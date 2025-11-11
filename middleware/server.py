import os, time, re, json, hashlib, secrets, uuid, asyncio
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator
import httpx
from starlette.middleware.base import BaseHTTPMiddleware

# --------- ENV ----------
RENDER_OCE_URL = os.getenv("RENDER_OCE_URL", "").rstrip("/")
RENDER_OCE_API_KEY = os.getenv("RENDER_OCE_API_KEY", "")
ACTION_KEY = os.getenv("ACTION_KEY", "")   # shared secret with GPT Action
# ------------------------

if not (RENDER_OCE_URL.startswith("https://") and RENDER_OCE_API_KEY and ACTION_KEY):
    print("[BOOT] Missing or invalid env variables.")

app = FastAPI(title="OCE Middleware", version="0.1.0")

# --- Optional: max body size ---
class MaxSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int = 256_000):
        super().__init__(app)
        self.max_bytes = max_bytes
    async def dispatch(self, request, call_next):
        cl = request.headers.get("content-length")
        if cl and int(cl) > self.max_bytes:
            return JSONResponse(status_code=413, content={"detail":"Payload too large"})
        return await call_next(request)

app.add_middleware(MaxSizeMiddleware, max_bytes=256_000)

# --- Rate limiting (IP+key) ---
RATE_LIMIT_WINDOW = 30  # s
RATE_LIMIT_MAX = 20     # req/window
_rate: Dict[str, Dict[str, Any]] = {}

def _rate_key(ip: str, action_key: str) -> str:
    return hashlib.sha256(f"{ip}:{action_key}".encode()).hexdigest()

def check_rate(ip: str, action_key: str):
    k = _rate_key(ip, action_key)
    now = time.time()
    slot = _rate.get(k)
    if not slot or now - slot["start"] > RATE_LIMIT_WINDOW:
        _rate[k] = {"start": now, "count": 1}
        return
    slot["count"] += 1
    if slot["count"] > RATE_LIMIT_MAX:
        hint = int(RATE_LIMIT_WINDOW - (now - slot["start"]))
        raise HTTPException(status_code=429, detail={"error":"rate_limited","hint":f"Try after {hint}s"})

def check_auth(x_action_key: Optional[str]):
    if not (ACTION_KEY and x_action_key) or not secrets.compare_digest(x_action_key, ACTION_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized.")

# --- Schemas ---
SAFE_SECTION = re.compile(r"^(Thesis|Hypothesis|Key Points|Counterpoints|Actions|Next Step|Criteria|Weights|Options|Scores|Recommendation|Uncertainty|Top Risks|Expected Loss|Mitigation|Simulation|Falsifiable Claim|Tests / Predictions|Status|Variables|Method|Data Needs)$")

class SessionCtx(BaseModel):
    project_id: str = Field(..., description="Opaque session/project id scoped by middleware.")
    mcda: Optional[Dict[str, Any]] = None
    risk: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

    @field_validator("project_id")
    @classmethod
    def projid_ok(cls, v: str):
        if not re.match(r"^[A-Za-z0-9_\-]{1,64}$", v):
            raise ValueError("Invalid project_id.")
        return v

class RunPayload(BaseModel):
    user_text: str = Field(..., min_length=1, max_length=8000)
    session_ctx: SessionCtx

class WarmupPayload(BaseModel):
    project_id: Optional[str] = None

# --- Helpers ---
def redact(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: ("***" if any(t in k.lower() for t in ["key","token","auth"]) else redact(v)) for k,v in o.items()}
    if isinstance(o, list):
        return [redact(x) for x in o]
    return o

async def _post_with_backoff(url, headers, payload, tries=3, base=0.5):
    async with httpx.AsyncClient(timeout=60.0) as client:
        last = None
        for i in range(tries):
            try:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last = e
                if i < tries - 1:
                    await asyncio.sleep(base * (2 ** i))
        raise last

async def oce_health() -> Dict[str, Any]:
    url = f"{RENDER_OCE_URL}/health"
    headers = {"Authorization": f"Bearer {RENDER_OCE_API_KEY}"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()

async def oce_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{RENDER_OCE_URL}/run_oce"
    headers = {"Authorization": f"Bearer {RENDER_OCE_API_KEY}", "Content-Type":"application/json"}
    return await _post_with_backoff(url, headers, payload, tries=3, base=0.5)

def shape_payload(p: RunPayload) -> Dict[str, Any]:
    bad = ["wipe memory", "delete memory", "shutdown", "reconfigure"]
    lower = p.user_text.lower()
    if any(b in lower for b in bad):
        raise HTTPException(status_code=400, detail="Dangerous instruction blocked by middleware.")
    return {"user_text": p.user_text.strip(), "session_ctx": p.session_ctx.dict()}

# --- Routes ---
ROOT = Path(__file__).resolve().parents[1]

@app.get("/openapi.yaml", include_in_schema=False)
def manifest():
    path = ROOT / "openapi.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found.")
    return FileResponse(path)

@app.get("/health")
async def health():
    try:
        oce = await oce_health()
        return {"status": "ok", "middleware": "alive", "oce": oce}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "degraded", "error": str(e)})

@app.post("/warmup")
async def warmup(_: WarmupPayload, x_action_key: str = Header(None)):
    check_auth(x_action_key)
    _ = await oce_health()
    return {"status": "warmed"}

@app.post("/bridge/run")
async def bridge_run(req: Request, body: RunPayload, x_action_key: str = Header(None)):
    check_auth(x_action_key)
    client_ip = req.client.host if req.client else "0.0.0.0"
    check_rate(client_ip, x_action_key)

    shaped = shape_payload(body)
    rid = str(uuid.uuid4())
    try:
        res = await oce_run(shaped)
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=f"OCE error: {he.response.text}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")

    try:
        js = res.get("json_summary", {})
        sections = js.get("sections_present", [])
        if any((not SAFE_SECTION.match(s)) for s in sections):
            res["text"] = (res.get("text") or "") + "\n\n[Note] Some sections were filtered for safety."
    except Exception:
        pass

    print("[BRIDGE] ok", json.dumps({"rid": rid, "ip": client_ip, "payload": redact(shaped)}, ensure_ascii=False))
    return JSONResponse(res, headers={"X-Request-ID": rid})
