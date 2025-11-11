import os, time, re, json, hashlib, secrets, uuid, asyncio, datetime
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
ACTION_KEY = os.getenv("ACTION_KEY", "")
# ------------------------

app = FastAPI(title="OCE Middleware", version="0.2.0")

# --- Optional: cap body size ---
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

# --- Simple rate limit ---
RATE_LIMIT_WINDOW = 30  # s
RATE_LIMIT_MAX = 20
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
BAD_INSTR = ["wipe memory", "delete memory", "shutdown", "reconfigure"]

class SessionCtx(BaseModel):
    project_id: str = Field(..., description="Opaque session/project id scoped by middleware.")
    user_id: Optional[str] = None
    message_id: Optional[str] = None
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
    if not RENDER_OCE_URL:
        return {"status":"degraded", "error":"RENDER_OCE_URL not set"}
    url = f"{RENDER_OCE_URL}/health"
    headers = {"Authorization": f"Bearer {RENDER_OCE_API_KEY}"} if RENDER_OCE_API_KEY else {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()

async def oce_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{RENDER_OCE_URL}/run_oce"
    headers = {"Authorization": f"Bearer {RENDER_OCE_API_KEY}", "Content-Type":"application/json"} if RENDER_OCE_API_KEY else {"Content-Type":"application/json"}
    return await _post_with_backoff(url, headers, payload, tries=3, base=0.5)

def shape_payload(body: RunPayload, ip: str) -> Dict[str, Any]:
    lower = body.user_text.lower()
    if any(b in lower for b in BAD_INSTR):
        raise HTTPException(status_code=400, detail="Dangerous instruction blocked by middleware.")
    shaped = body.model_dump()
    # enrich meta
    meta = shaped["session_ctx"].get("meta") or {}
    meta.update({
        "client": "gpt-actions",
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "ip_hash": hashlib.sha256(ip.encode()).hexdigest(),
    })
    shaped["session_ctx"]["meta"] = meta
    return shaped

SECTION_RE = re.compile(r"^#?\s*(Thesis|Recommendation|Actions|Top Risks)\s*:?", re.I | re.M)

def humanize_response(res: Dict[str, Any]) -> str:
    """
    Minimal 'assistant_text' from OCE output:
    - Thesis: 1–2 lauseen yhteenveto
    - Recommendation: suositus + lyhyt perustelu
    - Actions: 2–3 konkreettista askelta
    - Top Risks: 2 tärkeintä riskiä (nimi + lyhyt)
    Fallback: lyhyt tiivistelmä ensimmäisestä kappaleesta.
    """
    txt = (res.get("text") or "").strip()
    if not txt:
        return "OCE processed the request. No textual summary was returned."
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    thesis = next((l for l in lines if l.lower().startswith(("**thesis**","thesis:"))), None)
    reco = next((l for l in lines if l.lower().startswith(("**recommendation**","recommendation:"))), None)
    # Actions
    acts_idx = next((i for i,l in enumerate(lines) if l.lower().startswith(("**actions**","actions:"))), None)
    actions = []
    if acts_idx is not None:
        for l in lines[acts_idx+1:acts_idx+6]:
            if l.startswith(("*","-")): actions.append(l.lstrip("*- ").strip())
            else: break
    # Risks
    risk_idx = next((i for i,l in enumerate(lines) if l.lower().startswith(("**top risks**","top risks:"))), None)
    risks = []
    if risk_idx is not None:
        for l in lines[risk_idx+1:risk_idx+4]:
            if l.startswith(("*","-")): risks.append(l.lstrip("*- ").strip())
            else: break

    out = []
    if thesis: out.append(thesis.replace("**",""))
    if reco: out.append(reco.replace("**",""))
    if actions: out.append("Next steps: " + "; ".join(actions[:3]) + ".")
    if risks: out.append("Watch risks: " + "; ".join(risks[:2]) + ".")
    if not out:
        out.append(lines[0] if lines else "Summary unavailable.")
    return " ".join(out)

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
    # Älä koske OCE:en – palauta vain middleware-liveness.
    return {"status": "ok", "middleware": "alive"}

@app.get("/health/deep")
async def health_deep():
    try:
        oce = await oce_health()
        return {"status": "ok", "middleware": "alive", "oce": oce}
    except Exception as e:
        return {"status": "degraded", "middleware": "alive", "error": str(e)}


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

    shaped = shape_payload(body, client_ip)
    rid = str(uuid.uuid4())

    try:
        res = await oce_run(shaped)
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=f"OCE error: {he.response.text}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")

    # Build assistant_text for GPT to speak naturally
    assistant_text = humanize_response(res)
    res["assistant_text"] = assistant_text

    # Memory check hint
    js = res.get("json_summary", {}) or {}
    mem = js.get("memory", {})
    mem_ok = bool(mem.get("updated") is True)
    headers = {"X-Request-ID": rid, "X-Memory-Check": "ok" if mem_ok else "fail"}
    if not mem_ok:
        res["assistant_text"] += " (Note: upstream memory may not have updated; retry if session continuity is required.)"

    print("[BRIDGE] ok", json.dumps({"rid": rid, "ip": client_ip, "payload": redact(shaped)}, ensure_ascii=False))
    return JSONResponse(res, headers=headers)

