import os
import time
import re
import json
import hashlib
from typing import Optional, Dict, Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from pydantic import BaseModel, Field, validator
import httpx

# ========= ENV =========
RENDER_OCE_URL = os.getenv("RENDER_OCE_URL", "").rstrip("/")
RENDER_OCE_API_KEY = os.getenv("RENDER_OCE_API_KEY", "")
ACTION_KEY = os.getenv("ACTION_KEY", "")
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "https://chat.openai.com")
ALLOW_PUBLIC_WARMUP = os.getenv("ALLOW_PUBLIC_WARMUP", "false").lower() == "true"
# polku openapi.yamlille (oletus: repon juuressa)
OPENAPI_FILE = os.getenv(
    "OPENAPI_FILE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "openapi.yaml")),
)

if not ACTION_KEY:
    print("[BOOT] WARNING: ACTION_KEY missing; /warmup and /bridge/run will 401 unless ALLOW_PUBLIC_WARMUP=true")
# =======================

app = FastAPI(title="OCE Middleware", version="0.2.2")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate limit (IP+key) ---
RATE_LIMIT_WINDOW = 30  # seconds
RATE_LIMIT_MAX = 20     # requests/window
_rate: Dict[str, Dict[str, Any]] = {}

def _rate_key(ip: str, action_key: str) -> str:
    return hashlib.sha256(f"{ip}:{action_key}".encode()).hexdigest()

def check_rate(ip: str, action_key: str):
    k = _rate_key(ip, action_key or "")
    now = time.time()
    slot = _rate.get(k)
    if not slot or now - slot["start"] > RATE_LIMIT_WINDOW:
        _rate[k] = {"start": now, "count": 1}
        return
    slot["count"] += 1
    if slot["count"] > RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded; slow down.")

# --- Auth header helper (accept both names) ---
def _extract_action_key_from_headers(request: Request, explicit_header: Optional[str]) -> Optional[str]:
    if explicit_header:
        return explicit_header
    h = request.headers
    return (
        h.get("X-Action-Key")
        or h.get("x-action-key")
        or h.get("X-Api-Key")
        or h.get("x-api-key")
    )

def check_auth(key: Optional[str]):
    if not ALLOW_PUBLIC_WARMUP and (not key or key != ACTION_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized.")

# --- Models ---
class SessionCtx(BaseModel):
    project_id: str = Field(..., description="Opaque session/project id scoped by middleware.")
    user_id: Optional[str] = None
    message_id: Optional[str] = None
    mcda: Optional[Dict[str, Any]] = None
    risk: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

    @validator("project_id")
    def projid_ok(cls, v):
        if not re.match(r"^[A-Za-z0-9_\-]{1,64}$", v):
            raise ValueError("Invalid project_id.")
        return v

class RunPayload(BaseModel):
    user_text: str
    session_ctx: SessionCtx

# --- Helpers ---
def redact(o: Any) -> Any:
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if "key" in k.lower() or "token" in k.lower() or "auth" in k.lower():
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(o, list):
        return [redact(x) for x in o]
    return o

async def oce_health() -> Dict[str, Any]:
    if not RENDER_OCE_URL:
        raise RuntimeError("RENDER_OCE_URL not set")
    headers = {}
    if RENDER_OCE_API_KEY:
        headers["Authorization"] = f"Bearer {RENDER_OCE_API_KEY}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{RENDER_OCE_URL}/health", headers=headers)
        r.raise_for_status()
        return r.json()

async def oce_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not RENDER_OCE_URL:
        raise RuntimeError("RENDER_OCE_URL not set")
    headers = {"Content-Type": "application/json"}
    if RENDER_OCE_API_KEY:
        headers["Authorization"] = f"Bearer {RENDER_OCE_API_KEY}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{RENDER_OCE_URL}/run_oce", headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

SAFE_SECTION = re.compile(r"^(Thesis|Hypothesis|Key Points|Counterpoints|Actions|Next Step|Criteria|Weights|Options|Scores|Recommendation|Uncertainty|Top Risks|Expected Loss|Mitigation|Simulation|Falsifiable Claim|Tests / Predictions|Status|Variables|Method|Data Needs)$")

def shape_payload(p: RunPayload) -> Dict[str, Any]:
    bad = ["wipe memory", "delete memory", "shutdown", "reconfigure"]
    lower = p.user_text.lower()
    if any(b in lower for b in bad):
        raise HTTPException(status_code=400, detail="Dangerous instruction blocked by middleware.")
    return {
        "user_text": p.user_text.strip(),
        "session_ctx": p.session_ctx.dict(),
    }

def make_assistant_text(res: Dict[str, Any]) -> str:
    try:
        text = (res.get("text") or "").strip()
        js = res.get("json_summary", {}) or {}
        reco = None
        for line in text.splitlines():
            L = line.strip()
            if L.lower().startswith("**recommendation:**") or L.lower().startswith("recommendation:"):
                reco = L.split(":", 1)[-1].strip()
                break
        if not reco:
            maybe = js.get("recommendation") or js.get("Recommendation")
            if isinstance(maybe, str) and maybe.strip():
                reco = maybe.strip()
        bullets = []
        if reco:
            bullets.append(f"Suositus: {reco}")
        intent = js.get("intent")
        if intent:
            bullets.append(f"Intentio: {intent}")
        if "uncertainty" in text.lower():
            bullets.append("Huom: epävarmuusarvio mukana.")
        if "Top Risks" in text:
            bullets.append("Riskit arvioitu (Top Risks).")
        if bullets:
            return " • ".join(bullets)
        first = text.split("\n\n", 1)[0]
        return first[:280] + ("…" if len(first) > 280 else "")
    except Exception:
        return "OK."

# ============ ROUTES ============

@app.get("/health")
async def health():
    return {"status": "ok", "middleware": "alive"}

@app.get("/health/deep")
async def health_deep():
    try:
        oce = await oce_health()
        return {"status": "ok", "middleware": "alive", "oce": oce}
    except Exception as e:
        return {"status": "degraded", "middleware": "alive", "error": str(e)}

# palvele openapi.yaml staattisena tiedostona (GPT Actionsin importia varten)
@app.get("/openapi.yaml")
async def get_openapi_yaml():
    try:
        if not os.path.isfile(OPENAPI_FILE):
            raise FileNotFoundError(OPENAPI_FILE)
        return FileResponse(OPENAPI_FILE, media_type="application/yaml", filename="openapi.yaml")
    except Exception as e:
        return PlainTextResponse(f"openapi.yaml not found: {e}", status_code=404)

@app.post("/warmup")
async def warmup(request: Request, x_action_key: str = Header(None)):
    try:
        key = _extract_action_key_from_headers(request, x_action_key)
        if not ALLOW_PUBLIC_WARMUP:
            check_auth(key)
        warning = None
        if RENDER_OCE_URL:
            try:
                _ = await oce_health()
            except Exception as e:
                warning = f"oce_health failed: {e}"
        body = {"status": "warmed"}
        if ALLOW_PUBLIC_WARMUP:
            body["public"] = True
        if warning:
            body["warning"] = warning
        return JSONResponse(status_code=200, content=body)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=200, content={"status": "warmed", "warning": f"internal: {e}"})

@app.post("/bridge/run")
async def bridge_run(req: Request, body: RunPayload, x_action_key: str = Header(None)):
    key = _extract_action_key_from_headers(req, x_action_key)
    check_auth(key)
    client_ip = req.client.host if req.client else "0.0.0.0"
    check_rate(client_ip, key)
    shaped = shape_payload(body)
    try:
        res = await oce_run(shaped)
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=f"OCE error: {he.response.text}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")
    try:
        js = res.get("json_summary", {}) or {}
        sections = js.get("sections_present", [])
        if any((not SAFE_SECTION.match(s)) for s in sections):
            res["text"] = (res.get("text") or "") + "\n\n[Note] Some sections were filtered for safety."
    except Exception:
        pass
    res["assistant_text"] = make_assistant_text(res)
    print("[BRIDGE] ok", json.dumps({"ip": client_ip, "payload": redact(shaped)}, ensure_ascii=False))
    return JSONResponse(
        status_code=200,
        content=res,
        headers={
            "X-Request-ID": hashlib.sha256(f"{time.time()}:{client_ip}".encode()).hexdigest()[:16],
            "X-Memory-Check": "ok" if (body.session_ctx and body.session_ctx.project_id) else "fail",
        },
    )



