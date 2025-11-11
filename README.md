# OCE Middleware

FastAPI gateway that lets GPT Actions call a private OCE service on Render.

## Env
- `RENDER_OCE_URL`        e.g. https://oce-proto.onrender.com
- `RENDER_OCE_API_KEY`    secret for OCE upstream
- `ACTION_KEY`            shared secret with GPT Action

## Local run
uvicorn middleware.server:app --reload

## Endpoints
- GET /health
- GET /openapi.yaml
- POST /warmup (header: X-Action-Key)
- POST /bridge/run (header: X-Action-Key)
