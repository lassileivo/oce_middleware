# OCE Middleware

FastAPI gateway for GPT Actions -> OCE Core.

## Env
ACTION_KEY=...
RENDER_OCE_URL=https://oce-proto.onrender.com
RENDER_OCE_API_KEY=    # optional
CORS_ORIGIN=https://chat.openai.com
ALLOW_PUBLIC_WARMUP=false

## Run local
uvicorn middleware.server:app --reload

## Endpoints
GET /health
GET /health/deep
POST /warmup   (X-Action-Key or X-Api-Key)
POST /bridge/run (X-Action-Key or X-Api-Key)

