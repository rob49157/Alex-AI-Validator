import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from api.routes import tier1, tier2
from config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the embedding model so the first Tier 2 job doesn't stall
    logger.info("Pre-loading embedding model…")
    from services.semantic_similarity import _get_model
    await asyncio.to_thread(_get_model)
    logger.info("Embedding model ready.")
    yield


app = FastAPI(title="Alexandria AI Validator", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    key = request.headers.get("Authorization", "")
    if key != f"Bearer {settings.api_key}":
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Unauthorized"})
    return await call_next(request)


app.include_router(tier1.router, prefix="/api/validate")
app.include_router(tier2.router, prefix="/api/validate")


@app.get("/health")
def health():
    return {"status": "ok"}
