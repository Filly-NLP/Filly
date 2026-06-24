import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.database.database import init_db
from app.api.endpoints import router as api_router

# ─── Logging ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("filly")


# ─── Lifespan ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    logger.info("=" * 60)
    logger.info("  FILLY v%s starting up", settings.APP_VERSION)
    logger.info("  Database: %s", settings.DATABASE_URL)
    logger.info("=" * 60)

    # Initialize database tables
    init_db()
    logger.info("Database tables initialized")

    # Pre-warm NLP services (lazy singletons will init on first call,
    # but we can trigger them here for faster first-request response)
    try:
        from app.services.normalizer import get_normalizer
        from app.services.gec import get_gec_service
        get_normalizer()
        get_gec_service()
        logger.info("NLP services ready")
    except ImportError:
        logger.info("NLP services not available yet (Phase 1)")

    yield

    # ── Shutdown ──
    logger.info("FILLY shutting down")


# ─── App ────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Filipino writing assistant with text normalization and grammar correction.",
    lifespan=lifespan,
)

# ─── CORS Middleware ────────────────────────────────────────────────
# Allows the Vite frontend dev server (localhost:5173) to call the API.

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ─────────────────────────────────────────────────────────
# All endpoints are under the /api prefix to match the frontend proxy config.

app.include_router(api_router, prefix="/api")
