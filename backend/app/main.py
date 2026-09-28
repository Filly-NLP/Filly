import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.database.database import init_db
from app.api.endpoints import router as api_router
from app.api.endpoints import v1_router as api_v1_router

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

    app.state.filly_ready = False

    # Initialize database tables
    init_db()
    logger.info("Database tables initialized")

    # The GEC loader reads its checkpoint path from the process environment.
    # Resolve the configured path once at startup so resources are not loaded
    # for individual requests.
    os.environ["GECTOR_MODEL_PATH"] = str(settings.GECTOR_MODEL_PATH)

    from app.services.normalizer import get_normalizer
    from app.services.gec import get_gec_service
    from app.services.filly_pipeline import FillyPipeline

    normalizer = get_normalizer(
        rules_path=settings.NORMALIZER_RULES_PATH,
        vocabulary_path=settings.NORMALIZER_VOCAB_PATH,
    )
    requested_device = None if settings.DEVICE.lower() == "auto" else settings.DEVICE.lower()
    gec_service = get_gec_service(device=requested_device)
    app.state.filly_pipeline = FillyPipeline(
        normalizer,
        gec_service,
        iterations=settings.GECTOR_ITERATIONS,
    )
    app.state.filly_device = str(getattr(gec_service, "device", settings.DEVICE))
    app.state.filly_ready = True
    logger.info(
        "FILLY services ready (normalizer rules=%d, vocabulary=%d, GEC device=%s, iterations=%d)",
        len(normalizer.rules),
        len(normalizer.vocabulary),
        app.state.filly_device,
        settings.GECTOR_ITERATIONS,
    )

    try:
        yield
    finally:
        app.state.filly_ready = False

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
app.include_router(api_v1_router, prefix="/api/v1")
