import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .routers import datasets, augmentation, filtering, training, evaluation, annotation
from .config import LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Text Augment Platform",
    description="Low-resource text data augmentation and NLP model training experiment platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router, prefix="/api/v1")
app.include_router(augmentation.router, prefix="/api/v1")
app.include_router(filtering.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")
app.include_router(evaluation.router, prefix="/api/v1")
app.include_router(annotation.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "Text Augment Platform",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
