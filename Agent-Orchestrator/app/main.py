import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes.aggregate import router as aggregate_router
from app.routes.datasets import router as datasets_router
from app.routes.objectives import router as objectives_router
from app.routes.pipeline import router as pipeline_router
from app.services.dataset_registry import metadata_cache

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Dataset Registry (Stage 0A) init — both steps non-fatal, mirroring
    Analytics-Agent's init_db() pattern: a Postgres outage or an
    unwritable storage directory must degrade to "the cache always
    misses," never take down the core relay function this service exists
    for."""
    logger.info("Starting Agent Orchestrator...")
    metadata_cache.init_db()
    backend = (settings.MASTER_DATASET_BACKEND or "local").strip().lower()
    if backend == "blob":
        auth_mode = (
            "connection_string"
            if settings.AZURE_BLOB_CONNECTION_STRING
            else "managed_identity"
        )
        logger.info(
            "master_dataset_storage_config | backend=blob | container=%s | prefix=%s | auth=%s",
            settings.AZURE_BLOB_CONTAINER or "<missing>",
            settings.AZURE_BLOB_PREFIX or "<root>",
            auth_mode,
        )
    else:
        logger.info(
            "master_dataset_storage_config | backend=local | directory=%s",
            settings.MASTER_DATASET_STORAGE_DIR,
        )
        try:
            Path(settings.MASTER_DATASET_STORAGE_DIR).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create master dataset storage dir — Dataset Registry cache disabled: {e}")
    logger.info("Agent Orchestrator ready.")
    yield
    logger.info("Shutting down Agent Orchestrator.")


app = FastAPI(
    title="Agent Orchestrator",
    description=(
        "Coordinates the multi-agent data pipeline: uploads a dataset to the "
        "Schema Intelligence Layer, feeds its output into the Data Profiling "
        "Engine, and returns both results together. Designed to grow as more "
        "agents are added to the pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_router)
app.include_router(aggregate_router)
app.include_router(datasets_router)
app.include_router(objectives_router)


async def _ping(client: httpx.AsyncClient, name: str, url: str) -> dict:
    try:
        resp = await client.get(url, timeout=5.0)
        return {"name": name, "url": url, "reachable": resp.status_code == 200}
    except httpx.HTTPError as e:
        return {"name": name, "url": url, "reachable": False, "error": str(e)}


@app.get("/health")
async def health_check():
    """Health check — also pings each downstream agent so it's obvious what's actually reachable."""
    async with httpx.AsyncClient() as client:
        agents = [
            await _ping(client, "agent1_schema_intelligence", f"{settings.AGENT1_BASE_URL}/health"),
            await _ping(
                client,
                "agent2_data_profiling",
                f"{settings.AGENT2_BASE_URL}{settings.AGENT2_API_PREFIX}/health",
            ),
        ]
    return {
        "status": "healthy",
        "service": "Agent Orchestrator",
        "agents": agents,
    }
