"""FastAPI application entry point."""

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.api.error_handlers import register_error_handlers
from app.api.routes import health, profile_runs, rule_suggestions

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    setup_logging(settings)
    backend = (settings.master_dataset_backend or "local").strip().lower()
    if backend == "blob":
        auth_mode = (
            "connection_string"
            if settings.azure_blob_connection_string
            else "managed_identity"
        )
        logger.info(
            "master_dataset_storage_config",
            backend="blob",
            container=settings.azure_blob_container or "<missing>",
            prefix=settings.azure_blob_prefix or "<root>",
            auth=auth_mode,
        )
    else:
        logger.info(
            "master_dataset_storage_config",
            backend="local",
            directory=str(settings.master_dataset_storage_dir),
        )

    app = FastAPI(
        title="MVA Data Profiling Engine",
        description="Multi-Variance Analysis — Schema, Quality, Hierarchy, Readiness, and Chart Intelligence",
        version="0.1.0",
    )

    # Register error handlers
    register_error_handlers(app)

    # Register routers
    prefix = settings.api_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(profile_runs.router, prefix=prefix)
    app.include_router(rule_suggestions.router, prefix=prefix)

    return app


app = create_app()
