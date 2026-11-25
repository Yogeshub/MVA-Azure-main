"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field

# Azure OpenAI credentials (AZURE_OPENAI_*) genuinely identical across Agent
# 1/2/3 live in the repo-root .env — loaded here as a fallback UNDER this
# service's own .env, mirroring the exact pattern Agent 1/3 already use
# (local always wins on any key present in both).
_ROOT_ENV = str(Path(__file__).resolve().parent.parent.parent.parent / ".env")


class Settings(BaseSettings):
    """Central application settings with environment variable binding."""

    # Database
    database_url: str = Field(
        default="postgresql://mva_user:mva_password@localhost:5432/mva_profiling",
        alias="DATABASE_URL",
    )
    database_echo: bool = Field(default=False, alias="DATABASE_ECHO")

    # File Limits
    max_upload_size_mb: int = Field(default=25, alias="MAX_UPLOAD_SIZE_MB")
    max_dataset_rows: int = Field(default=200_000, alias="MAX_DATASET_ROWS")
    max_dataset_columns: int = Field(default=200, alias="MAX_DATASET_COLUMNS")
    processing_timeout_seconds: int = Field(default=120, alias="PROCESSING_TIMEOUT_SECONDS")

    # Temporary Storage
    temp_storage_dir: str = Field(default="./tmp/uploads", alias="TEMP_STORAGE_DIR")
    max_sample_values: int = Field(default=10, alias="MAX_SAMPLE_VALUES")

    # Drill-down Cubes
    min_cube_group_size: int = Field(default=5, alias="MIN_CUBE_GROUP_SIZE")

    # Ad-hoc aggregation (POST /profile-runs/{run_id}/aggregate).
    # The uploaded rows are the only source that can answer a dimension x metric
    # pairing the run did not pre-compute, so they are kept for a bounded window
    # after profiling instead of being deleted immediately. Set RETAIN_RUN_FILES
    # to false to restore the old delete-on-completion behaviour (ad-hoc
    # aggregation then falls back to pre-aggregated cubes only).
    retain_run_files: bool = Field(default=True, alias="RETAIN_RUN_FILES")
    run_file_retention_hours: int = Field(default=24, alias="RUN_FILE_RETENTION_HOURS")
    max_aggregation_groups: int = Field(default=200, alias="MAX_AGGREGATION_GROUPS")

    # Source dataset reference — how a finished run gets back to its rows.
    #
    # Retention above is a cache, not the source of truth. The permanent copy of
    # every upload lives in the orchestrator's Dataset Registry, content-addressed
    # by SHA-256 and never purged; a run records which one it came from
    # (profile_runs.source_fingerprint / source_dataset_uri). These two settings
    # say how to reach that copy once the retained scratch file has been swept, so
    # drill-down stops being bounded by RUN_FILE_RETENTION_HOURS.
    #
    # MASTER_DATASET_STORAGE_DIR: the orchestrator's master-dataset directory when
    #   both services share a filesystem (the normal single-box deployment). Point
    #   it at the same absolute path the orchestrator writes to. No network hop.
    # MASTER_DATASET_CONTENT_URL: fallback for a split deployment — the
    #   orchestrator's content endpoint, e.g.
    #   http://127.0.0.1:8002/datasets/masters/{fingerprint}/content
    #   The "{fingerprint}" placeholder is substituted; downloads are cached under
    #   TEMP_STORAGE_DIR/_master_cache so a drill session fetches once.
    # Leave either blank to disable that route. With both blank, behaviour is
    # exactly as before: retained rows, then cubes, then an honest failure.
    master_dataset_storage_dir: str = Field(default="", alias="MASTER_DATASET_STORAGE_DIR")
    master_dataset_content_url: str = Field(default="", alias="MASTER_DATASET_CONTENT_URL")
    master_dataset_fetch_timeout_seconds: float = Field(
        default=30.0, alias="MASTER_DATASET_FETCH_TIMEOUT_SECONDS"
    )

    # Where the master dataset store physically lives.
    #
    # "local" (default) — a directory on a filesystem this service can open,
    #   given by MASTER_DATASET_STORAGE_DIR above. Correct for local
    #   development and for any deployment where the orchestrator and this
    #   service share a disk. Nothing below is read, and the Azure SDK is
    #   never imported.
    # "blob" — an Azure Blob container both services can reach. Intended for
    #   Azure, where the two services have separate ephemeral filesystems and
    #   no shared path exists. The blob layout mirrors the local one exactly
    #   (<prefix>/<hash[:2]>/<hash><ext>), and downloaded bytes are still
    #   SHA-256 verified against the run's fingerprint before use.
    #
    # Deliberately an explicit switch rather than something inferred from
    # APP_ENV: where a service reads customer data from should never change
    # because an unrelated environment label was mistyped, and it must be
    # possible to exercise the blob path outside production before trusting it.
    master_dataset_backend: str = Field(default="local", alias="MASTER_DATASET_BACKEND")

    # Azure Blob settings — only read when master_dataset_backend == "blob".
    # Authenticate with Managed Identity (AZURE_BLOB_ACCOUNT_URL, no secret in
    # config) or, failing that, a connection string. AZURE_BLOB_CLIENT_ID
    # selects a user-assigned identity when more than one is attached.
    azure_blob_account_url: str = Field(default="", alias="AZURE_BLOB_ACCOUNT_URL")
    azure_blob_container: str = Field(default="", alias="AZURE_BLOB_CONTAINER")
    azure_blob_connection_string: str = Field(default="", alias="AZURE_BLOB_CONNECTION_STRING")
    azure_blob_client_id: str = Field(default="", alias="AZURE_BLOB_CLIENT_ID")
    azure_blob_prefix: str = Field(default="master-datasets", alias="AZURE_BLOB_PREFIX")

    # Cap on the "what can I drill into next" list returned with an aggregation.
    # Only computed when the caller asks for it (include_next_levels), so the
    # existing response shape and cost are unchanged for every other caller.
    max_next_level_candidates: int = Field(default=40, alias="MAX_NEXT_LEVEL_CANDIDATES")

    # LLM Provider — "azure" (default, real backend) or "mock" (tests only,
    # no network calls). No more "groq"/"openai" branch: this project runs
    # on Azure OpenAI exclusively.
    llm_provider: str = Field(default="azure", alias="LLM_PROVIDER")

    # Azure OpenAI — this project runs on Azure OpenAI, not Groq. These 4
    # normally come from the shared root .env (see _ROOT_ENV above); a
    # service-local .env value always wins if both define the same key.
    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str = Field(default="", alias="AZURE_OPENAI_DEPLOYMENT")
    # Only used by the feature-target/rule-suggestion ReAct agents'
    # AzureChatOpenAI client — LangChain's Azure integration needs an
    # explicit api-version, unlike the direct-httpx LLMProvider calls.
    azure_openai_api_version: str = Field(default="2024-10-21", alias="AZURE_OPENAI_API_VERSION")
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")
    llm_burst_cooldown_seconds: float = Field(default=5.0, alias="LLM_BURST_COOLDOWN_SECONDS")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # Application
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")

    # Authentication
    api_keys: list[str] = Field(default_factory=list, alias="API_KEYS")
    jwt_secret: str = Field(default="", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    model_config = {
        "env_file": (_ROOT_ENV, ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Factory function for settings singleton."""
    return Settings()
