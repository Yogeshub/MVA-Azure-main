"""Profile run model — the top-level entity for each processing job."""

import uuid
from datetime import datetime

from sqlalchemy import String, Integer, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ProfileRun(Base):
    __tablename__ = "profile_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    primary_domain: Mapped[str] = mapped_column(String(50), nullable=False)
    dominant_secondary_domain: Mapped[str | None] = mapped_column(String(100), nullable=True)
    source_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_file_type: Mapped[str] = mapped_column(String(10), nullable=False)
    # Durable pointer back to the bytes this run was built from. The retained
    # scratch copy under TEMP_STORAGE_DIR is a cache with a lifetime; this
    # reference outlives it, so a breakdown the run never pre-aggregated can be
    # recomputed from the source dataset instead of failing.
    # Nullable: a caller that supplies no reference behaves exactly as before.
    source_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_dataset_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # The worksheet that was profiled. Also written to the retained run's
    # sidecar, but that dies with the scratch copy — a multi-sheet workbook
    # cannot be re-opened correctly without it.
    source_sheet_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    configuration_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
