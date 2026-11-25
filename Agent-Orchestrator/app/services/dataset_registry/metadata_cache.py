"""Stage 0A — Master Dataset Repository (Postgres-backed metadata cache).

Mirrors Analytics-Agent/app/services/database.py's raw-psycopg2 style (same
shared native Postgres instance, one schema per service — see
Shared-Postgres/README.md) rather than Agent 2's SQLAlchemy+Alembic — a
handful of tables with no evolving relational structure doesn't need a
migration framework.

Non-fatal on failure, like Agent 3's init_db() (never Agent 1's, which
raises and crashes startup) — the Dataset Registry is an enhancement over
the existing pipeline, not core to it. A Postgres outage must degrade to
"the cache always misses, the pipeline runs exactly as it did before this
feature existed," never break the core relay function.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

import psycopg2
import psycopg2.extras

from app.config import settings
from app.services.dataset_registry.models import CachedPipelineResult, DatasetCopy, MasterDataset

logger = logging.getLogger(__name__)

RESPONSE_SCHEMA_VERSION = 1


class DatasetDeletedError(RuntimeError):
    """Raised when a write targets a lineage that has a durable tombstone."""


def _lock_dataset(cur, dataset_id: str) -> None:
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"dataset:{dataset_id}",))


def _lock_fingerprint(cur, fingerprint: str) -> None:
    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtext(%s))",
        (f"fingerprint:{fingerprint}",),
    )


def _raise_if_dataset_deleted(cur, dataset_id: str) -> None:
    cur.execute(
        "SELECT 1 FROM dataset_deletion_audits WHERE dataset_id = %s",
        (dataset_id,),
    )
    if cur.fetchone() is not None:
        raise DatasetDeletedError(f"Dataset '{dataset_id}' has been permanently deleted.")


def _raise_if_fingerprint_deleted(cur, fingerprint: str) -> None:
    cur.execute(
        "SELECT 1 FROM dataset_deletion_audits WHERE fingerprints @> %s::jsonb",
        (json.dumps([fingerprint]),),
    )
    if cur.fetchone() is not None:
        raise DatasetDeletedError(
            f"Fingerprint '{fingerprint}' belongs to a permanently deleted dataset."
        )

CREATE_MASTER_DATASETS_SQL = """
CREATE TABLE IF NOT EXISTS master_datasets (
    fingerprint          TEXT PRIMARY KEY,
    dataset_id           UUID NOT NULL,
    original_filename    TEXT NOT NULL,
    storage_location     TEXT NOT NULL,
    file_extension       TEXT NOT NULL,
    byte_size            BIGINT NOT NULL,
    row_count            INTEGER,
    column_count         INTEGER,
    latest_version       INTEGER NOT NULL DEFAULT 1,
    previous_fingerprint TEXT REFERENCES master_datasets(fingerprint),
    reference_count      INTEGER NOT NULL DEFAULT 0,
    first_uploaded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_referenced_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
CREATE_MASTER_DATASETS_DATASET_ID_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_master_datasets_dataset_id ON master_datasets(dataset_id);"
)
CREATE_MASTER_DATASETS_FILENAME_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_master_datasets_filename ON master_datasets(original_filename);"
)

CREATE_DATASET_COPIES_SQL = """
CREATE TABLE IF NOT EXISTS dataset_copies (
    copy_id           UUID PRIMARY KEY,
    fingerprint       TEXT NOT NULL REFERENCES master_datasets(fingerprint) ON DELETE CASCADE,
    user_id           TEXT,
    uploaded_filename TEXT NOT NULL,
    upload_timestamp  TIMESTAMPTZ NOT NULL DEFAULT now(),
    version           INTEGER NOT NULL,
    deleted_flag      BOOLEAN NOT NULL DEFAULT FALSE
);
"""
CREATE_DATASET_COPIES_FP_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_dataset_copies_fingerprint ON dataset_copies(fingerprint);"
)
CREATE_DATASET_COPIES_ACTIVE_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_dataset_copies_active ON dataset_copies(fingerprint) "
    "WHERE deleted_flag = FALSE;"
)

# A profiling run is answerable for as long as its rows are reachable. Agent 2
# keeps a working copy only for RUN_FILE_RETENTION_HOURS, but the Master Dataset
# it came from is stored here permanently and never purged — what was missing
# was the link between the two. This table is that link, written the moment a
# run_id exists (and on a cache hit, for the run the cached result belongs to),
# so any run can be pointed back at its own bytes long after the working copy
# has been swept. ON DELETE CASCADE: if the master is deleted the link is a lie,
# and a lie is worse than a missing row.
CREATE_RUN_DATASET_LINKS_SQL = """
CREATE TABLE IF NOT EXISTS run_dataset_links (
    run_id            TEXT PRIMARY KEY,
    fingerprint       TEXT NOT NULL REFERENCES master_datasets(fingerprint) ON DELETE CASCADE,
    copy_id           UUID,
    storage_location  TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    sheet_name        TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
CREATE_RUN_DATASET_LINKS_FP_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_run_dataset_links_fingerprint ON run_dataset_links(fingerprint);"
)

CREATE_MASTER_DATASET_RESULTS_SQL = """
CREATE TABLE IF NOT EXISTS master_dataset_results (
    fingerprint             TEXT PRIMARY KEY REFERENCES master_datasets(fingerprint) ON DELETE CASCADE,
    primary_domain          TEXT NOT NULL,
    agent1_body             JSONB NOT NULL,
    agent2_full_result      JSONB NOT NULL,
    response_schema_version INTEGER NOT NULL DEFAULT 1,
    cached_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# Dataset deletion is keyed by the stable dataset_id (and therefore covers all
# versions/fingerprints in that lineage).  This row intentionally outlives the
# master rows: it is both the tombstone that makes DELETE idempotent and the
# durable audit record for logical and physical cleanup.
CREATE_DATASET_DELETION_AUDITS_SQL = """
CREATE TABLE IF NOT EXISTS dataset_deletion_audits (
    dataset_id                UUID PRIMARY KEY,
    status                    TEXT NOT NULL,
    requested_by              TEXT NOT NULL DEFAULT 'unknown',
    requested_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at              TIMESTAMPTZ,
    fingerprints              JSONB NOT NULL,
    source_locations          JSONB NOT NULL,
    active_copy_count         INTEGER NOT NULL DEFAULT 0,
    cached_result_count       INTEGER NOT NULL DEFAULT 0,
    historical_run_count      INTEGER NOT NULL DEFAULT 0,
    physical_deleted_count    INTEGER NOT NULL DEFAULT 0,
    physical_cleanup_failures JSONB NOT NULL DEFAULT '[]'::jsonb,
    cleanup_claim_id          TEXT,
    cleanup_claimed_at        TIMESTAMPTZ
);
"""
MIGRATE_DATASET_DELETION_AUDITS_SQL = """
ALTER TABLE dataset_deletion_audits
ADD COLUMN IF NOT EXISTS requested_by TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE dataset_deletion_audits
ADD COLUMN IF NOT EXISTS cleanup_claim_id TEXT;
ALTER TABLE dataset_deletion_audits
ADD COLUMN IF NOT EXISTS cleanup_claimed_at TIMESTAMPTZ;
"""

# Older installations created run_dataset_links with an ON DELETE CASCADE
# foreign key.  Historical execution linkage is audit data and must survive
# source deletion, so migrate that table in place and allow its source locator
# to be redacted.  The run_id/fingerprint/filename/sheet linkage remains.
MIGRATE_RUN_DATASET_LINKS_FOR_DELETION_SQL = """
ALTER TABLE run_dataset_links ALTER COLUMN storage_location DROP NOT NULL;
DO $$
DECLARE constraint_name TEXT;
BEGIN
    SELECT tc.constraint_name INTO constraint_name
      FROM information_schema.table_constraints tc
      JOIN information_schema.constraint_column_usage ccu
        ON tc.constraint_name = ccu.constraint_name
       AND tc.constraint_schema = ccu.constraint_schema
     WHERE tc.table_schema = 'orchestrator'
       AND tc.table_name = 'run_dataset_links'
       AND tc.constraint_type = 'FOREIGN KEY'
       AND ccu.table_name = 'master_datasets'
     LIMIT 1;
    IF constraint_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE run_dataset_links DROP CONSTRAINT %I', constraint_name);
    END IF;
END $$;
"""


def get_connection() -> psycopg2.extensions.connection:
    """Public — reference_counter.py also opens direct connections against
    the dataset_copies/master_datasets tables for its own queries, within
    the same dataset_registry package."""
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        cursor_factory=psycopg2.extras.RealDictCursor,
        options="-c search_path=orchestrator",
    )


_get_connection = get_connection  # internal alias, this module's own historical call sites below


def init_db() -> None:
    """Create the orchestrator schema/tables if they don't exist. Never
    raises — logs and returns on failure so a Postgres outage doesn't take
    the whole service down at startup (see module docstring)."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute("CREATE SCHEMA IF NOT EXISTS orchestrator")
        cur.execute(CREATE_MASTER_DATASETS_SQL)
        cur.execute(CREATE_MASTER_DATASETS_DATASET_ID_IDX_SQL)
        cur.execute(CREATE_MASTER_DATASETS_FILENAME_IDX_SQL)
        cur.execute(CREATE_DATASET_COPIES_SQL)
        cur.execute(CREATE_DATASET_COPIES_FP_IDX_SQL)
        cur.execute(CREATE_DATASET_COPIES_ACTIVE_IDX_SQL)
        cur.execute(CREATE_MASTER_DATASET_RESULTS_SQL)
        cur.execute(CREATE_RUN_DATASET_LINKS_SQL)
        cur.execute(CREATE_RUN_DATASET_LINKS_FP_IDX_SQL)
        cur.execute(MIGRATE_RUN_DATASET_LINKS_FOR_DELETION_SQL)
        cur.execute(CREATE_DATASET_DELETION_AUDITS_SQL)
        cur.execute(MIGRATE_DATASET_DELETION_AUDITS_SQL)
        conn.commit()
        cur.close()
        conn.close()
        logger.info(
            f"Dataset Registry DB initialised at "
            f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB} (orchestrator)"
        )
    except Exception as e:
        logger.error(f"Failed to initialise Dataset Registry DB — cache disabled for this run: {e}")


def _row_to_master(row: dict[str, Any], results_row: dict[str, Any] | None) -> MasterDataset:
    m = MasterDataset(
        fingerprint=row["fingerprint"],
        dataset_id=str(row["dataset_id"]),
        original_filename=row["original_filename"],
        storage_location=row["storage_location"],
        file_extension=row["file_extension"],
        byte_size=row["byte_size"],
        row_count=row["row_count"],
        column_count=row["column_count"],
        latest_version=row["latest_version"],
        previous_fingerprint=row["previous_fingerprint"],
        reference_count=row["reference_count"],
        first_uploaded_at=row["first_uploaded_at"],
        last_referenced_at=row["last_referenced_at"],
    )
    if results_row is not None:
        agent1_body = results_row["agent1_body"]
        agent2 = results_row["agent2_full_result"]
        m.schema = agent1_body
        m.validation = agent1_body.get("quality_report") if isinstance(agent1_body, dict) else None
        m.column_profiles = agent2.get("column_profiles") if isinstance(agent2, dict) else None
        m.hierarchy = agent2.get("hierarchy") if isinstance(agent2, dict) else None
        m.charts = agent2.get("charts") if isinstance(agent2, dict) else None
        m.ml_readiness = agent2.get("ml_readiness_breakdown") if isinstance(agent2, dict) else None
        m.llm_readiness = agent2.get("llm_readiness_breakdown") if isinstance(agent2, dict) else None
        m.feature_recommendation = agent2.get("feature_recommendation") if isinstance(agent2, dict) else None
    return m


def lookup_master_dataset(fingerprint: str) -> MasterDataset | None:
    """Look up a Master Dataset by fingerprint, hydrated with cached
    pipeline metadata when available. Raises on DB error — callers (the
    DatasetRegistry facade) wrap this in try/except for the non-fatal
    degrade-to-miss behavior."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM master_datasets WHERE fingerprint = %s", (fingerprint,))
        row = cur.fetchone()
        if row is None:
            return None
        cur.execute("SELECT * FROM master_dataset_results WHERE fingerprint = %s", (fingerprint,))
        results_row = cur.fetchone()
        cur.close()
        return _row_to_master(row, results_row)
    finally:
        conn.close()


def lookup_latest_version_by_filename(original_filename: str) -> MasterDataset | None:
    """The lineage lookup version_manager.py uses: the most recent Master
    Dataset uploaded under this exact filename, regardless of content —
    the same "filename identity" signal Agent 1's own existing
    check_existing_dataset() already uses for its replace-vs-new
    mechanism, reused here to decide "is this a new version of something
    we've seen, or a genuinely unrelated new dataset."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM master_datasets WHERE original_filename = %s "
            "ORDER BY latest_version DESC LIMIT 1",
            (original_filename,),
        )
        row = cur.fetchone()
        cur.close()
        return _row_to_master(row, None) if row else None
    finally:
        conn.close()


def insert_master_dataset(master: MasterDataset) -> None:
    """ON CONFLICT DO NOTHING — race-safety for two concurrent uploads of
    genuinely new, identical content: Postgres's PK uniqueness on
    fingerprint is the real arbiter, whichever insert commits first wins,
    the other becomes a silent no-op."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        _lock_dataset(cur, master.dataset_id)
        _lock_fingerprint(cur, master.fingerprint)
        _raise_if_dataset_deleted(cur, master.dataset_id)
        cur.execute(
            """
            INSERT INTO master_datasets (
                fingerprint, dataset_id, original_filename, storage_location, file_extension,
                byte_size, row_count, column_count, latest_version, previous_fingerprint, reference_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (fingerprint) DO NOTHING
            """,
            (
                master.fingerprint, master.dataset_id, master.original_filename, master.storage_location,
                master.file_extension, master.byte_size, master.row_count, master.column_count,
                master.latest_version, master.previous_fingerprint, master.reference_count,
            ),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def update_master_counts(fingerprint: str, row_count: int | None, column_count: int | None) -> None:
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE master_datasets SET row_count = %s, column_count = %s WHERE fingerprint = %s",
            (row_count, column_count, fingerprint),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def touch_last_referenced(fingerprint: str) -> None:
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE master_datasets SET last_referenced_at = now() WHERE fingerprint = %s",
            (fingerprint,),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def get_cached_result(fingerprint: str) -> CachedPipelineResult | None:
    """What the graph node actually needs on a cache hit — the exact three
    fields finalize() reads off state (agent1_body, agent2_full_result,
    primary_domain), gated on response_schema_version matching the
    constant this module defines (bumping RESPONSE_SCHEMA_VERSION
    silently invalidates every pre-existing cache row at once, if
    finalize()'s response shape ever changes in a way that matters)."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM master_dataset_results WHERE fingerprint = %s", (fingerprint,))
        row = cur.fetchone()
        cur.close()
        if row is None or row["response_schema_version"] != RESPONSE_SCHEMA_VERSION:
            return None
        return CachedPipelineResult(
            primary_domain=row["primary_domain"],
            agent1_body=row["agent1_body"],
            agent2_full_result=row["agent2_full_result"],
            response_schema_version=row["response_schema_version"],
        )
    finally:
        conn.close()


def upsert_result(fingerprint: str, primary_domain: str, agent1_body: dict[str, Any],
                   agent2_full_result: dict[str, Any]) -> None:
    """Upsert (not insert) so a force_revalidate re-run correctly refreshes
    a previously-cached result instead of being blocked by the existing
    row."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        _lock_fingerprint(cur, fingerprint)
        _raise_if_fingerprint_deleted(cur, fingerprint)
        cur.execute(
            """
            INSERT INTO master_dataset_results (
                fingerprint, primary_domain, agent1_body, agent2_full_result, response_schema_version
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (fingerprint) DO UPDATE SET
                primary_domain = EXCLUDED.primary_domain,
                agent1_body = EXCLUDED.agent1_body,
                agent2_full_result = EXCLUDED.agent2_full_result,
                response_schema_version = EXCLUDED.response_schema_version,
                cached_at = now()
            """,
            (fingerprint, primary_domain, json.dumps(agent1_body), json.dumps(agent2_full_result),
             RESPONSE_SCHEMA_VERSION),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def upsert_run_dataset_link(*, run_id: str, fingerprint: str, copy_id: str | None,
                            storage_location: str, original_filename: str,
                            sheet_name: str | None = None,
                            preserve_existing: bool = False) -> None:
    """Record which stored dataset a profiling run was built from.

    Upsert rather than insert: a force_revalidate re-run of the same content
    must refresh the link instead of colliding with it. Raises on DB error —
    callers treat a failed link as non-fatal (the run still completes, it just
    cannot be repaired later), matching this module's degrade-to-miss contract.

    ``preserve_existing`` is for the cache-hit path, where the run being linked
    is an *older* one being replayed. The bytes are identical by construction,
    but the current request may have selected a different worksheet, and
    overwriting the stored ``sheet_name`` would later recompute that older run
    from the wrong sheet. An existing link is already correct for the run it
    describes, so it is left alone.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        _lock_fingerprint(cur, fingerprint)
        _raise_if_fingerprint_deleted(cur, fingerprint)
        cur.execute(
            """
            INSERT INTO run_dataset_links (
                run_id, fingerprint, copy_id, storage_location, original_filename, sheet_name
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """
            + (
                "ON CONFLICT (run_id) DO NOTHING"
                if preserve_existing
                else """ON CONFLICT (run_id) DO UPDATE SET
                fingerprint = EXCLUDED.fingerprint,
                copy_id = EXCLUDED.copy_id,
                storage_location = EXCLUDED.storage_location,
                original_filename = EXCLUDED.original_filename,
                sheet_name = EXCLUDED.sheet_name,
                created_at = now()"""
            ),
            (run_id, fingerprint, copy_id, storage_location, original_filename, sheet_name),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def lookup_run_dataset_link(run_id: str) -> dict[str, Any] | None:
    """The stored dataset a run was built from, or None if it was never linked."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM run_dataset_links WHERE run_id = %s", (run_id,))
        row = cur.fetchone()
        cur.close()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def insert_copy(copy: DatasetCopy) -> None:
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT dataset_id FROM master_datasets WHERE fingerprint = %s",
            (copy.fingerprint,),
        )
        master = cur.fetchone()
        if master is None:
            raise DatasetDeletedError(
                f"Fingerprint '{copy.fingerprint}' is no longer available."
            )
        dataset_id = str(master["dataset_id"])
        _lock_dataset(cur, dataset_id)
        _lock_fingerprint(cur, copy.fingerprint)
        _raise_if_dataset_deleted(cur, dataset_id)
        cur.execute(
            """
            INSERT INTO dataset_copies (copy_id, fingerprint, user_id, uploaded_filename, version, deleted_flag)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (copy.copy_id, copy.fingerprint, copy.user_id, copy.uploaded_filename, copy.version, copy.deleted_flag),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def insert_copy_and_increment_reference(copy: DatasetCopy) -> None:
    """Same as insert_copy() + reference_counter.increment(), but as one
    connection/one transaction instead of two separately-committed round
    trips. Closes a real gap: a crash between the two separate calls left
    a DatasetCopy row that was never counted, permanently under-counting
    reference_count (returned verbatim by GET /datasets/masters — a
    user-visible field, not just internal bookkeeping). The delete
    guardrail itself was never at risk, since it always recomputes a live
    COUNT(*) rather than trusting this cached column — but the exposed
    reference_count could still lie to an operator deciding whether it's
    safe to force-delete something."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT dataset_id FROM master_datasets WHERE fingerprint = %s",
            (copy.fingerprint,),
        )
        master = cur.fetchone()
        if master is None:
            raise DatasetDeletedError(
                f"Fingerprint '{copy.fingerprint}' is no longer available."
            )
        dataset_id = str(master["dataset_id"])
        _lock_dataset(cur, dataset_id)
        _lock_fingerprint(cur, copy.fingerprint)
        _raise_if_dataset_deleted(cur, dataset_id)
        cur.execute(
            """
            INSERT INTO dataset_copies (copy_id, fingerprint, user_id, uploaded_filename, version, deleted_flag)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (copy.copy_id, copy.fingerprint, copy.user_id, copy.uploaded_filename, copy.version, copy.deleted_flag),
        )
        cur.execute(
            "UPDATE master_datasets SET reference_count = reference_count + 1 WHERE fingerprint = %s",
            (copy.fingerprint,),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def list_master_datasets(limit: int = 100) -> list[MasterDataset]:
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM master_datasets ORDER BY first_uploaded_at DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close()
        return [_row_to_master(r, None) for r in rows]
    finally:
        conn.close()


def list_copies_for_fingerprint(fingerprint: str, include_deleted: bool = False) -> list[DatasetCopy]:
    conn = _get_connection()
    try:
        cur = conn.cursor()
        if include_deleted:
            cur.execute(
                "SELECT * FROM dataset_copies WHERE fingerprint = %s ORDER BY upload_timestamp DESC",
                (fingerprint,),
            )
        else:
            cur.execute(
                "SELECT * FROM dataset_copies WHERE fingerprint = %s AND deleted_flag = FALSE "
                "ORDER BY upload_timestamp DESC",
                (fingerprint,),
            )
        rows = cur.fetchall()
        cur.close()
        return [
            DatasetCopy(
                copy_id=str(r["copy_id"]), fingerprint=r["fingerprint"], uploaded_filename=r["uploaded_filename"],
                upload_timestamp=r["upload_timestamp"], version=r["version"], user_id=r["user_id"],
                deleted_flag=r["deleted_flag"],
            )
            for r in rows
        ]
    finally:
        conn.close()


def get_master_dataset_row(fingerprint: str) -> dict[str, Any] | None:
    """Raw row (not the hydrated dataclass) — used by reference_counter.py
    for the delete guardrail's storage_location lookup."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM master_datasets WHERE fingerprint = %s", (fingerprint,))
        row = cur.fetchone()
        cur.close()
        return dict(row) if row else None
    finally:
        conn.close()


def begin_dataset_deletion(
    dataset_id: str,
    *,
    requested_by: str = "unknown",
) -> tuple[dict[str, Any] | None, bool]:
    """Atomically tombstone a dataset lineage and remove its live metadata.

    Returns ``(audit, created)``.  A repeated call returns the durable existing
    audit and never recreates or re-deletes metadata.  Cached results and copy
    rows are removed by the master FK cascades; run links are retained but
    their source locator is redacted so historical runs/traces stay linked
    without retaining a route to source bytes.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        _lock_dataset(cur, dataset_id)
        cur.execute(
            "SELECT * FROM dataset_deletion_audits WHERE dataset_id = %s FOR UPDATE",
            (dataset_id,),
        )
        existing = cur.fetchone()
        if existing is not None:
            conn.commit()
            cur.close()
            return dict(existing), False

        cur.execute(
            "SELECT fingerprint, storage_location FROM master_datasets "
            "WHERE dataset_id = %s FOR UPDATE",
            (dataset_id,),
        )
        masters = cur.fetchall()
        if not masters:
            conn.rollback()
            cur.close()
            return None, False

        fingerprints = [row["fingerprint"] for row in masters]
        locations = [row["storage_location"] for row in masters]
        for fingerprint in sorted(fingerprints):
            _lock_fingerprint(cur, fingerprint)
        cur.execute(
            "SELECT COUNT(*) AS n FROM dataset_copies "
            "WHERE fingerprint = ANY(%s) AND deleted_flag = FALSE",
            (fingerprints,),
        )
        active_copies = int(cur.fetchone()["n"])
        cur.execute(
            "SELECT COUNT(*) AS n FROM master_dataset_results WHERE fingerprint = ANY(%s)",
            (fingerprints,),
        )
        cached_results = int(cur.fetchone()["n"])
        cur.execute(
            "SELECT COUNT(*) AS n FROM run_dataset_links WHERE fingerprint = ANY(%s)",
            (fingerprints,),
        )
        historical_runs = int(cur.fetchone()["n"])

        cur.execute(
            """
            INSERT INTO dataset_deletion_audits (
                dataset_id, status, requested_by, fingerprints, source_locations,
                active_copy_count, cached_result_count, historical_run_count,
                physical_cleanup_failures
            ) VALUES (%s, 'deletion_pending_cleanup', %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                dataset_id, requested_by, json.dumps(fingerprints), json.dumps(locations),
                active_copies, cached_results, historical_runs, json.dumps(locations),
            ),
        )
        audit = dict(cur.fetchone())
        cur.execute(
            "UPDATE run_dataset_links SET storage_location = NULL "
            "WHERE fingerprint = ANY(%s)",
            (fingerprints,),
        )
        cur.execute("DELETE FROM master_datasets WHERE dataset_id = %s", (dataset_id,))
        conn.commit()
        cur.close()
        return audit, True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def claim_dataset_cleanup(
    dataset_id: str,
) -> tuple[str, list[str]] | None:
    """Atomically claim one physical-cleanup attempt for this tombstone.

    A claim expires after five minutes so a crashed process cannot strand
    cleanup forever. Concurrent DELETE requests return the pending audit
    instead of both counting or overwriting the same cleanup attempt.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        _lock_dataset(cur, dataset_id)
        cur.execute(
            """
            SELECT *
              FROM dataset_deletion_audits
             WHERE dataset_id = %s
             FOR UPDATE
            """,
            (dataset_id,),
        )
        row = cur.fetchone()
        if row is None or row["status"] == "deleted":
            conn.commit()
            cur.close()
            return None
        if (
            row.get("cleanup_claim_id")
            and row.get("cleanup_claimed_at")
            and (datetime.now(row["cleanup_claimed_at"].tzinfo) - row["cleanup_claimed_at"]).total_seconds() < 300
        ):
            conn.commit()
            cur.close()
            return None

        claim_id = str(uuid4())
        locations = list(row["physical_cleanup_failures"] or [])
        cur.execute(
            """
            UPDATE dataset_deletion_audits
               SET cleanup_claim_id = %s,
                   cleanup_claimed_at = now()
             WHERE dataset_id = %s
            """,
            (claim_id, dataset_id),
        )
        conn.commit()
        cur.close()
        return claim_id, locations
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def finish_dataset_deletion(
    dataset_id: str,
    *,
    claim_id: str,
    failed_locations: list[str],
) -> dict[str, Any]:
    """Persist physical cleanup outcome and return the updated audit."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        status = "deletion_pending_cleanup" if failed_locations else "deleted"
        cur.execute(
            """
            UPDATE dataset_deletion_audits
               SET status = %s,
                    completed_at = CASE WHEN %s THEN NULL ELSE now() END,
                    physical_deleted_count =
                        jsonb_array_length(source_locations) - %s,
                    physical_cleanup_failures = %s,
                    cleanup_claim_id = NULL,
                    cleanup_claimed_at = NULL
             WHERE dataset_id = %s
               AND cleanup_claim_id = %s
             RETURNING *
            """,
            (
                status,
                bool(failed_locations),
                len(failed_locations),
                json.dumps(failed_locations),
                dataset_id,
                claim_id,
            ),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        if row is None:
            raise RuntimeError(f"Deletion audit for dataset '{dataset_id}' disappeared.")
        return dict(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_master_dataset(fingerprint: str) -> None:
    """Hard delete — cascades to dataset_copies and master_dataset_results
    via ON DELETE CASCADE. Callers (reference_counter.can_delete) are
    responsible for the force-unless-referenced guardrail before calling
    this."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        # Preserve historical run linkage while making the deleted source
        # unavailable.  This also keeps the legacy fingerprint API compatible
        # after the FK-removal migration above.
        cur.execute(
            "UPDATE run_dataset_links SET storage_location = NULL WHERE fingerprint = %s",
            (fingerprint,),
        )
        cur.execute("DELETE FROM master_datasets WHERE fingerprint = %s", (fingerprint,))
        conn.commit()
        cur.close()
    finally:
        conn.close()
