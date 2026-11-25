"""Stage 0A — reference counting and the master-deletion guardrail.

A Master Dataset's reference_count tracks how many non-deleted
DatasetCopy rows point at it. Deleting a copy is always safe and cheap
(decrement + soft-delete the copy row) — the Master Dataset itself is
only ever removed by an explicit, separate admin action, and that action
refuses unless forced when copies still reference it (confirmed with the
user: "refuse unless forced" over "always allow").
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.dataset_registry import metadata_cache, storage

logger = logging.getLogger(__name__)


def _connection():
    return metadata_cache.get_connection()


def increment(fingerprint: str) -> None:
    conn = _connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE master_datasets SET reference_count = reference_count + 1 WHERE fingerprint = %s",
            (fingerprint,),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def decrement(fingerprint: str) -> None:
    conn = _connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE master_datasets SET reference_count = GREATEST(reference_count - 1, 0) WHERE fingerprint = %s",
            (fingerprint,),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def active_copy_count(fingerprint: str) -> int:
    conn = _connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS n FROM dataset_copies WHERE fingerprint = %s AND deleted_flag = FALSE",
            (fingerprint,),
        )
        row = cur.fetchone()
        cur.close()
        return int(row["n"])
    finally:
        conn.close()


def soft_delete_copy(copy_id: str) -> str | None:
    """Soft-deletes the copy row and decrements the master's reference
    count. Returns the copy's fingerprint on success, None if the copy
    didn't exist (or was already deleted)."""
    conn = _connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE dataset_copies SET deleted_flag = TRUE "
            "WHERE copy_id = %s AND deleted_flag = FALSE RETURNING fingerprint",
            (copy_id,),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
    finally:
        conn.close()
    if row is None:
        return None
    fingerprint = row["fingerprint"]
    decrement(fingerprint)
    return fingerprint


def can_delete_master(fingerprint: str, force: bool) -> tuple[bool, str | None]:
    """The delete guardrail: refuse unless forced when active copies still
    reference this Master Dataset. Returns (allowed, reason_if_refused)."""
    count = active_copy_count(fingerprint)
    if count > 0 and not force:
        return False, (
            f"Master Dataset '{fingerprint}' still has {count} active copy/copies referencing it — "
            f"pass force=true to delete it anyway, or delete those copies first."
        )
    return True, None


def delete_master(fingerprint: str, force: bool) -> tuple[bool, str | None]:
    """Enforces the guardrail, then deletes the DB row (cascades to
    dataset_copies/master_dataset_results) and best-effort removes the
    physical file — DB commit happens first, file removal is best-effort
    and never blocks or reverses the DB delete (see storage.py)."""
    allowed, reason = can_delete_master(fingerprint, force)
    if not allowed:
        return False, reason

    row = metadata_cache.get_master_dataset_row(fingerprint)
    if row is None:
        return False, f"Master Dataset '{fingerprint}' not found."

    metadata_cache.delete_master_dataset(fingerprint)
    storage.delete_master_file(row["storage_location"])
    return True, None


def delete_dataset(
    dataset_id: str,
    *,
    requested_by: str = "unknown",
) -> dict[str, Any] | None:
    """Delete every stored version of a dataset_id and report cleanup status.

    Logical deletion and its tombstone commit together.  Physical deletion is
    then attempted through the storage abstraction.  Failed objects remain in
    the audit as explicit orphans and are retried by an identical DELETE.
    """
    audit, created = metadata_cache.begin_dataset_deletion(
        dataset_id,
        requested_by=requested_by,
    )
    if audit is None:
        return None

    status = audit["status"]
    if status == "deleted":
        result = dict(audit)
        result["idempotent"] = True
        return result

    claim = metadata_cache.claim_dataset_cleanup(dataset_id)
    if claim is None:
        result = dict(audit)
        result["idempotent"] = True
        return result

    claim_id, locations = claim
    failed: list[str] = []
    for location in locations:
        if not storage.delete_master_file(location):
            failed.append(location)

    updated = metadata_cache.finish_dataset_deletion(
        dataset_id,
        claim_id=claim_id,
        failed_locations=failed,
    )
    updated["idempotent"] = not created
    return updated
