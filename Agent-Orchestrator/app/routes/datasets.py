"""Dataset Registry admin routes — list Master Datasets, list/delete
copies, delete a Master Dataset (guarded — see reference_counter.py).
Read-only listing + lifecycle management only; uploads still go through
POST /pipeline/run, which is where Stage 0A's resolve_dataset_identity
node actually registers new content.
"""

import re
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.services.dataset_registry import blob_storage, metadata_cache, reference_counter

# A fingerprint is a SHA-256 hex digest. It reaches the filesystem, so it is
# validated as one rather than trusted for having come through a route.
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")

router = APIRouter(prefix="/datasets", tags=["Dataset Registry"])


class MasterDatasetOut(BaseModel):
    fingerprint: str
    dataset_id: str
    original_filename: str
    file_extension: str
    byte_size: int
    row_count: int | None
    column_count: int | None
    latest_version: int
    previous_fingerprint: str | None
    reference_count: int
    first_uploaded_at: str
    last_referenced_at: str
    has_cached_results: bool


class DatasetCopyOut(BaseModel):
    copy_id: str
    fingerprint: str
    uploaded_filename: str
    upload_timestamp: str
    version: int
    user_id: str | None
    deleted_flag: bool


class DatasetDeletionOut(BaseModel):
    dataset_id: str
    status: str
    idempotent: bool
    fingerprints: list[str]
    active_copy_count: int
    cached_result_count: int
    historical_run_count: int
    physical_deleted_count: int
    orphaned_object_count: int
    requested_at: str
    completed_at: str | None


@router.get("/masters", response_model=list[MasterDatasetOut])
def list_masters(limit: int = Query(default=100, le=500)):
    masters = metadata_cache.list_master_datasets(limit=limit)
    return [
        MasterDatasetOut(
            fingerprint=m.fingerprint, dataset_id=m.dataset_id, original_filename=m.original_filename,
            file_extension=m.file_extension, byte_size=m.byte_size, row_count=m.row_count,
            column_count=m.column_count, latest_version=m.latest_version,
            previous_fingerprint=m.previous_fingerprint, reference_count=m.reference_count,
            first_uploaded_at=m.first_uploaded_at.isoformat(), last_referenced_at=m.last_referenced_at.isoformat(),
            has_cached_results=m.has_cached_results,
        )
        for m in masters
    ]


@router.get("/masters/{fingerprint}/content")
def get_master_content(fingerprint: str):
    """Serve the stored bytes of a Master Dataset.

    One caller: Agent 2, recomputing a drill-down for a run whose retained
    working copy has aged out, when the two services do not share a filesystem.
    Where they do share one, Agent 2 reads the file directly and never comes
    here. Same trust boundary as the rest of this router — it is not exposed
    through the public API proxy.
    """
    if not _FINGERPRINT.match(fingerprint):
        raise HTTPException(status_code=400, detail="Not a valid dataset fingerprint.")

    row = metadata_cache.get_master_dataset_row(fingerprint)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No Master Dataset with fingerprint {fingerprint}.")

    location = row["storage_location"]
    filename = f"{fingerprint}{row['file_extension']}"

    # Under the blob backend there is no local path to serve — stream the
    # object through instead. Same contract either way: 410 when the row
    # promises bytes that are not actually there.
    if blob_storage.is_blob_uri(location):
        chunks = blob_storage.open_master_stream(location)
        if chunks is None:
            raise HTTPException(
                status_code=410,
                detail=f"The stored object for {fingerprint} is missing or unreachable.",
            )
        return StreamingResponse(
            chunks,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    path = Path(location)
    if not path.is_file():
        # The row promises bytes that are not there. Say so plainly rather than
        # letting the caller read it as "this dataset has no rows".
        raise HTTPException(
            status_code=410,
            detail=f"The stored file for {fingerprint} is missing from {path.parent}.",
        )

    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=filename,
    )


@router.get("/masters/{fingerprint}/copies", response_model=list[DatasetCopyOut])
def list_copies(fingerprint: str, include_deleted: bool = Query(default=False)):
    copies = metadata_cache.list_copies_for_fingerprint(fingerprint, include_deleted=include_deleted)
    return [
        DatasetCopyOut(
            copy_id=c.copy_id, fingerprint=c.fingerprint, uploaded_filename=c.uploaded_filename,
            upload_timestamp=c.upload_timestamp.isoformat(), version=c.version, user_id=c.user_id,
            deleted_flag=c.deleted_flag,
        )
        for c in copies
    ]


@router.delete("/copies/{copy_id}")
def delete_copy(copy_id: str):
    """Soft-deletes this copy and decrements the Master Dataset's
    reference_count. Never touches the Master Dataset or its physical
    file."""
    fingerprint = reference_counter.soft_delete_copy(copy_id)
    if fingerprint is None:
        raise HTTPException(status_code=404, detail=f"Copy '{copy_id}' not found or already deleted.")
    return {"copy_id": copy_id, "fingerprint": fingerprint, "deleted": True}


@router.delete("/masters/{fingerprint}")
def delete_master(fingerprint: str, force: bool = Query(default=False)):
    """Refuses unless forced when active copies still reference this
    Master Dataset (see reference_counter.can_delete_master)."""
    ok, reason = reference_counter.delete_master(fingerprint, force=force)
    if not ok:
        status_code = 404 if reason and "not found" in reason else 409
        raise HTTPException(status_code=status_code, detail=reason)
    return {"fingerprint": fingerprint, "deleted": True}


@router.delete("/{dataset_id}", response_model=DatasetDeletionOut)
def delete_dataset(
    dataset_id: str,
    confirm: bool = Query(...),
    deletion_actor: str | None = Header(default=None, alias="X-Deletion-Actor"),
):
    """Permanently delete all source versions for one stable dataset_id.

    ``confirm=true`` is deliberately mandatory.  Historical run linkage and
    deletion audit data remain, but source bytes, active copies, and cached
    derived master results are removed.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Permanent dataset deletion requires confirm=true.",
        )
    try:
        normalized_id = str(UUID(dataset_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="Not a valid dataset_id.") from None

    actor = (
        deletion_actor.strip()[:200]
        if isinstance(deletion_actor, str) and deletion_actor.strip()
        else "direct-orchestrator-client"
    )
    audit = reference_counter.delete_dataset(normalized_id, requested_by=actor)
    if audit is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{normalized_id}' not found.")

    failures = audit.get("physical_cleanup_failures") or []
    return DatasetDeletionOut(
        dataset_id=str(audit["dataset_id"]),
        status=audit["status"],
        idempotent=bool(audit["idempotent"]),
        fingerprints=list(audit["fingerprints"]),
        active_copy_count=audit["active_copy_count"],
        cached_result_count=audit["cached_result_count"],
        historical_run_count=audit["historical_run_count"],
        physical_deleted_count=audit["physical_deleted_count"],
        orphaned_object_count=len(failures),
        requested_at=audit["requested_at"].isoformat(),
        completed_at=audit["completed_at"].isoformat() if audit["completed_at"] else None,
    )
