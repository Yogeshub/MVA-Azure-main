"""Stage 0A — content-addressed physical storage for Master Dataset bytes.

Layout mirrors a git-object-store: <hash[:2]>/<hash>.<ext> under
settings.MASTER_DATASET_STORAGE_DIR. The original file extension is
preserved because Agent 1's loader (validate_and_load) branches on
filename extension to pick pd.read_csv vs pd.read_excel.

Two backends live behind this module, chosen by MASTER_DATASET_BACKEND:
the local filesystem (default) and Azure Blob Storage (see blob_storage.py).
Callers do not know which is in use — they hand over bytes and get back an
opaque ``storage_location`` string, and only this module and blob_storage
interpret it. The blob backend exists because on Azure the orchestrator and
Agent 2 have separate filesystems, so a path written here would be
unreadable to the service that needs the rows back for drill-down.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from app.services.dataset_registry import blob_storage

logger = logging.getLogger(__name__)


def master_file_path(base_dir: str, fingerprint: str, file_extension: str) -> Path:
    ext = file_extension if file_extension.startswith(".") else f".{file_extension}"
    return Path(base_dir) / fingerprint[:2] / f"{fingerprint}{ext}"


def write_master_file(base_dir: str, fingerprint: str, file_extension: str, content: bytes) -> str:
    """Atomic write: temp file in the same directory, then os.replace() —
    safe if two concurrent uploads of identical new content race each
    other (both write byte-identical content to the same final path;
    whichever replace() lands last wins, harmlessly, since the bytes are
    identical either way).

    Under the blob backend the same guarantee holds for a different reason:
    the blob name is the digest of the content, so a concurrent overwrite can
    only ever write identical bytes."""
    if blob_storage.is_enabled():
        logger.info(
            "master_dataset_storage_dispatch | backend=blob | fingerprint=%s",
            fingerprint,
        )
        return blob_storage.upload_master(fingerprint, file_extension, content)

    logger.info(
        "master_dataset_storage_dispatch | backend=local | fingerprint=%s | base_dir=%s",
        fingerprint,
        base_dir,
    )
    path = master_file_path(base_dir, fingerprint, file_extension)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{fingerprint}.tmp"
    tmp_path.write_bytes(content)
    os.replace(tmp_path, path)
    return str(path)


def delete_master_file(storage_location: str) -> bool:
    """Best-effort delete — never raises. Called only after the DB row is
    already gone, so a leftover orphaned file is the safe failure mode,
    never a DB row pointing at nothing."""
    if blob_storage.is_blob_uri(storage_location):
        return blob_storage.delete_master(storage_location)
    try:
        Path(storage_location).unlink(missing_ok=True)
        return True
    except OSError as e:
        logger.warning(f"Failed to delete master file at {storage_location}: {e}")
        return False
