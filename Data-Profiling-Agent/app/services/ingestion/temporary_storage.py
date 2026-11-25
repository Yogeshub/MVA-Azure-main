"""Temporary file storage — run-scoped directory with cleanup guarantees."""

import json
import time
import uuid
import shutil
from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class TemporaryStorage:
    """Manages run-scoped temporary directories for uploaded files."""

    # Sidecar holding what the loader needs to reproduce the original read
    # (e.g. which worksheet was profiled). Kept beside the upload so no schema
    # change is needed to re-open a retained run.
    META_FILENAME = ".run_meta.json"

    def __init__(self, settings: Settings):
        self._base_dir = Path(settings.temp_storage_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def create_run_directory(self, run_id: uuid.UUID) -> Path:
        """Create an isolated temporary directory for a processing run."""
        run_dir = self._base_dir / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_upload(self, run_id: uuid.UUID, content: bytes, safe_filename: str) -> Path:
        """
        Save uploaded file content to the run directory.

        Uses a sanitized filename to prevent path traversal.
        The original filename is NOT used as-is on the filesystem.
        """
        run_dir = self.create_run_directory(run_id)

        # Use only the base name, strip path components
        safe_name = Path(safe_filename).name
        # Additional safety: replace any remaining suspicious chars
        safe_name = safe_name.replace("..", "").replace("/", "").replace("\\", "")
        if not safe_name:
            safe_name = f"upload_{run_id.hex[:8]}"

        file_path = run_dir / safe_name
        file_path.write_bytes(content)

        logger.info(
            "file_saved",
            run_id=str(run_id),
            path=str(file_path),
            size_bytes=len(content),
        )
        return file_path

    def cleanup_run(self, run_id: uuid.UUID) -> bool:
        """
        Delete the entire run directory and its contents.

        Returns True if cleanup succeeded, False if directory didn't exist.
        """
        run_dir = self._base_dir / str(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
            logger.info("run_cleanup_complete", run_id=str(run_id))
            return True
        return False

    def get_run_directory(self, run_id: uuid.UUID) -> Path | None:
        """Get the run directory path if it exists."""
        run_dir = self._base_dir / str(run_id)
        return run_dir if run_dir.exists() else None

    def cleanup_all(self) -> int:
        """Remove all temporary directories. Returns count of removed dirs."""
        if not self._base_dir.exists():
            return 0
        count = 0
        for child in self._base_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                count += 1
        return count

    def find_run_file(self, run_id: uuid.UUID) -> Path | None:
        """Return the retained source file for a run, if it still exists.

        Used by ad-hoc aggregation to go back to the rows after profiling has
        finished. Returns None once the retention window has passed — callers
        must treat that as "cannot answer", never as "no data".
        """
        run_dir = self._base_dir / str(run_id)
        if not run_dir.exists():
            return None
        for child in sorted(run_dir.iterdir()):
            if child.is_file() and child.name != self.META_FILENAME:
                return child
        return None

    def write_run_metadata(self, run_id: uuid.UUID, meta: dict) -> None:
        """Record how the upload was read, so a retained run can be re-opened.

        Without this, a multi-sheet workbook could be profiled but never
        re-aggregated: the loader would not know which sheet was used.
        """
        run_dir = self.create_run_directory(run_id)
        try:
            (run_dir / self.META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")
        except OSError as e:
            logger.warning("run_metadata_write_failed", run_id=str(run_id), error=str(e))

    def read_run_metadata(self, run_id: uuid.UUID) -> dict:
        """Read the sidecar written at upload time. Empty dict if absent."""
        path = self._base_dir / str(run_id) / self.META_FILENAME
        if not path.exists():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError) as e:
            logger.warning("run_metadata_read_failed", run_id=str(run_id), error=str(e))
            return {}

    def purge_expired(self, max_age_hours: int) -> int:
        """Delete run directories older than the retention window.

        Retention is swept opportunistically (on each new run) rather than by a
        scheduler, so a long-idle service never accumulates files silently.
        Returns the number of directories removed.
        """
        if not self._base_dir.exists() or max_age_hours <= 0:
            return 0
        cutoff = time.time() - (max_age_hours * 3600)
        removed = 0
        for child in self._base_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        if removed:
            logger.info("run_files_purged", removed=removed, max_age_hours=max_age_hours)
        return removed
