"""Resolve a completed run back to the dataset it was built from.

Why this exists
---------------
Ad-hoc aggregation and drill-down need the rows again *after* the run has
finished. Until now the only copy Agent 2 could reach was the scratch copy it
wrote under ``TEMP_STORAGE_DIR/<run_id>/`` while profiling, which is swept after
``RUN_FILE_RETENTION_HOURS``. That made every post-run question time-bounded by
construction: once the sweep ran, drill-down could only be served from a
pre-aggregated cube, and a cube only answers the exact breakdown it was built
for.

The bytes were never actually gone. The orchestrator's Dataset Registry stores
every upload content-addressed by SHA-256 and never purges it. What was missing
was a *reference* from the run to that permanent copy. A run now records one
(``profile_runs.source_fingerprint`` / ``source_dataset_uri`` /
``source_sheet_name``), and this resolver turns it back into a readable file.

Resolution order (first hit wins, most local first):
  1. the run's retained scratch copy — no I/O beyond a directory listing;
  2. the master dataset file on a shared filesystem, addressed by fingerprint;
  3. an explicit local path / ``file://`` URI recorded on the run;
  4. an HTTP fetch of the master bytes, cached on disk so a drill session pays
     the download once.

Every step is optional and independently configurable, so a deployment where
Agent 2 and the orchestrator share a disk needs no network hop, and one where
they do not still works. If none of them can produce a file the caller gets
``None`` and must say so plainly — never a substituted or approximated answer.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.ingestion.temporary_storage import TemporaryStorage

logger = get_logger(__name__)

# A content fingerprint is a SHA-256 hex digest and nothing else. It is
# interpolated into a filesystem path, so it is validated as such rather than
# trusted because of where it came from.
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")

# The loaders only understand these. An extension outside the set cannot be
# read anyway, and must not be used to build a path.
ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls")

_MASTER_CACHE_DIRNAME = "_master_cache"

# Reading a whole file to hash it is cheap next to the aggregation that follows,
# but a drill session asks repeatedly. Memoised on identity-plus-mtime so an
# edited or swapped file is re-hashed rather than trusted.
_DIGEST_MEMO: dict[tuple[str, int, int], str] = {}
_DIGEST_MEMO_LIMIT = 256


def _sha256_of(path: Path) -> str | None:
    """Content digest of a file, or None if it cannot be read."""
    try:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return None
    cached = _DIGEST_MEMO.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    if len(_DIGEST_MEMO) >= _DIGEST_MEMO_LIMIT:
        _DIGEST_MEMO.clear()
    value = digest.hexdigest()
    _DIGEST_MEMO[key] = value
    return value


@dataclass(frozen=True)
class SourceDatasetReference:
    """A durable pointer to the bytes a run was built from."""

    fingerprint: str | None = None
    uri: str | None = None
    filename: str | None = None
    sheet_name: str | None = None

    @property
    def is_empty(self) -> bool:
        return not (self.fingerprint or self.uri)

    def merged_with(self, fallback: "SourceDatasetReference | None") -> "SourceDatasetReference":
        """Fill blanks from another reference without overwriting anything set."""
        if fallback is None:
            return self
        return SourceDatasetReference(
            fingerprint=self.fingerprint or fallback.fingerprint,
            uri=self.uri or fallback.uri,
            filename=self.filename or fallback.filename,
            sheet_name=self.sheet_name or fallback.sheet_name,
        )


@dataclass(frozen=True)
class ResolvedSource:
    """A readable copy of a run's source dataset, and where it came from."""

    path: Path
    origin: str
    sheet_name: str | None
    fingerprint: str | None

    @property
    def is_retained_run_file(self) -> bool:
        return self.origin == "retained_run_file"

    @property
    def provenance(self) -> str:
        return {
            "retained_run_file": "the source rows retained from this run",
            "master_dataset_file": "the stored source dataset for this run",
            "master_dataset_download": "the stored source dataset for this run",
        }.get(self.origin, "the source rows for this run")


def _extension_for(reference: SourceDatasetReference) -> str | None:
    """Pick the file extension to address the master copy with."""
    for candidate in (reference.uri, reference.filename):
        if not candidate:
            continue
        suffix = Path(urlparse(candidate).path if "://" in candidate else candidate).suffix.lower()
        if suffix in ALLOWED_EXTENSIONS:
            return suffix
    return None


class SourceDatasetResolver:
    """Turns a run's stored dataset reference back into a readable file."""

    def __init__(self, settings: Settings, temp_storage: TemporaryStorage | None = None):
        self._settings = settings
        self._temp_storage = temp_storage or TemporaryStorage(settings)

    # ------------------------------------------------------------------ #
    # Public                                                             #
    # ------------------------------------------------------------------ #

    def resolve(
        self,
        run_id: uuid.UUID,
        reference: SourceDatasetReference | None = None,
    ) -> ResolvedSource | None:
        """Return a readable source file for this run, or None if there is none."""
        reference = reference or SourceDatasetReference()

        retained = self._retained_run_file(run_id)
        if retained is not None:
            return retained

        if reference.is_empty:
            return None

        for step in (self._master_file_by_fingerprint, self._file_from_uri, self._download_master):
            try:
                resolved = step(reference)
            except Exception as e:  # noqa: BLE001 — a broken step must not mask the next one
                logger.warning(
                    "source_dataset_resolution_step_failed",
                    run_id=str(run_id),
                    step=step.__name__,
                    error=str(e),
                )
                continue
            if resolved is not None:
                logger.info(
                    "source_dataset_resolved",
                    run_id=str(run_id),
                    origin=resolved.origin,
                    path=str(resolved.path),
                )
                return resolved
        return None

    # ------------------------------------------------------------------ #
    # Steps                                                              #
    # ------------------------------------------------------------------ #

    def _allowed_roots(self) -> list[Path]:
        """Directories this resolver is permitted to read a dataset from.

        A reference reaches us from the database, but it was *written* from a
        request field, so it is treated as caller-controlled throughout. Only
        our own storage is in scope: the master store and Agent 2's own temp
        area. Anything else is refused rather than opened.
        """
        roots: list[Path] = []
        for value in (
            getattr(self._settings, "master_dataset_storage_dir", ""),
            getattr(self._settings, "temp_storage_dir", ""),
        ):
            candidate = (value or "").strip()
            if not candidate:
                continue
            try:
                roots.append(Path(candidate).resolve(strict=False))
            except OSError:
                continue
        return roots

    def _is_within_allowed_roots(self, path: Path) -> bool:
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            return False
        for root in self._allowed_roots():
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _content_matches(self, path: Path, fingerprint: str | None) -> bool:
        """True when the bytes are the ones the run was built from.

        A path can be right while its contents are not — a master file replaced
        underneath us, a cached download from an earlier, different reference.
        The fingerprint is the run's own claim about its data, so it is checked
        against the bytes rather than assumed from the filename.
        """
        expected = (fingerprint or "").strip().lower()
        if not _FINGERPRINT.match(expected):
            return False
        actual = _sha256_of(path)
        if actual == expected:
            return True
        logger.warning(
            "source_dataset_content_mismatch",
            path=str(path),
            expected_fingerprint=expected,
            actual_fingerprint=actual,
        )
        return False

    def _retained_run_file(self, run_id: uuid.UUID) -> ResolvedSource | None:
        path = self._temp_storage.find_run_file(run_id)
        if path is None:
            return None
        sheet_name = self._temp_storage.read_run_metadata(run_id).get("sheet_name")
        return ResolvedSource(
            path=path, origin="retained_run_file", sheet_name=sheet_name, fingerprint=None
        )

    def _master_file_by_fingerprint(self, reference: SourceDatasetReference) -> ResolvedSource | None:
        base_dir = (getattr(self._settings, "master_dataset_storage_dir", "") or "").strip()
        fingerprint = (reference.fingerprint or "").strip().lower()
        if not base_dir or not _FINGERPRINT.match(fingerprint):
            return None

        extension = _extension_for(reference)
        candidates = [extension] if extension else list(ALLOWED_EXTENSIONS)
        # Mirrors Agent-Orchestrator storage.master_file_path: a two-character
        # shard directory keyed by the fingerprint prefix.
        shard = Path(base_dir) / fingerprint[:2]
        for suffix in candidates:
            path = shard / f"{fingerprint}{suffix}"
            if path.is_file() and self._content_matches(path, fingerprint):
                return ResolvedSource(
                    path=path,
                    origin="master_dataset_file",
                    sheet_name=reference.sheet_name,
                    fingerprint=fingerprint,
                )
        return None

    def _file_from_uri(self, reference: SourceDatasetReference) -> ResolvedSource | None:
        uri = (reference.uri or "").strip()
        if not uri:
            return None

        parsed = urlparse(uri)
        if parsed.scheme in ("http", "https"):
            return None  # handled by _download_master
        if parsed.scheme == "file":
            raw = unquote(parsed.path)
            # file:///C:/... on Windows arrives with a leading slash before the drive.
            if re.match(r"^/[A-Za-z]:", raw):
                raw = raw[1:]
            path = Path(raw)
        elif parsed.scheme and len(parsed.scheme) > 1:
            return None  # an unknown scheme is not a local path
        else:
            path = Path(uri)

        if path.suffix.lower() not in ALLOWED_EXTENSIONS or not path.is_file():
            return None
        # A recorded URI is a hint about where our own copy lives, never a
        # licence to read the filesystem. Outside the configured roots it is
        # refused, and inside them the bytes must still be the run's own.
        if not self._is_within_allowed_roots(path):
            logger.warning("source_dataset_uri_outside_allowed_roots", path=str(path))
            return None
        if not self._content_matches(path, reference.fingerprint):
            return None
        return ResolvedSource(
            path=path,
            origin="master_dataset_file",
            sheet_name=reference.sheet_name,
            fingerprint=reference.fingerprint,
        )

    def _download_master(self, reference: SourceDatasetReference) -> ResolvedSource | None:
        url = self._download_url(reference)
        if url is None:
            return None

        fingerprint = (reference.fingerprint or "").strip().lower()
        if not _FINGERPRINT.match(fingerprint):
            return None
        extension = _extension_for(reference) or ".csv"
        cached = self._cache_path(fingerprint, extension)
        if cached.is_file() and self._content_matches(cached, fingerprint):
            return ResolvedSource(
                path=cached,
                origin="master_dataset_download",
                sheet_name=reference.sheet_name,
                fingerprint=fingerprint,
            )

        import httpx

        timeout = float(getattr(self._settings, "master_dataset_fetch_timeout_seconds", 30.0))
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=False) as response:
            if response.status_code != 200:
                logger.warning(
                    "master_dataset_fetch_rejected", url=url, status_code=response.status_code
                )
                return None
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cached.parent / f".{cached.name}.partial"
            digest = hashlib.sha256()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    digest.update(chunk)
                    handle.write(chunk)

        # Verify before the file is visible under its content address, so a
        # mismatched download can never be picked up by the cache branch above.
        if digest.hexdigest() != fingerprint:
            logger.warning(
                "master_dataset_fetch_content_mismatch",
                url=url,
                expected_fingerprint=fingerprint,
                actual_fingerprint=digest.hexdigest(),
            )
            tmp_path.unlink(missing_ok=True)
            return None
        shutil.move(str(tmp_path), str(cached))

        return ResolvedSource(
            path=cached,
            origin="master_dataset_download",
            sheet_name=reference.sheet_name,
            fingerprint=fingerprint,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _download_url(self, reference: SourceDatasetReference) -> str | None:
        """Build the fetch URL from configuration, never from the reference.

        Honouring an http(s) URI recorded on the run would let a request choose
        what this service connects to — the classic server-side request forgery
        shape. The only address we will fetch is the configured orchestrator
        endpoint, addressed by a validated fingerprint.
        """
        template = (getattr(self._settings, "master_dataset_content_url", "") or "").strip()
        fingerprint = (reference.fingerprint or "").strip().lower()
        if not template or not _FINGERPRINT.match(fingerprint):
            return None
        if "{fingerprint}" not in template:
            template = template.rstrip("/") + "/{fingerprint}/content"
        url = template.replace("{fingerprint}", fingerprint)
        return url if urlparse(url).scheme in ("http", "https") else None

    def _cache_path(self, fingerprint: str, extension: str) -> Path:
        base = Path(self._settings.temp_storage_dir) / _MASTER_CACHE_DIRNAME
        if not _FINGERPRINT.match(fingerprint) or extension not in ALLOWED_EXTENSIONS:
            raise ValueError("refusing to build a cache path from an unvalidated reference")
        return base / f"{fingerprint}{extension}"
