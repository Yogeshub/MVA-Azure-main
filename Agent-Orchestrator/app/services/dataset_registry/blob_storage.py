"""Stage 0A — Azure Blob Storage backend for Master Dataset bytes.

The local backend in ``storage.py`` writes ``<hash[:2]>/<hash>.<ext>`` under a
directory on disk. That works when every service that needs the bytes shares
that disk. On Azure they do not: the orchestrator and Agent 2 are separate
apps, each with its own ephemeral filesystem, so a path written by one is
meaningless to the other.

This backend keeps exactly the same content-addressed layout, in a Blob
container both services can reach. A master's ``storage_location`` then
becomes ``azure-blob://<container>/<name>`` instead of a filesystem path --
still an opaque locator as far as the rest of the registry is concerned, and
still resolved only through this module.

Selected by ``MASTER_DATASET_BACKEND=blob``. At the default (``local``)
nothing here runs and the Azure SDK is never imported, so local development is
unaffected and does not need the SDK installed.

Azure extras live with the shared dependency set:
``pip install -r Data-Profiling-Agent/requirements-azure.txt``
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Iterator

from app.config import settings

logger = logging.getLogger(__name__)

BLOB_URI_SCHEME = "azure-blob"

# Every component below is interpolated into a blob name. A name is not a
# filesystem path, but it is still an address built from stored input, and the
# cheapest place to refuse a malformed one is before it is built.
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls")

_client_lock = threading.Lock()
_client_cache: dict[str, Any] = {}
_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message)


def is_enabled() -> bool:
    """True when master datasets are stored in Blob rather than on disk."""
    return (settings.MASTER_DATASET_BACKEND or "local").strip().lower() == "blob"


def is_blob_uri(storage_location: str | None) -> bool:
    return isinstance(storage_location, str) and storage_location.startswith(
        f"{BLOB_URI_SCHEME}://"
    )


def parse_blob_uri(storage_location: str) -> tuple[str, str] | None:
    """Split ``azure-blob://container/name`` into its two parts."""
    if not is_blob_uri(storage_location):
        return None
    remainder = storage_location[len(BLOB_URI_SCHEME) + 3 :]
    container, _, name = remainder.partition("/")
    if not container or not name:
        return None
    return container, name


def _safe_prefix(raw: str | None) -> str:
    """A configured prefix, refused if it could climb out of its own namespace."""
    prefix = (raw or "").strip().strip("/")
    if not prefix:
        return ""
    if "\\" in prefix or ".." in prefix.split("/"):
        raise ValueError(f"AZURE_BLOB_PREFIX is not a usable blob prefix: {raw!r}")
    return prefix


def blob_name(fingerprint: str, file_extension: str) -> str:
    """The content address of a master dataset within the container.

    Mirrors ``storage.master_file_path`` so the two backends address the same
    dataset the same way, and a local store can be lifted into a container by
    a straight recursive copy. Raises ValueError on anything that is not a
    well-formed address -- on the write path that is fatal by design, since a
    dataset that cannot be addressed must not be registered.
    """
    ext = file_extension if file_extension.startswith(".") else f".{file_extension}"
    if not _FINGERPRINT.match(fingerprint or ""):
        raise ValueError("a master dataset is addressed by a SHA-256 digest and nothing else")
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"unsupported master dataset extension: {file_extension!r}")
    prefix = _safe_prefix(settings.AZURE_BLOB_PREFIX)
    shard = f"{fingerprint[:2]}/{fingerprint}{ext}"
    return f"{prefix}/{shard}" if prefix else shard


def _container_client(container_name: str | None = None) -> Any | None:
    """A cached container client, or None if Blob access is not usable."""
    container = (container_name or settings.AZURE_BLOB_CONTAINER or "").strip()
    account_url = (settings.AZURE_BLOB_ACCOUNT_URL or "").strip()
    connection_string = (settings.AZURE_BLOB_CONNECTION_STRING or "").strip()

    if not container or not (account_url or connection_string):
        _warn_once(
            "unconfigured",
            "MASTER_DATASET_BACKEND=blob but AZURE_BLOB_CONTAINER and one of "
            "AZURE_BLOB_ACCOUNT_URL / AZURE_BLOB_CONNECTION_STRING are not set.",
        )
        return None

    cache_key = f"{account_url}|{container}|{bool(connection_string)}"
    with _client_lock:
        cached = _client_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            _warn_once(
                "sdk_missing",
                "azure-storage-blob is not installed; install the Azure extras: "
                "pip install -r Data-Profiling-Agent/requirements-azure.txt",
            )
            return None

        try:
            if connection_string:
                service = BlobServiceClient.from_connection_string(connection_string)
            else:
                from azure.identity import DefaultAzureCredential

                client_id = (settings.AZURE_BLOB_CLIENT_ID or "").strip()
                credential = (
                    DefaultAzureCredential(managed_identity_client_id=client_id)
                    if client_id
                    else DefaultAzureCredential()
                )
                service = BlobServiceClient(account_url=account_url, credential=credential)
            client = service.get_container_client(container)
        except Exception as e:  # noqa: BLE001
            _warn_once("client_failed", f"Failed to create Blob client: {e}")
            return None

        _client_cache[cache_key] = client
        return client


def upload_master(fingerprint: str, file_extension: str, content: bytes) -> str:
    """Store master bytes and return their ``azure-blob://`` locator.

    Raises on failure. Unlike a read, a failed *write* must not be swallowed:
    the caller is about to insert a database row promising these bytes exist,
    and a row pointing at nothing is the one outcome worse than a failed
    upload. Overwrite is safe by construction -- the name is the SHA-256 of
    the content, so any existing blob at that address holds identical bytes.
    """
    configured_container = (settings.AZURE_BLOB_CONTAINER or "").strip() or "<missing>"
    try:
        name = blob_name(fingerprint, file_extension)
    except Exception:
        logger.exception(
            "master_blob_upload_failed | stage=build_address | container=%s | "
            "extension=%s | bytes=%d | fingerprint=%s",
            configured_container,
            file_extension,
            len(content),
            fingerprint,
        )
        raise

    logger.info(
        "master_blob_upload_started | stage=connect | container=%s | blob=%s | "
        "bytes=%d | fingerprint=%s",
        configured_container,
        name,
        len(content),
        fingerprint,
    )

    try:
        container = _container_client()
        if container is None:
            raise RuntimeError(
                "MASTER_DATASET_BACKEND=blob but Azure Blob Storage is not configured or "
                "unreachable; refusing to register a Master Dataset with no stored bytes."
            )
        container_name = container.container_name
        logger.info(
            "master_blob_upload_started | stage=upload_blob | container=%s | blob=%s | "
            "bytes=%d | fingerprint=%s",
            container_name,
            name,
            len(content),
            fingerprint,
        )
        container.upload_blob(name=name, data=content, overwrite=True)
    except Exception:
        logger.exception(
            "master_blob_upload_failed | container=%s | blob=%s | bytes=%d | fingerprint=%s",
            configured_container,
            name,
            len(content),
            fingerprint,
        )
        raise
    logger.info(
        "master_blob_upload_succeeded | container=%s | blob=%s | bytes=%d | fingerprint=%s",
        container_name,
        name,
        len(content),
        fingerprint,
    )
    return f"{BLOB_URI_SCHEME}://{container_name}/{name}"


def _guarded_chunks(downloader: Any, name: str) -> Iterator[bytes]:
    """Iterate a download, logging and releasing it if it fails part-way.

    Azure raises lazily, so a connection dropped mid-transfer surfaces during
    iteration rather than when the download was opened. Without this it would
    escape into whatever is consuming the generator -- for the content
    endpoint, into a StreamingResponse whose headers have already been sent.
    Re-raised rather than swallowed: ending the stream quietly would be
    indistinguishable from a complete short file, and the only consumer
    verifies the SHA-256 of what it receives.
    """
    try:
        for chunk in downloader.chunks():
            yield chunk
    except Exception as e:  # noqa: BLE001 — logged, then handed to the caller
        logger.warning(f"Master blob stream failed part-way for {name}: {e}")
        raise
    finally:
        close = getattr(downloader, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 — releasing a socket must not mask anything
                pass


def open_master_stream(storage_location: str) -> Iterator[bytes] | None:
    """Byte chunks for a stored master, or None if it is gone or unreachable."""
    parsed = parse_blob_uri(storage_location)
    if parsed is None:
        return None
    container_name, name = parsed

    container = _container_client(container_name)
    if container is None:
        return None
    try:
        downloader = container.get_blob_client(name).download_blob(max_concurrency=1)
        logger.info(
            "master_blob_read_opened | container=%s | blob=%s",
            container_name,
            name,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to read master blob {name}: {e}")
        return None
    return _guarded_chunks(downloader, name)


def delete_master(storage_location: str) -> bool:
    """Best-effort delete — never raises, mirroring delete_master_file()."""
    parsed = parse_blob_uri(storage_location)
    if parsed is None:
        return False
    container_name, name = parsed

    container = _container_client(container_name)
    if container is None:
        return False
    try:
        container.delete_blob(name)
        return True
    except Exception as e:  # noqa: BLE001
        # DELETE is idempotent: after a process crash, an audit retry may ask
        # us to remove an object that the prior attempt already removed.
        if e.__class__.__name__ == "ResourceNotFoundError":
            return True
        logger.warning(f"Failed to delete master blob {name}: {e}")
        return False
