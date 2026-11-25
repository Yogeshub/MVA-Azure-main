"""Read master datasets back from Azure Blob Storage.

Why this exists
---------------
Drill-down recovers a finished run's rows from the orchestrator's master
dataset store. On one machine that store is a directory both services can
open, and ``source_dataset_resolver._master_file_by_fingerprint`` reads it
directly. On Azure the two services are separate apps with separate disks, so
no such shared directory exists -- the bytes live in a Blob container instead.

This module is the Blob equivalent of that local lookup: the same
content-addressed layout, the same refusal to guess, and the caller still
verifies the SHA-256 of what comes back against the run's own fingerprint
before any of it is used. Nothing here decides that a file is the right one;
it only fetches the bytes at the address the fingerprint implies.

Enabled only when ``MASTER_DATASET_BACKEND=blob``. Left at the default
(``local``) this module is never called, the Azure SDK is never imported, and
a development deployment behaves exactly as it did before -- including not
needing the SDK installed at all.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Iterator

from app.core.logging import get_logger

logger = get_logger(__name__)

# Every component below is interpolated into a blob name. A name is not a
# filesystem path, but it is still an address built from stored input, and the
# cheapest place to refuse a malformed one is before it is built. Mirrors the
# validation the filesystem backend already applies in the resolver.
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls")

_client_lock = threading.Lock()
_client_cache: dict[str, Any] = {}

# Configuration problems are permanent for the life of the process and every
# drill-down would repeat them. Say each one once, loudly, then stay quiet.
_warned: set[str] = set()


def _warn_once(key: str, event: str, **fields: Any) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(event, **fields)


def is_blob_backend(settings: Any) -> bool:
    """True when this deployment keeps master datasets in Blob Storage."""
    value = (getattr(settings, "master_dataset_backend", "") or "local").strip().lower()
    return value == "blob"


def _safe_prefix(raw: str | None) -> str:
    """A configured prefix, refused if it could climb out of its own namespace."""
    prefix = (raw or "").strip().strip("/")
    if not prefix:
        return ""
    if "\\" in prefix or ".." in prefix.split("/"):
        raise ValueError(f"AZURE_BLOB_PREFIX is not a usable blob prefix: {raw!r}")
    return prefix


def blob_name(settings: Any, fingerprint: str, extension: str) -> str:
    """The content address of a master dataset within the container.

    Raises ValueError on anything that is not a well-formed address. Callers
    treat that as "this source cannot answer" rather than letting a malformed
    address reach the service.
    """
    if not _FINGERPRINT.match(fingerprint or ""):
        raise ValueError("a master dataset is addressed by a SHA-256 digest and nothing else")
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"unsupported master dataset extension: {extension!r}")
    prefix = _safe_prefix(getattr(settings, "azure_blob_prefix", ""))
    shard = f"{fingerprint[:2]}/{fingerprint}{extension}"
    return f"{prefix}/{shard}" if prefix else shard


def _container_client(settings: Any) -> Any | None:
    """A cached container client, or None if Blob access is not usable.

    Never raises. A misconfigured or unreachable container must degrade to
    "this source could not answer" so the resolver can try its next step and,
    failing that, return an honest 409 -- not crash an aggregation request.
    """
    container = (getattr(settings, "azure_blob_container", "") or "").strip()
    account_url = (getattr(settings, "azure_blob_account_url", "") or "").strip()
    connection_string = (getattr(settings, "azure_blob_connection_string", "") or "").strip()

    if not container or not (account_url or connection_string):
        _warn_once(
            "unconfigured",
            "master_blob_backend_unconfigured",
            detail=(
                "MASTER_DATASET_BACKEND=blob but AZURE_BLOB_CONTAINER and one of "
                "AZURE_BLOB_ACCOUNT_URL / AZURE_BLOB_CONNECTION_STRING are not set. "
                "Drill-down cannot reach stored datasets."
            ),
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
                "master_blob_sdk_missing",
                detail=(
                    "azure-storage-blob is not installed; install the Azure extras: "
                    "pip install -r requirements-azure.txt (in Data-Profiling-Agent/)"
                ),
            )
            return None

        try:
            if connection_string:
                service = BlobServiceClient.from_connection_string(connection_string)
            else:
                # Managed Identity is the intended production path -- no secret
                # in configuration, and the identity is granted "Storage Blob
                # Data Reader" on the container.
                from azure.identity import DefaultAzureCredential

                client_id = (getattr(settings, "azure_blob_client_id", "") or "").strip()
                credential = (
                    DefaultAzureCredential(managed_identity_client_id=client_id)
                    if client_id
                    else DefaultAzureCredential()
                )
                service = BlobServiceClient(account_url=account_url, credential=credential)
            client = service.get_container_client(container)
        except Exception as e:  # noqa: BLE001 — a bad client must not break aggregation
            _warn_once("client_failed", "master_blob_client_failed", error=str(e))
            return None

        _client_cache[cache_key] = client
        return client


def _guarded_chunks(downloader: Any, name: str) -> Iterator[bytes]:
    """Iterate a download, logging and releasing it if it fails part-way.

    Azure raises lazily: the failure that matters most -- a connection dropped
    mid-transfer -- surfaces during iteration, not when the download was
    opened. Left unhandled it would escape into whatever is consuming the
    generator. It is re-raised rather than swallowed, because silently ending
    the stream would look exactly like a complete short file; the caller's
    SHA-256 check is what turns a partial transfer into a refusal.
    """
    try:
        for chunk in downloader.chunks():
            yield chunk
    except Exception as e:  # noqa: BLE001 — logged, then handed to the caller
        logger.warning("master_blob_stream_failed", blob=name, error=str(e))
        raise
    finally:
        close = getattr(downloader, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 — releasing a socket must not mask anything
                pass


def open_master_stream(
    settings: Any, fingerprint: str, extension: str
) -> Iterator[bytes] | None:
    """Byte chunks of a stored master dataset, or None if it is not there.

    Streamed rather than loaded whole: the caller hashes as it writes, so a
    dataset never has to fit in memory to be verified.
    """
    container = _container_client(settings)
    if container is None:
        return None

    try:
        name = blob_name(settings, fingerprint, extension)
    except ValueError as e:
        logger.warning("master_blob_address_rejected", error=str(e))
        return None

    try:
        downloader = container.get_blob_client(name).download_blob(max_concurrency=1)
        logger.info(
            "master_blob_read_opened",
            container=getattr(container, "container_name", "<configured>"),
            blob=name,
            fingerprint=fingerprint,
        )
    except Exception as e:  # noqa: BLE001
        if _is_not_found(e):
            # Expected while probing extensions -- not every candidate exists.
            logger.warning(
                "master_blob_absent",
                container=getattr(container, "container_name", "<configured>"),
                blob=name,
                fingerprint=fingerprint,
                extension=extension,
                status_code=404,
            )
        else:
            _warn_once(f"fetch:{type(e).__name__}", "master_blob_fetch_failed", blob=name, error=str(e))
        return None

    return _guarded_chunks(downloader, name)


def _is_not_found(error: Exception) -> bool:
    if type(error).__name__ == "ResourceNotFoundError":
        return True
    return getattr(getattr(error, "response", None), "status_code", None) == 404
