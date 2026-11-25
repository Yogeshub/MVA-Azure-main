"""
Aggregation route — POST /pipeline/aggregate.

The analyst UI needs answers for dimension x metric pairings the profiling run
did not pre-compute: custom breakdowns the user builds themselves, and drill
paths deeper than the pre-built cubes. The orchestrator holds no rows of its
own, so this forwards to Agent 2 — which owns the dataset — and relays its
answer verbatim.

Input is validated here as well as downstream: the aggregator turns these names
into a group-by, so a crafted request must not be able to smuggle an expression
into it even if the aggregator is loose.
"""

import logging
import re
from typing import Any

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.services.dataset_registry import metadata_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

ALLOWED_AGGREGATIONS = {"sum", "mean", "min", "max", "count"}
MAX_FILTERS = 12
MAX_FILTER_VALUE_LENGTH = 256
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
MAX_COLUMN_NAME_LENGTH = 128


def _is_valid_column_name(name: str) -> bool:
    """Reject only what cannot be a column name.

    Real finance headers contain %, &, /, parentheses, currency symbols and
    non-ASCII text ("Variance %", "Actual/Budget", "P&L", "Revenue ($)"), so an
    allowlist of characters would refuse legitimate questions. The name is used
    as a pandas column lookup, never evaluated as an expression, so length and
    control-character checks are the meaningful guards — whether the column
    actually exists is Agent 2's answer to give (422 with the column named).
    """
    return bool(name) and len(name) <= MAX_COLUMN_NAME_LENGTH and not _CONTROL_CHARS.search(name)


class AggregateFilter(BaseModel):
    """One step of the ancestor drill path."""
    column: str
    value: str = ""


class AggregateRequest(BaseModel):
    run_id: str
    dimension: str
    metric: str | None = None
    aggregation: str = "sum"
    filters: list[AggregateFilter] = Field(default_factory=list)
    # Ask Agent 2 to also list the columns this result can be drilled into next,
    # measured on the rows actually in scope. Off by default; the analyst UI
    # turns it on so the hierarchy it offers reflects the data, not a guess.
    include_next_levels: bool = False


def _bad_request(detail: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": detail})


async def _post_aggregate(url: str, payload: dict[str, Any]) -> tuple[int, Any] | JSONResponse:
    """POST to Agent 2, returning (status, body) or a ready-made failure."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=settings.REQUEST_TIMEOUT_SECONDS)
    except httpx.HTTPError as e:
        return JSONResponse(
            status_code=502,
            content={"detail": f"Agent 2 is unreachable, so this breakdown cannot be computed: {e}"},
        )
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return JSONResponse(
            status_code=502,
            content={"detail": "Agent 2 returned a response that could not be parsed as JSON."},
        )


def _source_reference_for_run(run_id: str) -> dict[str, Any] | None:
    """Where this run's rows are permanently stored, per the Dataset Registry.

    The link was written when the run was created (or when its dataset was last
    uploaded), from the fingerprint of the exact bytes involved. It is a record,
    not an inference — there is no filename or row-count matching here, because
    a near-match would point a run at somebody else's data.
    """
    try:
        link = metadata_cache.lookup_run_dataset_link(run_id)
    except Exception as e:  # noqa: BLE001 — a registry outage must not mask Agent 2's answer
        logger.warning(f"Could not look up the source dataset for run {run_id}: {e}")
        return None
    if not link or not link.get("fingerprint"):
        return None
    return {
        "fingerprint": link.get("fingerprint"),
        "uri": link.get("storage_location"),
        "filename": link.get("original_filename"),
        "sheet_name": link.get("sheet_name"),
    }


@router.post("/aggregate")
async def aggregate(body: AggregateRequest):
    """Group a completed run's rows by one dimension and measure one metric."""
    run_id = body.run_id.strip()
    dimension = body.dimension.strip()
    metric = (body.metric or "").strip()
    aggregation = (body.aggregation or "sum").strip().lower()

    if not _RUN_ID.match(run_id):
        return _bad_request("run_id is not a valid identifier.")
    if not _is_valid_column_name(dimension):
        return _bad_request(f"'{dimension}' is not a valid column name.")
    if metric and not _is_valid_column_name(metric):
        return _bad_request(f"'{metric}' is not a valid column name.")
    if aggregation not in ALLOWED_AGGREGATIONS:
        return _bad_request(
            f"Unsupported aggregation '{aggregation}'. Allowed: {', '.join(sorted(ALLOWED_AGGREGATIONS))}."
        )
    if len(body.filters) > MAX_FILTERS:
        return _bad_request(f"The drill path is limited to {MAX_FILTERS} levels.")

    filters = []
    for f in body.filters:
        column = f.column.strip()
        if not _is_valid_column_name(column):
            return _bad_request(f"'{column}' is not a valid filter column.")
        if len(f.value) > MAX_FILTER_VALUE_LENGTH:
            return _bad_request("A filter value is too long.")
        filters.append({"column": column, "value": f.value})

    payload: dict[str, Any] = {
        "dimension": dimension,
        "metric": metric or None,
        "aggregation": aggregation,
        "filters": filters,
        "include_next_levels": bool(body.include_next_levels),
    }
    url = f"{settings.AGENT2_BASE_URL}{settings.AGENT2_API_PREFIX}/profile-runs/{run_id}/aggregate"

    outcome = await _post_aggregate(url, payload)
    if isinstance(outcome, JSONResponse):
        return outcome
    status_code, content = outcome

    # A 409 means Agent 2 could not get back to the run's rows. Often it simply
    # does not know where they are — a run profiled before source references
    # were recorded carries none. The Dataset Registry does know, so hand it
    # over and let Agent 2 try once more. Runs that predate this feature start
    # working without a backfill; a run with no link still fails honestly.
    if status_code == 409:
        reference = _source_reference_for_run(run_id)
        if reference is not None:
            retry = await _post_aggregate(url, {**payload, "source_reference": reference})
            if isinstance(retry, JSONResponse):
                return retry
            status_code, content = retry

    if status_code == 200:
        return content

    # Relay the upstream reason rather than flattening it — the UI shows the
    # analyst exactly why a breakdown could not be produced.
    detail = content.get("detail") if isinstance(content, dict) else None
    if isinstance(detail, dict):
        detail = detail.get("message") or str(detail)
    if not detail:
        detail = f"Agent 2 could not compute this breakdown (HTTP {status_code})."
    return JSONResponse(status_code=status_code, content={"detail": detail})
