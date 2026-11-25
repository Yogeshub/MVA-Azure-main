"""API request schemas."""

from pydantic import BaseModel, Field


class DrillDownRequest(BaseModel):
    """Drill-down request body."""
    selected_path: dict[str, str]


class RuleApprovalRequest(BaseModel):
    """Rule approval request."""
    comment: str | None = None


class RuleRejectionRequest(BaseModel):
    """Rule rejection request."""
    reason: str | None = None


class AggregateFilter(BaseModel):
    """One step of an ancestor drill path."""
    column: str
    value: str = ""


class SourceDatasetReferenceIn(BaseModel):
    """Where this run's rows can be found, supplied by the caller.

    Only ever used to fill blanks the run record itself does not carry — a run
    created before source references were persisted, for example. What the run
    records always wins, so a stale or crafted hint cannot point a run at
    somebody else's data.
    """
    fingerprint: str | None = None
    uri: str | None = None
    filename: str | None = None
    sheet_name: str | None = None


class AggregateRequest(BaseModel):
    """Ad-hoc aggregation request.

    metric may be null when aggregation is "count" (a pure row count).
    """
    dimension: str
    metric: str | None = None
    aggregation: str = "sum"
    filters: list[AggregateFilter] = Field(default_factory=list)
    # Opt-in: list the columns this result can be drilled into next. Off by
    # default so every existing caller's response shape and cost are unchanged.
    include_next_levels: bool = False
    source_reference: SourceDatasetReferenceIn | None = None
