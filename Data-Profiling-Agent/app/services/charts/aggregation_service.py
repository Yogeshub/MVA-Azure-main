"""Ad-hoc aggregation over a run's retained source rows.

The profiling pipeline pre-computes a fixed set of charts and drill-down cubes.
Anything outside that set — an arbitrary dimension x metric pairing, or a drill
path deeper than the pre-computed hierarchy — can only be answered by going back
to the rows themselves. This service does that, and only that.

Honesty contract (the analyst UI depends on it):
  * A group whose metric has no numeric value is DROPPED, never reported as 0.
  * Every response states how many rows it was built from, and how many rows the
    filters matched, so the caller can never present a partial answer as total.
  * Truncation is reported explicitly (`truncated` + `group_count`), because a
    top-N chart is not the same as an exhaustive one.
  * A request that cannot be answered as asked raises instead of degrading into
    a plausible-looking but different answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_AGGREGATIONS = ("sum", "mean", "min", "max", "count")

# Aggregations that need a numeric metric column. "count" does not.
_NUMERIC_AGGREGATIONS = ("sum", "mean", "min", "max")

# Column kinds that can serve as a drill level. A float measure is a metric, not
# a dimension; grouping by it produces one group per row and tells you nothing.
# "object" is how pandas 2.x reports a text column and "str"/"string" is how
# pandas 3.x does; both are listed so the level list does not silently empty out
# if the pin in requirements.txt is ever lifted.
_DIMENSION_KINDS = ("object", "str", "string", "category", "bool", "boolean")

# A candidate drill level must group its scope, not shatter it. Above this share
# of distinct values it is an identifier, not a category.
_MAX_CARDINALITY_RATIO = 0.5
_MIN_ROWS_FOR_CARDINALITY_RATIO = 10

# Beyond this a column's values are sentences, not category labels — the
# narrative/commentary columns finance datasets carry alongside their real
# dimensions. Word count is the sharper test of the two: on the reference
# banking dataset every genuine dimension averages at most four words
# ("Corporate & Institutional Banking"), while the commentary columns average
# eight, and some of those are short enough to slip past a length check alone.
_MAX_LABEL_LENGTH = 80
_MAX_LABEL_WORDS = 6


class AggregationRequestError(Exception):
    """The request cannot be answered as asked.

    Carries an HTTP status so the route can surface the real reason instead of
    a generic failure: 400 for a malformed request, 422 for a request that is
    well-formed but not answerable from this dataset.
    """

    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def load_run_dataframe(settings: Settings, file_path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Load a retained run file back into a dataframe using the ingestion loaders."""
    suffix = file_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        from app.services.ingestion.xlsx_loader import XLSXLoader

        return XLSXLoader(settings).load(file_path, sheet_name=sheet_name)

    from app.services.ingestion.csv_loader import CSVLoader

    return CSVLoader(settings).load(file_path)


class AggregationService:
    """Groups a dataframe by one dimension and measures one metric."""

    def __init__(self, max_groups: int = 200):
        self._max_groups = max(1, int(max_groups))

    def aggregate(
        self,
        df: pd.DataFrame,
        dimension: str,
        metric: str | None,
        aggregation: str,
        filters: list[dict[str, str]] | None = None,
        *,
        source: str = "dataset",
        source_note: str | None = None,
        include_next_levels: bool = False,
        max_next_levels: int = 40,
    ) -> dict[str, Any]:
        """Group `df` by `dimension` and measure `metric`.

        `source` / `source_note` describe which copy of the rows this was built
        from. The numbers are identical either way — a caller that recomputed
        from the permanent copy rather than the run's retained working copy must
        still be able to say so, rather than implying a freshness it cannot
        vouch for.

        `include_next_levels` adds the columns this result can be drilled into
        next, computed on the filtered subset so the answer reflects where the
        analyst actually is. Off by default: on a wide dataset it is real work,
        and no existing caller asked for it.
        """
        aggregation = (aggregation or "sum").strip().lower()
        if aggregation not in SUPPORTED_AGGREGATIONS:
            raise AggregationRequestError(
                "UNSUPPORTED_AGGREGATION",
                f"Aggregation '{aggregation}' is not supported. Supported: {', '.join(SUPPORTED_AGGREGATIONS)}.",
            )

        if dimension not in df.columns:
            raise AggregationRequestError(
                "UNKNOWN_DIMENSION",
                f"Column '{dimension}' is not present in this dataset.",
                status_code=422,
            )

        if aggregation in _NUMERIC_AGGREGATIONS:
            if not metric:
                raise AggregationRequestError(
                    "METRIC_REQUIRED",
                    f"Aggregation '{aggregation}' requires a metric column.",
                )
            if metric not in df.columns:
                raise AggregationRequestError(
                    "UNKNOWN_METRIC",
                    f"Column '{metric}' is not present in this dataset.",
                    status_code=422,
                )
        elif metric and metric not in df.columns:
            raise AggregationRequestError(
                "UNKNOWN_METRIC",
                f"Column '{metric}' is not present in this dataset.",
                status_code=422,
            )

        rows_total = int(len(df))
        subset, applied_filters = self._apply_filters(df, filters or [])
        rows_matched = int(len(subset))

        if rows_matched == 0:
            return self._empty_result(
                dimension, metric, aggregation, applied_filters, rows_total,
                reason="No rows match this drill path.",
                source=source,
            )

        if aggregation == "count":
            points, contributing_rows, dropped = self._count_by_dimension(subset, dimension)
        else:
            points, contributing_rows, dropped = self._measure_by_dimension(
                subset, dimension, str(metric), aggregation
            )

        group_count = len(points)
        truncated = group_count > self._max_groups
        if truncated:
            points = points[: self._max_groups]

        return {
            "dimension": dimension,
            "metric": metric,
            "aggregation": aggregation,
            "filters": applied_filters,
            "data": points,
            "rows_used": contributing_rows,
            "rows_matched": rows_matched,
            "rows_total": rows_total,
            "group_count": group_count,
            "returned_group_count": len(points),
            "truncated": truncated,
            "dropped_group_count": dropped,
            "coverage": "complete",
            "source": source,
            "next_level_candidates": (
                self._next_level_candidates(
                    subset,
                    exclude=[dimension, *(f["column"] for f in applied_filters)],
                    limit=max_next_levels,
                )
                if include_next_levels
                else None
            ),
            "title": self._title(dimension, metric, aggregation),
            "description": self._description(
                dimension=dimension,
                metric=metric,
                aggregation=aggregation,
                applied_filters=applied_filters,
                rows_used=contributing_rows,
                rows_matched=rows_matched,
                rows_total=rows_total,
                group_count=group_count,
                returned=len(points),
                truncated=truncated,
                dropped=dropped,
                source_note=source_note,
            ),
        }

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _apply_filters(
        self, df: pd.DataFrame, filters: list[dict[str, str]]
    ) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        """Narrow the frame by the ancestor drill path.

        Values arrive as strings (they came out of a chart label), so matching
        is done on the string rendering first, then on a numeric coercion for
        columns where 1 and "1.0" are the same value.
        """
        subset = df
        applied: list[dict[str, str]] = []
        for entry in filters:
            column = str(entry.get("column", "")).strip()
            value = "" if entry.get("value") is None else str(entry.get("value"))
            if not column:
                continue
            if column not in subset.columns:
                raise AggregationRequestError(
                    "UNKNOWN_FILTER_COLUMN",
                    f"Filter column '{column}' is not present in this dataset.",
                    status_code=422,
                )

            mask = subset[column].astype(str) == value
            if not bool(mask.any()):
                numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(numeric_value):
                    numeric_column = pd.to_numeric(subset[column], errors="coerce")
                    mask = numeric_column == numeric_value

            subset = subset[mask]
            applied.append({"column": column, "value": value})
            if subset.empty:
                break

        return subset, applied

    def _count_by_dimension(
        self, subset: pd.DataFrame, dimension: str
    ) -> tuple[list[dict[str, Any]], int, int]:
        counts = subset.groupby(subset[dimension].astype(str), dropna=False).size()
        points = [
            {"label": str(label), "value": int(count), "count": int(count)}
            for label, count in counts.items()
        ]
        points.sort(key=lambda p: p["value"], reverse=True)
        return points, int(len(subset)), 0

    def _measure_by_dimension(
        self, subset: pd.DataFrame, dimension: str, metric: str, aggregation: str
    ) -> tuple[list[dict[str, Any]], int, int]:
        numeric = pd.to_numeric(subset[metric], errors="coerce")
        if bool(numeric.isna().all()):
            raise AggregationRequestError(
                "METRIC_NOT_NUMERIC",
                f"Column '{metric}' holds no numeric values, so it cannot be aggregated with '{aggregation}'.",
                status_code=422,
            )

        frame = pd.DataFrame(
            {"_label": subset[dimension].astype(str), "_metric": numeric}
        )
        grouped = frame.groupby("_label", dropna=False)["_metric"]

        aggregated = getattr(grouped, aggregation)()
        counts = grouped.count()

        points: list[dict[str, Any]] = []
        dropped = 0
        contributing_rows = 0
        for label, value in aggregated.items():
            count = int(counts.get(label, 0))
            # A group with no numeric rows has no value. Reporting it as 0
            # would be indistinguishable from a real zero measurement.
            if count == 0 or pd.isna(value):
                dropped += 1
                continue
            points.append(
                {"label": str(label), "value": round(float(value), 4), "count": count}
            )
            contributing_rows += count

        points.sort(key=lambda p: abs(p["value"]), reverse=True)
        return points, contributing_rows, dropped

    def _next_level_candidates(
        self, subset: pd.DataFrame, exclude: list[str], limit: int
    ) -> list[dict[str, Any]]:
        """Columns this result can be drilled into next.

        The hierarchy the pipeline detects is a suggestion, not a cage — an
        analyst may want to split "this cost centre" by any dimension the data
        carries. Cardinality is measured on the rows actually in scope, so a
        column that has collapsed to a single value at this depth is not offered
        as a level (it would produce one group containing everything), and
        neither is one that is unique per row (one group per row).
        """
        excluded = {c for c in exclude if c}
        rows = int(len(subset))
        candidates: list[dict[str, Any]] = []

        for column in subset.columns:
            name = str(column)
            if name in excluded:
                continue
            dtype = str(subset[column].dtype).lower()
            if not any(dtype.startswith(kind) for kind in _DIMENSION_KINDS):
                continue
            distinct = int(subset[column].nunique(dropna=True))
            if distinct < 2 or (rows > 1 and distinct >= rows):
                continue
            # A level that shatters its own scope is an identifier wearing a
            # dimension's clothes: drilling into it produces a group per row or
            # two, which answers nothing. Only applied once there are enough
            # rows for the ratio to mean something.
            if rows >= _MIN_ROWS_FOR_CARDINALITY_RATIO and distinct > rows * _MAX_CARDINALITY_RATIO:
                continue
            # Narrative columns (commentary, explanations) repeat enough to look
            # like dimensions but read as prose. A label that long is unusable
            # as a chart axis and was never a category to begin with.
            if self._is_prose(subset[column]):
                continue
            candidates.append({"column": name, "distinct_count": distinct})

        # Coarse levels first: the fewer branches a level has, the more likely it
        # is the next useful cut rather than a long tail of near-unique keys.
        candidates.sort(key=lambda c: (c["distinct_count"], c["column"]))
        return candidates[: max(1, int(limit))]

    @staticmethod
    def _is_prose(series: pd.Series) -> bool:
        """True when a column reads as sentences rather than category labels."""
        sample = series.dropna()
        if sample.empty:
            return False
        sample = sample.head(200).astype(str)
        if sample.str.len().mean() > _MAX_LABEL_LENGTH:
            return True
        return bool(sample.str.split().str.len().mean() > _MAX_LABEL_WORDS)

    def _empty_result(
        self,
        dimension: str,
        metric: str | None,
        aggregation: str,
        applied_filters: list[dict[str, str]],
        rows_total: int,
        reason: str,
        source: str = "dataset",
    ) -> dict[str, Any]:
        return {
            "dimension": dimension,
            "metric": metric,
            "aggregation": aggregation,
            "filters": applied_filters,
            "data": [],
            "rows_used": 0,
            "rows_matched": 0,
            "rows_total": rows_total,
            "group_count": 0,
            "returned_group_count": 0,
            "truncated": False,
            "dropped_group_count": 0,
            "coverage": "complete",
            "source": source,
            "next_level_candidates": [],
            "title": self._title(dimension, metric, aggregation),
            "description": reason,
        }

    @staticmethod
    def _title(dimension: str, metric: str | None, aggregation: str) -> str:
        if aggregation == "count" or not metric:
            return f"Row count by {dimension}"
        return f"{aggregation.capitalize()} of {metric} by {dimension}"

    @staticmethod
    def _description(
        *,
        dimension: str,
        metric: str | None,
        aggregation: str,
        applied_filters: list[dict[str, str]],
        rows_used: int,
        rows_matched: int,
        rows_total: int,
        group_count: int,
        returned: int,
        truncated: bool,
        dropped: int,
        source_note: str | None = None,
    ) -> str:
        measure = (
            f"Row count grouped by {dimension}"
            if aggregation == "count" or not metric
            else f"{aggregation} of {metric} grouped by {dimension}"
        )
        parts = [f"{measure}, computed from the source rows of this run."]

        if source_note:
            parts.append(source_note)

        if applied_filters:
            path = ", ".join(f"{f['column']} = {f['value']}" for f in applied_filters)
            parts.append(f"Filtered to {path}: {rows_matched:,} of {rows_total:,} rows.")
        else:
            parts.append(f"All {rows_total:,} rows.")

        parts.append(f"{rows_used:,} rows carried a usable value.")

        if dropped:
            parts.append(
                f"{dropped} group(s) had no numeric value for {metric} and were omitted rather than shown as zero."
            )
        if truncated:
            parts.append(
                f"Showing the {returned} largest of {group_count} groups — this view is not exhaustive."
            )

        return " ".join(parts)
