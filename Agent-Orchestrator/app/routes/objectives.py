"""Lightweight, evidence-based objective recommendations for uploaded datasets.

This route intentionally does not invoke the pipeline or register the upload.  It
only reads a small schema sample, so callers can decide which analysis to run
before incurring a full pipeline execution.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings

router = APIRouter(prefix="/objectives", tags=["Objectives"])

_SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
_SCHEMA_SAMPLE_ROWS = 50
_ACTUAL_TOKEN = "actual"
_BUDGET_TOKEN = "budget"
_MEASURE_TOKENS = frozenset({
    "actual", "budget", "revenue", "profit", "nim", "margin", "sales",
    "cost", "expense", "amount", "value", "units", "quantity", "volume",
    "income", "spend",
})


def _tokens(column: Any) -> tuple[str, ...]:
    """Make a conservative token representation of a user-provided heading."""
    return tuple(
        token.lower()
        for token in re.findall(r"[A-Za-z]+|\d+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(column)))
    )


def _numeric_values(series: pd.Series) -> pd.Series:
    """Recognize common spreadsheet number spellings without calculating data."""
    text = series.astype("string").str.strip()
    cleaned = (
        text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        .str.replace(r"[$€£,%]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _has_numeric_pair(frame: pd.DataFrame, left: str, right: str) -> bool:
    # A pair must occur on at least one actual row.  Header names alone, or
    # unrelated string fields called "actual" and "budget", are not evidence.
    return bool((_numeric_values(frame[left]).notna() & _numeric_values(frame[right]).notna()).any())


def _actual_budget_pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    actuals: dict[tuple[str, ...], list[str]] = {}
    budgets: dict[tuple[str, ...], list[str]] = {}
    for column in frame.columns:
        column_name = str(column)
        column_tokens = _tokens(column_name)
        if _ACTUAL_TOKEN in column_tokens:
            key = tuple(token for token in column_tokens if token != _ACTUAL_TOKEN)
            actuals.setdefault(key, []).append(column_name)
        if _BUDGET_TOKEN in column_tokens:
            key = tuple(token for token in column_tokens if token != _BUDGET_TOKEN)
            budgets.setdefault(key, []).append(column_name)

    pairs: list[tuple[str, str]] = []
    for key in actuals.keys() & budgets.keys():
        for actual in actuals[key]:
            for budget in budgets[key]:
                if _has_numeric_pair(frame, actual, budget):
                    pairs.append((actual, budget))
    return pairs


def _pair_has_concept(pair: tuple[str, str], concept: str) -> bool:
    """Require the named business concept on both sides of a comparison."""
    left_tokens, right_tokens = _tokens(pair[0]), _tokens(pair[1])
    if concept == "nim":
        long_form = {"net", "interest", "margin"}
        return (
            "nim" in left_tokens and "nim" in right_tokens
        ) or (
            long_form.issubset(left_tokens) and long_form.issubset(right_tokens)
        )
    return concept in left_tokens and concept in right_tokens


def _objective(objective_id: str, title: str, business_question: str, rationale: str, columns: list[str]) -> dict[str, Any]:
    return {
        "id": objective_id,
        "title": title,
        "business_question": business_question,
        "rationale": rationale,
        "matched_columns": columns,
    }


def _dimension_and_measure(
    frame: pd.DataFrame,
    dimension_tokens: frozenset[str],
) -> tuple[str, str] | None:
    dimensions = [
        str(column) for column in frame.columns
        if set(_tokens(column)) & dimension_tokens
    ]
    measures = [
        str(column) for column in frame.columns
        if set(_tokens(column)) & _MEASURE_TOKENS
        and "id" not in _tokens(column)
        and _numeric_values(frame[str(column)]).notna().any()
    ]
    if dimensions and measures:
        return dimensions[0], measures[0]
    return None


def recommend_objectives(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return only recommendations supported by headers and sampled values."""
    frame = frame.copy()
    frame.columns = [str(column) for column in frame.columns]
    pairs = _actual_budget_pairs(frame)
    recommendations: list[dict[str, Any]] = []

    if pairs:
        columns = [column for pair in pairs for column in pair]
        recommendations.append(_objective(
            "actual-vs-budget", "Actual vs Budget",
            "How do actual results compare with budget?",
            "Found numeric actual/budget column pair(s).", columns,
        ))

    for objective_id, title, term, question in [
        ("revenue-variance", "Revenue Variance", "revenue", "Where does revenue differ from budget?"),
        ("profit-variance", "Profit Variance", "profit", "Where does profit differ from budget?"),
        ("nim-variance", "NIM Variance", "nim", "Where does net interest margin differ from budget?"),
    ]:
        matching_pairs = [pair for pair in pairs if _pair_has_concept(pair, term)]
        if matching_pairs:
            recommendations.append(_objective(
                objective_id, title, question,
                f"Found numeric actual/budget {term} column pair(s).",
                [column for pair in matching_pairs for column in pair],
            ))

    for objective_id, title, tokens, dimension_label, question in [
        (
            "regional-analysis",
            "Regional Analysis",
            frozenset({"region", "regional", "country", "state", "territory"}),
            "geographic",
            "How do results vary by region?",
        ),
        (
            "product-analysis",
            "Product Analysis",
            frozenset({"product", "sku", "item", "category"}),
            "product",
            "How do results vary by product?",
        ),
    ]:
        evidence = _dimension_and_measure(frame, tokens)
        if evidence:
            recommendations.append(_objective(
                objective_id, title, question,
                f"Found a {dimension_label} dimension and numeric measure-like column.",
                list(evidence),
            ))
    return recommendations


def _read_schema_sample(filename: str, content: bytes) -> pd.DataFrame:
    source = io.BytesIO(content)
    if Path(filename).suffix.lower() == ".csv":
        return pd.read_csv(source, nrows=_SCHEMA_SAMPLE_ROWS)
    return pd.read_excel(source, nrows=_SCHEMA_SAMPLE_ROWS)


@router.post("/recommend")
async def recommend_objectives_endpoint(
    file: UploadFile = File(..., description="CSV, XLS, or XLSX dataset to inspect"),
):
    """Recommend objectives from headers plus a small non-persisted schema sample."""
    filename = file.filename or "upload"
    extension = Path(filename).suffix.lower()
    if extension not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only CSV, XLS, and XLSX files are supported.")

    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds the maximum upload size of {settings.MAX_UPLOAD_SIZE_MB}MB.")
    try:
        frame = _read_schema_sample(filename, content)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read dataset schema: {exc}") from exc
    return {"objectives": recommend_objectives(frame)}