"""Focused evidence tests for POST /objectives/recommend recommendation logic."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.routes.objectives import recommend_objectives


def _ids(frame: pd.DataFrame) -> set[str]:
    return {item["id"] for item in recommend_objectives(frame)}


def test_recommends_actual_vs_budget_only_for_numeric_pair():
    assert "actual-vs-budget" in _ids(pd.DataFrame({
        "sales_actual": [100, 110], "sales_budget": [90, 105],
    }))


def test_recommends_revenue_variance_for_numeric_revenue_pair():
    assert "revenue-variance" in _ids(pd.DataFrame({
        "revenue_actual": [100, 110], "revenue_budget": [90, 105],
    }))


def test_recommends_profit_variance_for_numeric_profit_pair():
    assert "profit-variance" in _ids(pd.DataFrame({
        "profit_actual": [20, 22], "profit_budget": [18, 21],
    }))


def test_recommends_nim_variance_for_numeric_nim_pair():
    assert "nim-variance" in _ids(pd.DataFrame({
        "nim_actual": ["3.1%", "3.2%"], "nim_budget": ["3.0%", "3.0%"],
    }))


def test_recommends_nim_variance_for_full_net_interest_margin_name():
    assert "nim-variance" in _ids(pd.DataFrame({
        "net_interest_margin_actual": [3.1, 3.2],
        "net_interest_margin_budget": [3.0, 3.0],
    }))


def test_recommends_regional_analysis_with_numeric_measure():
    assert "regional-analysis" in _ids(pd.DataFrame({
        "region": ["East", "West"], "revenue": [100, 200],
    }))


def test_recommends_regional_analysis_for_supported_geography_aliases():
    for column in ("country", "state", "territory"):
        assert "regional-analysis" in _ids(pd.DataFrame({
            column: ["East", "West"], "revenue": [100, 200],
        }))


def test_recommends_product_analysis_with_numeric_measure():
    assert "product-analysis" in _ids(pd.DataFrame({
        "product": ["A", "B"], "units_sold": [4, 8],
    }))


def test_recommends_product_analysis_for_supported_catalog_aliases():
    for column in ("sku", "item", "category"):
        assert "product-analysis" in _ids(pd.DataFrame({
            column: ["A", "B"], "units": [4, 8],
        }))


def test_rejects_misleading_headers_without_numeric_actual_budget_evidence():
    ids = _ids(pd.DataFrame({
        "revenue_actual": ["unknown", "pending"],
        "revenue_budget": ["none", "tbd"],
    }))
    assert not {"actual-vs-budget", "revenue-variance"} & ids


def test_rejects_actual_and_budget_columns_for_different_metrics():
    ids = _ids(pd.DataFrame({
        "revenue_actual": [100, 110], "profit_budget": [18, 21],
    }))
    assert not {"actual-vs-budget", "revenue-variance", "profit-variance"} & ids


def test_rejects_dimension_with_only_identifier_not_measure():
    ids = _ids(pd.DataFrame({"region": ["East", "West"], "customer_id": [1, 2]}))
    assert "regional-analysis" not in ids