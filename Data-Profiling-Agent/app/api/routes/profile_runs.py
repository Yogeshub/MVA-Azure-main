"""Profile runs API routes."""

import uuid
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.enums import RunStatus
from app.core.logging import get_logger
from app.api.dependencies import get_cached_settings, get_db
from app.api.auth import get_current_user
from app.repositories.configuration_repository import ConfigurationRepository
from app.repositories.profile_run_repository import ProfileRunRepository
from app.repositories.rule_repository import RuleRepository
from app.services.ingestion.source_dataset_resolver import (
    SourceDatasetReference,
    SourceDatasetResolver,
)
from app.services.ingestion.temporary_storage import TemporaryStorage
from app.services.llm.factory import create_llm_provider
from app.services.quality.overall_score import calculate_overall_score
from app.services.schema_intelligence.local_provider import LocalSchemaIntelligenceProvider
from app.services.orchestration.job_manager import JobManager, ExecutionMode
from app.agents.data_profiling_agent.graph import run_profiling_graph
from app.schemas.requests import AggregateRequest, DrillDownRequest
from app.schemas.responses import RunCreatedResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/profile-runs", tags=["profile-runs"])

# Job manager with background thread execution
_job_manager = JobManager(mode=ExecutionMode.BACKGROUND_THREAD)


def _get_graph_dependencies(settings: Settings) -> tuple[ConfigurationRepository, LocalSchemaIntelligenceProvider, TemporaryStorage]:
    """Build the profiling graph's shared dependencies with current settings."""
    config_repo = ConfigurationRepository(config_dir="config")
    llm = create_llm_provider(settings)
    si = LocalSchemaIntelligenceProvider(llm)
    temp = TemporaryStorage(settings)
    # Retained source files are swept here rather than on a schedule, so an
    # idle service never accumulates uploads it will never be asked about.
    if getattr(settings, "retain_run_files", False):
        temp.purge_expired(getattr(settings, "run_file_retention_hours", 24))
    return config_repo, si, temp


def _parse_run_id(run_id: str) -> uuid.UUID:
    """Parse a run_id path param into a UUID, raising RunNotFoundError for malformed input."""
    try:
        return uuid.UUID(run_id)
    except ValueError:
        from app.core.exceptions import RunNotFoundError
        raise RunNotFoundError(run_id)


def _column_profile_to_dict(cp) -> dict[str, Any]:
    return {
        "physical_name": cp.physical_name,
        "normalized_key": cp.normalized_key,
        "pandas_dtype": cp.pandas_dtype,
        "refined_data_type": cp.refined_data_type,
        "statistics": cp.statistics_json,
        "candidate_semantic_type": cp.candidate_semantic_type,
        "candidate_column_role": cp.candidate_column_role,
        "candidate_confidence": cp.candidate_confidence,
        "confirmed_semantic_type": cp.confirmed_semantic_type,
        "confirmed_column_role": cp.confirmed_column_role,
        "schema_confidence": cp.schema_confidence,
        "identifier_score": cp.identifier_score,
        "is_grain_key": cp.is_grain_key,
    }


def _quality_assessment_to_dict(qa) -> dict[str, Any]:
    return {
        "dimension": qa.dimension,
        "score": qa.score,
        "display_score": qa.display_score,
        "status": qa.status,
        "assessed_count": qa.assessed_count,
        "violation_count": qa.violation_count,
        "evidence": qa.evidence_json,
        "reason": qa.reason,
    }


def _readiness_assessment_to_dict(ra) -> dict[str, Any]:
    return {
        "assessment_type": ra.assessment_type,
        "score": ra.score,
        "status": ra.status,
        "strengths": ra.strengths_json,
        "blocking_issues": ra.blockers_json,
        "recommendations": ra.recommendations_json,
        "evidence": ra.evidence_json,
        "weight_profile_version": ra.weight_profile_version,
        "dataset_score": ra.dataset_score,
        "task_compatibility_score": ra.task_compatibility_score,
    }


def _feature_recommendation_to_dict(fr) -> dict[str, Any]:
    return {
        "business_question": fr.business_question,
        "target_column": fr.target_column,
        "problem_type": fr.problem_type,
        "time_column": fr.time_column,
        "recommended_approach": fr.recommended_approach,
        "approach_reasoning": fr.approach_reasoning,
        "feature_columns": fr.feature_columns_json or [],
        "drop_columns": fr.drop_columns_json or [],
        "confidence": fr.confidence,
    }


def _rule_suggestion_to_dict(s) -> dict[str, Any]:
    definition = s.suggested_definition_json or {}
    return {
        "suggestion_id": str(s.id),
        "rule_key": definition.get("rule_key"),
        "type": definition.get("type"),
        "description": definition.get("description"),
        "confidence": s.confidence,
        "status": s.status,
    }


def _rule_evaluation_to_dict(e) -> dict[str, Any]:
    return {
        "rule_key": e.rule_key,
        "rule_type": e.rule_type,
        "source": e.source,
        "rule_definition_id": str(e.rule_definition_id) if e.rule_definition_id else None,
        "records_checked": e.records_checked,
        "pass_count": e.pass_count,
        "fail_count": e.fail_count,
        "score": e.score,
        "evidence": e.evidence_json,
    }


def _cube_to_dict(cube) -> dict[str, Any]:
    """Serialize a persisted drill-down cube in the shape the pipeline produced."""
    return {
        "level_column": cube.level_column,
        "level_depth": cube.level_depth,
        "dimension_path_json": cube.dimension_path_json or {},
        "aggregated_data_json": cube.aggregated_data_json or [],
        "metric_column": cube.metric_column,
        "aggregation": cube.aggregation,
        "record_count": cube.record_count,
    }


def _chart_to_dict(cs) -> dict[str, Any]:
    spec = dict(cs.specification_json or {})
    spec.setdefault("chart_key", cs.chart_key)
    spec.setdefault("category", cs.category)
    spec.setdefault("chart_type", cs.chart_type)
    spec.setdefault("title", cs.title)
    spec.setdefault("data", cs.aggregated_data_json)
    spec.setdefault("rank", cs.rank)
    return spec


@router.post("", response_model=RunCreatedResponse)
async def create_profile_run(
    file: UploadFile = File(...),
    primary_domain: str = Form(...),
    schema_metadata: str = Form(default="{}"),
    request_rules: str = Form(default="[]"),
    sheet_name: str | None = Form(default=None),
    business_question: str | None = Form(default=None),
    target_column: str | None = Form(default=None),
    # Durable reference to where this upload permanently lives, supplied by the
    # orchestrator's Dataset Registry. Optional — without it the run still
    # profiles normally, it just cannot recompute an unforeseen breakdown once
    # the retained working copy has been swept.
    source_fingerprint: str | None = Form(default=None),
    source_dataset_uri: str | None = Form(default=None),
    settings: Settings = Depends(get_cached_settings),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RunCreatedResponse:
    """
    Create and execute a profiling run.

    Accepts multipart/form-data with:
    - file: CSV or XLSX file
    - primary_domain: one of Payments, Customer, HR, Finance
    - schema_metadata: JSON string with column descriptions
    - request_rules: JSON string with additional rules
    - sheet_name: required only when the XLSX workbook has more than one
      non-empty sheet; names which sheet to load
    - business_question: optional — when supplied, the pipeline classifies the
      dataset's target/feature/drop columns and an ML-vs-LLM approach for this
      question (feature_recommendation) instead of a generic deterministic split
    - target_column: optional explicit override — if the caller already knows
      the target column, skip LLM target-guessing and use it directly
    """
    run_id = uuid.uuid4()

    # Parse JSON fields
    try:
        metadata = json.loads(schema_metadata) if schema_metadata else None
    except json.JSONDecodeError:
        metadata = None

    try:
        rules = json.loads(request_rules) if request_rules else None
    except json.JSONDecodeError:
        rules = None

    # Read file content
    content = await file.read()
    filename = file.filename or "upload"
    file_type = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()

    # Approved rule suggestions from prior runs apply to every future run in this primary
    # domain (secondary domain isn't known until classification runs mid-pipeline, so it
    # isn't filtered on here).
    rule_repo = RuleRepository(db)
    active_rules = rule_repo.list_active_definitions(primary_domain)
    approved_rule_dicts = [rd.definition_json for rd in active_rules]
    rule_definition_ids_by_key = {rd.rule_key: rd.id for rd in active_rules}

    # Execute pipeline (LangGraph-based) via job manager
    config_repo_dep, si_dep, temp_dep = _get_graph_dependencies(settings)
    # Record how this upload is being read so a retained run can be re-opened
    # identically later for ad-hoc aggregation (a multi-sheet workbook is
    # unreadable without knowing which sheet was profiled).
    if getattr(settings, "retain_run_files", False):
        temp_dep.write_run_metadata(run_id, {"sheet_name": sheet_name, "filename": filename})
    _job_manager.create_job(str(run_id))

    def _run_pipeline():
        return run_profiling_graph(
            settings=settings,
            config_repo=config_repo_dep,
            schema_intelligence=si_dep,
            temp_storage=temp_dep,
            run_id=run_id,
            file_content=content,
            filename=filename,
            primary_domain=primary_domain,
            schema_metadata=metadata,
            request_rules=rules,
            sheet_name=sheet_name,
            approved_rule_dicts=approved_rule_dicts,
            business_question=business_question,
            target_column_override=target_column,
        )

    _job_manager.submit(str(run_id), _run_pipeline)

    # Wait for completion (sync-like behavior for v1)
    completed_job = _job_manager.wait_for_completion(
        str(run_id), timeout=settings.processing_timeout_seconds,
    )

    if completed_job and completed_job.result:
        result = completed_job.result

        repo = ProfileRunRepository(db)
        repo.create_run(
            run_id, primary_domain, filename, file_type,
            source_fingerprint=(source_fingerprint or None),
            source_dataset_uri=(source_dataset_uri or None),
            source_sheet_name=sheet_name,
        )
        repo.mark_processing(run_id)

        if result.status == RunStatus.FAILED:
            error = result.error or {}
            repo.mark_failed(run_id, error.get("code", "PROCESSING_ERROR"), error.get("message", ""))
            repo.commit()
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=422,
                content={"error": result.error},
            )

        repo.persist_dataset_profile(run_id, result.dataset_profile)
        column_ids_by_name = repo.persist_column_profiles(run_id, result.column_profiles)
        repo.persist_secondary_domain_result(run_id, result.secondary_domain)
        repo.persist_column_classifications(run_id, column_ids_by_name, result.column_classifications)
        repo.persist_hierarchy(run_id, result.hierarchy)
        repo.persist_quality_assessments(run_id, result.quality_assessments)
        repo.persist_readiness_assessments(run_id, result.readiness_assessments)
        repo.persist_feature_recommendation(run_id, result.feature_recommendation, business_question)
        repo.persist_charts(run_id, result.charts)
        repo.persist_drill_down_cubes(run_id, getattr(result, "drill_down_cubes", None) or [])
        rule_repo.persist_suggestions(run_id, result.rule_suggestions)
        rule_repo.persist_evaluations(run_id, result.rule_evaluations, rule_definition_ids_by_key)

        dataset_profile = result.dataset_profile or {}
        repo.mark_completed(
            run_id,
            row_count=dataset_profile.get("row_count", 0),
            column_count=dataset_profile.get("column_count", 0),
            secondary_domain=(result.secondary_domain or {}).get("name"),
        )
        repo.commit()

        return RunCreatedResponse(run_id=str(run_id), status=result.status.value)

    # Job still processing or failed at job level
    if completed_job and completed_job.error:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": completed_job.error})

    return RunCreatedResponse(run_id=str(run_id), status="processing")


@router.get("/{run_id}")
def get_run_summary(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get profile run summary."""
    repo = ProfileRunRepository(db)
    run = repo.get_run(_parse_run_id(run_id))
    if not run:
        from app.core.exceptions import RunNotFoundError
        raise RunNotFoundError(run_id)

    dataset_profile = repo.get_dataset_profile(run.id)
    secondary_domain = repo.get_secondary_domain_result(run.id)

    return {
        "run_id": str(run.id),
        "status": run.status,
        "primary_domain": run.primary_domain,
        "secondary_domain": {
            "name": secondary_domain.domain_name,
            "confidence": secondary_domain.confidence,
            "status": secondary_domain.status,
            "evidence": secondary_domain.evidence_json,
        } if secondary_domain else None,
        "dataset_profile": dataset_profile.profile_json if dataset_profile else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


@router.get("/{run_id}/result")
def get_full_result(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get complete profiling result."""
    repo = ProfileRunRepository(db)
    run_uuid = _parse_run_id(run_id)
    run = repo.get_run(run_uuid)
    if not run:
        from app.core.exceptions import RunNotFoundError
        raise RunNotFoundError(run_id)

    dataset_profile = repo.get_dataset_profile(run_uuid)
    secondary_domain = repo.get_secondary_domain_result(run_uuid)
    column_profiles = repo.list_column_profiles(run_uuid)
    column_classifications = repo.list_column_classifications(run_uuid)
    hierarchy = repo.get_hierarchy(run_uuid)
    quality_assessments = repo.list_quality_assessments(run_uuid)
    readiness_assessments = repo.list_readiness_assessments(run_uuid)
    feature_recommendation = repo.get_feature_recommendation(run_uuid)
    charts = repo.list_charts(run_uuid)
    rule_suggestions = RuleRepository(db).list_suggestions_for_run(run_uuid)

    config_repo = ConfigurationRepository(config_dir="config")
    weights_config = config_repo.get_quality_weights()
    overall_quality = calculate_overall_score(
        [_quality_assessment_to_dict(qa) for qa in quality_assessments],
        weights_config.get("weights", {}),
    )

    return {
        "run_id": str(run.id),
        "status": run.status,
        "primary_domain": run.primary_domain,
        "secondary_domain": {
            "name": secondary_domain.domain_name,
            "confidence": secondary_domain.confidence,
            "status": secondary_domain.status,
            "evidence": secondary_domain.evidence_json,
        } if secondary_domain else {},
        "dataset_profile": dataset_profile.profile_json if dataset_profile else {},
        "column_profiles": [_column_profile_to_dict(cp) for cp in column_profiles],
        "column_classifications": [
            {
                "column_name": next((cp.physical_name for cp in column_profiles if cp.id == cc.column_profile_id), None),
                "primary_category": cc.primary_category,
                "secondary_categories": cc.secondary_categories_json,
                "confidence": cc.confidence,
                "evidence": cc.evidence_json,
            }
            for cc in column_classifications
        ],
        "hierarchy": hierarchy or {},
        "quality_assessments": [_quality_assessment_to_dict(qa) for qa in quality_assessments],
        "overall_quality": overall_quality,
        "readiness_assessments": [_readiness_assessment_to_dict(ra) for ra in readiness_assessments],
        "feature_recommendation": _feature_recommendation_to_dict(feature_recommendation) if feature_recommendation else {},
        "charts": [_chart_to_dict(c) for c in charts],
        "drill_down_cubes": [_cube_to_dict(c) for c in repo.list_drill_down_cubes(run_uuid)],
        "rule_suggestions": [_rule_suggestion_to_dict(s) for s in rule_suggestions],
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


@router.get("/{run_id}/columns")
def get_columns(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get column profiles."""
    repo = ProfileRunRepository(db)
    run_uuid = _parse_run_id(run_id)
    if not repo.get_run(run_uuid):
        from app.core.exceptions import RunNotFoundError
        raise RunNotFoundError(run_id)
    columns = repo.list_column_profiles(run_uuid)
    return {"columns": [_column_profile_to_dict(cp) for cp in columns]}


@router.get("/{run_id}/quality")
def get_quality(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get quality assessments."""
    repo = ProfileRunRepository(db)
    run_uuid = _parse_run_id(run_id)
    if not repo.get_run(run_uuid):
        from app.core.exceptions import RunNotFoundError
        raise RunNotFoundError(run_id)

    assessments = repo.list_quality_assessments(run_uuid)
    assessment_dicts = [_quality_assessment_to_dict(qa) for qa in assessments]

    config_repo = ConfigurationRepository(config_dir="config")
    weights_config = config_repo.get_quality_weights()
    overall = calculate_overall_score(assessment_dicts, weights_config.get("weights", {}))

    return {"assessments": assessment_dicts, "overall": overall}


@router.get("/{run_id}/readiness")
def get_readiness(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get readiness assessments."""
    repo = ProfileRunRepository(db)
    run_uuid = _parse_run_id(run_id)
    if not repo.get_run(run_uuid):
        from app.core.exceptions import RunNotFoundError
        raise RunNotFoundError(run_id)
    assessments = repo.list_readiness_assessments(run_uuid)
    return {"assessments": [_readiness_assessment_to_dict(ra) for ra in assessments]}


@router.get("/{run_id}/feature-recommendation")
def get_feature_recommendation(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get the target/feature/drop recommendation for a run."""
    repo = ProfileRunRepository(db)
    run_uuid = _parse_run_id(run_id)
    if not repo.get_run(run_uuid):
        from app.core.exceptions import RunNotFoundError
        raise RunNotFoundError(run_id)
    fr = repo.get_feature_recommendation(run_uuid)
    return {"feature_recommendation": _feature_recommendation_to_dict(fr) if fr else {}}


@router.get("/{run_id}/hierarchy")
def get_hierarchy(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get hierarchy chain."""
    repo = ProfileRunRepository(db)
    run_uuid = _parse_run_id(run_id)
    if not repo.get_run(run_uuid):
        from app.core.exceptions import RunNotFoundError
        raise RunNotFoundError(run_id)
    hierarchy = repo.get_hierarchy(run_uuid)
    return {"hierarchy": hierarchy or {}}


@router.get("/{run_id}/rule-evaluations")
def get_rule_evaluations(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get persisted business rule evaluation results for a run."""
    repo = ProfileRunRepository(db)
    run_uuid = _parse_run_id(run_id)
    if not repo.get_run(run_uuid):
        from app.core.exceptions import RunNotFoundError
        raise RunNotFoundError(run_id)
    rule_repo = RuleRepository(db)
    evaluations = rule_repo.list_evaluations(run_uuid)
    return {"evaluations": [_rule_evaluation_to_dict(e) for e in evaluations]}


@router.get("/{run_id}/charts")
def get_charts(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get chart specifications."""
    repo = ProfileRunRepository(db)
    run_uuid = _parse_run_id(run_id)
    if not repo.get_run(run_uuid):
        from app.core.exceptions import RunNotFoundError
        raise RunNotFoundError(run_id)
    charts = repo.list_charts(run_uuid)
    return {"charts": [_chart_to_dict(c) for c in charts]}


@router.post("/{run_id}/charts/{chart_id}/drill-down")
def drill_down(run_id: str, chart_id: str, body: DrillDownRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Execute a drill-down request."""
    repo = ProfileRunRepository(db)
    run_uuid = _parse_run_id(run_id)
    if not repo.get_run(run_uuid):
        from app.core.exceptions import RunNotFoundError
        raise RunNotFoundError(run_id)

    from app.services.charts.drilldown_service import DrillDownService
    service = DrillDownService()
    hierarchy = repo.get_hierarchy(run_uuid)
    hierarchy_levels = (hierarchy or {}).get("level_columns", [])

    return service.execute_drill_down(
        cubes=[_cube_to_dict(c) for c in repo.list_drill_down_cubes(run_uuid)],
        chart_id=chart_id,
        hierarchy_levels=hierarchy_levels,
        selected_path=body.selected_path,
    )


# The drill path is bounded so one request cannot turn into an unbounded chain
# of group-bys. Twelve levels is deeper than any hierarchy this pipeline detects
# and matches the orchestrator's own validation.
MAX_DRILL_DEPTH = 12

# Cube generation stops storing groups at this many (see the profiling
# pipeline's cube node). A cube sitting on the cap cannot prove it is complete.
CUBE_GROUP_CAP = 100


def _cube_matches(cube, dimension: str, metric: str | None, aggregation: str,
                  filters: list[dict[str, str]]) -> bool:
    """A cube may only answer a query it matches exactly.

    Dimension, metric, aggregation and the full ancestor path must all agree.
    A near-match would show real numbers under the wrong label, which is worse
    than returning nothing.
    """
    if cube.level_column != dimension:
        return False
    if (cube.metric_column or None) != (metric or None):
        return False
    if (cube.aggregation or "").lower() != aggregation:
        return False
    columns = [f["column"] for f in filters]
    if len(columns) != len(set(columns)):
        # A column pinned twice is applied sequentially over the rows, so
        # "A = x" then "A = y" matches nothing. A cube's dimension_path is a
        # mapping, so the same pair collapses to one entry and would match a
        # cube built for "A = y" alone — answering a different, non-empty
        # question and labelling it complete. Recompute instead; the row path
        # gets the contradiction right.
        return False
    path = {str(k): str(v) for k, v in (cube.dimension_path_json or {}).items()}
    requested = {f["column"]: str(f["value"]) for f in filters}
    return path == requested


def _cube_points(cube) -> tuple[list[dict[str, Any]], int]:
    """The cube's groups, and how many rows back them."""
    data = [
        {"label": str(row.get("label")), "value": row.get("value"), "count": row.get("count")}
        for row in (cube.aggregated_data_json or [])
        if isinstance(row, dict) and row.get("label") is not None
    ]
    contributing = 0
    for row in data:
        try:
            contributing += int(row.get("count") or 0)
        except (TypeError, ValueError):
            continue
    return data, contributing


def _cube_is_complete(cube) -> bool:
    """True only when the cube provably kept every row in its scope.

    Cube generation drops groups below MIN_CUBE_GROUP_SIZE, drops groups with no
    numeric value, and stops at CUBE_GROUP_CAP groups. None of that is recorded
    on the cube — but the per-group counts and the cube's own record_count are,
    and when they agree nothing was dropped. Anything else is treated as partial
    and recomputed from the source dataset instead, so a cube is only ever used
    as a free shortcut to the same answer, never as a cheaper different one.
    """
    data, contributing = _cube_points(cube)
    if not data or len(data) >= CUBE_GROUP_CAP:
        return False
    record_count = getattr(cube, "record_count", None)
    if not isinstance(record_count, int) or record_count <= 0:
        return False
    return contributing == record_count


def _cube_response(cube, filters: list[dict[str, str]], *, complete: bool) -> dict[str, Any]:
    """Serve a breakdown from a pre-aggregated cube, stating its coverage."""
    data, contributing = _cube_points(cube)
    title = (
        f"{(cube.aggregation or 'sum').capitalize()} of {cube.metric_column} by {cube.level_column}"
    )
    if complete:
        description = (
            f"Pre-aggregated during profiling from all {cube.record_count:,} rows in scope. "
            "Every row in scope is accounted for in the groups shown, so this is the same "
            "answer recomputing from the source rows would give."
        )
    else:
        description = (
            f"Pre-aggregated during profiling from {cube.record_count:,} rows in scope; "
            f"{contributing:,} rows back the groups shown. Groups below the configured minimum "
            f"size were dropped then, and at most {CUBE_GROUP_CAP} were kept, so this view may "
            "not be exhaustive. The source dataset for this run could not be read back, so it "
            "could not be recomputed exactly."
        )
    return {
        "dimension": cube.level_column,
        "metric": cube.metric_column,
        "aggregation": cube.aggregation,
        "filters": filters,
        "data": data,
        "rows_used": contributing,
        # A partial cube cannot say how many rows the filters matched. A complete
        # one can: its scope is exactly this filter path. Neither knows how large
        # the whole dataset was, so rows_total stays unknown either way.
        "rows_matched": cube.record_count if complete else None,
        "rows_total": None,
        "rows_in_cube_scope": cube.record_count,
        "group_count": len(data),
        "returned_group_count": len(data),
        "truncated": False,
        "coverage": "complete" if complete else "partial",
        "source": "cube",
        "title": title,
        "description": description,
    }


def _source_reference_for(run, hint) -> SourceDatasetReference:
    """The run's own dataset reference, with caller-supplied blanks filled in.

    What the run recorded always wins. A hint can only supply something the run
    does not know — which is how a run created before references were persisted
    becomes answerable again — never redirect a run at different data.
    """
    recorded = SourceDatasetReference(
        fingerprint=getattr(run, "source_fingerprint", None) or None,
        uri=getattr(run, "source_dataset_uri", None) or None,
        filename=getattr(run, "source_filename", None) or None,
        sheet_name=getattr(run, "source_sheet_name", None) or None,
    )
    if hint is None:
        return recorded
    return recorded.merged_with(
        SourceDatasetReference(
            fingerprint=(hint.fingerprint or None),
            uri=(hint.uri or None),
            filename=(hint.filename or None),
            sheet_name=(hint.sheet_name or None),
        )
    )


def _unresolvable_detail(run) -> str:
    """Say which of the two failures this is, because the fixes differ."""
    if getattr(run, "source_fingerprint", None) or getattr(run, "source_dataset_uri", None):
        return (
            "This run references a stored source dataset, but it could not be read back, and "
            "no pre-aggregated cube matches this breakdown — so it cannot be computed. Check "
            "that this service can reach the master dataset store (MASTER_DATASET_STORAGE_DIR "
            "or MASTER_DATASET_CONTENT_URL)."
        )
    return (
        "This run recorded no reference to its source dataset, its retained working copy is "
        "gone, and no pre-aggregated cube matches this breakdown — so it cannot be computed. "
        "Runs profiled before source references were recorded can only answer the breakdowns "
        "their cubes already cover; re-run the analysis on this dataset to make every "
        "breakdown available."
    )


@router.post("/{run_id}/aggregate")
def aggregate_run(
    run_id: str,
    body: AggregateRequest,
    settings: Settings = Depends(get_cached_settings),
    db: Session = Depends(get_db),
    user: str = Depends(get_current_user),
) -> dict[str, Any]:
    """Group a completed run's rows by any dimension and measure any metric.

    Source precedence, cheapest exact answer first:
      1. a pre-aggregated cube matching this breakdown exactly that can prove it
         kept every row in scope — free, and identical to recomputing;
      2. the run's source dataset — the working copy retained from the run if it
         is still there, otherwise the permanent copy the run references. Exact,
         at any drill depth, for any dimension x metric pairing;
      3. an exactly-matching cube that cannot prove completeness, labelled
         partial;
      4. nothing — a 409 naming which reference failed, so the caller tells the
         analyst the truth instead of rendering a fabricated breakdown.

    Retention (RETAIN_RUN_FILES / RUN_FILE_RETENTION_HOURS) only decides whether
    step 2 needs to go back to the permanent copy. It no longer decides whether
    a breakdown can be answered at all.
    """
    from app.services.charts.aggregation_service import (
        AggregationRequestError,
        AggregationService,
        load_run_dataframe,
    )

    repo = ProfileRunRepository(db)
    run_uuid = _parse_run_id(run_id)
    run = repo.get_run(run_uuid)
    if not run:
        from app.core.exceptions import RunNotFoundError
        raise RunNotFoundError(run_id)

    if len(body.filters) > MAX_DRILL_DEPTH:
        raise HTTPException(
            status_code=400, detail=f"The drill path is limited to {MAX_DRILL_DEPTH} levels."
        )

    dimension = body.dimension.strip()
    metric = (body.metric or "").strip() or None
    aggregation = (body.aggregation or "sum").strip().lower()
    filters = [{"column": f.column.strip(), "value": f.value} for f in body.filters if f.column.strip()]

    if not dimension:
        raise HTTPException(status_code=400, detail="A dimension column is required.")

    matching_cube = next(
        (
            cube
            for cube in repo.list_drill_down_cubes(run_uuid)
            if _cube_matches(cube, dimension, metric, aggregation, filters)
        ),
        None,
    )
    if matching_cube is not None and _cube_is_complete(matching_cube):
        return _cube_response(matching_cube, filters, complete=True)

    service = AggregationService(max_groups=getattr(settings, "max_aggregation_groups", 200))
    resolver = SourceDatasetResolver(settings, TemporaryStorage(settings))
    resolved = resolver.resolve(run_uuid, _source_reference_for(run, body.source_reference))

    if resolved is not None:
        try:
            df = load_run_dataframe(settings, resolved.path, sheet_name=resolved.sheet_name)
        except Exception as e:  # noqa: BLE001 — fall back to cubes, but say why in the log
            logger.warning(
                "aggregate_source_reload_failed",
                run_id=run_id,
                origin=resolved.origin,
                error=str(e),
            )
        else:
            recomputed = not resolved.is_retained_run_file
            try:
                return service.aggregate(
                    df,
                    dimension,
                    metric,
                    aggregation,
                    filters,
                    source="source_dataset" if recomputed else "dataset",
                    source_note=(
                        "Recomputed from the stored source dataset for this run — the working "
                        "copy kept during profiling had already been swept."
                        if recomputed
                        else None
                    ),
                    include_next_levels=body.include_next_levels,
                    max_next_levels=getattr(settings, "max_next_level_candidates", 40),
                )
            except AggregationRequestError as e:
                raise HTTPException(status_code=e.status_code, detail=e.message)

    if matching_cube is not None:
        return _cube_response(matching_cube, filters, complete=False)

    raise HTTPException(status_code=409, detail=_unresolvable_detail(run))
