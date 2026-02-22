import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.runtime.contracts import AnalysisSpec, ConclusionCard, Metric, RunRecord, RunSummary
from server.runtime.experiment_store import ExperimentTimelineStore
from scripts.start_structure_recovery_process import to_markdown


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _resolve_h3_summary(tempdata: Path, date_tag: str) -> Path:
    candidates = [
        tempdata / f"task_level_causal_eval_summary_{date_tag}_v3.json",
        tempdata / f"task_level_causal_eval_summary_{date_tag}_v2.json",
        tempdata / f"task_level_causal_eval_summary_{date_tag}.json",
        tempdata / "task_level_causal_eval_summary_20260220_v3.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No task-level causal summary file found for A0.")


def build_stage_a0(
    *,
    metrics_path: Path,
    stage_c_path: Path,
    h3_summary_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    metrics_data = _load_json(metrics_path)
    agg = metrics_data.get("aggregate", {})
    layer_metrics = metrics_data.get("layer_metrics", [])
    stage_c = _load_json(stage_c_path)
    h3 = _load_json(h3_summary_path)

    avg_k95 = float(agg.get("avg_k95", 0.0))
    avg_rank = float(agg.get("avg_effective_rank", 0.0))
    avg_recon = float(agg.get("avg_recon_error_k64", 1.0))
    systematicity = float(agg.get("systemicity_index", 0.0))
    best_val_acc = float((stage_c.get("metrics") or {}).get("best_val_acc", 0.0))

    k95_norm = _clamp01((avg_k95 - 40.0) / 30.0)
    rank_norm = _clamp01(avg_rank / 64.0)
    abstraction_score = round(0.6 * k95_norm + 0.4 * rank_norm, 4)

    precision_score = round(
        0.5 * _clamp01(1.0 - avg_recon) + 0.5 * _clamp01(best_val_acc),
        4,
    )

    selectivities: List[float] = []
    for lm in layer_metrics:
        c = float(lm.get("concept_score", 0.0))
        s = float(lm.get("syntax_score", 0.0))
        d = float(lm.get("domain_score", 0.0))
        denom = abs(c) + abs(s) + abs(d) + 1e-8
        selectivities.append(max(abs(c), abs(s), abs(d)) / denom)
    specificity_score = round(sum(selectivities) / len(selectivities), 4) if selectivities else 0.0

    systematicity_score = round(_clamp01(systematicity), 4)

    h3_aggs = h3.get("aggregates", {})
    h3_status = str(h3_aggs.get("h3_status", "unknown"))
    support_count = int(h3_aggs.get("support_count", 0))
    falsify_count = int(h3_aggs.get("falsify_count", 0))
    contradiction_penalty = 0.0
    if support_count > 0 and falsify_count > 0:
        contradiction_penalty = 0.12
    elif h3_status in {"mixed_open", "mixed_open_positive"}:
        contradiction_penalty = 0.06

    encoding_core_score = round(
        max(
            0.0,
            0.30 * abstraction_score
            + 0.30 * precision_score
            + 0.20 * specificity_score
            + 0.20 * systematicity_score
            - contradiction_penalty,
        ),
        4,
    )

    status = "pass" if encoding_core_score >= 0.62 else "watch"

    summary = {
        "stage": "A0",
        "stage_name": "encoding_genesis",
        "test_date": datetime.now(timezone.utc).date().isoformat(),
        "status": status,
        "metrics": {
            "abstraction_score": abstraction_score,
            "precision_score": precision_score,
            "specificity_score": specificity_score,
            "systematicity_score": systematicity_score,
            "encoding_core_score": encoding_core_score,
            "contradiction_penalty": contradiction_penalty,
            "h3_status": h3_status,
        },
        "conclusion": (
            "Encoding formation baseline established: representation quality is sufficient "
            "to proceed, but architecture-sensitive transfer risk remains."
            if status == "pass"
            else "Encoding formation quality remains unstable; improve feature extraction dynamics before promotion."
        ),
        "focus_questions": [
            "How are stable feature bases formed under local update dynamics?",
            "Which encoding subspaces are architecture-invariant vs architecture-specific?",
            "How does encoding transfer break across task categories?",
        ],
        "artifacts": [
            str(metrics_path).replace("\\", "/"),
            str(stage_c_path).replace("\\", "/"),
            str(h3_summary_path).replace("\\", "/"),
        ],
    }
    _save_json(output_path, summary)
    return summary


def append_timeline(stage_a0: Dict[str, Any], timeline_path: Path, output_path: Path) -> None:
    store = ExperimentTimelineStore(path=str(timeline_path))
    now = time.time()
    metrics = stage_a0.get("metrics", {})
    record = RunRecord(
        run_id=f"run_stage_a0_encoding_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        spec=AnalysisSpec(
            route="fiber_bundle",
            analysis_type="encoding_genesis",
            model="cross_model",
            params={"source_report": str(output_path).replace("\\", "/")},
            input_payload={},
        ),
        status="completed",
        created_at=now,
        updated_at=now + 0.001,
        summary=RunSummary(
            metrics=[
                Metric(key="abstraction_score", value=float(metrics.get("abstraction_score", 0.0)), min_value=0.0, max_value=1.0),
                Metric(key="precision_score", value=float(metrics.get("precision_score", 0.0)), min_value=0.0, max_value=1.0),
                Metric(key="specificity_score", value=float(metrics.get("specificity_score", 0.0)), min_value=0.0, max_value=1.0),
                Metric(key="systematicity_score", value=float(metrics.get("systematicity_score", 0.0)), min_value=0.0, max_value=1.0),
                Metric(key="encoding_core_score", value=float(metrics.get("encoding_core_score", 0.0)), min_value=0.0, max_value=1.0),
            ],
            conclusion=ConclusionCard(
                objective="Reconstruct the base mechanism of feature extraction and encoding formation (A0).",
                method="Derive encoding capability scores from structure metrics + task-transfer evidence.",
                evidence=[
                    f"encoding_core_score={metrics.get('encoding_core_score')}",
                    f"h3_status={metrics.get('h3_status')}",
                    f"penalty={metrics.get('contradiction_penalty')}",
                ],
                result=stage_a0.get("conclusion", ""),
                confidence=0.74 if stage_a0.get("status") == "pass" else 0.6,
                limitations=["Current encoding score is proxy-based; add training-trajectory probes next."],
                next_action="Add trajectory-level probes for encoding birth and stabilization.",
            ),
            artifacts=[{"path": str(output_path).replace("\\", "/")}],
        ),
        event_count=0,
    )
    store.append_run(record)


def update_pipeline_summary(stage_a0: Dict[str, Any], pipeline_path: Path, pipeline_md: Path, output_path: Path) -> None:
    pipeline = _load_json(pipeline_path)
    stages = pipeline.get("stages", [])
    replaced = False
    for i, s in enumerate(stages):
        if s.get("stage") == "A0":
            stages[i] = stage_a0
            replaced = True
            break
    if not replaced:
        stages.insert(0, stage_a0)
    pipeline["stages"] = stages

    # Keep overall status conservative:
    # pass only when A0..D all pass
    required = ["A0", "A", "B", "C", "D"]
    status_map = {s.get("stage"): s.get("status") for s in stages}
    pipeline["status"] = "pass" if all(status_map.get(k) == "pass" for k in required) else "in_progress"

    arts = set(pipeline.get("artifacts", []))
    arts.add(str(output_path).replace("\\", "/"))
    pipeline["artifacts"] = sorted(arts)

    _save_json(pipeline_path, pipeline)
    pipeline_md.write_text(to_markdown(pipeline), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A0 encoding genesis stage and merge into pipeline summary.")
    parser.add_argument("--date-tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--timeline", default="tempdata/agi_route_test_timeline.json")
    args = parser.parse_args()

    tempdata = ROOT / "tempdata"
    date_tag = args.date_tag

    metrics_path = tempdata / "qwen3_4b_structure" / "metrics.json"
    stage_c_path = tempdata / f"pipeline_stage_c_minimal_rebuild_{date_tag}.json"
    if not stage_c_path.exists():
        stage_c_path = tempdata / "pipeline_stage_c_minimal_rebuild_20260220.json"
    h3_summary_path = _resolve_h3_summary(tempdata, date_tag)
    output_path = tempdata / f"pipeline_stage_a0_encoding_genesis_{date_tag}.json"

    stage_a0 = build_stage_a0(
        metrics_path=metrics_path,
        stage_c_path=stage_c_path,
        h3_summary_path=h3_summary_path,
        output_path=output_path,
    )
    append_timeline(stage_a0, timeline_path=Path(args.timeline), output_path=output_path)

    pipeline_path = tempdata / f"structure_recovery_pipeline_kickoff_{date_tag}.json"
    if not pipeline_path.exists():
        pipeline_path = tempdata / "structure_recovery_pipeline_kickoff_20260220.json"
    pipeline_md = tempdata / pipeline_path.name.replace(".json", ".md")
    update_pipeline_summary(stage_a0, pipeline_path, pipeline_md, output_path)

    print(
        json.dumps(
            {
                "stage": "A0",
                "status": stage_a0.get("status"),
                "encoding_core_score": stage_a0.get("metrics", {}).get("encoding_core_score"),
                "output": str(output_path),
                "h3_summary": str(h3_summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
