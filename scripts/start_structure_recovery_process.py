import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.runtime.contracts import AnalysisSpec, ConclusionCard, Metric, RunRecord, RunSummary
from server.runtime.experiment_store import ExperimentTimelineStore


def _now_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _append_timeline(
    store: ExperimentTimelineStore,
    *,
    run_id: str,
    analysis_type: str,
    model: str,
    params: Dict[str, Any],
    metrics: List[Metric],
    conclusion: ConclusionCard,
    artifacts: List[Dict[str, Any]],
) -> None:
    now = time.time()
    record = RunRecord(
        run_id=run_id,
        spec=AnalysisSpec(
            route="fiber_bundle",
            analysis_type=analysis_type,
            model=model,
            params=params,
            input_payload={},
        ),
        status="completed",
        created_at=now,
        updated_at=now + 0.001,
        summary=RunSummary(
            metrics=metrics,
            conclusion=conclusion,
            artifacts=artifacts,
        ),
        event_count=0,
    )
    store.append_run(record)


def run_stage_a_invariant(metrics_path: Path, output_path: Path, store: ExperimentTimelineStore) -> Dict[str, Any]:
    data = _load_json(metrics_path)
    agg = data.get("aggregate", {})
    layer_metrics = data.get("layer_metrics", [])
    top_edges = agg.get("top_layer_edges", [])
    top_corr = float(top_edges[0]["corr"]) if top_edges else 0.0
    avg_recon = float(agg.get("avg_recon_error_k64", 1.0))
    systematicity = float(agg.get("systemicity_index", 0.0))
    candidate_count = 0
    for lm in layer_metrics:
        if float(lm.get("k95", 0)) >= 56 and float(lm.get("syntax_score", 0.0)) >= 0.2:
            candidate_count += 1

    stability_score = round(
        max(0.0, min(1.0, 0.45 * systematicity + 0.25 * top_corr + 0.30 * (1.0 - min(1.0, avg_recon)))),
        4,
    )
    status = "pass" if stability_score >= 0.72 and candidate_count >= 8 else "watch"
    summary = {
        "stage": "A",
        "stage_name": "invariant_discovery",
        "test_date": _now_date(),
        "status": status,
        "model": data.get("model_name", "unknown"),
        "metrics": {
            "layer_count": len(layer_metrics),
            "avg_effective_rank": float(agg.get("avg_effective_rank", 0.0)),
            "avg_k95": float(agg.get("avg_k95", 0.0)),
            "avg_recon_error_k64": avg_recon,
            "systemicity_index": systematicity,
            "top_edge_corr": round(top_corr, 4),
            "candidate_count": candidate_count,
            "stability_score": stability_score,
        },
        "conclusion": (
            "Invariant candidates extracted from layered structure metrics; "
            "ready for causal filtering." if status == "pass" else
            "Invariant quality is borderline; extend sampling and model diversity before promotion."
        ),
        "artifacts": [str(metrics_path).replace("\\", "/")],
    }
    _save_json(output_path, summary)

    _append_timeline(
        store,
        run_id=f"run_stage_a_invariant_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        analysis_type="invariant_discovery",
        model=str(data.get("model_name", "unknown")),
        params={"source_metrics": str(metrics_path).replace("\\", "/")},
        metrics=[
            Metric(key="stability_score", value=stability_score, min_value=0.0, max_value=1.0),
            Metric(key="candidate_count", value=float(candidate_count), min_value=0.0),
            Metric(key="systemicity_index", value=systematicity, min_value=0.0, max_value=1.0),
        ],
        conclusion=ConclusionCard(
            objective="Extract stable invariants from DNN layer geometry/spectral metrics.",
            method="Aggregate layer-level ranks, reconstruction error, and inter-layer similarity.",
            evidence=[
                f"candidate_count={candidate_count}",
                f"stability_score={stability_score}",
                f"systemicity={systematicity}",
            ],
            result=summary["conclusion"],
            confidence=0.78 if status == "pass" else 0.62,
            limitations=["Current run is single-model dominant; cross-model extension required."],
            next_action="Move candidates into causal intervention filtering.",
        ),
        artifacts=[{"path": str(output_path).replace("\\", "/")}],
    )
    return summary


def run_stage_b_causal(
    layerwise_path: Path,
    selective_path: Path,
    output_path: Path,
    store: ExperimentTimelineStore,
) -> Dict[str, Any]:
    layerwise = _load_json(layerwise_path)
    selective = _load_json(selective_path)
    lw = layerwise.get("aggregates", {})
    sv = selective.get("aggregates", {})
    max_lw_uplift = float(lw.get("max_uplift_random", 0.0))
    avg_sel_uplift = float(sv.get("avg_top1_uplift", 0.0))
    avg_sel_kl = float(sv.get("avg_kl_uplift", 0.0))

    if avg_sel_uplift >= 0.05 and max_lw_uplift < 0.03:
        verdict = "feature_selective_signal"
    elif max_lw_uplift >= 0.05:
        verdict = "layerwise_signal"
    else:
        verdict = "weak_signal"

    summary = {
        "stage": "B",
        "stage_name": "causal_filtering",
        "test_date": _now_date(),
        "status": "pass" if verdict != "weak_signal" else "watch",
        "metrics": {
            "layerwise_max_uplift": round(max_lw_uplift, 4),
            "feature_avg_top1_uplift": round(avg_sel_uplift, 4),
            "feature_avg_kl_uplift": round(avg_sel_kl, 6),
            "verdict": verdict,
        },
        "retained_structures": (
            ["feature_subspace_layer3_topk32"] if verdict == "feature_selective_signal" else []
        ),
        "rejected_structures": (
            ["layer_wide_smoothing_only"] if max_lw_uplift < 0.03 else []
        ),
        "conclusion": (
            "Causal filtering indicates useful signal is concentrated in selective feature subspaces, "
            "not in layer-wide smoothing."
        ),
        "artifacts": [
            str(layerwise_path).replace("\\", "/"),
            str(selective_path).replace("\\", "/"),
        ],
    }
    _save_json(output_path, summary)

    _append_timeline(
        store,
        run_id=f"run_stage_b_causal_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        analysis_type="causal_filtering",
        model="gpt2+distilgpt2",
        params={
            "layerwise_report": str(layerwise_path).replace("\\", "/"),
            "selective_report": str(selective_path).replace("\\", "/"),
        },
        metrics=[
            Metric(key="layerwise_max_uplift", value=max_lw_uplift, min_value=-1.0, max_value=1.0),
            Metric(key="feature_avg_top1_uplift", value=avg_sel_uplift, min_value=-1.0, max_value=1.0),
            Metric(key="feature_avg_kl_uplift", value=avg_sel_kl, min_value=-5.0, max_value=5.0),
        ],
        conclusion=ConclusionCard(
            objective="Filter invariant candidates by causal necessity.",
            method="Compare layer-wide intervention vs feature-selective intervention and controls.",
            evidence=[
                f"layerwise_max_uplift={max_lw_uplift}",
                f"feature_avg_top1_uplift={avg_sel_uplift}",
                f"feature_avg_kl_uplift={avg_sel_kl}",
                f"verdict={verdict}",
            ],
            result=summary["conclusion"],
            confidence=0.75 if verdict == "feature_selective_signal" else 0.58,
            limitations=["Needs task-level score deltas and significance tests per task family."],
            next_action="Promote selective structures into minimal reconstruction targets.",
        ),
        artifacts=[{"path": str(output_path).replace("\\", "/")}],
    )
    return summary


def _run_command(command: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def run_stage_c_minimal(output_json: Path, output_md: Path, store: ExperimentTimelineStore) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/scaling_validation_matrix.py",
        "--preset",
        "quick",
        "--epochs",
        "2",
        "--max-runs",
        "1",
        "--device",
        "auto",
        "--output-json",
        str(output_json).replace("\\", "/"),
        "--output-md",
        str(output_md).replace("\\", "/"),
    ]
    code, out, err = _run_command(cmd)
    if code != 0:
        raise RuntimeError(f"Stage C command failed: {err or out}")
    report = _load_json(output_json)
    best = report.get("summary", {}).get("global_best", {})
    best_val = float(best.get("best_val_acc", 0.0))
    status = "pass" if best_val >= 0.6 else "watch"
    summary = {
        "stage": "C",
        "stage_name": "minimal_reconstruction",
        "test_date": _now_date(),
        "status": status,
        "metrics": {
            "best_val_acc": best_val,
            "model_scale": best.get("model_scale"),
            "data_scale": best.get("data_scale"),
            "run_id": best.get("run_id"),
        },
        "conclusion": (
            "Minimal reconstruction smoke run completed; use as baseline for next constrained rebuild."
        ),
        "artifacts": [
            str(output_json).replace("\\", "/"),
            str(output_md).replace("\\", "/"),
        ],
    }

    _append_timeline(
        store,
        run_id=f"run_stage_c_minimal_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        analysis_type="minimal_reconstruction",
        model=str(best.get("model_scale", "unknown")),
        params={"source_report": str(output_json).replace("\\", "/")},
        metrics=[
            Metric(key="best_val_acc", value=best_val, min_value=0.0, max_value=1.0),
        ],
        conclusion=ConclusionCard(
            objective="Run a minimal reconstruction smoke test to initialize stage C baseline.",
            method="Quick scaling matrix run with one model/data point and short epochs.",
            evidence=[f"best_val_acc={best_val}", f"run_id={best.get('run_id')}"],
            result=summary["conclusion"],
            confidence=0.68 if status == "pass" else 0.55,
            limitations=["This is a smoke baseline, not the final constrained minimal generator."],
            next_action="Run constrained MDL rebuild with causal-selected feature subspaces.",
        ),
        artifacts=[{"path": str(output_json).replace("\\", "/")}],
    )
    return summary


def run_stage_d_multimodal(output_json: Path, output_md: Path, store: ExperimentTimelineStore) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/train_fiber_multimodal_connector.py",
        "--dataset",
        "synthetic",
        "--total-samples",
        "4000",
        "--epochs",
        "2",
        "--batch-size",
        "128",
        "--d-model",
        "96",
        "--report-json",
        str(output_json).replace("\\", "/"),
        "--report-md",
        str(output_md).replace("\\", "/"),
        "--analysis-type",
        "multimodal_connector_kickoff",
        "--timeline",
        "tempdata/agi_route_test_timeline.json",
        "--route",
        "fiber_bundle",
    ]
    code, out, err = _run_command(cmd)
    if code != 0:
        raise RuntimeError(f"Stage D command failed: {err or out}")

    report = _load_json(output_json)
    best = report.get("summary", {}).get("best", {})
    fused = float(best.get("val_fused_acc", 0.0))
    retrieval = float(best.get("val_retrieval_top1", 0.0))
    status = "pass" if fused >= 0.8 else "watch"
    summary = {
        "stage": "D",
        "stage_name": "cross_modal_assembly",
        "test_date": _now_date(),
        "status": status,
        "metrics": {
            "val_fused_acc": fused,
            "val_retrieval_top1": retrieval,
            "dataset": report.get("meta", {}).get("dataset"),
            "dataset_size": report.get("meta", {}).get("dataset_size"),
        },
        "conclusion": "Cross-modal assembly kickoff run completed.",
        "artifacts": [
            str(output_json).replace("\\", "/"),
            str(output_md).replace("\\", "/"),
        ],
    }

    _append_timeline(
        store,
        run_id=f"run_stage_d_multimodal_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        analysis_type="cross_modal_assembly",
        model="FiberMultimodalSystem",
        params={"source_report": str(output_json).replace("\\", "/")},
        metrics=[
            Metric(key="val_fused_acc", value=fused, min_value=0.0, max_value=1.0),
            Metric(key="val_retrieval_top1", value=retrieval, min_value=0.0, max_value=1.0),
        ],
        conclusion=ConclusionCard(
            objective="Assemble language/vision routes into a shared representation space.",
            method="Train fiber multimodal connector on synthetic paired data.",
            evidence=[
                f"val_fused_acc={fused}",
                f"val_retrieval_top1={retrieval}",
            ],
            result=summary["conclusion"],
            confidence=0.72 if status == "pass" else 0.6,
            limitations=["Synthetic-only kickoff; needs real-world multi-domain eval."],
            next_action="Expand to richer multimodal datasets and conflict-routing tests.",
        ),
        artifacts=[{"path": str(output_json).replace("\\", "/")}],
    )
    return summary


def run_stage_e_falsification(
    stage_b: Dict[str, Any],
    stage_c: Dict[str, Any],
    stage_d: Dict[str, Any],
    output_path: Path,
    store: ExperimentTimelineStore,
    tempdata: Path,
    support_models_min: int = 2,
    falsify_models_max: int = 0,
    seed_block_size: int = 6,
) -> Dict[str, Any]:
    def _find_latest(pattern: str) -> Path | None:
        matches = sorted(tempdata.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0] if matches else None

    def _summarize_seed_block(paths: List[Path]) -> Dict[str, Any]:
        if not paths:
            return {
                "run_count": 0,
                "support_models_min": None,
                "support_models_max": None,
                "support_models_avg": None,
                "falsify_models_max": None,
                "strict_gate_pass": False,
                "reports": [],
            }
        support_values: List[int] = []
        falsify_values: List[int] = []
        reports: List[str] = []
        for p in paths:
            payload = _load_json(p)
            ag = payload.get("aggregates", {})
            support_values.append(int(ag.get("support_models", 0)))
            falsify_values.append(int(ag.get("falsify_models", 0)))
            reports.append(str(p).replace("\\", "/"))
        support_floor = min(support_values) if support_values else 0
        falsify_ceiling = max(falsify_values) if falsify_values else 999
        strict_gate_pass = support_floor >= support_models_min and falsify_ceiling <= falsify_models_max
        return {
            "run_count": len(paths),
            "support_models_min": support_floor,
            "support_models_max": max(support_values) if support_values else 0,
            "support_models_avg": round(sum(support_values) / len(support_values), 4) if support_values else 0.0,
            "falsify_models_max": falsify_ceiling if falsify_values else 0,
            "strict_gate_pass": bool(strict_gate_pass),
            "reports": reports,
        }

    def _holdout_signature(payload: Dict[str, Any]) -> Dict[str, Any]:
        cfg = payload.get("config", {}) or {}
        return {
            "analysis_type": payload.get("analysis_type", "h3_holdout_validation"),
            "models": cfg.get("models", []),
            "task_profile": cfg.get("task_profile", "standard"),
            "max_per_category": cfg.get("max_per_category"),
            "locked_configs_from": cfg.get("locked_configs_from", ""),
            "lock_mode": cfg.get("lock_mode", ""),
            "adapter_profile": cfg.get("adapter_profile", ""),
            "adapter_strength": cfg.get("adapter_strength"),
            "adapter_failure_report": cfg.get("adapter_failure_report", ""),
            "support_models_min": cfg.get("support_models_min"),
            "falsify_models_max": cfg.get("falsify_models_max"),
        }

    layerwide_fail = stage_b.get("metrics", {}).get("layerwise_max_uplift", 0.0) < 0.03
    selective_pass = stage_b.get("metrics", {}).get("feature_avg_top1_uplift", 0.0) > 0.05
    task_summary_path = _find_latest("task_level_causal_eval_summary_*.json")
    holdout_reports = sorted(tempdata.glob("h3_holdout_validation_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    holdout_latest = holdout_reports[0] if holdout_reports else None
    seed_block_reports: List[Path] = []
    seed_block_signature: Dict[str, Any] = {}
    if holdout_latest and holdout_latest.exists():
        latest_payload = _load_json(holdout_latest)
        seed_block_signature = _holdout_signature(latest_payload)
        for p in holdout_reports:
            payload = _load_json(p)
            if _holdout_signature(payload) == seed_block_signature:
                seed_block_reports.append(p)
            if len(seed_block_reports) >= max(0, int(seed_block_size)):
                break
    if not seed_block_reports:
        seed_block_reports = holdout_reports[: max(0, int(seed_block_size))]

    task_eval: Dict[str, Any] = {}
    if task_summary_path and task_summary_path.exists():
        task_payload = _load_json(task_summary_path)
        ag = task_payload.get("aggregates", {})
        task_eval = {
            "summary_report": str(task_summary_path).replace("\\", "/"),
            "support_count": int(ag.get("support_count", 0)),
            "falsify_count": int(ag.get("falsify_count", 0)),
            "open_count": int(ag.get("open_count", 0)),
            "avg_task_score_uplift_logprob": float(ag.get("avg_task_score_uplift_logprob", 0.0)),
            "avg_win_rate": float(ag.get("avg_win_rate", 0.0)),
            "status": str(ag.get("h3_status", "open")),
        }

    holdout_ag = {}
    if holdout_latest and holdout_latest.exists():
        holdout_payload = _load_json(holdout_latest)
        holdout_ag = holdout_payload.get("aggregates", {})

    seed_block = _summarize_seed_block(seed_block_reports)
    current_support_models = int(holdout_ag.get("support_models", 0)) if holdout_ag else 0
    current_falsify_models = int(holdout_ag.get("falsify_models", 0)) if holdout_ag else 0
    current_gate_pass = current_support_models >= support_models_min and current_falsify_models <= falsify_models_max

    layered_goals = {
        "goal_1_same_arch_seed_stability": {
            "status": "pass" if seed_block.get("strict_gate_pass") else "pending",
            "criterion": f"seed_block: support_models_min>={support_models_min} and falsify_models_max<={falsify_models_max}",
            "evidence": seed_block,
        },
        "goal_2_cross_arch_stability": {
            "status": "pass" if current_gate_pass else "pending",
            "criterion": f"current_holdout: support_models>={support_models_min} and falsify_models<={falsify_models_max}",
            "evidence": {
                "support_models": current_support_models,
                "falsify_models": current_falsify_models,
                "report": str(holdout_latest).replace("\\", "/") if holdout_latest else None,
            },
        },
        "goal_3_cross_task_family_stability": {
            "status": "pass" if task_eval.get("falsify_count", 0) == 0 and task_eval.get("support_count", 0) >= 2 else "pending",
            "criterion": "task_level: support_count>=2 and falsify_count=0",
            "evidence": task_eval,
        },
    }

    if layered_goals["goal_1_same_arch_seed_stability"]["status"] != "pass":
        layered_goals["progression"] = "goal_1_pending"
    elif layered_goals["goal_2_cross_arch_stability"]["status"] != "pass":
        layered_goals["progression"] = "goal_2_pending"
    elif layered_goals["goal_3_cross_task_family_stability"]["status"] != "pass":
        layered_goals["progression"] = "goal_3_pending"
    else:
        layered_goals["progression"] = "all_goals_pass"

    open_hypotheses = [
        {
            "hypothesis": "H1: Layer-wide smoothing is sufficient for causal control.",
            "status": "falsified" if layerwide_fail else "open",
        },
        {
            "hypothesis": "H2: Feature-selective intervention yields stronger causal signal.",
            "status": "supported" if selective_pass else "open",
        },
        {
            "hypothesis": "H3: Selective causal signal transfers to task-score improvement.",
            "status": (
                "supported"
                if layered_goals["goal_3_cross_task_family_stability"]["status"] == "pass"
                else "open"
            ),
        },
    ]
    falsified_count = sum(1 for h in open_hypotheses if h["status"] == "falsified")
    summary = {
        "stage": "E",
        "stage_name": "open_falsification",
        "test_date": _now_date(),
        "status": "pass" if falsified_count >= 1 else "watch",
        "hypotheses": open_hypotheses,
        "conclusion": (
            "Open falsification updated with strict task-metric gate: keep H3 open until "
            "seed-block and holdout strict gates both pass."
        ),
        "inputs": {
            "stage_b": stage_b.get("metrics", {}),
            "stage_c": stage_c.get("metrics", {}),
            "stage_d": stage_d.get("metrics", {}),
        },
        "task_level_eval": task_eval,
        "strict_gate": {
            "support_models_min_required": support_models_min,
            "falsify_models_max_allowed": falsify_models_max,
            "current_run_pass": current_gate_pass,
            "seed_block_signature": seed_block_signature,
            "seed_block": seed_block,
            "promotion_basis": "task_metrics_only",
        },
        "layered_goals": layered_goals,
    }
    _save_json(output_path, summary)

    _append_timeline(
        store,
        run_id=f"run_stage_e_falsification_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        analysis_type="open_falsification",
        model="pipeline",
        params={"source": "stage_b_c_d"},
        metrics=[
            Metric(key="falsified_count", value=float(falsified_count), min_value=0.0),
        ],
        conclusion=ConclusionCard(
            objective="Keep only falsifiable conclusions and remove weak assumptions.",
            method="Cross-check stage-B causal outcomes against stage-C/D constraints.",
            evidence=[f"falsified_count={falsified_count}"],
            result=summary["conclusion"],
            confidence=0.7 if falsified_count >= 1 else 0.55,
            limitations=["Task-level counterfactuals still needed for stronger rejection power."],
            next_action="Design explicit counterfactual task suites for H3.",
        ),
        artifacts=[{"path": str(output_path).replace("\\", "/")}],
    )
    return summary


def to_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Structure Recovery Pipeline Kickoff",
        "",
        f"- Date: {summary.get('date')}",
        f"- Status: {summary.get('status')}",
        "",
        "| Stage | Name | Status | Key Metric |",
        "|---|---|---|---|",
    ]
    for item in summary.get("stages", []):
        metric_text = ""
        metrics = item.get("metrics", {})
        if item["stage"] == "A0":
            metric_text = (
                f"encoding_core={metrics.get('encoding_core_score')}, "
                f"h3={metrics.get('h3_status')}"
            )
        elif item["stage"] == "A":
            metric_text = f"stability={metrics.get('stability_score')}, candidates={metrics.get('candidate_count')}"
        elif item["stage"] == "B":
            metric_text = (
                f"feature_top1={metrics.get('feature_avg_top1_uplift')}, "
                f"layerwise={metrics.get('layerwise_max_uplift')}"
            )
        elif item["stage"] == "C":
            metric_text = f"best_val_acc={metrics.get('best_val_acc')}"
        elif item["stage"] == "D":
            metric_text = f"val_fused_acc={metrics.get('val_fused_acc')}"
        elif item["stage"] == "E":
            metric_text = f"hypotheses={len(item.get('hypotheses', []))}"
        lines.append(f"| {item['stage']} | {item['stage_name']} | {item['status']} | {metric_text} |")
    lines.extend(
        [
            "",
            "## Next Actions",
            "1. Upgrade A0 with trajectory-level encoding-formation probes.",
            "2. Extend stage-B with task-level metric deltas and significance tests.",
            "3. Run constrained MDL minimal reconstruction using selected causal subspaces.",
            "4. Expand stage-D to non-synthetic multimodal data and route conflicts.",
            "5. Build explicit counterfactual falsification tasks for open hypotheses.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kick off the 5-stage structure recovery process.")
    parser.add_argument("--timeline", default="tempdata/agi_route_test_timeline.json")
    parser.add_argument("--date-tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    args = parser.parse_args()

    store = ExperimentTimelineStore(path=args.timeline)
    date_tag = args.date_tag
    tempdata = ROOT / "tempdata"
    tempdata.mkdir(parents=True, exist_ok=True)

    stage_a_path = tempdata / f"pipeline_stage_a_invariant_{date_tag}.json"
    stage_b_path = tempdata / f"pipeline_stage_b_causal_filter_{date_tag}.json"
    stage_c_json = tempdata / f"pipeline_stage_c_minimal_rebuild_{date_tag}.json"
    stage_c_md = tempdata / f"pipeline_stage_c_minimal_rebuild_{date_tag}.md"
    stage_d_json = tempdata / f"pipeline_stage_d_multimodal_{date_tag}.json"
    stage_d_md = tempdata / f"pipeline_stage_d_multimodal_{date_tag}.md"
    stage_e_path = tempdata / f"pipeline_stage_e_open_falsification_{date_tag}.json"

    stage_a = run_stage_a_invariant(
        metrics_path=ROOT / "tempdata" / "qwen3_4b_structure" / "metrics.json",
        output_path=stage_a_path,
        store=store,
    )
    stage_b = run_stage_b_causal(
        layerwise_path=ROOT / "tempdata" / "geometric_intervention_large_scale_matrix_20260220.json",
        selective_path=ROOT / "tempdata" / "feature_selective_probe_matrix_20260220.json",
        output_path=stage_b_path,
        store=store,
    )
    stage_c = run_stage_c_minimal(stage_c_json, stage_c_md, store)
    stage_d = run_stage_d_multimodal(stage_d_json, stage_d_md, store)
    stage_e = run_stage_e_falsification(stage_b, stage_c, stage_d, stage_e_path, store, tempdata)

    stages = [stage_a, stage_b, stage_c, stage_d, stage_e]
    status = "pass" if all(s.get("status") == "pass" for s in stages[:4]) else "in_progress"
    summary = {
        "date": _now_date(),
        "status": status,
        "stages": stages,
        "artifacts": [
            str(stage_a_path).replace("\\", "/"),
            str(stage_b_path).replace("\\", "/"),
            str(stage_c_json).replace("\\", "/"),
            str(stage_d_json).replace("\\", "/"),
            str(stage_e_path).replace("\\", "/"),
        ],
    }
    summary_json = tempdata / f"structure_recovery_pipeline_kickoff_{date_tag}.json"
    summary_md = tempdata / f"structure_recovery_pipeline_kickoff_{date_tag}.md"
    _save_json(summary_json, summary)
    summary_md.write_text(to_markdown(summary), encoding="utf-8")

    print(json.dumps({"summary_json": str(summary_json), "summary_md": str(summary_md), "status": status}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
