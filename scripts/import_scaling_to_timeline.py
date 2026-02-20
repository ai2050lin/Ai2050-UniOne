import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.runtime.contracts import (
    AnalysisSpec,
    ConclusionCard,
    Metric,
    RunRecord,
    RunSummary,
)
from server.runtime.experiment_store import ExperimentTimelineStore


def _safe_load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_report_paths(paths: List[str]) -> Iterable[Path]:
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            yield p
            continue
        for item in sorted(Path().glob(raw)):
            if item.is_file() and item.suffix.lower() == ".json":
                yield item


def _metrics_from_run(run: dict) -> List[Metric]:
    m = run.get("metrics", {})
    return [
        Metric(
            key="best_val_acc",
            value=float(m.get("best_val_acc", 0.0)),
            min_value=0.0,
            max_value=1.0,
            description="Best validation accuracy in this run.",
        ),
        Metric(
            key="final_val_acc",
            value=float(m.get("final_val_acc", 0.0)),
            min_value=0.0,
            max_value=1.0,
            description="Final validation accuracy in this run.",
        ),
        Metric(
            key="train_seconds",
            value=float(m.get("train_seconds", 0.0)),
            min_value=0.0,
            description="Training wall-clock seconds.",
        ),
        Metric(
            key="samples_per_second",
            value=float(m.get("samples_per_second", 0.0)),
            min_value=0.0,
            description="Observed throughput.",
        ),
        Metric(
            key="generalization_gap",
            value=float(m.get("generalization_gap", 0.0)),
            min_value=-1.0,
            max_value=1.0,
            description="Train/val gap.",
        ),
    ]


def _confidence_from_acc(best_val_acc: float) -> float:
    if best_val_acc >= 0.999:
        return 0.98
    if best_val_acc >= 0.99:
        return 0.92
    if best_val_acc >= 0.90:
        return 0.82
    if best_val_acc >= 0.75:
        return 0.72
    if best_val_acc >= 0.50:
        return 0.62
    return 0.45


def _append_run_record(
    store: ExperimentTimelineStore,
    *,
    run_id: str,
    route: str,
    analysis_type: str,
    model_name: str,
    params: dict,
    metrics: List[Metric],
    conclusion: ConclusionCard,
    created_at: float,
):
    record = RunRecord(
        run_id=run_id,
        spec=AnalysisSpec(
            route=route,
            analysis_type=analysis_type,
            model=model_name,
            params=params,
            input_payload={},
        ),
        status="completed",
        created_at=created_at,
        updated_at=created_at + 0.001,
        summary=RunSummary(
            metrics=metrics,
            conclusion=conclusion,
            artifacts=[],
        ),
        event_count=0,
    )
    store.append_run(record)


def import_scaling_reports(
    timeline_path: str,
    route: str,
    report_paths: List[str],
) -> dict:
    store = ExperimentTimelineStore(path=timeline_path)
    now = time.time()
    imported = 0
    skipped = 0
    idx = 0

    for path in _iter_report_paths(report_paths):
        payload = _safe_load_json(path)

        # Report type A: run matrix report
        if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
            meta = payload.get("meta", {})
            for run in payload["runs"]:
                run_name = run.get("run_id", "unknown")
                run_id = f"run_scale_{path.stem}_{run_name}"
                model_scale = (run.get("model_scale") or {}).get("name", "unknown_model")
                data_scale = (run.get("data_scale") or {}).get("name", "unknown_data")
                best_val = float((run.get("metrics") or {}).get("best_val_acc", 0.0))
                confidence = _confidence_from_acc(best_val)

                params = {
                    "source_report": path.name,
                    "preset": meta.get("preset"),
                    "epochs": meta.get("epochs"),
                    "batch_size": meta.get("batch_size"),
                    "eval_batch_size": meta.get("eval_batch_size"),
                    "lr": meta.get("lr"),
                    "weight_decay": meta.get("weight_decay"),
                    "warmup_ratio": meta.get("warmup_ratio"),
                    "grad_accum_steps": meta.get("grad_accum_steps"),
                    "dropout": meta.get("dropout"),
                    "seed": run.get("seed"),
                    "model_scale": model_scale,
                    "data_scale": data_scale,
                    "param_count": run.get("param_count"),
                    "data_total_samples": (run.get("data_scale") or {}).get("total_samples"),
                }

                conclusion = ConclusionCard(
                    objective="Validate scaling behavior for large-model training.",
                    method="Run matrix training over model/data scales and record convergence metrics.",
                    evidence=[
                        f"Model={model_scale}",
                        f"Data={data_scale}",
                        f"best_val_acc={best_val:.6f}",
                        f"source={path.name}",
                    ],
                    result=(
                        f"Scaling run completed: {model_scale} on {data_scale}, "
                        f"best_val_acc={best_val:.6f}."
                    ),
                    confidence=confidence,
                    limitations=[
                        "Synthetic modular-addition task.",
                        "Needs OOD and robustness follow-up.",
                    ],
                    next_action="Continue with larger data scale and OOD stress tests.",
                )

                _append_run_record(
                    store,
                    run_id=run_id,
                    route=route,
                    analysis_type="scaling_validation",
                    model_name=model_scale,
                    params=params,
                    metrics=_metrics_from_run(run),
                    conclusion=conclusion,
                    created_at=now + idx * 0.01,
                )
                imported += 1
                idx += 1
            continue

        # Report type B: multi-seed aggregate summary
        if isinstance(payload, dict) and isinstance(payload.get("by_data_scale"), dict):
            run_id = f"run_scale_summary_{path.stem}"
            by_data = payload["by_data_scale"]
            evidence = []
            metrics = []
            for key, info in by_data.items():
                best_mean = float(info.get("best_mean", 0.0))
                best_std = float(info.get("best_std", 0.0))
                evidence.append(f"{key}: mean={best_mean:.6f}, std={best_std:.6f}")
                metrics.append(
                    Metric(
                        key=f"{key}_best_mean",
                        value=best_mean,
                        min_value=0.0,
                        max_value=1.0,
                        description=f"Multi-seed best accuracy mean for {key}.",
                    )
                )
                metrics.append(
                    Metric(
                        key=f"{key}_best_std",
                        value=best_std,
                        min_value=0.0,
                        description=f"Multi-seed best accuracy std for {key}.",
                    )
                )

            overall = payload.get("overall", {})
            overall_mean = float(overall.get("best_mean", 0.0))
            confidence = _confidence_from_acc(overall_mean)
            metrics.append(
                Metric(
                    key="overall_best_mean",
                    value=overall_mean,
                    min_value=0.0,
                    max_value=1.0,
                    description="Overall best accuracy mean across all seeds/scales.",
                )
            )

            conclusion = ConclusionCard(
                objective="Track multi-seed scaling stability.",
                method="Aggregate multiple tuned large-scale runsets and compute mean/std per data scale.",
                evidence=evidence[:8],
                result=f"Multi-seed summary imported from {path.name}.",
                confidence=confidence,
                limitations=[
                    "Aggregate stats depend on selected seed set.",
                    "Task family is still synthetic.",
                ],
                next_action="Increase seed count and add non-synthetic benchmarks.",
            )

            _append_run_record(
                store,
                run_id=run_id,
                route=route,
                analysis_type="scaling_validation_multiseed",
                model_name="m_8.5m",
                params={
                    "source_report": path.name,
                    "total_runs": payload.get("total_runs"),
                    "seed_runsets": len(payload.get("files", [])),
                },
                metrics=metrics,
                conclusion=conclusion,
                created_at=now + idx * 0.01,
            )
            imported += 1
            idx += 1
            continue

        skipped += 1

    return {"imported": imported, "skipped": skipped, "timeline_path": timeline_path}


def parse_args():
    parser = argparse.ArgumentParser(description="Import scaling test reports into AGI timeline JSON.")
    parser.add_argument(
        "--timeline",
        default="tempdata/agi_route_test_timeline.json",
        help="Target timeline file path.",
    )
    parser.add_argument("--route", default="fiber_bundle")
    parser.add_argument(
        "--reports",
        nargs="+",
        default=[
            "tempdata/scaling_validation_report_full_max.json",
            "tempdata/scaling_validation_report_m85_tuned.json",
            "tempdata/scaling_validation_report_m85_tuned_seed314.json",
            "tempdata/scaling_validation_report_m85_tuned_seed2026.json",
            "tempdata/scaling_validation_report_m85_tuned_seed4096.json",
            "tempdata/scaling_validation_report_m85_tuned_seed8192.json",
            "tempdata/scaling_validation_report_m85_d100k_e36_seed10001.json",
            "tempdata/scaling_validation_report_m85_d100k_e36_seed20002.json",
            "tempdata/scaling_validation_report_m85_d100k_e36_seed30003.json",
            "tempdata/scaling_validation_report_m85_xlarge_2m_3m.json",
            "tempdata/scaling_validation_m85_multiseed_summary_5seeds.json",
        ],
        help="Report paths or glob patterns.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = import_scaling_reports(
        timeline_path=args.timeline,
        route=args.route,
        report_paths=args.reports,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
