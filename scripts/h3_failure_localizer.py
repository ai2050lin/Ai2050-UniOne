import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.runtime.contracts import AnalysisSpec, ConclusionCard, Metric, RunRecord, RunSummary
from server.runtime.experiment_store import ExperimentTimelineStore


def _cfg_key(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg or {}, sort_keys=True, ensure_ascii=False)


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def aggregate_reports(paths: List[Path]) -> Dict[str, Any]:
    model_cat = defaultdict(lambda: defaultdict(lambda: {"n": 0, "support": 0, "falsify": 0, "open": 0, "uplift_sum": 0.0, "win_sum": 0.0}))
    model_cfg_cat = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"n": 0, "support": 0, "falsify": 0, "open": 0, "uplift_sum": 0.0})))
    run_meta: List[Dict[str, Any]] = []

    for p in paths:
        payload = json.loads(p.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        run_meta.append(
            {
                "path": str(p).replace("\\", "/"),
                "seed": cfg.get("seed"),
                "task_profile": cfg.get("task_profile", "standard"),
                "lock_mode": cfg.get("lock_mode", "per_category"),
                "max_per_category": cfg.get("max_per_category"),
                "aggregates": payload.get("aggregates", {}),
            }
        )
        for run in payload.get("runs", []):
            model_name = run.get("model", "unknown_model")
            categories = run.get("categories", {}) or {}
            for category, obj in categories.items():
                best = (obj or {}).get("best", {}) or {}
                verdict = best.get("verdict", "open")
                uplift = _to_float(best.get("uplift_logprob"), 0.0)
                win = _to_float(best.get("win_rate"), 0.0)
                cfg_item = (obj or {}).get("selected_config") or best.get("config") or {}
                ckey = _cfg_key(cfg_item)

                bucket = model_cat[model_name][category]
                bucket["n"] += 1
                bucket["uplift_sum"] += uplift
                bucket["win_sum"] += win
                if verdict == "support":
                    bucket["support"] += 1
                elif verdict == "falsify":
                    bucket["falsify"] += 1
                else:
                    bucket["open"] += 1

                cfg_bucket = model_cfg_cat[model_name][ckey][category]
                cfg_bucket["n"] += 1
                cfg_bucket["uplift_sum"] += uplift
                if verdict == "support":
                    cfg_bucket["support"] += 1
                elif verdict == "falsify":
                    cfg_bucket["falsify"] += 1
                else:
                    cfg_bucket["open"] += 1

    model_category_summary: Dict[str, Dict[str, Any]] = {}
    strict_failure_clusters: List[Dict[str, Any]] = []
    for model_name, cat_map in model_cat.items():
        model_category_summary[model_name] = {}
        for category, stat in cat_map.items():
            n = max(1, int(stat["n"]))
            support_rate = stat["support"] / n
            falsify_rate = stat["falsify"] / n
            open_rate = stat["open"] / n
            avg_uplift = stat["uplift_sum"] / n
            avg_win = stat["win_sum"] / n
            row = {
                "n": n,
                "support_count": stat["support"],
                "falsify_count": stat["falsify"],
                "open_count": stat["open"],
                "support_rate": round(support_rate, 4),
                "falsify_rate": round(falsify_rate, 4),
                "open_rate": round(open_rate, 4),
                "avg_uplift_logprob": round(avg_uplift, 8),
                "avg_win_rate": round(avg_win, 4),
            }
            model_category_summary[model_name][category] = row
            if n >= 2 and falsify_rate >= 0.5:
                strict_failure_clusters.append(
                    {
                        "model": model_name,
                        "category": category,
                        **row,
                    }
                )

    config_risk_summary: Dict[str, List[Dict[str, Any]]] = {}
    for model_name, cfg_map in model_cfg_cat.items():
        rows: List[Dict[str, Any]] = []
        for cfg_key, cat_map in cfg_map.items():
            for category, stat in cat_map.items():
                n = max(1, int(stat["n"]))
                rows.append(
                    {
                        "config": json.loads(cfg_key),
                        "category": category,
                        "n": n,
                        "falsify_rate": round(stat["falsify"] / n, 4),
                        "support_rate": round(stat["support"] / n, 4),
                        "open_rate": round(stat["open"] / n, 4),
                        "avg_uplift_logprob": round(stat["uplift_sum"] / n, 8),
                    }
                )
        rows.sort(key=lambda x: (x["falsify_rate"], -x["avg_uplift_logprob"]), reverse=True)
        config_risk_summary[model_name] = rows

    strict_failure_clusters.sort(key=lambda x: (x["falsify_rate"], -x["avg_uplift_logprob"]), reverse=True)
    fragile = len(strict_failure_clusters) > 0
    status = "fragile_single_config" if fragile else "open"

    findings: List[str] = []
    for item in strict_failure_clusters[:6]:
        findings.append(
            f"{item['model']}::{item['category']} falsify_rate={item['falsify_rate']} avg_uplift={item['avg_uplift_logprob']}"
        )
    if not findings:
        findings.append("No persistent falsify cluster detected in current report set.")

    recommendations = [
        "For promotion gate, keep per_model_single as mandatory stress test.",
        "Prioritize categories with persistent falsify clusters for subspace redesign.",
        "Separate category-specialized config from universal config claims.",
        "Promote H3 only if strict single-config holdout clears falsify_rate constraints.",
    ]

    return {
        "schema_version": "1.0",
        "test_date": datetime.now(timezone.utc).date().isoformat(),
        "analysis_type": "h3_failure_localization",
        "inputs": [str(p).replace("\\", "/") for p in paths],
        "runs": run_meta,
        "status": status,
        "strict_failure_clusters": strict_failure_clusters,
        "model_category_summary": model_category_summary,
        "config_risk_summary": config_risk_summary,
        "key_findings": findings,
        "recommendations": recommendations,
        "conclusion": (
            "Strict single-config transfer shows persistent failure clusters; treat H3 as fragile under universal-config constraints."
            if fragile
            else "No persistent failure cluster found under current strict setting."
        ),
    }


def append_timeline(result: Dict[str, Any], summary_path: Path, timeline_path: Path) -> None:
    store = ExperimentTimelineStore(path=str(timeline_path))
    now = time.time()
    strict_count = float(len(result.get("strict_failure_clusters", [])))
    status = result.get("status", "open")
    record = RunRecord(
        run_id=f"run_h3_failure_localization_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        spec=AnalysisSpec(
            route="fiber_bundle",
            analysis_type="h3_failure_localization",
            model="multi_model",
            params={"source_report": str(summary_path).replace("\\", "/")},
            input_payload={},
        ),
        status="completed",
        created_at=now,
        updated_at=now + 0.001,
        summary=RunSummary(
            metrics=[
                Metric(key="strict_failure_cluster_count", value=strict_count, min_value=0.0),
                Metric(key="fragility_flag", value=1.0 if status == "fragile_single_config" else 0.0, min_value=0.0, max_value=1.0),
            ],
            conclusion=ConclusionCard(
                objective="Localize strict-transfer failure clusters for H3.",
                method="Aggregate model/category/config verdicts from strict holdout reports.",
                evidence=result.get("key_findings", [])[:4],
                result=result.get("conclusion", ""),
                confidence=0.74,
                limitations=["Localization relies on available strict runs and seeds."],
                next_action="Use localized failure clusters to redesign robust shared subspaces.",
            ),
            artifacts=[{"path": str(summary_path).replace("\\", "/")}],
        ),
        event_count=0,
    )
    store.append_run(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Localize strict-transfer failure clusters for H3.")
    parser.add_argument("--inputs", required=True, help="Comma-separated JSON report paths.")
    parser.add_argument("--timeline", default="tempdata/agi_route_test_timeline.json")
    parser.add_argument("--output", default="tempdata/h3_failure_localization_20260221.json")
    args = parser.parse_args()

    paths = [Path(x.strip()) for x in args.inputs.split(",") if x.strip()]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input reports: {missing}")

    result = aggregate_reports(paths)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_timeline(result, summary_path=out, timeline_path=Path(args.timeline))
    print(
        json.dumps(
            {
                "output": str(out),
                "status": result.get("status"),
                "strict_failure_cluster_count": len(result.get("strict_failure_clusters", [])),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
