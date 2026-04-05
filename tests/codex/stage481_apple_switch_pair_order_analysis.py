#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE480_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / "stage480_apple_switch_exact_core_scan_20260403"
    / "summary.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / f"stage481_apple_switch_pair_order_analysis_{time.strftime('%Y%m%d')}"
)

MODEL_ORDER = ["qwen3", "deepseek7b"]


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_effect_map(subset_rows: Sequence[Dict[str, object]]) -> Dict[Tuple[str, ...], Dict[str, float]]:
    out: Dict[Tuple[str, ...], Dict[str, float]] = {
        tuple(): {
            "search_drop": 0.0,
            "heldout_drop": 0.0,
            "control_abs_shift": 0.0,
            "utility": 0.0,
        }
    }
    for row in subset_rows:
        out[tuple(sorted(row["subset_ids"]))] = row["effect"]
    return out


def pair_synergy(effect_map: Dict[Tuple[str, ...], Dict[str, float]], a: str, b: str, *, metric_key: str) -> float:
    pair_key = tuple(sorted((a, b)))
    pair_value = float(effect_map[pair_key][metric_key])
    a_value = float(effect_map[(a,)][metric_key])
    b_value = float(effect_map[(b,)][metric_key])
    return pair_value - a_value - b_value


def permutation_trace(effect_map: Dict[Tuple[str, ...], Dict[str, float]], order: Sequence[str], *, metric_key: str) -> Dict[str, object]:
    chosen: List[str] = []
    prefix_rows = []
    total_prefix_metric = 0.0
    for cid in order:
        before_key = tuple(sorted(chosen))
        chosen.append(cid)
        after_key = tuple(sorted(chosen))
        before_val = float(effect_map[before_key][metric_key])
        after_val = float(effect_map[after_key][metric_key])
        gain = after_val - before_val
        prefix_rows.append(
            {
                "added_candidate": cid,
                "prefix_ids": list(chosen),
                "prefix_metric": after_val,
                "marginal_gain": gain,
            }
        )
        total_prefix_metric += after_val
    return {
        "order": list(order),
        "metric_key": metric_key,
        "prefix_rows": prefix_rows,
        "total_prefix_metric": total_prefix_metric,
        "final_metric": float(effect_map[tuple(sorted(order))][metric_key]),
    }


def summarize_average_marginals(permutation_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_candidate: Dict[str, Dict[str, float]] = {}
    for row in permutation_rows:
        for pos, prefix_row in enumerate(row["prefix_rows"], start=1):
            cid = str(prefix_row["added_candidate"])
            slot = by_candidate.setdefault(
                cid,
                {
                    "sum_gain": 0.0,
                    "count": 0.0,
                    "sum_position": 0.0,
                    "first_gain_sum": 0.0,
                    "first_count": 0.0,
                    "late_gain_sum": 0.0,
                    "late_count": 0.0,
                },
            )
            gain = float(prefix_row["marginal_gain"])
            slot["sum_gain"] += gain
            slot["count"] += 1.0
            slot["sum_position"] += float(pos)
            if pos == 1:
                slot["first_gain_sum"] += gain
                slot["first_count"] += 1.0
            else:
                slot["late_gain_sum"] += gain
                slot["late_count"] += 1.0
    out = []
    for cid, slot in by_candidate.items():
        out.append(
            {
                "candidate_id": cid,
                "avg_marginal_gain": slot["sum_gain"] / max(1.0, slot["count"]),
                "avg_position": slot["sum_position"] / max(1.0, slot["count"]),
                "avg_first_gain": slot["first_gain_sum"] / max(1.0, slot["first_count"]),
                "avg_late_gain": slot["late_gain_sum"] / max(1.0, slot["late_count"]),
            }
        )
    out.sort(key=lambda row: row["avg_marginal_gain"], reverse=True)
    return {"rows": out}


def classify_roles(model_key: str, model_row: Dict[str, object]) -> Dict[str, object]:
    utility50 = model_row["min_subsets"]["utility"]["50pct"]
    utility70 = model_row["min_subsets"]["utility"]["70pct"]
    utility90 = model_row["min_subsets"]["utility"]["90pct"]
    heldout90 = model_row["min_subsets"]["heldout_drop"]["90pct"]
    best_utility = model_row["best_subset_by_utility"]

    utility50_ids = set(utility50["subset_ids"]) if utility50 else set()
    utility70_ids = set(utility70["subset_ids"]) if utility70 else set()
    utility90_ids = set(utility90["subset_ids"]) if utility90 else set()
    heldout90_ids = set(heldout90["subset_ids"]) if heldout90 else set()
    best_ids = set(best_utility["subset_ids"]) if best_utility else set()

    if model_key == "qwen3":
        return {
            "skeleton": sorted(utility70_ids),
            "bridge_to_90pct_utility": sorted(utility90_ids - utility70_ids),
            "heldout_boosters": sorted(heldout90_ids - utility90_ids),
            "max_utility_boosters": sorted(best_ids - utility90_ids),
        }

    return {
        "anchor": sorted(utility50_ids),
        "main_boosters": sorted(utility70_ids - utility50_ids),
        "heldout_boosters": sorted(heldout90_ids - utility70_ids),
        "max_utility_boosters": sorted(best_ids - utility70_ids),
    }


def analyze_focus_set(model_key: str, effect_map: Dict[Tuple[str, ...], Dict[str, float]], focus_ids: Sequence[str], *, metric_key: str) -> Dict[str, object]:
    orders = []
    for order in itertools.permutations(focus_ids):
        orders.append(permutation_trace(effect_map, order, metric_key=metric_key))
    orders.sort(key=lambda row: (row["total_prefix_metric"], row["final_metric"]), reverse=True)

    pair_rows = []
    for a, b in itertools.combinations(focus_ids, 2):
        pair_rows.append(
            {
                "pair": [a, b],
                "utility_pair_value": float(effect_map[tuple(sorted((a, b)))]["utility"]),
                "utility_synergy": pair_synergy(effect_map, a, b, metric_key="utility"),
                "heldout_pair_value": float(effect_map[tuple(sorted((a, b)))]["heldout_drop"]),
                "heldout_synergy": pair_synergy(effect_map, a, b, metric_key="heldout_drop"),
            }
        )
    synergy_key = "utility_synergy" if metric_key == "utility" else "heldout_synergy"
    pair_rows.sort(key=lambda row: row[synergy_key], reverse=True)

    return {
        "focus_ids": list(focus_ids),
        "metric_key": metric_key,
        "best_order": orders[0] if orders else None,
        "all_orders": orders,
        "average_marginals": summarize_average_marginals(orders),
        "pair_rows": pair_rows,
    }


def analyze_model(model_key: str, model_row: Dict[str, object]) -> Dict[str, object]:
    effect_map = build_effect_map(model_row["subset_rows"])
    roles = classify_roles(model_key, model_row)

    if model_key == "qwen3":
        utility_focus_ids = list(model_row["min_subsets"]["utility"]["90pct"]["subset_ids"])
        heldout_focus_ids = list(model_row["min_subsets"]["heldout_drop"]["90pct"]["subset_ids"])
    else:
        utility_focus_ids = list(model_row["min_subsets"]["utility"]["70pct"]["subset_ids"])
        heldout_focus_ids = list(model_row["min_subsets"]["heldout_drop"]["90pct"]["subset_ids"])

    return {
        "model_key": model_key,
        "model_name": model_row["model_name"],
        "roles": roles,
        "utility_focus": analyze_focus_set(model_key, effect_map, utility_focus_ids, metric_key="utility"),
        "heldout_focus": analyze_focus_set(model_key, effect_map, heldout_focus_ids, metric_key="heldout_drop"),
    }


def build_cross_model_summary(model_rows: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    out = {}
    for model_key, row in model_rows.items():
        out[model_key] = {
            "role_summary": row["roles"],
            "utility_best_order": row["utility_focus"]["best_order"]["order"] if row["utility_focus"]["best_order"] else [],
            "heldout_best_order": row["heldout_focus"]["best_order"]["order"] if row["heldout_focus"]["best_order"] else [],
        }
    return out


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 实验设置",
        f"- 时间戳: {summary['timestamp_utc']}",
        "- 输入来源: stage480 的精确子集穷举结果",
        "- 目标: 从已有全子集数据中抽取骨架、增强器、配对协同和最佳加入顺序",
        "",
    ]
    for model_key in MODEL_ORDER:
        row = summary["models"][model_key]
        lines.extend(
            [
                f"## 模型 {model_key}",
                f"- 角色划分: {row['roles']}",
                f"- utility 焦点集合: {row['utility_focus']['focus_ids']}",
                f"- utility 最优顺序: {row['utility_focus']['best_order']['order'] if row['utility_focus']['best_order'] else []}",
                f"- heldout 焦点集合: {row['heldout_focus']['focus_ids']}",
                f"- heldout 最优顺序: {row['heldout_focus']['best_order']['order'] if row['heldout_focus']['best_order'] else []}",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="苹果切换配对与顺序分析")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage480_summary = load_json(STAGE480_SUMMARY_PATH)
    start_time = time.time()
    model_rows = {}
    for model_key in MODEL_ORDER:
        model_rows[model_key] = analyze_model(model_key, stage480_summary["models"][model_key])
    elapsed = time.time() - start_time

    summary = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage481_apple_switch_pair_order_analysis",
        "title": "苹果切换配对与顺序分析",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "models": model_rows,
        "cross_model_summary": build_cross_model_summary(model_rows),
    }
    output_dir = Path(args.output_dir)
    write_outputs(summary, output_dir)
    print(
        json.dumps(
            {
                "status_short": "stage481_ready",
                "output_dir": str(output_dir),
                "elapsed_seconds": elapsed,
                "qwen3_utility_best_order": model_rows["qwen3"]["utility_focus"]["best_order"]["order"],
                "deepseek7b_utility_best_order": model_rows["deepseek7b"]["utility_focus"]["best_order"]["order"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
