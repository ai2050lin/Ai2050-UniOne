#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "codex_temp"
    / f"stage488_bridge_minimal_causal_circuit_protocol_{time.strftime('%Y%m%d')}"
)

STAGE439_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage439_binding_bridge_causal_ablation_20260402" / "summary.json"
STAGE440_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage440_attribute_graph_generalization_20260402" / "summary.json"
STAGE442_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage442_binding_mixed_subcircuit_search_20260402" / "summary.json"
STAGE443_SUMMARY_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage443_binding_family_split_probe_20260402" / "summary.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def extract_family_means(stage440: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    family_rows: Dict[str, List[Dict[str, float]]] = {}
    for model_row in stage440["model_results"]:
        for row in model_row["binding_rows"]:
            family_rows.setdefault(row["family_name"], []).append(row)
    out: Dict[str, Dict[str, float]] = {}
    for family_name, rows in family_rows.items():
        out[family_name] = {
            "count": len(rows),
            "mean_union_coverage": mean([float(row["union_coverage"]) for row in rows]),
            "mean_bridge_only_ratio": mean([float(row["bridge_only_ratio"]) for row in rows]),
            "law_support_rate": mean([1.0 if row["law_support"] else 0.0 for row in rows]),
        }
    return out


def extract_stage443_family(stage443: Dict[str, object], family_name: str) -> Dict[str, object]:
    for row in stage443["family_results"]:
        if row["family_name"] == family_name:
            return row
    raise KeyError(f"Missing family_name={family_name}")


def stage443_family_payload(row: Dict[str, object]) -> Dict[str, object]:
    if not row["ok"]:
        return {"ok": False, "error": row["error"], "mixed_support": False}
    result = row["result"]
    search_state = result.get("search_state", {})
    final_ids = search_state.get("pruned_subset_ids") or search_state.get("greedy_subset_ids") or []
    final_result = search_state.get("pruned_result") or search_state.get("greedy_result") or {}
    return {
        "ok": True,
        "mixed_support": bool(result.get("mixed_support", False)),
        "best_single_candidate": result.get("best_single_candidate"),
        "final_subset_ids": final_ids,
        "final_effect": final_result.get("effect", {}),
        "candidate_count": int(result.get("candidate_count", 0)),
        "shortlist_size": int(result.get("shortlist_size", 0)),
    }


def build_summary() -> Dict[str, object]:
    stage439 = load_json(STAGE439_SUMMARY_PATH)
    stage440 = load_json(STAGE440_SUMMARY_PATH)
    stage442 = load_json(STAGE442_SUMMARY_PATH)
    stage443 = load_json(STAGE443_SUMMARY_PATH)

    family_means = extract_family_means(stage440)
    color_payload = stage443_family_payload(extract_stage443_family(stage443, "color"))
    taste_payload = stage443_family_payload(extract_stage443_family(stage443, "taste"))
    size_payload = stage443_family_payload(extract_stage443_family(stage443, "size"))

    bridge_status = {
        "color": {"structure": family_means.get("color", {}), "family_probe": color_payload},
        "taste": {"structure": family_means.get("taste", {}), "family_probe": taste_payload},
        "size": {"structure": family_means.get("size", {}), "family_probe": size_payload},
    }

    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage488_bridge_minimal_causal_circuit_protocol",
        "title": "桥接项最小因果回路正式协议",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bridge_status": bridge_status,
        "aggregate": {
            "structural_law_support_rate": float(stage440["cross_model_summary"]["attribute_graph_support_rate"]),
            "mean_union_coverage": float(stage440["cross_model_summary"]["mean_union_coverage"]),
            "mean_bridge_only_ratio": float(stage440["cross_model_summary"]["mean_bridge_only_ratio"]),
            "pure_bridge_causal_support_rate": float(stage439["cross_model_summary"]["binding_causal_support_rate"]),
            "mixed_search_initial_support_rate": float(stage442["cross_model_summary"]["mixed_support_rate"]),
            "family_mixed_support_count": sum(1 for payload in [color_payload, taste_payload, size_payload] if bool(payload.get("mixed_support", False))),
            "family_probe_ok_count": sum(1 for payload in [color_payload, taste_payload, size_payload] if payload["ok"]),
            "core_answer": "桥接项目前最稳的图景是：结构规律已经非常强，但纯桥接神经元还没有打出稳定强因果；真正开始露出最小因果回路迹象的是 size（大小）家族，而且它表现成混合回路，不是纯神经元集合。",
        },
        "sources": {
            "stage439": str(STAGE439_SUMMARY_PATH),
            "stage440": str(STAGE440_SUMMARY_PATH),
            "stage442": str(STAGE442_SUMMARY_PATH),
            "stage443": str(STAGE443_SUMMARY_PATH),
        },
    }


def build_report(summary: Dict[str, object]) -> str:
    agg = summary["aggregate"]
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "## 汇总",
        f"- structural_law_support_rate = {agg['structural_law_support_rate']:.4f}",
        f"- mean_union_coverage = {agg['mean_union_coverage']:.4f}",
        f"- mean_bridge_only_ratio = {agg['mean_bridge_only_ratio']:.4f}",
        f"- pure_bridge_causal_support_rate = {agg['pure_bridge_causal_support_rate']:.4f}",
        f"- mixed_search_initial_support_rate = {agg['mixed_search_initial_support_rate']:.4f}",
        f"- family_mixed_support_count = {agg['family_mixed_support_count']}",
        f"- core_answer = {agg['core_answer']}",
        "",
    ]
    for family_name in ["color", "taste", "size"]:
        row = summary["bridge_status"][family_name]
        struct_row = row["structure"]
        probe_row = row["family_probe"]
        lines.extend(
            [
                f"## 家族 {family_name}",
                f"- structure_mean_union_coverage = {struct_row.get('mean_union_coverage', 0.0):.4f}",
                f"- structure_mean_bridge_only_ratio = {struct_row.get('mean_bridge_only_ratio', 0.0):.4f}",
                f"- structure_law_support_rate = {struct_row.get('law_support_rate', 0.0):.4f}",
                f"- probe_ok = {probe_row['ok']}",
                f"- mixed_support = {probe_row['mixed_support']}",
            ]
        )
        if not probe_row["ok"]:
            lines.append(f"- error = {probe_row['error']}")
        else:
            best = probe_row["best_single_candidate"]
            lines.append(f"- best_single_candidate = {best['candidate_id']} (utility={best['effect']['utility']:.4f}, binding_drop={best['effect']['binding_drop']:.4f})")
            lines.append(f"- final_subset_ids = {', '.join(probe_row['final_subset_ids']) if probe_row['final_subset_ids'] else '(empty)'}")
            if probe_row["final_effect"]:
                lines.append(
                    f"- final_effect = utility={probe_row['final_effect'].get('utility', 0.0):.4f}, "
                    f"binding_drop={probe_row['final_effect'].get('binding_drop', 0.0):.4f}, "
                    f"heldout_binding_drop={probe_row['final_effect'].get('heldout_binding_drop', 0.0):.4f}"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(summary: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="桥接项最小因果回路正式协议")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary()
    write_outputs(summary, Path(args.output_dir))
    print(json.dumps({"status_short": "stage488_ready", "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
