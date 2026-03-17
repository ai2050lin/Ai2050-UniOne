from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def aggregate_category_rows(rows: Sequence[Dict[str, object]], category: str) -> Dict[str, object]:
    target_rows = [row for row in rows if str(row.get("category")) == category]
    if not target_rows:
        return {
            "category": category,
            "row_count": 0,
            "pair_count": 0,
            "strict_positive_pair_count": 0,
            "strict_positive_pair_ratio": 0.0,
            "mean_union_joint_adv": 0.0,
            "mean_union_synergy_joint": 0.0,
        }
    pair_count = sum(int(row.get("pair_count", 0)) for row in target_rows)
    strict_positive_pair_count = sum(int(row.get("strict_positive_pair_count", 0)) for row in target_rows)
    mean_union_joint_adv = sum(safe_float(row.get("mean_union_joint_adv", 0.0)) for row in target_rows) / len(target_rows)
    mean_union_synergy_joint = sum(safe_float(row.get("mean_union_synergy_joint", 0.0)) for row in target_rows) / len(target_rows)
    return {
        "category": category,
        "row_count": len(target_rows),
        "pair_count": pair_count,
        "strict_positive_pair_count": strict_positive_pair_count,
        "strict_positive_pair_ratio": (strict_positive_pair_count / pair_count) if pair_count else 0.0,
        "mean_union_joint_adv": mean_union_joint_adv,
        "mean_union_synergy_joint": mean_union_synergy_joint,
    }


def analyze_model_gap(
    stage3_summary: Dict[str, object],
    stage5_prototype_summary: Dict[str, object],
    stage5_instance_summary: Dict[str, object],
    stage6_summary: Dict[str, object],
    category: str,
) -> Dict[str, object]:
    selected_categories = [str(x) for x in stage3_summary.get("selected_categories", [])]
    return {
        "selected_in_stage3": category in selected_categories,
        "selected_categories": selected_categories,
        "prototype_strict_joint_adv": safe_float(stage5_prototype_summary.get("mean_candidate_full_strict_joint_adv", 0.0)),
        "instance_strict_joint_adv": safe_float(stage5_instance_summary.get("mean_candidate_full_strict_joint_adv", 0.0)),
        "union_joint_adv": safe_float(stage6_summary.get("mean_union_joint_adv", 0.0)),
        "union_synergy_joint": safe_float(stage6_summary.get("mean_union_synergy_joint", 0.0)),
        "strict_positive_synergy_pair_count": int(stage6_summary.get("strict_positive_synergy_pair_count", 0)),
    }


def classify_failure_modes(
    fruit_discovery: Dict[str, object],
    apple_multiaxis: Dict[str, object],
    qwen_gap: Dict[str, object],
    deepseek_gap: Dict[str, object],
) -> List[str]:
    modes: List[str] = []
    if not qwen_gap["selected_in_stage3"] and not deepseek_gap["selected_in_stage3"]:
        modes.append("fruit 在严格真实类别词管线里首先卡在 stage3 选择层，说明早期聚焦机制就没有稳定抓住它。")
    if safe_float(fruit_discovery["strict_positive_pair_ratio"]) > 0.0 and safe_float(fruit_discovery["mean_union_synergy_joint"]) < 0.0:
        modes.append("fruit 在发现式轨道里并非完全不存在，但联合协同平均值仍为负，属于“有局部信号、无稳定协同”。")

    concept_axis = apple_multiaxis.get("concept_axis", {})
    if (
        safe_float(concept_axis.get("shared_base_ratio_mean", 0.0)) > 0.0
        and safe_float(concept_axis.get("meso_to_macro_jaccard_mean", 0.0)) > safe_float(concept_axis.get("micro_to_meso_jaccard_mean", 0.0))
    ):
        modes.append("apple 侧已经显示出 fruit 家族锚点和宏观桥接，但属性到实体的压缩仍弱，说明问题不在“完全没有家族结构”，而在“家族结构还不够硬”。")
    return modes


def build_payload(
    fruit_discovery: Dict[str, object],
    apple_multiaxis: Dict[str, object],
    qwen_gap: Dict[str, object],
    deepseek_gap: Dict[str, object],
) -> Dict[str, object]:
    failure_modes = classify_failure_modes(fruit_discovery, apple_multiaxis, qwen_gap, deepseek_gap)
    return {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "stage56_fruit_family_gap_report_v1",
        "title": "fruit 家族严格闭合缺口报告",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fruit_discovery_summary": fruit_discovery,
        "apple_multiaxis_summary": apple_multiaxis,
        "strict_pipeline_gap": {
            "qwen3_4b": qwen_gap,
            "deepseek_7b": deepseek_gap,
        },
        "failure_modes": failure_modes,
        "next_actions": [
            "把 fruit 直接放入严格真实类别词主战场，不再让它停留在发现式轨道边缘。",
            "优先用 fruit / food / nature / object 的对照块检查 fruit 的早期聚焦失败是否来自类别词本体弱。",
            "如果 strict 轨仍失败，再比较真实类别词 fruit 与 proxy 原型之间的性能差距，定位原型词缺口。",
        ],
    }


def build_markdown(payload: Dict[str, object]) -> str:
    fruit = payload["fruit_discovery_summary"]
    qwen = payload["strict_pipeline_gap"]["qwen3_4b"]
    deepseek = payload["strict_pipeline_gap"]["deepseek_7b"]
    concept = payload["apple_multiaxis_summary"]["concept_axis"]
    lines = [
        "# fruit 家族严格闭合缺口报告",
        "",
        "## 发现式轨道",
        f"- strict_positive_pair_ratio: {safe_float(fruit['strict_positive_pair_ratio']):.6f}",
        f"- mean_union_joint_adv: {safe_float(fruit['mean_union_joint_adv']):.6f}",
        f"- mean_union_synergy_joint: {safe_float(fruit['mean_union_synergy_joint']):.6f}",
        "",
        "## 严格真实类别词轨道",
        f"- qwen3_4b selected_in_stage3: {qwen['selected_in_stage3']}",
        f"- deepseek_7b selected_in_stage3: {deepseek['selected_in_stage3']}",
        f"- qwen3_4b stage6 strict_positive_synergy_pair_count: {qwen['strict_positive_synergy_pair_count']}",
        f"- deepseek_7b stage6 strict_positive_synergy_pair_count: {deepseek['strict_positive_synergy_pair_count']}",
        "",
        "## apple 家族探针",
        f"- micro_to_meso_jaccard_mean: {safe_float(concept['micro_to_meso_jaccard_mean']):.6f}",
        f"- meso_to_macro_jaccard_mean: {safe_float(concept['meso_to_macro_jaccard_mean']):.6f}",
        f"- shared_base_ratio_mean: {safe_float(concept['shared_base_ratio_mean']):.6f}",
        "",
        "## 缺口判断",
    ]
    lines.extend([f"- {line}" for line in payload["failure_modes"]])
    lines.extend(["", "## 下一步"])
    lines.extend([f"- {line}" for line in payload["next_actions"]])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fruit family closure gap report")
    parser.add_argument("--discovery-per-category-jsonl", default="tempdata/stage56_large_scale_discovery_multimodel_20260317_2105/discovery_per_category.jsonl")
    parser.add_argument("--apple-multiaxis-json", default="tests/codex_temp/stage56_multiaxis_language_analyzer_20260317_apple/multiaxis_language_analysis.json")
    parser.add_argument("--qwen-stage3-summary", default="tempdata/stage56_real_category_closure_block_strict_20260317_1922/qwen3_4b/stage3_causal_closure/summary.json")
    parser.add_argument("--qwen-stage5-prototype-summary", default="tempdata/stage56_real_category_closure_block_strict_20260317_1922/qwen3_4b/stage5_prototype/summary.json")
    parser.add_argument("--qwen-stage5-instance-summary", default="tempdata/stage56_real_category_closure_block_strict_20260317_1922/qwen3_4b/stage5_instance/summary.json")
    parser.add_argument("--qwen-stage6-summary", default="tempdata/stage56_real_category_closure_block_strict_20260317_1922/qwen3_4b/stage6_prototype_instance_decomposition/summary.json")
    parser.add_argument("--deepseek-stage3-summary", default="tempdata/stage56_real_category_closure_block_strict_20260317_1922/deepseek_7b/stage3_causal_closure/summary.json")
    parser.add_argument("--deepseek-stage5-prototype-summary", default="tempdata/stage56_real_category_closure_block_strict_20260317_1922/deepseek_7b/stage5_prototype/summary.json")
    parser.add_argument("--deepseek-stage5-instance-summary", default="tempdata/stage56_real_category_closure_block_strict_20260317_1922/deepseek_7b/stage5_instance/summary.json")
    parser.add_argument("--deepseek-stage6-summary", default="tempdata/stage56_real_category_closure_block_strict_20260317_1922/deepseek_7b/stage6_prototype_instance_decomposition/summary.json")
    parser.add_argument("--output-dir", default="tests/codex_temp/stage56_fruit_family_gap_report_20260317")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discovery_rows = read_jsonl(Path(args.discovery_per_category_jsonl))
    apple_multiaxis = read_json(Path(args.apple_multiaxis_json))
    fruit_discovery = aggregate_category_rows(discovery_rows, "fruit")
    qwen_gap = analyze_model_gap(
        read_json(Path(args.qwen_stage3_summary)),
        read_json(Path(args.qwen_stage5_prototype_summary)),
        read_json(Path(args.qwen_stage5_instance_summary)),
        read_json(Path(args.qwen_stage6_summary)),
        "fruit",
    )
    deepseek_gap = analyze_model_gap(
        read_json(Path(args.deepseek_stage3_summary)),
        read_json(Path(args.deepseek_stage5_prototype_summary)),
        read_json(Path(args.deepseek_stage5_instance_summary)),
        read_json(Path(args.deepseek_stage6_summary)),
        "fruit",
    )
    payload = build_payload(fruit_discovery, apple_multiaxis, qwen_gap, deepseek_gap)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fruit_family_gap_report.json"
    report_path = out_dir / "FRUIT_FAMILY_GAP_REPORT.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "json": str(json_path),
                "report": str(report_path),
                "failure_modes": payload["failure_modes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
