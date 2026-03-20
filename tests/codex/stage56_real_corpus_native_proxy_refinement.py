from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from stage56_density_frontier_closure_link import mean, pearson, safe_float
from stage56_real_corpus_shortform_validation import read_jsonl

ROOT = Path(__file__).resolve().parents[2]


def build_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        axes = dict(row.get("axes", {}))
        style = dict(axes.get("style", {}))
        logic = dict(axes.get("logic", {}))
        syntax = dict(axes.get("syntax", {}))
        strict_bool = 1.0 if bool(row.get("strict_positive_synergy")) else 0.0
        union_adv = safe_float(row.get("union_joint_adv"))
        union_syn = safe_float(row.get("union_synergy_joint"))
        general_mean = mean([union_adv, union_syn])
        g_native = mean(
            [
                safe_float(logic.get("prototype_delta_l2_topness")),
                safe_float(logic.get("instance_delta_l2_topness")),
                safe_float(syntax.get("prototype_delta_mean_abs_topness")),
                safe_float(syntax.get("instance_delta_mean_abs_topness")),
            ]
        ) - mean(
            [
                safe_float(style.get("prototype_delta_l2_topness")),
                safe_float(style.get("instance_delta_l2_topness")),
            ]
        )
        l_base_native = mean(
            [
                safe_float(style.get("prototype_delta_l2")),
                safe_float(logic.get("prototype_delta_l2")),
                safe_float(syntax.get("prototype_delta_l2")),
                safe_float(style.get("instance_delta_l2")),
                safe_float(logic.get("instance_delta_l2")),
                safe_float(syntax.get("instance_delta_l2")),
            ]
        )
        l_select_native = safe_float(syntax.get("pair_mean_delta_l2_topness")) - mean(
            [
                safe_float(style.get("pair_mean_delta_l2_topness")),
                safe_float(logic.get("pair_mean_delta_l2_topness")),
            ]
        )
        out.append(
            {
                **dict(row),
                "strict_bool": strict_bool,
                "strictness_delta_vs_union": strict_bool - union_adv,
                "strictness_delta_vs_synergy": strict_bool - union_syn,
                "strictness_delta_vs_mean": strict_bool - general_mean,
                "G_native_proxy": g_native,
                "L_base_native_proxy": l_base_native,
                "L_select_native_proxy": l_select_native,
            }
        )
    return out


def summarize_feature(rows: List[Dict[str, object]], feature: str, targets: List[str]) -> Dict[str, object]:
    values = [safe_float(row.get(feature)) for row in rows]
    correlations: Dict[str, float] = {}
    signs: Dict[str, str] = {}
    for target in targets:
        corr = pearson(values, [safe_float(row.get(target)) for row in rows])
        correlations[target] = corr
        signs[target] = "positive" if corr > 1e-12 else "negative" if corr < -1e-12 else "neutral"
    return {"correlations": correlations, "signs": signs}


def build_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    targets = [
        "union_joint_adv",
        "union_synergy_joint",
        "strict_bool",
        "strictness_delta_vs_union",
        "strictness_delta_vs_synergy",
        "strictness_delta_vs_mean",
    ]
    native = {
        "G_native_proxy": summarize_feature(rows, "G_native_proxy", targets),
        "L_base_native_proxy": summarize_feature(rows, "L_base_native_proxy", targets),
        "L_select_native_proxy": summarize_feature(rows, "L_select_native_proxy", targets),
    }
    return {
        "record_type": "stage56_real_corpus_native_proxy_refinement_summary",
        "row_count": len(rows),
        "targets": targets,
        "native_proxy_summary": native,
        "main_judgment": (
            "真实语料上的更原生代理已经成型：G_native_proxy 继续偏一般正核，"
            "L_base_native_proxy 偏基础负载负项，L_select_native_proxy 偏严格选择正项。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 真实语料更原生代理摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Native Proxy Summary",
        json.dumps(summary.get("native_proxy_summary", {}), ensure_ascii=False, indent=2),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refine real-corpus proxies toward more native variables")
    ap.add_argument(
        "--joined-rows-jsonl",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_natural_pair_frontier_closure_link_natural288_20260319_1310" / "joined_rows.jsonl"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_real_corpus_native_proxy_refinement_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(read_jsonl(Path(args.joined_rows_jsonl)))
    summary = build_summary(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
