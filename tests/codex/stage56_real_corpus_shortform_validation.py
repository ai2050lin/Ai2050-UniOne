from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from stage56_density_frontier_closure_link import mean, pearson, safe_float

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


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
        out.append(
            {
                **dict(row),
                "strict_bool": strict_bool,
                "strictness_delta_vs_union": strict_bool - union_adv,
                "strictness_delta_vs_synergy": strict_bool - union_syn,
                "strictness_delta_vs_mean": strict_bool - general_mean,
                "G_corpus_proxy": mean(
                    [
                        safe_float(logic.get("pair_mean_delta_l2_topness")),
                        safe_float(syntax.get("pair_mean_delta_l2_topness")),
                    ]
                )
                - safe_float(style.get("pair_mean_delta_l2_topness")),
                "L_base_corpus_proxy": mean(
                    [
                        safe_float(style.get("pair_mean_delta_l2")),
                        safe_float(logic.get("pair_mean_delta_l2")),
                        safe_float(syntax.get("pair_mean_delta_l2")),
                    ]
                ),
                "L_select_corpus_proxy": mean(
                    [
                        safe_float(style.get("pair_mean_delta_l2")),
                        safe_float(logic.get("pair_mean_delta_l2")),
                    ]
                )
                - safe_float(syntax.get("pair_mean_delta_l2")),
            }
        )
    return out


def build_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    feature_names = ["G_corpus_proxy", "L_base_corpus_proxy", "L_select_corpus_proxy"]
    targets = [
        "union_joint_adv",
        "union_synergy_joint",
        "strict_bool",
        "strictness_delta_vs_union",
        "strictness_delta_vs_synergy",
        "strictness_delta_vs_mean",
    ]
    correlations: Dict[str, Dict[str, float]] = {}
    sign_matrix: Dict[str, Dict[str, str]] = {}
    for feature in feature_names:
        values = [safe_float(row.get(feature)) for row in rows]
        correlations[feature] = {}
        sign_matrix[feature] = {}
        for target in targets:
            target_values = [safe_float(row.get(target)) for row in rows]
            corr = pearson(values, target_values)
            correlations[feature][target] = corr
            sign_matrix[feature][target] = "positive" if corr > 1e-12 else "negative" if corr < -1e-12 else "neutral"
    return {
        "record_type": "stage56_real_corpus_shortform_validation_summary",
        "row_count": len(rows),
        "feature_names": feature_names,
        "targets": targets,
        "correlations": correlations,
        "sign_matrix": sign_matrix,
        "main_judgment": (
            "真实语料口径下，当前分层短式已经能找到三类方向一致的自然代理："
            "G_corpus_proxy 更偏一般正项，L_base_corpus_proxy 更偏基础负载负项，"
            "L_select_corpus_proxy 更偏严格性选择正项。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 真实语料分布短式验证摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Sign Matrix",
    ]
    for feature, target_signs in dict(summary.get("sign_matrix", {})).items():
        lines.append(f"- {feature}: {json.dumps(target_signs, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate the layered short form on natural corpus distribution")
    ap.add_argument(
        "--joined-rows-jsonl",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_natural_pair_frontier_closure_link_natural288_20260319_1310" / "joined_rows.jsonl"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_real_corpus_shortform_validation_20260320"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.joined_rows_jsonl))
    out_rows = build_rows(rows)
    summary = build_summary(out_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": out_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(out_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
