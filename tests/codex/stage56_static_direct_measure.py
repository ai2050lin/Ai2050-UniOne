from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from stage56_fullsample_regression_runner import fit_linear_regression, read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def category_key(row: Dict[str, object]) -> Tuple[str, str]:
    return (str(row.get("model_id", "")), str(row.get("category", "")))


def build_rows(design_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in design_rows:
        grouped.setdefault(category_key(row), []).append(dict(row))

    out: List[Dict[str, object]] = []
    for row in design_rows:
        peers = grouped.get(category_key(row), [])
        atlas_mean = mean([safe_float(peer.get("atlas_static_proxy")) for peer in peers])
        frontier_mean = mean([safe_float(peer.get("frontier_dynamic_proxy")) for peer in peers])
        offset_mean = mean([safe_float(peer.get("offset_static_proxy")) for peer in peers])

        atlas = safe_float(row.get("atlas_static_proxy"))
        frontier = safe_float(row.get("frontier_dynamic_proxy"))
        offset = safe_float(row.get("offset_static_proxy"))

        family_patch_direct = mean([atlas_mean, frontier_mean])
        concept_offset_direct = mean(
            [
                abs(atlas - atlas_mean),
                abs(frontier - frontier_mean),
                abs(offset - offset_mean),
            ]
        )
        identity_margin_direct = family_patch_direct - concept_offset_direct

        out.append(
            {
                **dict(row),
                "family_patch_direct": family_patch_direct,
                "concept_offset_direct": concept_offset_direct,
                "identity_margin_direct": identity_margin_direct,
            }
        )
    return out


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    feature_names = [
        "family_patch_direct",
        "concept_offset_direct",
        "identity_margin_direct",
    ]
    fits = [
        fit_linear_regression(rows, feature_names, "union_joint_adv"),
        fit_linear_regression(rows, feature_names, "union_synergy_joint"),
        fit_linear_regression(rows, feature_names, "strict_positive_synergy"),
    ]
    return {
        "record_type": "stage56_static_direct_measure_summary",
        "row_count": len(rows),
        "feature_names": feature_names,
        "mean_family_patch_direct": mean([safe_float(row.get("family_patch_direct")) for row in rows]),
        "mean_concept_offset_direct": mean([safe_float(row.get("concept_offset_direct")) for row in rows]),
        "mean_identity_margin_direct": mean([safe_float(row.get("identity_margin_direct")) for row in rows]),
        "fits": fits,
        "main_judgment": (
            "静态本体层已经从摘要代理进一步推进到类别局部直测，"
            "现在可以直接检查 family patch 与 concept offset 在样本级的局部测度。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 静态项直测摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- mean_family_patch_direct: {safe_float(summary.get('mean_family_patch_direct')):+.6f}",
        f"- mean_concept_offset_direct: {safe_float(summary.get('mean_concept_offset_direct')):+.6f}",
        f"- mean_identity_margin_direct: {safe_float(summary.get('mean_identity_margin_direct')):+.6f}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Fits",
    ]
    for fit in list(summary.get("fits", [])):
        fit = dict(fit)
        lines.append(f"- target: {fit.get('target_name', '')}")
        for key, value in dict(fit.get("weights", {})).items():
            lines.append(f"  {key}: {safe_float(value):+.6f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Strengthen static terms using category-local direct measures")
    ap.add_argument(
        "--design-rows-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_fullsample_regression_runner_20260319" / "design_rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_static_direct_measure_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    design_rows = list(read_json(Path(args.design_rows_json)).get("rows", []))
    rows = build_rows(design_rows)
    summary = build_summary(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
