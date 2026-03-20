from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_native_stage_detector_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_native_stage_detector_summary() -> dict:
    local_native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_native_update_field_20260320" / "summary.json"
    )
    auto_align = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_stage_proxy_auto_alignment_20260320" / "summary.json"
    )
    local_eq = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_global_learning_equation_20260320" / "summary.json"
    )

    hm_local = local_native["headline_metrics"]
    hm_eq = local_eq["headline_metrics"]
    rows = []
    for row in auto_align["rows"]:
        frontier_detector = row["frontier_norm"] / max(hm_local["patch_update_native"], 1e-12)
        boundary_detector = row["boundary_norm"] / max(hm_local["boundary_response_native"], 1e-12)
        atlas_detector = row["atlas_norm"] / max(hm_local["atlas_consolidation_native"], 1e-12)
        rows.append(
            {
                "name": row["name"],
                "frontier_detector": frontier_detector,
                "boundary_detector": boundary_detector,
                "atlas_detector": atlas_detector,
                "ordered": frontier_detector <= boundary_detector <= atlas_detector,
            }
        )
    ordered_ratio = sum(1 for row in rows if row["ordered"]) / max(len(rows), 1)
    summary = {
        "case_count": len(rows),
        "ordered_ratio": ordered_ratio,
        "rows": rows,
        "detector_equation": {
            "frontier_stage": "T_frontier ~ frontier_norm / patch_update_native",
            "boundary_stage": "T_boundary ~ boundary_norm / boundary_response_native",
            "atlas_stage": "T_atlas ~ atlas_norm / atlas_consolidation_native",
        },
        "headline_metrics": {
            "frontier_detector_mean": mean(row["frontier_detector"] for row in rows),
            "boundary_detector_mean": mean(row["boundary_detector"] for row in rows),
            "atlas_detector_mean": mean(row["atlas_detector"] for row in rows),
        },
        "project_readout": {
            "summary": "当前阶段检测已经开始从规则式步数平移，推进到结合局部更新场和阶段归一化位置的原生检测器。",
            "next_question": "下一步要把这个检测器接到真实更长训练资产上，看它能否替代手工阶段定义。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 原生阶段检测器报告",
        "",
        f"- case_count: {summary['case_count']}",
        f"- ordered_ratio: {summary['ordered_ratio']:.6f}",
        f"- frontier_detector_mean: {hm['frontier_detector_mean']:.6f}",
        f"- boundary_detector_mean: {hm['boundary_detector_mean']:.6f}",
        f"- atlas_detector_mean: {hm['atlas_detector_mean']:.6f}",
        "",
        "## 个案",
    ]
    for row in summary["rows"]:
        lines.append(
            f"- {row['name']}: frontier={row['frontier_detector']:.6f}, boundary={row['boundary_detector']:.6f}, "
            f"atlas={row['atlas_detector']:.6f}, ordered={row['ordered']}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_native_stage_detector_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
