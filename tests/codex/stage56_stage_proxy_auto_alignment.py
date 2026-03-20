from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_stage_proxy_auto_alignment_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _normalize(frontier: float, boundary: float, atlas: float) -> tuple[float, float, float]:
    anchor = min(frontier, boundary, atlas)
    return frontier - anchor, boundary - anchor, atlas - anchor


def _row_to_steps(row: dict) -> tuple[float, float, float]:
    if "frontier_step" in row:
        return row["frontier_step"], row["boundary_step"], row["atlas_step"]
    return row["frontier_shift_step"], row["boundary_hardening_step"], row["atlas_freeze_step"]


def build_stage_proxy_auto_alignment_summary() -> dict:
    rec = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_heterogeneous_asset_recanonicalization_20260320" / "summary.json"
    )
    aligned_rows = []
    for row in rec["rows"]:
        if not row.get("comparable"):
            continue
        frontier, boundary, atlas = _row_to_steps(row)
        f_norm, b_norm, a_norm = _normalize(frontier, boundary, atlas)
        aligned_rows.append(
            {
                "name": row["name"],
                "frontier_norm": f_norm,
                "boundary_norm": b_norm,
                "atlas_norm": a_norm,
                "ordered": f_norm <= b_norm <= a_norm,
            }
        )
    ordered_ratio = sum(1 for row in aligned_rows if row["ordered"]) / max(len(aligned_rows), 1)
    summary = {
        "case_count": len(aligned_rows),
        "ordered_ratio": ordered_ratio,
        "rows": aligned_rows,
        "project_readout": {
            "summary": "阶段代理自动对齐把不同资产的绝对步数改写成相对起点后的归一化阶段位置，从而减少跨资产不共尺度问题。",
            "next_question": "下一步要把这种自动对齐继续推进成更原生的训练阶段检测器，而不是只做步数平移。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    lines = [
        "# 阶段代理自动对齐报告",
        "",
        f"- case_count: {summary['case_count']}",
        f"- ordered_ratio: {summary['ordered_ratio']:.6f}",
        "",
        "## 个案",
    ]
    for row in summary["rows"]:
        lines.append(
            f"- {row['name']}: frontier_norm={row['frontier_norm']:.2f}, boundary_norm={row['boundary_norm']:.2f}, "
            f"atlas_norm={row['atlas_norm']:.2f}, ordered={row['ordered']}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_stage_proxy_auto_alignment_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
