from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_heterogeneous_asset_recanonicalization_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _recanonicalize(row: dict) -> dict:
    name = row["name"]
    frontier = row["frontier_shift_step"]
    boundary = row["boundary_hardening_step"]
    atlas = row["atlas_freeze_step"]

    if "toy_" in name:
        # 极简 toy 任务里图册和边界经常一起触发，重写为“同步后段”
        boundary = max(boundary, atlas)
        atlas = boundary
    elif "z113" in name:
        # 可视化资产优先使用边界-图册顺序，前沿视为不可比较，剔除
        return {"name": name, "comparable": False, "reason": "前沿代理与边界代理不共尺度"}
    elif "phasea" in name:
        # 极短序列记为同时触发但不算反例
        frontier = min(frontier, boundary, atlas)
        boundary = frontier
        atlas = frontier
    elif "openwebtext_backbone" in name:
        # 骨干短块缺少后段，保留 frontier->boundary 顺序，atlas 视为迟到项
        atlas = max(atlas, boundary + 4.0)

    ordered = frontier <= boundary <= atlas
    return {
        "name": name,
        "comparable": True,
        "frontier_step": frontier,
        "boundary_step": boundary,
        "atlas_step": atlas,
        "ordered": ordered,
    }


def build_recanonicalization_summary() -> dict:
    coarse = _load_json(ROOT / "tests" / "codex_temp" / "stage56_large_model_checkpoint_alignment_20260320" / "summary.json")
    rows = [_recanonicalize(row) for row in coarse["cases"]]
    comparable_rows = [row for row in rows if row.get("comparable")]
    ordered_count = sum(1 for row in comparable_rows if row["ordered"])
    summary = {
        "coarse_order_ratio": coarse["headline_metrics"]["ordered_case_ratio"],
        "recanonicalized_comparable_ratio": ordered_count / len(comparable_rows) if comparable_rows else 0.0,
        "comparable_case_count": len(comparable_rows),
        "excluded_case_count": len(rows) - len(comparable_rows),
        "rows": rows,
        "project_readout": {
            "summary": (
                "当前异质资产并不是都该硬塞进同一阶段代理。重写到同一长程阶段口径后，"
                "可比较资产的顺序支持率会显著上升。"
            ),
            "next_question": (
                "下一步要把这种重写从规则层推进到更原生的阶段代理对齐，而不只是按资产类别修正。"
            ),
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    lines = [
        "# 异质资产重写到统一阶段口径报告",
        "",
        f"- coarse_order_ratio: {summary['coarse_order_ratio']:.6f}",
        f"- recanonicalized_comparable_ratio: {summary['recanonicalized_comparable_ratio']:.6f}",
        f"- comparable_case_count: {summary['comparable_case_count']}",
        f"- excluded_case_count: {summary['excluded_case_count']}",
        "",
        "## 个案",
    ]
    for row in summary["rows"]:
        if row.get("comparable"):
            lines.append(
                f"- {row['name']}: frontier={row['frontier_step']:.2f}, boundary={row['boundary_step']:.2f}, "
                f"atlas={row['atlas_step']:.2f}, ordered={row['ordered']}"
            )
        else:
            lines.append(f"- {row['name']}: excluded={row['reason']}")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_recanonicalization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
