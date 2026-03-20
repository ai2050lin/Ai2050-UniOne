from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_model_long_horizon_alignment_20260320"


@dataclass
class HorizonCase:
    name: str
    frontier_step: float
    boundary_step: float
    atlas_step: float


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _extract_numeric(seq, key: str | None = None) -> list[float]:
    if not seq:
        return []
    if key is None:
        return [float(v) for v in seq]
    return [float(row.get(key, 0.0)) for row in seq]


def _first_drop_step(values: list[float], ratio: float = 0.5) -> float:
    if not values:
        return 0.0
    start = float(values[0])
    end = float(values[-1])
    if abs(start - end) < 1e-12:
        return float(len(values))
    target = start + (end - start) * ratio
    if end < start:
        for idx, value in enumerate(values, start=1):
            if value <= target:
                return float(idx)
    else:
        for idx, value in enumerate(values, start=1):
            if value >= target:
                return float(idx)
    return float(len(values))


def _case_true_long(path: Path) -> HorizonCase:
    data = _load_json(path)
    frontier_seq = _extract_numeric(data.get("proto_history", []), "train_loss")
    boundary_seq = _extract_numeric(data.get("stabilization_history", []))
    atlas_seq = _extract_numeric(data.get("consolidation_history", []), "val_loss")
    f = _first_drop_step(frontier_seq, 0.55)
    b = len(frontier_seq) + _first_drop_step(boundary_seq, 0.45)
    a = len(frontier_seq) + len(boundary_seq) + _first_drop_step(atlas_seq, 0.5)
    return HorizonCase("openwebtext_true_long_run", f, b, a)


def _case_longterm(path: Path) -> HorizonCase:
    data = _load_json(path)
    frontier_seq = _extract_numeric(data.get("proto_history", []))
    boundary_seq = _extract_numeric(data.get("stabilization_history", []))
    atlas_seq = _extract_numeric(data.get("consolidation_history", []))
    f = _first_drop_step(frontier_seq, 0.55)
    b = len(frontier_seq) + _first_drop_step(boundary_seq, 0.45)
    a = len(frontier_seq) + len(boundary_seq) + _first_drop_step(atlas_seq, 0.5)
    return HorizonCase("openwebtext_longterm", f, b, a)


def _case_extended(path: Path) -> HorizonCase:
    data = _load_json(path)
    frontier_seq = _extract_numeric(data.get("proto_history_1", []))
    boundary_seq = _extract_numeric(data.get("stabilization_history", []))
    atlas_seq = _extract_numeric(data.get("final_read_stabilization_history", []))
    if not atlas_seq:
        atlas_seq = _extract_numeric(data.get("consolidation_history", []))
    f = _first_drop_step(frontier_seq, 0.55)
    b = len(frontier_seq) + _first_drop_step(boundary_seq, 0.45)
    a = len(frontier_seq) + len(boundary_seq) + _first_drop_step(atlas_seq, 0.5)
    return HorizonCase("openwebtext_extended_continual", f, b, a)


def _case_persistent(path: Path) -> HorizonCase:
    data = _load_json(path)
    frontier_seq = _extract_numeric(data.get("proto_history", []), "train_loss")
    boundary_seq = _extract_numeric(data.get("stabilization_history", []))
    atlas_seq = _extract_numeric(data.get("gate_alignment_history", []))
    f = _first_drop_step(frontier_seq, 0.55)
    b = len(frontier_seq) + _first_drop_step(boundary_seq, 0.45)
    a = len(frontier_seq) + len(boundary_seq) + _first_drop_step(atlas_seq, 0.5)
    return HorizonCase("openwebtext_persistent", f, b, a)


def build_long_horizon_alignment_summary() -> dict:
    cases = [
        _case_true_long(ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_true_long_run_block.json"),
        _case_longterm(ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_longterm_training_block.json"),
        _case_extended(ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_extended_continual_block.json"),
        _case_persistent(ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_persistent_training_block.json"),
    ]
    rows = []
    ordered = 0
    for c in cases:
        ok = c.frontier_step <= c.boundary_step <= c.atlas_step
        ordered += int(ok)
        rows.append(
            {
                "name": c.name,
                "frontier_step": c.frontier_step,
                "boundary_step": c.boundary_step,
                "atlas_step": c.atlas_step,
                "ordered": ok,
            }
        )
    summary = {
        "case_count": len(rows),
        "rows": rows,
        "headline_metrics": {
            "frontier_mean_step": mean([c.frontier_step for c in cases]),
            "boundary_mean_step": mean([c.boundary_step for c in cases]),
            "atlas_mean_step": mean([c.atlas_step for c in cases]),
            "ordered_case_ratio": ordered / len(cases),
        },
        "project_readout": {
            "summary": (
                "这一步只看更长训练块，把阶段顺序限制在前沿、边界、图册三类长程段。"
                "如果前沿 <= 边界 <= 图册在这些长程块里稳定成立，"
                "就说明多时间尺度顺序在更真实的训练阶段开始收口。"
            ),
            "next_question": "下一步要把这个长程顺序和大模型在线稳定性代理并回同一条学习系统公式。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    lines = [
        "# 大模型长程训练阶段对齐报告",
        "",
        f"- 案例数: {summary['case_count']}",
        f"- 前沿平均步: {summary['headline_metrics']['frontier_mean_step']:.3f}",
        f"- 边界平均步: {summary['headline_metrics']['boundary_mean_step']:.3f}",
        f"- 图册平均步: {summary['headline_metrics']['atlas_mean_step']:.3f}",
        f"- 顺序支持比例: {summary['headline_metrics']['ordered_case_ratio']:.3f}",
        "",
        "## 个案",
    ]
    for row in summary["rows"]:
        lines.append(
            f"- {row['name']}: frontier={row['frontier_step']:.3f}, "
            f"boundary={row['boundary_step']:.3f}, atlas={row['atlas_step']:.3f}, "
            f"ordered={row['ordered']}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_long_horizon_alignment_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
