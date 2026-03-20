from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_large_model_checkpoint_alignment_20260320"


@dataclass
class AlignmentCase:
    name: str
    atlas_freeze_step: float
    frontier_shift_step: float
    boundary_hardening_step: float


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _normalize_progress(values: Iterable[float], *, higher_better: bool) -> list[float]:
    seq = [float(v) for v in values]
    if not seq:
        return []
    if not higher_better:
        seq = [-v for v in seq]
    lo = min(seq)
    hi = max(seq)
    if hi - lo < 1e-12:
        return [0.0 for _ in seq]
    return [(v - lo) / (hi - lo) for v in seq]


def _first_reach(progress: list[float], threshold: float = 0.7) -> float:
    if not progress:
        return 0.0
    for idx, value in enumerate(progress, start=1):
        if value >= threshold:
            return float(idx)
    return float(len(progress))


def _from_phasea(path: Path) -> AlignmentCase:
    data = _load_json(path)
    history = data.get("history", [])
    atlas = _normalize_progress(
        [row.get("semantic_benchmark_score", 0.0) for row in history],
        higher_better=True,
    )
    frontier = _normalize_progress(
        [row.get("mean_train_loss", 0.0) for row in history],
        higher_better=False,
    )
    boundary = _normalize_progress(
        [row.get("generation_quality_score", 0.0) for row in history],
        higher_better=True,
    )
    return AlignmentCase(
        name="icspb_phasea",
        atlas_freeze_step=_first_reach(atlas),
        frontier_shift_step=_first_reach(frontier),
        boundary_hardening_step=_first_reach(boundary),
    )


def _from_toy(path: Path, key: str) -> AlignmentCase:
    data = _load_json(path)
    history = data.get(key, [])
    atlas = _normalize_progress([row.get("accuracy", 0.0) for row in history], higher_better=True)
    frontier = _normalize_progress([row.get("loss", 0.0) for row in history], higher_better=False)
    boundary = atlas
    return AlignmentCase(
        name=f"toy_{key.lower()}",
        atlas_freeze_step=_first_reach(atlas),
        frontier_shift_step=_first_reach(frontier),
        boundary_hardening_step=_first_reach(boundary, threshold=0.85),
    )


def _from_z113(path: Path) -> AlignmentCase:
    data = _load_json(path)
    train_acc = data.get("train_acc", [])
    test_acc = data.get("test_acc", [])
    epochs = data.get("epoch", [])
    atlas = _normalize_progress(train_acc, higher_better=True)
    frontier = _normalize_progress(test_acc, higher_better=True)
    generalization_gap = [float(t) - float(tr) for tr, t in zip(train_acc, test_acc)]
    boundary = _normalize_progress(generalization_gap, higher_better=True)
    step_scale = (float(len(epochs)) / float(len(train_acc))) if train_acc else 1.0
    return AlignmentCase(
        name="glm5_z113_visuals",
        atlas_freeze_step=_first_reach(atlas) * step_scale,
        frontier_shift_step=_first_reach(frontier) * step_scale,
        boundary_hardening_step=_first_reach(boundary) * step_scale,
    )


def _from_openwebtext_block(path: Path) -> AlignmentCase:
    data = _load_json(path)
    atlas = _normalize_progress(data.get("curve_history", {}).get("proto_baseline", []), higher_better=False)
    frontier = _normalize_progress(data.get("curve_history", {}).get("proto_warmup", []), higher_better=False)
    boundary = _normalize_progress(data.get("stabilization_history", []), higher_better=False)
    return AlignmentCase(
        name="openwebtext_backbone_v2",
        atlas_freeze_step=_first_reach(atlas),
        frontier_shift_step=_first_reach(frontier),
        boundary_hardening_step=_first_reach(boundary),
    )


def build_alignment_summary() -> dict:
    cases: list[AlignmentCase] = []
    phasea = ROOT / "tempdata" / "icspb_phasea_training_history.json"
    toy = ROOT / "research" / "glm5" / "experiments" / "toy_experiment" / "training_log.json"
    z113 = ROOT / "research" / "glm5" / "experiments" / "z113_visuals" / "training_log.json"
    openweb = ROOT / "tests" / "codex_temp" / "icspb_v2_openwebtext_real_training_curve_block.json"
    if phasea.exists():
        cases.append(_from_phasea(phasea))
    if toy.exists():
        cases.append(_from_toy(toy, "Transformer"))
        cases.append(_from_toy(toy, "FiberNet"))
    if z113.exists():
        cases.append(_from_z113(z113))
    if openweb.exists():
        cases.append(_from_openwebtext_block(openweb))

    case_rows = [
        {
            "name": c.name,
            "atlas_freeze_step": c.atlas_freeze_step,
            "frontier_shift_step": c.frontier_shift_step,
            "boundary_hardening_step": c.boundary_hardening_step,
            "ordering_signature": [
                c.atlas_freeze_step <= c.frontier_shift_step,
                c.frontier_shift_step <= c.boundary_hardening_step,
            ],
        }
        for c in cases
    ]
    ordering_support = sum(
        1
        for c in cases
        if c.atlas_freeze_step <= c.frontier_shift_step <= c.boundary_hardening_step
    )
    summary = {
        "case_count": len(cases),
        "cases": case_rows,
        "headline_metrics": {
            "atlas_mean_step": mean([c.atlas_freeze_step for c in cases]) if cases else 0.0,
            "frontier_mean_step": mean([c.frontier_shift_step for c in cases]) if cases else 0.0,
            "boundary_mean_step": mean([c.boundary_hardening_step for c in cases]) if cases else 0.0,
            "ordered_case_ratio": ordering_support / len(cases) if cases else 0.0,
        },
        "project_readout": {
            "summary": (
                "这一步把现有较大训练历史和真实训练曲线统一到图册、前沿、边界三阶段口径。"
                "如果图册冻结 <= 前沿迁移 <= 边界硬化在多数资产上成立，"
                "就说明当前多时间尺度学习顺序开始跨资产稳定。"
            ),
            "next_question": "下一步要检查这种顺序是否也在更真实的大模型在线恢复与长期训练资产上成立。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    lines = [
        "# 大模型检查点对齐报告",
        "",
        f"- 资产数量: {summary['case_count']}",
        f"- 图册平均冻结步: {summary['headline_metrics']['atlas_mean_step']:.3f}",
        f"- 前沿平均迁移步: {summary['headline_metrics']['frontier_mean_step']:.3f}",
        f"- 边界平均硬化步: {summary['headline_metrics']['boundary_mean_step']:.3f}",
        f"- 顺序支持比例: {summary['headline_metrics']['ordered_case_ratio']:.3f}",
        "",
        "## 个案",
    ]
    for row in summary["cases"]:
        lines.append(
            f"- {row['name']}: atlas={row['atlas_freeze_step']:.3f}, "
            f"frontier={row['frontier_shift_step']:.3f}, boundary={row['boundary_hardening_step']:.3f}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_alignment_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
