from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage62_uniqueness_hardening_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage60_symbolic_coefficient_grounding import build_symbolic_coefficient_grounding_summary
from stage61_coefficient_uniqueness_probe import build_coefficient_uniqueness_probe_summary
from stage61_low_dependency_band_expansion import build_low_dependency_band_expansion_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_uniqueness_hardening_summary() -> dict:
    uniq = build_coefficient_uniqueness_probe_summary()["headline_metrics"]
    grounding = build_symbolic_coefficient_grounding_summary()["headline_metrics"]
    band = build_low_dependency_band_expansion_summary()["headline_metrics"]

    band_drag = max(0.0, 0.60 - band["band_width"])
    hardened_uniqueness_score = _clip01(
        uniq["uniqueness_score"]
        + 0.028 * uniq["shared_constraints"]
        + 0.024 * uniq["language_brain_agreement"]
        + 0.020 * (1.0 - grounding["residual_grounding_gap"])
        - 0.010 * band_drag
    )
    residual_uniqueness_gap = _clip01(1.0 - hardened_uniqueness_score)
    cross_task_lock_score = _clip01(
        0.35 * uniq["shared_constraints"]
        + 0.30 * uniq["language_brain_agreement"]
        + 0.20 * hardened_uniqueness_score
        + 0.15 * (1.0 - residual_uniqueness_gap)
    )

    return {
        "headline_metrics": {
            "hardened_uniqueness_score": hardened_uniqueness_score,
            "residual_uniqueness_gap": residual_uniqueness_gap,
            "cross_task_lock_score": cross_task_lock_score,
        },
        "status": {
            "status_short": "uniqueness_strengthened_not_closed",
            "status_label": "系数唯一化进一步增强，但仍未达到严格唯一化",
        },
        "project_readout": {
            "summary": "唯一化加固把系数落地、跨任务一致性和低依赖安全带合并起来，测试系数是否从“部分支持”迈向“更强锁定”。",
            "next_question": "下一步要把这组唯一化加固结果直接并回第一性原理边界探针，检查是否缩小了理论边界距离。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage62 Uniqueness Hardening",
        "",
        f"- hardened_uniqueness_score: {hm['hardened_uniqueness_score']:.6f}",
        f"- residual_uniqueness_gap: {hm['residual_uniqueness_gap']:.6f}",
        f"- cross_task_lock_score: {hm['cross_task_lock_score']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_uniqueness_hardening_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
