from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage68_direct_theorem_probe_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_task_level_repair_comparison import build_task_level_repair_comparison_summary
from stage61_coefficient_uniqueness_probe import build_coefficient_uniqueness_probe_summary
from stage68_direct_signal_bundle import build_direct_signal_bundle_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_direct_theorem_probe_summary() -> dict:
    signals = build_direct_signal_bundle_summary()["headline_metrics"]
    repair = build_task_level_repair_comparison_summary()["candidate_repairs"]["sqrt"]
    uniq = build_coefficient_uniqueness_probe_summary()["headline_metrics"]

    direct_existence_support = _clip01(
        0.34 * signals["direct_structural_coherence"]
        + 0.26 * signals["direct_weight_grounding"]
        + 0.20 * repair["repaired_direct_structure"]
        + 0.20 * repair["repaired_direct_route"]
    )
    direct_uniqueness_support = _clip01(
        0.28 * uniq["shared_constraints"]
        + 0.24 * uniq["language_brain_agreement"]
        + 0.24 * signals["direct_weight_grounding"]
        + 0.24 * (1.0 - repair["repaired_brain_gap"])
    )
    direct_stability_support = _clip01(
        0.30 * signals["direct_boundary_resilience"]
        + 0.28 * signals["direct_task_recovery_support"]
        + 0.22 * (1.0 - repair["repaired_long_forgetting"])
        + 0.20 * (1.0 - min(1.0, repair["repaired_base_perplexity_delta"] / 1200.0))
    )
    direct_theorem_readiness = _clip01(
        0.30 * direct_existence_support
        + 0.34 * direct_uniqueness_support
        + 0.24 * direct_stability_support
        + 0.12 * signals["direct_weight_grounding"]
    )
    direct_theorem_gap = _clip01(1.0 - direct_theorem_readiness)

    return {
        "headline_metrics": {
            "direct_existence_support": direct_existence_support,
            "direct_uniqueness_support": direct_uniqueness_support,
            "direct_stability_support": direct_stability_support,
            "direct_theorem_readiness": direct_theorem_readiness,
            "direct_theorem_gap": direct_theorem_gap,
        },
        "status": {
            "status_short": "direct_theorem_probe_active",
            "status_label": "存在性、唯一性、稳定性已经被改写成更直接的探针，但严格定理仍未完成",
        },
        "project_readout": {
            "summary": "这一轮直接从任务修复量、边界韧性和系数约束出发构造存在性、唯一性、稳定性探针，不再套用中间闭合变量。",
            "next_question": "下一步要把这三个直接探针并回最终身份判断，验证不走嵌套链时，理论身份是否仍然成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage68 Direct Theorem Probe",
        "",
        f"- direct_existence_support: {hm['direct_existence_support']:.6f}",
        f"- direct_uniqueness_support: {hm['direct_uniqueness_support']:.6f}",
        f"- direct_stability_support: {hm['direct_stability_support']:.6f}",
        f"- direct_theorem_readiness: {hm['direct_theorem_readiness']:.6f}",
        f"- direct_theorem_gap: {hm['direct_theorem_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_direct_theorem_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
