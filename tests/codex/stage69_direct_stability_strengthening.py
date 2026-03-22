from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage69_direct_stability_strengthening_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage57_task_level_repair_comparison import build_task_level_repair_comparison_summary
from stage68_direct_signal_bundle import build_direct_signal_bundle_summary
from stage68_direct_theorem_probe import build_direct_theorem_probe_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_summary(relpath: str, builder) -> dict:
    path = ROOT / relpath
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return builder()


def build_direct_stability_strengthening_summary() -> dict:
    signals = _load_summary(
        "tests/codex_temp/stage68_direct_signal_bundle_20260322/summary.json",
        build_direct_signal_bundle_summary,
    )["headline_metrics"]
    theorem = _load_summary(
        "tests/codex_temp/stage68_direct_theorem_probe_20260322/summary.json",
        build_direct_theorem_probe_summary,
    )["headline_metrics"]
    context = build_context_native_grounding_summary()["headline_metrics"]
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    repair = build_task_level_repair_comparison_summary()["candidate_repairs"]["sqrt"]

    stability_gain = _clip01(
        0.24 * context["conditional_gate_stability"]
        + 0.18 * context["context_route_alignment"]
        + 0.18 * fiber["cross_region_share_stability"]
        + 0.14 * fiber["pressure_under_reuse"]
        + 0.14 * repair["repaired_direct_route"]
        + 0.12 * (1.0 - repair["repaired_brain_gap"])
    )
    strengthened_direct_stability_support = _clip01(
        theorem["direct_stability_support"]
        + 0.10 * stability_gain
        + 0.05 * signals["direct_boundary_resilience"]
    )
    residual_stability_gap = _clip01(1.0 - strengthened_direct_stability_support)

    return {
        "headline_metrics": {
            "stability_gain": stability_gain,
            "strengthened_direct_stability_support": strengthened_direct_stability_support,
            "residual_stability_gap": residual_stability_gap,
        },
        "status": {
            "status_short": "direct_stability_strengthened",
            "status_label": "直算链中最弱的稳定性支撑已被单独补强，但距离严格稳定性定理仍有缺口",
        },
        "project_readout": {
            "summary": "这一轮专门围绕上下文门控、纤维复用、压力控制和任务恢复来补强直算链里的稳定性支持项。",
            "next_question": "下一步要把增强后的稳定性支撑并回直算定理缺口，看看能否进一步压低 direct_theorem_gap。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage69 Direct Stability Strengthening",
        "",
        f"- stability_gain: {hm['stability_gain']:.6f}",
        f"- strengthened_direct_stability_support: {hm['strengthened_direct_stability_support']:.6f}",
        f"- residual_stability_gap: {hm['residual_stability_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_direct_stability_strengthening_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
