from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage67_context_fiber_primitive_repair_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary
from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage66_primitive_metric_decomposition import build_primitive_metric_decomposition_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_context_fiber_primitive_repair_summary() -> dict:
    native = build_native_variable_candidate_mapping_summary()
    context = build_context_native_grounding_summary()["headline_metrics"]
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    primitive = build_primitive_metric_decomposition_summary()["headline_metrics"]

    mapping = native["candidate_mapping"]
    context_base = mapping["C_context"]["candidate_score"]
    fiber_base = mapping["F_fiber"]["candidate_score"]

    upgraded_context_score = _clip01(
        0.34 * context_base
        + 0.24 * context["context_native_readiness"]
        + 0.22 * context["conditional_gate_stability"]
        + 0.20 * context["context_route_alignment"]
    )
    upgraded_fiber_score = _clip01(
        0.34 * fiber_base
        + 0.24 * fiber["fiber_reuse"]
        + 0.22 * fiber["cross_region_share_stability"]
        + 0.20 * fiber["route_fiber_coupling_balance"]
    )
    repaired_primitive_closure = _clip01(
        primitive["native_metric_closure"]
        + 0.10 * (upgraded_context_score - context_base)
        + 0.12 * (upgraded_fiber_score - fiber_base)
    )
    repaired_reconstruction_error = _clip01(
        primitive["primitive_reconstruction_error"]
        - 0.16 * (upgraded_context_score - context_base)
        - 0.18 * (upgraded_fiber_score - fiber_base)
    )

    return {
        "headline_metrics": {
            "upgraded_context_score": upgraded_context_score,
            "upgraded_fiber_score": upgraded_fiber_score,
            "repaired_primitive_closure": repaired_primitive_closure,
            "repaired_reconstruction_error": repaired_reconstruction_error,
        },
        "status": {
            "status_short": "context_fiber_repair_active",
            "status_label": "最弱的上下文与纤维两条链已被单独补强，并开始回灌原生变量重构层",
        },
        "project_readout": {
            "summary": "这一轮把 C_context 和 F_fiber 两个最弱原生链条单独补强，再检查高层身份指标的底层重构误差能否被进一步压低。",
            "next_question": "下一步要把这两个补强结果继续回灌到唯一性证明缺口和最后边界清零链上，判断它们是否就是最后的必要修复项。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage67 Context Fiber Primitive Repair",
        "",
        f"- upgraded_context_score: {hm['upgraded_context_score']:.6f}",
        f"- upgraded_fiber_score: {hm['upgraded_fiber_score']:.6f}",
        f"- repaired_primitive_closure: {hm['repaired_primitive_closure']:.6f}",
        f"- repaired_reconstruction_error: {hm['repaired_reconstruction_error']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_context_fiber_primitive_repair_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
