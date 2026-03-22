from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage70_native_observability_bridge_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage56_native_variable_candidate_mapping import build_native_variable_candidate_mapping_summary
from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage60_symbolic_coefficient_grounding import build_symbolic_coefficient_grounding_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_native_observability_bridge_summary() -> dict:
    native = build_native_variable_candidate_mapping_summary()
    mapping = native["candidate_mapping"]
    native_hm = native["headline_metrics"]
    context = build_context_native_grounding_summary()["headline_metrics"]
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    coeff = build_symbolic_coefficient_grounding_summary()["headline_metrics"]

    base_observability = sum(item["observability"] for item in mapping.values()) / len(mapping)
    locality_support = sum(item["locality"] for item in mapping.values()) / len(mapping)
    observability_bridge_score = _clip01(
        0.24 * base_observability
        + 0.18 * locality_support
        + 0.18 * context["context_native_readiness"]
        + 0.16 * fiber["reinforcement_readiness"]
        + 0.14 * coeff["native_coefficient_score"]
        + 0.10 * native_hm["native_mapping_completeness"]
    )
    proxy_traceability_score = _clip01(
        0.30 * context["conditional_gate_stability"]
        + 0.24 * context["context_route_alignment"]
        + 0.22 * fiber["cross_region_share_stability"]
        + 0.24 * coeff["coefficient_grounding_coverage"]
    )
    hidden_proxy_gap = _clip01(
        1.0 - (0.54 * observability_bridge_score + 0.46 * proxy_traceability_score)
    )

    return {
        "headline_metrics": {
            "base_observability": base_observability,
            "observability_bridge_score": observability_bridge_score,
            "proxy_traceability_score": proxy_traceability_score,
            "hidden_proxy_gap": hidden_proxy_gap,
        },
        "status": {
            "status_short": "native_observability_bridge_active",
            "status_label": "原生变量已经从候选概念进一步桥接到可观测、可追踪的代理量，但仍有剩余隐藏代理缺口",
        },
        "project_readout": {
            "summary": "这一轮专门评估原生变量到底有多少已经变成了可观测代理量，而不是停留在理论命名层。",
            "next_question": "下一步要把这些可观测桥接结果并回直算稳定性和最终身份锁定，检查理论是否继续受益于原生变量化。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage70 Native Observability Bridge",
        "",
        f"- base_observability: {hm['base_observability']:.6f}",
        f"- observability_bridge_score: {hm['observability_bridge_score']:.6f}",
        f"- proxy_traceability_score: {hm['proxy_traceability_score']:.6f}",
        f"- hidden_proxy_gap: {hm['hidden_proxy_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_native_observability_bridge_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
