from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage72_language_projection_covariance_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_context_native_grounding import build_context_native_grounding_summary
from stage57_fiber_reuse_reinforcement import build_fiber_reuse_reinforcement_summary
from stage70_native_observability_bridge import build_native_observability_bridge_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_language_projection_covariance_summary() -> dict:
    context = build_context_native_grounding_summary()["headline_metrics"]
    fiber = build_fiber_reuse_reinforcement_summary()["headline_metrics"]
    obs = build_native_observability_bridge_summary()["headline_metrics"]

    scenarios = [
        {
            "name": "neutral",
            "q": 0.71,
            "b": 0.69,
            "g": 0.70,
            "ctx_push": 0.00,
            "route_pull": 0.01,
            "expected_projection": 0.74,
        },
        {
            "name": "style_bias",
            "q": 0.77,
            "b": 0.82,
            "g": 0.74,
            "ctx_push": 0.07,
            "route_pull": 0.03,
            "expected_projection": 0.79,
        },
        {
            "name": "logic_route",
            "q": 0.79,
            "b": 0.72,
            "g": 0.80,
            "ctx_push": 0.05,
            "route_pull": 0.06,
            "expected_projection": 0.81,
        },
        {
            "name": "syntax_repair",
            "q": 0.75,
            "b": 0.68,
            "g": 0.78,
            "ctx_push": 0.04,
            "route_pull": 0.07,
            "expected_projection": 0.78,
        },
    ]

    scenario_records = []
    covariance_terms = []
    bias_transport_terms = []
    route_alignment_terms = []
    resilience_terms = []

    for scenario in scenarios:
        q = scenario["q"]
        b = scenario["b"]
        g = scenario["g"]

        q_plus = _clip01(
            0.36 * q
            + 0.24 * b
            + 0.18 * context["context_native_readiness"]
            + 0.10 * scenario["ctx_push"]
            + 0.12 * (1.0 - obs["hidden_proxy_gap"])
        )
        b_plus = _clip01(
            0.52 * b
            + 0.22 * q
            + 0.12 * g
            + 0.08 * scenario["ctx_push"]
            + 0.06 * context["conditional_gate_stability"]
        )
        g_plus = _clip01(
            0.34 * g
            + 0.24 * q
            + 0.18 * context["context_route_alignment"]
            + 0.10 * scenario["route_pull"]
            + 0.08 * context["conditional_gate_stability"]
            + 0.06 * fiber["route_fiber_coupling_balance"]
        )
        projection = _clip01(
            0.34 * q_plus
            + 0.28 * g_plus
            + 0.22 * b_plus
            + 0.16 * context["context_route_alignment"]
        )

        covariance = _clip01(1.0 - abs((q_plus - q) - 0.85 * (g_plus - g)))
        bias_transport = _clip01(1.0 - abs((b_plus - b) - 0.70 * (q_plus - q)))
        route_alignment = _clip01(1.0 - abs(projection - scenario["expected_projection"]))
        resilience = _clip01(
            1.0 - max(0.0, abs(q_plus - g_plus) - 0.08) - max(0.0, abs(b_plus - q_plus) - 0.12)
        )

        covariance_terms.append(covariance)
        bias_transport_terms.append(bias_transport)
        route_alignment_terms.append(route_alignment)
        resilience_terms.append(resilience)
        scenario_records.append(
            {
                "name": scenario["name"],
                "q_plus": q_plus,
                "b_plus": b_plus,
                "g_plus": g_plus,
                "projection": projection,
                "covariance": covariance,
                "bias_transport": bias_transport,
                "route_alignment": route_alignment,
                "resilience": resilience,
            }
        )

    context_covariance_stability = sum(covariance_terms) / len(covariance_terms)
    bias_gate_transport = sum(bias_transport_terms) / len(bias_transport_terms)
    route_conditioned_projection = sum(route_alignment_terms) / len(route_alignment_terms)
    context_shift_resilience = sum(resilience_terms) / len(resilience_terms)

    stress_q = _clip01(
        0.32 * 0.68
        + 0.18 * 0.84
        + 0.16 * context["context_native_readiness"]
        + 0.08 * (1.0 - obs["hidden_proxy_gap"])
    )
    stress_g = _clip01(
        0.30 * 0.61
        + 0.18 * 0.68
        + 0.18 * context["context_route_alignment"]
        + 0.08 * fiber["route_fiber_coupling_balance"]
    )
    stress_projection = _clip01(0.36 * stress_q + 0.30 * stress_g + 0.18 * 0.84 + 0.16 * context["context_route_alignment"])
    projection_counterexample_resistance = _clip01(
        1.0 - max(0.0, 0.76 - stress_projection) - max(0.0, abs(stress_q - stress_g) - 0.10)
    )
    projection_gap = _clip01(1.0 - route_conditioned_projection)

    language_projection_repair_score = _clip01(
        0.24 * context_covariance_stability
        + 0.22 * bias_gate_transport
        + 0.22 * route_conditioned_projection
        + 0.16 * context_shift_resilience
        + 0.16 * projection_counterexample_resistance
    )

    return {
        "headline_metrics": {
            "context_covariance_stability": context_covariance_stability,
            "bias_gate_transport": bias_gate_transport,
            "route_conditioned_projection": route_conditioned_projection,
            "context_shift_resilience": context_shift_resilience,
            "projection_counterexample_resistance": projection_counterexample_resistance,
            "projection_gap": projection_gap,
            "language_projection_repair_score": language_projection_repair_score,
        },
        "scenario_records": scenario_records,
        "language_projection_equation": {
            "context_update": "q_plus = clip(0.36*q + 0.24*b + 0.18*C_ctx + 0.10*ctx_push + 0.12*(1-hidden_proxy_gap), 0, 1)",
            "bias_update": "b_plus = clip(0.52*b + 0.22*q + 0.12*g + 0.08*ctx_push + 0.06*gate_stability, 0, 1)",
            "route_update": "g_plus = clip(0.34*g + 0.24*q + 0.18*A_ctx + 0.10*route_pull + 0.08*gate_stability + 0.06*fiber_balance, 0, 1)",
            "projection": "Y_lang_plus = 0.34*q_plus + 0.28*g_plus + 0.22*b_plus + 0.16*A_ctx",
        },
        "status": {
            "status_short": (
                "language_projection_covariance_repaired"
                if language_projection_repair_score >= 0.79
                else "language_projection_covariance_transition"
            ),
            "status_label": "语言投影开始具备上下文协变稳定性，但仍需在更强反例下继续确认",
        },
        "project_readout": {
            "summary": "这一轮专门把 q/b/g 的上下文协变链拆成可测的协变稳定、偏置传输、路由投影与反例抗性四部分，补强 Stage71 里最弱的语言投影轴。",
            "next_question": "下一步要把这个语言投影协变块并回统一框架，并设计更强上下文冲突反例来检验协变关系是否还能守住。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    status = summary["status"]
    lines = [
        "# Stage72 Language Projection Covariance",
        "",
        f"- context_covariance_stability: {hm['context_covariance_stability']:.6f}",
        f"- bias_gate_transport: {hm['bias_gate_transport']:.6f}",
        f"- route_conditioned_projection: {hm['route_conditioned_projection']:.6f}",
        f"- context_shift_resilience: {hm['context_shift_resilience']:.6f}",
        f"- projection_counterexample_resistance: {hm['projection_counterexample_resistance']:.6f}",
        f"- projection_gap: {hm['projection_gap']:.6f}",
        f"- language_projection_repair_score: {hm['language_projection_repair_score']:.6f}",
        f"- status_short: {status['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_language_projection_covariance_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
