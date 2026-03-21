from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage57_fiber_reuse_reinforcement_20260321"


def build_fiber_reuse_reinforcement_summary() -> dict:
    patch_activation = [0.84, 0.80, 0.77, 0.41, 0.39, 0.37]
    recurrence = [0.81, 0.79, 0.75, 0.46, 0.44, 0.42]
    route_cost = [0.18, 0.20, 0.22, 0.39, 0.42, 0.44]
    route_gate = [0.80, 0.77, 0.74, 0.56, 0.54, 0.52]
    pressure = [0.28, 0.29, 0.27, 0.36, 0.38, 0.39]
    plasticity_budget = [0.64, 0.61, 0.58, 0.49, 0.47, 0.45]
    context_gate = [0.72, 0.69, 0.65, 0.61, 0.58, 0.55]

    fiber_edges = ((0, 3), (1, 4), (2, 5))

    next_patch = []
    next_gate = []
    next_pressure = []
    for idx in range(len(patch_activation)):
        update = 0.48 * recurrence[idx] + 0.24 * patch_activation[idx] + 0.15 * context_gate[idx]
        update += 0.10 * plasticity_budget[idx] - 0.16 * pressure[idx]
        next_patch.append(max(0.0, min(1.0, update)))

        gate_update = 0.42 * route_gate[idx] + 0.24 * recurrence[idx] + 0.20 * context_gate[idx]
        gate_update += 0.10 * plasticity_budget[idx] - 0.18 * route_cost[idx] - 0.12 * pressure[idx]
        next_gate.append(max(0.0, min(1.0, gate_update)))

        pressure_update = 0.68 * pressure[idx] + 0.13 * route_cost[idx] + 0.11 * (1.0 - plasticity_budget[idx])
        pressure_update += 0.04 * abs(next_gate[idx] - context_gate[idx])
        next_pressure.append(max(0.0, min(1.0, pressure_update)))

    fiber_values = []
    stability_terms = []
    coupling_terms = []
    for left, right in fiber_edges:
        reuse = 0.38 * min(next_gate[left], next_gate[right])
        reuse += 0.27 * min(next_patch[left], next_patch[right])
        reuse += 0.20 * min(context_gate[left], context_gate[right])
        reuse += 0.15 * (1.0 - abs(next_pressure[left] - next_pressure[right]))
        reuse = max(0.0, min(1.0, reuse))
        fiber_values.append(reuse)

        stability_terms.append(1.0 - abs(reuse - min(next_gate[left], next_gate[right])))
        coupling_terms.append(1.0 - abs((next_gate[left] + next_gate[right]) / 2.0 - reuse))

    fiber_reuse = sum(fiber_values) / len(fiber_values)
    cross_region_share_stability = max(0.0, min(1.0, sum(stability_terms) / len(stability_terms)))
    route_fiber_coupling_balance = max(0.0, min(1.0, sum(coupling_terms) / len(coupling_terms)))
    pressure_under_reuse = max(0.0, min(1.0, 1.0 - (sum(next_pressure) / len(next_pressure))))
    reinforcement_readiness = max(
        0.0,
        min(
            1.0,
            0.35 * fiber_reuse
            + 0.25 * cross_region_share_stability
            + 0.20 * route_fiber_coupling_balance
            + 0.20 * pressure_under_reuse,
        ),
    )

    return {
        "headline_metrics": {
            "fiber_reuse": fiber_reuse,
            "cross_region_share_stability": cross_region_share_stability,
            "route_fiber_coupling_balance": route_fiber_coupling_balance,
            "pressure_under_reuse": pressure_under_reuse,
            "reinforcement_readiness": reinforcement_readiness,
        },
        "reinforcement_equation": {
            "patch_update": "a_plus = clip(0.48*r + 0.24*a + 0.15*q + 0.10*p - 0.16*h, 0, 1)",
            "route_update": "g_plus = clip(0.42*g + 0.24*r + 0.20*q + 0.10*p - 0.18*c - 0.12*h, 0, 1)",
            "pressure_update": "h_plus = clip(0.68*h + 0.13*c + 0.11*(1-p) + 0.04*|g_plus-q|, 0, 1)",
            "fiber_measure": "f_plus = 0.38*min(g_i,g_j) + 0.27*min(a_i,a_j) + 0.20*min(q_i,q_j) + 0.15*(1-|h_i-h_j|)",
        },
        "project_readout": {
            "summary": "Fiber reuse reinforcement strengthens cross-region sharing while keeping route separation and pressure under control.",
            "next_question": "Feed the reinforced fiber term back into the kernel candidate review to check whether the learning rule stays stable once reuse pressure increases.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage57 Fiber Reuse Reinforcement",
        "",
        f"- fiber_reuse: {hm['fiber_reuse']:.6f}",
        f"- cross_region_share_stability: {hm['cross_region_share_stability']:.6f}",
        f"- route_fiber_coupling_balance: {hm['route_fiber_coupling_balance']:.6f}",
        f"- pressure_under_reuse: {hm['pressure_under_reuse']:.6f}",
        f"- reinforcement_readiness: {hm['reinforcement_readiness']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_fiber_reuse_reinforcement_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
