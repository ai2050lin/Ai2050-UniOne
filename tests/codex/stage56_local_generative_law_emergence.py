from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_local_generative_law_emergence_20260321"


def build_local_generative_law_emergence_summary() -> dict:
    # 六节点局部场，包含三组局部近邻与两条跨区桥。
    patch_activation = [0.82, 0.78, 0.74, 0.31, 0.27, 0.22]
    recurrence = [0.79, 0.76, 0.73, 0.34, 0.30, 0.26]
    route_cost = [0.18, 0.21, 0.24, 0.49, 0.53, 0.57]
    route_gate = [0.77, 0.74, 0.71, 0.38, 0.34, 0.31]
    pressure = [0.28, 0.30, 0.27, 0.43, 0.45, 0.47]
    plasticity_budget = [0.62, 0.59, 0.57, 0.44, 0.41, 0.39]

    neighbors = {
        0: (1, 2),
        1: (0, 2),
        2: (0, 1, 3),
        3: (2, 4, 5),
        4: (3, 5),
        5: (3, 4),
    }
    fiber_edges = ((0, 3), (1, 4), (2, 5))

    next_patch = []
    next_gate = []
    next_pressure = []
    for idx in range(len(patch_activation)):
        local_mean = sum(patch_activation[n] for n in neighbors[idx]) / len(neighbors[idx])
        recurrence_pull = 0.55 * recurrence[idx] + 0.25 * local_mean
        inhibition = 0.20 * pressure[idx]
        update = max(0.0, min(1.0, recurrence_pull + 0.12 * plasticity_budget[idx] - inhibition))
        next_patch.append(update)

        gate_drive = 0.46 * patch_activation[idx] + 0.24 * recurrence[idx] + 0.18 * plasticity_budget[idx]
        gate_drag = 0.28 * route_cost[idx] + 0.16 * pressure[idx]
        next_gate.append(max(0.0, min(1.0, gate_drive - gate_drag + 0.08)))

        pressure_decay = 0.72 * pressure[idx]
        pressure_load = 0.18 * (1.0 - plasticity_budget[idx]) + 0.10 * route_cost[idx]
        next_pressure.append(max(0.0, min(1.0, pressure_decay + pressure_load)))

    patch_means = (sum(next_patch[:3]) / 3.0, sum(next_patch[3:]) / 3.0)
    intra_patch_gap = abs(patch_means[0] - patch_means[1])
    local_variance = sum(abs(next_patch[i] - next_patch[i + 1]) for i in range(2)) / 2.0
    patch_coherence = max(0.0, min(1.0, 0.75 * intra_patch_gap + 0.25 * (1.0 - local_variance)))

    fiber_values = []
    for left, right in fiber_edges:
        reuse = 0.55 * min(next_gate[left], next_gate[right]) + 0.25 * min(next_patch[left], next_patch[right])
        reuse += 0.20 * (1.0 - abs(next_pressure[left] - next_pressure[right]))
        fiber_values.append(max(0.0, min(1.0, reuse)))
    fiber_reuse = sum(fiber_values) / len(fiber_values)

    route_separation = max(
        0.0,
        min(
            1.0,
            0.50 * ((sum(next_gate[:3]) / 3.0) - (sum(next_gate[3:]) / 3.0) + 0.5)
            + 0.50 * (1.0 - sum(next_pressure) / len(next_pressure)),
        ),
    )
    pressure_balance = max(0.0, min(1.0, 1.0 - (max(next_pressure) - min(next_pressure))))

    local_law_emergence_score = max(
        0.0,
        min(
            1.0,
            0.32 * patch_coherence
            + 0.24 * fiber_reuse
            + 0.24 * route_separation
            + 0.20 * pressure_balance,
        ),
    )
    derivability_score = max(
        0.0,
        min(1.0, 0.45 * local_law_emergence_score + 0.30 * fiber_reuse + 0.25 * pressure_balance),
    )

    return {
        "headline_metrics": {
            "patch_coherence": patch_coherence,
            "fiber_reuse": fiber_reuse,
            "route_separation": route_separation,
            "pressure_balance": pressure_balance,
            "local_law_emergence_score": local_law_emergence_score,
            "derivability_score": derivability_score,
        },
        "local_law_system": {
            "patch_update": "a'(x) = clip(0.55*r(x) + 0.25*mean_neighbor(a) + 0.12*p(x) - 0.20*h(x), 0, 1)",
            "route_update": "g'(x) = clip(0.46*a(x) + 0.24*r(x) + 0.18*p(x) - 0.28*c(x) - 0.16*h(x) + 0.08, 0, 1)",
            "pressure_update": "h'(x) = clip(0.72*h(x) + 0.18*(1-p(x)) + 0.10*c(x), 0, 1)",
            "fiber_measure": "f(i,j) = 0.55*min(g_i,g_j) + 0.25*min(a_i,a_j) + 0.20*(1-|h_i-h_j|)",
        },
        "project_readout": {
            "summary": "Local generative law emergence checks whether patch, fiber, and route objects can arise from local update rules instead of being directly named.",
            "next_question": "Next bind the weakest context primitive into the same local-law system and test whether conditional routing also emerges without manual naming.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Local Generative Law Emergence",
        "",
        f"- patch_coherence: {hm['patch_coherence']:.6f}",
        f"- fiber_reuse: {hm['fiber_reuse']:.6f}",
        f"- route_separation: {hm['route_separation']:.6f}",
        f"- pressure_balance: {hm['pressure_balance']:.6f}",
        f"- local_law_emergence_score: {hm['local_law_emergence_score']:.6f}",
        f"- derivability_score: {hm['derivability_score']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_local_generative_law_emergence_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
