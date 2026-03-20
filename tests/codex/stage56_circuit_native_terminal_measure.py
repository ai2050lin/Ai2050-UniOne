from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_circuit_native_terminal_measure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_circuit_native_terminal_measure_summary() -> dict:
    circuit_v2 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_refinement_v2_20260320" / "summary.json"
    )
    terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_form_20260320" / "summary.json"
    )
    dominance = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_primary_dominance_20260320" / "summary.json"
    )

    hc = circuit_v2["headline_metrics"]
    ht = terminal["headline_metrics"]
    hd = dominance["headline_metrics"]

    direct_binding_v3 = hc["direct_binding_v2"] + 0.15 * ht["terminal_feature"]
    direct_gate_v3 = hc["direct_gate_v2"] / (1.0 + 0.1 * hd["dominance_ratio"])
    direct_attractor_v3 = hc["direct_attractor_v2"] + 0.15 * ht["terminal_structure"]
    direct_margin_v3 = direct_binding_v3 + direct_attractor_v3 - direct_gate_v3

    return {
        "headline_metrics": {
            "direct_binding_v3": direct_binding_v3,
            "direct_gate_v3": direct_gate_v3,
            "direct_attractor_v3": direct_attractor_v3,
            "direct_margin_v3": direct_margin_v3,
        },
        "terminal_measure_equation": {
            "binding_term": "B_direct_v3 = direct_binding_v2 + 0.15 * terminal_feature",
            "gate_term": "G_direct_v3 = direct_gate_v2 / (1 + 0.1 * dominance_ratio)",
            "attractor_term": "A_direct_v3 = direct_attractor_v2 + 0.15 * terminal_structure",
            "margin_term": "M_direct_v3 = B_direct_v3 + A_direct_v3 - G_direct_v3",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 回路级终式直测报告",
        "",
        f"- direct_binding_v3: {hm['direct_binding_v3']:.6f}",
        f"- direct_gate_v3: {hm['direct_gate_v3']:.6f}",
        f"- direct_attractor_v3: {hm['direct_attractor_v3']:.6f}",
        f"- direct_margin_v3: {hm['direct_margin_v3']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_circuit_native_terminal_measure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
