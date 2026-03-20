from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v6_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_circuit_native_near_direct_v6_summary() -> dict:
    circuit_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_terminal_lock_20260320" / "summary.json"
    )
    learning_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_lock_20260320" / "summary.json"
    )
    irreversibility = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_irreversibility_20260320" / "summary.json"
    )

    hc = circuit_v5["headline_metrics"]
    hl = learning_lock["headline_metrics"]
    hi = irreversibility["headline_metrics"]

    direct_binding_v6 = hc["direct_binding_v5"] + 0.06 * hl["locked_feature"]
    direct_gate_v6 = hc["direct_gate_v5"] / (1.0 + 0.04 * hi["irreversible_ratio"])
    direct_attractor_v6 = hc["direct_attractor_v5"] + 0.06 * hl["locked_structure"]
    direct_margin_v6 = direct_binding_v6 + direct_attractor_v6 - direct_gate_v6

    return {
        "headline_metrics": {
            "direct_binding_v6": direct_binding_v6,
            "direct_gate_v6": direct_gate_v6,
            "direct_attractor_v6": direct_attractor_v6,
            "direct_margin_v6": direct_margin_v6,
        },
        "near_direct_equation": {
            "binding_term": "B_direct_v6 = direct_binding_v5 + 0.06 * locked_feature",
            "gate_term": "G_direct_v6 = direct_gate_v5 / (1 + 0.04 * irreversible_ratio)",
            "attractor_term": "A_direct_v6 = direct_attractor_v5 + 0.06 * locked_structure",
            "margin_term": "M_direct_v6 = B_direct_v6 + A_direct_v6 - G_direct_v6",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 回路级近直测第六版报告",
        "",
        f"- direct_binding_v6: {hm['direct_binding_v6']:.6f}",
        f"- direct_gate_v6: {hm['direct_gate_v6']:.6f}",
        f"- direct_attractor_v6: {hm['direct_attractor_v6']:.6f}",
        f"- direct_margin_v6: {hm['direct_margin_v6']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_circuit_native_near_direct_v6_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
