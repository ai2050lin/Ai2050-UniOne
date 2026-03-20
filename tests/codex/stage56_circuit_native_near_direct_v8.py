from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v8_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_circuit_native_near_direct_v8_summary() -> dict:
    circuit_v7 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v7_20260320" / "summary.json"
    )
    closure_v2 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_final_closure_20260320" / "summary.json"
    )
    absolute_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_absolute_lock_20260320" / "summary.json"
    )

    hc = circuit_v7["headline_metrics"]
    hl = closure_v2["headline_metrics"]
    ha = absolute_lock["headline_metrics"]

    direct_binding_v8 = hc["direct_binding_v7"] + 0.04 * hl["closure_feature_v2"]
    direct_gate_v8 = hc["direct_gate_v7"] / (1.0 + 0.025 * ha["absolute_ratio"])
    direct_attractor_v8 = hc["direct_attractor_v7"] + 0.04 * hl["closure_structure_v2"]
    direct_margin_v8 = direct_binding_v8 + direct_attractor_v8 - direct_gate_v8

    return {
        "headline_metrics": {
            "direct_binding_v8": direct_binding_v8,
            "direct_gate_v8": direct_gate_v8,
            "direct_attractor_v8": direct_attractor_v8,
            "direct_margin_v8": direct_margin_v8,
        },
        "near_direct_equation": {
            "binding_term": "B_direct_v8 = direct_binding_v7 + 0.04 * closure_feature_v2",
            "gate_term": "G_direct_v8 = direct_gate_v7 / (1 + 0.025 * absolute_ratio)",
            "attractor_term": "A_direct_v8 = direct_attractor_v7 + 0.04 * closure_structure_v2",
            "margin_term": "M_direct_v8 = B_direct_v8 + A_direct_v8 - G_direct_v8",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 回路级近直测第八版报告",
        "",
        f"- direct_binding_v8: {hm['direct_binding_v8']:.6f}",
        f"- direct_gate_v8: {hm['direct_gate_v8']:.6f}",
        f"- direct_attractor_v8: {hm['direct_attractor_v8']:.6f}",
        f"- direct_margin_v8: {hm['direct_margin_v8']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_circuit_native_near_direct_v8_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
