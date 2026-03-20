from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_closure_v4_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_circuit_direct_closure_v4_summary() -> dict:
    circuit_v3 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_terminal_measure_20260320" / "summary.json"
    )
    closure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_closure_20260320" / "summary.json"
    )
    final_feature = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_finalization_20260320" / "summary.json"
    )

    hc = circuit_v3["headline_metrics"]
    hl = closure["headline_metrics"]
    hf = final_feature["headline_metrics"]

    direct_binding_v4 = hc["direct_binding_v3"] + 0.1 * hl["closure_feature"]
    direct_gate_v4 = hc["direct_gate_v3"] / (1.0 + 0.05 * hf["final_ratio"])
    direct_attractor_v4 = hc["direct_attractor_v3"] + 0.1 * hl["closure_structure"]
    direct_margin_v4 = direct_binding_v4 + direct_attractor_v4 - direct_gate_v4

    return {
        "headline_metrics": {
            "direct_binding_v4": direct_binding_v4,
            "direct_gate_v4": direct_gate_v4,
            "direct_attractor_v4": direct_attractor_v4,
            "direct_margin_v4": direct_margin_v4,
        },
        "closure_equation": {
            "binding_term": "B_direct_v4 = direct_binding_v3 + 0.1 * closure_feature",
            "gate_term": "G_direct_v4 = direct_gate_v3 / (1 + 0.05 * final_ratio)",
            "attractor_term": "A_direct_v4 = direct_attractor_v3 + 0.1 * closure_structure",
            "margin_term": "M_direct_v4 = B_direct_v4 + A_direct_v4 - G_direct_v4",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 回路级终式收口第四版报告",
        "",
        f"- direct_binding_v4: {hm['direct_binding_v4']:.6f}",
        f"- direct_gate_v4: {hm['direct_gate_v4']:.6f}",
        f"- direct_attractor_v4: {hm['direct_attractor_v4']:.6f}",
        f"- direct_margin_v4: {hm['direct_margin_v4']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_circuit_direct_closure_v4_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
