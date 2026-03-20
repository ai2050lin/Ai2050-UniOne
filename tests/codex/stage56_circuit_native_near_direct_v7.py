from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v7_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_circuit_native_near_direct_v7_summary() -> dict:
    circuit_v6 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v6_20260320" / "summary.json"
    )
    irrev_learning = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_irreversible_20260320" / "summary.json"
    )
    feature_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_irreversible_lock_20260320" / "summary.json"
    )

    hc = circuit_v6["headline_metrics"]
    hl = irrev_learning["headline_metrics"]
    hf = feature_lock["headline_metrics"]

    direct_binding_v7 = hc["direct_binding_v6"] + 0.05 * hl["irreversible_feature"]
    direct_gate_v7 = hc["direct_gate_v6"] / (1.0 + 0.03 * hf["lock_ratio"])
    direct_attractor_v7 = hc["direct_attractor_v6"] + 0.05 * hl["irreversible_structure"]
    direct_margin_v7 = direct_binding_v7 + direct_attractor_v7 - direct_gate_v7

    return {
        "headline_metrics": {
            "direct_binding_v7": direct_binding_v7,
            "direct_gate_v7": direct_gate_v7,
            "direct_attractor_v7": direct_attractor_v7,
            "direct_margin_v7": direct_margin_v7,
        },
        "near_direct_equation": {
            "binding_term": "B_direct_v7 = direct_binding_v6 + 0.05 * irreversible_feature",
            "gate_term": "G_direct_v7 = direct_gate_v6 / (1 + 0.03 * lock_ratio)",
            "attractor_term": "A_direct_v7 = direct_attractor_v6 + 0.05 * irreversible_structure",
            "margin_term": "M_direct_v7 = B_direct_v7 + A_direct_v7 - G_direct_v7",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 回路级近直测第七版报告",
        "",
        f"- direct_binding_v7: {hm['direct_binding_v7']:.6f}",
        f"- direct_gate_v7: {hm['direct_gate_v7']:.6f}",
        f"- direct_attractor_v7: {hm['direct_attractor_v7']:.6f}",
        f"- direct_margin_v7: {hm['direct_margin_v7']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_circuit_native_near_direct_v7_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
