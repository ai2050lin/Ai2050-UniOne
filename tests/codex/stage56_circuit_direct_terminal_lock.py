from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_terminal_lock_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_circuit_direct_terminal_lock_summary() -> dict:
    circuit_v4 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_closure_v4_20260320" / "summary.json"
    )
    learning_final = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_final_20260320" / "summary.json"
    )
    feature_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_locking_20260320" / "summary.json"
    )

    hc = circuit_v4["headline_metrics"]
    hl = learning_final["headline_metrics"]
    hf = feature_lock["headline_metrics"]

    direct_binding_v5 = hc["direct_binding_v4"] + 0.08 * hl["final_feature"]
    direct_gate_v5 = hc["direct_gate_v4"] / (1.0 + 0.05 * hf["locking_ratio"])
    direct_attractor_v5 = hc["direct_attractor_v4"] + 0.08 * hl["final_structure"]
    direct_margin_v5 = direct_binding_v5 + direct_attractor_v5 - direct_gate_v5

    return {
        "headline_metrics": {
            "direct_binding_v5": direct_binding_v5,
            "direct_gate_v5": direct_gate_v5,
            "direct_attractor_v5": direct_attractor_v5,
            "direct_margin_v5": direct_margin_v5,
        },
        "terminal_lock_equation": {
            "binding_term": "B_direct_v5 = direct_binding_v4 + 0.08 * final_feature",
            "gate_term": "G_direct_v5 = direct_gate_v4 / (1 + 0.05 * locking_ratio)",
            "attractor_term": "A_direct_v5 = direct_attractor_v4 + 0.08 * final_structure",
            "margin_term": "M_direct_v5 = B_direct_v5 + A_direct_v5 - G_direct_v5",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 回路级终式锁定报告",
        "",
        f"- direct_binding_v5: {hm['direct_binding_v5']:.6f}",
        f"- direct_gate_v5: {hm['direct_gate_v5']:.6f}",
        f"- direct_attractor_v5: {hm['direct_attractor_v5']:.6f}",
        f"- direct_margin_v5: {hm['direct_margin_v5']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_circuit_direct_terminal_lock_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
