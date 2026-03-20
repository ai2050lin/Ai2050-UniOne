from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_learning_dynamics_terminal_closure_summary() -> dict:
    terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_form_20260320" / "summary.json"
    )
    circuit_v3 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_terminal_measure_20260320" / "summary.json"
    )
    reinforce = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_reinforcement_20260320" / "summary.json"
    )

    ht = terminal["headline_metrics"]
    hc = circuit_v3["headline_metrics"]
    hr = reinforce["headline_metrics"]

    closure_seed = ht["terminal_seed"] / (1.0 + hc["direct_gate_v3"])
    closure_feature = ht["terminal_feature"] + hr["reinforced_margin"]
    closure_structure = ht["terminal_structure"] + hc["direct_margin_v3"]
    closure_global = closure_seed + closure_feature + closure_structure

    return {
        "headline_metrics": {
            "closure_seed": closure_seed,
            "closure_feature": closure_feature,
            "closure_structure": closure_structure,
            "closure_global": closure_global,
        },
        "terminal_closure_equation": {
            "seed_term": "C_seed = terminal_seed / (1 + direct_gate_v3)",
            "feature_term": "C_feature = terminal_feature + reinforced_margin",
            "structure_term": "C_structure = terminal_structure + direct_margin_v3",
            "global_term": "C_global = C_seed + C_feature + C_structure",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 连续学习动力学终式收口报告",
        "",
        f"- closure_seed: {hm['closure_seed']:.6f}",
        f"- closure_feature: {hm['closure_feature']:.6f}",
        f"- closure_structure: {hm['closure_structure']:.6f}",
        f"- closure_global: {hm['closure_global']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_learning_dynamics_terminal_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
