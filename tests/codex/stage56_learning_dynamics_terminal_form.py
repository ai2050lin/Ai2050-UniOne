from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_form_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_learning_dynamics_terminal_form_summary() -> dict:
    learning = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_pulse_to_network_learning_dynamics_20260320" / "summary.json"
    )
    direct = _load_json(ROOT / "tests" / "codex_temp" / "stage56_circuit_native_direct_measure_20260320" / "summary.json")
    threshold = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_primary_threshold_closure_20260320" / "summary.json"
    )

    hl = learning["headline_metrics"]
    hd = direct["headline_metrics"]
    ht = threshold["headline_metrics"]

    terminal_seed = hl["learning_seed"] / (1.0 + hd["direct_gate_measure"])
    terminal_feature = hl["learning_feature"] + ht["primary_threshold_margin"]
    terminal_structure = hl["learning_structure"] + hd["direct_attractor_measure"]
    terminal_global = terminal_seed + terminal_feature + terminal_structure

    return {
        "headline_metrics": {
            "terminal_seed": terminal_seed,
            "terminal_feature": terminal_feature,
            "terminal_structure": terminal_structure,
            "terminal_global": terminal_global,
        },
        "terminal_equation": {
            "seed_term": "T_seed = learning_seed / (1 + direct_gate_measure)",
            "feature_term": "T_feature = learning_feature + primary_threshold_margin",
            "structure_term": "T_structure = learning_structure + direct_attractor_measure",
            "global_term": "T_global = T_seed + T_feature + T_structure",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 连续学习动力学终式报告",
        "",
        f"- terminal_seed: {hm['terminal_seed']:.6f}",
        f"- terminal_feature: {hm['terminal_feature']:.6f}",
        f"- terminal_structure: {hm['terminal_structure']:.6f}",
        f"- terminal_global: {hm['terminal_global']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_learning_dynamics_terminal_form_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
