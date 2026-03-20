from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_final_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_learning_dynamics_terminal_final_summary() -> dict:
    closure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_closure_20260320" / "summary.json"
    )
    circuit_v4 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_closure_v4_20260320" / "summary.json"
    )
    final_feature = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_finalization_20260320" / "summary.json"
    )

    hc = closure["headline_metrics"]
    hd = circuit_v4["headline_metrics"]
    hf = final_feature["headline_metrics"]

    final_seed = hc["closure_seed"] / (1.0 + hd["direct_gate_v4"])
    final_feature_term = hc["closure_feature"] + hf["final_margin"]
    final_structure = hc["closure_structure"] + hd["direct_margin_v4"]
    final_global = final_seed + final_feature_term + final_structure

    return {
        "headline_metrics": {
            "final_seed": final_seed,
            "final_feature": final_feature_term,
            "final_structure": final_structure,
            "final_global": final_global,
        },
        "final_equation": {
            "seed_term": "F_seed = closure_seed / (1 + direct_gate_v4)",
            "feature_term": "F_feature = closure_feature + final_margin",
            "structure_term": "F_structure = closure_structure + direct_margin_v4",
            "global_term": "F_global = F_seed + F_feature + F_structure",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 连续学习动力学终式最终版报告",
        "",
        f"- final_seed: {hm['final_seed']:.6f}",
        f"- final_feature: {hm['final_feature']:.6f}",
        f"- final_structure: {hm['final_structure']:.6f}",
        f"- final_global: {hm['final_global']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_learning_dynamics_terminal_final_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
