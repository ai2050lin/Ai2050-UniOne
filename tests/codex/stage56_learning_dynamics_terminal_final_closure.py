from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_final_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_learning_dynamics_terminal_final_closure_summary() -> dict:
    irrev_learning = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_irreversible_20260320" / "summary.json"
    )
    circuit_v7 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v7_20260320" / "summary.json"
    )
    feature_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_irreversible_lock_20260320" / "summary.json"
    )

    hl = irrev_learning["headline_metrics"]
    hc = circuit_v7["headline_metrics"]
    hf = feature_lock["headline_metrics"]

    closure_seed_v2 = hl["irreversible_seed"] / (1.0 + hc["direct_gate_v7"])
    closure_feature_v2 = hl["irreversible_feature"] + hf["lock_margin"]
    closure_structure_v2 = hl["irreversible_structure"] + hc["direct_margin_v7"]
    closure_global_v2 = closure_seed_v2 + closure_feature_v2 + closure_structure_v2

    return {
        "headline_metrics": {
            "closure_seed_v2": closure_seed_v2,
            "closure_feature_v2": closure_feature_v2,
            "closure_structure_v2": closure_structure_v2,
            "closure_global_v2": closure_global_v2,
        },
        "final_closure_equation": {
            "seed_term": "C2_seed = irreversible_seed / (1 + direct_gate_v7)",
            "feature_term": "C2_feature = irreversible_feature + lock_margin",
            "structure_term": "C2_structure = irreversible_structure + direct_margin_v7",
            "global_term": "C2_global = C2_seed + C2_feature + C2_structure",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 连续学习动力学最终闭合报告",
        "",
        f"- closure_seed_v2: {hm['closure_seed_v2']:.6f}",
        f"- closure_feature_v2: {hm['closure_feature_v2']:.6f}",
        f"- closure_structure_v2: {hm['closure_structure_v2']:.6f}",
        f"- closure_global_v2: {hm['closure_global_v2']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_learning_dynamics_terminal_final_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
