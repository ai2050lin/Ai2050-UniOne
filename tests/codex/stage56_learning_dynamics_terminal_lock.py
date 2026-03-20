from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_lock_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_learning_dynamics_terminal_lock_summary() -> dict:
    learning_final = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_final_20260320" / "summary.json"
    )
    circuit_v5 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_terminal_lock_20260320" / "summary.json"
    )
    feature_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_locking_20260320" / "summary.json"
    )

    hl = learning_final["headline_metrics"]
    hc = circuit_v5["headline_metrics"]
    hf = feature_lock["headline_metrics"]

    locked_seed = hl["final_seed"] / (1.0 + hc["direct_gate_v5"])
    locked_feature = hl["final_feature"] + hf["locking_margin"]
    locked_structure = hl["final_structure"] + hc["direct_margin_v5"]
    locked_global = locked_seed + locked_feature + locked_structure

    return {
        "headline_metrics": {
            "locked_seed": locked_seed,
            "locked_feature": locked_feature,
            "locked_structure": locked_structure,
            "locked_global": locked_global,
        },
        "locking_equation": {
            "seed_term": "L_seed = final_seed / (1 + direct_gate_v5)",
            "feature_term": "L_feature = final_feature + locking_margin",
            "structure_term": "L_structure = final_structure + direct_margin_v5",
            "global_term": "L_global = L_seed + L_feature + L_structure",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 连续学习动力学锁定报告",
        "",
        f"- locked_seed: {hm['locked_seed']:.6f}",
        f"- locked_feature: {hm['locked_feature']:.6f}",
        f"- locked_structure: {hm['locked_structure']:.6f}",
        f"- locked_global: {hm['locked_global']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_learning_dynamics_terminal_lock_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
