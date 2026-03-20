from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_irreversible_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_learning_dynamics_terminal_irreversible_summary() -> dict:
    learning_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_lock_20260320" / "summary.json"
    )
    circuit_v6 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v6_20260320" / "summary.json"
    )
    irreversibility = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_irreversibility_20260320" / "summary.json"
    )

    hl = learning_lock["headline_metrics"]
    hc = circuit_v6["headline_metrics"]
    hi = irreversibility["headline_metrics"]

    irreversible_seed = hl["locked_seed"] / (1.0 + hc["direct_gate_v6"])
    irreversible_feature = hl["locked_feature"] + hi["irreversible_margin"]
    irreversible_structure = hl["locked_structure"] + hc["direct_margin_v6"]
    irreversible_global = irreversible_seed + irreversible_feature + irreversible_structure

    return {
        "headline_metrics": {
            "irreversible_seed": irreversible_seed,
            "irreversible_feature": irreversible_feature,
            "irreversible_structure": irreversible_structure,
            "irreversible_global": irreversible_global,
        },
        "irreversible_equation": {
            "seed_term": "I_seed = locked_seed / (1 + direct_gate_v6)",
            "feature_term": "I_feature = locked_feature + irreversible_margin",
            "structure_term": "I_structure = locked_structure + direct_margin_v6",
            "global_term": "I_global = I_seed + I_feature + I_structure",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 连续学习动力学不可逆版报告",
        "",
        f"- irreversible_seed: {hm['irreversible_seed']:.6f}",
        f"- irreversible_feature: {hm['irreversible_feature']:.6f}",
        f"- irreversible_structure: {hm['irreversible_structure']:.6f}",
        f"- irreversible_global: {hm['irreversible_global']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_learning_dynamics_terminal_irreversible_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
