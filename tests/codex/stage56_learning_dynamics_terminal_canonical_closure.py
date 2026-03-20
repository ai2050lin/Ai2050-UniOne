from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_canonical_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_learning_dynamics_terminal_canonical_closure_summary() -> dict:
    closure_v2 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_final_closure_20260320" / "summary.json"
    )
    circuit_v8 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v8_20260320" / "summary.json"
    )
    absolute_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_absolute_lock_20260320" / "summary.json"
    )

    hc = closure_v2["headline_metrics"]
    hd = circuit_v8["headline_metrics"]
    ha = absolute_lock["headline_metrics"]

    canonical_seed = hc["closure_seed_v2"] / (1.0 + hd["direct_gate_v8"])
    canonical_feature = hc["closure_feature_v2"] + ha["absolute_margin"]
    canonical_structure = hc["closure_structure_v2"] + hd["direct_margin_v8"]
    canonical_global = canonical_seed + canonical_feature + canonical_structure

    return {
        "headline_metrics": {
            "canonical_seed": canonical_seed,
            "canonical_feature": canonical_feature,
            "canonical_structure": canonical_structure,
            "canonical_global": canonical_global,
        },
        "canonical_equation": {
            "seed_term": "Q_seed = closure_seed_v2 / (1 + direct_gate_v8)",
            "feature_term": "Q_feature = closure_feature_v2 + absolute_margin",
            "structure_term": "Q_structure = closure_structure_v2 + direct_margin_v8",
            "global_term": "Q_global = Q_seed + Q_feature + Q_structure",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 连续学习动力学规范闭合报告",
        "",
        f"- canonical_seed: {hm['canonical_seed']:.6f}",
        f"- canonical_feature: {hm['canonical_feature']:.6f}",
        f"- canonical_structure: {hm['canonical_structure']:.6f}",
        f"- canonical_global: {hm['canonical_global']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_learning_dynamics_terminal_canonical_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
