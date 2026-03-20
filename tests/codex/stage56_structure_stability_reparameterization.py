from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_structure_stability_reparameterization_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_structure_stability_reparameterization_summary() -> dict:
    stabilized = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_confidence_stabilization_20260320" / "summary.json"
    )
    terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_terminal_closure_20260320" / "summary.json"
    )

    hs = stabilized["headline_metrics"]
    ht = terminal["headline_metrics"]

    stability_intensity = hs["stabilized_margin"]
    stability_strength = stability_intensity / (1.0 + stability_intensity)
    closure_alignment = hs["stabilized_margin"] / (1.0 + ht["terminal_closure_margin_v3"])
    stability_balance = stability_strength * (1.0 + closure_alignment)

    return {
        "headline_metrics": {
            "stability_intensity": stability_intensity,
            "stability_strength": stability_strength,
            "closure_alignment": closure_alignment,
            "stability_balance": stability_balance,
        },
        "reparameterized_equation": {
            "intensity_term": "I_struct = M_struct_v4",
            "strength_term": "S_strength = I_struct / (1 + I_struct)",
            "alignment_term": "A_closure = I_struct / (1 + Tc_margin)",
            "balance_term": "B_struct = S_strength * (1 + A_closure)",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 结构稳定强度重参数化报告",
        "",
        f"- stability_intensity: {hm['stability_intensity']:.6f}",
        f"- stability_strength: {hm['stability_strength']:.6f}",
        f"- closure_alignment: {hm['closure_alignment']:.6f}",
        f"- stability_balance: {hm['stability_balance']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_structure_stability_reparameterization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
