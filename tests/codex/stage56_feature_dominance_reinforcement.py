from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_reinforcement_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_dominance_reinforcement_summary() -> dict:
    dominance = _load_json(ROOT / "tests" / "codex_temp" / "stage56_feature_primary_dominance_20260320" / "summary.json")
    threshold = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_primary_threshold_closure_20260320" / "summary.json"
    )
    circuit_v2 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_direct_refinement_v2_20260320" / "summary.json"
    )

    hd = dominance["headline_metrics"]
    ht = threshold["headline_metrics"]
    hc = circuit_v2["headline_metrics"]

    reinforced_gain = hd["dominance_gain"] + ht["threshold_lift"]
    reinforced_gap = 0.8 * hd["dominance_gap"] + 0.2 * hc["direct_gate_v2"]
    reinforced_margin = reinforced_gain - reinforced_gap
    reinforced_ratio = reinforced_gain / max(reinforced_gap, 1e-9)

    return {
        "headline_metrics": {
            "reinforced_gain": reinforced_gain,
            "reinforced_gap": reinforced_gap,
            "reinforced_margin": reinforced_margin,
            "reinforced_ratio": reinforced_ratio,
        },
        "reinforcement_equation": {
            "gain_term": "G_reinforce = dominance_gain + threshold_lift",
            "gap_term": "P_reinforce = 0.8 * dominance_gap + 0.2 * direct_gate_v2",
            "margin_term": "M_reinforce = G_reinforce - P_reinforce",
            "ratio_term": "R_reinforce = G_reinforce / P_reinforce",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征提取压倒性优势强化报告",
        "",
        f"- reinforced_gain: {hm['reinforced_gain']:.6f}",
        f"- reinforced_gap: {hm['reinforced_gap']:.6f}",
        f"- reinforced_margin: {hm['reinforced_margin']:.6f}",
        f"- reinforced_ratio: {hm['reinforced_ratio']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_dominance_reinforcement_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
