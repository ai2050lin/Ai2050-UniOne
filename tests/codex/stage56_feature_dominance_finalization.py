from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_finalization_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_dominance_finalization_summary() -> dict:
    reinforce = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_reinforcement_20260320" / "summary.json"
    )
    circuit_v3 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_terminal_measure_20260320" / "summary.json"
    )

    hr = reinforce["headline_metrics"]
    hc = circuit_v3["headline_metrics"]

    final_gain = hr["reinforced_gain"] + hr["reinforced_margin"]
    final_gap = 0.75 * hr["reinforced_gap"] + 0.25 * hc["direct_gate_v3"]
    final_margin = final_gain - final_gap
    final_ratio = final_gain / max(final_gap, 1e-9)

    return {
        "headline_metrics": {
            "final_gain": final_gain,
            "final_gap": final_gap,
            "final_margin": final_margin,
            "final_ratio": final_ratio,
        },
        "finalization_equation": {
            "gain_term": "G_final = reinforced_gain + reinforced_margin",
            "gap_term": "P_final = 0.75 * reinforced_gap + 0.25 * direct_gate_v3",
            "margin_term": "M_final = G_final - P_final",
            "ratio_term": "R_final = G_final / P_final",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征提取主导性定型报告",
        "",
        f"- final_gain: {hm['final_gain']:.6f}",
        f"- final_gap: {hm['final_gap']:.6f}",
        f"- final_margin: {hm['final_margin']:.6f}",
        f"- final_ratio: {hm['final_ratio']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_dominance_finalization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
