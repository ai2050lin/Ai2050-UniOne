from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_primary_dominance_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_primary_dominance_summary() -> dict:
    threshold = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_primary_threshold_closure_20260320" / "summary.json"
    )
    feature = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_primary_structure_20260320" / "summary.json"
    )
    native = _load_json(ROOT / "tests" / "codex_temp" / "stage56_spike_feature_native_variables_20260320" / "summary.json")

    ht = threshold["headline_metrics"]
    hf = feature["headline_metrics"]
    hn = native["headline_metrics"]

    dominance_gain = ht["primary_threshold_margin"] + hn["native_feature"] + hn["native_selectivity"]
    dominance_gap = 0.25 * hf["primary_feature_core"]
    dominance_margin = dominance_gain - dominance_gap
    dominance_ratio = dominance_gain / max(dominance_gap, 1e-9)

    return {
        "headline_metrics": {
            "dominance_gain": dominance_gain,
            "dominance_gap": dominance_gap,
            "dominance_margin": dominance_margin,
            "dominance_ratio": dominance_ratio,
        },
        "dominance_equation": {
            "gain_term": "G_dom = primary_threshold_margin + native_feature + native_selectivity",
            "gap_term": "P_dom = 0.25 * primary_feature_core",
            "margin_term": "M_dom = G_dom - P_dom",
            "ratio_term": "R_dom = G_dom / P_dom",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征提取压倒性主结构报告",
        "",
        f"- dominance_gain: {hm['dominance_gain']:.6f}",
        f"- dominance_gap: {hm['dominance_gap']:.6f}",
        f"- dominance_margin: {hm['dominance_margin']:.6f}",
        f"- dominance_ratio: {hm['dominance_ratio']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_primary_dominance_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
