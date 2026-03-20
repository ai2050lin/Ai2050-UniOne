from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_primary_structure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_extraction_primary_structure_summary() -> dict:
    balance = _load_json(ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_balance_refinement_20260320" / "summary.json")
    native = _load_json(ROOT / "tests" / "codex_temp" / "stage56_spike_feature_native_variables_20260320" / "summary.json")
    v7 = _load_json(ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v7_20260320" / "summary.json")

    hb = balance["headline_metrics"]
    hn = native["headline_metrics"]
    hv7 = v7["headline_metrics"]

    primary_feature_core = hb["balanced_feature_gain"] + hn["native_feature"] + hn["native_selectivity"]
    feature_structure_support = primary_feature_core / (1.0 + hv7["pressure_term_v7"])
    feature_primary_margin = feature_structure_support - hb["seed_normalized"] * 0.5
    feature_primary_ratio = primary_feature_core / max(hb["seed_normalized"], 1e-9)

    return {
        "headline_metrics": {
            "primary_feature_core": primary_feature_core,
            "feature_structure_support": feature_structure_support,
            "feature_primary_margin": feature_primary_margin,
            "feature_primary_ratio": feature_primary_ratio,
        },
        "primary_equation": {
            "core_term": "F_core = balanced_feature_gain + native_feature + native_selectivity",
            "support_term": "F_support = F_core / (1 + pressure_term_v7)",
            "margin_term": "M_feature_primary = F_support - 0.5 * seed_normalized",
            "ratio_term": "R_feature_primary = F_core / seed_normalized",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征提取主结构化报告",
        "",
        f"- primary_feature_core: {hm['primary_feature_core']:.6f}",
        f"- feature_structure_support: {hm['feature_structure_support']:.6f}",
        f"- feature_primary_margin: {hm['feature_primary_margin']:.6f}",
        f"- feature_primary_ratio: {hm['feature_primary_ratio']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_extraction_primary_structure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
