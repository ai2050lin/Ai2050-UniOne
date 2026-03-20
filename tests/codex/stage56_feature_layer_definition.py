from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_layer_definition_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_layer_definition_summary() -> dict:
    absolute_lock = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_absolute_lock_20260320" / "summary.json"
    )
    feature_primary = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_primary_structure_20260320" / "summary.json"
    )
    native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_feature_native_variables_20260320" / "summary.json"
    )

    ha = absolute_lock["headline_metrics"]
    hp = feature_primary["headline_metrics"]
    hn = native["headline_metrics"]

    feature_basis = hn["native_feature"] + hn["native_selectivity"]
    feature_separation = hp["feature_primary_ratio"]
    feature_lock = ha["absolute_margin"]
    feature_layer_core = feature_basis + feature_separation + feature_lock

    return {
        "headline_metrics": {
            "feature_basis": feature_basis,
            "feature_separation": feature_separation,
            "feature_lock": feature_lock,
            "feature_layer_core": feature_layer_core,
        },
        "definition_equation": {
            "basis_term": "F_basis = native_feature + native_selectivity",
            "separation_term": "F_sep = feature_primary_ratio",
            "lock_term": "F_lock = absolute_margin",
            "core_term": "F_core = F_basis + F_sep + F_lock",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征层定义报告",
        "",
        f"- feature_basis: {hm['feature_basis']:.6f}",
        f"- feature_separation: {hm['feature_separation']:.6f}",
        f"- feature_lock: {hm['feature_lock']:.6f}",
        f"- feature_layer_core: {hm['feature_layer_core']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_layer_definition_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
