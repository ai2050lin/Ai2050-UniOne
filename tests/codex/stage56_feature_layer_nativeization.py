from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_layer_nativeization_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_layer_nativeization_summary() -> dict:
    feature_layer = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_definition_20260320" / "summary.json"
    )
    feature_primary = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_extraction_primary_structure_20260320" / "summary.json"
    )
    native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_feature_native_variables_20260320" / "summary.json"
    )

    hf = feature_layer["headline_metrics"]
    hp = feature_primary["headline_metrics"]
    hn = native["headline_metrics"]

    native_basis_v2 = hf["feature_basis"] + hp["feature_structure_support"]
    native_separation_v2 = hf["feature_separation"] * (1.0 + hn["native_selectivity"])
    native_lock_v2 = hf["feature_lock"] / (1.0 + hn["native_inhibition"])
    feature_native_core_v2 = native_basis_v2 + native_separation_v2 + native_lock_v2

    return {
        "headline_metrics": {
            "native_basis_v2": native_basis_v2,
            "native_separation_v2": native_separation_v2,
            "native_lock_v2": native_lock_v2,
            "feature_native_core_v2": feature_native_core_v2,
        },
        "nativeization_equation": {
            "basis_term": "F_basis_v2 = feature_basis + feature_structure_support",
            "separation_term": "F_sep_v2 = feature_separation * (1 + native_selectivity)",
            "lock_term": "F_lock_v2 = feature_lock / (1 + native_inhibition)",
            "core_term": "F_native_v2 = F_basis_v2 + F_sep_v2 + F_lock_v2",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征层原生化报告",
        "",
        f"- native_basis_v2: {hm['native_basis_v2']:.6f}",
        f"- native_separation_v2: {hm['native_separation_v2']:.6f}",
        f"- native_lock_v2: {hm['native_lock_v2']:.6f}",
        f"- feature_native_core_v2: {hm['feature_native_core_v2']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_layer_nativeization_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
