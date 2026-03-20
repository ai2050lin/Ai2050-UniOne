from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_layer_native_direct_measure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_layer_native_direct_measure_summary() -> dict:
    nativeization = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_nativeization_20260320" / "summary.json"
    )
    spike_native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spike_feature_native_variables_20260320" / "summary.json"
    )
    dominance = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_dominance_absolute_lock_20260320" / "summary.json"
    )

    hn = nativeization["headline_metrics"]
    hs = spike_native["headline_metrics"]
    hd = dominance["headline_metrics"]

    direct_basis_v3 = hn["native_basis_v2"] / (1.0 + hs["native_inhibition"])
    direct_selectivity_v3 = hn["native_separation_v2"] * (1.0 + hs["native_selectivity"])
    direct_lock_v3 = hn["native_lock_v2"] + hd["absolute_margin"]
    feature_direct_core_v3 = direct_basis_v3 + direct_selectivity_v3 + direct_lock_v3

    return {
        "headline_metrics": {
            "direct_basis_v3": direct_basis_v3,
            "direct_selectivity_v3": direct_selectivity_v3,
            "direct_lock_v3": direct_lock_v3,
            "feature_direct_core_v3": feature_direct_core_v3,
        },
        "direct_measure_equation": {
            "basis_term": "F_basis_v3 = native_basis_v2 / (1 + native_inhibition)",
            "selectivity_term": "F_sel_v3 = native_separation_v2 * (1 + native_selectivity)",
            "lock_term": "F_lock_v3 = native_lock_v2 + absolute_margin",
            "core_term": "F_direct_v3 = F_basis_v3 + F_sel_v3 + F_lock_v3",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征层近直测报告",
        "",
        f"- direct_basis_v3: {hm['direct_basis_v3']:.6f}",
        f"- direct_selectivity_v3: {hm['direct_selectivity_v3']:.6f}",
        f"- direct_lock_v3: {hm['direct_lock_v3']:.6f}",
        f"- feature_direct_core_v3: {hm['feature_direct_core_v3']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_layer_native_direct_measure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
