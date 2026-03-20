from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_layer_direct_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_layer_direct_closure_summary() -> dict:
    direct = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_native_direct_measure_20260320" / "summary.json"
    )
    v19 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v19_20260320" / "summary.json"
    )
    circuit = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_circuit_native_near_direct_v8_20260320" / "summary.json"
    )

    hd = direct["headline_metrics"]
    hv = v19["headline_metrics"]
    hc = circuit["headline_metrics"]

    direct_basis_v4 = hd["direct_basis_v3"] * (1.0 + hc["direct_binding_v8"] / (1.0 + hc["direct_attractor_v8"]))
    direct_selectivity_v4 = hd["direct_selectivity_v3"] * (1.0 + hc["direct_gate_v8"])
    direct_lock_v4 = hd["direct_lock_v3"] + hv["pressure_term_v19"]
    feature_direct_closure_v4 = direct_basis_v4 + direct_selectivity_v4 + direct_lock_v4

    return {
        "headline_metrics": {
            "direct_basis_v4": direct_basis_v4,
            "direct_selectivity_v4": direct_selectivity_v4,
            "direct_lock_v4": direct_lock_v4,
            "feature_direct_closure_v4": feature_direct_closure_v4,
        },
        "direct_closure_equation": {
            "basis_term": "F_basis_v4 = direct_basis_v3 * (1 + direct_binding_v8 / (1 + direct_attractor_v8))",
            "selectivity_term": "F_sel_v4 = direct_selectivity_v3 * (1 + direct_gate_v8)",
            "lock_term": "F_lock_v4 = direct_lock_v3 + pressure_term_v19",
            "core_term": "F_close_v4 = F_basis_v4 + F_sel_v4 + F_lock_v4",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征层直测收口报告",
        "",
        f"- direct_basis_v4: {hm['direct_basis_v4']:.6f}",
        f"- direct_selectivity_v4: {hm['direct_selectivity_v4']:.6f}",
        f"- direct_lock_v4: {hm['direct_lock_v4']:.6f}",
        f"- feature_direct_closure_v4: {hm['feature_direct_closure_v4']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_layer_direct_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
