from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_local_native_update_field_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_local_native_update_field_summary() -> dict:
    cascade = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_first_plasticity_cascade_20260320" / "summary.json"
    )
    long_ctx = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_long_context_online_language_suite_20260320" / "summary.json"
    )
    attractor = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_attractor_circuit_bridge_v1_20260320" / "summary.json"
    )

    frontier_peak = cascade["headline_metrics"]["frontier_peak"]
    boundary_peak = cascade["headline_metrics"]["boundary_peak"]
    atlas_peak = cascade["headline_metrics"]["atlas_peak"]
    gap_shift = attractor["gap_shift"]
    short_forgetting = long_ctx["short_context"]["deltas"]["forgetting"]
    long_forgetting = long_ctx["long_context"]["deltas"]["forgetting"]
    short_gate_shift = abs(long_ctx["short_context"]["deltas"]["strict_gate_shift"])
    long_gate_shift = abs(long_ctx["long_context"]["deltas"]["strict_gate_shift"])

    patch_update_native = frontier_peak / (frontier_peak + boundary_peak + atlas_peak)
    boundary_response_native = boundary_peak / (frontier_peak + boundary_peak + atlas_peak)
    atlas_consolidation_native = atlas_peak / (frontier_peak + boundary_peak + atlas_peak)
    attractor_rearrangement_native = gap_shift / max(attractor["final_attractor_gap"], 1e-12)
    forgetting_pressure_native = (short_forgetting + long_forgetting) / 2.0
    gate_drift_native = (short_gate_shift + long_gate_shift) / 2.0
    locality_margin = patch_update_native - boundary_response_native - atlas_consolidation_native

    summary = {
        "headline_metrics": {
            "patch_update_native": patch_update_native,
            "boundary_response_native": boundary_response_native,
            "atlas_consolidation_native": atlas_consolidation_native,
            "attractor_rearrangement_native": attractor_rearrangement_native,
            "forgetting_pressure_native": forgetting_pressure_native,
            "gate_drift_native": gate_drift_native,
            "locality_margin": locality_margin,
        },
        "local_field_equation": {
            "patch_field": "P_local ~ patch_update_native",
            "boundary_field": "B_global ~ boundary_response_native + attractor_rearrangement_native",
            "atlas_field": "A_slow ~ atlas_consolidation_native",
            "stability_field": "R_risk ~ forgetting_pressure_native + gate_drift_native",
        },
        "project_readout": {
            "summary": "这一版把局部更新、边界响应、图册固化和风险漂移压成了更接近原生的更新场对象，便于后续直接并入学习方程。",
            "next_question": "下一步要把这些原生更新场直接并回跨规模学习方程，而不是继续停在单独摘要对象上。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 局部原生更新场报告",
        "",
        f"- patch_update_native: {hm['patch_update_native']:.6f}",
        f"- boundary_response_native: {hm['boundary_response_native']:.6f}",
        f"- atlas_consolidation_native: {hm['atlas_consolidation_native']:.6f}",
        f"- attractor_rearrangement_native: {hm['attractor_rearrangement_native']:.6f}",
        f"- forgetting_pressure_native: {hm['forgetting_pressure_native']:.6f}",
        f"- gate_drift_native: {hm['gate_drift_native']:.6f}",
        f"- locality_margin: {hm['locality_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_local_native_update_field_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
