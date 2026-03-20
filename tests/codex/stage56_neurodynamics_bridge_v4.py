from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_neurodynamics_bridge_v4_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_neurodynamics_bridge_v4_summary() -> dict:
    local_native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_native_update_field_20260320" / "summary.json"
    )
    spike = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_spiking_dynamics_bridge_v3_20260320" / "summary.json"
    )
    attractor = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_attractor_circuit_bridge_v1_20260320" / "summary.json"
    )

    local = local_native["headline_metrics"]
    spike_state = spike["spike_bridge_state"]

    local_excitation = spike_state["excitatory_drive"] * local["patch_update_native"]
    competitive_inhibition = spike_state["inhibitory_load"] + local["forgetting_pressure_native"]
    synchrony_gain = spike_state["select_synchrony"] * (1.0 - local["gate_drift_native"])
    basin_separation = attractor["final_attractor_gap"] * local["attractor_rearrangement_native"]

    summary = {
        "headline_metrics": {
            "local_excitation": local_excitation,
            "competitive_inhibition": competitive_inhibition,
            "synchrony_gain": synchrony_gain,
            "basin_separation": basin_separation,
            "dynamic_margin": local_excitation + synchrony_gain - competitive_inhibition,
        },
        "bridge_equations": {
            "local_excitation_eq": "E_local ~ excitatory_drive * patch_update_native",
            "competition_eq": "I_comp ~ inhibitory_load + forgetting_pressure_native",
            "synchrony_eq": "S_sync ~ select_synchrony * (1 - gate_drift_native)",
            "basin_eq": "B_sep ~ final_attractor_gap * attractor_rearrangement_native",
        },
        "project_readout": {
            "summary": "当前桥接已经能把局部更新场和脉冲近似量接成更接近神经动力学的四元组：局部兴奋、竞争抑制、同步增益、吸引域分离。",
            "next_question": "下一步要把这四元组推进到连续时间和回路级，而不是继续停在中层桥接量。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 神经动力学桥接第四版报告",
        "",
        f"- local_excitation: {hm['local_excitation']:.6f}",
        f"- competitive_inhibition: {hm['competitive_inhibition']:.6f}",
        f"- synchrony_gain: {hm['synchrony_gain']:.6f}",
        f"- basin_separation: {hm['basin_separation']:.6f}",
        f"- dynamic_margin: {hm['dynamic_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_neurodynamics_bridge_v4_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
