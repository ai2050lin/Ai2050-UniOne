from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_formation_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_circuit_formation_summary() -> dict:
    local_native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_native_update_field_20260320" / "summary.json"
    )
    neuro = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neurodynamics_bridge_v4_20260320" / "summary.json"
    )
    local_eq = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_local_global_learning_equation_20260320" / "summary.json"
    )

    hm_local = local_native["headline_metrics"]
    hm_neuro = neuro["headline_metrics"]
    hm_eq = local_eq["headline_metrics"]

    local_stimulation = hm_neuro["local_excitation"] * hm_local["patch_update_native"]
    circuit_binding = hm_neuro["synchrony_gain"] * hm_eq["local_patch_drive"]
    structure_embedding = hm_eq["global_boundary_drive"] + hm_local["boundary_response_native"]
    steady_state_pressure = hm_neuro["competitive_inhibition"] + hm_eq["risk_drag"]
    circuit_margin = local_stimulation + circuit_binding - steady_state_pressure

    summary = {
        "headline_metrics": {
            "local_stimulation": local_stimulation,
            "circuit_binding": circuit_binding,
            "structure_embedding": structure_embedding,
            "steady_state_pressure": steady_state_pressure,
            "circuit_margin": circuit_margin,
        },
        "formation_equation": {
            "stimulation": "C_stim ~ local_excitation * patch_update_native",
            "binding": "C_bind ~ synchrony_gain * local_patch_drive",
            "embedding": "N_embed ~ global_boundary_drive + boundary_response_native",
            "pressure": "P_steady ~ competitive_inhibition + risk_drag",
        },
        "project_readout": {
            "summary": "当前已经能把局部受刺激、编码回路形成、网络结构嵌入和稳态压力四个环节写成连续链条。",
            "next_question": "下一步要把这些回路形成量和真实长期在线稳态分区对上，检验哪些回路会走向高平衡区，哪些会走向高风险区。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码回路形成报告",
        "",
        f"- local_stimulation: {hm['local_stimulation']:.6f}",
        f"- circuit_binding: {hm['circuit_binding']:.6f}",
        f"- structure_embedding: {hm['structure_embedding']:.6f}",
        f"- steady_state_pressure: {hm['steady_state_pressure']:.6f}",
        f"- circuit_margin: {hm['circuit_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_circuit_formation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
