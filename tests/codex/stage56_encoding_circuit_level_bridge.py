from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_level_bridge_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_circuit_level_bridge_summary() -> dict:
    refined = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_native_refinement_20260320" / "summary.json"
    )
    circuit = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_formation_20260320" / "summary.json"
    )
    neuro = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_continuous_neurodynamics_bridge_20260320" / "summary.json"
    )

    rhm = refined["headline_metrics"]
    chm = circuit["headline_metrics"]
    nhm = neuro["headline_metrics"]

    excitatory_seed = rhm["seed_refined"] * nhm["dV_dt"]
    synchrony_binding = rhm["bind_refined"] * nhm["dS_dt"]
    embedding_recruitment = rhm["embed_refined"] * chm["structure_embedding"]
    inhibitory_pressure = rhm["pressure_refined"] * max(nhm["dB_dt"], 0.0)
    circuit_level_margin = excitatory_seed + synchrony_binding + embedding_recruitment - inhibitory_pressure

    return {
        "headline_metrics": {
            "excitatory_seed": excitatory_seed,
            "synchrony_binding": synchrony_binding,
            "embedding_recruitment": embedding_recruitment,
            "inhibitory_pressure": inhibitory_pressure,
            "circuit_level_margin": circuit_level_margin,
        },
        "circuit_bridge_equation": {
            "seed_term": "E_seed = seed_refined * dV/dt",
            "bind_term": "B_sync = bind_refined * dS/dt",
            "embed_term": "R_embed = embed_refined * structure_embedding",
            "pressure_term": "I_pressure = pressure_refined * max(dB/dt, 0)",
            "margin_term": "M_circuit = E_seed + B_sync + R_embed - I_pressure",
        },
        "project_readout": {
            "summary": "这一版把编码核继续往回路级桥接，明确区分兴奋种子、同步绑定、嵌入招募和抑制压力四种回路量。",
            "next_question": "下一步要检验回路级边距是否能进一步提升编码核对稳态区和长期学习区的预测力。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码回路级桥接报告",
        "",
        f"- excitatory_seed: {hm['excitatory_seed']:.6f}",
        f"- synchrony_binding: {hm['synchrony_binding']:.6f}",
        f"- embedding_recruitment: {hm['embedding_recruitment']:.6f}",
        f"- inhibitory_pressure: {hm['inhibitory_pressure']:.6f}",
        f"- circuit_level_margin: {hm['circuit_level_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_circuit_level_bridge_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
