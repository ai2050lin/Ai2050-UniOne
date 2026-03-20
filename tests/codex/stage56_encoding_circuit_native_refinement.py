from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_native_refinement_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_circuit_native_refinement_summary() -> dict:
    circuit = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_formation_20260320" / "summary.json"
    )
    neuro = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neurodynamics_bridge_v4_20260320" / "summary.json"
    )
    ode = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_continuous_learning_ode_20260320" / "summary.json"
    )

    chm = circuit["headline_metrics"]
    nhm = neuro["headline_metrics"]
    ohm = ode["headline_metrics"]

    seed_raw = chm["local_stimulation"]
    bind_raw = chm["circuit_binding"] + nhm["synchrony_gain"] + nhm["basin_separation"]
    embed_raw = chm["structure_embedding"] + ohm["d_boundary"] + ohm["d_frontier"]
    pressure_raw = chm["steady_state_pressure"] + nhm["competitive_inhibition"] + max(0.0, -ohm["d_circuit"])
    total = seed_raw + bind_raw + embed_raw + pressure_raw

    seed_refined = seed_raw / total
    bind_refined = bind_raw / total
    embed_refined = embed_raw / total
    pressure_refined = pressure_raw / total
    encode_balance_refined = seed_refined + bind_refined + embed_refined - pressure_refined
    structure_yield_refined = (seed_refined + bind_refined + embed_refined) / max(pressure_refined, 1e-12)

    return {
        "headline_metrics": {
            "seed_refined": seed_refined,
            "bind_refined": bind_refined,
            "embed_refined": embed_refined,
            "pressure_refined": pressure_refined,
            "encode_balance_refined": encode_balance_refined,
            "structure_yield_refined": structure_yield_refined,
        },
        "refined_equation": {
            "seed_term": "C_seed_refined ~ local_stimulation / total_mass_v2",
            "bind_term": "C_bind_refined ~ (circuit_binding + synchrony_gain + basin_separation) / total_mass_v2",
            "embed_term": "N_embed_refined ~ (structure_embedding + d_boundary + d_frontier) / total_mass_v2",
            "pressure_term": "P_refined ~ (steady_state_pressure + competitive_inhibition) / total_mass_v2",
        },
        "project_readout": {
            "summary": "这一版把编码回路变量做了二次压缩，增强绑定量和嵌入量，减少编码核只被种子量单独主导的问题。",
            "next_question": "下一步要检验 refined 编码核是否能更稳地预测稳态区，而不是继续只做静态解释。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码回路原生变量强化报告",
        "",
        f"- seed_refined: {hm['seed_refined']:.6f}",
        f"- bind_refined: {hm['bind_refined']:.6f}",
        f"- embed_refined: {hm['embed_refined']:.6f}",
        f"- pressure_refined: {hm['pressure_refined']:.6f}",
        f"- encode_balance_refined: {hm['encode_balance_refined']:.6f}",
        f"- structure_yield_refined: {hm['structure_yield_refined']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_circuit_native_refinement_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
