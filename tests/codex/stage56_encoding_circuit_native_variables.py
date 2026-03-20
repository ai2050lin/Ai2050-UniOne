from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_native_variables_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_circuit_native_variable_summary() -> dict:
    circuit = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_formation_20260320" / "summary.json"
    )
    hm = circuit["headline_metrics"]
    total = (
        hm["local_stimulation"]
        + hm["circuit_binding"]
        + hm["structure_embedding"]
        + hm["steady_state_pressure"]
    )
    seed_native = hm["local_stimulation"] / total
    bind_native = hm["circuit_binding"] / total
    embed_native = hm["structure_embedding"] / total
    pressure_native = hm["steady_state_pressure"] / total
    encode_balance_native = seed_native + bind_native + embed_native - pressure_native
    structure_yield_native = (seed_native + bind_native) / max(embed_native + pressure_native, 1e-12)

    summary = {
        "headline_metrics": {
            "seed_native": seed_native,
            "bind_native": bind_native,
            "embed_native": embed_native,
            "pressure_native": pressure_native,
            "encode_balance_native": encode_balance_native,
            "structure_yield_native": structure_yield_native,
        },
        "native_equation": {
            "seed_field": "C_seed ~ local_stimulation / total_mass",
            "bind_field": "C_bind_native ~ circuit_binding / total_mass",
            "embed_field": "N_embed_native ~ structure_embedding / total_mass",
            "pressure_field": "P_native ~ steady_state_pressure / total_mass",
        },
        "project_readout": {
            "summary": "这一版把编码回路形成链压成更接近原生的四个相对结构量，方便后续直接做稳态预测和闭式压缩。",
            "next_question": "下一步要看这些原生编码变量是否比旧的桥接量更能预测高平衡区和高风险区。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码回路原生变量报告",
        "",
        f"- seed_native: {hm['seed_native']:.6f}",
        f"- bind_native: {hm['bind_native']:.6f}",
        f"- embed_native: {hm['embed_native']:.6f}",
        f"- pressure_native: {hm['pressure_native']:.6f}",
        f"- encode_balance_native: {hm['encode_balance_native']:.6f}",
        f"- structure_yield_native: {hm['structure_yield_native']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_circuit_native_variable_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
