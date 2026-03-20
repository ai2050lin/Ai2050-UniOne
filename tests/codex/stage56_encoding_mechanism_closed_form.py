from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_summary() -> dict:
    native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_native_variables_20260320" / "summary.json"
    )
    ode = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_continuous_learning_ode_20260320" / "summary.json"
    )
    nhm = native["headline_metrics"]
    ohm = ode["headline_metrics"]

    encoding_core = nhm["seed_native"] + nhm["bind_native"] + nhm["embed_native"] - nhm["pressure_native"]
    structural_growth = ohm["d_frontier"] + ohm["d_boundary"] - abs(ohm["d_atlas"])
    circuit_pressure = max(0.0, -ohm["d_circuit"]) + nhm["pressure_native"]
    closed_form_margin = encoding_core + structural_growth - circuit_pressure

    summary = {
        "headline_metrics": {
            "encoding_core": encoding_core,
            "structural_growth": structural_growth,
            "circuit_pressure": circuit_pressure,
            "closed_form_margin": closed_form_margin,
        },
        "closed_form_equation": {
            "encoding_kernel": "K_enc = C_seed + C_bind + N_embed - P_native",
            "growth_term": "G_struct = dF/dt + dB/dt - |dA/dt|",
            "pressure_term": "P_circuit = max(0, -dC/dt) + P_native",
            "closed_form": "M_enc = K_enc + G_struct - P_circuit",
        },
        "project_readout": {
            "summary": "这一版尝试把编码机制本身压成更短的闭式核，重点不再是单个桥接量，而是编码核、结构增长项和回路压力项三者的平衡。",
            "next_question": "下一步要看这个闭式核能否跨资产预测稳态区和长期在线风险。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式核报告",
        "",
        f"- encoding_core: {hm['encoding_core']:.6f}",
        f"- structural_growth: {hm['structural_growth']:.6f}",
        f"- circuit_pressure: {hm['circuit_pressure']:.6f}",
        f"- closed_form_margin: {hm['closed_form_margin']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
