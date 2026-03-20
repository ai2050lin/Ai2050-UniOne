from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v2_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v2_summary() -> dict:
    refined = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_native_refinement_20260320" / "summary.json"
    )
    ode = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_continuous_learning_ode_20260320" / "summary.json"
    )

    rhm = refined["headline_metrics"]
    ohm = ode["headline_metrics"]

    encoding_kernel_v2 = rhm["seed_refined"] + rhm["bind_refined"] + 0.5 * rhm["embed_refined"] - rhm["pressure_refined"]
    structural_growth_v2 = ohm["d_frontier"] + ohm["d_boundary"] + max(0.0, ohm["d_circuit"]) - abs(ohm["d_atlas"])
    circuit_pressure_v2 = rhm["pressure_refined"] + max(0.0, -ohm["d_frontier"]) + max(0.0, -ohm["d_boundary"])
    closed_form_margin_v2 = encoding_kernel_v2 + structural_growth_v2 - circuit_pressure_v2

    return {
        "headline_metrics": {
            "encoding_kernel_v2": encoding_kernel_v2,
            "structural_growth_v2": structural_growth_v2,
            "circuit_pressure_v2": circuit_pressure_v2,
            "closed_form_margin_v2": closed_form_margin_v2,
        },
        "closed_form_equation": {
            "kernel_v2": "K_enc_v2 = C_seed_refined + C_bind_refined + 0.5 * N_embed_refined - P_refined",
            "growth_v2": "G_v2 = dF/dt + dB/dt + max(0, dC/dt) - |dA/dt|",
            "pressure_v2": "P_v2 = P_refined + max(0, -dF/dt) + max(0, -dB/dt)",
            "margin_v2": "M_enc_v2 = K_enc_v2 + G_v2 - P_v2",
        },
        "project_readout": {
            "summary": "这一版把编码机制闭式核继续压短，把回路增长和结构增长合并进同一个增长项，检查是否能得到更稳的编码核边距。",
            "next_question": "下一步要比较 M_enc_v2 和旧的 M_enc，检查它是否在稳态预测上更稳定。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制第二版闭式核报告",
        "",
        f"- encoding_kernel_v2: {hm['encoding_kernel_v2']:.6f}",
        f"- structural_growth_v2: {hm['structural_growth_v2']:.6f}",
        f"- circuit_pressure_v2: {hm['circuit_pressure_v2']:.6f}",
        f"- closed_form_margin_v2: {hm['closed_form_margin_v2']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v2_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
