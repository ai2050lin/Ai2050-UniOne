from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v3_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v3_summary() -> dict:
    refined = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_native_refinement_20260320" / "summary.json"
    )
    v2 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v2_20260320" / "summary.json"
    )
    cross_asset = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_kernel_cross_asset_validation_20260320" / "summary.json"
    )
    circuit_level = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_circuit_level_bridge_20260320" / "summary.json"
    )

    rhm = refined["headline_metrics"]
    vhm = v2["headline_metrics"]
    xhm = cross_asset["headline_metrics"]
    chm = circuit_level["headline_metrics"]

    encoding_kernel_v3 = rhm["seed_refined"] + rhm["bind_refined"] + rhm["embed_refined"] - rhm["pressure_refined"]
    structure_growth_v3 = 0.5 * vhm["structural_growth_v2"] + 0.5 * chm["circuit_level_margin"]
    cross_asset_pressure_v3 = xhm["support_gap"] + vhm["circuit_pressure_v2"]
    closed_form_margin_v3 = encoding_kernel_v3 + structure_growth_v3 + xhm["cross_asset_support"] - cross_asset_pressure_v3

    return {
        "headline_metrics": {
            "encoding_kernel_v3": encoding_kernel_v3,
            "structure_growth_v3": structure_growth_v3,
            "cross_asset_pressure_v3": cross_asset_pressure_v3,
            "closed_form_margin_v3": closed_form_margin_v3,
        },
        "closed_form_equation": {
            "kernel_v3": "K_enc_v3 = seed_refined + bind_refined + embed_refined - pressure_refined",
            "growth_v3": "G_v3 = 0.5 * structural_growth_v2 + 0.5 * circuit_level_margin",
            "pressure_v3": "P_v3 = support_gap + circuit_pressure_v2",
            "margin_v3": "M_enc_v3 = K_enc_v3 + G_v3 + cross_asset_support - P_v3",
        },
        "project_readout": {
            "summary": "这一版把编码核、回路级桥接和跨资产支持度并回同一个闭式核，检查编码机制能否同时保住结构解释、稳态预测和跨资产稳定性。",
            "next_question": "下一步要比较 M_enc_v3 和 M_enc_v2 在更多资产上的稳定性，判断第三版是否足够成为阶段性最终编码核。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制第三版闭式核报告",
        "",
        f"- encoding_kernel_v3: {hm['encoding_kernel_v3']:.6f}",
        f"- structure_growth_v3: {hm['structure_growth_v3']:.6f}",
        f"- cross_asset_pressure_v3: {hm['cross_asset_pressure_v3']:.6f}",
        f"- closed_form_margin_v3: {hm['closed_form_margin_v3']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v3_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
