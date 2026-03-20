from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v33_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v33_summary() -> dict:
    v32 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v32_20260320" / "summary.json"
    )
    remap = _load_json(ROOT / "tests" / "codex_temp" / "stage56_icspb_object_remapping_20260320" / "summary.json")
    transport = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_bridge_reintegration_20260320" / "summary.json"
    )

    hv = v32["headline_metrics"]
    hr = remap["headline_metrics"]
    ht = transport["headline_metrics"]

    feature_term_v33 = (
        hv["feature_term_v32"]
        + hr["attribute_fiber_to_feature"]
        + ht["restricted_readout_term"]
        + hr["concept_offset_to_feature"]
    )
    structure_term_v33 = (
        hv["structure_term_v32"]
        + hr["family_patch_to_structure"]
        + ht["admissible_update_term"]
    )
    learning_term_v33 = (
        hv["learning_term_v32"]
        + ht["stage_transport_term"]
        + ht["successor_transport_term"]
        + ht["protocol_bridge_term"]
    )
    pressure_term_v33 = (
        hv["pressure_term_v32"]
        + (1.0 - ht["protocol_bridge_strength"])
        + (1.0 - ht["admissible_update_strength"])
    )
    encoding_margin_v33 = feature_term_v33 + structure_term_v33 + learning_term_v33 - pressure_term_v33

    return {
        "headline_metrics": {
            "feature_term_v33": feature_term_v33,
            "structure_term_v33": structure_term_v33,
            "learning_term_v33": learning_term_v33,
            "pressure_term_v33": pressure_term_v33,
            "encoding_margin_v33": encoding_margin_v33,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v33 = K_f_v32 + A_fiber + R_readout + D_offset",
            "structure_term": "K_s_v33 = K_s_v32 + F_patch + U_admissible",
            "learning_term": "K_l_v33 = K_l_v32 + T_stage + T_successor + B_protocol",
            "pressure_term": "P_v33 = P_v32 + (1 - G_bridge) + (1 - G_update)",
            "margin_term": "M_encoding_v33 = K_f_v33 + K_s_v33 + K_l_v33 - P_v33",
        },
        "project_readout": {
            "summary": "第 33 版把旧 ICSPB 中最关键的 transport/readout/bridge 层正式并回了当前主核。现在 family patch / concept offset / attribute fiber 主要回到特征与结构层，admissible update / restricted readout / stage-successor / protocol bridge 则被吸收入学习反馈与压力补偿层。",
            "next_question": "下一步要继续检验旧版运输几何对象回并之后，是否能长期稳定地留在同一条主方程里，而不是再次分裂成平行理论。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第三十三版报告",
        "",
        f"- feature_term_v33: {hm['feature_term_v33']:.6f}",
        f"- structure_term_v33: {hm['structure_term_v33']:.6f}",
        f"- learning_term_v33: {hm['learning_term_v33']:.6f}",
        f"- pressure_term_v33: {hm['pressure_term_v33']:.6f}",
        f"- encoding_margin_v33: {hm['encoding_margin_v33']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v33_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
