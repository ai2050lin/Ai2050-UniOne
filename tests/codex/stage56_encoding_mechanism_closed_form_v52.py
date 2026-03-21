from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v52_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v52_summary() -> dict:
    v51 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v51_20260320" / "summary.json"
    )
    brain_native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_nativeization_20260320" / "summary.json"
    )
    proto_expanded = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_object_attribute_structure_prototype_20260320" / "summary.json"
    )

    hv = v51["headline_metrics"]
    hb = brain_native["headline_metrics"]
    hp = proto_expanded["headline_metrics"]

    feature_term_v52 = hv["feature_term_v51"] + hv["feature_term_v51"] * hb["feature_nativeization"] * 0.01
    structure_term_v52 = hv["structure_term_v51"] + hv["structure_term_v51"] * hp["structure_route_split"] * 0.01
    learning_term_v52 = hv["learning_term_v51"] + hv["learning_term_v51"] * hb["brain_native_chain_strength"] + hp["expanded_prototype_margin"] * 1000.0
    pressure_term_v52 = max(0.0, hv["pressure_term_v51"] + hb["brain_native_gap"] + hp["context_route_split"] - hp["shared_red_reuse"])
    encoding_margin_v52 = feature_term_v52 + structure_term_v52 + learning_term_v52 - pressure_term_v52

    return {
        "headline_metrics": {
            "feature_term_v52": feature_term_v52,
            "structure_term_v52": structure_term_v52,
            "learning_term_v52": learning_term_v52,
            "pressure_term_v52": pressure_term_v52,
            "encoding_margin_v52": encoding_margin_v52,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v52 = K_f_v51 + K_f_v51 * N_feature_native * 0.01",
            "structure_term": "K_s_v52 = K_s_v51 + K_s_v51 * R_structure_split * 0.01",
            "learning_term": "K_l_v52 = K_l_v51 + K_l_v51 * M_brain_native + M_proto_expand * 1000",
            "pressure_term": "P_v52 = P_v51 + G_native + G_context - H_red_shared",
            "margin_term": "M_encoding_v52 = K_f_v52 + K_s_v52 + K_l_v52 - P_v52",
        },
        "project_readout": {
            "summary": "第五十二版主核把逆向脑编码原生化和对象-属性-结构扩展原型一起并回主式，用来表达语言主入口之后的脑编码闭合和对象属性结构扩展关系。",
            "next_question": "下一步要把这条主核继续推进到训练终式，检验解释主核能否转成可施工的训练规则。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第五十二版报告",
        "",
        f"- feature_term_v52: {hm['feature_term_v52']:.6f}",
        f"- structure_term_v52: {hm['structure_term_v52']:.6f}",
        f"- learning_term_v52: {hm['learning_term_v52']:.6f}",
        f"- pressure_term_v52: {hm['pressure_term_v52']:.6f}",
        f"- encoding_margin_v52: {hm['encoding_margin_v52']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v52_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
