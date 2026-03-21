from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v53_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v53_summary() -> dict:
    v52 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v52_20260320" / "summary.json"
    )
    brain_direct = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_native_direct_measure_20260320" / "summary.json"
    )
    proto_train = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_contextual_trainable_prototype_20260320" / "summary.json"
    )

    hv = v52["headline_metrics"]
    hb = brain_direct["headline_metrics"]
    hp = proto_train["headline_metrics"]

    feature_term_v53 = hv["feature_term_v52"] + hv["feature_term_v52"] * hb["direct_feature_measure"] * 0.01
    structure_term_v53 = hv["structure_term_v52"] + hv["structure_term_v52"] * hb["direct_structure_measure"] * 0.01
    learning_term_v53 = hv["learning_term_v52"] + hv["learning_term_v52"] * hp["heldout_generalization"] + hp["trainable_prototype_margin"] * 1000.0
    pressure_term_v53 = max(0.0, hv["pressure_term_v52"] + hb["direct_brain_gap"] + hp["context_split_consistency"] - hp["shared_red_consistency"])
    encoding_margin_v53 = feature_term_v53 + structure_term_v53 + learning_term_v53 - pressure_term_v53

    return {
        "headline_metrics": {
            "feature_term_v53": feature_term_v53,
            "structure_term_v53": structure_term_v53,
            "learning_term_v53": learning_term_v53,
            "pressure_term_v53": pressure_term_v53,
            "encoding_margin_v53": encoding_margin_v53,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v53 = K_f_v52 + K_f_v52 * D_feature * 0.01",
            "structure_term": "K_s_v53 = K_s_v52 + K_s_v52 * D_structure * 0.01",
            "learning_term": "K_l_v53 = K_l_v52 + K_l_v52 * G_hold + M_proto_trainable * 1000",
            "pressure_term": "P_v53 = P_v52 + G_direct + S_ctx - H_red_ctx",
            "margin_term": "M_encoding_v53 = K_f_v53 + K_s_v53 + K_l_v53 - P_v53",
        },
        "project_readout": {
            "summary": "第五十三版主核把逆向脑编码近直测和带上下文的可训练扩展原型一起并回主式，用来表达从解释主核向训练主核推进的过渡关系。",
            "next_question": "下一步要把这条主核继续推进到即时学习和旧知识回落测试，检验它能否真正过渡到训练终式。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第五十三版报告",
        "",
        f"- feature_term_v53: {hm['feature_term_v53']:.6f}",
        f"- structure_term_v53: {hm['structure_term_v53']:.6f}",
        f"- learning_term_v53: {hm['learning_term_v53']:.6f}",
        f"- pressure_term_v53: {hm['pressure_term_v53']:.6f}",
        f"- encoding_margin_v53: {hm['encoding_margin_v53']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v53_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
