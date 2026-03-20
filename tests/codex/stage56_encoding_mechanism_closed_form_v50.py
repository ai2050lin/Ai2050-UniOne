from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v50_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v50_summary() -> dict:
    v49 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v49_20260320" / "summary.json"
    )
    color_native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_color_fiber_nativeization_20260320" / "summary.json"
    )
    proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_object_attribute_coupling_prototype_20260320" / "summary.json"
    )

    hv = v49["headline_metrics"]
    hc = color_native["headline_metrics"]
    hp = proto["headline_metrics"]

    feature_term_v50 = hv["feature_term_v49"] + hv["feature_term_v49"] * hc["native_color_binding"] * 0.01
    structure_term_v50 = hv["structure_term_v49"] + hv["structure_term_v49"] * hp["same_attribute_different_route"] * 0.01
    learning_term_v50 = hv["learning_term_v49"] + hv["learning_term_v49"] * hp["shared_attribute_reuse"] + hc["color_native_margin"] * 1000.0
    pressure_term_v50 = max(
        0.0,
        hv["pressure_term_v49"] + hp["same_attribute_different_route"] + hc["native_color_route_split"] - hp["shared_attribute_reuse"]
    )
    encoding_margin_v50 = feature_term_v50 + structure_term_v50 + learning_term_v50 - pressure_term_v50

    return {
        "headline_metrics": {
            "feature_term_v50": feature_term_v50,
            "structure_term_v50": structure_term_v50,
            "learning_term_v50": learning_term_v50,
            "pressure_term_v50": pressure_term_v50,
            "encoding_margin_v50": encoding_margin_v50,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v50 = K_f_v49 + K_f_v49 * B_red * 0.01",
            "structure_term": "K_s_v50 = K_s_v49 + K_s_v49 * R_split * 0.01",
            "learning_term": "K_l_v50 = K_l_v49 + K_l_v49 * H_attr + M_red * 1000",
            "pressure_term": "P_v50 = P_v49 + R_split + S_red - H_attr",
            "margin_term": "M_encoding_v50 = K_f_v50 + K_s_v50 + K_l_v50 - P_v50",
        },
        "project_readout": {
            "summary": "第五十版主核把颜色纤维原生化和对象-属性耦合原型一起并回主式，用来表达共享属性支路与对象路径分叉的统一关系。",
            "next_question": "下一步要把这条共享属性支路推进到更多颜色和更多上下文任务，确认它是不是可泛化的原理。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第五十版报告",
        "",
        f"- feature_term_v50: {hm['feature_term_v50']:.6f}",
        f"- structure_term_v50: {hm['structure_term_v50']:.6f}",
        f"- learning_term_v50: {hm['learning_term_v50']:.6f}",
        f"- pressure_term_v50: {hm['pressure_term_v50']:.6f}",
        f"- encoding_margin_v50: {hm['encoding_margin_v50']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v50_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
