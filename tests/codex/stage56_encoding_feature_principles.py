from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_feature_principles_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_feature_principles_summary() -> dict:
    feature_terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_terminal_direct_20260320" / "summary.json"
    )
    structure_terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_terminal_closure_20260320" / "summary.json"
    )
    equal_level = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_equal_level_closure_20260320" / "summary.json"
    )
    v25 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v25_20260320" / "summary.json"
    )

    hf = feature_terminal["headline_metrics"]
    hs = structure_terminal["headline_metrics"]
    he = equal_level["headline_metrics"]
    hv = v25["headline_metrics"]

    extraction_stack = hf["direct_basis_v5"] + hf["direct_selectivity_v5"] + hf["direct_lock_v5"]
    structure_stack = hs["terminal_circuit_closure"] + hs["terminal_structure_closure"] + hs["terminal_feedback_closure"]
    equalized_core = he["equal_geometric_core"]
    principle_margin = hv["encoding_margin_v25"] / max(hv["learning_term_v25"], 1e-9)

    return {
        "headline_metrics": {
            "extraction_stack": extraction_stack,
            "structure_stack": structure_stack,
            "equalized_core": equalized_core,
            "principle_margin": principle_margin,
        },
        "principle_equation": {
            "extraction_term": "E_stack = F_basis_v5 + F_sel_v5 + F_lock_v5",
            "structure_term": "S_stack = Tc_fc + Tc_fs + Tc_fb",
            "equalized_term": "C_equal = E_core",
            "margin_term": "M_principle = M_encoding_v25 / K_l_v25",
        },
        "principles": [
            "\u7f16\u7801\u7ed3\u6784\u4e0d\u662f\u5355\u70b9\u5411\u91cf\uff0c\u800c\u662f\u7279\u5f81\u5c42\u3001\u7ed3\u6784\u5c42\u3001\u5b66\u4e60\u5c42\u548c\u538b\u529b\u5c42\u7684\u590d\u5408\u5bf9\u8c61\u3002",
            "\u7279\u5f81\u63d0\u53d6\u4e0d\u662f\u4e00\u6b21\u8bfb\u51fa\uff0c\u800c\u662f\u57fa\u7840\u5dee\u5f02\u3001\u9009\u62e9\u6027\u653e\u5927\u548c\u9501\u5b9a\u7ef4\u6301\u4e09\u6b65\u53e0\u52a0\u3002",
            "\u7f51\u7edc\u7ed3\u6784\u4e0d\u662f\u5148\u9a8c\u7ed9\u5b9a\uff0c\u800c\u662f\u7279\u5f81\u5c42\u63a8\u52a8\u56de\u8def\u4e0e\u7ed3\u6784\u95ed\u5408\u540e\u9010\u6b65\u5f62\u6210\u3002",
            "\u5f53\u524d\u6700\u7a33\u7684\u9636\u6bb5\u5224\u65ad\u662f\uff1a\u7279\u5f81\u5c42\u4e0e\u7ed3\u6784\u5c42\u5df2\u7ecf\u8fdb\u5165\u540c\u91cf\u7ea7\u95ed\u5408\uff0c\u4f46\u8fd9\u79cd\u95ed\u5408\u4ecd\u7136\u662f\u8fd1\u539f\u751f\u5bf9\u8c61\u4e0a\u7684\u95ed\u5408\u3002",
        ],
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码结构与特征提取原理报告",
        "",
        f"- extraction_stack: {hm['extraction_stack']:.6f}",
        f"- structure_stack: {hm['structure_stack']:.6f}",
        f"- equalized_core: {hm['equalized_core']:.6f}",
        f"- principle_margin: {hm['principle_margin']:.6f}",
        "",
        "## 原理",
    ]
    lines.extend([f"- {item}" for item in summary["principles"]])
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_feature_principles_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
