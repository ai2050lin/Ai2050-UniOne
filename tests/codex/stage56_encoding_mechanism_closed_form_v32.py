from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v32_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_encoding_mechanism_closed_form_v32_summary() -> dict:
    v31 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v31_20260320" / "summary.json"
    )
    stability_native = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_stability_native_approximation_20260320" / "summary.json"
    )

    hv = v31["headline_metrics"]
    hs = stability_native["headline_metrics"]

    feature_term_v32 = hv["feature_term_v31"] + hs["native_stability_ratio"] * hs["native_stability_seed"]
    structure_term_v32 = hv["structure_term_v31"] + hs["native_stability_core"]
    learning_term_v32 = hv["learning_term_v31"] + hs["native_stability_feedback"] + hs["native_stability_core"] * hs["native_stability_ratio"]
    pressure_term_v32 = hv["pressure_term_v31"] + (1.0 - hs["native_stability_ratio"])
    encoding_margin_v32 = feature_term_v32 + structure_term_v32 + learning_term_v32 - pressure_term_v32

    return {
        "headline_metrics": {
            "feature_term_v32": feature_term_v32,
            "structure_term_v32": structure_term_v32,
            "learning_term_v32": learning_term_v32,
            "pressure_term_v32": pressure_term_v32,
            "encoding_margin_v32": encoding_margin_v32,
        },
        "closed_form_equation": {
            "feature_term": "K_f_v32 = K_f_v31 + R_native * S_seed_native",
            "structure_term": "K_s_v32 = K_s_v31 + S_native",
            "learning_term": "K_l_v32 = K_l_v31 + S_fb_native + S_native * R_native",
            "pressure_term": "P_v32 = P_v31 + (1 - R_native)",
            "margin_term": "M_encoding_v32 = K_f_v32 + K_s_v32 + K_l_v32 - P_v32",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 编码机制闭式第三十二版报告",
        "",
        f"- feature_term_v32: {hm['feature_term_v32']:.6f}",
        f"- structure_term_v32: {hm['structure_term_v32']:.6f}",
        f"- learning_term_v32: {hm['learning_term_v32']:.6f}",
        f"- pressure_term_v32: {hm['pressure_term_v32']:.6f}",
        f"- encoding_margin_v32: {hm['encoding_margin_v32']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_encoding_mechanism_closed_form_v32_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
