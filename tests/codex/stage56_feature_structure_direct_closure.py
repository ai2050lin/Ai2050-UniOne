from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_structure_direct_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_structure_direct_closure_summary() -> dict:
    feature_close = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_direct_closure_20260320" / "summary.json"
    )
    native_close = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_native_closure_20260320" / "summary.json"
    )
    canonical = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_canonical_closure_20260320" / "summary.json"
    )

    hf = feature_close["headline_metrics"]
    hn = native_close["headline_metrics"]
    hl = canonical["headline_metrics"]

    direct_circuit_closure = hn["closure_circuit_link"] * (1.0 + hf["feature_direct_closure_v4"] / 100.0)
    direct_structure_closure = hn["closure_structure_link"] * (1.0 + hf["feature_direct_closure_v4"] / 100.0)
    direct_feedback_closure = hn["closure_feedback"] + hl["canonical_seed"] / max(hf["feature_direct_closure_v4"], 1e-9)
    direct_closure_margin_v2 = direct_circuit_closure + direct_structure_closure + direct_feedback_closure

    return {
        "headline_metrics": {
            "direct_circuit_closure": direct_circuit_closure,
            "direct_structure_closure": direct_structure_closure,
            "direct_feedback_closure": direct_feedback_closure,
            "direct_closure_margin_v2": direct_closure_margin_v2,
        },
        "direct_structure_equation": {
            "circuit_term": "Ds_fc = closure_circuit_link * (1 + F_close_v4 / 100)",
            "structure_term": "Ds_fs = closure_structure_link * (1 + F_close_v4 / 100)",
            "feedback_term": "Ds_fb = closure_feedback + canonical_seed / F_close_v4",
            "margin_term": "Ds_margin = Ds_fc + Ds_fs + Ds_fb",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征到结构闭合直测报告",
        "",
        f"- direct_circuit_closure: {hm['direct_circuit_closure']:.6f}",
        f"- direct_structure_closure: {hm['direct_structure_closure']:.6f}",
        f"- direct_feedback_closure: {hm['direct_feedback_closure']:.6f}",
        f"- direct_closure_margin_v2: {hm['direct_closure_margin_v2']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_structure_direct_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
