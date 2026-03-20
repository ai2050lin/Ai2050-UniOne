from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_feature_structure_terminal_closure_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_feature_structure_terminal_closure_summary() -> dict:
    feature_terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_layer_terminal_direct_20260320" / "summary.json"
    )
    structure_close = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_direct_closure_20260320" / "summary.json"
    )
    canonical = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_learning_dynamics_terminal_canonical_closure_20260320" / "summary.json"
    )

    hf = feature_terminal["headline_metrics"]
    hs = structure_close["headline_metrics"]
    hc = canonical["headline_metrics"]

    terminal_circuit_closure = hs["direct_circuit_closure"] * (1.0 + hf["feature_terminal_core_v5"] / 200.0)
    terminal_structure_closure = hs["direct_structure_closure"] * (1.0 + hf["feature_terminal_core_v5"] / 200.0)
    terminal_feedback_closure = hs["direct_feedback_closure"] + hc["canonical_global"] / max(hf["feature_terminal_core_v5"], 1e-9)
    terminal_closure_margin_v3 = terminal_circuit_closure + terminal_structure_closure + terminal_feedback_closure

    return {
        "headline_metrics": {
            "terminal_circuit_closure": terminal_circuit_closure,
            "terminal_structure_closure": terminal_structure_closure,
            "terminal_feedback_closure": terminal_feedback_closure,
            "terminal_closure_margin_v3": terminal_closure_margin_v3,
        },
        "terminal_closure_equation": {
            "circuit_term": "Tc_fc = direct_circuit_closure * (1 + F_terminal_v5 / 200)",
            "structure_term": "Tc_fs = direct_structure_closure * (1 + F_terminal_v5 / 200)",
            "feedback_term": "Tc_fb = direct_feedback_closure + canonical_global / F_terminal_v5",
            "margin_term": "Tc_margin = Tc_fc + Tc_fs + Tc_fb",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 特征到结构终块闭合报告",
        "",
        f"- terminal_circuit_closure: {hm['terminal_circuit_closure']:.6f}",
        f"- terminal_structure_closure: {hm['terminal_structure_closure']:.6f}",
        f"- terminal_feedback_closure: {hm['terminal_feedback_closure']:.6f}",
        f"- terminal_closure_margin_v3: {hm['terminal_closure_margin_v3']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_feature_structure_terminal_closure_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
