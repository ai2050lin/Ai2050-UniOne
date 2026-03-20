from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_direct_measure_v2_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_structure_genesis_direct_measure_v2_summary() -> dict:
    structure_terminal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_feature_structure_terminal_closure_20260320" / "summary.json"
    )
    chain = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_feature_network_chain_20260320" / "summary.json"
    )
    genesis = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_network_structure_genesis_probe_20260320" / "summary.json"
    )

    hs = structure_terminal["headline_metrics"]
    hc = chain["headline_metrics"]
    hg = genesis["headline_metrics"]

    structure_branching_direct = hs["terminal_structure_closure"] / (1.0 + hc["feature_lock_signal"])
    closure_binding_direct = hs["terminal_circuit_closure"] / (1.0 + hc["feature_selection_signal"])
    feedback_stability_direct = hs["terminal_feedback_closure"] * (
        1.0 + hg["genesis_margin"] / (1.0 + hs["terminal_structure_closure"])
    )
    structure_genesis_direct_core = (
        structure_branching_direct + closure_binding_direct + feedback_stability_direct
    )
    structure_direct_confidence = structure_genesis_direct_core / (1.0 + hs["terminal_closure_margin_v3"])

    return {
        "headline_metrics": {
            "structure_branching_direct": structure_branching_direct,
            "closure_binding_direct": closure_binding_direct,
            "feedback_stability_direct": feedback_stability_direct,
            "structure_genesis_direct_core": structure_genesis_direct_core,
            "structure_direct_confidence": structure_direct_confidence,
        },
        "direct_equation": {
            "branch_term": "S_branch = Tc_fs / (1 + N_lock)",
            "bind_term": "S_bind = Tc_fc / (1 + N_feat)",
            "feedback_term": "S_fb = Tc_fb * (1 + M_genesis / (1 + Tc_fs))",
            "core_term": "S_core = S_branch + S_bind + S_fb",
            "confidence_term": "C_struct = S_core / (1 + Tc_margin)",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 结构生成直测第二版报告",
        "",
        f"- structure_branching_direct: {hm['structure_branching_direct']:.6f}",
        f"- closure_binding_direct: {hm['closure_binding_direct']:.6f}",
        f"- feedback_stability_direct: {hm['feedback_stability_direct']:.6f}",
        f"- structure_genesis_direct_core: {hm['structure_genesis_direct_core']:.6f}",
        f"- structure_direct_confidence: {hm['structure_direct_confidence']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_structure_genesis_direct_measure_v2_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
