from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_prototype_network_readiness_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_prototype_network_readiness_summary() -> dict:
    train = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_rule_20260320" / "summary.json"
    )
    feas = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_architecture_feasibility_20260320" / "summary.json"
    )
    cross_modal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_modal_unification_strengthening_20260320" / "summary.json"
    )
    falsi = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_falsifiability_closure_strengthening_20260320" / "summary.json"
    )

    ht = train["headline_metrics"]
    hf = feas["headline_metrics"]
    hm = cross_modal["headline_metrics"]
    hc = falsi["headline_metrics"]

    language_stack_readiness = (
        hf["language_capability_readiness"]
        + ht["prototype_trainability"]
        + hm["language_to_general_stable"]
    ) / 3.0
    online_learning_readiness = (
        hf["online_stability_readiness"]
        + ht["terminal_stability_guard"]
        + hc["falsifiability_closure_stable"]
    ) / 3.0
    prototype_network_readiness = (
        language_stack_readiness + online_learning_readiness + ht["training_terminal_readiness"]
    ) / 3.0
    agi_delivery_gap = max(0.0, 1.0 - prototype_network_readiness + hf["rollback_penalty"] / 3.0)

    return {
        "headline_metrics": {
            "language_stack_readiness": language_stack_readiness,
            "online_learning_readiness": online_learning_readiness,
            "prototype_network_readiness": prototype_network_readiness,
            "agi_delivery_gap": agi_delivery_gap,
        },
        "readiness_equation": {
            "language_term": "R_lang_stack = mean(R_lang, T_term, T_lang_star)",
            "online_term": "R_online_stack = mean(R_online, G_term, C_false_star)",
            "prototype_term": "R_proto = mean(R_lang_stack, R_online_stack, R_train)",
            "gap_term": "G_agi = 1 - R_proto + R_risk / 3",
        },
        "project_readout": {
            "summary": "原型网络就绪度块把语言能力、即时学习稳定性和训练终式准备度压成了统一对象，用来判断当前是否适合启动新的实验网络。",
            "next_question": "下一步要把这个就绪度对象和真实小型网络训练结果对齐，确认理论准备度能否转成工程准备度。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 原型网络就绪度报告",
        "",
        f"- language_stack_readiness: {hm['language_stack_readiness']:.6f}",
        f"- online_learning_readiness: {hm['online_learning_readiness']:.6f}",
        f"- prototype_network_readiness: {hm['prototype_network_readiness']:.6f}",
        f"- agi_delivery_gap: {hm['agi_delivery_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_prototype_network_readiness_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
