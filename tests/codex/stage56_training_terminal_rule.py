from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_training_terminal_rule_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_training_terminal_rule_summary() -> dict:
    falsi = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_falsifiability_closure_strengthening_20260320" / "summary.json"
    )
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )
    keep = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_high_retention_cross_version_validation_20260320" / "summary.json"
    )
    feas = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_architecture_feasibility_20260320" / "summary.json"
    )
    cross_modal = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_modal_unification_strengthening_20260320" / "summary.json"
    )

    hf = falsi["headline_metrics"]
    hc = cross["headline_metrics"]
    hk = keep["headline_metrics"]
    ha = feas["headline_metrics"]
    hm = cross_modal["headline_metrics"]

    terminal_update_strength = (
        hf["falsifiability_closure_stable"]
        + hc["cross_version_stability_stable"]
        + hk["cross_keep_core"]
    ) / 3.0
    terminal_stability_guard = 1.0 - min(
        1.0,
        (ha["rollback_penalty"] + hm["modality_residual_stable"] + hf["residual_nonfalsifiable_stable"]) / 3.0,
    )
    prototype_trainability = (
        ha["architecture_feasibility"]
        + hm["cross_modal_unification_stable"]
        + hf["falsifiability_closure_stable"]
    ) / 3.0
    training_terminal_readiness = (
        terminal_update_strength + terminal_stability_guard + prototype_trainability
    ) / 3.0
    terminal_training_gap = max(0.0, 1.0 - training_terminal_readiness)

    return {
        "headline_metrics": {
            "terminal_update_strength": terminal_update_strength,
            "terminal_stability_guard": terminal_stability_guard,
            "prototype_trainability": prototype_trainability,
            "training_terminal_readiness": training_terminal_readiness,
            "terminal_training_gap": terminal_training_gap,
        },
        "terminal_equation": {
            "update_term": "U_term = mean(C_false_star, S_cross, K_cross)",
            "guard_term": "G_term = 1 - mean(R_risk, R_mod_star, R_false_star)",
            "trainability_term": "T_term = mean(F_arch, T_cross_star, C_false_star)",
            "readiness_term": "R_train = mean(U_term, G_term, T_term)",
            "gap_term": "G_train = 1 - R_train",
        },
        "project_readout": {
            "summary": "训练终式块把更新强度、稳定守卫和原型可训练性压成了同一个对象，用来判断当前主线离可施工训练规则还有多远。",
            "next_question": "下一步要把这个训练终式对象推进到真实小型原型网络训练里，确认它能指导稳定收敛，而不只是停留在理论层。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 训练终式报告",
        "",
        f"- terminal_update_strength: {hm['terminal_update_strength']:.6f}",
        f"- terminal_stability_guard: {hm['terminal_stability_guard']:.6f}",
        f"- prototype_trainability: {hm['prototype_trainability']:.6f}",
        f"- training_terminal_readiness: {hm['training_terminal_readiness']:.6f}",
        f"- terminal_training_gap: {hm['terminal_training_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_training_terminal_rule_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
