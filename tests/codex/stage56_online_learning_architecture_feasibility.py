from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_online_learning_architecture_feasibility_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_online_learning_architecture_feasibility_summary() -> dict:
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320" / "summary.json"
    )
    keep = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_high_retention_cross_version_validation_20260320" / "summary.json"
    )
    theory = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_total_theory_bridge_expansion_20260320" / "summary.json"
    )

    hc = cross["headline_metrics"]
    hk = keep["headline_metrics"]
    ht = theory["headline_metrics"]

    language_capability_readiness = min(1.0, ht["dnn_to_brain_alignment"] / 2.0 + 0.35)
    online_stability_readiness = (hc["cross_version_stability_stable"] + hk["cross_keep_core"]) / 2.0
    rollback_penalty = hc["rollback_risk_reduced"]
    architecture_feasibility = (
        language_capability_readiness
        + online_stability_readiness
        + ht["total_bridge_strength_expanded"] / 2.0
    ) / 3.0
    production_gap = max(0.0, 1.0 - architecture_feasibility + rollback_penalty / 2.0)

    return {
        "headline_metrics": {
            "language_capability_readiness": language_capability_readiness,
            "online_stability_readiness": online_stability_readiness,
            "rollback_penalty": rollback_penalty,
            "architecture_feasibility": architecture_feasibility,
            "production_gap": production_gap,
        },
        "feasibility_equation": {
            "language_term": "R_lang = min(1, A_db / 2 + 0.35)",
            "online_term": "R_online = mean(S_cross_star, K_cross)",
            "risk_term": "R_risk = R_back_star",
            "feasibility_term": "F_arch = mean(R_lang, R_online, T_bridge_plus / 2)",
            "gap_term": "G_prod = 1 - F_arch + R_risk / 2",
        },
        "project_readout": {
            "summary": "即时学习网络可行性块把语言能力准备度、在线稳定度和总理论桥强度并成了一个更接近工程判断的对象。",
            "next_question": "下一步要把这个可行性对象和真正的小型原型网络训练结果对齐，而不是只停留在理论综合层。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 即时学习网络可行性报告",
        "",
        f"- language_capability_readiness: {hm['language_capability_readiness']:.6f}",
        f"- online_stability_readiness: {hm['online_stability_readiness']:.6f}",
        f"- rollback_penalty: {hm['rollback_penalty']:.6f}",
        f"- architecture_feasibility: {hm['architecture_feasibility']:.6f}",
        f"- production_gap: {hm['production_gap']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_online_learning_architecture_feasibility_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
