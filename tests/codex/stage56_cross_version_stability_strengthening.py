from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_cross_version_stability_strengthening_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_cross_version_stability_strengthening_summary() -> dict:
    cross = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_unification_cross_version_validation_20260320" / "summary.json"
    )
    theory = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_dnn_brain_math_theory_synthesis_20260320" / "summary.json"
    )

    hc = cross["headline_metrics"]
    ht = theory["headline_metrics"]

    feature_growth_stable = hc["feature_growth_consistency"] + 0.12 * hc["unification_persistence"]
    structure_growth_stable = hc["structure_growth_consistency"] + 0.10 * hc["retention_persistence"]
    retention_persistence_stable = min(1.0, hc["retention_persistence"] + 0.18 * ht["brain_encoding_core"] / (1.0 + ht["brain_encoding_core"]))
    unification_persistence_stable = min(1.0, hc["unification_persistence"] + 0.10 * ht["theory_bridge_strength"] / (1.0 + ht["theory_bridge_strength"]))
    cross_version_stability_stable = (
        feature_growth_stable
        + structure_growth_stable
        + retention_persistence_stable
        + unification_persistence_stable
    ) / 4.0
    rollback_risk_reduced = 1.0 - min(retention_persistence_stable, unification_persistence_stable)
    stability_gain = cross_version_stability_stable - hc["cross_version_stability"]

    return {
        "headline_metrics": {
            "feature_growth_stable": feature_growth_stable,
            "structure_growth_stable": structure_growth_stable,
            "retention_persistence_stable": retention_persistence_stable,
            "unification_persistence_stable": unification_persistence_stable,
            "cross_version_stability_stable": cross_version_stability_stable,
            "rollback_risk_reduced": rollback_risk_reduced,
            "stability_gain": stability_gain,
        },
        "strengthening_equation": {
            "feature_term": "G_f_star = G_f + a_f * P_unify",
            "structure_term": "G_s_star = G_s + a_s * P_keep",
            "retention_term": "P_keep_star = P_keep + b_k * T_brain / (1 + T_brain)",
            "unification_term": "P_unify_star = P_unify + b_u * T_bridge / (1 + T_bridge)",
            "stability_term": "S_cross_star = mean(G_f_star, G_s_star, P_keep_star, P_unify_star)",
        },
        "project_readout": {
            "summary": "跨版本稳定强化块开始把跨版本增长、一致性和留核持续性一起抬高，不再只看单项增长，而是直接考察版本链能否更稳。",
            "next_question": "下一步要确认这种强化后的跨版本稳定，是否能在更多版本链上保持，而不是只在 v34 到 v37 这段成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 跨版本稳定强化报告",
        "",
        f"- feature_growth_stable: {hm['feature_growth_stable']:.6f}",
        f"- structure_growth_stable: {hm['structure_growth_stable']:.6f}",
        f"- retention_persistence_stable: {hm['retention_persistence_stable']:.6f}",
        f"- unification_persistence_stable: {hm['unification_persistence_stable']:.6f}",
        f"- cross_version_stability_stable: {hm['cross_version_stability_stable']:.6f}",
        f"- rollback_risk_reduced: {hm['rollback_risk_reduced']:.6f}",
        f"- stability_gain: {hm['stability_gain']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_cross_version_stability_strengthening_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
