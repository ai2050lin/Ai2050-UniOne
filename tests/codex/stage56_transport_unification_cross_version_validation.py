from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_transport_unification_cross_version_validation_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_transport_unification_cross_version_validation_summary() -> dict:
    v34 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v34_20260320" / "summary.json"
    )
    v35 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v35_20260320" / "summary.json"
    )
    v36 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_encoding_mechanism_closed_form_v36_20260320" / "summary.json"
    )
    r35 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_retention_reinforcement_20260320" / "summary.json"
    )
    r36 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_transport_kernel_stability_strengthening_20260320" / "summary.json"
    )
    u35 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_icspb_unification_reinforcement_20260320" / "summary.json"
    )
    u36 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_icspb_unification_high_closure_20260320" / "summary.json"
    )

    hv34 = v34["headline_metrics"]
    hv35 = v35["headline_metrics"]
    hv36 = v36["headline_metrics"]
    hr35 = r35["headline_metrics"]
    hr36 = r36["headline_metrics"]
    hu35 = u35["headline_metrics"]
    hu36 = u36["headline_metrics"]

    feature_growth_consistency = (hv35["feature_term_v35"] - hv34["feature_term_v34"]) / hv34["feature_term_v34"]
    structure_growth_consistency = (hv36["structure_term_v36"] - hv35["structure_term_v35"]) / hv35["structure_term_v35"]
    retention_persistence = (hr35["transport_kernel_stability_reinforced"] + hr36["transport_kernel_stability_stable"]) / 2.0
    unification_persistence = (hu35["unification_stability_reinforced"] + hu36["unification_high_stability"]) / 2.0
    cross_version_stability = (
        feature_growth_consistency + structure_growth_consistency + retention_persistence + unification_persistence
    ) / 4.0
    rollback_risk = 1.0 - min(retention_persistence, unification_persistence)

    return {
        "headline_metrics": {
            "feature_growth_consistency": feature_growth_consistency,
            "structure_growth_consistency": structure_growth_consistency,
            "retention_persistence": retention_persistence,
            "unification_persistence": unification_persistence,
            "cross_version_stability": cross_version_stability,
            "rollback_risk": rollback_risk,
        },
        "validation_equation": {
            "feature_term": "G_f = (K_f_v35 - K_f_v34) / K_f_v34",
            "structure_term": "G_s = (K_s_v36 - K_s_v35) / K_s_v35",
            "retention_term": "P_keep = mean(K_keep_plus, K_keep_star)",
            "unification_term": "P_unify = mean(S_unify_plus, S_unify_high)",
            "stability_term": "S_cross = mean(G_f, G_s, P_keep, P_unify)",
        },
        "project_readout": {
            "summary": "跨版本验证块开始检查高闭合和高留核是不是能连续保持，而不只是单版局部强化。",
            "next_question": "下一步要把这种跨版本稳定性并回主核，避免主核只在单一版本上成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 跨版本稳定性验证报告",
        "",
        f"- feature_growth_consistency: {hm['feature_growth_consistency']:.6f}",
        f"- structure_growth_consistency: {hm['structure_growth_consistency']:.6f}",
        f"- retention_persistence: {hm['retention_persistence']:.6f}",
        f"- unification_persistence: {hm['unification_persistence']:.6f}",
        f"- cross_version_stability: {hm['cross_version_stability']:.6f}",
        f"- rollback_risk: {hm['rollback_risk']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_transport_unification_cross_version_validation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
