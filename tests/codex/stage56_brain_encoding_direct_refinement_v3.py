from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v3_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_brain_encoding_direct_refinement_v3_summary() -> dict:
    v2 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v2_20260321" / "summary.json"
    )
    horizon = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_long_horizon_stability_20260321" / "summary.json"
    )
    online = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_online_learning_rollback_probe_20260321" / "summary.json"
    )
    proto = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_contextual_trainable_prototype_20260320" / "summary.json"
    )

    hv2 = v2["headline_metrics"]
    hh = horizon["headline_metrics"]
    ho = online["headline_metrics"]
    hp = proto["headline_metrics"]

    direct_origin_measure_v3 = _clip01((hv2["direct_origin_measure_v2"] + hh["long_horizon_retention"]) / 2.0)
    direct_feature_measure_v3 = _clip01((hv2["direct_feature_measure_v2"] + hh["shared_fiber_survival"] + ho["route_split_retention"]) / 3.0)
    direct_structure_measure_v3 = _clip01((hv2["direct_structure_measure_v2"] + hh["structural_survival"] + hp["route_split_consistency"]) / 3.0)
    direct_route_measure_v3 = _clip01((hv2["direct_route_measure_v2"] + hh["contextual_survival"] + hp["context_split_consistency"]) / 3.0)
    direct_brain_measure_v3 = (
        direct_origin_measure_v3
        + direct_feature_measure_v3
        + direct_structure_measure_v3
        + direct_route_measure_v3
    ) / 4.0
    direct_brain_gap_v3 = 1.0 - direct_brain_measure_v3
    dynamic_structure_balance_v3 = (direct_structure_measure_v3 + direct_route_measure_v3 + hh["structural_survival"]) / 3.0

    return {
        "headline_metrics": {
            "direct_origin_measure_v3": direct_origin_measure_v3,
            "direct_feature_measure_v3": direct_feature_measure_v3,
            "direct_structure_measure_v3": direct_structure_measure_v3,
            "direct_route_measure_v3": direct_route_measure_v3,
            "direct_brain_measure_v3": direct_brain_measure_v3,
            "direct_brain_gap_v3": direct_brain_gap_v3,
            "dynamic_structure_balance_v3": dynamic_structure_balance_v3,
        },
        "direct_equation_v3": {
            "origin_term": "D_origin_v3 = mean(D_origin_v2, R_h)",
            "feature_term": "D_feature_v3 = mean(D_feature_v2, H_fiber, R_route)",
            "structure_term": "D_structure_v3 = mean(D_structure_v2, H_structure, route_split_consistency)",
            "route_term": "D_route_v3 = mean(D_route_v2, H_context, context_split_consistency)",
            "system_term": "M_brain_direct_v3 = mean(D_origin_v3, D_feature_v3, D_structure_v3, D_route_v3)",
        },
        "project_readout": {
            "summary": "逆向脑编码直测强化第三版把长时间尺度在线稳定性并回直测链，使结构层和路线层不再只依赖单轮一致性，而开始受动态生存率约束。",
            "next_question": "下一步要把这个直测强化链继续推进到更接近原生回路记录，否则它仍然主要是动态代理量而不是真实直测量。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 逆向脑编码直测强化第三版报告",
        "",
        f"- direct_origin_measure_v3: {hm['direct_origin_measure_v3']:.6f}",
        f"- direct_feature_measure_v3: {hm['direct_feature_measure_v3']:.6f}",
        f"- direct_structure_measure_v3: {hm['direct_structure_measure_v3']:.6f}",
        f"- direct_route_measure_v3: {hm['direct_route_measure_v3']:.6f}",
        f"- direct_brain_measure_v3: {hm['direct_brain_measure_v3']:.6f}",
        f"- direct_brain_gap_v3: {hm['direct_brain_gap_v3']:.6f}",
        f"- dynamic_structure_balance_v3: {hm['dynamic_structure_balance_v3']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_brain_encoding_direct_refinement_v3_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
