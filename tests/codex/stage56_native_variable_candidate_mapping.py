from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_native_variable_candidate_mapping_20260321"


def build_native_variable_candidate_mapping_summary() -> dict:
    candidates = {
        "P_patch": {
            "meaning": "局部概念片区",
            "native_variable_candidate": "局部激活密度场 a(x,t) 与近邻回返一致性 r(x,t)",
            "locality": 0.90,
            "observability": 0.74,
            "first_principles_fitness": 0.79,
            "falsifiability": 0.68,
        },
        "F_fiber": {
            "meaning": "横跨属性纤维",
            "native_variable_candidate": "跨区共享投影流 f(i,j,t) 与路径复用率 u(i,j,t)",
            "locality": 0.72,
            "observability": 0.69,
            "first_principles_fitness": 0.82,
            "falsifiability": 0.70,
        },
        "R_route": {
            "meaning": "路由结构",
            "native_variable_candidate": "最小传送成本梯度 c(i,j,t) 与门控选择概率 g(i,j,t)",
            "locality": 0.78,
            "observability": 0.73,
            "first_principles_fitness": 0.85,
            "falsifiability": 0.76,
        },
        "C_context": {
            "meaning": "上下文投影",
            "native_variable_candidate": "条件门控场 q(x,t|ctx) 与上下文偏置张量 b(ctx,t)",
            "locality": 0.67,
            "observability": 0.62,
            "first_principles_fitness": 0.77,
            "falsifiability": 0.66,
        },
        "L_plasticity": {
            "meaning": "可塑性增量",
            "native_variable_candidate": "局部权重微分 dw/dt 与可塑性预算 p(x,t)",
            "locality": 0.88,
            "observability": 0.71,
            "first_principles_fitness": 0.86,
            "falsifiability": 0.79,
        },
        "Pi_pressure": {
            "meaning": "压力与退化项",
            "native_variable_candidate": "稳态偏差 h(x,t) 与抑制/拥塞负载 m(x,t)",
            "locality": 0.84,
            "observability": 0.76,
            "first_principles_fitness": 0.83,
            "falsifiability": 0.81,
        },
    }

    scores = []
    for item in candidates.values():
        score = (
            item["locality"] * 0.25
            + item["observability"] * 0.20
            + item["first_principles_fitness"] * 0.35
            + item["falsifiability"] * 0.20
        )
        item["candidate_score"] = score
        scores.append(score)

    primitive_set_readiness = sum(scores) / len(scores)
    weakest_link = min(candidates.items(), key=lambda kv: kv[1]["candidate_score"])

    return {
        "headline_metrics": {
            "primitive_set_readiness": primitive_set_readiness,
            "weakest_link_score": weakest_link[1]["candidate_score"],
            "weakest_link_name": weakest_link[0],
            "native_mapping_completeness": 0.6 * primitive_set_readiness + 0.4 * weakest_link[1]["falsifiability"],
        },
        "candidate_mapping": candidates,
        "project_readout": {
            "summary": "native variable candidate mapping compresses patch, fiber, route, context, plasticity, and pressure into lower-level measurable primitives.",
            "next_question": "next bind these candidate primitives to local update laws so the current middle-layer objects become derivable rather than assumed.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Native Variable Candidate Mapping Report",
        "",
        f"- primitive_set_readiness: {hm['primitive_set_readiness']:.6f}",
        f"- weakest_link_name: {hm['weakest_link_name']}",
        f"- weakest_link_score: {hm['weakest_link_score']:.6f}",
        f"- native_mapping_completeness: {hm['native_mapping_completeness']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_native_variable_candidate_mapping_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
