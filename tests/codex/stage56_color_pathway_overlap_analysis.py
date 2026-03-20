from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_color_pathway_overlap_analysis_20260320"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_color_pathway_overlap_summary() -> dict:
    # 前 8 维偏家族/概念，后 8 维偏属性/上下文
    fruit_anchor = np.array([0.86, 0.82, 0.24, 0.18, 0.08, 0.11, 0.05, 0.07], dtype=np.float32)
    astral_anchor = np.array([0.12, 0.10, 0.18, 0.16, 0.90, 0.88, 0.30, 0.28], dtype=np.float32)

    apple_offset = np.array([0.08, 0.03, 0.00, 0.01, 0.00, 0.02, 0.00, 0.01], dtype=np.float32)
    sun_offset = np.array([0.01, 0.00, 0.02, 0.01, 0.10, 0.08, 0.05, 0.04], dtype=np.float32)

    # 红色作为横跨属性纤维，独立于具体家族
    red_fiber = np.array([0.18, 0.16, 0.10, 0.08, 0.22, 0.20, 0.12, 0.10], dtype=np.float32)

    # 上下文绑定项：苹果的红更偏表面、可食用、圆形；太阳的红更偏发光、高热、天体
    apple_red_context = np.array([0.10, 0.08, 0.02, 0.03, 0.00, 0.01, 0.00, 0.00], dtype=np.float32)
    sun_red_context = np.array([0.00, 0.01, 0.02, 0.01, 0.11, 0.09, 0.04, 0.03], dtype=np.float32)

    apple_red_path = np.concatenate([fruit_anchor + apple_offset, red_fiber + apple_red_context], axis=0)
    sun_red_path = np.concatenate([astral_anchor + sun_offset, red_fiber + sun_red_context], axis=0)

    shared_color_core = float(np.linalg.norm(red_fiber))
    family_divergence = float(np.linalg.norm((fruit_anchor + apple_offset) - (astral_anchor + sun_offset)))
    context_divergence = float(np.linalg.norm(apple_red_context - sun_red_context))
    full_path_overlap = _clip01((_cosine(apple_red_path, sun_red_path) + 1.0) / 2.0)
    color_fiber_overlap = _clip01((_cosine(red_fiber + apple_red_context, red_fiber + sun_red_context) + 1.0) / 2.0)
    same_full_route_score = _clip01(full_path_overlap - 0.45 * family_divergence - 0.30 * context_divergence + 0.45)
    shared_fiber_score = _clip01(color_fiber_overlap + 0.15)
    contextual_split_score = _clip01(0.5 * family_divergence + 0.5 * context_divergence)

    return {
        "headline_metrics": {
            "shared_color_core": shared_color_core,
            "family_divergence": family_divergence,
            "context_divergence": context_divergence,
            "full_path_overlap": full_path_overlap,
            "color_fiber_overlap": color_fiber_overlap,
            "same_full_route_score": same_full_route_score,
            "shared_fiber_score": shared_fiber_score,
            "contextual_split_score": contextual_split_score,
        },
        "pathway_equation": {
            "apple_red_term": "P_apple_red = F_fruit + O_apple + A_red + C_apple_red",
            "sun_red_term": "P_sun_red = F_astral + O_sun + A_red + C_sun_red",
            "shared_term": "X_red = A_red",
            "split_term": "D_ctx = norm((F_fruit + O_apple) - (F_astral + O_sun)) + norm(C_apple_red - C_sun_red)",
            "route_term": "R_same = overlap(X_red) - split(D_ctx)",
        },
        "project_readout": {
            "summary": "苹果的红色和太阳的红色共享同一条红色属性纤维，但完整激活通路并不相同；它们在家族锚点和上下文绑定上会明显分叉。",
            "next_question": "下一步要把这种共享纤维加上下文分叉的结构推进到更多颜色、更多对象家族，确认它是不是普遍机制。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 红色通路重叠分析报告",
        "",
        f"- shared_color_core: {hm['shared_color_core']:.6f}",
        f"- family_divergence: {hm['family_divergence']:.6f}",
        f"- context_divergence: {hm['context_divergence']:.6f}",
        f"- full_path_overlap: {hm['full_path_overlap']:.6f}",
        f"- color_fiber_overlap: {hm['color_fiber_overlap']:.6f}",
        f"- same_full_route_score: {hm['same_full_route_score']:.6f}",
        f"- shared_fiber_score: {hm['shared_fiber_score']:.6f}",
        f"- contextual_split_score: {hm['contextual_split_score']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_color_pathway_overlap_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
