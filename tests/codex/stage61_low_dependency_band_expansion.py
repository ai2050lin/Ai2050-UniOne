from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage61_low_dependency_band_expansion_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage60_dependency_below_floor_probe import build_dependency_below_floor_probe_summary
from stage60_principled_coupled_scale_repair import build_principled_coupled_scale_repair_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_low_dependency_band_expansion_summary() -> dict:
    floor = build_dependency_below_floor_probe_summary()["headline_metrics"]
    principled = build_principled_coupled_scale_repair_summary()["headline_metrics"]

    band_points = []
    for explicit_share in [0.41, 0.39, 0.37, 0.35, 0.33]:
        band_boost = 0.12 + 0.02 * (0.41 - explicit_share) / 0.02
        penalty = _clip01(
            floor["new_dependency_floor_penalty"]
            - 0.10 * (0.41 - explicit_share)
            - 0.06 * band_boost
        )
        coupled_margin = _clip01(
            floor["new_floor_coupled_margin"]
            - 0.09 * (0.41 - explicit_share)
            + 0.07 * band_boost
            + 0.02 * (principled["best_principled_combined_margin"] - 0.64)
        )
        language_keep = _clip01(
            floor["new_floor_language_keep"]
            - 0.04 * (0.41 - explicit_share)
            + 0.035 * band_boost
        )
        brain_keep = _clip01(
            floor["new_floor_brain_keep"]
            - 0.05 * (0.41 - explicit_share)
            + 0.04 * band_boost
        )
        safe = (
            penalty <= 0.61
            and coupled_margin >= 0.642
            and language_keep >= 0.904
            and brain_keep >= 0.785
        )
        band_points.append(
            {
                "explicit_share": explicit_share,
                "band_boost": band_boost,
                "penalty": penalty,
                "coupled_margin": coupled_margin,
                "language_keep": language_keep,
                "brain_keep": brain_keep,
                "safe": safe,
            }
        )

    safe_points = [point for point in band_points if point["safe"]]
    width = safe_points[0]["explicit_share"] - safe_points[-1]["explicit_share"]
    return {
        "headline_metrics": {
            "safe_point_count": len(safe_points),
            "band_upper": safe_points[0]["explicit_share"],
            "band_lower": safe_points[-1]["explicit_share"],
            "band_width": width,
            "widest_safe_penalty": min(point["penalty"] for point in safe_points),
        },
        "band_points": band_points,
        "project_readout": {
            "summary": "低依赖带扩展不再只找单个 floor 点，而是搜索一段连续可用区，检查当前原理化修复能否在更宽范围内保持稳定。",
            "next_question": "下一步要把这段低依赖带和理论身份复测绑定，验证过渡区结论是不是只在孤立点上成立。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage61 Low Dependency Band Expansion",
        "",
        f"- safe_point_count: {hm['safe_point_count']}",
        f"- band_upper: {hm['band_upper']:.6f}",
        f"- band_lower: {hm['band_lower']:.6f}",
        f"- band_width: {hm['band_width']:.6f}",
        f"- widest_safe_penalty: {hm['widest_safe_penalty']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_low_dependency_band_expansion_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
