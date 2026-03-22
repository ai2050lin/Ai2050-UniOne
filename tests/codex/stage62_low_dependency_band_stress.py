from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage62_low_dependency_band_stress_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage61_low_dependency_band_expansion import build_low_dependency_band_expansion_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_low_dependency_band_stress_summary() -> dict:
    band = build_low_dependency_band_expansion_summary()["band_points"]

    stressed_points = []
    for point in band:
        delta = 0.41 - point["explicit_share"]
        stressed_penalty = _clip01(point["penalty"] + 0.002 + 0.020 * delta)
        stressed_margin = _clip01(point["coupled_margin"] - 0.003 - 0.045 * delta)
        stressed_language_keep = _clip01(point["language_keep"] - 0.001 - 0.015 * delta)
        stressed_brain_keep = _clip01(point["brain_keep"] - 0.0015 - 0.012 * delta)
        stressed_safe = (
            stressed_penalty <= 0.602
            and stressed_margin >= 0.6465
            and stressed_language_keep >= 0.9075
            and stressed_brain_keep >= 0.7875
        )
        stressed_points.append(
            {
                "explicit_share": point["explicit_share"],
                "stressed_penalty": stressed_penalty,
                "stressed_margin": stressed_margin,
                "stressed_language_keep": stressed_language_keep,
                "stressed_brain_keep": stressed_brain_keep,
                "stressed_safe": stressed_safe,
            }
        )

    safe_points = [point for point in stressed_points if point["stressed_safe"]]
    stressed_band_upper = safe_points[0]["explicit_share"]
    stressed_band_lower = safe_points[-1]["explicit_share"]
    stressed_band_width = stressed_band_upper - stressed_band_lower
    band_resilience_score = _clip01(
        0.34 * (len(safe_points) / len(stressed_points))
        + 0.26 * (sum(point["stressed_margin"] for point in safe_points) / len(safe_points))
        + 0.20 * (1.0 - min(point["stressed_penalty"] for point in safe_points))
        + 0.20 * (sum(point["stressed_brain_keep"] for point in safe_points) / len(safe_points))
    )

    return {
        "headline_metrics": {
            "stressed_safe_point_count": len(safe_points),
            "stressed_band_upper": stressed_band_upper,
            "stressed_band_lower": stressed_band_lower,
            "stressed_band_width": stressed_band_width,
            "band_resilience_score": band_resilience_score,
        },
        "stressed_points": stressed_points,
        "project_readout": {
            "summary": "低依赖带测压把 stage61 的连续安全带放到更强扰动下，检查这条带到底是真稳，还是一压就只剩窄条。",
            "next_question": "下一步要把这条带和唯一化加固一起并回理论身份，看过渡区是不是在扰动下仍能守住。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage62 Low Dependency Band Stress",
        "",
        f"- stressed_safe_point_count: {hm['stressed_safe_point_count']}",
        f"- stressed_band_upper: {hm['stressed_band_upper']:.6f}",
        f"- stressed_band_lower: {hm['stressed_band_lower']:.6f}",
        f"- stressed_band_width: {hm['stressed_band_width']:.6f}",
        f"- band_resilience_score: {hm['band_resilience_score']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_low_dependency_band_stress_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
