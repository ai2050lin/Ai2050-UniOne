from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage60_dependency_below_floor_probe_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage59_dependency_floor_search import build_dependency_floor_search_summary
from stage60_principled_coupled_scale_repair import build_principled_coupled_scale_repair_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_dependency_below_floor_probe_summary() -> dict:
    floor = build_dependency_floor_search_summary()["headline_metrics"]
    principled = build_principled_coupled_scale_repair_summary()["headline_metrics"]

    search_points = []
    for explicit_share in [0.46, 0.44, 0.43, 0.41, 0.39]:
        coherence_boost = 0.11 + 0.22 * (0.46 - explicit_share)
        candidate_penalty = _clip01(
            principled["best_principled_dependency_penalty"]
            - 0.42 * (0.46 - explicit_share)
            - 0.18 * coherence_boost
        )
        coupled_margin = _clip01(
            principled["best_principled_combined_margin"]
            - 0.17 * (0.46 - explicit_share)
            + 0.12 * coherence_boost
        )
        language_keep = _clip01(
            floor["floor_language_keep"]
            - 0.05 * (0.46 - explicit_share)
            + 0.04 * coherence_boost
        )
        brain_keep = _clip01(
            floor["floor_brain_keep"]
            - 0.06 * (0.46 - explicit_share)
            + 0.05 * coherence_boost
        )
        safe = (
            candidate_penalty <= 0.62
            and coupled_margin >= 0.618
            and language_keep >= 0.902
            and brain_keep >= 0.782
        )
        search_points.append(
            {
                "explicit_share": explicit_share,
                "coherence_boost": coherence_boost,
                "candidate_penalty": candidate_penalty,
                "coupled_margin": coupled_margin,
                "language_keep": language_keep,
                "brain_keep": brain_keep,
                "safe": safe,
            }
        )

    safe_points = [point for point in search_points if point["safe"]]
    best_floor = min(safe_points, key=lambda item: item["explicit_share"])

    return {
        "headline_metrics": {
            "safe_point_count": len(safe_points),
            "new_dependency_floor_explicit_share": best_floor["explicit_share"],
            "new_dependency_floor_penalty": best_floor["candidate_penalty"],
            "new_floor_coupled_margin": best_floor["coupled_margin"],
            "new_floor_language_keep": best_floor["language_keep"],
            "new_floor_brain_keep": best_floor["brain_keep"],
        },
        "search_points": search_points,
        "project_readout": {
            "summary": "依赖地板下探在 principled bundle 的帮助下继续往 0.46 以下搜索，目标是验证原理化修复是否真的能把显式依赖继续压低一档。",
            "next_question": "下一步要继续测试 0.43 以下是否还可以通过更强的系数落地把脑桥接一起稳住。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage60 Dependency Below Floor Probe",
        "",
        f"- safe_point_count: {hm['safe_point_count']}",
        f"- new_dependency_floor_explicit_share: {hm['new_dependency_floor_explicit_share']:.6f}",
        f"- new_dependency_floor_penalty: {hm['new_dependency_floor_penalty']:.6f}",
        f"- new_floor_coupled_margin: {hm['new_floor_coupled_margin']:.6f}",
        f"- new_floor_language_keep: {hm['new_floor_language_keep']:.6f}",
        f"- new_floor_brain_keep: {hm['new_floor_brain_keep']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_dependency_below_floor_probe_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
