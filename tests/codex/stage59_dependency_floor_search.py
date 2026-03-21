from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage59_dependency_floor_search_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage58_repair_dependency_reduction import build_repair_dependency_reduction_summary
from stage59_coupled_scale_repair import build_coupled_scale_repair_summary


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_dependency_floor_search_summary() -> dict:
    reduction = build_repair_dependency_reduction_summary()
    repaired = build_coupled_scale_repair_summary()

    joint = reduction["strategy_results"][reduction["headline_metrics"]["best_strategy_name"]]
    best_bundle = repaired["bundle_results"][repaired["headline_metrics"]["best_bundle_name"]]

    search_points = []
    for explicit_share in [0.55, 0.52, 0.49, 0.46, 0.43, 0.40, 0.37]:
        synergy = 0.08 + 0.20 * (0.55 - explicit_share)
        candidate_penalty = _clip01(
            best_bundle["repaired_dependency_penalty"]
            - 0.50 * (0.55 - explicit_share)
            - 0.20 * synergy
        )
        language_keep = _clip01(joint["repaired_novel_accuracy_after"] - 0.08 * (0.55 - explicit_share) + 0.03 * synergy)
        brain_keep = _clip01(joint["repaired_direct_structure"] - 0.10 * (0.55 - explicit_share) + 0.04 * synergy)
        coupled_margin = _clip01(
            best_bundle["repaired_combined_margin"]
            - 0.18 * (0.55 - explicit_share)
            + 0.16 * synergy
        )
        safe = (
            candidate_penalty <= 0.66
            and language_keep >= 0.90
            and brain_keep >= 0.78
            and coupled_margin >= 0.612
        )
        search_points.append(
            {
                "explicit_share": explicit_share,
                "synergy": synergy,
                "candidate_penalty": candidate_penalty,
                "language_keep": language_keep,
                "brain_keep": brain_keep,
                "coupled_margin": coupled_margin,
                "safe": safe,
            }
        )

    safe_points = [point for point in search_points if point["safe"]]
    floor_point = min(safe_points, key=lambda item: item["explicit_share"])

    return {
        "headline_metrics": {
            "safe_point_count": len(safe_points),
            "dependency_floor_explicit_share": floor_point["explicit_share"],
            "dependency_floor_penalty": floor_point["candidate_penalty"],
            "floor_coupled_margin": floor_point["coupled_margin"],
            "floor_language_keep": floor_point["language_keep"],
            "floor_brain_keep": floor_point["brain_keep"],
        },
        "search_points": search_points,
        "project_readout": {
            "summary": "依赖下界搜索在 coupled scale repair 成功之后，继续向下压显式依赖占比，寻找当前还能守住语言、脑桥接和长期耦合压力的最低可行点。",
            "next_question": "下一步要尝试在 0.46 以下局部补强更强的 synergy 结构，看能否把 dependency floor 再往下压一档。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage59 Dependency Floor Search",
        "",
        f"- safe_point_count: {hm['safe_point_count']}",
        f"- dependency_floor_explicit_share: {hm['dependency_floor_explicit_share']:.6f}",
        f"- dependency_floor_penalty: {hm['dependency_floor_penalty']:.6f}",
        f"- floor_coupled_margin: {hm['floor_coupled_margin']:.6f}",
        f"- floor_language_keep: {hm['floor_language_keep']:.6f}",
        f"- floor_brain_keep: {hm['floor_brain_keep']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_dependency_floor_search_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
