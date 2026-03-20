from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_cross_scale_learning_equation_unification_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _norm_triplet(a: float, b: float, c: float) -> tuple[float, float, float]:
    total = a + b + c
    if total <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (a / total, b / total, c / total)


def build_cross_scale_unification_summary() -> dict:
    small = _load_json(ROOT / "tests" / "codex_temp" / "stage56_learning_equation_direct_fit_20260320" / "summary.json")
    large = _load_json(ROOT / "tests" / "codex_temp" / "stage56_large_model_learning_equation_bridge_20260320" / "summary.json")
    small_drives = small.get("drives", small)

    small_triplet = _norm_triplet(
        small_drives["atlas_learning_drive_v2"],
        small_drives["frontier_learning_drive_v2"],
        small_drives["closure_learning_drive_v2"],
    )
    large_triplet = _norm_triplet(
        large["headline_metrics"]["atlas_learning_drive_large"],
        large["headline_metrics"]["frontier_learning_drive_large"],
        large["headline_metrics"]["boundary_learning_drive_large"],
    )
    diff = [abs(a - b) for a, b in zip(small_triplet, large_triplet)]
    summary = {
        "headline_metrics": {
            "small_scale_triplet": {
                "atlas": small_triplet[0],
                "frontier": small_triplet[1],
                "boundary": small_triplet[2],
            },
            "large_scale_triplet": {
                "atlas": large_triplet[0],
                "frontier": large_triplet[1],
                "boundary": large_triplet[2],
            },
            "mean_absolute_gap": sum(diff) / 3.0,
            "same_ordering": sorted(range(3), key=lambda i: small_triplet[i], reverse=True)
            == sorted(range(3), key=lambda i: large_triplet[i], reverse=True),
        },
        "project_readout": {
            "summary": (
                "这一步直接比较小原型学习方程和大模型学习桥接的三元驱动比例。"
                "如果两边顺序一致且差距不大，就说明当前学习理论开始出现跨规模同构。"
            ),
            "next_question": "下一步要看这种同构能否继续跨到更原生变量，而不是只停在驱动比例层。",
        },
    }
    return summary


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 跨规模学习方程统一报告",
        "",
        f"- 小尺度 atlas/frontier/boundary: {hm['small_scale_triplet']['atlas']:.6f}, {hm['small_scale_triplet']['frontier']:.6f}, {hm['small_scale_triplet']['boundary']:.6f}",
        f"- 大尺度 atlas/frontier/boundary: {hm['large_scale_triplet']['atlas']:.6f}, {hm['large_scale_triplet']['frontier']:.6f}, {hm['large_scale_triplet']['boundary']:.6f}",
        f"- 平均绝对差距: {hm['mean_absolute_gap']:.6f}",
        f"- 顺序一致: {hm['same_ordering']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_cross_scale_unification_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
