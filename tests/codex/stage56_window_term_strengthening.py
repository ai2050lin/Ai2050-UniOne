from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from stage56_fullsample_regression_runner import fit_linear_regression, read_json, safe_float

ROOT = Path(__file__).resolve().parents[2]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def pair_key(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("model_id", "")),
        str(row.get("category", "")),
        str(row.get("prototype_term", "")),
        str(row.get("instance_term", "")),
    )


def build_rows(joined_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = {}
    for row in joined_rows:
        grouped.setdefault(pair_key(row), []).append(dict(row))

    out: List[Dict[str, object]] = []
    for key, rows in grouped.items():
        generated_window_mass = mean([safe_float(row.get("complete_generated_energy")) for row in rows])
        prompt_window_mass = mean([safe_float(row.get("complete_prompt_energy")) for row in rows])
        generated_window_gap = mean([safe_float(row.get("complete_energy_gap")) for row in rows])
        hidden_center_mean = mean([safe_float(row.get("hidden_window_center")) for row in rows])
        mlp_center_mean = mean([safe_float(row.get("mlp_window_center")) for row in rows])
        generated_dominance_mean = mean([safe_float(row.get("generated_dominance_score")) for row in rows])
        window_center_mean = mean([hidden_center_mean, mlp_center_mean])
        window_center_gap = abs(hidden_center_mean - mlp_center_mean)
        sample = rows[0]
        out.append(
            {
                "model_id": key[0],
                "category": key[1],
                "prototype_term": key[2],
                "instance_term": key[3],
                "generated_window_mass": generated_window_mass,
                "prompt_window_mass": prompt_window_mass,
                "generated_window_gap": generated_window_gap,
                "generated_dominance_mean": generated_dominance_mean,
                "window_center_mean": window_center_mean,
                "window_center_gap": window_center_gap,
                "union_joint_adv": safe_float(sample.get("union_joint_adv")),
                "union_synergy_joint": safe_float(sample.get("union_synergy_joint")),
                "strict_positive_synergy": 1.0 if bool(sample.get("strict_positive_synergy")) else 0.0,
            }
        )
    return out


def build_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    feature_names = [
        "generated_window_mass",
        "prompt_window_mass",
        "generated_window_gap",
        "generated_dominance_mean",
        "window_center_mean",
        "window_center_gap",
    ]
    fits = [
        fit_linear_regression(rows, feature_names, "union_joint_adv"),
        fit_linear_regression(rows, feature_names, "union_synergy_joint"),
        fit_linear_regression(rows, feature_names, "strict_positive_synergy"),
    ]
    return {
        "record_type": "stage56_window_term_strengthening_summary",
        "row_count": len(rows),
        "feature_names": feature_names,
        "mean_generated_window_gap": mean([safe_float(row.get("generated_window_gap")) for row in rows]),
        "mean_generated_dominance_mean": mean([safe_float(row.get("generated_dominance_mean")) for row in rows]),
        "fits": fits,
        "main_judgment": (
            "窗口项已经从简单中心位置推进到生成侧质量、提示侧质量、能量差和窗口一致性等更强测度，"
            "可以直接检查窗口层到底是在促进还是拖累闭包。"
        ),
    }


def build_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Stage56 窗口项强化摘要",
        "",
        f"- row_count: {summary.get('row_count', 0)}",
        f"- mean_generated_window_gap: {safe_float(summary.get('mean_generated_window_gap')):+.6f}",
        f"- mean_generated_dominance_mean: {safe_float(summary.get('mean_generated_dominance_mean')):+.6f}",
        f"- main_judgment: {summary.get('main_judgment', '')}",
        "",
        "## Fits",
    ]
    for fit in list(summary.get("fits", [])):
        fit = dict(fit)
        lines.append(f"- target: {fit.get('target_name', '')}")
        for key, value in dict(fit.get("weights", {})).items():
            lines.append(f"  {key}: {safe_float(value):+.6f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Strengthen window terms using complete highdim joined rows")
    ap.add_argument(
        "--complete-joined-json",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_complete_highdim_field_20260319_1645" / "joined_rows.json"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_window_term_strengthening_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    joined_rows = list(read_json(Path(args.complete_joined_json)).get("rows", []))
    rows = build_rows(joined_rows)
    summary = build_summary(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
