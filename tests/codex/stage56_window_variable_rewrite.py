from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def average(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = average(xs)
    mean_y = average(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    cov = sum(a * b for a, b in zip(dx, dy))
    std_x = sum(a * a for a in dx) ** 0.5
    std_y = sum(b * b for b in dy) ** 0.5
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    return float(cov / (std_x * std_y))


def direction_label(value: float, threshold: float = 0.12) -> str:
    if value > threshold:
        return "positive"
    if value < -threshold:
        return "negative"
    return "neutral"


def contiguous_windows(length: int, min_width: int, max_width: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for width in range(max(1, min_width), max(1, max_width) + 1):
        if width > length:
            continue
        for start in range(0, length - width + 1):
            out.append((start, start + width))
    return out


def window_label(labels: Sequence[str], start: int, end: int) -> str:
    if not labels:
        return f"{start}:{end}"
    return f"{labels[start]}..{labels[end - 1]}"


def window_sum(profile: Sequence[float], start: int, end: int) -> float:
    return float(sum(abs(safe_float(value)) for value in profile[start:end]))


def top_weighted_rows(rows: Sequence[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            safe_float(row.get("weight")),
            safe_float(row.get("union_synergy_joint")),
            safe_float(row.get("union_joint_adv")),
        ),
        reverse=True,
    )
    return ordered[:limit]


def scan_component_windows(
    rows: Sequence[Dict[str, object]],
    profile_key: str,
    target_key: str,
    min_width: int,
    max_width: int,
) -> Dict[str, object]:
    if not rows:
        return {
            "best_window": "none",
            "best_corr": 0.0,
            "direction": "neutral",
            "window_mean": 0.0,
        }
    labels = list(rows[0].get("tail_position_labels", []))
    width = len(labels)
    best = {
        "best_window": "none",
        "best_corr": 0.0,
        "direction": "neutral",
        "window_mean": 0.0,
    }
    best_abs_corr = -1.0
    for start, end in contiguous_windows(width, min_width=min_width, max_width=max_width):
        xs = [window_sum(list(row.get(profile_key, [])), start, end) for row in rows]
        ys = [safe_float(row.get(target_key)) for row in rows]
        corr = pearson_corr(xs, ys)
        abs_corr = abs(corr)
        if abs_corr > best_abs_corr:
            best_abs_corr = abs_corr
            best = {
                "best_window": window_label(labels, start, end),
                "best_corr": corr,
                "direction": direction_label(corr),
                "window_mean": average(xs),
            }
    return best


def summarize_component(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    positive_rows = [row for row in rows if safe_float(row.get("weight")) > 0.0]
    return {
        "case_count": len(positive_rows),
        "mean_union_synergy_joint": average([safe_float(row.get("union_synergy_joint")) for row in positive_rows]),
        "mean_union_joint_adv": average([safe_float(row.get("union_joint_adv")) for row in positive_rows]),
        "hidden_to_synergy": scan_component_windows(
            positive_rows,
            profile_key="hidden_token_profile",
            target_key="union_synergy_joint",
            min_width=2,
            max_width=4,
        ),
        "mlp_to_synergy": scan_component_windows(
            positive_rows,
            profile_key="mlp_token_profile",
            target_key="union_synergy_joint",
            min_width=2,
            max_width=4,
        ),
        "hidden_to_joint_adv": scan_component_windows(
            positive_rows,
            profile_key="hidden_token_profile",
            target_key="union_joint_adv",
            min_width=2,
            max_width=4,
        ),
        "mlp_to_joint_adv": scan_component_windows(
            positive_rows,
            profile_key="mlp_token_profile",
            target_key="union_joint_adv",
            min_width=2,
            max_width=4,
        ),
        "top_cases": [
            {
                "model_id": str(row["model_id"]),
                "category": str(row["category"]),
                "prototype_term": str(row["prototype_term"]),
                "instance_term": str(row["instance_term"]),
                "weight": safe_float(row["weight"]),
                "union_synergy_joint": safe_float(row["union_synergy_joint"]),
            }
            for row in top_weighted_rows(positive_rows, limit=8)
        ],
    }


def build_summary(joined_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    component_labels = sorted({str(row["component_label"]) for row in joined_rows})
    return {
        "record_type": "stage56_window_variable_rewrite_summary",
        "joined_row_count": len(joined_rows),
        "component_labels": component_labels,
        "per_component": {
            label: summarize_component([row for row in joined_rows if str(row["component_label"]) == label])
            for label in component_labels
        },
    }


def write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Stage56 窗口变量重写报告",
        "",
        f"- joined_row_count: {summary['joined_row_count']}",
        "",
        "## Per Component",
    ]
    for label in summary["component_labels"]:
        row = dict(summary["per_component"][label])
        hidden_syn = dict(row["hidden_to_synergy"])
        mlp_syn = dict(row["mlp_to_synergy"])
        lines.append(
            f"- {label}: "
            f"cases={row['case_count']}, "
            f"hidden_synergy={hidden_syn['best_window']} ({hidden_syn['best_corr']:.4f}), "
            f"mlp_synergy={mlp_syn['best_window']} ({mlp_syn['best_corr']:.4f}), "
            f"mean_synergy={row['mean_union_synergy_joint']:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rewrite token totals into window variables for closure analysis")
    ap.add_argument(
        "--joined-rows-json",
        default=str(
            ROOT
            / "tests"
            / "codex_temp"
            / "stage56_component_trajectory_window_map_all3_12cat_allpairs_20260319_0137"
            / "joined_rows.json"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "codex_temp" / "stage56_window_variable_rewrite_20260319"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    joined_rows = list(read_json(Path(args.joined_rows_json)).get("rows", []))
    summary = build_summary(joined_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "joined_row_count": summary["joined_row_count"],
                "component_labels": summary["component_labels"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
