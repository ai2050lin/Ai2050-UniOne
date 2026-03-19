from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from stage56_natural_generation_decoupling import (
    attach_zone_metrics,
    join_component_rows,
    read_json,
    read_jsonl,
    safe_float,
)


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


def split_windows(profile: Sequence[float], generated_token_count: int) -> Dict[str, float]:
    values = [abs(safe_float(value)) for value in profile]
    total = float(sum(values))
    length = len(values)
    generated = max(0, min(int(generated_token_count), length))
    prompt_end = max(0, length - generated)
    prompt_values = values[:prompt_end]
    generated_values = values[prompt_end:]
    prompt_tail = prompt_values[-4:] if len(prompt_values) >= 4 else prompt_values
    generated_head = generated_values[:4] if len(generated_values) >= 4 else generated_values
    generated_tail = generated_values[-4:] if len(generated_values) >= 4 else generated_values
    return {
        "prompt_sum": float(sum(prompt_values)),
        "generated_sum": float(sum(generated_values)),
        "prompt_tail_sum": float(sum(prompt_tail)),
        "generated_head_sum": float(sum(generated_head)),
        "generated_tail_sum": float(sum(generated_tail)),
        "generated_token_count": generated,
        "tail_token_count": length,
        "generated_share": float(sum(generated_values) / total) if total else 0.0,
        "prompt_share": float(sum(prompt_values) / total) if total else 0.0,
    }


def enrich_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        enriched = attach_zone_metrics(row)
        hidden = split_windows(
            enriched.get("mean_hidden_token_profile", enriched.get("hidden_token_profile", [])),
            int(enriched.get("generated_token_count", 0)),
        )
        mlp = split_windows(
            enriched.get("mean_mlp_token_profile", enriched.get("mlp_token_profile", [])),
            int(enriched.get("generated_token_count", 0)),
        )
        out.append(
            {
                **enriched,
                "hidden_window_metrics": hidden,
                "mlp_window_metrics": mlp,
                "generated_prompt_gap_hidden": safe_float(hidden["generated_sum"]) - safe_float(hidden["prompt_sum"]),
                "generated_prompt_gap_mlp": safe_float(mlp["generated_sum"]) - safe_float(mlp["prompt_sum"]),
            }
        )
    return out


def summarize_axis(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "case_count": 0,
            "mean_hidden_generated_share": 0.0,
            "mean_mlp_generated_share": 0.0,
            "hidden_generated_dominant_ratio": 0.0,
            "mlp_generated_dominant_ratio": 0.0,
        }
    hidden_gen_shares = [safe_float(dict(row["hidden_window_metrics"]).get("generated_share")) for row in rows]
    mlp_gen_shares = [safe_float(dict(row["mlp_window_metrics"]).get("generated_share")) for row in rows]
    return {
        "case_count": len(rows),
        "mean_hidden_generated_share": average(hidden_gen_shares),
        "mean_mlp_generated_share": average(mlp_gen_shares),
        "hidden_generated_dominant_ratio": average(
            [
                1.0
                if safe_float(dict(row["hidden_window_metrics"]).get("generated_sum"))
                > safe_float(dict(row["hidden_window_metrics"]).get("prompt_sum"))
                else 0.0
                for row in rows
            ]
        ),
        "mlp_generated_dominant_ratio": average(
            [
                1.0
                if safe_float(dict(row["mlp_window_metrics"]).get("generated_sum"))
                > safe_float(dict(row["mlp_window_metrics"]).get("prompt_sum"))
                else 0.0
                for row in rows
            ]
        ),
    }


def summarize_component(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    positive = [row for row in rows if safe_float(row.get("weight")) > 0.0]
    if not positive:
        return {
            "case_count": 0,
            "signal_origin": "none",
            "corr_hidden_prompt_to_synergy": 0.0,
            "corr_hidden_generated_to_synergy": 0.0,
            "corr_mlp_prompt_to_synergy": 0.0,
            "corr_mlp_generated_to_synergy": 0.0,
        }
    synergy = [safe_float(row.get("union_synergy_joint")) for row in positive]
    hidden_prompt = [safe_float(dict(row["hidden_window_metrics"]).get("prompt_sum")) for row in positive]
    hidden_generated = [safe_float(dict(row["hidden_window_metrics"]).get("generated_sum")) for row in positive]
    mlp_prompt = [safe_float(dict(row["mlp_window_metrics"]).get("prompt_sum")) for row in positive]
    mlp_generated = [safe_float(dict(row["mlp_window_metrics"]).get("generated_sum")) for row in positive]
    corr_hidden_prompt = pearson_corr(hidden_prompt, synergy)
    corr_hidden_generated = pearson_corr(hidden_generated, synergy)
    corr_mlp_prompt = pearson_corr(mlp_prompt, synergy)
    corr_mlp_generated = pearson_corr(mlp_generated, synergy)
    generated_edge = average([corr_hidden_generated - corr_hidden_prompt, corr_mlp_generated - corr_mlp_prompt])
    if generated_edge > 0.05:
        signal_origin = "generated_dominant"
    elif generated_edge < -0.05:
        signal_origin = "prompt_contaminated"
    else:
        signal_origin = "mixed"
    return {
        "case_count": len(positive),
        "signal_origin": signal_origin,
        "mean_hidden_generated_share": average([safe_float(dict(row["hidden_window_metrics"]).get("generated_share")) for row in positive]),
        "mean_mlp_generated_share": average([safe_float(dict(row["mlp_window_metrics"]).get("generated_share")) for row in positive]),
        "corr_hidden_prompt_to_synergy": corr_hidden_prompt,
        "corr_hidden_generated_to_synergy": corr_hidden_generated,
        "corr_mlp_prompt_to_synergy": corr_mlp_prompt,
        "corr_mlp_generated_to_synergy": corr_mlp_generated,
        "corr_generated_prompt_gap_hidden_to_synergy": pearson_corr(
            [safe_float(row.get("generated_prompt_gap_hidden")) for row in positive],
            synergy,
        ),
        "corr_generated_prompt_gap_mlp_to_synergy": pearson_corr(
            [safe_float(row.get("generated_prompt_gap_mlp")) for row in positive],
            synergy,
        ),
    }


def build_summary(
    natural_rows: Sequence[Dict[str, object]],
    component_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    enriched = enrich_rows(natural_rows)
    joined = join_component_rows(component_rows, enriched)
    for row in joined:
        match = next(
            item
            for item in enriched
            if (
                str(item.get("model_id")) == str(row.get("model_id"))
                and str(item.get("category")) == str(row.get("category"))
                and str(item.get("prototype_term")) == str(row.get("prototype_term"))
                and str(item.get("instance_term")) == str(row.get("instance_term"))
                and str(item.get("axis")) == str(row.get("axis"))
            )
        )
        row["hidden_window_metrics"] = match["hidden_window_metrics"]
        row["mlp_window_metrics"] = match["mlp_window_metrics"]
        row["generated_prompt_gap_hidden"] = match["generated_prompt_gap_hidden"]
        row["generated_prompt_gap_mlp"] = match["generated_prompt_gap_mlp"]

    axes = sorted({str(row.get("axis", "")) for row in enriched})
    components = sorted({str(row.get("component_label", "")) for row in joined})
    return {
        "record_type": "stage56_natural_generation_strong_decoupling_summary",
        "case_count": len(enriched),
        "component_joined_row_count": len(joined),
        "per_axis": {
            axis: summarize_axis([row for row in enriched if str(row.get("axis")) == axis])
            for axis in axes
        },
        "per_component": {
            component: summarize_component([row for row in joined if str(row.get("component_label")) == component])
            for component in components
        },
    }


def build_markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# 自然生成强解耦摘要",
        "",
        f"- case_count: {summary['case_count']}",
        f"- component_joined_row_count: {summary['component_joined_row_count']}",
        "",
        "## Per Axis",
    ]
    for axis, row in dict(summary.get("per_axis", {})).items():
        lines.append(
            f"- {axis}: hidden_generated_share={safe_float(row.get('mean_hidden_generated_share')):.4f}, "
            f"mlp_generated_share={safe_float(row.get('mean_mlp_generated_share')):.4f}, "
            f"hidden_generated_dominant_ratio={safe_float(row.get('hidden_generated_dominant_ratio')):.4f}, "
            f"mlp_generated_dominant_ratio={safe_float(row.get('mlp_generated_dominant_ratio')):.4f}"
        )
    lines.extend(["", "## Per Component"])
    for component, row in dict(summary.get("per_component", {})).items():
        lines.append(
            f"- {component}: signal_origin={row.get('signal_origin', 'none')}, "
            f"hidden_prompt_corr={safe_float(row.get('corr_hidden_prompt_to_synergy')):+.4f}, "
            f"hidden_generated_corr={safe_float(row.get('corr_hidden_generated_to_synergy')):+.4f}, "
            f"mlp_prompt_corr={safe_float(row.get('corr_mlp_prompt_to_synergy')):+.4f}, "
            f"mlp_generated_corr={safe_float(row.get('corr_mlp_generated_to_synergy')):+.4f}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stronger natural-generation decoupling over prompt and generated windows")
    ap.add_argument("--natural-cases-jsonl", required=True)
    ap.add_argument("--component-joined-json", required=True)
    ap.add_argument("--output-dir", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    natural_rows = read_jsonl(Path(args.natural_cases_jsonl))
    component_rows = list(read_json(Path(args.component_joined_json)).get("rows", []))
    summary = build_summary(natural_rows, component_rows)
    out_dir = Path(args.output_dir) if args.output_dir else Path("tests/codex_temp/stage56_natural_generation_strong_decoupling")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "SUMMARY.md").write_text(build_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": out_dir.as_posix(),
                "case_count": int(summary["case_count"]),
                "component_joined_row_count": int(summary["component_joined_row_count"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
