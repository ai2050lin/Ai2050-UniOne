from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage57_language_task_boundary_trigger_20260321"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage57_real_boundary_stress_generator import build_real_boundary_stress_generator_summary


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_language_task_boundary_trigger_summary() -> dict:
    suite = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_long_context_online_language_suite_20260320" / "summary.json"
    )
    stress = build_real_boundary_stress_generator_summary()

    comparison = suite["comparison"]
    long_ctx = suite["long_context"]
    context_overload = stress["scenario_results"]["context_overload"]
    scale = stress["scale_metrics"]

    overload_strength = max(0.0, 0.70 - context_overload["context_route_alignment"])
    pressure_strength = max(0.0, 0.60 - context_overload["pressure_under_reuse"])

    stressed_long_forgetting = min(
        1.0,
        comparison["long_forgetting"] + 0.85 * overload_strength + 0.40 * pressure_strength + 0.25 * scale["scale_collapse_risk"],
    )
    stressed_base_perplexity_delta = comparison["long_base_perplexity_delta"] * (
        1.0 + 0.60 * overload_strength + 0.25 * scale["scale_forgetting_penalty"]
    )
    stressed_novel_accuracy_after = max(
        0.0,
        comparison["long_novel_accuracy_after"] - 0.42 * overload_strength - 0.10 * scale["scale_collapse_risk"],
    )
    stressed_gate_shift = abs(long_ctx["deltas"]["strict_gate_shift"]) + 0.006 * overload_strength + 0.003 * pressure_strength

    task_failure = (
        stressed_long_forgetting > 0.20
        or stressed_base_perplexity_delta > 1000.0
        or stressed_novel_accuracy_after < 0.90
    )

    task_boundary_readiness = max(
        0.0,
        min(
            1.0,
            0.30 * (1.0 - min(1.0, stressed_long_forgetting))
            + 0.25 * (1.0 - min(1.0, stressed_base_perplexity_delta / 1200.0))
            + 0.25 * stressed_novel_accuracy_after
            + 0.20 * (1.0 - min(1.0, stressed_gate_shift / 0.02)),
        ),
    )

    return {
        "headline_metrics": {
            "stressed_long_forgetting": stressed_long_forgetting,
            "stressed_base_perplexity_delta": stressed_base_perplexity_delta,
            "stressed_novel_accuracy_after": stressed_novel_accuracy_after,
            "stressed_gate_shift": stressed_gate_shift,
            "task_boundary_readiness": task_boundary_readiness,
        },
        "task_trigger": {
            "triggered": task_failure,
            "reason": "long-context language task fails under context overload if forgetting spikes, base perplexity blows up, or novel accuracy collapses.",
        },
        "project_readout": {
            "summary": "Language task boundary trigger injects the current context-overload stress into the long-context online language suite and checks whether the task itself, not just the metric layer, crosses the declared failure boundary.",
            "next_question": "Use this trigger to compare different bounded learning laws under task pressure instead of relying only on static kernel scores.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage57 Language Task Boundary Trigger",
        "",
        f"- stressed_long_forgetting: {hm['stressed_long_forgetting']:.6f}",
        f"- stressed_base_perplexity_delta: {hm['stressed_base_perplexity_delta']:.6f}",
        f"- stressed_novel_accuracy_after: {hm['stressed_novel_accuracy_after']:.6f}",
        f"- stressed_gate_shift: {hm['stressed_gate_shift']:.6f}",
        f"- task_boundary_readiness: {hm['task_boundary_readiness']:.6f}",
        f"- triggered: {summary['task_trigger']['triggered']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_language_task_boundary_trigger_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
