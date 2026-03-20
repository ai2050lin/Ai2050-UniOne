from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_direct_refinement_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_neuron_origin_direct_refinement_summary() -> dict:
    origin = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_native_probe_20260320" / "summary.json"
    )
    neuron = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_native_direct_closure_20260320" / "summary.json"
    )

    ho = origin["headline_metrics"]
    hn = neuron["headline_metrics"]

    origin_source_refined = ho["pulse_source_strength"]
    origin_focus_refined = ho["selectivity_focus"] * (1.0 + hn["neuron_feature_ratio"])
    origin_retention_refined = ho["lock_retention"] * (1.0 + hn["neuron_closure_confidence"])
    neuron_origin_margin_v2 = origin_source_refined + origin_focus_refined + origin_retention_refined
    origin_stability_v2 = neuron_origin_margin_v2 / (1.0 + hn["neuron_native_core"])

    return {
        "headline_metrics": {
            "origin_source_refined": origin_source_refined,
            "origin_focus_refined": origin_focus_refined,
            "origin_retention_refined": origin_retention_refined,
            "neuron_origin_margin_v2": neuron_origin_margin_v2,
            "origin_stability_v2": origin_stability_v2,
        },
        "refinement_equation": {
            "source_term": "N_source_v2 = N_source",
            "focus_term": "N_focus_v2 = N_focus * (1 + R_neuron)",
            "retention_term": "N_retention_v2 = N_retention * (1 + C_neuron)",
            "margin_term": "M_origin_v2 = N_source_v2 + N_focus_v2 + N_retention_v2",
            "stability_term": "S_origin_v2 = M_origin_v2 / (1 + N_core)",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 神经元起点直测强化报告",
        "",
        f"- origin_source_refined: {hm['origin_source_refined']:.6f}",
        f"- origin_focus_refined: {hm['origin_focus_refined']:.6f}",
        f"- origin_retention_refined: {hm['origin_retention_refined']:.6f}",
        f"- neuron_origin_margin_v2: {hm['neuron_origin_margin_v2']:.6f}",
        f"- origin_stability_v2: {hm['origin_stability_v2']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_neuron_origin_direct_refinement_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
