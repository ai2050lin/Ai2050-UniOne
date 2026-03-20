from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_structure_stability_native_approximation_20260320"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def build_structure_stability_native_approximation_summary() -> dict:
    stability = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_stability_reparameterization_20260320" / "summary.json"
    )
    structure = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_structure_genesis_confidence_refinement_20260320" / "summary.json"
    )
    origin = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_neuron_origin_direct_refinement_20260320" / "summary.json"
    )

    hs = stability["headline_metrics"]
    hg = structure["headline_metrics"]
    ho = origin["headline_metrics"]

    native_stability_seed = ho["origin_stability_v2"] * hs["stability_intensity"]
    native_stability_binding = hg["binding_refined_v2"] * hs["stability_strength"]
    native_stability_feedback = hg["feedback_refined_v2"] * hs["closure_alignment"]
    native_stability_core = native_stability_seed + native_stability_binding + native_stability_feedback
    native_stability_ratio = native_stability_core / (1.0 + hg["structure_genesis_margin_v3"])

    return {
        "headline_metrics": {
            "native_stability_seed": native_stability_seed,
            "native_stability_binding": native_stability_binding,
            "native_stability_feedback": native_stability_feedback,
            "native_stability_core": native_stability_core,
            "native_stability_ratio": native_stability_ratio,
        },
        "approximation_equation": {
            "seed_term": "S_seed_native = S_origin_v2 * I_struct",
            "binding_term": "S_bind_native = M_bind_v2 * S_strength",
            "feedback_term": "S_fb_native = S_fb_v2 * A_closure",
            "core_term": "S_native = S_seed_native + S_bind_native + S_fb_native",
            "ratio_term": "R_native = S_native / (1 + M_struct_v3)",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# 结构稳定原生逼近报告",
        "",
        f"- native_stability_seed: {hm['native_stability_seed']:.6f}",
        f"- native_stability_binding: {hm['native_stability_binding']:.6f}",
        f"- native_stability_feedback: {hm['native_stability_feedback']:.6f}",
        f"- native_stability_core: {hm['native_stability_core']:.6f}",
        f"- native_stability_ratio: {hm['native_stability_ratio']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_structure_stability_native_approximation_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
