from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage56_learning_rule_bounded_law_comparison_20260321"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _headline(version: int) -> dict:
    path = ROOT / "tests" / "codex_temp" / f"stage56_encoding_mechanism_closed_form_v{version}_20260321" / "summary.json"
    return _load_json(path)["headline_metrics"]


def _bounded_terms(mode: str, previous_learning: float, drive: float, reference_scale: float) -> tuple[float, float]:
    if mode == "log":
        latent = math.log1p(previous_learning)
        growth = 0.27 + 0.22 * drive + 0.008 * latent
    elif mode == "sqrt":
        latent = math.sqrt(previous_learning)
        growth = 0.26 + 0.18 * drive + 0.0035 * math.log1p(latent)
    elif mode == "rational":
        latent = previous_learning / (previous_learning + 6000.0)
        growth = 0.29 + 0.20 * drive + 0.11 * latent
    else:
        raise ValueError(f"unsupported mode: {mode}")

    bounded_term = reference_scale * growth
    return latent, bounded_term


def _law_metrics(mode: str, previous_learning: float, current_learning: float, drive_prev: float, drive_now: float,
                 reference_prev: float, reference_now: float, feature_now: float, structure_now: float,
                 pressure_now: float, brain_score: float, train_gap: float) -> dict:
    latent_prev, bounded_prev = _bounded_terms(mode, previous_learning, drive_prev, reference_prev)
    latent_now, bounded_now = _bounded_terms(mode, current_learning, drive_now, reference_now)
    ratio = bounded_now / bounded_prev
    learning_share = bounded_now / (bounded_now + feature_now + structure_now + pressure_now)
    domination_penalty = max(0.0, min(1.0, learning_share))
    stability = max(0.0, min(1.0, 1.0 - abs(ratio - 1.0) * 4.5))
    readiness = max(
        0.0,
        min(
            1.0,
            0.32 * (1.0 - domination_penalty)
            + 0.24 * stability
            + 0.22 * (1.0 - train_gap)
            + 0.22 * brain_score,
        ),
    )
    return {
        "mode": mode,
        "latent_prev": latent_prev,
        "latent_now": latent_now,
        "bounded_learning_prev": bounded_prev,
        "bounded_learning_now": bounded_now,
        "bounded_ratio": ratio,
        "domination_penalty": domination_penalty,
        "stability": stability,
        "readiness": readiness,
    }


def build_learning_rule_bounded_law_comparison_summary() -> dict:
    v99 = _headline(99)
    v100 = _headline(100)
    v101 = _headline(101)
    bridge_v44 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v44_20260321" / "summary.json"
    )["headline_metrics"]
    bridge_v45 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_training_terminal_bridge_v45_20260321" / "summary.json"
    )["headline_metrics"]
    brain_v38 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v38_20260321" / "summary.json"
    )["headline_metrics"]
    brain_v39 = _load_json(
        ROOT / "tests" / "codex_temp" / "stage56_brain_encoding_direct_refinement_v39_20260321" / "summary.json"
    )["headline_metrics"]

    drive_v100 = max(
        0.0,
        min(
            1.0,
            bridge_v44["topology_training_readiness_v44"] * 0.52
            + brain_v38["direct_brain_measure_v38"] * 0.18
            + bridge_v44["plasticity_rule_alignment_v44"] * 0.12
            - bridge_v44["topology_training_gap_v44"] * 0.16
            - v99["pressure_term_v99"] * 1e-4 * 0.02,
        ),
    )
    drive_v101 = max(
        0.0,
        min(
            1.0,
            bridge_v45["topology_training_readiness_v45"] * 0.52
            + brain_v39["direct_brain_measure_v39"] * 0.18
            + bridge_v45["plasticity_rule_alignment_v45"] * 0.12
            - bridge_v45["topology_training_gap_v45"] * 0.16
            - v100["pressure_term_v100"] * 1e-4 * 0.02,
        ),
    )

    reference_v100 = (v99["feature_term_v99"] + v99["structure_term_v99"]) / 2.0
    reference_v101 = (v100["feature_term_v100"] + v100["structure_term_v100"]) / 2.0

    candidates = {}
    for mode in ("log", "sqrt", "rational"):
        candidates[mode] = _law_metrics(
            mode=mode,
            previous_learning=v99["learning_term_v99"],
            current_learning=v100["learning_term_v100"],
            drive_prev=drive_v100,
            drive_now=drive_v101,
            reference_prev=reference_v100,
            reference_now=reference_v101,
            feature_now=v100["feature_term_v100"],
            structure_now=v100["structure_term_v100"],
            pressure_now=v100["pressure_term_v100"],
            brain_score=brain_v39["direct_brain_measure_v39"],
            train_gap=bridge_v45["topology_training_gap_v45"],
        )

    best = max(candidates.items(), key=lambda kv: kv[1]["readiness"])
    readiness_gap = best[1]["readiness"] - min(item["readiness"] for item in candidates.values())

    return {
        "headline_metrics": {
            "best_law_name": best[0],
            "best_law_readiness": best[1]["readiness"],
            "best_law_bounded_ratio": best[1]["bounded_ratio"],
            "best_law_domination_penalty": best[1]["domination_penalty"],
            "law_readiness_gap": readiness_gap,
            "comparison_readiness": 0.45 * best[1]["readiness"] + 0.30 * (1.0 - best[1]["domination_penalty"]) + 0.25 * best[1]["stability"],
        },
        "candidate_laws": candidates,
        "project_readout": {
            "summary": "Compare log, sqrt, and rational bounded learning updates to choose the most stable K_l replacement candidate before rewriting the closed form.",
            "next_question": "Use the best bounded law as the learning update inside local generative rules and test whether patch, fiber, and route structures still emerge.",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Learning Rule Bounded Law Comparison",
        "",
        f"- best_law_name: {hm['best_law_name']}",
        f"- best_law_readiness: {hm['best_law_readiness']:.6f}",
        f"- best_law_bounded_ratio: {hm['best_law_bounded_ratio']:.6f}",
        f"- best_law_domination_penalty: {hm['best_law_domination_penalty']:.6f}",
        f"- law_readiness_gap: {hm['law_readiness_gap']:.6f}",
        f"- comparison_readiness: {hm['comparison_readiness']:.6f}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_learning_rule_bounded_law_comparison_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
