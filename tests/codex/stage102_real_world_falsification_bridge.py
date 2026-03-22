from __future__ import annotations

import json
import statistics
import sys
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage102_real_world_falsification_bridge_20260322"
CODEX_DIR = ROOT / "tests" / "codex"
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from stage99_real_external_data_counterexample_pack import build_real_external_data_counterexample_pack_summary
from stage100_backfeed_suppression_hardening import build_backfeed_suppression_hardening_summary
from stage101_brain_evidence_joint_closure import build_brain_evidence_joint_closure_summary


PRIMARY_PROBE = ROOT / "tempdata" / "deepseek7b_multidim_encoding_probe_20260305_220444" / "multidim_encoding_probe.json"
MULTISEED_ROOT = ROOT / "tempdata" / "deepseek7b_multidim_multiseed_v1"
DIMENSIONS = ["style", "logic", "syntax"]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_any(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in patterns)


@lru_cache(maxsize=1)
def build_real_world_falsification_bridge_summary() -> dict:
    primary = _load_json(PRIMARY_PROBE)
    multiseed_paths = sorted(MULTISEED_ROOT.glob("seed*/probe/multidim_encoding_probe.json"))
    multiseed = [_load_json(path) for path in multiseed_paths]

    real_external = build_real_external_data_counterexample_pack_summary()["headline_metrics"]
    backfeed = build_backfeed_suppression_hardening_summary()["headline_metrics"]
    joint = build_brain_evidence_joint_closure_summary()["headline_metrics"]

    dimension_records = []
    task_context_strengths = []
    multiseed_stability_terms = []

    for dimension in DIMENSIONS:
        dim = primary["dimensions"][dimension]
        pairs = dim["pairs"]

        pair_realism_terms = []
        for pair in pairs:
            combined = f"{pair['a']} {pair['b']}"
            dialogue_marker = 1.0 if _has_any(combined, ["user:", "assistant:", "tell me", "explain", "why"]) else 0.0
            question_marker = 1.0 if "?" in combined else 0.0
            formality_shift = 1.0 if _has_any(combined, ["formal", "academic", "scientific", "technical report", "definition"]) else 0.0
            length_norm = _clip01(len(combined.split()) / 28.0)
            pair_realism = _clip01(
                0.26 * dialogue_marker
                + 0.20 * question_marker
                + 0.24 * formality_shift
                + 0.30 * length_norm
            )
            pair_realism_terms.append(pair_realism)

        task_context_realism = sum(pair_realism_terms) / len(pair_realism_terms)
        delta_energy = _clip01(dim["mean_pair_delta_l2"] / 1200.0)
        abs_energy = _clip01(dim["mean_pair_delta_abs"] / 1.20)
        contrast_separation = _clip01((1.10 - dim["pair_delta_cosine_mean"]) / 1.20)
        task_trigger_strength = _clip01(
            0.34 * delta_energy
            + 0.28 * abs_energy
            + 0.22 * contrast_separation
            + 0.16 * task_context_realism
        )

        seed_l2 = [item["dimensions"][dimension]["mean_pair_delta_l2"] for item in multiseed]
        seed_cos = [item["dimensions"][dimension]["pair_delta_cosine_mean"] for item in multiseed]
        l2_stability = _clip01(1.0 - (statistics.pstdev(seed_l2) / max(1e-6, statistics.mean(seed_l2))) * 2.0)
        cos_stability = _clip01(1.0 - statistics.pstdev(seed_cos) / 0.25)
        multiseed_stability = _clip01(0.56 * l2_stability + 0.44 * cos_stability)

        dimension_records.append(
            {
                "dimension": dimension,
                "task_context_realism": task_context_realism,
                "task_trigger_strength": task_trigger_strength,
                "multiseed_stability": multiseed_stability,
                "mean_pair_delta_l2": dim["mean_pair_delta_l2"],
                "mean_pair_delta_abs": dim["mean_pair_delta_abs"],
                "pair_delta_cosine_mean": dim["pair_delta_cosine_mean"],
                "pair_preview": [{"a": pair["a"], "b": pair["b"]} for pair in pairs[:2]],
            }
        )
        task_context_strengths.append(task_trigger_strength * task_context_realism)
        multiseed_stability_terms.append(multiseed_stability)

    task_context_bridge_strength = sum(task_context_strengths) / len(task_context_strengths)
    multiseed_probe_stability = sum(multiseed_stability_terms) / len(multiseed_stability_terms)
    bridge_alignment_support = _clip01(
        0.28 * real_external["path_alignment_rate"]
        + 0.22 * real_external["receiver_alignment_rate"]
        + 0.18 * real_external["clause_alignment_rate"]
        + 0.18 * joint["real_world_bridge_joint"]
        + 0.14 * joint["brain_evidence_joint_closure_score"]
    )
    falsification_triggerability = _clip01(
        0.30 * task_context_bridge_strength
        + 0.24 * real_external["real_trigger_rate"]
        + 0.20 * real_external["mean_strongest_path_intensity"]
        + 0.14 * bridge_alignment_support
        + 0.12 * (1.0 - backfeed["summary_backfeed_risk_after"])
    )
    remaining_real_world_gap = _clip01(
        1.0
        - min(
            task_context_bridge_strength,
            multiseed_probe_stability,
            bridge_alignment_support,
            falsification_triggerability,
        )
    )
    real_world_falsification_bridge_score = _clip01(
        0.24 * task_context_bridge_strength
        + 0.22 * multiseed_probe_stability
        + 0.22 * bridge_alignment_support
        + 0.20 * falsification_triggerability
        + 0.12 * (1.0 - remaining_real_world_gap)
    )

    return {
        "headline_metrics": {
            "task_context_bridge_strength": task_context_bridge_strength,
            "multiseed_probe_stability": multiseed_probe_stability,
            "bridge_alignment_support": bridge_alignment_support,
            "falsification_triggerability": falsification_triggerability,
            "remaining_real_world_gap": remaining_real_world_gap,
            "real_world_falsification_bridge_score": real_world_falsification_bridge_score,
        },
        "dimension_records": dimension_records,
        "status": {
            "status_short": (
                "real_world_falsification_bridge_ready"
                if real_world_falsification_bridge_score >= 0.60
                and multiseed_probe_stability >= 0.65
                else "real_world_falsification_bridge_transition"
            ),
            "status_label": "真实世界判伪桥已经从词表级外部样本推进到任务语境级探针，但距离真实任务闭合仍有明显缺口。",
        },
        "project_readout": {
            "summary": "这一轮用自然语言对比句和多随机种子探针，搭起了从外部样本级反例到更接近真实任务语境的判伪桥。",
            "next_question": "下一步要把这座桥真正接到真实任务数据和真实错误案例，而不是长期停留在探针句对层面。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage102 Real World Falsification Bridge",
        "",
        f"- task_context_bridge_strength: {hm['task_context_bridge_strength']:.6f}",
        f"- multiseed_probe_stability: {hm['multiseed_probe_stability']:.6f}",
        f"- bridge_alignment_support: {hm['bridge_alignment_support']:.6f}",
        f"- falsification_triggerability: {hm['falsification_triggerability']:.6f}",
        f"- remaining_real_world_gap: {hm['remaining_real_world_gap']:.6f}",
        f"- real_world_falsification_bridge_score: {hm['real_world_falsification_bridge_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_real_world_falsification_bridge_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
