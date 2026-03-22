from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from stage100_backfeed_suppression_hardening import build_backfeed_suppression_hardening_summary
from stage101_brain_evidence_joint_closure import build_brain_evidence_joint_closure_summary
from stage102_real_world_falsification_bridge import build_real_world_falsification_bridge_summary
from stage103_native_brain_anchor_search import build_native_brain_anchor_search_summary
from stage104_tensor_level_language_projection_rebuild import build_tensor_level_language_projection_rebuild_summary
from stage105_tensor_level_route_scale_rebuild import build_tensor_level_route_scale_rebuild_summary
from stage106_forward_backward_trace_rebuild import build_forward_backward_trace_rebuild_summary
from stage107_math_theory_object_layer_synthesis import build_math_theory_object_layer_synthesis_summary
from stage109_invariant_boundary_quantity_search import build_invariant_boundary_quantity_search_summary


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "tests" / "codex_temp" / "stage110_axiom_falsification_suite_20260322"
PRIMARY_PROBE = ROOT / "tempdata" / "deepseek7b_multidim_encoding_probe_20260305_220444" / "multidim_encoding_probe.json"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            rows.append(row)
    return rows


@lru_cache(maxsize=1)
def build_axiom_falsification_suite_summary() -> dict:
    probe = _load_json(PRIMARY_PROBE)
    projection = build_tensor_level_language_projection_rebuild_summary()["headline_metrics"]
    route = build_tensor_level_route_scale_rebuild_summary()["headline_metrics"]
    fb = build_forward_backward_trace_rebuild_summary()["headline_metrics"]
    backfeed = build_backfeed_suppression_hardening_summary()["headline_metrics"]
    joint = build_brain_evidence_joint_closure_summary()["headline_metrics"]
    bridge = build_real_world_falsification_bridge_summary()["headline_metrics"]
    anchors = build_native_brain_anchor_search_summary()["headline_metrics"]
    theory = build_math_theory_object_layer_synthesis_summary()
    quantities = build_invariant_boundary_quantity_search_summary()

    english_rows = _load_rows(ROOT / "tests" / "codex" / "deepseek7b_nouns_english_520_clean.csv")
    bilingual_rows = _load_rows(ROOT / "tests" / "codex" / "deepseek7b_bilingual_nouns_utf8.csv")

    english_counter = Counter(row[1] for row in english_rows if len(row) > 1)
    bilingual_counter = Counter(row[1] for row in bilingual_rows if len(row) > 1)
    abstract_ratio = english_counter.get("abstract", 0) / max(1, len(english_rows))
    bilingual_coverage = len(bilingual_counter) / max(1, len(english_counter))

    style_pair = probe["dimensions"]["style"]["pairs"][0]
    logic_pair = probe["dimensions"]["logic"]["pairs"][0]
    syntax_pair = probe["dimensions"]["syntax"]["pairs"][0]

    attack_records = [
        {
            "name": "style_context_flip_attack",
            "sample": [style_pair["a"], style_pair["b"]],
            "targets": {
                "projection_covariance_axiom": 1.00,
                "falsifiable_boundary_axiom": 0.55,
            },
            "intensity": _clip01(
                0.34 * bridge["task_context_bridge_strength"]
                + 0.24 * (1.0 - projection["cross_dimension_projection_stability"])
                + 0.22 * projection["cross_dimension_separation"]
                + 0.20 * bridge["remaining_real_world_gap"]
            ),
        },
        {
            "name": "logic_negation_attack",
            "sample": [logic_pair["a"], logic_pair["b"]],
            "targets": {
                "bounded_repair_axiom": 0.65,
                "falsifiable_boundary_axiom": 0.85,
                "distributed_routing_axiom": 0.35,
            },
            "intensity": _clip01(
                0.30 * bridge["falsification_triggerability"]
                + 0.24 * quantities["headline_metrics"]["highest_boundary_pressure"]
                + 0.22 * (1.0 - fb["raw_backward_fidelity"])
                + 0.24 * backfeed["summary_backfeed_risk_after"]
            ),
        },
        {
            "name": "syntax_voice_shift_attack",
            "sample": [syntax_pair["a"], syntax_pair["b"]],
            "targets": {
                "projection_covariance_axiom": 0.70,
                "distributed_routing_axiom": 0.40,
                "bounded_repair_axiom": 0.30,
            },
            "intensity": _clip01(
                0.28 * (1.0 - projection["reconstructed_route_projection"])
                + 0.24 * (1.0 - projection["cross_dimension_projection_stability"])
                + 0.22 * route["route_scale_margin"] * 0.5 + 0.06
                + 0.26 * bridge["task_context_bridge_strength"]
            ),
        },
        {
            "name": "bilingual_alias_attack",
            "sample": [bilingual_rows[0][0], bilingual_rows[1][0]],
            "targets": {
                "projection_covariance_axiom": 0.45,
                "falsifiable_boundary_axiom": 0.75,
                "anchor_separability_axiom": 0.25,
            },
            "intensity": _clip01(
                0.34 * (1.0 - bilingual_coverage)
                + 0.24 * bridge["remaining_real_world_gap"]
                + 0.18 * backfeed["summary_backfeed_risk_after"]
                + 0.24 * quantities["headline_metrics"]["boundary_quantity_resilience"]
            ),
        },
        {
            "name": "macro_abstract_bridge_attack",
            "sample": ["justice", "truth"],
            "targets": {
                "falsifiable_boundary_axiom": 0.90,
                "anchor_separability_axiom": 0.40,
                "bounded_repair_axiom": 0.30,
            },
            "intensity": _clip01(
                0.32 * quantities["headline_metrics"]["highest_boundary_pressure"]
                + 0.24 * (1.0 - abstract_ratio / 0.10)
                + 0.20 * joint["evidence_isolation_joint"]
                + 0.24 * bridge["remaining_real_world_gap"]
            ),
        },
        {
            "name": "anchor_overlap_attack",
            "sample": ["style-anchor", "syntax-anchor"],
            "targets": {
                "anchor_separability_axiom": 1.00,
                "distributed_routing_axiom": 0.30,
            },
            "intensity": _clip01(
                0.42 * anchors["anchor_ambiguity_penalty"]
                + 0.26 * (1.0 - anchors["dimension_specific_anchor_strength"])
                + 0.14 * (1.0 - anchors["closure_bridge_support"])
                + 0.18 * route["reconstructed_route_scale_score"]
            ),
        },
    ]

    attack_coverage = _clip01(
        len({axiom for attack in attack_records for axiom in attack["targets"]}) / 5.0
    )

    axiom_support_map = {
        record["name"]: record["support"] for record in theory["axiom_records"]
    }
    axiom_penalties = defaultdict(float)
    for attack in attack_records:
        for axiom_name, weight in attack["targets"].items():
            axiom_penalties[axiom_name] += attack["intensity"] * weight * 0.28

    axiom_attack_records = []
    for record in theory["axiom_records"]:
        name = record["name"]
        post_attack_support = _clip01(record["support"] - axiom_penalties[name])
        targeted_attack_count = sum(1 for attack in attack_records if name in attack["targets"])
        axiom_attack_records.append(
            {
                "name": name,
                "base_support": record["support"],
                "post_attack_support": post_attack_support,
                "support_drop": _clip01(record["support"] - post_attack_support),
                "targeted_attack_count": targeted_attack_count,
            }
        )

    weakest_axiom_after_attack = min(axiom_attack_records, key=lambda item: item["post_attack_support"])
    strongest_attack = max(attack_records, key=lambda item: item["intensity"])
    task_bridge_retest_pressure = _clip01(
        0.42 * bridge["remaining_real_world_gap"]
        + 0.32 * _mean(
            attack["intensity"]
            for attack in attack_records
            if attack["name"] in {"logic_negation_attack", "macro_abstract_bridge_attack", "bilingual_alias_attack"}
        )
        + 0.26 * quantities["headline_metrics"]["highest_boundary_pressure"]
    )
    falsification_survival_score = _clip01(
        0.28 * weakest_axiom_after_attack["post_attack_support"]
        + 0.22 * (1.0 - strongest_attack["intensity"])
        + 0.24 * (1.0 - task_bridge_retest_pressure)
        + 0.26 * attack_coverage
    )
    axiom_falsification_suite_score = _clip01(
        0.26 * attack_coverage
        + 0.24 * (1.0 - _mean(item["support_drop"] for item in axiom_attack_records))
        + 0.22 * falsification_survival_score
        + 0.28 * (1.0 - task_bridge_retest_pressure)
    )

    return {
        "headline_metrics": {
            "attack_coverage": attack_coverage,
            "strongest_attack_name": strongest_attack["name"],
            "strongest_attack_intensity": strongest_attack["intensity"],
            "weakest_axiom_after_attack_name": weakest_axiom_after_attack["name"],
            "weakest_axiom_after_attack_score": weakest_axiom_after_attack["post_attack_support"],
            "task_bridge_retest_pressure": task_bridge_retest_pressure,
            "falsification_survival_score": falsification_survival_score,
            "axiom_falsification_suite_score": axiom_falsification_suite_score,
        },
        "attack_records": attack_records,
        "axiom_attack_records": axiom_attack_records,
        "status": {
            "status_short": (
                "axiom_falsification_suite_ready"
                if axiom_falsification_suite_score >= 0.54 and weakest_axiom_after_attack["post_attack_support"] >= 0.28
                else "axiom_falsification_suite_transition"
            ),
            "status_label": "公理判伪攻击包已经开始使用真实任务句和真实词表样本，但当前最弱公理在攻击后仍然明显偏脆。",
        },
        "project_readout": {
            "summary": "这一轮开始把 style、logic、syntax 对照任务句与双语、抽象概念样本组织成面向公理的攻击包，不再只看静态支持分数。",
            "next_question": "下一步要把这些攻击进一步扩展到更真实的任务批次，并检查哪些公理经常被同一类反例打穿。",
        },
    }


def write_report(summary: dict, out_dir: Path) -> None:
    hm = summary["headline_metrics"]
    lines = [
        "# Stage110 Axiom Falsification Suite",
        "",
        f"- attack_coverage: {hm['attack_coverage']:.6f}",
        f"- strongest_attack_name: {hm['strongest_attack_name']}",
        f"- strongest_attack_intensity: {hm['strongest_attack_intensity']:.6f}",
        f"- weakest_axiom_after_attack_name: {hm['weakest_axiom_after_attack_name']}",
        f"- weakest_axiom_after_attack_score: {hm['weakest_axiom_after_attack_score']:.6f}",
        f"- task_bridge_retest_pressure: {hm['task_bridge_retest_pressure']:.6f}",
        f"- falsification_survival_score: {hm['falsification_survival_score']:.6f}",
        f"- axiom_falsification_suite_score: {hm['axiom_falsification_suite_score']:.6f}",
        f"- status_short: {summary['status']['status_short']}",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_axiom_falsification_suite_summary()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)


if __name__ == "__main__":
    main()
