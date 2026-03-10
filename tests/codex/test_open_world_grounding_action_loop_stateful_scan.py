#!/usr/bin/env python
"""
Stateful scan for the open-world grounding action loop.

Goal:
- extend the minimal action loop with long-state action trust
- test whether the grounding gains can transfer into a minimal
  perception-action-correction loop once action policy is folded into the
  same update law family
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import test_open_world_continuous_grounding_stream as stream_bench
import test_open_world_grounding_action_loop as action_loop


ROOT = Path(__file__).resolve().parents[2]


class StatefulSharedActionAgent(stream_bench.SharedOffsetStreamGrounder):
    def __init__(
        self,
        family_alpha: float,
        offset_alpha: float,
        action_beta: float,
        correction_mix: float,
        trust_temp: float,
    ) -> None:
        super().__init__()
        self.family_alpha_fixed = float(family_alpha)
        self.offset_alpha_fixed = float(offset_alpha)
        self.action_beta = float(action_beta)
        self.correction_mix = float(correction_mix)
        self.trust_temp = float(trust_temp)
        self.family_trust: Dict[str, float] = {family: 0.0 for family in stream_bench.FAMILIES}
        self.family_seen: Dict[str, int] = {family: 0 for family in stream_bench.FAMILIES}

    def train(self, x: np.ndarray, family: str, concept: str) -> None:
        family_prev = self.family_basis.get(family)
        family_mean = self._mean(
            family_prev,
            self.family_count.get(family, 0),
            x,
            alpha_override=self.family_alpha_fixed if family_prev is not None else None,
        )
        self.family_basis[family] = family_mean
        self.family_count[family] = self.family_count.get(family, 0) + 1

        offset = (x - family_mean).astype(np.float32)
        offset_prev = self.concept_offset.get(concept)
        self.concept_offset[concept] = self._mean(
            offset_prev,
            self.concept_count.get(concept, 0),
            offset,
            alpha_override=self.offset_alpha_fixed if offset_prev is not None else None,
        )
        self.concept_count[concept] = self.concept_count.get(concept, 0) + 1
        self.concept_family[concept] = family

        pred_family, _pred_concept, _margin = self.predict_with_margin(x)
        correct = float(pred_family == family)
        self.family_seen[family] += 1
        trust_prev = self.family_trust[family]
        self.family_trust[family] = (1.0 - self.action_beta) * trust_prev + self.action_beta * correct
        if pred_family != family:
            wrong_prev = self.family_trust[pred_family]
            self.family_trust[pred_family] = (1.0 - self.action_beta) * wrong_prev

    def family_candidates(self, x: np.ndarray) -> List[Tuple[str, float]]:
        rows = []
        for name, proto in self.family_basis.items():
            base_dist = stream_bench.sq_dist(x, proto)
            trust = self.family_trust.get(name, 0.0)
            adjusted = float(base_dist - self.trust_temp * trust)
            rows.append((name, adjusted))
        rows.sort(key=lambda item: item[1])
        return rows

    def concept_candidates(self, x: np.ndarray, family: str) -> List[Tuple[str, float]]:
        rows = []
        for concept, fam in self.concept_family.items():
            if fam != family:
                continue
            proto = self.family_basis[family] + self.concept_offset[concept]
            rows.append((concept, stream_bench.sq_dist(x, proto)))
        rows.sort(key=lambda item: item[1])
        return rows

    def predict_with_margin(self, x: np.ndarray) -> Tuple[str, str, float]:
        fam_rows = self.family_candidates(x)
        family = fam_rows[0][0]
        concept_rows = self.concept_candidates(x, family)
        concept = concept_rows[0][0]
        if len(fam_rows) < 2:
            margin = 0.0
        else:
            margin = float(fam_rows[1][1] - fam_rows[0][1])
        return family, concept, margin

    def corrected_family(self, x: np.ndarray, margin_threshold: float) -> Tuple[str, bool]:
        pred_family, _pred_concept, margin = self.predict_with_margin(x)
        if margin >= margin_threshold:
            return pred_family, False

        family_rows = self.family_candidates(x)
        if len(family_rows) >= 2:
            first_family = family_rows[0][0]
            second_family = family_rows[1][0]
            first_trust = self.family_trust.get(first_family, 0.0)
            second_trust = self.family_trust.get(second_family, 0.0)
            if second_trust > first_trust + self.correction_mix * max(1e-6, abs(margin)):
                return second_family, True
        return pred_family, True


def run_stateful_system(
    seed: int,
    noise: float,
    dropout_p: float,
    missing_modality_p: float,
    drift_scale: float,
    margin_threshold: float,
    family_alpha: float,
    offset_alpha: float,
    action_beta: float,
    correction_mix: float,
    trust_temp: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    agent = StatefulSharedActionAgent(
        family_alpha=family_alpha,
        offset_alpha=offset_alpha,
        action_beta=action_beta,
        correction_mix=correction_mix,
        trust_temp=trust_temp,
    )
    dim = 24
    stream = stream_bench.build_stream(seed)

    action_ok = 0
    corrected_action_ok = 0
    action_total = 0
    update_count = 0

    for item in stream:
        if item["kind"] == "concept":
            x = stream_bench.sample_stream_input(
                rng,
                item["concept"],
                noise=noise,
                dropout_p=dropout_p,
                missing_modality_p=missing_modality_p,
                drift_scale=drift_scale,
            )
            agent.train(x, item["family"], item["concept"])
            update_count += 1

            pred_family, _pred_concept, _margin = agent.predict_with_margin(x)
            first_ok = action_loop.action_for_family(pred_family) == action_loop.action_for_family(item["family"])
            corrected_family, _used_correction = agent.corrected_family(x, margin_threshold=margin_threshold)
            corrected_ok = action_loop.action_for_family(corrected_family) == action_loop.action_for_family(item["family"])
            action_ok += int(first_ok)
            corrected_action_ok += int(corrected_ok)
            action_total += 1
        else:
            _ = stream_bench.sample_noise_chunk(rng, dim=dim, noise_scale=noise * 1.2)

    old_retention = action_loop.evaluate_old_concepts(
        agent=agent,
        rng=rng,
        repeats=18,
        noise=noise,
        dropout_p=dropout_p,
        missing_modality_p=missing_modality_p,
        drift_scale=drift_scale,
    )
    action_accuracy = float(action_ok / max(1, action_total))
    corrected_action_accuracy = float(corrected_action_ok / max(1, action_total))
    loop_score = float((1.3 * corrected_action_accuracy + 1.0 * action_accuracy + 1.1 * old_retention) / 3.4)
    mean_trust = float(np.mean(list(agent.family_trust.values())))

    return {
        "action_accuracy": action_accuracy,
        "corrected_action_accuracy": corrected_action_accuracy,
        "old_concept_retention": old_retention,
        "loop_score": loop_score,
        "mean_family_trust": mean_trust,
        "update_count": float(update_count),
    }


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(rows[0].keys())
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan long-state action trust for the open-world grounding action loop")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.18)
    ap.add_argument("--dropout-p", type=float, default=0.10)
    ap.add_argument("--missing-modality-p", type=float, default=0.25)
    ap.add_argument("--drift-scale", type=float, default=0.06)
    ap.add_argument("--margin-threshold", type=float, default=0.035)
    ap.add_argument("--json-out", type=str, default="tests/codex_temp/open_world_grounding_action_loop_stateful_scan_20260310.json")
    args = ap.parse_args()

    t0 = time.time()
    baseline_payload = json.loads((ROOT / "tests" / "codex_temp" / "open_world_grounding_action_loop_20260310.json").read_text(encoding="utf-8"))
    direct = baseline_payload["systems"]["direct_action"]
    tuned = baseline_payload["systems"]["shared_action_tuned"]

    rows = []
    best = None
    for action_beta in [0.05, 0.10, 0.18, 0.28]:
        for correction_mix in [0.05, 0.10, 0.18, 0.28]:
            for trust_temp in [0.02, 0.05, 0.08, 0.12]:
                seeds = []
                for offset in range(int(args.num_seeds)):
                    seeds.append(
                        run_stateful_system(
                            seed=int(args.seed) + offset,
                            noise=float(args.noise),
                            dropout_p=float(args.dropout_p),
                            missing_modality_p=float(args.missing_modality_p),
                            drift_scale=float(args.drift_scale),
                            margin_threshold=float(args.margin_threshold),
                            family_alpha=0.05,
                            offset_alpha=0.32,
                            action_beta=float(action_beta),
                            correction_mix=float(correction_mix),
                            trust_temp=float(trust_temp),
                        )
                    )
                summary = summarize(seeds)
                row = {
                    "action_beta": float(action_beta),
                    "correction_mix": float(correction_mix),
                    "trust_temp": float(trust_temp),
                    **summary,
                    "loop_score_gain_vs_direct": float(summary["loop_score"] - direct["loop_score"]),
                    "loop_score_gain_vs_tuned": float(summary["loop_score"] - tuned["loop_score"]),
                    "corrected_action_gain_vs_direct": float(summary["corrected_action_accuracy"] - direct["corrected_action_accuracy"]),
                    "old_retention_gain_vs_direct": float(summary["old_concept_retention"] - direct["old_concept_retention"]),
                }
                rows.append(row)
                objective = -(row["loop_score_gain_vs_direct"]) - 0.4 * row["corrected_action_gain_vs_direct"]
                if best is None or objective < best[0]:
                    best = (objective, row)

    assert best is not None
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": float(time.time() - t0),
            "num_seeds": int(args.num_seeds),
            "config_count": len(rows),
            "source_files": [
                "open_world_grounding_action_loop_20260310.json",
            ],
        },
        "baseline_direct_action": direct,
        "baseline_tuned_action": tuned,
        "best_config": best[1],
        "rows": rows,
        "project_readout": {
            "summary": "这一版把长期动作信任状态并入统一更新律，测试动作策略和自纠错是否能借助长状态从负边界走向可用区。",
            "next_question": "如果 stateful action trust 仍然无法翻正 loop_score，下一步就必须把长期目标状态也接进来。"
        },
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["best_config"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
