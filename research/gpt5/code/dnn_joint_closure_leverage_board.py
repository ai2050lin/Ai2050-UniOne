from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import itertools
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class DnnJointClosureLeverageBoard:
    joint_closure: Dict[str, Any]

    @classmethod
    def from_artifacts(cls, root: Path) -> "DnnJointClosureLeverageBoard":
        temp = root / "tests" / "codex_temp"
        return cls(
            joint_closure=load_json(temp / "dnn_joint_exact_closure_board_block_20260315.json"),
        )

    @staticmethod
    def _coupled_score(dense: float, family: float, successor: float) -> float:
        return 0.30 * dense + 0.35 * family + 0.35 * successor

    def summary(self) -> Dict[str, Any]:
        metrics = self.joint_closure["headline_metrics"]
        baseline_dense = float(metrics["dense_evidence_score"])
        baseline_family = float(metrics["family_specific_closure_score"])
        baseline_successor = float(metrics["successor_closure_score"])
        baseline_coupled = float(metrics["coupled_exact_closure_score"])
        baseline_readiness = float(metrics["theorem_readiness_under_coupling"])

        uplift_unit = 0.10
        axis_names = {
            "dense": "dense exact evidence",
            "family": "family-to-specific exact closure",
            "successor": "successor exact closure",
        }

        scenarios: List[Dict[str, Any]] = []
        for r in (1, 2, 3):
            for combo in itertools.combinations(axis_names.keys(), r):
                dense = baseline_dense + (uplift_unit if "dense" in combo else 0.0)
                family = baseline_family + (uplift_unit if "family" in combo else 0.0)
                successor = baseline_successor + (uplift_unit if "successor" in combo else 0.0)
                new_coupled = self._coupled_score(
                    clamp01(dense),
                    clamp01(family),
                    clamp01(successor),
                )
                delta_coupled = new_coupled - baseline_coupled
                new_readiness = baseline_readiness + 0.35 * delta_coupled
                scenarios.append({
                    "combo": [axis_names[key] for key in combo],
                    "combo_key": "+".join(combo),
                    "delta_coupled": round(delta_coupled, 4),
                    "new_coupled": round(new_coupled, 4),
                    "new_readiness": round(new_readiness, 4),
                })

        scenarios = sorted(scenarios, key=lambda item: (item["delta_coupled"], item["new_readiness"]), reverse=True)

        best_single = next(item for item in scenarios if len(item["combo"]) == 1)
        best_pair = next(item for item in scenarios if len(item["combo"]) == 2)
        best_all = next(item for item in scenarios if len(item["combo"]) == 3)

        metric_lines_cn = [
            f"（单块最佳杠杆提升）best_single_delta = {best_single['delta_coupled']:.4f}",
            f"（双块最佳杠杆提升）best_pair_delta = {best_pair['delta_coupled']:.4f}",
            f"（三块联动总提升）best_all_delta = {best_all['delta_coupled']:.4f}",
            f"（当前联合闭合基线）baseline_coupled_exact_closure = {baseline_coupled:.4f}",
            f"（当前系统准备度基线）baseline_theorem_readiness = {baseline_readiness:.4f}",
        ]

        return {
            "baseline_coupled_exact_closure": baseline_coupled,
            "baseline_theorem_readiness": baseline_readiness,
            "best_single_scenario": best_single,
            "best_pair_scenario": best_pair,
            "best_all_scenario": best_all,
            "all_scenarios": scenarios,
            "metric_lines_cn": metric_lines_cn,
            "critical_path": [
                f"单块优先级最高的是 {best_single['combo'][0]}",
                f"双块联动优先级最高的是 {' + '.join(best_pair['combo'])}",
                "最终突破仍然需要三块联动，不能长期停在单块最优策略上",
            ],
        }
