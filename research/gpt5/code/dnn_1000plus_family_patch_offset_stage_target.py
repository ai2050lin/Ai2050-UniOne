from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json

from research.gpt5.code.dnn_1000plus_dense_execution_bundle import Dnn1000PlusDenseExecutionBundle


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Dnn1000PlusFamilyPatchOffsetStageTarget:
    bundle: Dict[str, Any]
    thousand_program: Dict[str, Any]
    specific_bridge: Dict[str, Any]
    math_process: Dict[str, Any]

    @classmethod
    def from_repo(cls, root: Path) -> "Dnn1000PlusFamilyPatchOffsetStageTarget":
        temp = root / "tests" / "codex_temp"
        bundle_path = temp / "dnn_1000plus_dense_execution_bundle_block_20260316.json"
        if bundle_path.exists():
            bundle = load_json(bundle_path)
        else:
            bundle = Dnn1000PlusDenseExecutionBundle.from_repo(root).summary()
        return cls(
            bundle=bundle,
            thousand_program=load_json(temp / "dnn_thousand_noun_family_patch_offset_program_block_20260316.json"),
            specific_bridge=load_json(temp / "dnn_specific_math_bridge_block_20260315.json"),
            math_process=load_json(temp / "family_patch_offset_math_process_explainer_block_20260316.json"),
        )

    def summary(self) -> Dict[str, Any]:
        bundle_metrics = self.bundle["headline_metrics"]
        current_program = self.thousand_program["headline_metrics"]
        specific_metrics = self.specific_bridge["headline_metrics"]
        family_patch_steps = self.math_process["math_process"]["family_patch_process"]
        concept_offset_steps = self.math_process["math_process"]["concept_offset_process"]

        current_family_fit = float(family_patch_steps[3]["current_reading"]["mean_family_fit_strength"])
        current_wrong_margin = float(family_patch_steps[3]["current_reading"]["mean_wrong_family_margin"])
        current_specific_closure = float(specific_metrics["exact_specific_closure_score"])

        stage_ready = float(bundle_metrics["execution_stage_readiness"])
        projected_family_fit = min(0.93, current_family_fit + 0.09 * stage_ready)
        projected_wrong_margin = min(0.88, current_wrong_margin + 0.05 * stage_ready)
        projected_specific_closure = min(0.68, current_specific_closure + 0.12 * stage_ready)
        projected_offset_top32_energy = min(
            0.42,
            float(concept_offset_steps[2]["current_reading"]["mean_offset_top32_energy_ratio"]) + 0.05 * stage_ready,
        )
        thousand_base = self.thousand_program["program"]["current_base"]
        thousand_scale_readiness = min(
            1.0,
            0.30 * min(1.0, float(thousand_base["current_mass_noun_total"]) / 600.0)
            + 0.20 * min(1.0, float(thousand_base["inventory_concepts"]) / 384.0)
            + 0.25 * 1.0
            + 0.25 * min(1.0, stage_ready),
        )

        stage_gain = projected_specific_closure - current_specific_closure
        metric_lines_cn = [
            f"（当前family贴合强度）current_family_fit_strength = {current_family_fit:.4f}",
            f"（阶段目标family贴合强度）projected_family_fit_strength = {projected_family_fit:.4f}",
            f"（当前错误family间隔）current_wrong_family_margin = {current_wrong_margin:.4f}",
            f"（阶段目标错误family间隔）projected_wrong_family_margin = {projected_wrong_margin:.4f}",
            f"（当前specific精确闭合）current_exact_specific_closure = {current_specific_closure:.4f}",
            f"（阶段目标specific精确闭合）projected_exact_specific_closure = {projected_specific_closure:.4f}",
            f"（阶段目标offset前32维能量占比）projected_offset_top32_energy_ratio = {projected_offset_top32_energy:.4f}",
            f"（本轮阶段specific提升量）projected_specific_closure_gain = {stage_gain:.4f}",
            f"（数千名词主线准备度）thousand_scale_readiness = {thousand_scale_readiness:.4f}",
        ]

        return {
            "metric_lines_cn": metric_lines_cn,
            "strict_conclusion": {
                "core_answer": "After wiring the 1000+ noun source into launchable balanced batches, the next honest stage target is not theorem closure yet. It is a stronger family-patch stability estimate, a cleaner wrong-family margin, and a first material lift on family-to-specific exact closure.",
                "main_hard_gaps": [
                    "these are stage targets projected from execution readiness, not post-harvest measured results",
                    "successor exact closure remains outside the direct benefit range of this noun-only stage",
                    "true theorem closure still requires dense neuron-level exports, not just batched noun coverage",
                ],
            },
            "headline_metrics": {
                "current_family_fit_strength": current_family_fit,
                "projected_family_fit_strength": projected_family_fit,
                "current_wrong_family_margin": current_wrong_margin,
                "projected_wrong_family_margin": projected_wrong_margin,
                "current_exact_specific_closure": current_specific_closure,
                "projected_exact_specific_closure": projected_specific_closure,
                "projected_offset_top32_energy_ratio": projected_offset_top32_energy,
                "projected_specific_closure_gain": stage_gain,
                "metric_lines_cn": metric_lines_cn,
            },
        }
