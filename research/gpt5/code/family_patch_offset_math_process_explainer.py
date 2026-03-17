from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class FamilyPatchOffsetMathProcessExplainer:
    family_math: Dict[str, Any]
    triscale_map: Dict[str, Any]
    specific_bridge: Dict[str, Any]
    exact_system: Dict[str, Any]

    @classmethod
    def from_artifacts(cls, root: Path) -> "FamilyPatchOffsetMathProcessExplainer":
        temp = root / "tests" / "codex_temp"
        return cls(
            family_math=load_json(temp / "qwen3_deepseek_family_patch_offset_math_mechanism_20260315.json"),
            triscale_map=load_json(temp / "qwen_deepseek_micro_meso_macro_encoding_map_20260315.json"),
            specific_bridge=load_json(temp / "dnn_specific_math_bridge_block_20260315.json"),
            exact_system=load_json(temp / "dnn_exact_encoding_system_block_20260315.json"),
        )

    def summary(self) -> Dict[str, Any]:
        cross = self.family_math["cross_model_summary"]
        candidate_math = self.family_math["candidate_math"]
        meso = self.triscale_map["three_scales"]["meso"]["direct_evidence"]
        micro = self.triscale_map["three_scales"]["micro"]["direct_evidence"]
        specific = self.specific_bridge["headline_metrics"]
        exact = self.exact_system["headline_metrics"]

        metric_lines_cn = [
            f"（family贴合强度）mean_family_fit_strength = {cross['mean_family_fit_strength']:.4f}",
            f"（错误family间隔）mean_wrong_family_margin = {cross['mean_wrong_family_margin']:.4f}",
            f"（offset前32维能量占比）mean_offset_top32_energy_ratio = {cross['mean_offset_top32_energy_ratio']:.4f}",
            f"（共享基底范数比例）mean_shared_norm_ratio = {cross['mean_shared_norm_ratio']:.4f}",
            f"（specific参数恢复强度）specific_parametric_restoration_score = {specific['specific_parametric_restoration_score']:.4f}",
            f"（系统精确闭合度）exact_system_closure_score = {exact['exact_system_closure_score']:.4f}",
        ]

        test_chain = [
            {
                "test": "tests/codex/test_qwen3_deepseek_family_patch_offset_math_mechanism.py",
                "role": "从 qwen3 和 deepseek 的真实结构结果里，先建立 family patch 和 concept offset 的候选数学式。",
                "main_outputs": [
                    "mean_family_fit_strength",
                    "mean_wrong_family_margin",
                    "mean_offset_top32_energy_ratio",
                    "mean_shared_norm_ratio",
                    "equation_1_family_projection",
                    "equation_2_state_decomposition",
                    "equation_3_offset_factorization",
                ],
            },
            {
                "test": "tests/codex/test_qwen_deepseek_micro_meso_macro_encoding_map.py",
                "role": "把抽象公式放回苹果、香蕉、梨、颜色、味道、类别提升这些具体例子里，验证它是否真的能解释对象与属性。",
                "main_outputs": [
                    "apple_banana_distance",
                    "apple_pear_distance",
                    "round_axis_alignment",
                    "apple_color_taste candidate_formula",
                ],
            },
            {
                "test": "tests/codex/test_dnn_specific_math_bridge_block.py",
                "role": "检查 family basis + offset 这条解释，能不能进一步桥接到具体概念细节。",
                "main_outputs": [
                    "specific_parametric_restoration_score",
                    "exact_specific_closure_score",
                    "family_to_specific_gap",
                ],
            },
            {
                "test": "tests/codex/test_dnn_exact_encoding_system_block.py",
                "role": "检查 family patch + concept offset 放进整个系统以后，还能不能站住。",
                "main_outputs": [
                    "basis_offset_core_score",
                    "system_parametric_score",
                    "exact_system_closure_score",
                ],
            },
        ]

        family_process = [
            {
                "step": "第一步，先采样概念状态",
                "meaning": "先从 qwen3 和 deepseek 里拿到苹果、香蕉、梨等概念在多层里的表示。",
            },
            {
                "step": "第二步，按 family 分组",
                "meaning": "把苹果、香蕉、梨放进 fruit family，把猫、狗放进 animal family。",
            },
            {
                "step": "第三步，拟合 family patch",
                "meaning": "对每个家族，拟合一块共享局部底板，而不是只取一个平均点。",
                "formula": candidate_math["equation_1_family_projection"],
            },
            {
                "step": "第四步，检查贴合度和错误间隔",
                "meaning": "看概念投到自己家族底板后的残差是不是更小，看投到错误家族后的残差是不是明显更大。",
                "current_reading": {
                    "mean_family_fit_strength": cross["mean_family_fit_strength"],
                    "mean_wrong_family_margin": cross["mean_wrong_family_margin"],
                },
            },
            {
                "step": "第五步，用苹果、香蕉、梨做直观验证",
                "meaning": "如果苹果、香蕉、梨真共享 fruit patch，那它们彼此距离应该小于跨家族距离。",
                "current_reading": {
                    "apple_banana_distance": meso["apple_banana_distance"],
                    "apple_pear_distance": meso["apple_pear_distance"],
                },
            },
        ]

        offset_process = [
            {
                "step": "第一步，把具体概念拆成底板加偏移",
                "meaning": "先假设苹果不是独立整块编码，而是水果底板加苹果偏移。",
                "formula": candidate_math["equation_2_state_decomposition"],
            },
            {
                "step": "第二步，单独计算偏移项",
                "meaning": "把苹果状态减去水果底板，得到 Delta_apple；香蕉同理得到 Delta_banana。",
            },
            {
                "step": "第三步，看偏移是不是稀疏小偏移",
                "meaning": "如果 offset 是合理机制，它的主要信息应该集中在少量方向上，而不是全空间重建。",
                "current_reading": {
                    "mean_offset_top32_energy_ratio": cross["mean_offset_top32_energy_ratio"],
                    "mean_shared_norm_ratio": cross["mean_shared_norm_ratio"],
                },
            },
            {
                "step": "第四步，把 offset 再分成局部和共享部分",
                "meaning": "检查 offset 里到底有多少是 family 内部局部基底，有多少是跨 family 共享 scaffold。",
                "formula": candidate_math["equation_3_offset_factorization"],
            },
            {
                "step": "第五步，看它能不能恢复具体概念",
                "meaning": "如果 family patch + offset 真的对，就应该能解释为什么苹果是苹果，而不是香蕉。",
                "current_reading": {
                    "specific_parametric_restoration_score": specific["specific_parametric_restoration_score"],
                    "exact_specific_closure_score": specific["exact_specific_closure_score"],
                },
            },
        ]

        apple_example = {
            "question": "苹果这个概念现在是怎么计算的",
            "walkthrough": [
                "先在 fruit family 里定位共享底板 B_fruit。",
                "再把苹果表示减去 B_fruit，得到 Delta_apple。",
                "再看 Delta_apple 的主要方向是否集中在少数维度里，而不是全空间散开。",
                "如果还要解释苹果的圆、甜、红，就在 B_fruit + Delta_apple 之外再叠加属性方向。",
            ],
            "candidate_formula": "h_apple_attr = B_fruit + Delta_apple + a_round * u_round + a_sweet * u_sweet + a_red * u_color + epsilon",
            "plain_meaning": "普通话说，就是先找到“水果共有的那部分”，再加上“苹果自己的那一点”，最后再加上圆、甜、红这些属性。",
            "current_numbers": {
                "apple_banana_distance": meso["apple_banana_distance"],
                "apple_pear_distance": meso["apple_pear_distance"],
                "round_axis_alignment": micro["round_axis_alignment"],
                "specific_parametric_restoration_score": specific["specific_parametric_restoration_score"],
            },
        }

        current_conclusion = {
            "what_is_strong_now": [
                "family patch 已经不是弱猜想，而是有跨模型贴合度和错误间隔支撑的强候选结构。",
                "concept offset 已经不是口头说法，而是能被参数恢复显著支持的候选机制。",
                "苹果、香蕉、梨这些具体例子已经能被同一套 family patch + offset 逻辑统一解释。",
            ],
            "what_is_not_solved": [
                "还不能从 family patch 精确推出苹果的全部细节，family-to-specific exact closure 还没打穿。",
                "offset 是怎么在新概念第一次进入模型时被写进去的，动态学习律还没闭合。",
                "当前大部分证据还是 row/signature 级，dense neuron-level exact tensor 还不够厚。",
            ],
        }

        return {
            "metric_lines_cn": metric_lines_cn,
            "test_chain": test_chain,
            "math_process": {
                "family_patch_process": family_process,
                "concept_offset_process": offset_process,
            },
            "apple_example": apple_example,
            "current_conclusion": current_conclusion,
        }
