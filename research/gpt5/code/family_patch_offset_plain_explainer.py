from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class FamilyPatchOffsetPlainExplainer:
    family_math: Dict[str, Any]
    triscale_map: Dict[str, Any]
    specific_bridge: Dict[str, Any]
    exact_system: Dict[str, Any]

    @classmethod
    def from_artifacts(cls, root: Path) -> "FamilyPatchOffsetPlainExplainer":
        temp = root / "tests" / "codex_temp"
        return cls(
            family_math=load_json(temp / "qwen3_deepseek_family_patch_offset_math_mechanism_20260315.json"),
            triscale_map=load_json(temp / "qwen_deepseek_micro_meso_macro_encoding_map_20260315.json"),
            specific_bridge=load_json(temp / "dnn_specific_math_bridge_block_20260315.json"),
            exact_system=load_json(temp / "dnn_exact_encoding_system_block_20260315.json"),
        )

    def summary(self) -> Dict[str, Any]:
        family_direct = self.family_math["cross_model_summary"]
        meso = self.triscale_map["three_scales"]["meso"]
        examples = self.triscale_map["example_mechanisms"]
        specific = self.specific_bridge["headline_metrics"]
        exact = self.exact_system["headline_metrics"]

        metric_lines_cn = [
            f"（family贴合强度）mean_family_fit_strength = {family_direct['mean_family_fit_strength']:.4f}",
            f"（错误family间隔）mean_wrong_family_margin = {family_direct['mean_wrong_family_margin']:.4f}",
            f"（apple到banana距离）apple_banana_distance = {meso['direct_evidence']['apple_banana_distance']:.4f}",
            f"（apple到pear距离）apple_pear_distance = {meso['direct_evidence']['apple_pear_distance']:.4f}",
            f"（concept offset参数恢复）specific_parametric_restoration_score = {specific['specific_parametric_restoration_score']:.4f}",
            f"（系统精确闭合度）exact_system_closure_score = {exact['exact_system_closure_score']:.4f}",
        ]

        return {
            "metric_lines_cn": metric_lines_cn,
            "plain_answer": {
                "family_patch": {
                    "what_data_was_extracted": [
                        "从 qwen3 和 deepseek 里抽取了同族概念的中心、投影残差和跨族间隔，观察同族概念能不能落到同一块局部区域里。",
                        "直接指标包括 family_fit_strength、wrong_family_margin，以及 apple / banana / pear 这些同族概念之间的距离。",
                    ],
                    "logic": [
                        "先问一个最朴素的问题：苹果、香蕉、梨在模型里是不是彼此更靠近，而不是随机散开。",
                        "如果同族概念总能落到同一块局部空间，那就说明模型里确实存在一块共享的“水果公共底板”，这就是 family patch。",
                    ],
                    "result": [
                        f"当前跨模型 mean_family_fit_strength = {family_direct['mean_family_fit_strength']:.4f}，说明 family patch 不是弱信号。",
                        f"mean_wrong_family_margin = {family_direct['mean_wrong_family_margin']:.4f}，说明它和错误家族之间还有明显边界。",
                        f"apple_banana_distance = {meso['direct_evidence']['apple_banana_distance']:.4f}，apple_pear_distance = {meso['direct_evidence']['apple_pear_distance']:.4f}，都支持“苹果和梨、香蕉共享 fruit family patch”。",
                    ],
                    "remaining_problems": [
                        "现在可以比较强地说 family patch 存在，但还没有唯一 final theorem（最终定理）。",
                        "不同模型里 family patch 最强的层位并不完全一样，说明机制相似，但工作点还没有统一。",
                    ],
                },
                "concept_offset": {
                    "what_data_was_extracted": [
                        "抽取了苹果相对 fruit family 的偏移、同族概念之间的差异，以及苹果颜色、味道、圆润度等属性轴。",
                        "直接指标包括 shared_norm_ratio、offset_top32_energy_ratio、specific_parametric_restoration_score。",
                    ],
                    "logic": [
                        "如果苹果只是 fruit family 里的一个具体成员，那它就应该等于“水果公共底板 + 苹果自己的小偏移”。",
                        "这个小偏移不能太大，否则它就不再是“水果中的苹果”，而会变成完全独立的新编码。",
                    ],
                    "result": [
                        "当前 candidate law（候选规律）已经比较稳定：苹果不是整块重建，而是在 family basis（家族基底）上叠加 concept offset（概念偏移），再叠加上下文和协议修正。",
                        f"specific_parametric_restoration_score = {specific['specific_parametric_restoration_score']:.4f}，说明这种解释已经很强。",
                        f"exact_system_closure_score = {exact['exact_system_closure_score']:.4f}，说明“参数上讲得通”和“最终精确闭合”之间还有明显距离。",
                    ],
                    "remaining_problems": [
                        "family-to-specific exact closure（从家族层精确推回具体概念）还没打穿，也就是还不能完全从 family patch 精确算出苹果的全部细节。",
                        "新概念第一次进入模型时，这个 offset（偏移）到底是怎么写进去的，动态学习律还没有闭合。",
                    ],
                },
                "apple_walkthrough": {
                    "steps": [
                        "第一步，把苹果、香蕉、梨这些概念放进同一个 fruit family 里比较，确认它们彼此比跨家族概念更近。",
                        "第二步，把苹果看成“水果公共底板 + 苹果偏移”。公共底板解释“它为什么还是水果”，偏移解释“它为什么是苹果不是香蕉”。",
                        "第三步，再在苹果偏移上叠加属性轴，比如圆、甜、红。这样就能解释苹果的颜色和味道。",
                        "第四步，如果把“苹果”提升到“水果”或“物体”，就不是简单换词，而是从对象层进入更高层的类别/角色桥接。",
                    ],
                    "candidate_formula": examples["apple_color_taste"]["candidate_formula"],
                    "plain_meaning": "普通话来说，就是：先有一块“水果地盘”，苹果先站进这块地盘；然后再往“苹果自己的方向”偏一点；最后再挂上红、甜、圆这些属性。这样模型里才会同时保留“它是水果”和“它是苹果”。",
                },
            },
        }
