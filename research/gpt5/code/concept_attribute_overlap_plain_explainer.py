from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ConceptAttributeOverlapPlainExplainer:
    triscale_map: Dict[str, Any]
    family_math: Dict[str, Any]
    apple_prediction: Dict[str, Any]
    exact_system: Dict[str, Any]

    @classmethod
    def from_artifacts(cls, root: Path) -> "ConceptAttributeOverlapPlainExplainer":
        temp = root / "tests" / "codex_temp"
        return cls(
            triscale_map=load_json(temp / "qwen_deepseek_micro_meso_macro_encoding_map_20260315.json"),
            family_math=load_json(temp / "qwen3_deepseek_family_patch_offset_math_mechanism_20260315.json"),
            apple_prediction=load_json(temp / "apple_dnn_brain_prediction_block.json"),
            exact_system=load_json(temp / "dnn_exact_encoding_system_block_20260315.json"),
        )

    def summary(self) -> Dict[str, Any]:
        micro = self.triscale_map["three_scales"]["micro"]["direct_evidence"]
        family = self.family_math["cross_model_summary"]
        apple = self.apple_prediction["dnn_prediction"]
        exact = self.exact_system["headline_metrics"]

        metric_lines_cn = [
            f"（圆形属性轴对齐度）round_axis_alignment = {micro['round_axis_alignment']:.4f}",
            f"（family贴合强度）mean_family_fit_strength = {family['mean_family_fit_strength']:.4f}",
            f"（错误family间隔）mean_wrong_family_margin = {family['mean_wrong_family_margin']:.4f}",
            f"（apple稀疏偏移支持）sparse_offset_support = {apple['predicted_mechanism']['sparse_offset_support']:.4f}",
            f"（跨维解耦指数）deepseek_cross_dim_decoupling_index = {apple['predicted_mechanism']['deepseek_cross_dim_decoupling_index']:.4f}",
            f"（系统精确闭合度）exact_system_closure_score = {exact['exact_system_closure_score']:.4f}",
        ]

        return {
            "metric_lines_cn": metric_lines_cn,
            "plain_answer": {
                "short_answer": "更合理的答案是：有交集，但不是完全相同的一组神经元。共享的是一部分“圆形相关线路”，不同的是苹果和月亮各自的对象身份、上下文和角色线路。",
                "what_data_supports_this": [
                    "现有三尺度结果里，round_axis_alignment 较高，说明“圆形”不是纯噪声，而是相对稳定的属性方向。",
                    "family_fit_strength 和 wrong_family_margin 较高，说明苹果和月亮首先会落在不同 family patch 里：苹果更偏 fruit，月亮更偏 celestial。",
                    "apple_dnn_brain_prediction 里已经把苹果描述成“fruit family patch + apple-specific sparse offset + local attribute fibers”，这说明圆形更像挂在对象上的属性纤维，而不是脱离对象独立存在的整块编码。",
                ],
                "logic": [
                    "先分清对象层和属性层。苹果和月亮不是一个对象家族，所以它们的主编码不会相同。",
                    "再看属性层。圆形更像一条可复用的属性方向，苹果可以沿这条方向偏，月亮也可以沿这条方向偏。",
                    "所以最终不是“苹果的圆形神经元 = 月亮的圆形神经元”，而是“它们会部分共用圆形相关子空间，但落点不同”。",
                ],
                "apple_moon_walkthrough": [
                    "苹果：先进入 fruit family patch，再叠加 Delta_apple，然后再叠加 round 这条属性纤维。",
                    "月亮：先进入 celestial family patch，再叠加 Delta_moon，然后再叠加 round 这条属性纤维。",
                    "因为前面的 family patch 和 concept offset 不同，所以完整激活图不会一样；但因为后面都叠加了 round 属性纤维，所以会有部分交集。",
                ],
                "candidate_formula": [
                    "h_apple_round ~= B_fruit + Delta_apple + a_round * u_round + epsilon",
                    "h_moon_round ~= B_celestial + Delta_moon + b_round * u_round + epsilon",
                ],
                "plain_meaning": "翻成普通话，就是：苹果和月亮都可能会用到一条“圆形相关线路”，但苹果先是苹果，月亮先是月亮。圆形只是挂在它们各自身上的共同属性，不会把它们变成同一个对象编码。",
                "remaining_problems": [
                    "现在还没有 apple-round vs moon-round 的 dense neuron overlap 直接实测，所以这仍然是强推断，不是最终定理。",
                    "现有证据更强地支持“属性方向可复用”，还没有把“重叠到具体哪些 neuron（神经元）”这一级完全量出来。",
                    "exact_system_closure 仍然偏低，说明系统级精确闭合还没完成。",
                ],
            },
        }
