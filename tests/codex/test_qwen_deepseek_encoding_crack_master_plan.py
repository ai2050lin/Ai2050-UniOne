from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]


def load_json(rel_path: str) -> Dict[str, Any]:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def build_blocks() -> List[Dict[str, Any]]:
    core_gaps = load_json("tests/codex_temp/theory_track_encoding_mechanism_core_gaps_20260312.json")
    math_mechanism = load_json("tests/codex_temp/qwen3_deepseek_family_patch_offset_math_mechanism_20260315.json")
    micro_meso_macro = load_json("tests/codex_temp/qwen_deepseek_micro_meso_macro_encoding_map_20260315.json")
    analytic_transfer = load_json("tests/codex_temp/qwen_deepseek_analytic_family_transfer_law_20260315.json")
    universal_generator = load_json("tests/codex_temp/qwen_deepseek_universal_family_state_generator_20260315.json")

    gaps = {row["name"]: row for row in core_gaps["gaps"]}
    math_progress = math_mechanism["progress_estimate"]
    scale_progress = micro_meso_macro["progress_estimate"]
    transfer_progress = analytic_transfer["progress_estimate"]
    generator_progress = universal_generator["progress_estimate"]

    return [
        {
            "order": 1,
            "block": "universal_family_operator_closure",
            "goal": "把单族到多族的候选生成器升级为连续跨族算子族，而不是支撑维度重映射。",
            "why_now": "这是从“知道一个族推其他族”走向可泛化闭式的第一门槛。",
            "current_percent": round(
                (transfer_progress["closed_form_family_transfer_percent"] + generator_progress["single_family_to_multi_family_generator_percent"]) / 2.0,
                1,
            ),
            "closes_gaps": [
                "object_to_readout_compatibility",
                "bridge_role_dense_coupling",
            ],
            "depends_on": [],
            "deliverables": [
                "统一的 T_(a->b) 连续算子族定义",
                "至少 5 个大族的基底与原型偏置可闭式生成",
                "跨 Qwen/DeepSeek 的同口径算子比较",
            ],
            "exit_criteria": [
                "不再依赖手工 support slot remap",
                "同一算子族可覆盖 fruit/animal/vehicle/object/abstract",
                "跨模型平均 family basis 生成一致性达到可接受阈值",
            ],
        },
        {
            "order": 2,
            "block": "concept_local_residual_auto_factorization",
            "goal": "把概念局部残差从手工属性包升级为自动分解，得到可扩展的 concept offset 生成律。",
            "why_now": "如果 concept-local residual 还靠手工指定，就不能声称已经破解概念编码机制。",
            "current_percent": round(
                (math_progress["concept_offset_math_percent"] + generator_progress["single_family_to_multi_concept_generator_percent"]) / 2.0,
                1,
            ),
            "closes_gaps": [
                "object_to_readout_compatibility",
                "reasoning_slice_engineering_integration",
            ],
            "depends_on": ["universal_family_operator_closure"],
            "deliverables": [
                "自动分解出的 local basis / shared scaffold / residual 三层结构",
                "hundreds-scale 概念集的 offset 稀疏展开",
                "Micro/Meso/Macro 三尺度统一到同一分解器",
            ],
            "exit_criteria": [
                "新概念状态不再需要手工属性包指定",
                "offset 分解可稳定解释同族与跨族概念差异",
                "Qwen 与 DeepSeek 上的主要因子方向可对齐",
            ],
        },
        {
            "order": 3,
            "block": "adaptive_offset_dynamic_law",
            "goal": "闭合新概念写入、保留、切换、回放、固化的统一动态学习律。",
            "why_now": "当前最大的数学缺口仍是 offset 如何形成和更新，而不是静态几何本身。",
            "current_percent": float(math_progress["dynamic_learning_law_percent"]),
            "closes_gaps": [
                "stress_bound_dynamic_update_closure",
            ],
            "depends_on": [
                "universal_family_operator_closure",
                "concept_local_residual_auto_factorization",
            ],
            "deliverables": [
                "adaptive_offset_(t+1)=F(offset_t, novelty, routing, replay, stabilization) 的统一形式",
                "新概念首次进入时的状态生成与后续固化轨迹",
                "novelty-retention-switching 联合基准",
            ],
            "exit_criteria": [
                "动态更新不再只靠静态拟合解释",
                "同一学习律可同时解释写入、保留、切换",
                "压力条件下仍能维持 patch 与 offset 的稳定结构",
            ],
        },
        {
            "order": 4,
            "block": "readout_transport_bridge_unified_state_equation",
            "goal": "把 restricted readout、stage-conditioned transport、successor transport、protocol bridge 并入同一状态方程。",
            "why_now": "只生成概念状态还不够，必须解释这些状态如何被读出、运输并进入任务接口。",
            "current_percent": round(
                (
                    math_progress["family_patch_plus_offset_joint_mechanism_percent"]
                    + scale_progress["macro_abstraction_relation_protocol_mechanism_percent"]
                    + generator_progress["whole_network_state_generator_percent"]
                )
                / 3.0,
                1,
            ),
            "closes_gaps": [
                "object_to_readout_compatibility",
                "bridge_role_dense_coupling",
                "reasoning_slice_engineering_integration",
            ],
            "depends_on": [
                "universal_family_operator_closure",
                "concept_local_residual_auto_factorization",
                "adaptive_offset_dynamic_law",
            ],
            "deliverables": [
                "统一状态方程：对象基底、偏置、关系扰动、阶段门控、读出约束、桥接接口",
                "successor / readout / protocol 三线联合评估",
                "从概念状态到任务端口的可追踪链路",
            ],
            "exit_criteria": [
                "readout 不再是独立附加模块",
                "stage / successor / protocol 可以在同一方程内解释",
                "对象状态到外部接口的路径可连续追踪",
            ],
        },
        {
            "order": 5,
            "block": "whole_network_state_generator",
            "goal": "把族级与概念级闭式推进到全网候选状态生成器，覆盖更多层、更多概念、更多神经元子状态。",
            "why_now": "这是你提出的终极目标的直接工程形态：从局部编码规律走向全网状态生成。",
            "current_percent": float(generator_progress["whole_network_state_generator_percent"]),
            "closes_gaps": [
                "object_to_readout_compatibility",
                "bridge_role_dense_coupling",
                "stress_bound_dynamic_update_closure",
            ],
            "depends_on": [
                "universal_family_operator_closure",
                "concept_local_residual_auto_factorization",
                "adaptive_offset_dynamic_law",
                "readout_transport_bridge_unified_state_equation",
            ],
            "deliverables": [
                "覆盖 hundreds-scale 概念和多族的状态生成器",
                "分层神经元状态近似生成与误差分解",
                "Qwen / DeepSeek 双模型同口径全网候选状态图谱",
            ],
            "exit_criteria": [
                "从单族锚点可以外推到大规模多族多概念",
                "全网状态生成误差可被明确分解到少数剩余项",
                "不同模型上生成规律具有稳定可迁移性",
            ],
        },
        {
            "order": 6,
            "block": "brain_side_causal_falsification_closure",
            "goal": "把 DNN 侧闭式外推到脑侧因果验证，证明这不是只适用于 Transformer 的工程拟合。",
            "why_now": "如果没有脑侧因果检验，就不能声称已经破解更一般的大脑编码机制。",
            "current_percent": 22.0,
            "closes_gaps": [
                "brain_side_causal_closure",
            ],
            "depends_on": [
                "whole_network_state_generator",
            ],
            "deliverables": [
                "脑区级 family patch / concept offset / readout / transport 对应预测",
                "干预与证伪报告",
                "多模态一致性与脑区异同的统一解释",
            ],
            "exit_criteria": [
                "至少形成可证伪的脑侧预测集",
                "DNN 侧闭式能解释脑区间共性与差异",
                "失败案例可明确回写到哪一个算子或学习律假设错误",
            ],
        },
    ]


def build_payload() -> Dict[str, Any]:
    t0 = time.time()
    core_gaps = load_json("tests/codex_temp/theory_track_encoding_mechanism_core_gaps_20260312.json")
    math_mechanism = load_json("tests/codex_temp/qwen3_deepseek_family_patch_offset_math_mechanism_20260315.json")
    universal_generator = load_json("tests/codex_temp/qwen_deepseek_universal_family_state_generator_20260315.json")
    blocks = build_blocks()

    highest_severity = [gap["name"] for gap in core_gaps["gaps"] if gap["severity"] == "highest"]
    average_plan_progress = round(sum(block["current_percent"] for block in blocks) / len(blocks), 1)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "QwenDeepSeek_encoding_crack_master_plan",
        },
        "strict_goal": {
            "statement": "从单族编码规律出发，逐步推进到多族、多概念、全网候选状态生成，并最终外推到脑侧可证伪编码机制。",
            "boundary": "当前还远未达到终局，只是把剩余硬伤整理成一条可执行主路线。",
        },
        "current_hard_gaps": core_gaps["gaps"],
        "current_progress_reference": {
            "family_patch_math_percent": math_mechanism["progress_estimate"]["family_patch_math_percent"],
            "concept_offset_math_percent": math_mechanism["progress_estimate"]["concept_offset_math_percent"],
            "dynamic_learning_law_percent": math_mechanism["progress_estimate"]["dynamic_learning_law_percent"],
            "whole_network_state_generator_percent": universal_generator["progress_estimate"]["whole_network_state_generator_percent"],
            "full_brain_encoding_mechanism_percent": universal_generator["progress_estimate"]["full_brain_encoding_mechanism_percent"],
        },
        "execution_strategy": {
            "principle": "按大块推进，每一块都必须绑定明确硬伤、交付物和验收门槛，避免继续零散补丁化。",
            "highest_severity_first": highest_severity,
            "recommended_now": blocks[0]["block"],
            "average_plan_progress_percent": average_plan_progress,
        },
        "plan_blocks": blocks,
        "completion_definition": {
            "for_dnn": "至少能从单族锚点稳定外推到大规模多族多概念，并对主要神经元状态形成低残差候选生成。",
            "for_brain": "至少能提出脑侧 family patch / concept offset / readout / transport 的成体系可证伪预测。",
            "not_enough": [
                "只有静态几何拟合",
                "只有少量概念的局部解释",
                "只有前端动画或口头理论",
            ],
        },
    }
    return payload


def test_qwen_deepseek_encoding_crack_master_plan() -> None:
    payload = build_payload()
    assert len(payload["plan_blocks"]) == 6
    assert payload["execution_strategy"]["recommended_now"] == "universal_family_operator_closure"
    assert "brain_side_causal_closure" in payload["execution_strategy"]["highest_severity_first"]
    assert payload["plan_blocks"][2]["block"] == "adaptive_offset_dynamic_law"


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen/DeepSeek 编码机制总计划")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/qwen_deepseek_encoding_crack_master_plan_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["execution_strategy"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
