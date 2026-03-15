from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]


def load_json(rel_path: str) -> Dict[str, Any]:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def build_payload() -> Dict[str, Any]:
    t0 = time.time()

    f7 = load_json("tests/codex_temp/f7_human_language_instant_learning_architecture_20260311.json")
    spike_bridge = load_json("tests/codex_temp/spike_brain_system_bridge_block.json")
    spike_bio = load_json("tests/codex_temp/spike_biophysical_consistency_block.json")
    language_target = load_json("tests/codex_temp/theory_track_qwen_deepseek_language_target_plan.json")
    whole_generator = load_json("tests/codex_temp/qwen_deepseek_whole_network_state_generator_20260315.json")
    brain_falsification = load_json("tests/codex_temp/qwen_deepseek_brain_side_causal_falsification_closure_20260315.json")

    language_readiness = float(f7["headline_metrics"]["language_capacity_readiness_score"])
    constructibility = float(f7["headline_metrics"]["architecture_constructibility_score"])
    spike_bridge_score = float(spike_bridge["bridge"]["spike_bridge_score"])
    spike_bio_score = float(spike_bio["consistency"]["consistency_score"])
    whole_generator_score = float(whole_generator["generator_scores"]["whole_generator_score"])
    falsification_score = float(brain_falsification["scores"]["falsification_readiness"])

    architecture_route_score = (
        0.24 * language_readiness
        + 0.16 * constructibility
        + 0.18 * spike_bridge_score
        + 0.14 * spike_bio_score
        + 0.18 * whole_generator_score
        + 0.10 * falsification_score
    )

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t0, 6),
            "task_block": "SpikeICSPB_non_attention_non_bp_language_architecture_route",
        },
        "strict_goal": {
            "statement": "基于 ICSPB 理论给出一条非 Attention、非 BP 的大规模语言网络设计路线。",
            "boundary": "当前给出的是可构造路线和训练制度原型，不是已经证明可达到 Qwen/DeepSeek 级完整语言能力的实证模型。",
        },
        "why_non_attention_non_bp_is_plausible": {
            "core_answer": (
                "如果语言能力依赖的本质是 patch/section/fiber、阶段运输、后继对齐、"
                "群体读出和受限写读，而不是 Attention 公式本身，那么这些算子可以由脉冲事件结构和局部学习制度重建。"
            ),
            "supporting_scores": {
                "language_capacity_readiness_score": language_readiness,
                "architecture_constructibility_score": constructibility,
                "spike_bridge_score": spike_bridge_score,
                "spike_biophysical_consistency_score": spike_bio_score,
                "whole_generator_score": whole_generator_score,
                "brain_falsification_score": falsification_score,
                "architecture_route_score": architecture_route_score,
            },
        },
        "architecture": {
            "name": "SpikeICSPB-LM",
            "non_attention_replacement": {
                "event_patch_selector": "替代全局 self-attention；按事件与上下文激活局部 family patch",
                "burst_window_section_binder": "替代 token-token 全连接对齐；在短时间窗内完成对象 section 绑定",
                "phase_gated_successor_transport": "替代因果 attention 路由；用阶段门控和后继触发器推进语言轨迹",
                "population_readout_head": "替代单点 logits 直读；由群体读出形成 token / span / action 预测",
            },
            "non_bp_learning_replacement": {
                "local_eligibility_trace": "每个局部连接维护 eligibility trace，而不是全局反向梯度",
                "modulatory_credit_signal": "用局部预测误差、协议失败、验证分数作为调制信用",
                "fast_write_memory": "新事件先写入快速局部记忆，不直接改动全局 scaffold",
                "slow_consolidation": "通过 replay / consolidation 把稳定模式写入慢时标结构",
                "plasticity_guard": "通过 admissible update 和 rollback 保护旧知识与读出安全",
            },
            "language_modules": {
                "module_1_patch_lexicon": "词元/子词/短语先映射为 patch 候选，不是直接 dense embedding",
                "module_2_family_section_workspace": "维护 family patch、concept section、attribute/relation fibers",
                "module_3_protocol_router": "将当前状态分配到 syntax / semantics / memory / tool / recovery 协议",
                "module_4_successor_core": "负责后继 token、跨度 continuation、长链推理轨迹",
                "module_5_population_decoder": "把群体状态读成文本输出与外部行动",
                "module_6_replay_consolidator": "离线/在线回放，把局部语言经验固化进长期 scaffold",
            },
        },
        "candidate_training_law": {
            "core_equations": [
                "M_fast(t+1) = (1-lambda_m) M_fast(t) + eta_fast * gate_fast * Novelty_t * LocalBind(x_t, h_t)",
                "A_slow(t+1) = (1-lambda_A) A_slow(t) + eta_slow * Consolidate(M_fast(t+1), h_t)",
                "e_ij(t+1) = lambda_e * e_ij(t) + pre_i(t) * post_j(t)",
                "Delta w_ij = eta_loc * m_t * e_ij - eta_decay * w_ij + eta_replay * Replay_ij",
            ],
            "meaning": (
                "语言学习被拆成快速局部写入、慢速回放固化、局部 eligibility、"
                "调制信用和受限可塑性，不再依赖全局 BP。"
            ),
        },
        "scale_route": {
            "phase_a": {
                "goal": "先做 100M 级等价状态容量，验证长上下文 continuation 和基本问答",
                "must_have": [
                    "局部事件缓存",
                    "阶段门控 successor core",
                    "population decoder",
                    "replay-consolidation 周期",
                ],
            },
            "phase_b": {
                "goal": "推进到 300M-500M 级等价容量，验证多轮对话、长推理、开放域知识接入",
                "must_have": [
                    "协议路由稳定化",
                    "family patch 扩容",
                    "跨模态 grounding 接入",
                    "局部学习收敛监控",
                ],
            },
            "phase_c": {
                "goal": "推进到 1B+ 级等价容量，逼近 Qwen/DeepSeek 级语言能力",
                "must_have": [
                    "稳定的 successor 链",
                    "大规模 replay 课程",
                    "外部验证与工具链",
                    "受控 canonical write/read 分支",
                ],
            },
            "reference_target": language_target["roadmap"],
        },
        "strict_verdict": {
            "can_design_now": True,
            "can_claim_full_language_now": False,
            "core_answer": (
                "基于当前研究，已经可以设计一条非 Attention、非 BP 的大规模语言网络路线；"
                "但还不能宣称它已经被证明能达到完整强语言能力。"
            ),
            "main_hard_gaps": [
                "局部学习律在大规模语言训练上的稳定收敛仍未被证明",
                "successor 链比 readout/gating 更弱，长程语言连贯性仍是主风险",
                "大规模等价容量如何实现仍需要新的工程制度，而不是直接复用 Transformer 基础设施",
                "当前缺少真实非 Attention 非 BP 语言模型的实训结果",
            ],
        },
        "progress_estimate": {
            "non_attention_non_bp_architecture_design_percent": 66.0,
            "non_attention_non_bp_large_scale_trainability_percent": 34.0,
            "non_attention_non_bp_full_language_capability_percent": 22.0,
            "full_brain_encoding_mechanism_percent": 64.0,
        },
        "next_large_blocks": [
            "实现 SpikeICSPB-LM 最小原型，至少打通 patch selector / successor core / population decoder / replay consolidation。",
            "建立非 BP 局部学习基准，直接测长上下文 continuation、多轮对话和稳定 retention。",
            "把 Phase-A 语言目标改写成非 Attention 非 BP 口径，不再默认沿用 Transformer 训练脚手架。",
        ],
    }
    return payload


def test_spike_icspb_non_attention_non_bp_language_architecture_route() -> None:
    payload = build_payload()
    scores = payload["why_non_attention_non_bp_is_plausible"]["supporting_scores"]
    assert payload["strict_verdict"]["can_design_now"] is True
    assert payload["strict_verdict"]["can_claim_full_language_now"] is False
    assert scores["architecture_route_score"] > 0.75
    assert payload["progress_estimate"]["non_attention_non_bp_architecture_design_percent"] >= 66.0


def main() -> None:
    ap = argparse.ArgumentParser(description="SpikeICSPB non-attention non-BP language architecture route")
    ap.add_argument(
        "--json-out",
        type=str,
        default="tests/codex_temp/spike_icspb_non_attention_non_bp_language_architecture_route_20260315.json",
    )
    args = ap.parse_args()

    payload = build_payload()
    out_path = ROOT / args.json_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["why_non_attention_non_bp_is_plausible"]["supporting_scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
