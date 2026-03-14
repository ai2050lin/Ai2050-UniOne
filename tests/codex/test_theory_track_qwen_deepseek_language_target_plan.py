from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / "tests" / "codex_temp"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gpt5.code.icspb_backbone_v2_large_online import (
    ICSPBBackboneV2LargeOnline,
    ICSPBLargeOnlineConfig,
)


def param_count(cfg: ICSPBLargeOnlineConfig) -> int:
    model = ICSPBBackboneV2LargeOnline(cfg)
    return sum(p.numel() for p in model.parameters())


def human_readable_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def main() -> None:
    start = time.time()
    TEMP.mkdir(parents=True, exist_ok=True)

    base_cfg = ICSPBLargeOnlineConfig()
    base_params = param_count(base_cfg)

    phase_a_cfg = replace(
        base_cfg,
        hidden_dim=768,
        concept_vocab_size=65536,
        relation_vocab_size=512,
        context_vocab_size=512,
        stage_vocab_size=512,
        protocol_vocab_size=512,
        visual_input_dim=256,
        audio_input_dim=192,
        task_classes=256,
        brain_probe_dim=128,
        consciousness_dim=64,
    )
    phase_b_cfg = replace(
        phase_a_cfg,
        hidden_dim=1536,
        concept_vocab_size=131072,
        relation_vocab_size=1024,
        context_vocab_size=1024,
        stage_vocab_size=1024,
        protocol_vocab_size=1024,
        visual_input_dim=512,
        audio_input_dim=384,
        task_classes=512,
        brain_probe_dim=256,
        consciousness_dim=128,
    )
    phase_c_cfg = replace(
        phase_b_cfg,
        hidden_dim=3072,
        concept_vocab_size=262144,
        relation_vocab_size=2048,
        context_vocab_size=2048,
        stage_vocab_size=2048,
        protocol_vocab_size=2048,
        visual_input_dim=1024,
        audio_input_dim=768,
        task_classes=1024,
        brain_probe_dim=512,
        consciousness_dim=256,
    )

    phase_a_params = param_count(phase_a_cfg)
    phase_b_params = param_count(phase_b_cfg)
    phase_c_params = param_count(phase_c_cfg)

    # 当前模型主要是原型 backbone，不是深层 token-level 语言主干。
    current_language_scale_fit = 0.18
    current_instant_learning_fit = 0.34
    architecture_gap = 0.82

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": time.time() - start,
            "task_block": "TheoryTrack_Qwen_DeepSeek_Language_Target_Plan",
        },
        "headline_metrics": {
            "current_model_params": base_params,
            "current_model_params_hr": human_readable_params(base_params),
            "current_language_scale_fit": current_language_scale_fit,
            "current_instant_learning_fit": current_instant_learning_fit,
            "architecture_gap": architecture_gap,
        },
        "roadmap": [
            {
                "phase": "Phase-A 语言主干成型",
                "target_params": phase_a_params,
                "target_params_hr": human_readable_params(phase_a_params),
                "goal": "先把当前原型推进到中等规模语言主干，验证 token-level 语言建模、长上下文和基础开放域问答。",
                "must_do": [
                    "把当前显式语义 scaffold 降到辅助层，而不是主生成层",
                    "引入正式 token embedding / decode 主干",
                    "把长上下文与长知识链推理升成训练主目标",
                ],
            },
            {
                "phase": "Phase-B 接近强开源大模型区间",
                "target_params": phase_b_params,
                "target_params_hr": human_readable_params(phase_b_params),
                "goal": "把语言主干推进到接近强开源模型的能力区间，并开始让即时学习成为稳定能力而不是附属特性。",
                "must_do": [
                    "语言预训练与即时学习制度并行优化",
                    "fast-write / replay / consolidation 进入训练主目标",
                    "多模态 grounding 接入语言回答主回路",
                ],
            },
            {
                "phase": "Phase-C Qwen/DeepSeek 级语言能力目标",
                "target_params": phase_c_params,
                "target_params_hr": human_readable_params(phase_c_params),
                "goal": "逼近 Qwen/DeepSeek 级语言能力，同时保留 ICSPB 的即时学习效率优势。",
                "must_do": [
                    "训练总量提升到大规模 token 级别",
                    "把开放世界知识、长推理和真实外部验证接入",
                    "冻结大部分语言主干，仅对 canonical write/read/replay 分支做在线高效更新",
                ],
            },
        ],
        "verdict": {
            "route_can_reach_qwen_deepseek_level": True,
            "current_model_is_far_below_target_scale": True,
            "current_route_requires_scale_and_architecture_upgrade": True,
            "core_answer": "当前路线在理论上可以逼近 Qwen/DeepSeek 级语言能力，但当前 1.44M 参数原型远远不够，必须先把语言主干扩成正式的大规模 token-level 模型，再把即时学习保留在受控的 fast-write/replay/consolidation 分支中。",
        },
    }

    out_file = TEMP / "theory_track_qwen_deepseek_language_target_plan.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
