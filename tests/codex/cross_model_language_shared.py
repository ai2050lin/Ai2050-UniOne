#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GPT2_STAGE123_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage123_route_shift_layer_localization_20260323" / "summary.json"
GPT2_STAGE124_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage124_noun_neuron_basic_probe_20260323" / "summary.json"
GPT2_STAGE130_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage130_multisyntax_noun_context_probe_20260323" / "summary.json"
GPT2_STAGE133_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage133_complex_discourse_noun_propagation_20260323" / "summary.json"
GPT2_STAGE134_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage134_noun_verb_joint_propagation_20260323" / "summary.json"
GPT2_STAGE136_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage136_anaphora_ellipsis_propagation_20260323" / "summary.json"
GPT2_STAGE137_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage137_noun_verb_result_chain_20260323" / "summary.json"
GPT2_STAGE138_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage138_conditional_gating_field_reconstruction_20260323" / "summary.json"

QWEN_STAGE139_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage139_qwen3_language_validation_suite_20260323" / "summary.json"
DEEPSEEK_STAGE140_PATH = PROJECT_ROOT / "tests" / "codex_temp" / "stage140_deepseek_language_validation_suite_20260323" / "summary.json"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def corr_to_unit(value: float) -> float:
    return clamp01((float(value) + 1.0) / 2.0)


def gpt2_bundle() -> Dict[str, object]:
    return {
        "model_key": "gpt2",
        "display_name": "GPT-2",
        "stage123": load_json(GPT2_STAGE123_PATH),
        "stage124": load_json(GPT2_STAGE124_PATH),
        "stage130": load_json(GPT2_STAGE130_PATH),
        "stage133": load_json(GPT2_STAGE133_PATH),
        "stage134": load_json(GPT2_STAGE134_PATH),
        "stage136": load_json(GPT2_STAGE136_PATH),
        "stage137": load_json(GPT2_STAGE137_PATH),
        "stage138": load_json(GPT2_STAGE138_PATH),
    }


def suite_bundle(summary_path: Path, model_key: str, display_name: str) -> Dict[str, object]:
    summary = load_json(summary_path)
    return {
        "model_key": model_key,
        "display_name": display_name,
        "summary": summary,
        "transfer": summary["transfer_summary"],
        "discourse": summary["discourse_summary"],
        "joint": summary["joint_summary"],
        "anaphora": summary["anaphora_summary"],
        "result": summary["result_summary"],
        "field": summary["field_summary"],
    }


def qwen_bundle() -> Dict[str, object]:
    return suite_bundle(QWEN_STAGE139_PATH, "qwen3", "Qwen3-4B")


def deepseek_bundle() -> Dict[str, object]:
    return suite_bundle(DEEPSEEK_STAGE140_PATH, "deepseek7b", "DeepSeek-R1-Distill-Qwen-7B")


def build_all_model_bundles() -> Dict[str, Dict[str, object]]:
    return {
        "gpt2": gpt2_bundle(),
        "qwen3": qwen_bundle(),
        "deepseek7b": deepseek_bundle(),
    }
