#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_language_shared import (
    GEMMA4_MODEL_PATH,
    GLM4_MODEL_PATH,
    PROJECT_ROOT,
    QWEN3_MODEL_PATH,
    discover_layers,
    load_gemma4_model,
    load_glm4_model,
    load_qwen3_model,
    move_batch_to_model_device,
)


DEEPSEEK_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)

MODEL_SPECS = {
    "qwen3": {
        "label": "Qwen3-4B",
        "model_path": QWEN3_MODEL_PATH,
    },
    "deepseek7b": {
        "label": "DeepSeek-R1-Distill-Qwen-7B",
        "model_path": DEEPSEEK_MODEL_PATH,
    },
    "glm4": {
        "label": "GLM-4-9B-Chat-HF",
        "model_path": GLM4_MODEL_PATH,
    },
    "gemma4": {
        "label": "Gemma-4-E2B-it",
        "model_path": GEMMA4_MODEL_PATH,
    },
}


def set_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_deepseek_model(*, prefer_cuda: bool = True):
    set_offline_env()
    want_cuda = bool(prefer_cuda and torch.cuda.is_available())
    kwargs = {
        "pretrained_model_name_or_path": str(DEEPSEEK_MODEL_PATH),
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    if want_cuda:
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = "cpu"
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(DEEPSEEK_MODEL_PATH),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_model_bundle(model_key: str, *, prefer_cuda: bool = True):
    if model_key == "qwen3":
        return load_qwen3_model(prefer_cuda=prefer_cuda)
    if model_key == "deepseek7b":
        return load_deepseek_model(prefer_cuda=prefer_cuda)
    if model_key == "glm4":
        return load_glm4_model(prefer_cuda=prefer_cuda)
    if model_key == "gemma4":
        return load_gemma4_model(prefer_cuda=prefer_cuda)
    raise KeyError(f"未知模型: {model_key}")


def free_model(model) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class ZeroModule(torch.nn.Module):
    def __init__(self, *, return_tuple: bool = False):
        super().__init__()
        self.return_tuple = return_tuple

    def forward(self, *args, **kwargs):
        hidden = None
        if args:
            hidden = args[0]
        elif "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
        if hidden is None:
            return None
        zeros = torch.zeros_like(hidden)
        return (zeros, None) if self.return_tuple else zeros


def evenly_spaced_layers(model, *, count: int = 7) -> List[int]:
    total = len(discover_layers(model))
    if total <= count:
        return list(range(total))
    layers = {0, total - 1}
    for i in range(1, count - 1):
        layers.add(round((total - 1) * i / (count - 1)))
    return sorted(layers)


def score_candidate_avg_logprob(model, tokenizer, prompt: str, candidate: str) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + candidate, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        return float("-inf")
    device = get_model_device(model)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    count = 0
    for pos in range(len(prompt_ids), len(full_ids)):
        prev = pos - 1
        token_id = full_ids[pos]
        total += float(log_probs[prev, token_id].item())
        count += 1
    if count == 0:
        return float("-inf")
    return total / count


def candidate_score_map(model, tokenizer, prompt: str, candidates: Sequence[str]) -> Dict[str, float]:
    return {
        cand: score_candidate_avg_logprob(model, tokenizer, prompt, cand)
        for cand in candidates
    }


def ablate_layer_component(model, layer_idx: int, component: str):
    layer = discover_layers(model)[layer_idx]
    if component == "attn":
        original = layer.self_attn
        layer.self_attn = ZeroModule(return_tuple=True)
        return layer, original
    if component == "mlp":
        original = layer.mlp
        layer.mlp = ZeroModule(return_tuple=False)
        return layer, original
    raise ValueError(f"未知组件: {component}")


def restore_layer_component(layer, component: str, original) -> None:
    if component == "attn":
        layer.self_attn = original
    elif component == "mlp":
        layer.mlp = original
    else:
        raise ValueError(f"未知组件: {component}")


def encode_to_device(model, tokenizer, text: str, *, max_length: int = 128) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return move_batch_to_model_device(model, encoded)
