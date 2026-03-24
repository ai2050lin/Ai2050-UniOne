#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = Path(r"d:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60")


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=True,
        trust_remote_code=True,
    )
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    return model, tokenizer


def run_forward(model, tokenizer, text: str):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True,
        )
    return inputs, outputs


def suffix_span(longer: List[int], suffix: List[int]) -> Tuple[int, int]:
    for start in range(len(longer) - len(suffix) + 1):
        if longer[start : start + len(suffix)] == suffix:
            return start, start + len(suffix)
    raise ValueError("未找到内容后缀跨度")


def layer_band(layer_idx: int, layer_count: int) -> str:
    if layer_idx < layer_count / 3:
        return "early"
    if layer_idx < 2 * layer_count / 3:
        return "mid"
    return "late"


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


class GateCollector:
    def __init__(self, model):
        self.layers = list(model.model.layers)
        self.buffers = [None for _ in self.layers]
        self.handles = []
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(self._hook(li)))

    def _hook(self, layer_idx: int):
        def inner(_module, _inputs, output):
            self.buffers[layer_idx] = output[0, -1, :].detach().float().cpu()
            return output

        return inner

    def reset(self):
        for i in range(len(self.buffers)):
            self.buffers[i] = None

    def get(self) -> List[torch.Tensor]:
        return [buf for buf in self.buffers if buf is not None]

    def close(self):
        for handle in self.handles:
            handle.remove()
