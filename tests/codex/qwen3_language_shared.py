#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QWEN3_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"
)
GLM4_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"
)
GEMMA4_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"
)


def set_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def discover_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return list(model.model.language_model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError("未识别到可用的解码层结构")


def mapped_layer_index(layer_count: int, gpt2_layer_index: int) -> int:
    if layer_count <= 1:
        return 0
    if gpt2_layer_index <= 0:
        return 0
    if gpt2_layer_index >= 11:
        return layer_count - 1
    return max(0, min(layer_count - 1, round((layer_count - 1) * gpt2_layer_index / 11.0)))


def resolve_anchor_layers(model) -> Dict[str, int]:
    layer_count = len(discover_layers(model))
    return {
        "layer_count": layer_count,
        "early_layer": mapped_layer_index(layer_count, 1),
        "route_layer": mapped_layer_index(layer_count, 3),
        "late_layer": mapped_layer_index(layer_count, 11),
    }


def load_qwen3_tokenizer():
    set_offline_env()
    tokenizer = AutoTokenizer.from_pretrained(
        str(QWEN3_MODEL_PATH),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qwen3_model(*, prefer_cuda: bool = True):
    set_offline_env()
    want_cuda = bool(prefer_cuda and torch.cuda.is_available())
    load_kwargs = {
        "pretrained_model_name_or_path": str(QWEN3_MODEL_PATH),
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    if want_cuda:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    return model, load_qwen3_tokenizer()


def load_glm4_tokenizer():
    set_offline_env()
    tokenizer = AutoTokenizer.from_pretrained(
        str(GLM4_MODEL_PATH),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_glm4_model(*, prefer_cuda: bool = True):
    set_offline_env()
    want_cuda = bool(prefer_cuda and torch.cuda.is_available())
    load_kwargs = {
        "pretrained_model_name_or_path": str(GLM4_MODEL_PATH),
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    if want_cuda:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    return model, load_glm4_tokenizer()


def load_gemma4_processor():
    set_offline_env()
    processor = AutoProcessor.from_pretrained(
        str(GEMMA4_MODEL_PATH),
        local_files_only=True,
    )
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def load_gemma4_model(*, prefer_cuda: bool = True):
    set_offline_env()
    want_cuda = bool(prefer_cuda and torch.cuda.is_available())
    load_kwargs = {
        "pretrained_model_name_or_path": str(GEMMA4_MODEL_PATH),
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    if want_cuda:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["attn_implementation"] = "eager"
    model = AutoModelForImageTextToText.from_pretrained(**load_kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    processor = load_gemma4_processor()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return model, tokenizer


def load_qwen3_embedding_weight() -> torch.Tensor:
    for shard_path in sorted(QWEN3_MODEL_PATH.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            if "model.embed_tokens.weight" in handle.keys():
                return handle.get_tensor("model.embed_tokens.weight").detach().cpu()
    raise FileNotFoundError("未在 Qwen3 safetensors 中找到 model.embed_tokens.weight")


def qwen_neuron_dim(model) -> int:
    layer = discover_layers(model)[0]
    if hasattr(layer.mlp, "gate_proj"):
        return int(layer.mlp.gate_proj.out_features)
    if hasattr(layer.mlp, "gate_up_proj"):
        return int(layer.mlp.gate_up_proj.out_features // 2)
    raise RuntimeError("未识别到 Qwen3 MLP 神经元维度")


def qwen_hidden_dim(model) -> int:
    layer = discover_layers(model)[0]
    if hasattr(layer.mlp, "down_proj"):
        return int(layer.mlp.down_proj.out_features)
    raise RuntimeError("未识别到 Qwen3 隐状态维度")


def capture_qwen_mlp_payloads(
    model,
    layer_payload_map: Dict[int, str],
) -> Tuple[Dict[int, torch.Tensor | None], List[object]]:
    layers = discover_layers(model)
    buffers: Dict[int, torch.Tensor | None] = {idx: None for idx in layer_payload_map}
    handles = []

    for layer_idx, payload_kind in layer_payload_map.items():
        layer = layers[layer_idx]
        mlp_probe = layer.mlp.down_proj
        if payload_kind == "neuron_in":
            def make_pre_hook(target_idx: int):
                def hook(_module, inputs):
                    buffers[target_idx] = inputs[0].detach().float().cpu()
                return hook

            handles.append(mlp_probe.register_forward_pre_hook(make_pre_hook(layer_idx)))
        elif payload_kind == "hidden_out":
            def make_post_hook(target_idx: int):
                def hook(_module, _inputs, output):
                    buffers[target_idx] = output.detach().float().cpu()
                return hook

            handles.append(mlp_probe.register_forward_hook(make_post_hook(layer_idx)))
        else:
            raise ValueError(f"未知 payload_kind: {payload_kind}")
    return buffers, handles


def remove_hooks(handles: Sequence[object]) -> None:
    for handle in handles:
        handle.remove()


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B"
)

def _find_deepseek_snapshot():
    base = DEEPSEEK7B_MODEL_PATH
    if not base.exists():
        return None
    snapshots = base / "snapshots"
    if not snapshots.exists():
        return None
    dirs = [d for d in snapshots.iterdir() if d.is_dir()]
    return dirs[0] if dirs else None


def load_deepseek7b_model(*, prefer_cuda: bool = True):
    set_offline_env()
    snapshot = _find_deepseek_snapshot()
    if snapshot is None:
        raise FileNotFoundError("未找到 DeepSeek 模型快照目录")
    print(f"  DeepSeek 模型路径: {snapshot}")
    want_cuda = bool(prefer_cuda and torch.cuda.is_available())
    load_kwargs = {
        "pretrained_model_name_or_path": str(snapshot),
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    if want_cuda:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def move_batch_to_model_device(model, encoded: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return encoded
    return {key: value.to(device) for key, value in encoded.items()}
