# -*- coding: utf-8 -*-
"""Debug: 测试 Qwen3 模型加载和 run_with_cache"""
import os
import sys
import time
import traceback as tb

import torch

os.environ['HF_HOME'] = r'D:\develop\model'
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

snapshot_path = r'D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c'

import transformer_lens.utils as tl_utils
import transformers.configuration_utils as config_utils
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # 步骤1: 加载 HF 模型
    hf_model = AutoModelForCausalLM.from_pretrained(
        snapshot_path, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True, add_bos_token=False)
    
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id

    # monkey-patch rope_theta
    _orig = config_utils.PretrainedConfig.__getattribute__
    def _patched(self, key):
        if key == "rope_theta":
            try:
                return _orig(self, key)
            except AttributeError:
                try:
                    rs = _orig(self, "rope_scaling")
                    if isinstance(rs, dict) and "rope_theta" in rs:
                        return rs["rope_theta"]
                except:
                    pass
                return 1000000
        return _orig(self, key)
    config_utils.PretrainedConfig.__getattribute__ = _patched
    
    # monkey-patch get_tokenizer_with_bos
    _orig_tok = tl_utils.get_tokenizer_with_bos
    tl_utils.get_tokenizer_with_bos = lambda tok: tok
    
    try:
        model = HookedTransformer.from_pretrained(
            "Qwen/Qwen3-4B", hf_model=hf_model, device=device, tokenizer=tokenizer,
            fold_ln=False, center_writing_weights=False, center_unembed=False,
            dtype=torch.float16, default_prepend_bos=False
        )
    finally:
        config_utils.PretrainedConfig.__getattribute__ = _orig
        tl_utils.get_tokenizer_with_bos = _orig_tok
    
    model.eval()
    print(f"Model loaded OK, device={next(model.parameters()).device}")
    print(f"n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")
    
    # 测试 run_with_cache
    print("Testing run_with_cache...")
    with torch.no_grad():
        logits, cache = model.run_with_cache("Hello world")
    print(f"run_with_cache OK, logits shape: {logits.shape}")
    
except Exception as e:
    err = tb.format_exc()
    print(err)
    with open('tempdata/qwen3_device_error.txt', 'w', encoding='utf-8') as f:
        f.write(err)
    print("Error saved to tempdata/qwen3_device_error.txt")
