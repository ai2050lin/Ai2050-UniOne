# -*- coding: utf-8 -*-
import os
import sys
import traceback as tb

import torch

os.environ['HF_HOME'] = r'D:\develop\model'
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

snapshot_path = r'D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c'

from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    hf_model = AutoModelForCausalLM.from_pretrained(
        snapshot_path, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16, device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    
    # Fix rope_theta
    rs = getattr(hf_model.config, 'rope_scaling', None)
    if rs and isinstance(rs, dict) and 'rope_theta' in rs:
        hf_model.config.rope_theta = rs['rope_theta']
    print(f'rope_theta after fix: {hf_model.config.rope_theta}')
    
    model = HookedTransformer.from_pretrained(
        'qwen2.5-7b', hf_model=hf_model, device='cuda', tokenizer=tokenizer,
        fold_ln=False, center_writing_weights=False, center_unembed=False,
        dtype=torch.float16
    )
    print('SUCCESS')
except Exception as e:
    err_text = tb.format_exc()
    print(err_text)
    with open('tempdata/qwen3_load_error.txt', 'w', encoding='utf-8') as f:
        f.write(err_text)
    print('Error saved to tempdata/qwen3_load_error.txt')
