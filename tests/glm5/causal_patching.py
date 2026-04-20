"""
Phase CCVII: Activation Patching 因果干预验证
==============================================

核心目标: 证明相关性 != 因果性, 用activation patching验证因果性!

方案:
  S1: Tense Patching — 把past tense句子的head输出patch到present tense, 看模型输出是否改变
  S2: Polarity Patching — 把肯定句的head输出patch到否定句
  S3: 跨层因果追踪 — 逐层patch tense head, 找到因果关键层
  S4: 因果效应量化 — patch前后logit差异, 量化因果效应大小

原理:
  1. 正常运行: 模型输入 "The cat sat" → 输出 "on" (past context)
  2. Patching: 运行 "The cat sits" (present), 但把某层某head的输出替换为 "sat" 时的
  3. 如果输出变成past context → 该层该head对tense有因果性!

样本: 400对/特征 (适中, 不太慢)
模型: 先测DS7B (7B 8bit, 显存可控), 再测Qwen3和GLM4

运行:
  python tests/glm5/causal_patching.py --model deepseek7b
  python tests/glm5/causal_patching.py --model qwen3
  python tests/glm5/causal_patching.py --model glm4
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import json, gc, time, argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from causal_megasample import (
    generate_polarity_pairs, generate_tense_pairs, generate_semantic_pairs,
    generate_sentiment_pairs, generate_number_pairs
)

FEATURE_GENERATORS = {
    'polarity': generate_polarity_pairs,
    'tense': generate_tense_pairs,
    'semantic': generate_semantic_pairs,
    'sentiment': generate_sentiment_pairs,
    'number': generate_number_pairs,
}
FNAMES = list(FEATURE_GENERATORS.keys())

N_PAIRS = 400  # 每特征400对


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    PATHS = {
        "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    path = PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if model_name in ["glm4", "deepseek7b"]:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb_config, device_map="auto",
                                                      trust_remote_code=True, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cpu',
                                                      trust_remote_code=True, local_files_only=True)
        model = model.to('cuda')
    model.eval()
    device = next(model.parameters()).device
    cfg = model.config
    info = {
        'd_model': cfg.hidden_size, 'n_layers': cfg.num_hidden_layers,
        'n_heads': cfg.num_attention_heads, 'd_head': cfg.hidden_size // cfg.num_attention_heads,
        'n_kv_heads': getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads),
        'device': device
    }
    return model, tokenizer, info


def get_next_token_logit(model, tokenizer, text, device):
    """获取模型对输入文本的下一个token的logits"""
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        out = model(**toks)
    last_logit = out.logits[0, -1, :]  # [vocab_size]
    return last_logit


def patch_head_and_get_logit(model, tokenizer, text_source, text_target, layer_idx, head_idx, info):
    """
    Activation Patching: 
    1. 运行source文本, 捕获指定层指定head的输出
    2. 运行target文本, 在指定层替换该head的输出为source的
    3. 返回patched后的logits
    """
    d_model = info['d_model']
    n_heads = info['n_heads']
    d_head = info['d_head']
    device = info['device']
    
    # Step 1: 获取source的head输出
    source_head_output = [None]
    
    def source_hook(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            h = input[0].detach()  # [batch, seq, n_heads*d_head]
            # 保存指定head的输出
            s, e = head_idx * d_head, (head_idx + 1) * d_head
            source_head_output[0] = h[0, -1, s:e].clone()  # [d_head]
    
    o_proj = model.model.layers[layer_idx].self_attn.o_proj
    if o_proj.weight.is_meta:
        return None
    
    handle = o_proj.register_forward_hook(source_hook)
    _ = get_next_token_logit(model, tokenizer, text_source, device)
    handle.remove()
    
    if source_head_output[0] is None:
        return None
    
    # Step 2: 运行target, patch指定head
    patched_logit = [None]
    
    def target_hook(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            h = input[0]  # [batch, seq, n_heads*d_head]
            s, e = head_idx * d_head, (head_idx + 1) * d_head
            # 替换target的head输出为source的
            h[0, -1, s:e] = source_head_output[0].to(h.device)
    
    handle = o_proj.register_forward_hook(target_hook)
    patched_logit[0] = get_next_token_logit(model, tokenizer, text_target, device)
    handle.remove()
    
    return patched_logit[0]


def compute_causal_effect(clean_logit, patched_logit, tokenizer, feature_name):
    """
    计算因果效应:
    - logit_diff: patched和clean的logit差异 (L2距离)
    - top_token_change: top-1 token是否改变
    - feature_logit_diff: 与特征相关的token的logit变化
    """
    diff = (patched_logit - clean_logit).float()
    l2_dist = float(torch.norm(diff))
    cos_sim = float(torch.nn.functional.cosine_similarity(
        clean_logit.float().unsqueeze(0), patched_logit.float().unsqueeze(0)))
    
    # Top-5 token变化
    clean_top5 = torch.argsort(clean_logit, descending=True)[:5]
    patched_top5 = torch.argsort(patched_logit, descending=True)[:5]
    top1_change = int(clean_top5[0] != patched_top5[0])
    
    # Top-5 overlap
    overlap = len(set(clean_top5.tolist()) & set(patched_top5.tolist()))
    
    # 特征相关的logit变化
    # tense: 过去vs现在 -> "was"/"were"/"had" vs "is"/"are"/"has"
    # polarity: 肯定vs否定 -> "not"/"no"/"never" vs "yes"/"always"/"very"
    feature_tokens = {
        'tense': {
            'past': ['was', 'were', 'had', 'did', 'went', 'came', 'made', 'took'],
            'present': ['is', 'are', 'has', 'does', 'goes', 'comes', 'makes', 'takes']
        },
        'polarity': {
            'positive': ['yes', 'very', 'always', 'great', 'good', 'true', 'right'],
            'negative': ['not', 'no', 'never', 'bad', 'wrong', 'false', 'nothing']
        },
        'number': {
            'singular': ['is', 'was', 'has', 'does', 'it', 'he', 'she'],
            'plural': ['are', 'were', 'have', 'do', 'they', 'we', 'these']
        },
        'sentiment': {
            'positive': ['happy', 'good', 'great', 'love', 'wonderful', 'excellent'],
            'negative': ['sad', 'bad', 'terrible', 'hate', 'awful', 'horrible']
        },
        'semantic': {
            'A': ['animal', 'cat', 'dog', 'bird', 'fish', 'horse'],
            'B': ['city', 'place', 'building', 'road', 'street', 'town']
        }
    }
    
    feature_logit_change = {}
    if feature_name in feature_tokens:
        for sub_feat, tokens in feature_tokens[feature_name].items():
            token_ids = []
            for t in tokens:
                ids = tokenizer.encode(t, add_special_tokens=False)
                token_ids.extend(ids)
            if token_ids:
                clean_sum = float(clean_logit[token_ids].sum())
                patched_sum = float(patched_logit[token_ids].sum())
                feature_logit_change[sub_feat] = {
                    'clean': clean_sum,
                    'patched': patched_sum,
                    'diff': patched_sum - clean_sum
                }
    
    return {
        'l2_dist': l2_dist,
        'cos_sim': cos_sim,
        'top1_change': top1_change,
        'top5_overlap': overlap,
        'feature_logit_change': feature_logit_change
    }


def test_s1_single_layer_patching(model, tokenizer, info):
    """S1: 单层head patching — 找到因果关键head"""
    print("\n" + "="*70)
    print("S1: 单层Head Patching (因果关键head)")
    print("="*70)
    
    n_layers = info['n_layers']
    n_heads = info['n_heads']
    device = info['device']
    
    # 采样层: 末8层 + 每8层采样
    sample_layers = sorted(set(
        list(range(max(0, n_layers - 8), n_layers)) +
        list(range(0, n_layers, max(1, n_layers // 8)))
    ))
    
    results = {}
    
    for fname in ['tense', 'polarity', 'number']:
        print(f"\n  Feature: {fname}")
        pairs = FEATURE_GENERATORS[fname]()[:N_PAIRS]
        
        layer_results = {}
        
        for li in sample_layers:
            o_proj = model.model.layers[li].self_attn.o_proj
            if o_proj.weight.is_meta:
                continue
            
            head_effects = {}
            
            for h_idx in range(n_heads):
                effects = []
                
                for text_a, text_b in pairs[:100]:  # 先用100对做快速扫描
                    try:
                        # Clean: text_b (e.g., present tense)
                        clean_logit = get_next_token_logit(model, tokenizer, text_b, device)
                        
                        # Patched: 把text_a (e.g., past tense)的head输出patch到text_b
                        patched_logit = patch_head_and_get_logit(
                            model, tokenizer, text_a, text_b, li, h_idx, info)
                        
                        if patched_logit is None:
                            continue
                        
                        effect = compute_causal_effect(clean_logit, patched_logit, tokenizer, fname)
                        effects.append(effect)
                    except:
                        continue
                
                if effects:
                    avg_l2 = np.mean([e['l2_dist'] for e in effects])
                    avg_cos = np.mean([e['cos_sim'] for e in effects])
                    avg_top1_change = np.mean([e['top1_change'] for e in effects])
                    
                    # 特征相关logit变化
                    feat_changes = defaultdict(list)
                    for e in effects:
                        for sub_feat, vals in e['feature_logit_change'].items():
                            feat_changes[sub_feat].append(vals['diff'])
                    
                    feat_avg = {k: float(np.mean(v)) for k, v in feat_changes.items()}
                    
                    head_effects[f'h{h_idx}'] = {
                        'avg_l2': avg_l2,
                        'avg_cos_sim': avg_cos,
                        'avg_top1_change': avg_top1_change,
                        'n_pairs': len(effects),
                        'feature_logit_diff': feat_avg
                    }
            
            if head_effects:
                # 按因果效应排序 (l2距离越大 = 因果效应越强)
                sorted_heads = sorted(head_effects.items(), key=lambda x: x[1]['avg_l2'], reverse=True)
                
                print(f"    L{li}: top5 heads by causal effect (l2_dist):")
                for h_name, hd in sorted_heads[:5]:
                    feat_str = ', '.join(f"{k}={v:.2f}" for k, v in sorted(hd['feature_logit_diff'].items(), key=lambda x: -abs(x[1]))[:3])
                    print(f"      {h_name}: l2={hd['avg_l2']:.4f}, cos={hd['avg_cos_sim']:.6f}, top1_chg={hd['avg_top1_change']:.3f} | {feat_str}")
                
                layer_results[f'L{li}'] = {
                    'sorted_heads': [(n, d['avg_l2'], d['avg_cos_sim'], d['avg_top1_change']) for n, d in sorted_heads[:10]],
                    'all_heads': head_effects
                }
        
        results[fname] = layer_results
    
    return results


def test_s2_cross_layer_patching(model, tokenizer, info):
    """S2: 跨层因果追踪 — 逐层patch top head, 量化因果效应"""
    print("\n" + "="*70)
    print("S2: 跨层因果追踪")
    print("="*70)
    
    n_layers = info['n_layers']
    n_heads = info['n_heads']
    device = info['device']
    
    results = {}
    
    for fname in ['tense', 'polarity']:
        print(f"\n  Feature: {fname}")
        pairs = FEATURE_GENERATORS[fname]()[:N_PAIRS]
        
        layer_effects = {}
        
        for li in range(n_layers):
            o_proj = model.model.layers[li].self_attn.o_proj
            if o_proj.weight.is_meta:
                layer_effects[f'L{li}'] = {'error': 'meta_device'}
                continue
            
            # 对该层所有head做patching, 找最强因果head
            best_l2 = 0
            best_head = None
            best_effects = None
            
            for h_idx in range(n_heads):
                effects = []
                
                for text_a, text_b in pairs[:50]:  # 快速扫描, 50对
                    try:
                        clean_logit = get_next_token_logit(model, tokenizer, text_b, device)
                        patched_logit = patch_head_and_get_logit(
                            model, tokenizer, text_a, text_b, li, h_idx, info)
                        if patched_logit is None:
                            continue
                        effect = compute_causal_effect(clean_logit, patched_logit, tokenizer, fname)
                        effects.append(effect)
                    except:
                        continue
                
                if effects:
                    avg_l2 = np.mean([e['l2_dist'] for e in effects])
                    if avg_l2 > best_l2:
                        best_l2 = avg_l2
                        best_head = h_idx
                        best_effects = effects
            
            if best_effects:
                avg_l2 = np.mean([e['l2_dist'] for e in best_effects])
                avg_cos = np.mean([e['cos_sim'] for e in best_effects])
                avg_top1_change = np.mean([e['top1_change'] for e in best_effects])
                
                feat_changes = defaultdict(list)
                for e in best_effects:
                    for sub_feat, vals in e['feature_logit_change'].items():
                        feat_changes[sub_feat].append(vals['diff'])
                feat_avg = {k: float(np.mean(v)) for k, v in feat_changes.items()}
                
                layer_effects[f'L{li}'] = {
                    'best_head': best_head,
                    'avg_l2': avg_l2,
                    'avg_cos_sim': avg_cos,
                    'avg_top1_change': avg_top1_change,
                    'feature_logit_diff': feat_avg,
                    'n_pairs': len(best_effects)
                }
                
                print(f"    L{li}: best=h{best_head}, l2={avg_l2:.4f}, cos={avg_cos:.6f}, top1_chg={avg_top1_change:.3f}")
            else:
                layer_effects[f'L{li}'] = {'error': 'no_data'}
                print(f"    L{li}: no data")
        
        results[fname] = layer_effects
    
    return results


def test_s3_head_specificity_patching(model, tokenizer, info):
    """S3: Head特异性验证 — tense head是否只影响tense, 不影响其他特征?"""
    print("\n" + "="*70)
    print("S3: Head特异性验证 (cross-feature patching)")
    print("="*70)
    
    n_layers = info['n_layers']
    device = info['device']
    
    results = {}
    
    # 选取5个关键层
    sample_layers = sorted(set(
        [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    ))
    
    for li in sample_layers:
        o_proj = model.model.layers[li].self_attn.o_proj
        if o_proj.weight.is_meta:
            continue
        
        print(f"\n  L{li}:")
        layer_results = {}
        
        # 对每个特征, 找最强head
        for source_fname in ['tense', 'polarity', 'number']:
            source_pairs = FEATURE_GENERATORS[source_fname]()[:50]
            
            # 找最强head
            best_l2 = 0
            best_head = None
            
            for h_idx in range(info['n_heads']):
                effects = []
                for text_a, text_b in source_pairs[:20]:
                    try:
                        clean_logit = get_next_token_logit(model, tokenizer, text_b, device)
                        patched_logit = patch_head_and_get_logit(
                            model, tokenizer, text_a, text_b, li, h_idx, info)
                        if patched_logit is None:
                            continue
                        effects.append(float(torch.norm((patched_logit - clean_logit).float())))
                    except:
                        continue
                if effects:
                    avg_l2 = np.mean(effects)
                    if avg_l2 > best_l2:
                        best_l2 = avg_l2
                        best_head = h_idx
            
            if best_head is None:
                continue
            
            # 用该head做跨特征patching
            cross_effects = {}
            for target_fname in FNAMES:
                target_pairs = FEATURE_GENERATORS[target_fname]()[:30]
                effects = []
                
                for text_a, text_b in target_pairs[:20]:
                    try:
                        # source的head输出
                        clean_logit = get_next_token_logit(model, tokenizer, text_b, device)
                        patched_logit = patch_head_and_get_logit(
                            model, tokenizer, text_a, text_b, li, best_head, info)
                        if patched_logit is None:
                            continue
                        effects.append(float(torch.norm((patched_logit - clean_logit).float())))
                    except:
                        continue
                
                if effects:
                    cross_effects[target_fname] = float(np.mean(effects))
            
            print(f"    {source_fname} best_head=h{best_head}: " + 
                  ', '.join(f"{k}={v:.4f}" for k, v in sorted(cross_effects.items(), key=lambda x: -x[1])))
            
            layer_results[source_fname] = {
                'best_head': best_head,
                'cross_effects': cross_effects
            }
        
        results[f'L{li}'] = layer_results
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'deepseek7b', 'qwen3'])
    parser.add_argument('--test', type=str, default='all', choices=['s1', 's2', 's3', 'all'])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"Loading {model_name}...")
    model, tokenizer, info = load_model(model_name)
    print(f"  d_model={info['d_model']}, n_layers={info['n_layers']}, n_heads={info['n_heads']}, d_head={info['d_head']}")
    
    out_dir = Path(f"results/causal_fiber/{model_name}_patching")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    
    if args.test in ['s1', 'all']:
        s1_results = test_s1_single_layer_patching(model, tokenizer, info)
        with open(out_dir / 's1_single_layer.json', 'w') as f:
            json.dump(s1_results, f, indent=2, default=str)
        print(f"\n  S1 done in {time.time()-t0:.0f}s")
    
    if args.test in ['s2', 'all']:
        s2_results = test_s2_cross_layer_patching(model, tokenizer, info)
        with open(out_dir / 's2_cross_layer.json', 'w') as f:
            json.dump(s2_results, f, indent=2, default=str)
        print(f"\n  S2 done in {time.time()-t0:.0f}s")
    
    if args.test in ['s3', 'all']:
        s3_results = test_s3_head_specificity_patching(model, tokenizer, info)
        with open(out_dir / 's3_specificity.json', 'w') as f:
            json.dump(s3_results, f, indent=2, default=str)
        print(f"\n  S3 done in {time.time()-t0:.0f}s")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"\nDone! Free VRAM: {torch.cuda.mem_get_info(0)[0]/1024**3:.1f}GB")
