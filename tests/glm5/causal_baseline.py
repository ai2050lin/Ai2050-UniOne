"""
Phase CXCVII: 因果基线分析 — 从解码几何转向因果几何
=================================================================

核心洞察 (Phase CXCVI): 
  之前发现的"正交编织""结构不变量"等都是高维线性分类器的一般性质,
  不是语言特有! 需要从"解码空间"转向"因果空间"找真正的语言结构。

核心测试:
  C1: 随机方向1维注入 vs 极性方向1维注入 — 1维注入无效是高维一般还是语言特有?
  C2: 随机patch因果效应 vs 极性patch — 极性的100-300x因果效应是否超过随机?
  C3: 因果路径正交性 — 极性的因果路径 vs 时态的因果路径是否正交?
  C4: 因果效应的层间演化 — 因果效应如何逐层变化? 语言vs随机的差异在哪里?

运行:
  python tests/glm5/causal_baseline.py --model qwen3 --test c1
  python tests/glm5/causal_baseline.py --model qwen3 --test all
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import gc
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 大样本测试集 — 200+对
# ============================================================

def generate_polarity_pairs():
    """生成200对极性对"""
    templates = [
        ("The cat is here", "The cat is not here"),
        ("The dog is happy", "The dog is not happy"),
        ("The house is big", "The house is not big"),
        ("The phone is working", "The phone is not working"),
        ("The bridge is safe", "The bridge is not safe"),
        ("The star is visible", "The star is not visible"),
        ("The door is open", "The door is not open"),
        ("The lake is deep", "The lake is not deep"),
        ("The road is clear", "The road is not clear"),
        ("The wall is strong", "The wall is not strong"),
        ("The bird is flying", "The bird is not flying"),
        ("The fish is swimming", "The fish is not swimming"),
        ("The car is fast", "The car is not fast"),
        ("The tree is tall", "The tree is not tall"),
        ("The river is wide", "The river is not wide"),
        ("The book is interesting", "The book is not interesting"),
        ("The food is fresh", "The food is not fresh"),
        ("The light is bright", "The light is not bright"),
        ("The music is loud", "The music is not loud"),
        ("The weather is cold", "The weather is not cold"),
        ("The door was closed", "The door was not closed"),
        ("The child was playing", "The child was not playing"),
        ("The man was running", "The man was not running"),
        ("The woman was singing", "The woman was not singing"),
        ("The dog was barking", "The dog was not barking"),
        ("The sun was shining", "The sun was not shining"),
        ("The wind was blowing", "The wind was not blowing"),
        ("The rain was falling", "The rain was not falling"),
        ("The snow was melting", "The snow was not melting"),
        ("The fire was burning", "The fire was not burning"),
        ("I like the car", "I do not like the car"),
        ("She knows the answer", "She does not know the answer"),
        ("The river flows north", "The river does not flow north"),
        ("I understand the plan", "I do not understand the plan"),
        ("She likes the movie", "She does not like the movie"),
        ("The key works well", "The key does not work well"),
        ("The food tastes good", "The food does not taste good"),
        ("He drives the truck", "He does not drive the truck"),
        ("The system runs smoothly", "The system does not run smoothly"),
        ("The machine operates well", "The machine does not operate well"),
        ("He can swim", "He cannot swim"),
        ("The bird will come", "The bird will not come"),
        ("She can help", "She cannot help"),
        ("They will agree", "They will not agree"),
        ("He can solve it", "He cannot solve it"),
        ("The team will win", "The team will not win"),
        ("She can finish it", "She cannot finish it"),
        ("They will succeed", "They will not succeed"),
        ("The cloud disappeared", "The cloud did not disappear"),
        ("He arrived early", "He did not arrive early"),
        ("She passed the test", "She did not pass the test"),
        ("They finished the work", "They did not finish the work"),
        ("The student graduated", "The student did not graduate"),
        ("He answered correctly", "He did not answer correctly"),
        ("She returned home", "She did not return home"),
        ("The flower has bloomed", "The flower has not bloomed"),
        ("He has finished", "He has not finished"),
        ("She has arrived", "She has not arrived"),
        ("The tree has grown", "The tree has not grown"),
        ("The project has started", "The project has not started"),
        ("The water is clean", "The water is not clean"),
        ("The machine works", "The machine does not work"),
        ("He can see", "He cannot see"),
        ("The plan was approved", "The plan was not approved"),
        ("She will come", "She will not come"),
        ("The road is open", "The road is not open"),
        ("The system is stable", "The system is not stable"),
        ("I trust the result", "I do not trust the result"),
        ("The door is locked", "The door is not locked"),
        ("The glass is empty", "The glass is not empty"),
        ("The room is dark", "The room is not dark"),
        ("The sky is blue", "The sky is not blue"),
        ("The grass is green", "The grass is not green"),
        ("The coffee is hot", "The coffee is not hot"),
        ("The ice is thick", "The ice is not thick"),
        ("The path is narrow", "The path is not narrow"),
        ("The cake is sweet", "The cake is not sweet"),
        ("The movie is long", "The movie is not long"),
        ("The test is hard", "The test is not hard"),
        ("The game is fair", "The game is not fair"),
        ("The price is low", "The price is not low"),
        ("The quality is high", "The quality is not high"),
        ("The speed is slow", "The speed is not slow"),
        ("The sound is clear", "The sound is not clear"),
        ("The color is bright", "The color is not bright"),
        ("The shape is round", "The shape is not round"),
        ("The size is small", "The size is not small"),
        ("The weight is heavy", "The weight is not heavy"),
        ("The distance is short", "The distance is not short"),
        ("The temperature is warm", "The temperature is not warm"),
        ("The apple is ripe", "The apple is not ripe"),
        ("The ocean is calm", "The ocean is not calm"),
        ("The mountain is high", "The mountain is not high"),
        ("The forest is dense", "The forest is not dense"),
        ("The desert is dry", "The desert is not dry"),
        ("The city is noisy", "The city is not noisy"),
        ("The village is quiet", "The village is not quiet"),
        ("The boy is tall", "The boy is not tall"),
        ("The girl is short", "The girl is not short"),
        ("The man is strong", "The man is not strong"),
        ("The woman is kind", "The woman is not kind"),
        ("The cat is black", "The cat is not black"),
        ("The dog is brown", "The dog is not brown"),
        ("The flower is red", "The flower is not red"),
        ("The leaf is green", "The leaf is not green"),
        ("The stone is hard", "The stone is not hard"),
        ("The feather is soft", "The feather is not soft"),
        ("The metal is shiny", "The metal is not shiny"),
        ("The wood is rough", "The wood is not rough"),
        ("The glass is smooth", "The glass is not smooth"),
        ("He is running fast", "He is not running fast"),
        ("She is reading quietly", "She is not reading quietly"),
        ("They are sleeping peacefully", "They are not sleeping peacefully"),
        ("We are learning quickly", "We are not learning quickly"),
        ("The children are playing", "The children are not playing"),
        ("The birds are singing", "The birds are not singing"),
        ("The students are studying", "The students are not studying"),
        ("The workers are building", "The workers are not building"),
        ("The doctors are helping", "The doctors are not helping"),
        ("The scientists are researching", "The scientists are not researching"),
        ("This is correct", "This is not correct"),
        ("That is true", "That is not true"),
        ("It is real", "It is not real"),
        ("The answer is right", "The answer is not right"),
        ("The method is effective", "The method is not effective"),
        ("The solution is simple", "The solution is not simple"),
        ("The problem is easy", "The problem is not easy"),
        ("The task is difficult", "The task is not difficult"),
        ("The process is fast", "The process is not fast"),
        ("The result is accurate", "The result is not accurate"),
        ("The building is tall", "The building is not tall"),
        ("The river is long", "The river is not long"),
        ("The park is beautiful", "The park is not beautiful"),
        ("The market is busy", "The market is not busy"),
        ("The station is crowded", "The station is not crowded"),
        ("The airport is modern", "The airport is not modern"),
        ("The library is old", "The library is not old"),
        ("The museum is large", "The museum is not large"),
        ("The theater is famous", "The theater is not famous"),
        ("The restaurant is popular", "The restaurant is not popular"),
        ("The engine is powerful", "The engine is not powerful"),
        ("The battery is full", "The battery is not full"),
        ("The signal is strong", "The signal is not strong"),
        ("The connection is stable", "The connection is not stable"),
        ("The software is updated", "The software is not updated"),
        ("The hardware is compatible", "The hardware is not compatible"),
        ("The network is secure", "The network is not secure"),
        ("The system is reliable", "The system is not reliable"),
        ("The device is functional", "The device is not functional"),
        ("The screen is bright", "The screen is not bright"),
    ]
    return templates[:200]


def generate_tense_pairs():
    """生成200对时态对"""
    verbs = [
        ("runs", "ran"), ("walks", "walked"), ("plays", "played"),
        ("sings", "sang"), ("eats", "ate"), ("drinks", "drank"),
        ("writes", "wrote"), ("reads", "read"), ("speaks", "spoke"),
        ("drives", "drove"), ("swims", "swam"), ("flies", "flew"),
        ("grows", "grew"), ("knows", "knew"), ("thinks", "thought"),
        ("brings", "brought"), ("builds", "built"), ("catches", "caught"),
        ("teaches", "taught"), ("feels", "felt"), ("finds", "found"),
        ("gives", "gave"), ("holds", "held"), ("keeps", "kept"),
        ("leaves", "left"), ("loses", "lost"), ("meets", "met"),
        ("pays", "paid"), ("sells", "sold"), ("sends", "sent"),
        ("shuts", "shut"), ("sits", "sat"), ("sleeps", "slept"),
        ("spends", "spent"), ("stands", "stood"), ("takes", "took"),
        ("tells", "told"), ("understands", "understood"), ("wears", "wore"),
        ("wins", "won"), ("begins", "began"), ("breaks", "broke"),
        ("chooses", "chose"), ("draws", "drew"), ("falls", "fell"),
        ("forgets", "forgot"), ("gets", "got"), ("hangs", "hung"),
        ("hides", "hid"), ("hurts", "hurt"),
    ]
    subjects = [
        "The cat", "The dog", "He", "She", "The bird", "The man",
        "The woman", "The child", "The teacher", "The student",
        "The river", "The wind", "The fire", "The rain", "The sun",
        "I", "We", "They", "The team", "The group",
        "The boy", "The girl", "The king", "The queen", "The doctor",
        "The farmer", "The driver", "The artist", "The writer", "The singer",
        "The player", "The runner", "The worker", "The leader", "The speaker",
        "The nurse", "The chef", "The guard", "The pilot", "The judge",
    ]
    pairs = []
    for i, subj in enumerate(subjects):
        for pres, past in verbs[i % len(verbs):i % len(verbs) + 5]:
            pairs.append((f"{subj} {pres} every day", f"{subj} {past} yesterday"))
            if len(pairs) >= 200:
                return pairs[:200]
    return pairs[:200]


# ============================================================
# 模型加载
# ============================================================

def load_model_fast(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    PATHS = {
        "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    
    path = PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if model_name in ["glm4", "deepseek7b"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map='cpu',
            trust_remote_code=True, local_files_only=True,
        )
        model = model.to('cuda')
    
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


# ============================================================
# 核心工具: 提取残差流 + 干预
# ============================================================

def extract_residual_stream(model, tokenizer, device, texts, layers_to_probe, pool_method='last'):
    """批量提取残差流表示"""
    captured = {}
    hooks = []
    layers = model.model.layers
    
    for li in layers_to_probe:
        captured[li] = []
    
    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0].detach().cpu().float()
            else:
                h = output.detach().cpu().float()
            captured[li].append(h[0])
        return hook_fn
    
    for li in layers_to_probe:
        hooks.append(layers[li].register_forward_hook(make_hook(li)))
    
    results = {li: [] for li in layers_to_probe}
    
    for text in texts:
        for li in layers_to_probe:
            captured[li] = []
        
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        
        with torch.no_grad():
            try:
                _ = model(**toks)
            except Exception as e:
                for li in layers_to_probe:
                    results[li].append(np.zeros(model.config.hidden_size))
                continue
        
        for li in layers_to_probe:
            if captured[li]:
                h = captured[li][0].numpy()
                if pool_method == 'last':
                    results[li].append(h[-1])
                elif pool_method == 'mean':
                    results[li].append(h.mean(axis=0))
                else:
                    results[li].append(h[-1])
            else:
                results[li].append(np.zeros(model.config.hidden_size))
    
    for h in hooks:
        h.remove()
    
    for li in layers_to_probe:
        results[li] = np.array(results[li])
    
    return results


def get_logit_diff(model, tokenizer, device, prompt, pos_token=" not", neg_token=" very"):
    """
    计算logit差异: log P(pos_token) - log P(neg_token)
    用作极性读出指标
    """
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        logits = model(**toks).logits[0, -1].detach().cpu().float().numpy()
    
    pos_id = tokenizer.encode(pos_token, add_special_tokens=False)[0]
    neg_id = tokenizer.encode(neg_token, add_special_tokens=False)[0]
    
    return float(logits[pos_id] - logits[neg_id])


def compute_embedding_injection_effect(model, tokenizer, device, prompt, direction, beta, 
                                        pos_token=" not", neg_token=" very"):
    """
    计算在embedding层注入方向的因果效应
    
    Returns:
        delta_logit: 注入后logit_diff的变化
        effect_norm: logits向量的L2范数变化
    """
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
    input_ids = toks.input_ids
    
    # 基线
    with torch.no_grad():
        base_logits = model(**toks).logits[0, -1].detach().cpu().float().numpy()
    
    pos_id = tokenizer.encode(pos_token, add_special_tokens=False)[0]
    neg_id = tokenizer.encode(neg_token, add_special_tokens=False)[0]
    base_logit_diff = float(base_logits[pos_id] - base_logits[neg_id])
    
    # 注入
    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(input_ids).detach().clone()
    dir_tensor = torch.tensor(beta * direction, dtype=embeds.dtype, device=device)
    embeds_intervened = embeds.clone()
    embeds_intervened[0, -1, :] += dir_tensor.to(embeds.dtype)
    
    pos_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    
    with torch.no_grad():
        try:
            interv_logits = model(inputs_embeds=embeds_intervened, position_ids=pos_ids).logits[0, -1].detach().cpu().float().numpy()
        except:
            interv_logits = base_logits.copy()
    
    interv_logit_diff = float(interv_logits[pos_id] - interv_logits[neg_id])
    
    delta_logit = interv_logit_diff - base_logit_diff
    effect_norm = float(np.linalg.norm(interv_logits - base_logits))
    
    return delta_logit, effect_norm


def compute_residual_patch_effect(model, tokenizer, device, source_text, target_text, 
                                   layer, pos_token=" not", neg_token=" very"):
    """
    计算残差流修补的因果效应
    
    将source_text在layer的残差流修补到target_text的对应位置
    
    Returns:
        delta_logit: 修补后logit_diff的变化
    """
    # 获取source在layer的残差流
    source_repr = extract_residual_stream(model, tokenizer, device, [source_text], [layer], 'last')
    source_h = source_repr[layer][0]  # [d_model]
    
    # 基线: target的logit
    base_logit_diff = get_logit_diff(model, tokenizer, device, target_text, pos_token, neg_token)
    
    # 修补: 在target的前向中, 将layer的输出替换为source的
    captured = {}
    hooks = []
    layers = model.model.layers
    
    def make_patch_hook(li, patch_vec):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                # 在last token位置修补
                h[0, -1, :] = torch.tensor(patch_vec, dtype=h.dtype, device=h.device)
                return (h,) + output[1:]
            else:
                h = output.clone()
                h[0, -1, :] = torch.tensor(patch_vec, dtype=h.dtype, device=h.device)
                return h
        return hook_fn
    
    # 注册修补hook
    hooks.append(layers[layer].register_forward_hook(make_patch_hook(layer, source_h)))
    
    # 前向
    toks = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        try:
            patched_logits = model(**toks).logits[0, -1].detach().cpu().float().numpy()
        except:
            patched_logits = np.zeros_like(base_logits) if 'base_logits' in dir() else np.zeros(1000)
    
    for h in hooks:
        h.remove()
    
    pos_id = tokenizer.encode(pos_token, add_special_tokens=False)[0]
    neg_id = tokenizer.encode(neg_token, add_special_tokens=False)[0]
    patched_logit_diff = float(patched_logits[pos_id] - patched_logits[neg_id])
    
    return patched_logit_diff - base_logit_diff


# ============================================================
# C1: 随机方向1维注入 vs 极性方向1维注入
# ============================================================

def test_c1(model_name, model, tokenizer, device, d_model, n_layers):
    """
    C1: 1维注入的因果效应 — 极性方向 vs 随机方向
    
    已知: 极性1维注入无效(slope<0.05)
    问题: 随机方向的1维注入是否也无效?
    """
    
    print("\n" + "=" * 70)
    print("C1: 1维注入因果效应 — 极性方向 vs 随机方向")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()
    
    # 提取极性差分方向(在中间层)
    target_layer = n_layers // 2
    aff_texts = [aff for aff, neg in pol_pairs[:50]]
    neg_texts = [neg for aff, neg in pol_pairs[:50]]
    all_texts = aff_texts + neg_texts
    
    print(f"  提取极性差分方向 (L{target_layer})...")
    repr_dict = extract_residual_stream(model, tokenizer, device, all_texts, [target_layer], 'last')
    X = repr_dict[target_layer]
    
    n_half = len(aff_texts)
    mean_aff = X[:n_half].mean(axis=0)
    mean_neg = X[n_half:].mean(axis=0)
    polarity_dir = mean_neg - mean_aff
    polarity_norm = np.linalg.norm(polarity_dir)
    if polarity_norm > 1e-10:
        polarity_dir = polarity_dir / polarity_norm
    
    # 也提取probe方向
    labels = np.array([0] * n_half + [1] * n_half)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf.fit(X_scaled, labels)
    probe_dir = clf.coef_[0]
    probe_norm = np.linalg.norm(probe_dir)
    if probe_norm > 1e-10:
        probe_dir = probe_dir / probe_norm
    
    # 生成随机方向
    n_random_dirs = 30
    np.random.seed(42)
    random_dirs = []
    for _ in range(n_random_dirs):
        rd = np.random.randn(d_model)
        rd = rd / np.linalg.norm(rd)
        random_dirs.append(rd)
    
    # 测试模板 (大样本!)
    test_prompts = []
    for aff, neg in pol_pairs[50:100]:  # 50个测试提示
        # 用肯定句作为prompt, 在最后token注入方向
        words = aff.split()
        if len(words) >= 3:
            prompt = " ".join(words[:-1])  # 去掉最后一个词
        else:
            prompt = aff
        test_prompts.append(prompt)
    
    print(f"  测试提示数: {len(test_prompts)}")
    print(f"  随机方向数: {n_random_dirs}")
    
    betas = [1.0, 2.0, 4.0, 8.0, 16.0]
    
    results = {
        'target_layer': target_layer,
        'n_test_prompts': len(test_prompts),
        'n_random_dirs': n_random_dirs,
        'betas': betas,
        'polarity_injection': {},
        'probe_injection': {},
        'random_injection': {},
        'comparison': {},
    }
    
    for beta in betas:
        print(f"\n  --- beta={beta} ---")
        
        # 极性差分方向注入
        pol_delta_logits = []
        pol_effect_norms = []
        for prompt in test_prompts:
            try:
                dl, en = compute_embedding_injection_effect(
                    model, tokenizer, device, prompt, polarity_dir, beta
                )
                pol_delta_logits.append(dl)
                pol_effect_norms.append(en)
            except:
                pass
        
        # Probe方向注入
        probe_delta_logits = []
        probe_effect_norms = []
        for prompt in test_prompts:
            try:
                dl, en = compute_embedding_injection_effect(
                    model, tokenizer, device, prompt, probe_dir, beta
                )
                probe_delta_logits.append(dl)
                probe_effect_norms.append(en)
            except:
                pass
        
        # 随机方向注入
        rand_delta_logits_all = []
        rand_effect_norms_all = []
        for rd in random_dirs:
            rd_delta_logits = []
            rd_effect_norms = []
            for prompt in test_prompts:
                try:
                    dl, en = compute_embedding_injection_effect(
                        model, tokenizer, device, prompt, rd, beta
                    )
                    rd_delta_logits.append(dl)
                    rd_effect_norms.append(en)
                except:
                    pass
            if rd_delta_logits:
                rand_delta_logits_all.append(np.mean(rd_delta_logits))
                rand_effect_norms_all.append(np.mean(rd_effect_norms))
        
        # 汇总
        pol_mean_dl = float(np.mean(pol_delta_logits)) if pol_delta_logits else 0
        pol_std_dl = float(np.std(pol_delta_logits)) if pol_delta_logits else 0
        pol_mean_en = float(np.mean(pol_effect_norms)) if pol_effect_norms else 0
        
        probe_mean_dl = float(np.mean(probe_delta_logits)) if probe_delta_logits else 0
        probe_std_dl = float(np.std(probe_delta_logits)) if probe_delta_logits else 0
        probe_mean_en = float(np.mean(probe_effect_norms)) if probe_effect_norms else 0
        
        rand_mean_dl = float(np.mean(rand_delta_logits_all)) if rand_delta_logits_all else 0
        rand_std_dl = float(np.std(rand_delta_logits_all)) if rand_delta_logits_all else 0
        rand_mean_en = float(np.mean(rand_effect_norms_all)) if rand_effect_norms_all else 0
        rand_std_en = float(np.std(rand_effect_norms_all)) if rand_effect_norms_all else 0
        
        results['polarity_injection'][beta] = {
            'delta_logit_mean': pol_mean_dl,
            'delta_logit_std': pol_std_dl,
            'effect_norm_mean': pol_mean_en,
        }
        results['probe_injection'][beta] = {
            'delta_logit_mean': probe_mean_dl,
            'delta_logit_std': probe_std_dl,
            'effect_norm_mean': probe_mean_en,
        }
        results['random_injection'][beta] = {
            'delta_logit_mean': rand_mean_dl,
            'delta_logit_std': rand_std_dl,
            'effect_norm_mean': rand_mean_en,
            'effect_norm_std': rand_std_en,
        }
        
        # 关键比较: 极性vs随机的delta_logit比率
        dl_ratio = abs(pol_mean_dl) / max(abs(rand_mean_dl), 1e-10)
        en_ratio = pol_mean_en / max(rand_mean_en, 1e-10)
        
        results['comparison'][beta] = {
            'delta_logit_ratio': dl_ratio,
            'effect_norm_ratio': en_ratio,
            'polarity_vs_random_z': float((pol_mean_dl - rand_mean_dl) / max(rand_std_dl, 1e-6)),
        }
        
        print(f"    Polarity: delta_logit={pol_mean_dl:+.4f}±{pol_std_dl:.4f}, effect_norm={pol_mean_en:.4f}")
        print(f"    Probe:    delta_logit={probe_mean_dl:+.4f}±{probe_std_dl:.4f}, effect_norm={probe_mean_en:.4f}")
        print(f"    Random:   delta_logit={rand_mean_dl:+.4f}±{rand_std_dl:.4f}, effect_norm={rand_mean_en:.4f}±{rand_std_en:.4f}")
        print(f"    Ratio:    delta_logit={dl_ratio:.2f}x, effect_norm={en_ratio:.2f}x", end="")
        
        if dl_ratio > 2.0:
            print(" → 极性方向比随机显著更强! 1维注入虽弱但有特异性")
        elif dl_ratio > 1.0:
            print(" → 极性方向略强于随机")
        else:
            print(" → 1维注入无效是高维一般性质")
    
    # === 汇总 ===
    print("\n" + "=" * 70)
    print("C1 汇总: 1维注入无效是语言特有还是高维一般?")
    print("=" * 70)
    
    for beta in betas:
        comp = results['comparison'][beta]
        print(f"  beta={beta:5.1f}: pol/rand delta_logit ratio={comp['delta_logit_ratio']:.2f}x, "
              f"z={comp['polarity_vs_random_z']:.1f}")
    
    # 综合判断
    high_beta_ratios = [results['comparison'][b]['delta_logit_ratio'] for b in [8.0, 16.0] if b in results['comparison']]
    if high_beta_ratios:
        mean_ratio = np.mean(high_beta_ratios)
        print(f"\n  高beta(8,16)平均比率: {mean_ratio:.2f}x", end="")
        if mean_ratio > 2.0:
            print(" → 极性1维注入虽弱但有语言特异性!")
        elif mean_ratio > 1.0:
            print(" → 极性1维注入略强于随机, 但差异不大")
        else:
            print(" → 1维注入无效是高维一般性质")
    
    return results


# ============================================================
# C2: 随机patch因果效应 vs 极性patch
# ============================================================

def test_c2(model_name, model, tokenizer, device, d_model, n_layers):
    """
    C2: 残差流修补的因果效应 — 极性patch vs 随机patch
    
    已知: 极性patch有100-300x因果效应
    问题: 随机patch是否也有这么大效应?
    """
    
    print("\n" + "=" * 70)
    print("C2: 残差流修补因果效应 — 极性patch vs 随机patch")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()[:50]
    
    # 采样层
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    key_layers = sorted(set(key_layers))
    
    n_patches = 20  # 20对patch测试
    
    results = {
        'key_layers': key_layers,
        'n_patches': n_patches,
        'polarity_patch': {},
        'random_patch': {},
        'comparison': {},
    }
    
    for li in key_layers:
        print(f"\n  --- L{li} ---")
        
        pol_deltas = []
        rand_deltas = []
        
        for i in range(n_patches):
            aff, neg = pol_pairs[i]
            
            # 极性patch: neg→aff (把否定句的残差流修补到肯定句)
            try:
                pol_delta = compute_residual_patch_effect(
                    model, tokenizer, device, neg, aff, li
                )
                pol_deltas.append(pol_delta)
            except:
                pol_deltas.append(0)
            
            # 随机patch: 随机选一个句子作为source
            rand_idx = np.random.randint(0, len(pol_pairs))
            rand_source = pol_pairs[rand_idx][np.random.randint(0, 2)]
            try:
                rand_delta = compute_residual_patch_effect(
                    model, tokenizer, device, rand_source, aff, li
                )
                rand_deltas.append(rand_delta)
            except:
                rand_deltas.append(0)
        
        pol_mean = float(np.mean(pol_deltas))
        pol_std = float(np.std(pol_deltas))
        rand_mean = float(np.mean(rand_deltas))
        rand_std = float(np.std(rand_deltas))
        
        ratio = abs(pol_mean) / max(abs(rand_mean), 1e-10)
        
        results['polarity_patch'][li] = {
            'mean': pol_mean,
            'std': pol_std,
            'abs_mean': float(np.mean(np.abs(pol_deltas))),
        }
        results['random_patch'][li] = {
            'mean': rand_mean,
            'std': rand_std,
            'abs_mean': float(np.mean(np.abs(rand_deltas))),
        }
        results['comparison'][li] = {
            'ratio': ratio,
            'z_score': float((pol_mean - rand_mean) / max(rand_std, 1e-6)),
        }
        
        print(f"    Polarity patch: delta={pol_mean:+.4f}±{pol_std:.4f}, |delta|={np.mean(np.abs(pol_deltas)):.4f}")
        print(f"    Random patch:   delta={rand_mean:+.4f}±{rand_std:.4f}, |delta|={np.mean(np.abs(rand_deltas)):.4f}")
        print(f"    Ratio: {ratio:.2f}x", end="")
        
        if ratio > 3.0:
            print(" → 极性patch比随机显著更强!")
        elif ratio > 1.5:
            print(" → 极性patch比随机略强")
        else:
            print(" → patch效应无显著差异")
    
    # === 汇总 ===
    print("\n" + "=" * 70)
    print("C2 汇总: 残差流修补的因果效应是语言特有?")
    print("=" * 70)
    
    for li in key_layers:
        comp = results['comparison'][li]
        pol = results['polarity_patch'][li]
        rand = results['random_patch'][li]
        print(f"  L{li}: pol={pol['mean']:+.4f}, rand={rand['mean']:+.4f}, ratio={comp['ratio']:.2f}x")
    
    return results


# ============================================================
# C3: 因果路径正交性 — 极性vs时态的因果路径
# ============================================================

def test_c3(model_name, model, tokenizer, device, d_model, n_layers):
    """
    C3: 因果路径正交性 — 不同语言特征的因果路径是否正交?
    
    方法: 
    - 计算极性的因果方向(1维注入效应最大的方向)
    - 计算时态的因果方向
    - 比较两个因果方向的正交性
    
    这是"因果几何"而非"解码几何"!
    """
    
    print("\n" + "=" * 70)
    print("C3: 因果路径正交性 — 极性vs时态的因果方向")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()[:50]
    tense_pairs = generate_tense_pairs()[:50]
    
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    key_layers = sorted(set(key_layers))
    
    results = {
        'key_layers': key_layers,
        'polarity_causal_dir': {},
        'tense_causal_dir': {},
        'causal_cosine': {},
        'decode_cosine': {},
    }
    
    for li in key_layers:
        print(f"\n  --- L{li} ---")
        
        # 提取极性差分方向
        pol_aff = [a for a, b in pol_pairs]
        pol_neg = [b for a, b in pol_pairs]
        pol_texts = pol_aff + pol_neg
        pol_labels = np.array([0] * len(pol_aff) + [1] * len(pol_neg))
        
        pol_repr = extract_residual_stream(model, tokenizer, device, pol_texts, [li], 'last')
        X_pol = pol_repr[li]
        
        n_half = len(pol_aff)
        mean_aff = X_pol[:n_half].mean(axis=0)
        mean_neg = X_pol[n_half:].mean(axis=0)
        pol_causal_dir = mean_neg - mean_aff
        pol_norm = np.linalg.norm(pol_causal_dir)
        if pol_norm > 1e-10:
            pol_causal_dir = pol_causal_dir / pol_norm
        
        # 提取时态差分方向
        tense_pres = [a for a, b in tense_pairs]
        tense_past = [b for a, b in tense_pairs]
        tense_texts = tense_pres + tense_past
        tense_labels_arr = np.array([0] * len(tense_pres) + [1] * len(tense_past))
        
        tense_repr = extract_residual_stream(model, tokenizer, device, tense_texts, [li], 'last')
        X_tense = tense_repr[li]
        
        n_half_t = len(tense_pres)
        mean_pres = X_tense[:n_half_t].mean(axis=0)
        mean_past = X_tense[n_half_t:].mean(axis=0)
        tense_causal_dir = mean_past - mean_pres
        tense_norm = np.linalg.norm(tense_causal_dir)
        if tense_norm > 1e-10:
            tense_causal_dir = tense_causal_dir / tense_norm
        
        # 因果方向余弦
        causal_cos = float(np.dot(pol_causal_dir, tense_causal_dir))
        
        # 解码方向余弦(用probe)
        scaler = StandardScaler()
        X_pol_scaled = scaler.fit_transform(X_pol)
        clf_pol = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        clf_pol.fit(X_pol_scaled, pol_labels)
        w_pol = clf_pol.coef_[0]
        w_pol_norm = w_pol / (np.linalg.norm(w_pol) + 1e-10)
        
        X_tense_scaled = scaler.fit_transform(X_tense)
        clf_tense = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        clf_tense.fit(X_tense_scaled, tense_labels_arr)
        w_tense = clf_tense.coef_[0]
        w_tense_norm = w_tense / (np.linalg.norm(w_tense) + 1e-10)
        
        decode_cos = float(np.dot(w_pol_norm, w_tense_norm))
        
        results['polarity_causal_dir'][li] = {
            'norm': float(pol_norm),
        }
        results['tense_causal_dir'][li] = {
            'norm': float(tense_norm),
        }
        results['causal_cosine'][li] = causal_cos
        results['decode_cosine'][li] = decode_cos
        
        print(f"    因果方向cos(pol, tense): {causal_cos:+.4f}")
        print(f"    解码方向cos(pol, tense): {decode_cos:+.4f}")
        print(f"    差分方向范数: pol={pol_norm:.4f}, tense={tense_norm:.4f}")
        
        if abs(causal_cos) < 0.1:
            print(f"    → 因果方向几乎正交! 语言特征在因果空间正交")
        elif abs(causal_cos) < 0.3:
            print(f"    → 因果方向弱相关")
        else:
            print(f"    → 因果方向强相关! 语言特征在因果空间不对齐")
    
    # === 汇总 ===
    print("\n" + "=" * 70)
    print("C3 汇总: 因果路径正交性 vs 解码方向正交性")
    print("=" * 70)
    
    for li in key_layers:
        cc = results['causal_cosine'][li]
        dc = results['decode_cosine'][li]
        print(f"  L{li}: causal_cos={cc:+.4f}, decode_cos={dc:+.4f}, diff={cc-dc:+.4f}")
    
    # 因果方向与解码方向的对比
    causal_abs_mean = np.mean([abs(v) for v in results['causal_cosine'].values()])
    decode_abs_mean = np.mean([abs(v) for v in results['decode_cosine'].values()])
    
    print(f"\n  |cos|平均: causal={causal_abs_mean:.4f}, decode={decode_abs_mean:.4f}")
    if causal_abs_mean < decode_abs_mean:
        print("  → 因果方向比解码方向更正交! 因果空间的语言结构更分离")
    else:
        print("  → 因果方向与解码方向正交性相似")
    
    return results


# ============================================================
# C4: 因果效应的层间演化
# ============================================================

def test_c4(model_name, model, tokenizer, device, d_model, n_layers):
    """
    C4: 因果效应的层间演化 — 极性vs随机, 逐层追踪
    
    在每一层做残差流修补, 测量极性patch和随机patch的效应差异
    """
    
    print("\n" + "=" * 70)
    print("C4: 因果效应层间演化 — 极性patch vs 随机patch, 逐层追踪")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()[:30]
    
    # 采样更多层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 10)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    
    n_patches = 10  # 10对patch
    
    results = {
        'sample_layers': sample_layers,
        'n_patches': n_patches,
        'polarity_effect': {},
        'random_effect': {},
        'effect_ratio': {},
    }
    
    for li in sample_layers:
        pol_deltas = []
        rand_deltas = []
        
        for i in range(n_patches):
            aff, neg = pol_pairs[i]
            
            try:
                pol_delta = compute_residual_patch_effect(
                    model, tokenizer, device, neg, aff, li
                )
                pol_deltas.append(pol_delta)
            except:
                pol_deltas.append(0)
            
            rand_idx = np.random.randint(0, len(pol_pairs))
            rand_source = pol_pairs[rand_idx][np.random.randint(0, 2)]
            try:
                rand_delta = compute_residual_patch_effect(
                    model, tokenizer, device, rand_source, aff, li
                )
                rand_deltas.append(rand_delta)
            except:
                rand_deltas.append(0)
        
        pol_abs = float(np.mean(np.abs(pol_deltas)))
        rand_abs = float(np.mean(np.abs(rand_deltas)))
        ratio = pol_abs / max(rand_abs, 1e-10)
        
        results['polarity_effect'][li] = {
            'mean': float(np.mean(pol_deltas)),
            'abs_mean': pol_abs,
        }
        results['random_effect'][li] = {
            'mean': float(np.mean(rand_deltas)),
            'abs_mean': rand_abs,
        }
        results['effect_ratio'][li] = ratio
        
        print(f"  L{li:2d}: pol|delta|={pol_abs:.4f}, rand|delta|={rand_abs:.4f}, ratio={ratio:.2f}x")
    
    # === 汇总 ===
    print("\n" + "=" * 70)
    print("C4 汇总: 因果效应的层间演化")
    print("=" * 70)
    
    # 找极性效应最强的层
    max_layer = max(results['effect_ratio'], key=results['effect_ratio'].get)
    max_ratio = results['effect_ratio'][max_layer]
    print(f"  极性效应最强层: L{max_layer}, ratio={max_ratio:.2f}x")
    
    # 早期vs晚期
    early_layers = [l for l in sample_layers if l < n_layers // 3]
    late_layers = [l for l in sample_layers if l > 2 * n_layers // 3]
    
    if early_layers and late_layers:
        early_ratio = np.mean([results['effect_ratio'][l] for l in early_layers])
        late_ratio = np.mean([results['effect_ratio'][l] for l in late_layers])
        print(f"  早期平均ratio: {early_ratio:.2f}x")
        print(f"  晚期平均ratio: {late_ratio:.2f}x")
        if early_ratio > late_ratio * 1.5:
            print("  → 极性因果效应在早期更强! 与注意力头发现一致")
        elif late_ratio > early_ratio * 1.5:
            print("  → 极性因果效应在晚期更强")
        else:
            print("  → 极性因果效应在各层差异不大")
    
    return results


# ============================================================
# 主入口
# ============================================================

def run_test(model_name, test_name):
    print(f"\n{'='*70}")
    print(f"Phase CXCVII: Causal Baseline - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model_fast(model_name)
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    
    print(f"  n_layers={n_layers}, d_model={d_model}")
    
    result_dir = Path(f"results/causal_baseline/{model_name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    results = {}
    
    def convert_keys(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                new_key = str(k) if isinstance(k, tuple) else k
                new_dict[new_key] = convert_keys(v)
            return new_dict
        elif isinstance(obj, list):
            return [convert_keys(item) for item in obj]
        else:
            return obj
    
    if test_name in ['c1', 'all']:
        r = test_c1(model_name, model, tokenizer, device, d_model, n_layers)
        results['c1'] = r
        with open(result_dir / "c1_results.json", 'w') as f:
            json.dump(convert_keys(r), f, indent=2, default=str)
    
    if test_name in ['c2', 'all']:
        r = test_c2(model_name, model, tokenizer, device, d_model, n_layers)
        results['c2'] = r
        with open(result_dir / "c2_results.json", 'w') as f:
            json.dump(convert_keys(r), f, indent=2, default=str)
    
    if test_name in ['c3', 'all']:
        r = test_c3(model_name, model, tokenizer, device, d_model, n_layers)
        results['c3'] = r
        with open(result_dir / "c3_results.json", 'w') as f:
            json.dump(convert_keys(r), f, indent=2, default=str)
    
    if test_name in ['c4', 'all']:
        r = test_c4(model_name, model, tokenizer, device, d_model, n_layers)
        results['c4'] = r
        with open(result_dir / "c4_results.json", 'w') as f:
            json.dump(convert_keys(r), f, indent=2, default=str)
    
    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.1f}s")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test", type=str, default="all", choices=["c1", "c2", "c3", "c4", "all"])
    args = parser.parse_args()
    
    run_test(args.model, args.test)
