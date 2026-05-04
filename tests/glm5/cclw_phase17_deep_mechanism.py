"""
CCL-W(Phase 17): 语法方向因果效力的深层机制与论元-修饰二元性的数学证明
=============================================================================
核心问题(基于Phase 16的发现):
  1. ★★★★★★★★★ 语法方向的层间传播
     → Phase 16发现Qwen3的语法方向100倍于随机方向(DS7B仅0.8倍)
     → 但干预只在L0(embedding层)进行
     → 在哪一层注入语法方向最有效? L0? L5? L10? L20?
     → 语法方向在不同层是否保持方向一致?
  
  2. ★★★★★★★★★ 语法头的精确功能
     → Phase 16发现75-89%的头是语法头
     → Top语法头到底关注什么?
     → 语法头关注语法标记还是语义信息?
  
  3. ★★★★★★★ 论元-修饰二元性的因果验证
     → Phase 16发现二维正交模型(名词轴+修饰语轴)
     → 在embedding中只注入名词轴方向: 预测是否偏向名词模式?
     → 只注入修饰语轴方向: 预测是否偏向修饰模式?
     → 正交性是否因果地影响模型行为?
  
  4. ★★★★★★★ advmod的第三维: 方式vs时间
     → Phase 16发现advmod偏离二维平面最大(残差12-17)
     → 第三维编码了什么? 方式vs时间的区分?
     → 系统地比较方式副词(wisely/quickly) vs 时间副词(yesterday/today)

实验:
  Exp1: ★★★★★★★★★ 语法方向的层间传播
    → 在不同层(L0, L5, L10, L20, 最后一层)注入语法方向
    → 比较各层的干预效力(KL散度)
    → 追踪语法方向在层间的传播路径

  Exp2: ★★★★★★★★★ 语法头的精确功能与信息流
    → Top语法头的功能分析
    → 逐层逐头的语法方向投影
    → 语法头的注意力模式(能获取的话)

  Exp3: ★★★★★★★ 论元-修饰二元性的因果验证
    → 在embedding中只注入名词轴方向
    → 只注入修饰语轴方向
    → 验证两个轴的因果独立性

  Exp4: ★★★★★★★ advmod的第三维: 方式vs时间
    → 方式副词(wisely/quickly/carefully) vs 时间副词(yesterday/today/always)
    → 它们在第三维上的位置
    → 第三维是否编码了方式vs时间的对立?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== 复用Phase 15/16的数据 =====
MANIFOLD_ROLES_DATA = {
    "nsubj": {
        "sentences": [
            "The king ruled the kingdom wisely",
            "The doctor treated the patient carefully",
            "The artist painted the portrait beautifully",
            "The soldier defended the castle bravely",
            "The teacher explained the lesson clearly",
            "The chef cooked the meal perfectly",
            "The cat chased the mouse quickly",
            "The dog found the bone happily",
            "The woman drove the car safely",
            "The man fixed the roof carefully",
            "The student read the book quietly",
            "The singer performed the song brilliantly",
            "The baker made the bread daily",
            "The pilot flew the plane smoothly",
            "The farmer grew the crops diligently",
            "The writer wrote the novel slowly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
    "poss": {
        "sentences": [
            "The king's crown glittered brightly",
            "The doctor's office opened early",
            "The artist's studio looked beautiful",
            "The soldier's uniform was clean",
            "The teacher's book sold quickly",
            "The chef's restaurant opened today",
            "The cat's tail swished gently",
            "The dog's bark echoed loudly",
            "The woman's dress looked elegant",
            "The man's car drove fast",
            "The student's essay read well",
            "The singer's voice rang clearly",
            "The baker's shop smelled wonderful",
            "The pilot's license was renewed",
            "The farmer's land was fertile",
            "The writer's pen wrote smoothly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
    "dobj": {
        "sentences": [
            "They crowned the king yesterday",
            "She visited the doctor recently",
            "He admired the artist greatly",
            "We honored the soldier today",
            "You thanked the teacher warmly",
            "The customer tipped the chef generously",
            "The hawk chased the cat swiftly",
            "The boy found the dog outside",
            "The police arrested the woman quickly",
            "The company hired the man recently",
            "I praised the student loudly",
            "They applauded the singer warmly",
            "She visited the baker often",
            "He admired the pilot greatly",
            "We thanked the farmer sincerely",
            "The editor praised the writer highly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
    "amod": {
        "sentences": [
            "The brave king fought hard",
            "The kind doctor helped many",
            "The creative artist worked well",
            "The strong soldier marched far",
            "The wise teacher explained clearly",
            "The skilled chef cooked perfectly",
            "The quick cat ran fast",
            "The loyal dog stayed close",
            "The old woman walked slowly",
            "The tall man stood quietly",
            "The bright student read carefully",
            "The talented singer performed brilliantly",
            "The patient baker waited calmly",
            "The careful pilot landed smoothly",
            "The hardworking farmer harvested early",
            "The thoughtful writer reflected deeply",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "wise",
            "skilled", "quick", "loyal", "old", "tall",
            "bright", "talented", "patient", "careful", "hardworking", "thoughtful",
        ],
    },
    "advmod": {
        "sentences": [
            "The king ruled wisely forever",
            "The doctor worked carefully always",
            "The artist painted beautifully daily",
            "The soldier fought bravely there",
            "The teacher spoke clearly again",
            "The chef worked quickly then",
            "The cat ran swiftly home",
            "The dog barked loudly today",
            "The woman drove slowly forward",
            "The man spoke quietly now",
            "The student studied carefully alone",
            "The singer performed brilliantly tonight",
            "The baker baked freshly daily",
            "The pilot flew steadily onward",
            "The farmer worked diligently always",
            "The writer typed quickly away",
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely", "clearly",
            "quickly", "swiftly", "loudly", "slowly", "quietly",
            "carefully", "brilliantly", "freshly", "steadily", "diligently", "quickly",
        ],
    },
    "pobj": {
        "sentences": [
            "They looked at the king closely",
            "She waited for the doctor patiently",
            "He thought about the artist often",
            "We marched toward the soldier steadily",
            "You listened to the teacher attentively",
            "They paid the chef generously",
            "She played with the cat happily",
            "He walked toward the dog slowly",
            "The gift belonged to the woman originally",
            "The letter was for the man personally",
            "I read about the student recently",
            "They talked about the singer excitedly",
            "She ordered from the baker regularly",
            "He flew with the pilot recently",
            "We learned from the farmer carefully",
            "They wrote about the writer frequently",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
}

# ===== Exp4: 方式副词 vs 时间副词数据 =====
ADVERB_TYPE_DATA = {
    "manner": {
        "description": "方式副词(描述动作方式)",
        "sentences": [
            "The king ruled wisely and justly",
            "The doctor worked carefully and precisely",
            "The artist painted beautifully and skillfully",
            "The soldier fought bravely and fiercely",
            "The teacher spoke clearly and patiently",
            "The chef cooked perfectly and expertly",
            "The cat ran swiftly and gracefully",
            "The dog barked loudly and aggressively",
            "The woman drove slowly and cautiously",
            "The man spoke quietly and gently",
            "The student studied diligently and thoroughly",
            "The singer performed brilliantly and passionately",
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely", "clearly",
            "perfectly", "swiftly", "loudly", "slowly", "quietly",
            "diligently", "brilliantly",
        ],
    },
    "temporal": {
        "description": "时间副词(描述动作时间)",
        "sentences": [
            "The king ruled yesterday and today",
            "The doctor worked early and late",
            "The artist painted daily and weekly",
            "The soldier fought recently and formerly",
            "The teacher spoke always and never",
            "The chef cooked now and then",
            "The cat ran today and tomorrow",
            "The dog barked tonight and yesterday",
            "The woman drove already and soon",
            "The man spoke before and after",
            "The student studied always and forever",
            "The singer performed recently and currently",
        ],
        "target_words": [
            "yesterday", "early", "daily", "recently", "always",
            "now", "today", "tonight", "already", "before",
            "always", "recently",
        ],
    },
}


def find_token_index(tokens, word):
    word_lower = word.lower()
    word_start = word_lower[:3]
    for i, tok in enumerate(tokens):
        tok_lower = tok.lower().strip()
        if word_lower in tok_lower or tok_lower.startswith(word_start):
            return i
    for i, tok in enumerate(tokens):
        if word_lower[:2] in tok.lower():
            return i
    return None


def collect_hs_at_layer(model, tokenizer, device, sentences, target_words, layer_idx):
    """在指定层收集target token的hidden states"""
    layers = get_layers(model)
    if layer_idx >= len(layers):
        layer_idx = len(layers) - 1
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    target_layer = layers[layer_idx]

    all_h = []
    valid_words = []

    for sent, target_word in zip(sentences, target_words):
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]

        dep_idx = find_token_index(tokens_list, target_word)
        if dep_idx is None:
            continue

        captured = {}
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured['h'] = output[0].detach().float().cpu().numpy()
            else:
                captured['h'] = output.detach().float().cpu().numpy()

        h_handle = target_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(**toks)
        h_handle.remove()

        if 'h' not in captured:
            continue

        h_vec = captured['h'][0, dep_idx, :]
        all_h.append(h_vec)
        valid_words.append(target_word)

    return np.array(all_h) if all_h else None


def get_syntax_directions_at_layer(model, tokenizer, device, model_info, layer_idx):
    """获取指定层的语法方向(nsubj-dobj, nsubj-amod等)"""
    role_names = ["nsubj", "poss", "dobj", "amod", "advmod", "pobj"]
    role_h = {}
    for role in role_names:
        data = MANIFOLD_ROLES_DATA[role]
        H = collect_hs_at_layer(model, tokenizer, device,
                                data["sentences"], data["target_words"], layer_idx)
        if H is not None and len(H) > 0:
            role_h[role] = H

    centers = {}
    for role in role_names:
        if role in role_h:
            centers[role] = np.mean(role_h[role], axis=0)

    directions = {}
    # nsubj→dobj方向
    if 'nsubj' in centers and 'dobj' in centers:
        d = centers['dobj'] - centers['nsubj']
        norm = np.linalg.norm(d)
        if norm > 0:
            directions['nsubj_dobj'] = d / norm
            directions['nsubj_dobj_norm'] = norm
            directions['nsubj_dobj_raw'] = d

    # nsubj→amod方向 (修饰语轴)
    if 'nsubj' in centers and 'amod' in centers:
        d = centers['amod'] - centers['nsubj']
        norm = np.linalg.norm(d)
        if norm > 0:
            directions['nsubj_amod'] = d / norm
            directions['nsubj_amod_norm'] = norm
            directions['nsubj_amod_raw'] = d

    # 名词轴和修饰语轴(正交分解)
    if 'nsubj_dobj_raw' in directions and 'nsubj_amod_raw' in directions:
        # 名词轴 = nsubj→dobj方向
        noun_axis = directions['nsubj_dobj_raw']
        noun_axis_norm = np.linalg.norm(noun_axis)
        if noun_axis_norm > 0:
            noun_axis = noun_axis / noun_axis_norm

        # 修饰语轴 = nsubj→amod去掉名词轴分量
        amod_raw = directions['nsubj_amod_raw']
        amod_proj = np.dot(amod_raw, noun_axis) * noun_axis
        modifier_axis = amod_raw - amod_proj
        modifier_norm = np.linalg.norm(modifier_axis)
        if modifier_norm > 0:
            modifier_axis = modifier_axis / modifier_norm

        directions['noun_axis'] = noun_axis
        directions['modifier_axis'] = modifier_axis
        # 正交性检验
        directions['axis_orthogonality'] = float(np.dot(noun_axis, modifier_axis))

    return directions, centers


# ===== Exp1: 语法方向的层间传播 =====
def exp1_layer_propagation(model, tokenizer, device):
    """语法方向在不同层的干预效力"""
    print("\n" + "="*70)
    print("Exp1: 语法方向的层间传播 ★★★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Step 1: 获取L0的语法方向
    print("\n  Step 1: 获取L0的语法方向")
    dirs_l0, centers_l0 = get_syntax_directions_at_layer(model, tokenizer, device, model_info, 0)
    
    if 'nsubj_dobj' not in dirs_l0:
        print("  无法获取L0语法方向!")
        return results

    nsubj_dobj_dir_l0 = dirs_l0['nsubj_dobj']
    print(f"  L0 nsubj-dobj方向范数: {dirs_l0.get('nsubj_dobj_norm', 0):.4f}")

    # Step 2: 在多个层注入语法方向, 比较效力
    print("\n  Step 2: 多层干预效力比较")
    
    # 采样层
    sample_layers = [0]
    if n_layers > 5:
        sample_layers += [1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    else:
        sample_layers = list(range(n_layers))
    sample_layers = sorted(set(sample_layers))
    
    test_sent = "They crowned the king yesterday"
    test_word = "king"
    
    layer_kl_results = {}
    
    for layer_idx in sample_layers:
        print(f"\n  === 层 {layer_idx} ===")
        
        # 获取该层的语法方向
        dirs_l, centers_l = get_syntax_directions_at_layer(
            model, tokenizer, device, model_info, layer_idx)
        
        if 'nsubj_dobj' not in dirs_l:
            print(f"  L{layer_idx}: 无法获取语法方向, 跳过")
            continue
        
        nsubj_dobj_dir_l = dirs_l['nsubj_dobj']
        nsubj_dobj_norm_l = dirs_l.get('nsubj_dobj_norm', 0)
        
        # L0方向与该层方向的余弦相似度
        cos_with_l0 = float(np.dot(nsubj_dobj_dir_l0, nsubj_dobj_dir_l))
        print(f"  L{layer_idx} nsubj-dobj方向范数: {nsubj_dobj_norm_l:.4f}")
        print(f"  与L0方向余弦相似度: {cos_with_l0:.4f}")
        
        # 在该层进行干预
        toks = tokenizer(test_sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens_list, test_word)
        
        if dep_idx is None:
            continue
        
        # 基线: 正常前向传播
        layers_list = get_layers(model)
        
        # 获取embedding
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
        
        with torch.no_grad():
            base_logits = model(inputs_embeds=inputs_embeds).logits[0, dep_idx, :]
            base_probs = torch.softmax(base_logits, dim=-1)
        
        # 干预: 在target层注入语法方向
        # 使用hook在特定层的输出上添加方向
        captured_h = {}
        def make_intervention_hook(layer_target, direction, alpha):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    # 在target token位置注入方向
                    direction_t = torch.tensor(direction, dtype=h.dtype, device=h.device)
                    h[0, dep_idx, :] += (alpha * direction_t).to(h.dtype)
                    return (h,) + output[1:]
                return output
            return hook
        
        for alpha in [1.0, 2.0, 4.0]:
            hook = layers_list[layer_idx].register_forward_hook(
                make_intervention_hook(layer_idx, nsubj_dobj_dir_l, alpha))
            
            with torch.no_grad():
                int_logits = model(inputs_embeds=inputs_embeds).logits[0, dep_idx, :]
                int_probs = torch.softmax(int_logits, dim=-1)
            
            hook.remove()
            
            kl_div = torch.nn.functional.kl_div(
                torch.log(int_probs + 1e-10),
                base_probs + 1e-10,
                reduction='sum'
            ).item()
            
            print(f"  L{layer_idx} alpha={alpha}: KL={kl_div:.4f}")
            
            if alpha == 2.0:
                layer_kl_results[layer_idx] = {
                    'kl_div': float(kl_div),
                    'dir_norm': float(nsubj_dobj_norm_l),
                    'cos_with_l0': float(cos_with_l0),
                }
    
    # 随机方向对照
    print("\n  Step 3: 随机方向对照")
    np.random.seed(42)
    random_dirs = [np.random.randn(d_model) for _ in range(3)]
    random_dirs = [d / np.linalg.norm(d) for d in random_dirs]
    
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
    
    with torch.no_grad():
        base_logits = model(inputs_embeds=inputs_embeds).logits[0, dep_idx, :]
        base_probs = torch.softmax(base_logits, dim=-1)
    
    random_kls = []
    for layer_idx in sample_layers[:3]:  # 只测试前3层
        for rdir in random_dirs:
            hook = layers_list[layer_idx].register_forward_hook(
                make_intervention_hook(layer_idx, rdir, 2.0))
            with torch.no_grad():
                int_logits = model(inputs_embeds=inputs_embeds).logits[0, dep_idx, :]
                int_probs = torch.softmax(int_logits, dim=-1)
            hook.remove()
            
            kl = torch.nn.functional.kl_div(
                torch.log(int_probs + 1e-10),
                base_probs + 1e-10,
                reduction='sum'
            ).item()
            random_kls.append(kl)
    
    avg_random_kl = np.mean(random_kls) if random_kls else 1e-10
    print(f"  随机方向平均KL(alpha=2.0): {avg_random_kl:.4f}")
    
    # 比较各层的语法/随机比
    print("\n  Step 4: 各层语法方向效力总结")
    for layer_idx in sorted(layer_kl_results.keys()):
        r = layer_kl_results[layer_idx]
        ratio = r['kl_div'] / max(avg_random_kl, 1e-10)
        print(f"  L{layer_idx}: 语法KL={r['kl_div']:.4f}, 语法/随机={ratio:.2f}x, "
              f"方向范数={r['dir_norm']:.4f}, cos(L0)={r['cos_with_l0']:.4f}")
    
    results['layer_kl_results'] = {str(k): v for k, v in layer_kl_results.items()}
    results['avg_random_kl'] = float(avg_random_kl)
    results['sample_layers'] = sample_layers
    
    return results


# ===== Exp2: 语法头的精确功能与信息流 =====
def exp2_syntax_head_function(model, tokenizer, device):
    """语法头的逐层逐头分析"""
    print("\n" + "="*70)
    print("Exp2: 语法头的精确功能与信息流 ★★★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)

    # 采样层
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    sample_layers = sorted(set([l for l in sample_layers if l < n_layers]))

    # Step 1: 逐层逐头的语法方向投影
    print("\n  Step 1: 逐层逐头的语法方向投影")
    
    # 先获取L0语法方向
    dirs_l0, _ = get_syntax_directions_at_layer(model, tokenizer, device, model_info, 0)
    if 'nsubj_dobj' not in dirs_l0:
        print("  无法获取L0语法方向!")
        return results
    nsubj_dobj_dir = dirs_l0['nsubj_dobj']

    # 采样句子
    nsubj_sents = MANIFOLD_ROLES_DATA["nsubj"]["sentences"][:4]
    nsubj_words = MANIFOLD_ROLES_DATA["nsubj"]["target_words"][:4]
    dobj_sents = MANIFOLD_ROLES_DATA["dobj"]["sentences"][:4]
    dobj_words = MANIFOLD_ROLES_DATA["dobj"]["target_words"][:4]

    # 获取n_heads
    layer0 = layers[0]
    sa = layer0.self_attn
    if hasattr(sa, 'num_heads'):
        n_heads = sa.num_heads
    elif hasattr(sa, 'num_attention_heads'):
        n_heads = sa.num_attention_heads
    else:
        n_heads = d_model // 64
    head_dim = d_model // n_heads
    print(f"  n_heads={n_heads}, head_dim={head_dim}")

    layer_head_syntax = {}
    
    for layer_idx in sample_layers:
        target_layer = layers[layer_idx]
        
        # 收集各头在nsubj/dobj句子中的输出
        nsubj_head_h = {h: [] for h in range(n_heads)}
        dobj_head_h = {h: [] for h in range(n_heads)}
        
        for sents, words, role in [(nsubj_sents, nsubj_words, "nsubj"),
                                    (dobj_sents, dobj_words, "dobj")]:
            for sent, tw in zip(sents, words):
                toks = tokenizer(sent, return_tensors="pt").to(device)
                input_ids = toks.input_ids
                tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
                dep_idx = find_token_index(tokens_list, tw)
                if dep_idx is None:
                    continue
                
                captured = {}
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        captured['h'] = output[0].detach().float().cpu().numpy()
                    else:
                        captured['h'] = output.detach().float().cpu().numpy()
                
                h_handle = target_layer.register_forward_hook(hook_fn)
                with torch.no_grad():
                    _ = model(**toks)
                h_handle.remove()
                
                if 'h' not in captured:
                    continue
                
                h_vec = captured['h'][0, dep_idx, :]
                for h in range(n_heads):
                    head_seg = h_vec[h*head_dim:(h+1)*head_dim]
                    if role == "nsubj":
                        nsubj_head_h[h].append(head_seg)
                    else:
                        dobj_head_h[h].append(head_seg)
        
        # 计算每个头的语法区分力
        head_scores = []
        for h in range(n_heads):
            ns_arr = np.array(nsubj_head_h[h])
            do_arr = np.array(dobj_head_h[h])
            if len(ns_arr) >= 1 and len(do_arr) >= 1:
                ns_center = np.mean(ns_arr, axis=0)
                do_center = np.mean(do_arr, axis=0)
                sep = np.linalg.norm(do_center - ns_center)
                
                # 各头中心在语法方向上的投影
                ns_proj = np.dot(ns_center, nsubj_dobj_dir[h*head_dim:(h+1)*head_dim])
                do_proj = np.dot(do_center, nsubj_dobj_dir[h*head_dim:(h+1)*head_dim])
                proj_diff = do_proj - ns_proj
                
                head_scores.append({
                    'head': h,
                    'separation': float(sep),
                    'proj_diff': float(proj_diff),
                    'ns_proj': float(ns_proj),
                    'do_proj': float(do_proj),
                })
        
        # 排序
        head_scores.sort(key=lambda x: -abs(x['proj_diff']))
        layer_head_syntax[layer_idx] = head_scores
        
        # 打印Top-5头
        print(f"\n  L{layer_idx} Top-5语法头:")
        for s in head_scores[:5]:
            print(f"    Head {s['head']}: sep={s['separation']:.4f}, "
                  f"proj_diff={s['proj_diff']:.4f}, "
                  f"ns_proj={s['ns_proj']:.4f}, do_proj={s['do_proj']:.4f}")

    # Step 2: 跨层语法头的连续性
    print("\n  Step 2: 跨层语法头的连续性分析")
    
    # 检查: 同一个头在不同层是否持续是语法头?
    if len(sample_layers) >= 2:
        # 找到每层Top-5语法头
        layer_top_heads = {}
        for layer_idx in sample_layers:
            if layer_idx in layer_head_syntax:
                top_h = [s['head'] for s in layer_head_syntax[layer_idx][:5]]
                layer_top_heads[layer_idx] = top_h
        
        print("  各层Top-5语法头:")
        for layer_idx in sorted(layer_top_heads.keys()):
            print(f"    L{layer_idx}: {layer_top_heads[layer_idx]}")
        
        # 连续性: 是否有头在多层都是Top-5?
        all_top_heads = set()
        for heads in layer_top_heads.values():
            all_top_heads.update(heads)
        
        head_persistence = {}
        for h in all_top_heads:
            count = sum(1 for heads in layer_top_heads.values() if h in heads)
            head_persistence[h] = count
        
        persistent_heads = {h: c for h, c in head_persistence.items() if c >= 2}
        if persistent_heads:
            print(f"\n  ★ 在≥2层为Top-5语法头的头:")
            for h, c in sorted(persistent_heads.items(), key=lambda x: -x[1]):
                layers_present = [l for l, heads in layer_top_heads.items() if h in heads]
                print(f"    Head {h}: 出现在{c}层: L{layers_present}")
        else:
            print(f"\n  ★ 没有头在≥2层都是Top-5语法头")
            print(f"    → 语法功能在不同层由不同头承担!")
    
    results['layer_head_syntax'] = {str(k): v for k, v in layer_head_syntax.items()}
    results['n_heads'] = n_heads
    results['head_dim'] = head_dim
    results['sample_layers'] = sample_layers
    
    return results


# ===== Exp3: 论元-修饰二元性的因果验证 =====
def exp3_argument_modifier_causal(model, tokenizer, device):
    """论元-修饰二元性的因果验证"""
    print("\n" + "="*70)
    print("Exp3: 论元-修饰二元性的因果验证 ★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Step 1: 获取L0的名词轴和修饰语轴
    print("\n  Step 1: 获取名词轴和修饰语轴(L0)")
    dirs_l0, centers_l0 = get_syntax_directions_at_layer(model, tokenizer, device, model_info, 0)
    
    if 'noun_axis' not in dirs_l0 or 'modifier_axis' not in dirs_l0:
        print("  无法获取名词轴和修饰语轴!")
        return results
    
    noun_axis = dirs_l0['noun_axis']
    modifier_axis = dirs_l0['modifier_axis']
    ortho = dirs_l0['axis_orthogonality']
    
    print(f"  名词轴范数: {np.linalg.norm(noun_axis):.4f}")
    print(f"  修饰语轴范数: {np.linalg.norm(modifier_axis):.4f}")
    print(f"  正交性(点积): {ortho:.6f}")

    # Step 2: 注入名词轴方向
    print("\n  Step 2: 注入名词轴方向")
    
    # 测试: 在不同句子中注入名词轴
    # 名词轴方向应该让模型更倾向于预测名词/论元模式
    test_cases = [
        ("The king", "king"),  # nsubj
        ("They crowned the", "the"),  # 后面应该跟dobj
    ]
    
    noun_axis_results = []
    for sent, target in test_cases:
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        
        # 在最后一个token位置注入
        last_idx = len(tokens_list) - 1
        
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
        
        with torch.no_grad():
            base_logits = model(inputs_embeds=inputs_embeds).logits[0, last_idx, :]
            base_top10 = torch.topk(base_logits, 10)
            base_tokens = [safe_decode(tokenizer, t.item()) for t in base_top10.indices]
            base_probs = torch.softmax(base_logits, dim=-1)
        
        for alpha in [1.0, 2.0, 4.0]:
            inputs_intervened = inputs_embeds.clone()
            noun_tensor = torch.tensor(noun_axis, dtype=inputs_embeds.dtype, device=device)
            # 注入名词轴 = 向dobj方向移动
            inputs_intervened[0, last_idx, :] += (alpha * noun_tensor).to(model.dtype)
            
            with torch.no_grad():
                int_logits = model(inputs_embeds=inputs_intervened).logits[0, last_idx, :]
                int_top10 = torch.topk(int_logits, 10)
                int_tokens = [safe_decode(tokenizer, t.item()) for t in int_top10.indices]
                int_probs = torch.softmax(int_logits, dim=-1)
            
            kl = torch.nn.functional.kl_div(
                torch.log(int_probs + 1e-10),
                base_probs + 1e-10,
                reduction='sum'
            ).item()
            
            print(f"  '{sent}' alpha={alpha}:")
            print(f"    Base: {base_tokens[:5]}")
            print(f"    +Noun: {int_tokens[:5]} (KL={kl:.4f})")
            
            noun_axis_results.append({
                'sentence': sent,
                'axis': 'noun',
                'alpha': alpha,
                'base_top5': base_tokens[:5],
                'int_top5': int_tokens[:5],
                'kl': float(kl),
            })

    # Step 3: 注入修饰语轴方向
    print("\n  Step 3: 注入修饰语轴方向")
    
    modifier_axis_results = []
    for sent, target in test_cases:
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        last_idx = len(tokens_list) - 1
        
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
        
        with torch.no_grad():
            base_logits = model(inputs_embeds=inputs_embeds).logits[0, last_idx, :]
            base_probs = torch.softmax(base_logits, dim=-1)
            base_top10 = torch.topk(base_logits, 10)
            base_tokens = [safe_decode(tokenizer, t.item()) for t in base_top10.indices]
        
        for alpha in [1.0, 2.0, 4.0]:
            inputs_intervened = inputs_embeds.clone()
            mod_tensor = torch.tensor(modifier_axis, dtype=inputs_embeds.dtype, device=device)
            inputs_intervened[0, last_idx, :] += (alpha * mod_tensor).to(model.dtype)
            
            with torch.no_grad():
                int_logits = model(inputs_embeds=inputs_intervened).logits[0, last_idx, :]
                int_top10 = torch.topk(int_logits, 10)
                int_tokens = [safe_decode(tokenizer, t.item()) for t in int_top10.indices]
                int_probs = torch.softmax(int_logits, dim=-1)
            
            kl = torch.nn.functional.kl_div(
                torch.log(int_probs + 1e-10),
                base_probs + 1e-10,
                reduction='sum'
            ).item()
            
            print(f"  '{sent}' alpha={alpha}:")
            print(f"    Base: {base_tokens[:5]}")
            print(f"    +Mod: {int_tokens[:5]} (KL={kl:.4f})")
            
            modifier_axis_results.append({
                'sentence': sent,
                'axis': 'modifier',
                'alpha': alpha,
                'base_top5': base_tokens[:5],
                'int_top5': int_tokens[:5],
                'kl': float(kl),
            })

    # Step 4: 名词轴 vs 修饰语轴的因果独立性
    print("\n  Step 4: 名词轴 vs 修饰语轴的因果独立性")
    
    # 在"The king"上分别注入两个轴, 比较预测变化
    sent = "The king"
    toks = tokenizer(sent, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    last_idx = input_ids.shape[1] - 1
    
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
    
    with torch.no_grad():
        base_logits = model(inputs_embeds=inputs_embeds).logits[0, last_idx, :]
        base_probs = torch.softmax(base_logits, dim=-1)
    
    alpha = 2.0
    
    # 名词轴干预
    inputs_noun = inputs_embeds.clone()
    noun_tensor = torch.tensor(noun_axis, dtype=inputs_embeds.dtype, device=device)
    inputs_noun[0, last_idx, :] += (alpha * noun_tensor).to(model.dtype)
    with torch.no_grad():
        noun_logits = model(inputs_embeds=inputs_noun).logits[0, last_idx, :]
        noun_probs = torch.softmax(noun_logits, dim=-1)
    
    # 修饰语轴干预
    inputs_mod = inputs_embeds.clone()
    mod_tensor = torch.tensor(modifier_axis, dtype=inputs_embeds.dtype, device=device)
    inputs_mod[0, last_idx, :] += (alpha * mod_tensor).to(model.dtype)
    with torch.no_grad():
        mod_logits = model(inputs_embeds=inputs_mod).logits[0, last_idx, :]
        mod_probs = torch.softmax(mod_logits, dim=-1)
    
    # 两个轴同时干预
    inputs_both = inputs_embeds.clone()
    inputs_both[0, last_idx, :] += (alpha * noun_tensor).to(model.dtype)
    inputs_both[0, last_idx, :] += (alpha * mod_tensor).to(model.dtype)
    with torch.no_grad():
        both_logits = model(inputs_embeds=inputs_both).logits[0, last_idx, :]
        both_probs = torch.softmax(both_logits, dim=-1)
    
    # 计算KL散度
    kl_noun = torch.nn.functional.kl_div(
        torch.log(noun_probs + 1e-10), base_probs + 1e-10, reduction='sum').item()
    kl_mod = torch.nn.functional.kl_div(
        torch.log(mod_probs + 1e-10), base_probs + 1e-10, reduction='sum').item()
    kl_both = torch.nn.functional.kl_div(
        torch.log(both_probs + 1e-10), base_probs + 1e-10, reduction='sum').item()
    
    # 如果两个轴因果独立: KL(both) ≈ KL(noun) + KL(mod)
    # 如果两个轴因果耦合: KL(both) >> KL(noun) + KL(mod) 或 KL(both) << KL(noun) + KL(mod)
    expected_independent = kl_noun + kl_mod
    coupling_ratio = kl_both / max(expected_independent, 1e-10)
    
    print(f"  名词轴KL: {kl_noun:.4f}")
    print(f"  修饰语轴KL: {kl_mod:.4f}")
    print(f"  双轴KL: {kl_both:.4f}")
    print(f"  独立性预期KL(noun+mod): {expected_independent:.4f}")
    print(f"  耦合比(KL_both / KL_noun+KL_mod): {coupling_ratio:.4f}")
    
    if 0.8 < coupling_ratio < 1.2:
        print(f"  ★★★ 两个轴因果独立! (耦合比≈1.0)")
    elif coupling_ratio < 0.8:
        print(f"  ★★ 两个轴因果冗余! (耦合比<0.8, 同时注入效果弱于预期)")
    else:
        print(f"  ★★ 两个轴因果超加性! (耦合比>1.2, 同时注入效果强于预期)")
    
    # Top token变化
    base_top10 = [safe_decode(tokenizer, t.item()) for t in torch.topk(base_logits, 10).indices]
    noun_top10 = [safe_decode(tokenizer, t.item()) for t in torch.topk(noun_logits, 10).indices]
    mod_top10 = [safe_decode(tokenizer, t.item()) for t in torch.topk(mod_logits, 10).indices]
    both_top10 = [safe_decode(tokenizer, t.item()) for t in torch.topk(both_logits, 10).indices]
    
    print(f"\n  Top-5 tokens:")
    print(f"    Base:   {base_top10[:5]}")
    print(f"    +Noun:  {noun_top10[:5]}")
    print(f"    +Mod:   {mod_top10[:5]}")
    print(f"    +Both:  {both_top10[:5]}")
    
    results['noun_axis_results'] = noun_axis_results
    results['modifier_axis_results'] = modifier_axis_results
    results['kl_noun'] = float(kl_noun)
    results['kl_mod'] = float(kl_mod)
    results['kl_both'] = float(kl_both)
    results['coupling_ratio'] = float(coupling_ratio)
    results['axis_orthogonality'] = float(ortho)
    results['base_top5'] = base_top10[:5]
    results['noun_top5'] = noun_top10[:5]
    results['mod_top5'] = mod_top10[:5]
    results['both_top5'] = both_top10[:5]
    
    return results


# ===== Exp4: advmod的第三维: 方式vs时间 =====
def exp4_adverb_third_dimension(model, tokenizer, device):
    """方式副词vs时间副词的第三维分析"""
    print("\n" + "="*70)
    print("Exp4: advmod的第三维: 方式vs时间 ★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Step 1: 收集所有角色 + 方式/时间副词的hidden states
    print("\n  Step 1: 收集hidden states")
    role_names = ["nsubj", "poss", "dobj", "amod", "advmod", "pobj"]
    
    all_h = []
    all_labels = []
    all_types = []
    
    # 六类语法角色
    for role_idx, role in enumerate(role_names):
        data = MANIFOLD_ROLES_DATA[role]
        H = collect_hs_at_layer(model, tokenizer, device,
                                data["sentences"], data["target_words"], -1)
        if H is not None and len(H) > 0:
            all_h.append(H)
            all_labels.extend([role_idx] * len(H))
            all_types.extend([role] * len(H))
            print(f"  {role}: {len(H)} samples")
    
    # 方式副词
    manner_data = ADVERB_TYPE_DATA["manner"]
    H_manner = collect_hs_at_layer(model, tokenizer, device,
                                   manner_data["sentences"], manner_data["target_words"], -1)
    if H_manner is not None and len(H_manner) > 0:
        all_h.append(H_manner)
        all_labels.extend([6] * len(H_manner))  # 6 = manner
        all_types.extend(["manner"] * len(H_manner))
        print(f"  manner: {len(H_manner)} samples")
    
    # 时间副词
    temporal_data = ADVERB_TYPE_DATA["temporal"]
    H_temporal = collect_hs_at_layer(model, tokenizer, device,
                                     temporal_data["sentences"], temporal_data["target_words"], -1)
    if H_temporal is not None and len(H_temporal) > 0:
        all_h.append(H_temporal)
        all_labels.extend([7] * len(H_temporal))  # 7 = temporal
        all_types.extend(["temporal"] * len(H_temporal))
        print(f"  temporal: {len(H_temporal)} samples")
    
    if len(all_h) < 3:
        print("  样本不足!")
        return results
    
    H_all = np.vstack(all_h)
    labels = np.array(all_labels)
    
    # Step 2: PCA 3D分析
    print("\n  Step 2: PCA 3D分析")
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H_all)
    pca3 = PCA(n_components=3)
    H_3d = pca3.fit_transform(H_scaled)
    print(f"  3D方差保留: {sum(pca3.explained_variance_ratio_):.4f}")
    
    # 各类中心
    all_type_names = role_names + ["manner", "temporal"]
    centers_3d = {}
    for type_idx, type_name in enumerate(all_type_names):
        mask = labels == type_idx
        if np.sum(mask) > 0:
            centers_3d[type_name] = np.mean(H_3d[mask], axis=0)
    
    # Step 3: 二维平面重建(名词轴+修饰语轴)
    print("\n  Step 3: 二维平面拟合")
    
    if 'nsubj' in centers_3d and 'dobj' in centers_3d and 'amod' in centers_3d:
        # 名词轴
        noun_axis = centers_3d['dobj'] - centers_3d['nsubj']
        noun_norm = np.linalg.norm(noun_axis)
        if noun_norm > 0:
            noun_axis = noun_axis / noun_norm
        
        # 修饰语轴(正交于名词轴)
        amod_vec = centers_3d['amod'] - centers_3d['nsubj']
        amod_proj = np.dot(amod_vec, noun_axis) * noun_axis
        modifier_axis = amod_vec - amod_proj
        mod_norm = np.linalg.norm(modifier_axis)
        if mod_norm > 0:
            modifier_axis = modifier_axis / mod_norm
        
        # 第三轴 = 名词轴 × 修饰语轴
        third_axis = np.cross(noun_axis, modifier_axis)
        third_norm = np.linalg.norm(third_axis)
        if third_norm > 0:
            third_axis = third_axis / third_norm
        
        print(f"  名词轴: {noun_axis}")
        print(f"  修饰语轴: {modifier_axis}")
        print(f"  第三轴: {third_axis}")
        
        # 各角色在三个轴上的投影
        print(f"\n  各类型在三轴上的投影:")
        for type_name in all_type_names:
            if type_name in centers_3d:
                vec = centers_3d[type_name] - centers_3d['nsubj']
                proj1 = np.dot(vec, noun_axis)
                proj2 = np.dot(vec, modifier_axis)
                proj3 = np.dot(vec, third_axis)
                residual = np.linalg.norm(vec - proj1*noun_axis - proj2*modifier_axis - proj3*third_axis)
                print(f"    {type_name}: noun={proj1:.4f}, mod={proj2:.4f}, "
                      f"third={proj3:.4f}, residual={residual:.4f}")
                results[f'{type_name}_noun_proj'] = float(proj1)
                results[f'{type_name}_mod_proj'] = float(proj2)
                results[f'{type_name}_third_proj'] = float(proj3)
        
        # 关键: advmod vs manner vs temporal 在第三维上的比较
        print(f"\n  ★ advmod vs manner vs temporal 第三维分析:")
        for type_name in ["advmod", "manner", "temporal"]:
            if type_name in results:
                print(f"    {type_name} 第三维投影: {results[f'{type_name}_third_proj']:.4f}")
        
        # 方式副词 vs 时间副词在第三维上的区分
        if 'manner' in centers_3d and 'temporal' in centers_3d:
            manner_vec = centers_3d['manner'] - centers_3d['nsubj']
            temporal_vec = centers_3d['temporal'] - centers_3d['nsubj']
            
            manner_third = np.dot(manner_vec, third_axis)
            temporal_third = np.dot(temporal_vec, third_axis)
            
            # 方式vs时间的第三维距离
            third_dim_separation = abs(manner_third - temporal_third)
            
            # 方式vs时间在二维平面上的距离
            manner_2d = np.dot(manner_vec, noun_axis)*noun_axis + np.dot(manner_vec, modifier_axis)*modifier_axis
            temporal_2d = np.dot(temporal_vec, noun_axis)*noun_axis + np.dot(temporal_vec, modifier_axis)*modifier_axis
            planar_separation = np.linalg.norm(manner_2d - temporal_2d)
            
            print(f"\n  方式-时间在第三维的距离: {third_dim_separation:.4f}")
            print(f"  方式-时间在二维平面的距离: {planar_separation:.4f}")
            print(f"  第三维/平面距离比: {third_dim_separation/max(planar_separation, 1e-10):.4f}")
            
            results['manner_temporal_third_dim_sep'] = float(third_dim_separation)
            results['manner_temporal_planar_sep'] = float(planar_separation)
            results['third_to_planar_ratio'] = float(third_dim_separation / max(planar_separation, 1e-10))
    
    # Step 4: 方式vs时间的线性可分性
    print("\n  Step 4: 方式vs时间的线性可分性")
    
    manner_mask = labels == 6
    temporal_mask = labels == 7
    
    if np.sum(manner_mask) >= 5 and np.sum(temporal_mask) >= 5:
        # 在全空间中的可分性
        H_mt = H_scaled[manner_mask | temporal_mask]
        labels_mt = labels[manner_mask | temporal_mask]
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv = cross_val_score(probe, H_mt, labels_mt, cv=min(5, min(np.sum(manner_mask), np.sum(temporal_mask))), scoring='accuracy')
        print(f"  全空间 方式vs时间 CV: {cv.mean():.4f}")
        results['manner_temporal_full_cv'] = float(cv.mean())
        
        # 在第三维上的可分性
        H_mt_3d = H_3d[manner_mask | temporal_mask]
        labels_mt = labels[manner_mask | temporal_mask]
        
        # 只用第三维
        if 'third_axis' in dir():
            # 计算各样本在第三维上的投影
            third_proj_all = np.dot(H_mt_3d, third_axis).reshape(-1, 1)
            probe_1d = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            cv_1d = cross_val_score(probe_1d, third_proj_all, labels_mt, 
                                    cv=min(5, min(np.sum(manner_mask), np.sum(temporal_mask))), scoring='accuracy')
            print(f"  仅第三维 方式vs时间 CV: {cv_1d.mean():.4f}")
            results['manner_temporal_third_only_cv'] = float(cv_1d.mean())
        
        # 只用前两维(名词+修饰语)
        H_mt_2d = H_mt_3d[:, :2]
        probe_2d = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_2d = cross_val_score(probe_2d, H_mt_2d, labels_mt,
                                cv=min(5, min(np.sum(manner_mask), np.sum(temporal_mask))), scoring='accuracy')
        print(f"  仅前两维 方式vs时间 CV: {cv_2d.mean():.4f}")
        results['manner_temporal_2d_cv'] = float(cv_2d.mean())
    
    results['pca3_variance'] = float(sum(pca3.explained_variance_ratio_))
    
    return results


# ===== 主函数 =====
def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3, 4])
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"CCL-W Phase17 语法方向深层机制+论元修饰因果验证 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    try:
        if args.exp == 1:
            results = exp1_layer_propagation(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_syntax_head_function(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_argument_modifier_causal(model, tokenizer, device)
        elif args.exp == 4:
            results = exp4_adverb_third_dimension(model, tokenizer, device)

        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                               f"cclw_exp{args.exp}_{args.model}_results.json")

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, tuple):
                return list(convert(v) for v in obj)
            return obj

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {out_path}")

    finally:
        release_model(model)
        print(f"  模型已释放")


if __name__ == "__main__":
    main()
