"""
CCL-V(Phase 16): 注意力回传的数学结构与语法编码机制
=============================================================================
核心问题(基于Phase 15的发现):
  1. 哪些注意力头负责nsubj/dobj/amod方向?
     → Phase 15发现语法角色有统一3D几何
     → 注意力回传决定语法角色编码
     → 需要找到具体是哪些头在编码语法方向
  
  2. 语法角色方向是否因果地影响模型行为?
     → 在embedding中注入nsubj方向, 预测是否改变?
     → 将dobj方向的表示转向nsubj, 输出是否变化?
  
  3. 跨语言(nsubj-poss)等价是否普遍?
     → 中文没有's, nsubj-poss是否仍然等价?
     → 不同语言的语法几何是否有共性?
  
  4. 语法角色几何是否可以用参数化模型描述?
     → 用正交投影+旋转模型拟合6类角色
     → 参数是否与模型性能相关?

实验:
  Exp1: ★★★★★★★★★ 注意力头与语法角色方向
    → 在语法最强层收集各注意力头的输出
    → 测量每个头在nsubj-dobj方向上的投影
    → 识别"语法头"和"非语法头"

  Exp2: ★★★★★★★★★ 语法角色方向的因果干预
    → 在embedding中注入nsubj/dobj方向
    → 测量模型预测的变化
    → 验证语法方向是否因果地影响行为

  Exp3: ★★★★★★★ 跨语言的语法几何比较
    → 中文句子: nsubj vs poss vs dobj
    → 中文没有's, nsubj-poss是否仍然等价?
    → 英文 vs 中文的几何比较

  Exp4: ★★★★★ 语法角色几何的参数化模型
    → 用正交投影+旋转模型拟合6类角色
    → 参数: nsubj-dobj角度, amod偏移角, pobj偏移量
    → 三模型的参数比较
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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== 复用Phase 15的数据 =====
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

# ===== Exp3: 中文数据 =====
CHINESE_ROLES_DATA = {
    "nsubj": {
        "sentences": [
            "国王统治着王国",
            "医生治疗了病人",
            "艺术家画了肖像",
            "士兵保卫了城堡",
            "老师解释了课程",
            "厨师做了饭菜",
            "猫追赶着老鼠",
            "狗找到了骨头",
            "女人开着车",
            "男人修了屋顶",
            "学生读了书",
            "歌手唱了歌",
            "面包师做了面包",
            "飞行员驾驶飞机",
            "农民种了庄稼",
            "作家写了小说",
        ],
        "target_words": [
            "国王", "医生", "艺术家", "士兵", "老师",
            "厨师", "猫", "狗", "女人", "男人",
            "学生", "歌手", "面包师", "飞行员", "农民", "作家",
        ],
    },
    "poss": {
        "sentences": [
            "国王的王冠闪闪发光",
            "医生的诊所很早就开了",
            "艺术家的工作室很漂亮",
            "士兵的制服很干净",
            "老师的书卖得很快",
            "厨师的餐厅今天开业",
            "猫的尾巴轻轻摆动",
            "狗的叫声很响亮",
            "女人的裙子很优雅",
            "男人的车开得很快",
            "学生的文章写得很好",
            "歌手的声音很清澈",
            "面包师的店很香",
            "飞行员的执照更新了",
            "农民的土地很肥沃",
            "作家的笔写得很流畅",
        ],
        "target_words": [
            "国王", "医生", "艺术家", "士兵", "老师",
            "厨师", "猫", "狗", "女人", "男人",
            "学生", "歌手", "面包师", "飞行员", "农民", "作家",
        ],
    },
    "dobj": {
        "sentences": [
            "他们加冕了国王",
            "她拜访了医生",
            "他钦佩艺术家",
            "我们尊敬士兵",
            "你感谢老师",
            "顾客打赏了厨师",
            "老鹰追赶猫",
            "男孩找到了狗",
            "警察逮捕了女人",
            "公司雇用了男人",
            "我表扬了学生",
            "他们为歌手鼓掌",
            "她光顾了面包师",
            "他钦佩飞行员",
            "我们感谢农民",
            "编辑赞扬了作家",
        ],
        "target_words": [
            "国王", "医生", "艺术家", "士兵", "老师",
            "厨师", "猫", "狗", "女人", "男人",
            "学生", "歌手", "面包师", "飞行员", "农民", "作家",
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


# ===== Exp1: 注意力头与语法角色方向 =====
def exp1_attention_heads_syntax(model, tokenizer, device):
    """分析各注意力头对语法角色方向的贡献"""
    print("\n" + "="*70)
    print("Exp1: 注意力头与语法角色方向 ★★★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)

    # Step 1: 获取语法角色方向
    print("\n  Step 1: 获取语法角色方向")
    role_names = ["nsubj", "poss", "dobj", "amod", "advmod", "pobj"]
    
    # 在语法最强层收集hidden states
    # Phase 15发现: Qwen3 L20, GLM4 L5, DS7B L4
    # 但统一用L5作为"语法强层"
    if args.model == 'qwen3':
        syntax_layer = min(20, n_layers - 1)
    elif args.model == 'glm4':
        syntax_layer = min(5, n_layers - 1)
    else:
        syntax_layer = min(4, n_layers - 1)
    
    role_h = {}
    for role in role_names:
        data = MANIFOLD_ROLES_DATA[role]
        H = collect_hs_at_layer(model, tokenizer, device,
                                data["sentences"], data["target_words"], syntax_layer)
        if H is not None and len(H) > 0:
            role_h[role] = H
            print(f"  {role}: {len(H)} samples at L{syntax_layer}")

    # 计算各角色中心
    centers = {}
    for role in role_names:
        if role in role_h:
            centers[role] = np.mean(role_h[role], axis=0)

    # 语法方向: nsubj→dobj方向
    if 'nsubj' in centers and 'dobj' in centers:
        nsubj_dobj_dir = centers['dobj'] - centers['nsubj']
        nsubj_dobj_norm = np.linalg.norm(nsubj_dobj_dir)
        if nsubj_dobj_norm > 0:
            nsubj_dobj_dir = nsubj_dobj_dir / nsubj_dobj_norm
        print(f"  nsubj→dobj方向范数: {nsubj_dobj_norm:.4f}")
        results['nsubj_dobj_dir_norm'] = float(nsubj_dobj_norm)

    # Step 2: 各注意力头的语法贡献
    print("\n  Step 2: 各注意力头在语法方向上的投影")
    # 对于语法最强层的每个句子, 收集各注意力头的输出
    # 使用hook获取每个头的attn_output
    
    # 采样3个nsubj句子和3个dobj句子
    nsubj_sents = MANIFOLD_ROLES_DATA["nsubj"]["sentences"][:6]
    nsubj_words = MANIFOLD_ROLES_DATA["nsubj"]["target_words"][:6]
    dobj_sents = MANIFOLD_ROLES_DATA["dobj"]["sentences"][:6]
    dobj_words = MANIFOLD_ROLES_DATA["dobj"]["target_words"][:6]
    
    target_layer = layers[syntax_layer]
    
    # 获取n_heads
    sa = target_layer.self_attn
    if hasattr(sa, 'num_heads'):
        n_heads = sa.num_heads
    elif hasattr(sa, 'num_attention_heads'):
        n_heads = sa.num_attention_heads
    else:
        n_heads = d_model // 64  # 假设head_dim=64
    head_dim = d_model // n_heads
    print(f"  n_heads={n_heads}, head_dim={head_dim}")

    # 收集各头的attn_output在target token位置
    head_projections = {f"head_{h}": [] for h in range(n_heads)}
    head_labels = []

    all_sents = list(zip(nsubj_sents, nsubj_words, ["nsubj"]*6)) + \
                list(zip(dobj_sents, dobj_words, ["dobj"]*6))

    for sent, tw, role in all_sents:
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens_list, tw)
        if dep_idx is None:
            continue

        # Hook: 获取self_attn的输出, 然后手动拆分为各头
        captured_attn = {}
        def attn_hook(module, input, output):
            # output是(attn_output, attn_weights, ...) 或 (attn_output, ...)
            # attn_output: [batch, seq_len, d_model]
            if isinstance(output, tuple):
                captured_attn['out'] = output[0].detach().float().cpu()
            else:
                captured_attn['out'] = output.detach().float().cpu()

        handle = sa.register_forward_hook(attn_hook)
        with torch.no_grad():
            _ = model(**toks)
        handle.remove()

        if 'out' not in captured_attn:
            continue

        attn_out = captured_attn['out'][0, dep_idx, :].numpy()  # [d_model]
        # 拆分为各头
        for h in range(n_heads):
            head_out = attn_out[h*head_dim:(h+1)*head_dim]
            if 'nsubj_dobj_dir' in dir():
                # 投影到语法方向上
                head_dir_segment = nsubj_dobj_dir[h*head_dim:(h+1)*head_dim]
                dir_norm = np.linalg.norm(head_dir_segment)
                if dir_norm > 0:
                    proj = np.dot(head_out, head_dir_segment) / dir_norm
                else:
                    proj = 0.0
            else:
                proj = 0.0
            head_projections[f"head_{h}"].append(proj)
        head_labels.append(role)

    # 分析各头的语法区分力
    print(f"\n  各注意力头在nsubj-dobj方向上的区分力:")
    nsubj_proj = {f"head_{h}": [] for h in range(n_heads)}
    dobj_proj = {f"head_{h}": [] for h in range(n_heads)}
    
    for i, role in enumerate(head_labels):
        for h in range(n_heads):
            if i < len(head_projections[f"head_{h}"]):
                if role == "nsubj":
                    nsubj_proj[f"head_{h}"].append(head_projections[f"head_{h}"][i])
                else:
                    dobj_proj[f"head_{h}"].append(head_projections[f"head_{h}"][i])

    syntax_heads = []
    for h in range(n_heads):
        ns_vals = nsubj_proj[f"head_{h}"]
        do_vals = dobj_proj[f"head_{h}"]
        if len(ns_vals) >= 2 and len(do_vals) >= 2:
            ns_mean = np.mean(ns_vals)
            do_mean = np.mean(do_vals)
            separation = abs(do_mean - ns_mean)
            # 用t检验简化: 均值差/标准差
            ns_std = max(np.std(ns_vals), 1e-10)
            do_std = max(np.std(do_vals), 1e-10)
            pooled_std = np.sqrt((ns_std**2 + do_std**2) / 2)
            effect_size = separation / pooled_std if pooled_std > 0 else 0
            
            if h < 10 or effect_size > 0.5:  # 打印前10个或显著的
                print(f"    Head {h}: nsubj_mean={ns_mean:.4f}, dobj_mean={do_mean:.4f}, "
                      f"sep={separation:.4f}, d={effect_size:.3f}")
            
            results[f'head_{h}_nsubj_mean'] = float(ns_mean)
            results[f'head_{h}_dobj_mean'] = float(do_mean)
            results[f'head_{h}_separation'] = float(separation)
            results[f'head_{h}_effect_size'] = float(effect_size)
            
            if effect_size > 1.0:  # 大效应
                syntax_heads.append((h, effect_size, separation))

    syntax_heads.sort(key=lambda x: -x[1])
    print(f"\n  语法头(effect_size>1.0): {len(syntax_heads)}")
    for h, d, sep in syntax_heads[:10]:
        print(f"    Head {h}: d={d:.3f}, separation={sep:.4f}")
    results['syntax_heads'] = [(h, float(d), float(sep)) for h, d, sep in syntax_heads]
    results['n_heads'] = n_heads
    results['head_dim'] = head_dim

    # Step 3: 多层语法头分布
    print(f"\n  Step 3: 多层语法头分布")
    sample_layers = [0, syntax_layer, n_layers - 1]
    if n_layers > 10:
        sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    layer_syntax_strength = {}
    for layer_idx in sample_layers:
        target_l = layers[layer_idx]
        sa_l = target_l.self_attn
        
        # 简化: 只用3个句子, 快速评估
        test_sents = MANIFOLD_ROLES_DATA["nsubj"]["sentences"][:3]
        test_words = MANIFOLD_ROLES_DATA["nsubj"]["target_words"][:3]
        test_dobj_sents = MANIFOLD_ROLES_DATA["dobj"]["sentences"][:3]
        test_dobj_words = MANIFOLD_ROLES_DATA["dobj"]["target_words"][:3]
        
        layer_ns = {f"head_{h}": [] for h in range(n_heads)}
        layer_do = {f"head_{h}": [] for h in range(n_heads)}
        
        for sents, words, role in [(test_sents, test_words, "nsubj"), 
                                     (test_dobj_sents, test_dobj_words, "dobj")]:
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
                
                h_handle = target_l.register_forward_hook(hook_fn)
                with torch.no_grad():
                    _ = model(**toks)
                h_handle.remove()
                
                if 'h' not in captured:
                    continue
                
                h_vec = captured['h'][0, dep_idx, :]
                for hh in range(n_heads):
                    head_seg = h_vec[hh*head_dim:(hh+1)*head_dim]
                    if role == "nsubj":
                        layer_ns[f"head_{hh}"].append(head_seg)
                    else:
                        layer_do[f"head_{hh}"].append(head_seg)
        
        # 计算每层的总语法区分力
        total_syntax_strength = 0
        for hh in range(n_heads):
            ns_arr = np.array(layer_ns[f"head_{hh}"])
            do_arr = np.array(layer_do[f"head_{hh}"])
            if len(ns_arr) >= 1 and len(do_arr) >= 1:
                ns_center = np.mean(ns_arr, axis=0)
                do_center = np.mean(do_arr, axis=0)
                sep = np.linalg.norm(do_center - ns_center)
                total_syntax_strength += sep
        
        layer_syntax_strength[layer_idx] = float(total_syntax_strength)
        print(f"    L{layer_idx}: 总语法区分力={total_syntax_strength:.4f}")
    
    results['layer_syntax_strength'] = layer_syntax_strength

    # Step 4: 注意力模式分析
    print(f"\n  Step 4: 语法头关注哪些token?")
    # 对syntax_heads中top-3头, 查看注意力权重
    if len(syntax_heads) >= 1:
        top_heads = [h for h, _, _ in syntax_heads[:3]]
        
        for head_idx in top_heads:
            print(f"\n    Head {head_idx} 注意力模式:")
            
            for role, sents_key in [("nsubj", "nsubj"), ("dobj", "dobj")]:
                sent = MANIFOLD_ROLES_DATA[sents_key]["sentences"][0]
                tw = MANIFOLD_ROLES_DATA[sents_key]["target_words"][0]
                
                toks = tokenizer(sent, return_tensors="pt").to(device)
                input_ids = toks.input_ids
                tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
                dep_idx = find_token_index(tokens_list, tw)
                if dep_idx is None:
                    continue
                
                # 需要获取注意力权重
                captured_attn_w = {}
                def attn_weight_hook(module, input, output):
                    # output可能包含(attn_output, attn_weights, past_key_value)
                    if len(output) >= 2 and isinstance(output[1], torch.Tensor):
                        captured_attn_w['weights'] = output[1].detach().float().cpu()
                
                handle = sa_l.register_forward_hook(attn_weight_hook)
                with torch.no_grad():
                    _ = model(**toks)
                handle.remove()
                
                if 'weights' in captured_attn_w:
                    attn_w = captured_attn_w['weights']  # [batch, n_heads, seq, seq]
                    # 提取目标头的注意力
                    head_attn = attn_w[0, head_idx, dep_idx, :].numpy()
                    top_k = min(5, len(tokens_list))
                    top_indices = np.argsort(head_attn)[-top_k:][::-1]
                    top_tokens = [(tokens_list[i], float(head_attn[i])) for i in top_indices]
                    print(f"      {role}({sent}): target={tw}@pos{dep_idx}")
                    print(f"        Top attention: {top_tokens}")
                    results[f'head{head_idx}_{role}_attn'] = top_tokens
                else:
                    print(f"      {role}: 无法获取注意力权重")

    results['syntax_layer'] = syntax_layer
    return results


# ===== Exp2: 语法角色方向的因果干预 =====
def exp2_causal_intervention(model, tokenizer, device):
    """语法角色方向的因果干预实验"""
    print("\n" + "="*70)
    print("Exp2: 语法角色方向的因果干预 ★★★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)

    # Step 1: 获取语法方向
    print("\n  Step 1: 获取语法方向(在embedding层)")
    
    # 使用Phase 15的发现: nsubj和dobj的差就是语法方向
    # 在L0(embedding后)获取方向
    role_h_l0 = {}
    for role in ["nsubj", "dobj", "amod"]:
        data = MANIFOLD_ROLES_DATA[role]
        H = collect_hs_at_layer(model, tokenizer, device,
                                data["sentences"], data["target_words"], 0)
        if H is not None:
            role_h_l0[role] = H
            print(f"  {role} at L0: {len(H)} samples")

    if 'nsubj' not in role_h_l0 or 'dobj' not in role_h_l0:
        print("  无法获取语法方向!")
        return results

    nsubj_center = np.mean(role_h_l0['nsubj'], axis=0)
    dobj_center = np.mean(role_h_l0['dobj'], axis=0)
    
    # nsubj→dobj方向 (归一化)
    nsubj_dobj_vec = dobj_center - nsubj_center
    nsubj_dobj_norm = np.linalg.norm(nsubj_dobj_vec)
    if nsubj_dobj_norm > 0:
        nsubj_dobj_dir = nsubj_dobj_vec / nsubj_dobj_norm
    else:
        print("  nsubj-dobj方向范数为0!")
        return results
    
    print(f"  nsubj-dobj方向范数: {nsubj_dobj_norm:.4f}")

    # Step 2: 注入nsubj方向到dobj位置的embedding
    print("\n  Step 2: 注入nsubj方向干预")
    
    # 测试句子: "They crowned the king yesterday" (king是dobj)
    # 注入nsubj方向: 应该让模型更倾向于把king理解为nsubj
    
    test_pairs = [
        # (dobj句子, target词, 期望nsubj句子)
        ("They crowned the king yesterday", "king", "The king ruled"),
        ("She visited the doctor recently", "doctor", "The doctor treated"),
        ("The hawk chased the cat swiftly", "cat", "The cat chased"),
    ]
    
    intervention_results = []
    
    for sent, target_word, expected_pattern in test_pairs:
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens_list, target_word)
        if dep_idx is None:
            continue
        
        # 获取embedding
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
        
        # 基线预测
        with torch.no_grad():
            base_logits = model(inputs_embeds=inputs_embeds).logits[0, dep_idx, :]
            base_top5 = torch.topk(base_logits, 5)
            base_tokens = [safe_decode(tokenizer, t.item()) for t in base_top5.indices]
            base_probs = torch.softmax(base_logits, dim=-1)
        
        # 干预: 在target token位置减去nsubj→dobj方向(使其更像nsubj)
        # 即: h' = h - alpha * nsubj_dobj_dir (让表示向nsubj方向移动)
        for alpha in [0.5, 1.0, 2.0, 4.0]:
            inputs_intervened = inputs_embeds.clone()
            direction_tensor = torch.tensor(nsubj_dobj_dir, dtype=inputs_embeds.dtype, device=device)
            # 减去方向 = 向nsubj方向移动
            inputs_intervened[0, dep_idx, :] -= (alpha * direction_tensor).to(model.dtype)
            
            with torch.no_grad():
                int_logits = model(inputs_embeds=inputs_intervened).logits[0, dep_idx, :]
                int_top5 = torch.topk(int_logits, 5)
                int_tokens = [safe_decode(tokenizer, t.item()) for t in int_top5.indices]
                int_probs = torch.softmax(int_logits, dim=-1)
            
            # 计算KL散度
            kl_div = torch.nn.functional.kl_div(
                torch.log(int_probs + 1e-10), 
                base_probs + 1e-10, 
                reduction='sum'
            ).item()
            
            print(f"  '{target_word}' in '{sent}' alpha={alpha}:")
            print(f"    Base top5: {base_tokens}")
            print(f"    Intervention top5: {int_tokens}")
            print(f"    KL divergence: {kl_div:.4f}")
            
            intervention_results.append({
                'sentence': sent,
                'target': target_word,
                'alpha': alpha,
                'base_top5': base_tokens,
                'int_top5': int_tokens,
                'kl_divergence': float(kl_div),
            })

    # Step 3: 反向干预 - 在nsubj位置注入dobj方向
    print("\n  Step 3: 注入dobj方向到nsubj位置")
    
    test_pairs_nsubj = [
        ("The king ruled the kingdom wisely", "king"),
        ("The doctor treated the patient carefully", "doctor"),
        ("The cat chased the mouse quickly", "cat"),
    ]
    
    for sent, target_word in test_pairs_nsubj:
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens_list, target_word)
        if dep_idx is None:
            continue
        
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
        
        with torch.no_grad():
            base_logits = model(inputs_embeds=inputs_embeds).logits[0, dep_idx, :]
            base_top5 = torch.topk(base_logits, 5)
            base_tokens = [safe_decode(tokenizer, t.item()) for t in base_top5.indices]
            base_probs = torch.softmax(base_logits, dim=-1)
        
        for alpha in [0.5, 1.0, 2.0, 4.0]:
            inputs_intervened = inputs_embeds.clone()
            direction_tensor = torch.tensor(nsubj_dobj_dir, dtype=inputs_embeds.dtype, device=device)
            # 加上方向 = 向dobj方向移动
            inputs_intervened[0, dep_idx, :] += (alpha * direction_tensor).to(model.dtype)
            
            with torch.no_grad():
                int_logits = model(inputs_embeds=inputs_intervened).logits[0, dep_idx, :]
                int_top5 = torch.topk(int_logits, 5)
                int_tokens = [safe_decode(tokenizer, t.item()) for t in int_top5.indices]
                int_probs = torch.softmax(int_logits, dim=-1)
            
            kl_div = torch.nn.functional.kl_div(
                torch.log(int_probs + 1e-10), 
                base_probs + 1e-10, 
                reduction='sum'
            ).item()
            
            print(f"  '{target_word}' in '{sent}' alpha={alpha}:")
            print(f"    Base top5: {base_tokens}")
            print(f"    Intervention top5: {int_tokens}")
            print(f"    KL divergence: {kl_div:.4f}")
            
            intervention_results.append({
                'sentence': sent,
                'target': target_word,
                'direction': 'nsubj->dobj',
                'alpha': alpha,
                'base_top5': base_tokens,
                'int_top5': int_tokens,
                'kl_divergence': float(kl_div),
            })

    # Step 4: 随机方向对照
    print("\n  Step 4: 随机方向对照")
    np.random.seed(42)
    random_dir = np.random.randn(d_model)
    random_dir = random_dir / np.linalg.norm(random_dir)
    
    sent = "They crowned the king yesterday"
    tw = "king"
    toks = tokenizer(sent, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
    dep_idx = find_token_index(tokens_list, tw)
    
    if dep_idx is not None:
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).detach().clone().to(model.dtype)
        
        with torch.no_grad():
            base_logits = model(inputs_embeds=inputs_embeds).logits[0, dep_idx, :]
            base_probs = torch.softmax(base_logits, dim=-1)
        
        for alpha in [0.5, 1.0, 2.0, 4.0]:
            inputs_intervened = inputs_embeds.clone()
            random_tensor = torch.tensor(random_dir, dtype=inputs_embeds.dtype, device=device)
            inputs_intervened[0, dep_idx, :] += (alpha * random_tensor).to(model.dtype)
            
            with torch.no_grad():
                int_logits = model(inputs_embeds=inputs_intervened).logits[0, dep_idx, :]
                int_probs = torch.softmax(int_logits, dim=-1)
            
            kl_div = torch.nn.functional.kl_div(
                torch.log(int_probs + 1e-10), 
                base_probs + 1e-10, 
                reduction='sum'
            ).item()
            
            print(f"  Random direction alpha={alpha}: KL={kl_div:.4f}")
            intervention_results.append({
                'sentence': sent,
                'target': tw,
                'direction': 'random',
                'alpha': alpha,
                'kl_divergence': float(kl_div),
            })
    
    # 比较语法方向 vs 随机方向的KL散度
    syntax_kls = [r['kl_divergence'] for r in intervention_results 
                  if r.get('direction') != 'random' and r['alpha'] == 2.0]
    random_kls = [r['kl_divergence'] for r in intervention_results 
                  if r.get('direction') == 'random' and r['alpha'] == 2.0]
    
    if syntax_kls and random_kls:
        print(f"\n  语法方向平均KL(alpha=2.0): {np.mean(syntax_kls):.4f}")
        print(f"  随机方向平均KL(alpha=2.0): {np.mean(random_kls):.4f}")
        print(f"  语法方向/随机方向: {np.mean(syntax_kls)/max(np.mean(random_kls), 1e-10):.4f}")
        results['syntax_vs_random_kl_ratio'] = float(np.mean(syntax_kls) / max(np.mean(random_kls), 1e-10))

    results['intervention_results'] = intervention_results
    results['nsubj_dobj_dir_norm'] = float(nsubj_dobj_norm)
    return results


# ===== Exp3: 跨语言的语法几何比较 =====
def exp3_cross_language(model, tokenizer, device):
    """跨语言的语法几何比较"""
    print("\n" + "="*70)
    print("Exp3: 跨语言的语法几何比较 ★★★★★★★")
    print("="*70)

    results = {}
    role_names = ["nsubj", "poss", "dobj"]

    # Step 1: 英文语法几何(复用Phase 15的数据)
    print("\n  Step 1: 英文语法几何")
    en_role_h = {}
    for role in role_names:
        data = MANIFOLD_ROLES_DATA[role]
        H = collect_hs_at_layer(model, tokenizer, device,
                                data["sentences"], data["target_words"], -1)
        if H is not None:
            en_role_h[role] = H
            print(f"  EN {role}: {len(H)} samples")

    # Step 2: 中文语法几何
    print("\n  Step 2: 中文语法几何")
    zh_role_h = {}
    for role in role_names:
        data = CHINESE_ROLES_DATA[role]
        H = collect_hs_at_layer(model, tokenizer, device,
                                data["sentences"], data["target_words"], -1)
        if H is not None:
            zh_role_h[role] = H
            print(f"  ZH {role}: {len(H)} samples")

    # Step 3: 英文几何分析
    print("\n  Step 3: 英文3D几何")
    en_centers = {}
    for role in role_names:
        if role in en_role_h:
            en_centers[role] = np.mean(en_role_h[role], axis=0)

    if len(en_centers) >= 3:
        # nsubj-dobj角度
        all_h_en = np.vstack([en_role_h[r] for r in role_names if r in en_role_h])
        labels_en = []
        for r in role_names:
            if r in en_role_h:
                labels_en.extend([role_names.index(r)] * len(en_role_h[r]))
        labels_en = np.array(labels_en)
        
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(all_h_en)
        pca3 = PCA(n_components=3)
        H_3d = pca3.fit_transform(H_scaled)
        
        en_centers_3d = {}
        for ridx, r in enumerate(role_names):
            if r in en_role_h:
                mask = labels_en == ridx
                en_centers_3d[r] = np.mean(H_3d[mask], axis=0)
        
        # 角度
        for r1, r2 in [("nsubj", "poss"), ("nsubj", "dobj"), ("poss", "dobj")]:
            if r1 in en_centers_3d and r2 in en_centers_3d:
                c1 = en_centers_3d[r1]
                c2 = en_centers_3d[r2]
                dist = np.linalg.norm(c1 - c2)
                cos = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10)
                angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
                print(f"    EN {r1}-{r2}: dist={dist:.4f}, cos={cos:.4f}, angle={angle:.1f}°")
                results[f'en_{r1}_{r2}_dist'] = float(dist)
                results[f'en_{r1}_{r2}_cos'] = float(cos)
                results[f'en_{r1}_{r2}_angle'] = float(angle)
        
        # nsubj vs poss CV
        nsubj_idx = role_names.index('nsubj')
        poss_idx = role_names.index('poss')
        mask_np = (labels_en == nsubj_idx) | (labels_en == poss_idx)
        H_np = H_scaled[mask_np]
        labels_np = labels_en[mask_np]
        if len(H_np) >= 10:
            probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            cv = cross_val_score(probe, H_np, labels_np, cv=5, scoring='accuracy')
            print(f"    EN nsubj vs poss CV: {cv.mean():.4f}")
            results['en_nsubj_vs_poss_cv'] = float(cv.mean())

    # Step 4: 中文几何分析
    print("\n  Step 4: 中文3D几何")
    zh_centers = {}
    for role in role_names:
        if role in zh_role_h:
            zh_centers[role] = np.mean(zh_role_h[role], axis=0)

    if len(zh_centers) >= 3:
        all_h_zh = np.vstack([zh_role_h[r] for r in role_names if r in zh_role_h])
        labels_zh = []
        for r in role_names:
            if r in zh_role_h:
                labels_zh.extend([role_names.index(r)] * len(zh_role_h[r]))
        labels_zh = np.array(labels_zh)
        
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(all_h_zh)
        pca3 = PCA(n_components=3)
        H_3d = pca3.fit_transform(H_scaled)
        
        zh_centers_3d = {}
        for ridx, r in enumerate(role_names):
            if r in zh_role_h:
                mask = labels_zh == ridx
                zh_centers_3d[r] = np.mean(H_3d[mask], axis=0)
        
        for r1, r2 in [("nsubj", "poss"), ("nsubj", "dobj"), ("poss", "dobj")]:
            if r1 in zh_centers_3d and r2 in zh_centers_3d:
                c1 = zh_centers_3d[r1]
                c2 = zh_centers_3d[r2]
                dist = np.linalg.norm(c1 - c2)
                cos = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10)
                angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
                print(f"    ZH {r1}-{r2}: dist={dist:.4f}, cos={cos:.4f}, angle={angle:.1f}°")
                results[f'zh_{r1}_{r2}_dist'] = float(dist)
                results[f'zh_{r1}_{r2}_cos'] = float(cos)
                results[f'zh_{r1}_{r2}_angle'] = float(angle)
        
        # nsubj vs poss CV
        nsubj_idx = role_names.index('nsubj')
        poss_idx = role_names.index('poss')
        mask_np = (labels_zh == nsubj_idx) | (labels_zh == poss_idx)
        H_np = H_scaled[mask_np]
        labels_np = labels_zh[mask_np]
        if len(H_np) >= 10:
            probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            cv = cross_val_score(probe, H_np, labels_np, cv=5, scoring='accuracy')
            print(f"    ZH nsubj vs poss CV: {cv.mean():.4f}")
            results['zh_nsubj_vs_poss_cv'] = float(cv.mean())

    # Step 5: 跨语言比较
    print("\n  Step 5: 跨语言比较总结")
    if 'en_nsubj_vs_poss_cv' in results and 'zh_nsubj_vs_poss_cv' in results:
        print(f"    EN nsubj-poss CV: {results['en_nsubj_vs_poss_cv']:.4f}")
        print(f"    ZH nsubj-poss CV: {results['zh_nsubj_vs_poss_cv']:.4f}")
        print(f"    差异: {abs(results['en_nsubj_vs_poss_cv'] - results['zh_nsubj_vs_poss_cv']):.4f}")
    
    # 中文没有's, nsubj和poss是否仍然不可分?
    if 'zh_nsubj_vs_poss_cv' in results:
        if results['zh_nsubj_vs_poss_cv'] < 0.6:
            print(f"    ★ 中文nsubj-poss仍然不可分(CV<0.6)!")
            print(f"      → nsubj-poss等价不是英文特有的, 跨语言普遍!")
        else:
            print(f"    ★ 中文nsubj-poss可分(CV={results['zh_nsubj_vs_poss_cv']:.4f})")
            print(f"      → nsubj-poss等价可能与's标记有关")

    return results


# ===== Exp4: 语法角色几何的参数化模型 =====
def exp4_parametric_model(model, tokenizer, device):
    """语法角色几何的参数化模型拟合"""
    print("\n" + "="*70)
    print("Exp4: 语法角色几何的参数化模型 ★★★★★")
    print("="*70)

    results = {}
    role_names = ["nsubj", "poss", "dobj", "amod", "advmod", "pobj"]

    # 收集6类语法角色的hidden states
    all_h = []
    all_labels = []

    for role_idx, role in enumerate(role_names):
        data = MANIFOLD_ROLES_DATA[role]
        H = collect_hs_at_layer(model, tokenizer, device,
                                data["sentences"], data["target_words"], -1)
        if H is not None and len(H) > 0:
            all_h.append(H)
            all_labels.extend([role_idx] * len(H))
            print(f"  {role}: {len(H)} samples")

    if len(all_h) < 6:
        print("  样本不足!")
        return results

    H_all = np.vstack(all_h)
    labels = np.array(all_labels)

    # PCA 3D
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H_all)
    pca3 = PCA(n_components=3)
    H_3d = pca3.fit_transform(H_scaled)
    var_3d = sum(pca3.explained_variance_ratio_)
    print(f"  3D方差保留: {var_3d:.4f}")

    centers_3d = {}
    for role_idx, role in enumerate(role_names):
        mask = labels == role_idx
        centers_3d[role] = np.mean(H_3d[mask], axis=0)

    # 模型1: 一维模型(所有角色在一条线上)
    print("\n  模型1: 一维线性模型")
    # 用nsubj-dobj轴作为唯一轴
    nsubj_dobj_vec = centers_3d['dobj'] - centers_3d['nsubj']
    nsubj_dobj_norm = np.linalg.norm(nsubj_dobj_vec)
    
    if nsubj_dobj_norm > 0:
        axis_dir = nsubj_dobj_vec / nsubj_dobj_norm
        
        # 各角色在轴上的投影
        print(f"    nsubj-dobj轴方向: {axis_dir}")
        for role in role_names:
            vec = centers_3d[role] - centers_3d['nsubj']
            proj = np.dot(vec, axis_dir)
            perp = np.linalg.norm(vec - proj * axis_dir)
            print(f"    {role}: projection={proj:.4f}, perp_dist={perp:.4f}")
            results[f'1d_{role}_proj'] = float(proj)
            results[f'1d_{role}_perp'] = float(perp)
        
        # 一维重建误差
        total_1d_error = 0
        for role in role_names:
            total_1d_error += results[f'1d_{role}_perp'] ** 2
        results['1d_total_error'] = float(total_1d_error)
        print(f"    一维模型总误差: {total_1d_error:.4f}")

    # 模型2: 二维模型(名词轴 + 修饰语轴)
    print("\n  模型2: 二维正交分解模型")
    # 轴1: nsubj-dobj (名词角色轴)
    # 轴2: 垂直于轴1的方向, 由amod/advmod确定
    
    if nsubj_dobj_norm > 0:
        axis1 = axis_dir  # 名词轴
        
        # amod的方向(去掉名词轴分量)
        amod_vec = centers_3d['amod'] - centers_3d['nsubj']
        amod_proj = np.dot(amod_vec, axis1)
        amod_perp = amod_vec - amod_proj * axis1
        amod_perp_norm = np.linalg.norm(amod_perp)
        
        if amod_perp_norm > 0:
            axis2 = amod_perp / amod_perp_norm  # 修饰语轴
            
            # 各角色在两个轴上的投影
            for role in role_names:
                vec = centers_3d[role] - centers_3d['nsubj']
                proj1 = np.dot(vec, axis1)
                proj2 = np.dot(vec, axis2)
                residual = np.linalg.norm(vec - proj1 * axis1 - proj2 * axis2)
                print(f"    {role}: axis1={proj1:.4f}, axis2={proj2:.4f}, residual={residual:.4f}")
                results[f'2d_{role}_axis1'] = float(proj1)
                results[f'2d_{role}_axis2'] = float(proj2)
                results[f'2d_{role}_residual'] = float(residual)
            
            # 二维重建误差
            total_2d_error = 0
            for role in role_names:
                total_2d_error += results[f'2d_{role}_residual'] ** 2
            results['2d_total_error'] = float(total_2d_error)
            print(f"    二维模型总误差: {total_2d_error:.4f}")
            
            # 轴间正交性
            ortho = np.dot(axis1, axis2)
            print(f"    轴1-轴2点积: {ortho:.6f} (应≈0)")
            results['axis_orthogonality'] = float(ortho)

    # 模型3: 参数化模型
    print("\n  模型3: 参数化几何模型")
    # 参数化模型:
    #   nsubj = origin
    #   poss = origin + eps (nsubj-poss等价)
    #   dobj = origin + L * [cos(theta/2), sin(theta/2), 0]
    #        其中theta是nsubj-dobj角度(在2D中)
    #   pobj = origin + L_pobj * [cos(theta/2 + delta), sin(theta/2 + delta), 0]
    #   amod = origin + d_amod * [cos(phi), sin(phi), 0]
    #   advmod = origin + d_advmod * [cos(phi + psi), sin(phi + psi), 0]
    
    # 从数据中提取参数
    if nsubj_dobj_norm > 0 and amod_perp_norm > 0:
        # theta: nsubj-dobj在2D平面上的角度
        # nsubj在原点, dobj在axis1正方向
        # 所以nsubj-dobj方向角 = 0 (by construction)
        # 但amod相对于nsubj的角度:
        amod_angle = np.degrees(np.arctan2(
            np.dot(centers_3d['amod'] - centers_3d['nsubj'], axis2),
            np.dot(centers_3d['amod'] - centers_3d['nsubj'], axis1)
        ))
        
        advmod_angle = np.degrees(np.arctan2(
            np.dot(centers_3d['advmod'] - centers_3d['nsubj'], axis2),
            np.dot(centers_3d['advmod'] - centers_3d['nsubj'], axis1)
        ))
        
        pobj_angle = np.degrees(np.arctan2(
            np.dot(centers_3d['pobj'] - centers_3d['nsubj'], axis2),
            np.dot(centers_3d['pobj'] - centers_3d['nsubj'], axis1)
        ))
        
        dobj_angle = np.degrees(np.arctan2(
            np.dot(centers_3d['dobj'] - centers_3d['nsubj'], axis2),
            np.dot(centers_3d['dobj'] - centers_3d['nsubj'], axis1)
        ))
        
        nsubj_angle = 0.0  # by construction
        poss_angle = np.degrees(np.arctan2(
            np.dot(centers_3d['poss'] - centers_3d['nsubj'], axis2),
            np.dot(centers_3d['poss'] - centers_3d['nsubj'], axis1)
        ))
        
        # 距离参数
        L_nsubj_dobj = nsubj_dobj_norm
        L_nsubj_amod = np.linalg.norm(centers_3d['amod'] - centers_3d['nsubj'])
        L_nsubj_advmod = np.linalg.norm(centers_3d['advmod'] - centers_3d['nsubj'])
        L_nsubj_pobj = np.linalg.norm(centers_3d['pobj'] - centers_3d['nsubj'])
        
        print(f"    参数:")
        print(f"    nsubj角度: {nsubj_angle:.1f}° (原点)")
        print(f"    poss角度: {poss_angle:.1f}°")
        print(f"    dobj角度: {dobj_angle:.1f}°")
        print(f"    pobj角度: {pobj_angle:.1f}°")
        print(f"    amod角度: {amod_angle:.1f}°")
        print(f"    advmod角度: {advmod_angle:.1f}°")
        print(f"    nsubj-dobj距离: {L_nsubj_dobj:.4f}")
        print(f"    nsubj-amod距离: {L_nsubj_amod:.4f}")
        print(f"    nsubj-advmod距离: {L_nsubj_advmod:.4f}")
        print(f"    nsubj-pobj距离: {L_nsubj_pobj:.4f}")
        
        # 关键角度参数
        print(f"\n    关键角度参数:")
        theta_ns_do = dobj_angle - nsubj_angle  # nsubj→dobj
        theta_ns_am = amod_angle - nsubj_angle  # nsubj→amod
        theta_am_adv = advmod_angle - amod_angle  # amod→advmod
        theta_do_pobj = pobj_angle - dobj_angle  # dobj→pobj
        
        print(f"    theta(nsubj→dobj): {theta_ns_do:.1f}°")
        print(f"    theta(nsubj→amod): {theta_ns_am:.1f}°")
        print(f"    theta(amod→advmod): {theta_am_adv:.1f}°")
        print(f"    theta(dobj→pobj): {theta_do_pobj:.1f}°")
        
        results['param_theta_ns_do'] = float(theta_ns_do)
        results['param_theta_ns_am'] = float(theta_ns_am)
        results['param_theta_am_adv'] = float(theta_am_adv)
        results['param_theta_do_pobj'] = float(theta_do_pobj)
        results['param_L_ns_do'] = float(L_nsubj_dobj)
        results['param_L_ns_am'] = float(L_nsubj_amod)
        results['param_L_ns_adv'] = float(L_nsubj_advmod)
        results['param_L_ns_pobj'] = float(L_nsubj_pobj)
        
        # 模型拟合质量
        # 用参数化模型重建中心, 比较与实际中心的误差
        reconstructed = {}
        reconstructed['nsubj'] = centers_3d['nsubj']
        reconstructed['poss'] = centers_3d['nsubj']  # poss = nsubj
        reconstructed['dobj'] = centers_3d['nsubj'] + L_nsubj_dobj * (
            np.cos(np.radians(dobj_angle)) * axis1 + 
            np.sin(np.radians(dobj_angle)) * axis2
        )
        reconstructed['pobj'] = centers_3d['nsubj'] + L_nsubj_pobj * (
            np.cos(np.radians(pobj_angle)) * axis1 + 
            np.sin(np.radians(pobj_angle)) * axis2
        )
        reconstructed['amod'] = centers_3d['nsubj'] + L_nsubj_amod * (
            np.cos(np.radians(amod_angle)) * axis1 + 
            np.sin(np.radians(amod_angle)) * axis2
        )
        reconstructed['advmod'] = centers_3d['nsubj'] + L_nsubj_advmod * (
            np.cos(np.radians(advmod_angle)) * axis1 + 
            np.sin(np.radians(advmod_angle)) * axis2
        )
        
        total_param_error = 0
        for role in role_names:
            error = np.linalg.norm(centers_3d[role] - reconstructed[role])
            total_param_error += error ** 2
            print(f"    {role} 重建误差: {error:.4f}")
        
        results['param_model_error'] = float(total_param_error)
        print(f"    参数化模型总误差: {total_param_error:.4f}")

    # 比较三个模型
    print(f"\n  模型比较:")
    if '1d_total_error' in results:
        print(f"    一维模型误差: {results['1d_total_error']:.4f}")
    if '2d_total_error' in results:
        print(f"    二维模型误差: {results['2d_total_error']:.4f}")
    if 'param_model_error' in results:
        print(f"    参数化模型误差: {results['param_model_error']:.4f}")
    
    # 误差减少率
    if '1d_total_error' in results and '2d_total_error' in results:
        reduction = (results['1d_total_error'] - results['2d_total_error']) / max(results['1d_total_error'], 1e-10)
        print(f"    1D→2D误差减少: {reduction*100:.1f}%")
        results['1d_to_2d_reduction'] = float(reduction)

    results['pca3_variance'] = float(var_3d)
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
    print(f"CCL-V Phase16 注意力回传+语法编码机制 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    try:
        if args.exp == 1:
            results = exp1_attention_heads_syntax(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_causal_intervention(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_cross_language(model, tokenizer, device)
        elif args.exp == 4:
            results = exp4_parametric_model(model, tokenizer, device)

        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                               f"cclv_exp{args.exp}_{args.model}_results.json")

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
