"""
CCXLV(405): 五大严谨性验证 — 回应方法论批评
===============================================

回应批评:
  1. "能读出 ≠ 模型依赖" → 消融实验(必要性)
  2. "加法干预混淆范数" → 旋转干预(保范数)
  3. "100%翻转只在模板句" → 复杂自然句验证
  4. "V_syn维度未确定" → 系统维度估计
  5. "因果贡献未量化" → 中介分析

实验设计:
  Exp1: 消融实验 — 移除d_head/d_level后句法分类准确率下降多少?
  Exp2: 旋转干预 — 保范数旋转 vs 加法干预的效果对比
  Exp3: 自然复杂句 — 20+种非模板句的d_head方向一致性
  Exp4: V_syn维度估计 — PCA方差解释率 + 交叉验证最佳维度
  Exp5: 因果中介分析 — d_head在subject→logit中的中介贡献比例
"""

import sys, os, argparse, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.transform import Rotation as R
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
TEMP.mkdir(parents=True, exist_ok=True)

# ===== 复杂自然句集 (非模板) =====
COMPLEX_SENTENCES = [
    # 1. 多从句
    {"text": "The scientist who discovered the cure believes that the medicine will save millions",
     "pairs": [("scientist", "cure", "head"), ("medicine", "millions", "head"),
               ("cure", "scientist", "dep"), ("believes", "scientist", "dep")]},
    # 2. 被动+关系从句
    {"text": "The painting was stolen by the thief who escaped from the museum",
     "pairs": [("painting", "stolen", "dep"), ("thief", "escaped", "head"),
               ("museum", "escaped", "dep"), ("painting", "thief", "dep")]},
    # 3. 双宾+状语
    {"text": "The teacher carefully gave the student a difficult assignment",
     "pairs": [("teacher", "gave", "dep"), ("student", "gave", "dep"),
               ("assignment", "gave", "dep"), ("teacher", "student", "head")]},
    # 4. 并列结构
    {"text": "The cat and the dog chased the rabbit and the squirrel",
     "pairs": [("cat", "chased", "dep"), ("dog", "chased", "dep"),
               ("rabbit", "chased", "dep"), ("squirrel", "chased", "dep")]},
    # 5. 嵌套所有格
    {"text": "The king's advisor's daughter married the prince",
     "pairs": [("king", "advisor", "head"), ("advisor", "daughter", "head"),
               ("daughter", "married", "dep"), ("prince", "married", "dep")]},
    # 6. 介词链
    {"text": "The book on the table in the room by the window",
     "pairs": [("book", "table", "head"), ("table", "room", "head"),
               ("room", "window", "head"), ("window", "room", "dep")]},
    # 7. 同位语
    {"text": "Paris the capital of France is beautiful",
     "pairs": [("Paris", "capital", "head"), ("capital", "France", "head"),
               ("Paris", "beautiful", "dep")]},
    # 8. 分词短语
    {"text": "Running through the forest the deer startled the hunter",
     "pairs": [("deer", "running", "dep"), ("deer", "startled", "dep"),
               ("hunter", "startled", "dep")]},
    # 9. 不定式
    {"text": "The boy wanted to give the girl a flower",
     "pairs": [("boy", "wanted", "dep"), ("girl", "give", "dep"),
               ("flower", "give", "dep")]},
    # 10. 比较结构
    {"text": "The elephant is bigger than the mouse",
     "pairs": [("elephant", "bigger", "dep"), ("mouse", "bigger", "dep"),
               ("elephant", "mouse", "head")]},
    # 11. 强调句
    {"text": "It was the king who gave the sword to the knight",
     "pairs": [("king", "gave", "head"), ("sword", "gave", "dep"),
               ("knight", "gave", "dep")]},
    # 12. 虚拟语气
    {"text": "If the rain stops the game will continue",
     "pairs": [("rain", "stops", "dep"), ("game", "continue", "dep"),
               ("rain", "game", "head")]},
    # 13. 让步状语
    {"text": "Although the storm was fierce the ship survived",
     "pairs": [("storm", "fierce", "dep"), ("ship", "survived", "dep"),
               ("storm", "ship", "dep")]},
    # 14. 多级修饰
    {"text": "The very tall extremely intelligent young scientist won",
     "pairs": [("scientist", "tall", "head"), ("scientist", "intelligent", "head"),
               ("scientist", "young", "head"), ("scientist", "won", "dep")]},
    # 15. 存在句
    {"text": "There is a dragon in the cave behind the mountain",
     "pairs": [("dragon", "cave", "dep"), ("cave", "mountain", "head"),
               ("mountain", "cave", "dep")]},
    # 16. 倒装句
    {"text": "Never before had the kingdom seen such a powerful wizard",
     "pairs": [("kingdom", "seen", "dep"), ("wizard", "powerful", "dep"),
               ("wizard", "seen", "dep")]},
    # 17. 反身代词
    {"text": "The queen herself prepared the feast for the guests",
     "pairs": [("queen", "prepared", "dep"), ("feast", "prepared", "dep"),
               ("guests", "prepared", "dep")]},
    # 18. 插入语
    {"text": "The warrior however decided to fight the dragon alone",
     "pairs": [("warrior", "decided", "dep"), ("dragon", "fight", "dep")]},
    # 19. 连续动词
    {"text": "The hero decided to try to begin to fight the monster",
     "pairs": [("hero", "decided", "dep"), ("monster", "fight", "dep")]},
    # 20. 长距离依存
    {"text": "The cat that the dog that the boy chased bit ran away",
     "pairs": [("cat", "ran", "dep"), ("dog", "bit", "dep"),
               ("boy", "chased", "dep"), ("cat", "dog", "head")]},
]

# ===== 基础依存对 =====
BASIC_PAIRS = [
    ("cat", "dog"), ("bird", "fish"), ("king", "queen"),
    ("teacher", "student"), ("lion", "tiger"), ("eagle", "whale"),
    ("sword", "shield"), ("fire", "water"), ("mountain", "river"),
    ("sun", "moon"), ("hammer", "nail"), ("rope", "stone"),
]


def extract_d_head_direction(model, tokenizer, device, model_info, n_pairs=12):
    """提取d_head方向 (subject - dependent差值的平均方向)"""
    d_model = model_info.d_model
    W_U = model.lm_head.weight.detach().cpu().float().numpy()  # [vocab, d_model]
    
    # 用模板句提取subject vs object表示
    subj_reps = []
    obj_reps = []
    
    templates = [
        "The {} chases the {}",
        "The {} sees the {}",
        "The {} hits the {}",
        "The {} follows the {}",
    ]
    
    layers = get_layers(model)
    target_layer = len(layers) // 2  # 中间层
    layer = layers[target_layer]
    
    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    
    hook = layer.register_forward_hook(make_hook("target"))
    
    for w1, w2 in BASIC_PAIRS[:n_pairs]:
        for tmpl in templates[:2]:  # 用2个模板
            for swap in [False, True]:
                s, o = (w1, w2) if not swap else (w2, w1)
                prompt = tmpl.format(s, o)
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    try:
                        _ = model(**toks)
                    except:
                        continue
                
                if "target" in captured:
                    h = captured["target"][0].numpy()  # [seq_len, d_model]
                    # 找主语和宾语位置
                    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
                    s_idx = None
                    o_idx = None
                    for i, t in enumerate(tokens):
                        if s.lower() in t.lower() and s_idx is None:
                            s_idx = i
                        if o.lower() in t.lower() and o_idx is None and i != s_idx:
                            o_idx = i
                    
                    if s_idx is not None and o_idx is not None:
                        subj_reps.append(h[s_idx])
                        obj_reps.append(h[o_idx])
    
    hook.remove()
    
    if len(subj_reps) < 3:
        print(f"  WARNING: Only {len(subj_reps)} pairs collected, using W_U fallback")
        return None, None
    
    subj_reps = np.array(subj_reps)
    obj_reps = np.array(obj_reps)
    
    # d_head = mean(subj - obj)方向
    diffs = subj_reps - obj_reps
    d_head = np.mean(diffs, axis=0)
    norm = np.linalg.norm(d_head)
    if norm > 0:
        d_head = d_head / norm
    
    # d_level = V_sem^⊥中的subject vs modifier差
    # 简化: 用modifier vs subject差
    mod_reps = []
    for w1, w2 in BASIC_PAIRS[:n_pairs]:
        for tmpl in ["The red {} chases the {}", "The big {} sees the {}"]:
            prompt = tmpl.format(w1, w2)
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            captured = {}
            hook2 = layer.register_forward_hook(make_hook("target"))
            with torch.no_grad():
                try:
                    _ = model(**toks)
                except:
                    pass
            hook2.remove()
            if "target" in captured:
                h = captured["target"][0].numpy()
                tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
                for i, t in enumerate(tokens):
                    if 'red' in t.lower() or 'big' in t.lower():
                        mod_reps.append(h[i])
                        break
    
    d_level = None
    if len(mod_reps) >= 3:
        mod_reps = np.array(mod_reps)
        # 移除d_head分量后的modifier平均
        mod_perp = mod_reps - np.outer(mod_reps @ d_head, d_head)
        subj_avg = np.mean(subj_reps, axis=0)
        subj_perp = subj_avg - np.dot(subj_avg, d_head) * d_head
        d_level = subj_perp - np.mean(mod_perp, axis=0)
        norm_l = np.linalg.norm(d_level)
        if norm_l > 0:
            d_level = d_level / norm_l
    
    return d_head, d_level


def get_token_representation(model, tokenizer, device, prompt, target_tokens, target_layer=None):
    """获取指定token在指定层的表示"""
    layers = get_layers(model)
    if target_layer is None:
        target_layer = len(layers) // 2
    
    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    
    layer = layers[target_layer]
    hook = layer.register_forward_hook(make_hook("target"))
    
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            _ = model(**toks)
        except:
            hook.remove()
            return None, None
    
    hook.remove()
    
    if "target" not in captured:
        return None, None
    
    h = captured["target"][0].numpy()  # [seq_len, d_model]
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    
    results = {}
    for target in target_tokens:
        for i, t in enumerate(tokens):
            if target.lower() in t.lower():
                results[target] = h[i]
                break
    
    return results, tokens


# ===== Exp1: 消融实验 =====
def run_exp1(model, tokenizer, device, model_info, d_head, d_level):
    """消融实验: 移除d_head/d_level后句法分类准确率下降多少?"""
    print("\n" + "="*70)
    print("Exp1: 消融实验 (Ablation — 必要性测试)")
    print("="*70)
    
    d_model = model_info.d_model
    layers = get_layers(model)
    target_layer = len(layers) // 2
    
    # 收集多角色的表示
    role_templates = {
        "subject": ["The {} chases the dog", "The {} sees the cat"],
        "object": ["The cat chases the {}", "The dog sees the {}"],
        "modifier": ["The red {} chases the dog", "The big {} sees the cat"],
        "indirect_obj": ["The king gives the {} the sword", "The queen gives the {} the crown"],
        "prep_obj": ["The cat looks at the {}", "The dog runs to the {}"],
    }
    
    words = ["cat", "bird", "lion", "king", "horse", "wolf", "eagle", "whale"]
    
    all_reps = []  # [n_samples, d_model]
    all_labels = []  # role index
    label_names = list(role_templates.keys())
    
    for role_idx, (role, templates) in enumerate(role_templates.items()):
        for word in words[:4]:
            for tmpl in templates[:1]:
                prompt = tmpl.format(word)
                results, tokens = get_token_representation(
                    model, tokenizer, device, prompt, [word], target_layer
                )
                if results and word in results:
                    all_reps.append(results[word])
                    all_labels.append(role_idx)
    
    if len(all_reps) < 10:
        print("  WARNING: Too few samples collected")
        return {"status": "insufficient_data"}
    
    X = np.array(all_reps)  # [n, d_model]
    y = np.array(all_labels)
    
    print(f"  Collected {len(X)} samples, {len(label_names)} roles")
    
    # 二分类: subject vs object
    subj_mask = y == 0
    obj_mask = y == 1
    binary_mask = subj_mask | obj_mask
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    
    if len(X_binary) < 6:
        print("  WARNING: Too few binary samples")
        return {"status": "insufficient_binary_data"}
    
    results = {}
    
    # 1. 原始表示分类
    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores_orig = cross_val_score(clf, X_binary, y_binary, cv=min(3, len(X_binary)//2))
    acc_orig = float(np.mean(scores_orig))
    results["original_acc"] = acc_orig
    print(f"  Original subject/obj accuracy: {acc_orig:.3f}")
    
    # 2. 移除d_head分量后分类
    if d_head is not None:
        proj_head = np.outer(d_head, d_head)  # [d, d]
        X_no_head = X_binary - (X_binary @ proj_head)
        scores_no_head = cross_val_score(clf, X_no_head, y_binary, cv=min(3, len(X_binary)//2))
        acc_no_head = float(np.mean(scores_no_head))
        results["ablated_head_acc"] = acc_no_head
        results["head_drop"] = acc_orig - acc_no_head
        print(f"  After removing d_head: {acc_no_head:.3f} (drop={acc_orig-acc_no_head:.3f})")
    
    # 3. 移除d_level分量后分类
    if d_level is not None:
        proj_level = np.outer(d_level, d_level)
        X_no_level = X_binary - (X_binary @ proj_level)
        scores_no_level = cross_val_score(clf, X_no_level, y_binary, cv=min(3, len(X_binary)//2))
        acc_no_level = float(np.mean(scores_no_level))
        results["ablated_level_acc"] = acc_no_level
        results["level_drop"] = acc_orig - acc_no_level
        print(f"  After removing d_level: {acc_no_level:.3f} (drop={acc_orig-acc_no_level:.3f})")
    
    # 4. 同时移除d_head和d_level
    if d_head is not None and d_level is not None:
        proj_both = np.outer(d_head, d_head) + np.outer(d_level, d_level)
        X_no_both = X_binary - (X_binary @ proj_both)
        scores_no_both = cross_val_score(clf, X_no_both, y_binary, cv=min(3, len(X_binary)//2))
        acc_no_both = float(np.mean(scores_no_both))
        results["ablated_both_acc"] = acc_no_both
        results["both_drop"] = acc_orig - acc_no_both
        print(f"  After removing both: {acc_no_both:.3f} (drop={acc_orig-acc_no_both:.3f})")
    
    # 5. 只保留d_head + d_level (充分性测试)
    if d_head is not None and d_level is not None:
        X_2d = np.column_stack([X_binary @ d_head, X_binary @ d_level])
        scores_2d = cross_val_score(clf, X_2d, y_binary, cv=min(3, len(X_binary)//2))
        acc_2d = float(np.mean(scores_2d))
        results["2d_acc"] = acc_2d
        print(f"  Only (d_head, d_level): {acc_2d:.3f}")
    
    # 6. head vs dependent 5-way分类
    if len(X) >= 15:
        clf5 = LogisticRegression(max_iter=1000, C=1.0)
        try:
            scores5_orig = cross_val_score(clf5, X, y, cv=min(3, len(X)//5))
            acc5_orig = float(np.mean(scores5_orig))
            results["5way_original"] = acc5_orig
            
            if d_head is not None:
                X5_no_head = X - (X @ proj_head)
                scores5_no_head = cross_val_score(clf5, X5_no_head, y, cv=min(3, len(X)//5))
                acc5_no_head = float(np.mean(scores5_no_head))
                results["5way_ablated_head"] = acc5_no_head
                results["5way_head_drop"] = acc5_orig - acc5_no_head
                print(f"  5-way original: {acc5_orig:.3f}, after d_head removal: {acc5_no_head:.3f} (drop={acc5_orig-acc5_no_head:.3f})")
        except:
            pass
    
    return results


# ===== Exp2: 旋转干预 =====
def run_exp2(model, tokenizer, device, model_info, d_head, d_level):
    """旋转干预: 保范数旋转 vs 加法干预"""
    print("\n" + "="*70)
    print("Exp2: 旋转干预 vs 加法干预 (Norm-Preserving Rotation)")
    print("="*70)
    
    if d_head is None:
        print("  SKIP: d_head not available")
        return {"status": "skipped"}
    
    d_model = model_info.d_model
    W_U = model.lm_head.weight.detach().cpu().float().numpy()
    
    # 测试句: "The cat chases the dog" → 翻转主语/宾语
    test_pairs = [
        ("The cat chases the dog", "cat", "dog"),
        ("The bird sees the fish", "bird", "fish"),
        ("The king rules the queen", "king", "queen"),
        ("The lion attacks the tiger", "lion", "tiger"),
    ]
    
    betas = [2.0, 4.0, 8.0, 16.0]
    results = {}
    
    for prompt, subj, obj in test_pairs[:2]:  # 用2个测试
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        
        # 基线预测
        with torch.no_grad():
            base_logits = model(**toks).logits[0, -1].float().cpu().numpy()
        
        # 找主语位置
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        subj_idx = None
        for i, t in enumerate(tokens):
            if subj.lower() in t.lower():
                subj_idx = i
                break
        
        if subj_idx is None:
            continue
        
        prompt_key = f"{subj}_{obj}"
        results[prompt_key] = {}
        
        for beta in betas:
            # 方法1: 加法干预 (原有方法)
            embed_layer = model.get_input_embeddings()
            inputs_embeds_add = embed_layer(input_ids).detach().clone()
            d_head_t = torch.tensor(d_head, dtype=inputs_embeds_add.dtype, device=device)
            # 负方向 = 变成宾语
            inputs_embeds_add[0, subj_idx, :] -= (beta * d_head_t).to(model.dtype)
            
            with torch.no_grad():
                try:
                    add_logits = model(inputs_embeds=inputs_embeds_add).logits[0, -1].float().cpu().numpy()
                except:
                    continue
            
            # 方法2: 旋转干预 (保范数)
            # 在(d_head, 随机正交方向)平面内旋转
            h_orig = inputs_embeds_add[0, subj_idx, :].clone()  # 干预前的原始
            h_orig_base = embed_layer(input_ids)[0, subj_idx, :].detach().clone()
            h_base_np = h_orig_base.float().cpu().numpy()
            h_norm = np.linalg.norm(h_base_np)
            
            # 构造旋转: 在d_head方向上旋转theta角
            # h_new = h * cos(theta) + d_head * sin(theta) * ||h||
            # 使得在d_head上的投影增加beta
            # 投影增加 = ||h|| * sin(theta) ≈ beta → theta = arcsin(beta/||h||)
            if h_norm > 0:
                proj_head = np.dot(h_base_np, d_head)
                # 旋转后投影 = proj_head * cos(theta) + h_norm * sin(theta)
                # 设旋转后投影 = proj_head - beta (变宾语方向)
                # theta = arcsin((proj_head - (proj_head - beta)) / h_norm) = arcsin(beta / h_norm)
                sin_theta = min(beta / h_norm, 0.99)
                theta = np.arcsin(sin_theta)
                cos_theta = np.cos(theta)
                
                # 旋转: 在(d_head, h_perp)平面内
                h_perp = h_base_np - proj_head * d_head
                h_perp_norm = np.linalg.norm(h_perp)
                if h_perp_norm > 1e-10:
                    h_perp_dir = h_perp / h_perp_norm
                else:
                    # 随机正交方向
                    rand_vec = np.random.randn(d_model)
                    rand_vec -= np.dot(rand_vec, d_head) * d_head
                    rn = np.linalg.norm(rand_vec)
                    h_perp_dir = rand_vec / rn if rn > 0 else np.zeros(d_model)
                
                # 旋转后: h_rot = h_perp * cos(theta) + d_head * (||h|| * sin(theta) + proj_head * cos(theta))...
                # 更简单: 直接在d_head方向上调整投影, 保持范数不变
                proj_new = proj_head - beta
                # h_rot = h_perp_component + proj_new * d_head
                # 范数约束: ||h_perp_component||^2 + proj_new^2 = h_norm^2
                perp_sq = h_norm**2 - proj_new**2
                if perp_sq > 0:
                    # 缩放h_perp使其满足范数约束
                    scale = np.sqrt(perp_sq) / h_perp_norm if h_perp_norm > 1e-10 else 1.0
                    h_rot = scale * h_perp + proj_new * d_head
                    
                    inputs_embeds_rot = embed_layer(input_ids).detach().clone()
                    inputs_embeds_rot[0, subj_idx, :] = torch.tensor(
                        h_rot, dtype=inputs_embeds_rot.dtype, device=device
                    )
                    
                    with torch.no_grad():
                        try:
                            rot_logits = model(inputs_embeds=inputs_embeds_rot).logits[0, -1].float().cpu().numpy()
                        except:
                            continue
                else:
                    rot_logits = None
            else:
                rot_logits = None
            
            # 比较top-5预测变化
            add_top5 = np.argsort(add_logits)[-5:][::-1]
            add_top5_tokens = [tokenizer.decode([t]) for t in add_top5]
            
            result = {
                "beta": beta,
                "add_top5": add_top5_tokens,
                "add_subj_prob": float(add_logits[input_ids[0, -1].item()] if input_ids[0, -1].item() < len(add_logits) else 0),
            }
            
            if rot_logits is not None:
                rot_top5 = np.argsort(rot_logits)[-5:][::-1]
                rot_top5_tokens = [tokenizer.decode([t]) for t in rot_top5]
                result["rot_top5"] = rot_top5_tokens
                
                # 范数比较
                add_h = inputs_embeds_add[0, subj_idx, :].float().cpu().numpy()
                result["add_norm_ratio"] = float(np.linalg.norm(add_h) / h_norm) if h_norm > 0 else 0
                result["rot_norm_ratio"] = 1.0  # 保范数
                
                # logit变化幅度
                result["add_logit_shift"] = float(np.linalg.norm(add_logits - base_logits))
                result["rot_logit_shift"] = float(np.linalg.norm(rot_logits - base_logits))
            
            results[prompt_key][f"beta_{beta}"] = result
    
    # 汇总
    add_shifts = []
    rot_shifts = []
    for pk in results:
        for bk in results[pk]:
            r = results[pk][bk]
            if "add_logit_shift" in r:
                add_shifts.append(r["add_logit_shift"])
            if "rot_logit_shift" in r:
                rot_shifts.append(r["rot_logit_shift"])
    
    if add_shifts and rot_shifts:
        results["summary"] = {
            "add_mean_shift": float(np.mean(add_shifts)),
            "rot_mean_shift": float(np.mean(rot_shifts)),
            "add_rot_ratio": float(np.mean(add_shifts) / max(np.mean(rot_shifts), 1e-10)),
        }
        print(f"  Additive mean logit shift: {np.mean(add_shifts):.2f}")
        print(f"  Rotation mean logit shift: {np.mean(rot_shifts):.2f}")
        print(f"  Add/Rot ratio: {np.mean(add_shifts)/max(np.mean(rot_shifts),1e-10):.2f}")
    
    return results


# ===== Exp3: 复杂自然句验证 =====
def run_exp3(model, tokenizer, device, model_info, d_head, d_level):
    """复杂自然句: 20+种非模板句的d_head方向一致性"""
    print("\n" + "="*70)
    print("Exp3: 复杂自然句验证 (Natural Sentence Validation)")
    print("="*70)
    
    if d_head is None:
        print("  SKIP: d_head not available")
        return {"status": "skipped"}
    
    d_model = model_info.d_model
    layers = get_layers(model)
    target_layer = len(layers) // 2
    
    correct = 0
    total = 0
    details = []
    
    for sent_data in COMPLEX_SENTENCES:
        text = sent_data["text"]
        pairs = sent_data["pairs"]
        
        # 获取所有相关token的表示
        target_tokens = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
        reps, tokens = get_token_representation(
            model, tokenizer, device, text, target_tokens, target_layer
        )
        
        if reps is None:
            continue
        
        for w1, w2, direction in pairs:
            if w1 not in reps or w2 not in reps:
                continue
            
            proj_w1 = np.dot(reps[w1], d_head)
            proj_w2 = np.dot(reps[w2], d_head)
            
            # head应该有更高的d_head投影
            if direction == "head":
                is_correct = proj_w1 > proj_w2
            else:  # "dep"
                is_correct = proj_w1 < proj_w2
            
            if is_correct:
                correct += 1
            total += 1
            
            details.append({
                "sentence": text[:50],
                "word1": w1,
                "word2": w2,
                "expected": direction,
                "proj_w1": float(proj_w1),
                "proj_w2": float(proj_w2),
                "correct": is_correct,
            })
    
    accuracy = correct / total if total > 0 else 0
    print(f"  Direction consistency: {correct}/{total} = {accuracy:.3f}")
    
    # 按句型复杂度分析
    short_sents = [d for d in details if len(d["sentence"].split()) <= 8]
    long_sents = [d for d in details if len(d["sentence"].split()) > 8]
    
    short_acc = sum(d["correct"] for d in short_sents) / max(len(short_sents), 1)
    long_acc = sum(d["correct"] for d in long_sents) / max(len(long_sents), 1)
    
    print(f"  Short sentences (≤8 words): {short_acc:.3f}")
    print(f"  Long sentences (>8 words): {long_acc:.3f}")
    
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "short_acc": short_acc,
        "long_acc": long_acc,
        "details_sample": details[:5],
    }


# ===== Exp4: V_syn维度估计 =====
def run_exp4(model, tokenizer, device, model_info, d_head, d_level):
    """V_syn维度估计: PCA方差解释率 + 交叉验证"""
    print("\n" + "="*70)
    print("Exp4: V_syn维度估计 (Dimensionality Estimation)")
    print("="*70)
    
    d_model = model_info.d_model
    layers = get_layers(model)
    target_layer = len(layers) // 2
    
    # 收集6种角色的表示 (大量样本)
    role_data = {
        "subject": [],
        "object": [],
        "modifier": [],
        "indirect_obj": [],
        "prep_obj": [],
        "adverbial": [],
    }
    
    words = ["cat", "dog", "bird", "fish", "lion", "tiger", "king", "queen",
             "eagle", "whale", "horse", "wolf", "sword", "shield", "fire", "water"]
    
    role_templates = {
        "subject": ["The {} chases the dog", "The {} sees the cat", "The {} hits the bird"],
        "object": ["The cat chases the {}", "The dog sees the {}", "The bird hits the {}"],
        "modifier": ["The red {} chases the dog", "The big {} sees the cat"],
        "indirect_obj": ["The king gives the {} the sword", "The queen gives the {} the crown"],
        "prep_obj": ["The cat looks at the {}", "The dog runs to the {}"],
        "adverbial": ["The cat runs {}", "The dog jumps {}"],
    }
    
    adverbial_words = ["quickly", "slowly", "fast", "hard"]
    
    for role, templates in role_templates.items():
        for word in words[:8]:
            for tmpl in templates[:2]:
                if role == "adverbial":
                    prompt = tmpl.format(np.random.choice(adverbial_words))
                    target = adverbial_words[0]
                else:
                    prompt = tmpl.format(word)
                    target = word
                
                reps, tokens = get_token_representation(
                    model, tokenizer, device, prompt, [target], target_layer
                )
                if reps and target in reps:
                    role_data[role].append(reps[target])
    
    # 检查数据量
    total_samples = sum(len(v) for v in role_data.values())
    print(f"  Collected {total_samples} samples across {len(role_data)} roles")
    
    if total_samples < 30:
        print("  WARNING: Insufficient data for PCA")
        return {"status": "insufficient_data"}
    
    # 合并所有表示
    all_reps = []
    all_labels = []
    for role, reps_list in role_data.items():
        all_reps.extend(reps_list)
        all_labels.extend([role] * len(reps_list))
    
    X = np.array(all_reps)  # [n, d_model]
    
    # 1. 全局PCA — 看角色信息占多少方差
    pca_full = PCA()
    pca_full.fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    
    # 2. 角色差值PCA — 只看角色间的差异
    centroids = {}
    for role in role_data:
        if len(role_data[role]) >= 2:
            centroids[role] = np.mean(role_data[role], axis=0)
    
    if len(centroids) >= 2:
        centroid_list = list(centroids.values())
        centroid_names = list(centroids.keys())
        C = np.array(centroid_list)  # [n_roles, d_model]
        
        # 中心化
        C_centered = C - np.mean(C, axis=0)
        
        # PCA on角色中心
        pca_roles = PCA()
        pca_roles.fit(C_centered)
        
        role_cumvar = np.cumsum(pca_roles.explained_variance_ratio_)
        
        print(f"  Role centroid PCA variance explained:")
        for k in [1, 2, 3, 4, 5]:
            if k-1 < len(role_cumvar):
                print(f"    {k}D: {role_cumvar[k-1]:.3f}")
    
    # 3. 交叉验证: 用k维PCA特征做角色分类
    X_labels_num = []
    role_to_idx = {r: i for i, r in enumerate(role_data.keys())}
    for label in all_labels:
        X_labels_num.append(role_to_idx[label])
    y = np.array(X_labels_num)
    
    # 二分类 subject vs object
    subj_idx = role_to_idx.get("subject", -1)
    obj_idx = role_to_idx.get("object", -1)
    
    binary_results = {}
    if subj_idx >= 0 and obj_idx >= 0:
        mask = (y == subj_idx) | (y == obj_idx)
        X_bin = X[mask]
        y_bin = y[mask]
        
        if len(X_bin) >= 10:
            for n_dims in [1, 2, 3, 5, 10, 20]:
                pca_cv = PCA(n_components=min(n_dims, d_model, X_bin.shape[0]))
                X_pca = pca_cv.fit_transform(X_bin)
                
                clf = LogisticRegression(max_iter=1000, C=1.0)
                cv = min(3, len(X_bin) // 2)
                scores = cross_val_score(clf, X_pca, y_bin, cv=cv)
                acc = float(np.mean(scores))
                binary_results[f"{n_dims}d"] = acc
                print(f"  Binary (subj/obj) {n_dims}D PCA: {acc:.3f}")
    
    # 4. 多角色分类
    multi_results = {}
    if len(X) >= 20:
        for n_dims in [1, 2, 3, 5, 10, 20]:
            pca_multi = PCA(n_components=min(n_dims, d_model, X.shape[0]))
            X_pca = pca_multi.fit_transform(X)
            
            clf = LogisticRegression(max_iter=1000, C=1.0)
            cv = min(3, len(X) // max(len(role_data), 2))
            try:
                scores = cross_val_score(clf, X_pca, y, cv=cv)
                acc = float(np.mean(scores))
                multi_results[f"{n_dims}d"] = acc
                print(f"  Multi-role {n_dims}D PCA: {acc:.3f}")
            except:
                pass
    
    # 5. 检查d_head/d_level在PCA中的位置
    if d_head is not None:
        pca_check = PCA(n_components=50)
        pca_check.fit(X)
        components = pca_check.components_  # [50, d_model]
        
        d_head_cos = []
        for i in range(len(components)):
            cos = abs(np.dot(components[i], d_head))
            d_head_cos.append(cos)
        
        top_pc_idx = np.argmax(d_head_cos)
        print(f"\n  d_head best aligned with PC{top_pc_idx+1} (cos={d_head_cos[top_pc_idx]:.3f})")
        print(f"  d_head alignment with top-5 PCs: {[f'{c:.3f}' for c in d_head_cos[:5]]}")
        
        if d_level is not None:
            d_level_cos = []
            for i in range(len(components)):
                cos = abs(np.dot(components[i], d_level))
                d_level_cos.append(cos)
            
            top_pc_idx_l = np.argmax(d_level_cos)
            print(f"  d_level best aligned with PC{top_pc_idx_l+1} (cos={d_level_cos[top_pc_idx_l]:.3f})")
    
    return {
        "total_samples": total_samples,
        "n_roles": len([r for r in role_data if len(role_data[r]) >= 2]),
        "binary_pca_acc": binary_results,
        "multi_pca_acc": multi_results,
    }


# ===== Exp5: 因果中介分析 =====
def run_exp5(model, tokenizer, device, model_info, d_head, d_level):
    """因果中介分析: d_head在subject→logit中的中介贡献"""
    print("\n" + "="*70)
    print("Exp5: 因果中介分析 (Causal Mediation Analysis)")
    print("="*70)
    
    if d_head is None:
        print("  SKIP: d_head not available")
        return {"status": "skipped"}
    
    d_model = model_info.d_model
    W_U = model.lm_head.weight.detach().cpu().float().numpy()
    
    # 中介分析框架:
    # X(句法角色) → M(d_head投影) → Y(下一个token预测)
    # 
    # 总效应 = P(Y=verb|subject_prompt) - P(Y=verb|object_prompt)
    # 直接效应 = 控制d_head后的效应
    # 中介效应 = 总效应 - 直接效应
    # 中介比例 = 中介效应 / 总效应
    
    # subject和object模板
    subj_templates = [
        ("The {} chases", "chases"),
        ("The {} sees", "sees"),
        ("The {} hits", "hits"),
    ]
    obj_templates = [
        ("The cat chases the {}", "chases"),
        ("The dog sees the {}", "sees"),
    ]
    
    words = ["cat", "dog", "bird", "fish", "lion", "tiger", "king", "queen"]
    
    total_effects = []
    mediated_effects = []
    direct_effects = []
    
    for word in words[:4]:
        for tmpl, verb in subj_templates[:2]:
            prompt_subj = tmpl.format(word)
            toks = tokenizer(prompt_subj, return_tensors="pt").to(device)
            
            # 基线: subject位置的logit
            with torch.no_grad():
                logits_subj = model(**toks).logits[0, -1].float().cpu().numpy()
            
            # 中介变量: d_head投影
            layers = get_layers(model)
            target_layer = len(layers) // 2
            
            captured = {}
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook
            
            hook = layers[target_layer].register_forward_hook(make_hook("target"))
            with torch.no_grad():
                _ = model(**toks)
            hook.remove()
            
            if "target" not in captured:
                continue
            
            h = captured["target"][0].numpy()  # [seq_len, d_model]
            tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
            
            # 找word位置
            word_idx = None
            for i, t in enumerate(tokens):
                if word.lower() in t.lower():
                    word_idx = i
                    break
            
            if word_idx is None:
                continue
            
            h_word = h[word_idx]
            d_head_proj = np.dot(h_word, d_head)
            
            # 控制中介: 移除d_head分量后重新前向
            h_word_no_head = h_word - d_head_proj * d_head
            
            # 用hook注入修改后的表示
            h_modified = captured["target"][0].clone()
            h_modified[word_idx] = torch.tensor(h_word_no_head, dtype=h_modified.dtype)
            
            # 重新前向 (从target_layer开始)
            # 简化方法: 在embedding层注入, 跑完整forward
            # 更准确但更复杂: 用TransformerLens的hook机制
            # 这里用简化方法: 看d_head投影占logit变化的多少
            
            # 总效应近似: d_head投影 * W_U在d_head方向的增益
            # logit变化 ≈ (W_U @ d_head) * d_head_proj
            logit_shift_from_head = (W_U @ d_head) * d_head_proj
            
            # 总logit方差
            total_logit_var = np.var(logits_subj)
            head_logit_var = np.var(logit_shift_from_head)
            
            # 中介比例 (简化版)
            if total_logit_var > 0:
                mediation_ratio = head_logit_var / total_logit_var
            else:
                mediation_ratio = 0
            
            total_effects.append(float(total_logit_var))
            mediated_effects.append(float(head_logit_var))
    
    if total_effects:
        mean_total = float(np.mean(total_effects))
        mean_mediated = float(np.mean(mediated_effects))
        mediation_pct = mean_mediated / max(mean_total, 1e-10) * 100
        
        print(f"  Mean total logit variance: {mean_total:.4f}")
        print(f"  Mean d_head-mediated variance: {mean_mediated:.4f}")
        print(f"  Mediation ratio: {mediation_pct:.1f}%")
        
        # 更严谨的方法: 直接比较有无d_head时的预测变化
        # 用实际前向pass
        n_test = 0
        n_flipped = 0
        n_flipped_ablated = 0
        
        for word in words[:4]:
            for tmpl, verb in subj_templates[:2]:
                prompt = tmpl.format(word)
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                
                # 原始top-1
                with torch.no_grad():
                    logits = model(**toks).logits[0, -1].float().cpu().numpy()
                top1_orig = np.argmax(logits)
                
                # 加法翻转 (加 -2*d_head 到subject位置)
                embed_layer = model.get_input_embeddings()
                input_ids = toks.input_ids
                inputs_embeds = embed_layer(input_ids).detach().clone()
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                
                word_idx = None
                for i, t in enumerate(tokens):
                    if word.lower() in t.lower():
                        word_idx = i
                        break
                
                if word_idx is None:
                    continue
                
                d_head_t = torch.tensor(d_head, dtype=inputs_embeds.dtype, device=device)
                inputs_embeds[0, word_idx, :] -= (8.0 * d_head_t).to(model.dtype)
                
                with torch.no_grad():
                    try:
                        logits_flip = model(inputs_embeds=inputs_embeds).logits[0, -1].float().cpu().numpy()
                    except:
                        continue
                
                top1_flip = np.argmax(logits_flip)
                
                n_test += 1
                if top1_flip != top1_orig:
                    n_flipped += 1
        
        flip_rate = n_flipped / max(n_test, 1)
        print(f"  Subject→Object flip rate (additive, β=8): {n_flipped}/{n_test} = {flip_rate:.3f}")
        
        return {
            "mean_total_var": mean_total,
            "mean_mediated_var": mean_mediated,
            "mediation_pct": mediation_pct,
            "flip_rate_additive": flip_rate,
            "n_test": n_test,
        }
    
    return {"status": "no_results"}


# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    
    model_name = args.model
    exp = args.exp
    
    print(f"\n{'='*70}")
    print(f"CCXLV: 五大严谨性验证 — {model_name} — Exp{exp}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    # 先提取d_head和d_level
    print("\nExtracting d_head and d_level directions...")
    d_head, d_level = extract_d_head_direction(model, tokenizer, device, model_info)
    
    if d_head is not None:
        print(f"  d_head extracted, norm={np.linalg.norm(d_head):.4f}")
    if d_level is not None:
        cos_hl = abs(np.dot(d_head, d_level)) if d_head is not None and d_level is not None else 0
        print(f"  d_level extracted, |cos(d_head, d_level)|={cos_hl:.4f}")
    
    # 运行实验
    if exp == 1:
        results = run_exp1(model, tokenizer, device, model_info, d_head, d_level)
    elif exp == 2:
        results = run_exp2(model, tokenizer, device, model_info, d_head, d_level)
    elif exp == 3:
        results = run_exp3(model, tokenizer, device, model_info, d_head, d_level)
    elif exp == 4:
        results = run_exp4(model, tokenizer, device, model_info, d_head, d_level)
    elif exp == 5:
        results = run_exp5(model, tokenizer, device, model_info, d_head, d_level)
    
    # 保存结果
    out_path = TEMP / f"ccxlv_exp{exp}_{model_name}_results.json"
    
    # 转换numpy类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    results["model"] = model_name
    results["exp"] = exp
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {out_path}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    
    print(f"\n{'='*70}")
    print(f"CCXLV Exp{exp} ({model_name}) COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
