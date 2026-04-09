"""
Phase XL: L0→L1非线性破解与Jacobian修饰模型
P246: MLP拟合L0→L1变换 (大数据集500+词)
P247: Jacobian修饰模型 (差分法估计J矩阵)
P248: 序列级上下文调制追踪 (逐步注入上下文)
P249: 因果方程最终可预测性验证
P250: 跨模型非线性统一理论分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, gc, time, json, argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置 =====================
OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_model_path(model_name):
    from pathlib import Path as _P
    paths = {
        "qwen3": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "deepseek7b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "glm4": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
    }
    return paths.get(model_name)

def load_model(model_name):
    import os
    p = get_model_path(model_name)
    # 确保路径存在
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Model path not found: {p}")
    # 使用os.path.abspath获取绝对路径，避免HuggingFace解析问题
    p_abs = os.path.abspath(p)
    tok = AutoTokenizer.from_pretrained(p_abs, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        p_abs, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager"
    )
    mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device)
    return mdl, tok, device

# ===================== 大词表 =====================
LARGE_WORD_LIST = [
    # 名词 (200个)
    "apple", "banana", "cherry", "dog", "cat", "bird", "car", "bus", "train",
    "king", "queen", "prince", "house", "building", "castle", "water", "fire", "earth",
    "book", "paper", "letter", "table", "chair", "desk", "sun", "moon", "star",
    "river", "mountain", "ocean", "tree", "flower", "grass", "fish", "bear", "wolf",
    "city", "village", "country", "school", "hospital", "church", "road", "bridge", "tunnel",
    "music", "art", "science", "history", "math", "language", "computer", "phone", "radio",
    "mother", "father", "sister", "brother", "friend", "teacher", "doctor", "lawyer",
    "bread", "rice", "meat", "cheese", "wine", "beer", "coffee", "tea", "milk",
    "gold", "silver", "iron", "stone", "wood", "glass", "paper", "cotton", "silk",
    "spring", "summer", "autumn", "winter", "morning", "evening", "night", "dawn", "dusk",
    "war", "peace", "love", "hate", "fear", "hope", "dream", "truth", "lie",
    "hand", "foot", "head", "eye", "ear", "mouth", "heart", "brain", "blood",
    "shirt", "dress", "coat", "hat", "shoe", "ring", "watch", "glasses",
    "garden", "park", "forest", "desert", "island", "beach", "lake", "cave",
    "airplane", "ship", "bicycle", "horse", "elephant", "lion", "tiger", "snake",
    "piano", "guitar", "violin", "drum", "trumpet", "flute", "bell", "horn",
    "diamond", "ruby", "pearl", "crystal", "marble", "clay", "sand", "dust",
    "cloud", "rain", "snow", "wind", "storm", "thunder", "lightning", "fog",
    "bank", "office", "factory", "market", "store", "restaurant", "hotel", "museum",
    "oxygen", "carbon", "hydrogen", "nitrogen", "helium", "sodium", "calcium", "iron",
    "idea", "plan", "rule", "law", "right", "duty", "power", "energy", "force",
    "speed", "weight", "size", "shape", "color", "sound", "light", "heat",
    "north", "south", "east", "west", "center", "edge", "top", "bottom",
    "beginning", "middle", "end", "past", "present", "future", "moment", "era",
    # 动词 (100个)
    "run", "walk", "jump", "swim", "fly", "climb", "fall", "rise", "grow",
    "eat", "drink", "sleep", "wake", "breathe", "think", "feel", "see", "hear",
    "speak", "write", "read", "sing", "dance", "play", "work", "rest", "wait",
    "give", "take", "buy", "sell", "send", "receive", "find", "lose", "keep",
    "open", "close", "start", "stop", "move", "turn", "push", "pull", "hold",
    "build", "break", "create", "destroy", "change", "choose", "decide", "try",
    "help", "hurt", "love", "hate", "fear", "trust", "believe", "doubt", "know",
    "learn", "teach", "lead", "follow", "win", "lose", "fight", "protect", "attack",
    "drive", "ride", "sail", "draw", "paint", "cook", "clean", "fix", "count",
    "measure", "compare", "explain", "understand", "remember", "forget", "imagine", "discover",
    "accept", "refuse", "agree", "argue", "promise", "warn", "invite", "visit",
    # 形容词 (100个)
    "big", "small", "tall", "short", "long", "wide", "narrow", "thick", "thin",
    "heavy", "light", "fast", "slow", "hot", "cold", "warm", "cool", "wet", "dry",
    "new", "old", "young", "beautiful", "ugly", "clean", "dirty", "rich", "poor",
    "strong", "weak", "hard", "soft", "sharp", "dull", "bright", "dark", "loud", "quiet",
    "happy", "sad", "angry", "calm", "brave", "scared", "kind", "cruel", "honest", "false",
    "smart", "stupid", "wise", "foolish", "good", "bad", "right", "wrong", "easy", "difficult",
    "simple", "complex", "safe", "dangerous", "full", "empty", "open", "closed", "free", "busy",
    "red", "blue", "green", "yellow", "white", "black", "brown", "pink", "purple", "orange",
    "fresh", "stale", "sweet", "bitter", "sour", "salty", "spicy", "plain", "rich", "plain",
    "deep", "shallow", "high", "low", "near", "far", "early", "late", "first", "last",
    # 副词/介词/代词 (100个)
    "quickly", "slowly", "carefully", "suddenly", "always", "never", "often", "rarely",
    "already", "still", "yet", "just", "very", "quite", "almost", "enough",
    "here", "there", "everywhere", "nowhere", "inside", "outside", "above", "below",
    "together", "apart", "forward", "backward", "up", "down", "away", "back",
    "in", "on", "at", "by", "with", "from", "to", "for", "of", "about",
    "between", "through", "under", "over", "across", "along", "around", "behind",
    "he", "she", "it", "they", "we", "you", "I", "me", "him", "her",
    "this", "that", "these", "those", "my", "your", "his", "her", "its", "our",
    "who", "what", "where", "when", "why", "how", "which", "whose",
    "some", "any", "all", "none", "many", "few", "more", "less", "much", "little",
    "each", "every", "both", "either", "neither", "other", "another", "such", "same", "different",
]

# ===================== P246: MLP拟合L0→L1 =====================
def p246_mlp_fit_L0_L1(mdl, tok, device, model_name):
    """用3层MLP拟合L0→L1变换, 500+词大数据集"""
    print(f"\n{'='*60}")
    print(f"P246: MLP拟合L0→L1变换 ({model_name})")
    print(f"{'='*60}")
    
    d_model = mdl.config.hidden_size
    n_layers = mdl.config.num_hidden_layers
    log_data = {}
    
    # 收集500+词的(L0, L1)对
    words = LARGE_WORD_LIST[:500]
    emb_list, L0_list, L1_list = [], [], []
    
    print(f"  Collecting {len(words)} (L0, L1) pairs...")
    for w in words:
        inputs = tok(f"The {w}", return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**inputs, output_hidden_states=True)
        # L0 = embedding layer output, L1 = first transformer layer output
        h_L0 = out.hidden_states[0][0, -1].float().cpu()
        h_L1 = out.hidden_states[1][0, -1].float().cpu()
        L0_list.append(h_L0)
        L1_list.append(h_L1)
        # embedding
        tid = inputs["input_ids"][0, -1].item()
        emb_row = mdl.model.embed_tokens.weight[tid:tid+1].detach().float().cpu()[0].clone()
        emb_list.append(emb_row)
        del out
    gc.collect()
    
    X_L0 = torch.stack(L0_list)  # [N, d_model]
    Y_L1 = torch.stack(L1_list)  # [N, d_model]
    X_emb = torch.stack(emb_list)  # [N, d_model]
    N = X_L0.shape[0]
    
    print(f"  N={N} pairs, d_model={d_model}")
    
    # Train/test split 80/20
    perm = torch.randperm(N)
    n_train = int(0.8 * N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    X_train, X_test = X_L0[train_idx], X_L0[test_idx]
    Y_train, Y_test = Y_L1[train_idx], Y_L1[test_idx]
    
    # ---- Baseline: Linear Regression ----
    print(f"\n  [1] Linear Regression baseline...")
    reg_linear = Ridge(alpha=1.0)
    reg_linear.fit(X_train.numpy(), Y_train.numpy())
    Y_pred_linear = reg_linear.predict(X_test.numpy())
    r2_linear = r2_score(Y_test.numpy(), Y_pred_linear)
    cos_linear = F.cosine_similarity(
        torch.from_numpy(Y_pred_linear).float(), 
        Y_test.float(), dim=1
    ).mean().item()
    print(f"    Linear R2={r2_linear:.4f}, avg_cos={cos_linear:.4f}")
    
    # ---- MLP拟合 ----
    print(f"\n  [2] MLP fitting...")
    
    class L0toL1MLP(nn.Module):
        def __init__(self, d_model, hidden_mult=4):
            super().__init__()
            d_hidden = d_model * hidden_mult
            self.net = nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_model),
            )
        def forward(self, x):
            return self.net(x)
    
    # 尝试不同hidden_mult
    for hidden_mult in [2, 4, 8]:
        mlp = L0toL1MLP(d_model, hidden_mult).float()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)
        
        # 训练
        X_tr = X_train.float()
        Y_tr = Y_train.float()
        X_te = X_test.float()
        Y_te = Y_test.float()
        
        best_r2 = -1
        best_cos = 0
        batch_size = 64
        
        for epoch in range(2000):
            perm_b = torch.randperm(n_train)
            epoch_loss = 0
            n_batches = 0
            for i in range(0, n_train, batch_size):
                batch_idx = perm_b[i:i+batch_size]
                x_b = X_tr[batch_idx]
                y_b = Y_tr[batch_idx]
                
                y_pred = mlp(x_b)
                loss = F.mse_loss(y_pred, y_b)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            
            if epoch % 200 == 0 or epoch == 1999:
                with torch.no_grad():
                    Y_pred_mlp = mlp(X_te)
                    r2_mlp = 1 - ((Y_te - Y_pred_mlp)**2).sum() / ((Y_te - Y_te.mean(0))**2).sum()
                    r2_mlp = r2_mlp.item()
                    cos_mlp = F.cosine_similarity(Y_pred_mlp, Y_te, dim=1).mean().item()
                    
                    if r2_mlp > best_r2:
                        best_r2 = r2_mlp
                        best_cos = cos_mlp
                    
                    print(f"    Epoch {epoch}: loss={epoch_loss/n_batches:.6f}, R2={r2_mlp:.4f}, cos={cos_mlp:.4f}")
        
        log_data[f"mlp_hidden{hidden_mult}x_R2"] = best_r2
        log_data[f"mlp_hidden{hidden_mult}x_cos"] = best_cos
        print(f"  MLP(hidden={hidden_mult}x): best R2={best_r2:.4f}, cos={best_cos:.4f}")
    
    # ---- Deep MLP (6层) ----
    print(f"\n  [3] Deep MLP (6 layers)...")
    
    class DeepMLP(nn.Module):
        def __init__(self, d_model, n_layers=6, hidden_mult=4):
            super().__init__()
            d_hidden = d_model * hidden_mult
            layers = []
            layers.append(nn.Linear(d_model, d_hidden))
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(d_hidden, d_hidden))
                layers.append(nn.GELU())
            layers.append(nn.Linear(d_hidden, d_model))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)
    
    deep_mlp = DeepMLP(d_model, n_layers=6, hidden_mult=4).float()
    optimizer = torch.optim.Adam(deep_mlp.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)
    
    best_r2_deep = -1
    best_cos_deep = 0
    
    for epoch in range(3000):
        perm_b = torch.randperm(n_train)
        epoch_loss = 0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            batch_idx = perm_b[i:i+batch_size]
            x_b = X_tr[batch_idx]
            y_b = Y_tr[batch_idx]
            
            y_pred = deep_mlp(x_b)
            loss = F.mse_loss(y_pred, y_b)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if epoch % 300 == 0 or epoch == 2999:
            with torch.no_grad():
                Y_pred_deep = deep_mlp(X_te)
                r2_deep = 1 - ((Y_te - Y_pred_deep)**2).sum() / ((Y_te - Y_te.mean(0))**2).sum()
                r2_deep = r2_deep.item()
                cos_deep = F.cosine_similarity(Y_pred_deep, Y_te, dim=1).mean().item()
                
                if r2_deep > best_r2_deep:
                    best_r2_deep = r2_deep
                    best_cos_deep = cos_deep
                
                print(f"    Epoch {epoch}: loss={epoch_loss/n_batches:.6f}, R2={r2_deep:.4f}, cos={cos_deep:.4f}")
    
    log_data["deep_mlp_6layer_R2"] = best_r2_deep
    log_data["deep_mlp_6layer_cos"] = best_cos_deep
    print(f"  Deep MLP(6L): best R2={best_r2_deep:.4f}, cos={best_cos_deep:.4f}")
    
    # ---- Emb→L0→L1 两阶段分解 ----
    print(f"\n  [4] Two-stage decomposition: emb→L0→L1...")
    
    # emb→L0
    X_emb_train, X_emb_test = X_emb[train_idx], X_emb[test_idx]
    
    reg_emb_to_L0 = Ridge(alpha=1.0)
    reg_emb_to_L0.fit(X_emb_train.numpy(), X_L0[train_idx].numpy())
    L0_pred = reg_emb_to_L0.predict(X_emb_test.numpy())
    r2_emb_L0 = r2_score(X_L0[test_idx].numpy(), L0_pred)
    cos_emb_L0 = F.cosine_similarity(
        torch.from_numpy(L0_pred).float(),
        X_L0[test_idx].float(), dim=1
    ).mean().item()
    print(f"    emb→L0: R2={r2_emb_L0:.4f}, cos={cos_emb_L0:.4f}")
    
    # L0→L1 (already have)
    print(f"    L0→L1: R2={r2_linear:.4f}, cos={cos_linear:.4f}")
    
    # emb→L1 端到端
    reg_emb_to_L1 = Ridge(alpha=1.0)
    reg_emb_to_L1.fit(X_emb_train.numpy(), Y_L1[train_idx].numpy())
    L1_pred_emb = reg_emb_to_L1.predict(X_emb_test.numpy())
    r2_emb_L1 = r2_score(Y_L1[test_idx].numpy(), L1_pred_emb)
    cos_emb_L1 = F.cosine_similarity(
        torch.from_numpy(L1_pred_emb).float(),
        Y_L1[test_idx].float(), dim=1
    ).mean().item()
    print(f"    emb→L1 (linear): R2={r2_emb_L1:.4f}, cos={cos_emb_L1:.4f}")
    
    # 逐维度分析非线性
    print(f"\n  [5] Per-dimension nonlinearity analysis...")
    dim_r2 = []
    for d in range(min(d_model, 100)):
        reg_d = Ridge(alpha=1.0)
        reg_d.fit(X_train.numpy(), Y_train[:, d].numpy())
        pred_d = reg_d.predict(X_test.numpy())
        r2_d = r2_score(Y_test[:, d].numpy(), pred_d)
        dim_r2.append(r2_d)
    
    dim_r2 = np.array(dim_r2)
    low_r2_dims = np.where(dim_r2 < 0.5)[0]
    print(f"    dims with R2<0.5: {len(low_r2_dims)}/{len(dim_r2)}")
    print(f"    min R2 dim: {dim_r2.min():.4f} (dim {dim_r2.argmin()})")
    print(f"    median R2: {np.median(dim_r2):.4f}")
    print(f"    mean R2: {dim_r2.mean():.4f}")
    
    log_data["linear_R2"] = r2_linear
    log_data["linear_cos"] = cos_linear
    log_data["emb_L0_R2"] = r2_emb_L0
    log_data["emb_L0_cos"] = cos_emb_L0
    log_data["emb_L1_R2"] = r2_emb_L1
    log_data["emb_L1_cos"] = cos_emb_L1
    log_data["n_low_r2_dims"] = int(len(low_r2_dims))
    log_data["min_dim_r2"] = float(dim_r2.min())
    log_data["median_dim_r2"] = float(np.median(dim_r2))
    
    # 保存结果
    result = {
        "model": model_name,
        "N": N,
        "d_model": d_model,
        "linear": {"R2": r2_linear, "cos": cos_linear},
        "mlp_2x": {"R2": log_data.get("mlp_hidden2x_R2", 0), "cos": log_data.get("mlp_hidden2x_cos", 0)},
        "mlp_4x": {"R2": log_data.get("mlp_hidden4x_R2", 0), "cos": log_data.get("mlp_hidden4x_cos", 0)},
        "mlp_8x": {"R2": log_data.get("mlp_hidden8x_R2", 0), "cos": log_data.get("mlp_hidden8x_cos", 0)},
        "deep_mlp_6L": {"R2": best_r2_deep, "cos": best_cos_deep},
        "emb_L0": {"R2": r2_emb_L0, "cos": cos_emb_L0},
        "emb_L1": {"R2": r2_emb_L1, "cos": cos_emb_L1},
        "per_dim": {"n_low_r2": int(len(low_r2_dims)), "min_r2": float(dim_r2.min()), "median_r2": float(np.median(dim_r2))},
    }
    
    print(f"\n  P246 Summary ({model_name}):")
    print(f"    Linear:    R2={r2_linear:.4f}, cos={cos_linear:.4f}")
    print(f"    MLP(4x):   R2={log_data.get('mlp_hidden4x_R2',0):.4f}, cos={log_data.get('mlp_hidden4x_cos',0):.4f}")
    print(f"    Deep(6L):  R2={best_r2_deep:.4f}, cos={best_cos_deep:.4f}")
    print(f"    emb→L0:    R2={r2_emb_L0:.4f}, cos={cos_emb_L0:.4f}")
    print(f"    emb→L1:    R2={r2_emb_L1:.4f}, cos={cos_emb_L1:.4f}")
    
    del X_L0, Y_L1, X_emb, X_train, X_test, Y_train, Y_test
    del mlp, deep_mlp
    gc.collect()
    
    return result


# ===================== P247: Jacobian修饰模型 =====================
def p247_jacobian_modification(mdl, tok, device, model_name):
    """用差分法估计Jacobian矩阵, 验证h(n+adj) = h(n) + J(h(n))·δ_adj"""
    print(f"\n{'='*60}")
    print(f"P247: Jacobian修饰模型 ({model_name})")
    print(f"{'='*60}")
    
    d_model = mdl.config.hidden_size
    
    nouns = ["apple", "dog", "car", "king", "house", "water", "book", "city", 
             "tree", "fish", "child", "woman", "man", "river", "mountain",
             "hand", "door", "road", "school", "garden", "bread", "shirt",
             "clock", "lamp", "table", "chair", "window", "flower", "stone", "cloud"]
    
    adjectives = ["red", "big", "small", "new", "old", "happy", "sad", "fast", "slow",
                  "green", "blue", "hot", "cold", "beautiful", "strong", "dark", "bright",
                  "long", "soft", "heavy"]
    
    # 收集 h(noun), h(adj+noun), h(adj_only) 数据
    print(f"  Collecting {len(nouns)}×{len(adjectives)} modification pairs...")
    
    noun_hiddens = {}  # noun -> hidden state
    adj_noun_hiddens = {}  # (adj, noun) -> hidden state
    adj_hiddens = {}  # adj -> hidden state (from "The adj")
    
    # 先收集名词
    for n in nouns:
        inputs = tok(f"The {n}", return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**inputs, output_hidden_states=True)
        noun_hiddens[n] = out.hidden_states[-1][0, -1].float().cpu()
        del out
    
    # 收集形容词+名词
    for adj in adjectives:
        # 形容词的hidden
        inputs = tok(f"The {adj}", return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**inputs, output_hidden_states=True)
        adj_hiddens[adj] = out.hidden_states[-1][0, -1].float().cpu()
        del out
        
        for n in nouns:
            inputs = tok(f"The {adj} {n}", return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl(**inputs, output_hidden_states=True)
            adj_noun_hiddens[(adj, n)] = out.hidden_states[-1][0, -1].float().cpu()
            del out
    
    gc.collect()
    
    # ---- 分析1: 通用修饰方向 vs Jacobian修正 ----
    print(f"\n  [1] Universal direction vs Jacobian correction...")
    
    results = {"universal": [], "jacobian": [], "additive": []}
    
    for adj in adjectives:
        deltas = []
        noun_vecs = []
        for n in nouns:
            delta = adj_noun_hiddens[(adj, n)] - noun_hiddens[n]
            deltas.append(delta)
            noun_vecs.append(noun_hiddens[n])
        
        # 通用方向 = 所有delta的均值
        delta_universal = torch.stack(deltas).mean(0)
        
        # 评估: 通用方向预测delta的能力
        cos_universal_list = []
        cos_jacobian_list = []
        cos_additive_list = []
        
        for i, n in enumerate(nouns):
            actual_delta = deltas[i]
            
            # 通用方向预测
            cos_u = F.cosine_similarity(delta_universal.unsqueeze(0), actual_delta.unsqueeze(0)).item()
            cos_universal_list.append(cos_u)
            
            # Jacobian修正: 用相邻名词的delta来估计J
            # 简化: δ_pred = J·h(n), J = Σ δ_i ⊗ h(n_i) / (h(n_i)·h(n_i))
            # 这里用留一法估计
            other_deltas = [deltas[j] for j in range(len(nouns)) if j != i]
            other_nouns = [noun_vecs[j] for j in range(len(nouns)) if j != i]
            
            # 用线性回归拟合 J: δ = J · h_noun
            # J ≈ (Δ^T · H) · (H^T · H)^{-1}
            H_other = torch.stack(other_nouns)  # [N-1, d_model]
            D_other = torch.stack(other_deltas)  # [N-1, d_model]
            
            # Ridge回归估计J
            try:
                reg = Ridge(alpha=1.0)
                reg.fit(H_other.numpy(), D_other.numpy())
                delta_pred = reg.predict(noun_vecs[i].unsqueeze(0).numpy())
                cos_j = F.cosine_similarity(
                    torch.from_numpy(delta_pred).float().squeeze(),
                    actual_delta, dim=0
                ).item()
                cos_jacobian_list.append(cos_j)
            except:
                cos_jacobian_list.append(cos_u)  # fallback
            
            # 简单加法: h(n+adj) ≈ h(n) + h(adj) - B_global
            # 用平均cos评估
            cos_a = F.cosine_similarity(
                (adj_hiddens[adj] - noun_hiddens[nouns[0]] + noun_hiddens[n]).unsqueeze(0),
                adj_noun_hiddens[(adj, n)].unsqueeze(0)
            ).item()
            cos_additive_list.append(cos_a)
        
        avg_universal = np.mean(cos_universal_list)
        avg_jacobian = np.mean(cos_jacobian_list)
        avg_additive = np.mean(cos_additive_list)
        
        results["universal"].append(avg_universal)
        results["jacobian"].append(avg_jacobian)
        results["additive"].append(avg_additive)
        
        print(f"    {adj:12s}: universal_cos={avg_universal:.3f}, jacobian_cos={avg_jacobian:.3f}, additive_cos={avg_additive:.3f}")
    
    # ---- 分析2: Jacobian矩阵的秩 ----
    print(f"\n  [2] Jacobian matrix rank analysis...")
    
    for adj in ["red", "big", "happy", "fast"]:
        # 构造delta矩阵和noun矩阵
        deltas = []
        noun_vecs = []
        for n in nouns:
            deltas.append(adj_noun_hiddens[(adj, n)] - noun_hiddens[n])
            noun_vecs.append(noun_hiddens[n])
        
        D = torch.stack(deltas)  # [N_nouns, d_model]
        H = torch.stack(noun_vecs)  # [N_nouns, d_model]
        
        # SVD of D (所有delta的奇异值分布)
        _, S_D, _ = torch.linalg.svd(D, full_matrices=False)
        
        # 有效秩: sum(S)^2 / sum(S^2)
        erank = (S_D.sum())**2 / (S_D**2).sum()
        
        # dim50/dim90
        cumvar = torch.cumsum(S_D**2, 0) / (S_D**2).sum()
        dim50 = (cumvar < 0.5).sum().item() + 1
        dim90 = (cumvar < 0.9).sum().item() + 1
        
        print(f"    {adj}: erank={erank:.1f}, dim50={dim50}, dim90={dim90}, S[0]={S_D[0]:.1f}, S[-1]={S_D[-1]:.1f}")
    
    # ---- 分析3: 名词间的修饰delta关系 ----
    print(f"\n  [3] Cross-noun delta similarity...")
    
    adj_delta_coss = {}
    for adj in adjectives:
        deltas = [adj_noun_hiddens[(adj, n)] - noun_hiddens[n] for n in nouns]
        # 两两cos
        coss = []
        for i in range(len(deltas)):
            for j in range(i+1, len(deltas)):
                c = F.cosine_similarity(deltas[i].unsqueeze(0), deltas[j].unsqueeze(0)).item()
                coss.append(c)
        adj_delta_coss[adj] = np.mean(coss)
    
    print(f"  Cross-noun delta cos (avg):")
    for adj, c in sorted(adj_delta_coss.items(), key=lambda x: -x[1]):
        print(f"    {adj}: {c:.3f}")
    
    # 总结
    result = {
        "model": model_name,
        "n_nouns": len(nouns),
        "n_adjectives": len(adjectives),
        "avg_universal_cos": float(np.mean(results["universal"])),
        "avg_jacobian_cos": float(np.mean(results["jacobian"])),
        "avg_additive_cos": float(np.mean(results["additive"])),
        "jacobian_improvement": float(np.mean(results["jacobian"]) - np.mean(results["universal"])),
        "avg_cross_noun_delta_cos": float(np.mean(list(adj_delta_coss.values()))),
    }
    
    print(f"\n  P247 Summary ({model_name}):")
    print(f"    Universal direction cos: {result['avg_universal_cos']:.4f}")
    print(f"    Jacobian model cos:      {result['avg_jacobian_cos']:.4f}")
    print(f"    Additive model cos:      {result['avg_additive_cos']:.4f}")
    print(f"    Jacobian improvement:    {result['jacobian_improvement']:+.4f}")
    print(f"    Cross-noun delta cos:    {result['avg_cross_noun_delta_cos']:.4f}")
    
    del noun_hiddens, adj_noun_hiddens, adj_hiddens
    gc.collect()
    
    return result


# ===================== P248: 序列级上下文调制追踪 =====================
def p248_context_modulation_trace(mdl, tok, device, model_name):
    """逐步注入上下文, 追踪h如何变化"""
    print(f"\n{'='*60}")
    print(f"P248: 序列级上下文调制追踪 ({model_name})")
    print(f"{'='*60}")
    
    d_model = mdl.config.hidden_size
    n_layers = mdl.config.num_hidden_layers
    
    # 不同长度的上下文
    contexts = [
        ("bank", [
            ("", "bank"),                                    # 无上下文
            ("The", "bank"),                                 # 1词
            ("I went to the", "bank"),                       # 中性
            ("I deposited money in the", "bank"),            # 金融
            ("The river flowed past the", "bank"),           # 河岸
            ("I sat on the grass near the river", "bank"),   # 河岸(长)
            ("The financial institution called the", "bank"), # 金融(长)
        ]),
        ("light", [
            ("", "light"),
            ("The", "light"),
            ("Turn on the", "light"),                        # 灯
            ("The feather is very", "light"),                # 轻
            ("The speed of", "light"),                       # 光速
            ("She turned on the bedroom", "light"),          # 灯(长)
            ("This bag is surprisingly", "light"),           # 轻(长)
        ]),
        ("right", [
            ("", "right"),
            ("The", "right"),
            ("Turn to the", "right"),                        # 右
            ("You have the", "right"),                       # 权利
            ("That is the", "right"),                        # 正确
            ("The correct answer is the", "right"),          # 正确(长)
        ]),
        ("play", [
            ("", "play"),
            ("The", "play"),
            ("Let's", "play"),                               # 玩(动词)
            ("I watched the", "play"),                       # 戏剧(名词)
            ("The children went outside to", "play"),        # 玩(长)
            ("Shakespeare wrote this famous", "play"),       # 戏剧(长)
        ]),
    ]
    
    all_results = {}
    
    for target_word, context_list in contexts:
        print(f"\n  Target word: '{target_word}'")
        word_results = []
        
        for ctx, word in context_list:
            full_text = f"{ctx} {word}".strip()
            inputs = tok(full_text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = mdl(**inputs, output_hidden_states=True)
            
            # 最终层的hidden state
            h_final = out.hidden_states[-1][0, -1].float().cpu()
            
            # 收集所有层的hidden state
            layer_hiddens = []
            for L in range(len(out.hidden_states)):
                layer_hiddens.append(out.hidden_states[L][0, -1].float().cpu())
            
            word_results.append({
                "context": ctx if ctx else "(none)",
                "full_text": full_text,
                "h_final": h_final,
                "layer_hiddens": layer_hiddens,
                "n_ctx_tokens": inputs["input_ids"].shape[1] - 1,  # 减去target word
            })
            
            del out
        
        # 计算不同上下文间的cos
        print(f"    Cross-context cosine similarities:")
        for i in range(len(word_results)):
            for j in range(i+1, len(word_results)):
                cos_ij = F.cosine_similarity(
                    word_results[i]["h_final"].unsqueeze(0),
                    word_results[j]["h_final"].unsqueeze(0)
                ).item()
                print(f"      '{word_results[i]['context'][:30]:30s}' vs '{word_results[j]['context'][:30]:30s}': cos={cos_ij:.4f}")
        
        # 逐层分析上下文效应
        print(f"    Per-layer context effect (cos with no-context baseline):")
        for L in [0, 1, 5, 10, 20, n_layers//2, n_layers-1]:
            if L >= len(word_results[0]["layer_hiddens"]):
                continue
            h_no_ctx = word_results[0]["layer_hiddens"][L]
            coss = []
            for wr in word_results[1:]:
                cos_L = F.cosine_similarity(h_no_ctx.unsqueeze(0), wr["layer_hiddens"][L].unsqueeze(0)).item()
                coss.append(cos_L)
            print(f"      L{L:2d}: avg_cos={np.mean(coss):.4f}, min_cos={min(coss):.4f}")
        
        all_results[target_word] = {
            "n_contexts": len(context_list),
            "pairwise_cos": [],
        }
        
        # 保存cos矩阵
        for i in range(len(word_results)):
            for j in range(i+1, len(word_results)):
                cos_ij = F.cosine_similarity(
                    word_results[i]["h_final"].unsqueeze(0),
                    word_results[j]["h_final"].unsqueeze(0)
                ).item()
                all_results[target_word]["pairwise_cos"].append(cos_ij)
        
        del word_results
        gc.collect()
    
    # 汇总
    result = {
        "model": model_name,
        "words_tested": [w for w, _ in contexts],
    }
    
    for word, data in all_results.items():
        coss = data["pairwise_cos"]
        result[f"{word}_avg_cos"] = float(np.mean(coss))
        result[f"{word}_min_cos"] = float(min(coss))
        result[f"{word}_max_cos"] = float(max(coss))
        print(f"\n  {word}: avg_cos={np.mean(coss):.4f}, min={min(coss):.4f}, max={max(coss):.4f}")
    
    return result


# ===================== P249: 因果方程最终可预测性 =====================
def p249_causal_equation_predictability(mdl, tok, device, model_name):
    """给定emb+上下文C, 用简化方程预测h(w,C)"""
    print(f"\n{'='*60}")
    print(f"P249: 因果方程最终可预测性 ({model_name})")
    print(f"{'='*60}")
    
    d_model = mdl.config.hidden_size
    
    # 大规模测试: 200词 × 5上下文模板
    words = LARGE_WORD_LIST[:200]
    templates = [
        "The {} is here.",          # 中性
        "I saw a {} yesterday.",    # 中性
        "The big {} was found.",    # 有修饰
        "She gave me the {}.",      # 宾语
        "The {} ran quickly.",      # 主语+动词
    ]
    
    print(f"  Testing {len(words)} words × {len(templates)} templates...")
    
    # 收集数据
    all_data = {}  # template -> list of (emb, h_final, h_L0, word)
    
    for t_idx, template in enumerate(templates):
        data_list = []
        for w in words:
            text = template.format(w)
            inputs = tok(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl(**inputs, output_hidden_states=True)
            
            h_final = out.hidden_states[-1][0, -1].float().cpu()
            h_L0 = out.hidden_states[0][0, -1].float().cpu()
            
            # 获取target token的embedding
            # 找到{}位置的token
            target_tokens = tok(w, add_special_tokens=False)["input_ids"]
            # 最后一个token的embedding
            tid = target_tokens[-1]
            emb = mdl.model.embed_tokens.weight[tid:tid+1].detach().float().cpu()[0].clone()
            
            data_list.append({"emb": emb, "h_final": h_final, "h_L0": h_L0, "word": w})
            del out
        
        all_data[t_idx] = data_list
        gc.collect()
    
    # ---- 预测模型1: 纯emb→h ----
    print(f"\n  [1] Pure emb→h prediction...")
    
    # 用template 0作为训练, 其他作为测试
    train_data = all_data[0]
    test_data = all_data[1]
    
    X_train = torch.stack([d["emb"] for d in train_data])
    Y_train = torch.stack([d["h_final"] for d in train_data])
    X_test = torch.stack([d["emb"] for d in test_data])
    Y_test = torch.stack([d["h_final"] for d in test_data])
    
    # Linear
    reg = Ridge(alpha=1.0)
    reg.fit(X_train.numpy(), Y_train.numpy())
    Y_pred = reg.predict(X_test.numpy())
    r2_emb = r2_score(Y_test.numpy(), Y_pred)
    cos_emb = F.cosine_similarity(torch.from_numpy(Y_pred).float(), Y_test, dim=1).mean().item()
    print(f"    emb→h (linear): R2={r2_emb:.4f}, cos={cos_emb:.4f}")
    
    # ---- 预测模型2: 跨模板预测 ----
    print(f"\n  [2] Cross-template prediction...")
    
    for t_test in range(1, len(templates)):
        test_data = all_data[t_test]
        X_test_t = torch.stack([d["emb"] for d in test_data])
        Y_test_t = torch.stack([d["h_final"] for d in test_data])
        
        Y_pred_t = reg.predict(X_test_t.numpy())
        r2_t = r2_score(Y_test_t.numpy(), Y_pred_t)
        cos_t = F.cosine_similarity(torch.from_numpy(Y_pred_t).float(), Y_test_t, dim=1).mean().item()
        print(f"    Template {t_test} '{templates[t_test][:30]}': R2={r2_t:.4f}, cos={cos_t:.4f}")
    
    # ---- 预测模型3: 上下文差分 ----
    print(f"\n  [3] Context differential analysis...")
    
    # h_ctx - h_noctx = 上下文调制
    for t_idx in range(1, len(templates)):
        ctx_hiddens = torch.stack([d["h_final"] for d in all_data[t_idx]])
        noctx_hiddens = torch.stack([d["h_final"] for d in all_data[0]])
        
        delta_ctx = ctx_hiddens - noctx_hiddens
        
        # delta_ctx的方向一致性
        delta_norms = delta_ctx.norm(dim=1)
        delta_mean = delta_ctx.mean(0)
        
        cos_delta_mean = F.cosine_similarity(delta_ctx, delta_mean.unsqueeze(0).expand_as(delta_ctx), dim=1).mean().item()
        cos_delta_range = F.cosine_similarity(delta_ctx[0:1], delta_ctx[-1:], dim=1).item()
        
        # delta与B_global的关系
        B_global = noctx_hiddens.mean(0)
        B_global_norm = B_global / B_global.norm()
        cos_delta_B = F.cosine_similarity(delta_ctx.float(), B_global_norm.unsqueeze(0).expand_as(delta_ctx), dim=1).mean().item()
        
        print(f"    Template {t_idx}: delta_mean_cos={cos_delta_mean:.4f}, delta_range_cos={cos_delta_range:.4f}, cos(delta,B_global)={cos_delta_B:.4f}")
    
    # ---- 预测模型4: 混合模型 ----
    print(f"\n  [4] Hybrid model: emb + context_code...")
    
    # 构造混合特征: [emb, template_onehot]
    # 用所有模板数据训练
    X_all, Y_all, T_all = [], [], []
    for t_idx in range(len(templates)):
        for d in all_data[t_idx]:
            # one-hot template
            t_code = torch.zeros(len(templates))
            t_code[t_idx] = 1.0
            X_all.append(torch.cat([d["emb"], t_code]))
            Y_all.append(d["h_final"])
            T_all.append(t_idx)
    
    X_all = torch.stack(X_all)
    Y_all = torch.stack(Y_all)
    
    # 训练/测试分割
    N = X_all.shape[0]
    perm = torch.randperm(N)
    n_tr = int(0.8 * N)
    
    reg_hybrid = Ridge(alpha=1.0)
    reg_hybrid.fit(X_all[perm[:n_tr]].numpy(), Y_all[perm[:n_tr]].numpy())
    Y_pred_hybrid = reg_hybrid.predict(X_all[perm[n_tr:]].numpy())
    r2_hybrid = r2_score(Y_all[perm[n_tr:]].numpy(), Y_pred_hybrid)
    cos_hybrid = F.cosine_similarity(torch.from_numpy(Y_pred_hybrid).float(), Y_all[perm[n_tr:]], dim=1).mean().item()
    print(f"    Hybrid(emb+template): R2={r2_hybrid:.4f}, cos={cos_hybrid:.4f}")
    
    # ---- 预测模型5: 最终因果方程误差分解 ----
    print(f"\n  [5] Causal equation error decomposition...")
    
    # 误差分解: 总误差 = emb映射误差 + 上下文调制误差 + 修饰误差
    # emb映射误差: 同一模板内, emb→h的R2
    # 上下文调制误差: 跨模板, 同一word的h变异
    # 修饰误差: 有修饰模板 vs 无修饰模板
    
    emb_errors = []
    ctx_variations = []
    
    for w_idx in range(min(len(words), 100)):
        # 同一词在不同模板的hidden states
        h_per_template = []
        for t_idx in range(len(templates)):
            h_per_template.append(all_data[t_idx][w_idx]["h_final"])
        
        h_stack = torch.stack(h_per_template)
        
        # 上下文调制: 跨模板的变异
        h_mean = h_stack.mean(0)
        h_var = ((h_stack - h_mean)**2).sum() / (h_mean**2).sum()
        ctx_variations.append(h_var.item())
        
        # cos range
        cos_range = F.cosine_similarity(h_stack[0:1], h_stack[1:], dim=1).tolist()
        emb_errors.extend(cos_range)
    
    print(f"    Context variation (avg ||δ_ctx||²/||h||²): {np.mean(ctx_variations):.6f}")
    print(f"    Cross-template cos (same word): {np.mean(emb_errors):.4f}")
    
    # 误差下界
    total_variance = np.mean(ctx_variations)
    emb_R2 = r2_emb
    ctx_R2_loss = 1 - r2_hybrid  # 混合模型也无法解释的部分
    
    print(f"\n    Error decomposition:")
    print(f"      emb→h R2:           {emb_R2:.4f} (emb mapping)")
    print(f"      hybrid R2:          {r2_hybrid:.4f} (emb + context code)")
    print(f"      context variance:   {total_variance:.6f} (||δ_ctx||²/||h||²)")
    print(f"      unexplained:        {1-r2_hybrid:.4f}")
    
    result = {
        "model": model_name,
        "n_words": len(words),
        "n_templates": len(templates),
        "emb_h_R2": r2_emb,
        "emb_h_cos": cos_emb,
        "hybrid_R2": r2_hybrid,
        "hybrid_cos": cos_hybrid,
        "ctx_variance": float(np.mean(ctx_variations)),
        "cross_template_cos": float(np.mean(emb_errors)),
    }
    
    print(f"\n  P249 Summary ({model_name}):")
    print(f"    emb→h:     R2={r2_emb:.4f}, cos={cos_emb:.4f}")
    print(f"    hybrid:    R2={r2_hybrid:.4f}, cos={cos_hybrid:.4f}")
    print(f"    ctx_var:   {np.mean(ctx_variations):.6f}")
    
    del all_data
    gc.collect()
    
    return result


# ===================== P250: 跨模型非线性统一理论 =====================
def p250_cross_model_nonlinearity(mdl, tok, device, model_name):
    """分析模型架构/训练与非线性度的关系"""
    print(f"\n{'='*60}")
    print(f"P250: 跨模型非线性统一理论 ({model_name})")
    print(f"{'='*60}")
    
    d_model = mdl.config.hidden_size
    n_layers = mdl.config.num_hidden_layers
    n_heads = mdl.config.num_attention_heads
    intermediate_size = getattr(mdl.config, 'intermediate_size', 4 * d_model)
    
    # 模型架构信息
    print(f"  Model architecture:")
    print(f"    d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
    print(f"    intermediate_size={intermediate_size}")
    print(f"    vocab_size={mdl.config.vocab_size}")
    
    # ---- 分析1: 逐层非线性度 ----
    print(f"\n  [1] Per-layer nonlinearity profile...")
    
    test_words = LARGE_WORD_LIST[:100]
    layer_hiddens = {L: [] for L in range(n_layers + 1)}
    
    for w in test_words:
        inputs = tok(f"The {w}", return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**inputs, output_hidden_states=True)
        for L in range(n_layers + 1):
            layer_hiddens[L].append(out.hidden_states[L][0, -1].float().cpu())
        del out
    gc.collect()
    
    # 逐层线性R2
    layer_R2 = {}
    for L in range(n_layers):
        H_in = torch.stack(layer_hiddens[L])    # [N, d_model]
        H_out = torch.stack(layer_hiddens[L+1])  # [N, d_model]
        
        reg = Ridge(alpha=1.0)
        reg.fit(H_in.numpy(), H_out.numpy())
        H_pred = reg.predict(H_in.numpy())
        r2 = r2_score(H_out.numpy(), H_pred)
        layer_R2[L] = r2
        
        if L <= 5 or L >= n_layers - 3 or L == n_layers // 2:
            cos_L = F.cosine_similarity(torch.from_numpy(H_pred).float(), H_out, dim=1).mean().item()
            print(f"    L{L}→L{L+1}: R2={r2:.6f}, cos={cos_L:.6f}")
    
    # 找最非线性的层
    min_r2_layer = min(layer_R2, key=layer_R2.get)
    print(f"\n    Most nonlinear layer: L{min_r2_layer}→L{min_r2_layer+1}, R2={layer_R2[min_r2_layer]:.6f}")
    
    # ---- 分析2: 每层残差范数比 ----
    print(f"\n  [2] Per-layer residual norm ratio...")
    
    for L in range(min(10, n_layers)):
        h_in = torch.stack(layer_hiddens[L])
        h_out = torch.stack(layer_hiddens[L+1])
        delta = h_out - h_in
        ratio = delta.norm(dim=1).mean().item() / h_in.norm(dim=1).mean().item()
        print(f"    L{L}: ||delta||/||h_in|| = {ratio:.4f}")
    
    # ---- 分析3: 第一层FFN/Attention分解 ----
    print(f"\n  [3] Layer 0 decomposition (attn vs ffn)...")
    
    # Hook法收集attn_out和ffn_out
    L0_attn_outs = []
    L0_ffn_outs = []
    
    def make_hooks(layer_idx):
        attn_outs = []
        ffn_outs = []
        
        def attn_hook(module, input, output):
            # output of attention = (hidden_states, attn_weights, past_key_value)
            if isinstance(output, tuple):
                attn_outs.append(output[0].detach().float().cpu())
            else:
                attn_outs.append(output.detach().float().cpu())
        
        def ffn_hook(module, input, output):
            ffn_outs.append(output.detach().float().cpu())
        
        return attn_hook, ffn_hook, attn_outs, ffn_outs
    
    # 注册hook
    layer0 = mdl.model.layers[0]
    attn_hook, ffn_hook, a_outs, f_outs = make_hooks(0)
    
    h_attn = layer0.self_attn.register_forward_hook(attn_hook)
    h_ffn = layer0.mlp.register_forward_hook(ffn_hook)
    
    for w in test_words[:20]:
        inputs = tok(f"The {w}", return_tensors="pt").to(device)
        with torch.no_grad():
            _ = mdl(**inputs)
    
    h_attn.remove()
    h_ffn.remove()
    
    if a_outs and f_outs:
        # 分析attn和ffn的贡献
        attn_norms = [a[0, -1].norm().item() for a in a_outs]
        ffn_norms = [f[0, -1].norm().item() for f in f_outs]
        
        print(f"    ||attn_out||: mean={np.mean(attn_norms):.2f}, std={np.std(attn_norms):.2f}")
        print(f"    ||ffn_out||:  mean={np.mean(ffn_norms):.2f}, std={np.std(ffn_norms):.2f}")
        print(f"    ||ffn||/||attn|| = {np.mean(ffn_norms)/np.mean(attn_norms):.4f}")
        
        # cos(attn, ffn)
        cos_attn_ffn = []
        for a, f in zip(a_outs, f_outs):
            cos_val = F.cosine_similarity(a[0, -1].unsqueeze(0), f[0, -1].unsqueeze(0)).item()
            cos_attn_ffn.append(cos_val)
        print(f"    cos(attn, ffn): mean={np.mean(cos_attn_ffn):.4f}")
    
    # ---- 分析4: 第一层注意力模式 ----
    print(f"\n  [4] Layer 0 attention pattern analysis...")
    
    for w in test_words[:5]:
        inputs = tok(f"The {w}", return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**inputs, output_attentions=True)
        
        # L0的attention weights
        if out.attentions and len(out.attentions) > 0:
            attn_L0 = out.attentions[0][0]  # [n_heads, seq_len, seq_len]
            n_heads_actual = attn_L0.shape[0]
            seq_len = attn_L0.shape[1]
            
            # 最后一个token对各token的attention
            last_token_attn = attn_L0[:, -1, :]  # [n_heads, seq_len]
            
            # 哪些head关注target word?
            # target word通常是最后一个token
            target_attn = last_token_attn[:, -1].mean().item()
            prev_attn = last_token_attn[:, -2].mean().item() if seq_len > 1 else 0
            
            if w == test_words[0]:
                print(f"    '{w}': self_attn={target_attn:.4f}, prev_attn={prev_attn:.4f}")
        
        del out
    
    # 汇总
    result = {
        "model": model_name,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "intermediate_size": intermediate_size,
        "vocab_size": mdl.config.vocab_size,
        "L0_L1_R2": layer_R2.get(0, 0),
        "most_nonlinear_layer": min_r2_layer,
        "most_nonlinear_R2": layer_R2.get(min_r2_layer, 0),
        "avg_R2_after_L1": float(np.mean([v for k, v in layer_R2.items() if k >= 1])),
    }
    
    # 逐层R2列表
    result["layer_R2_profile"] = {f"L{k}": float(v) for k, v in sorted(layer_R2.items())}
    
    print(f"\n  P250 Summary ({model_name}):")
    print(f"    Architecture: {d_model}d, {n_layers}L, {n_heads}H")
    print(f"    L0→L1 R2: {layer_R2.get(0, 0):.6f}")
    print(f"    Most nonlinear: L{min_r2_layer} (R2={layer_R2[min_r2_layer]:.6f})")
    print(f"    Avg R2 after L1: {result['avg_R2_after_L1']:.6f}")
    
    del layer_hiddens
    gc.collect()
    
    return result


# ===================== 主程序 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", 
                       choices=["qwen3", "deepseek7b", "glm4", "gemma4"])
    args = parser.parse_args()
    
    model_name = args.model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    print(f"\n{'#'*60}")
    print(f"Phase XL: L0→L1 Nonlinear Mapping & Jacobian Model")
    print(f"Model: {model_name}, Time: {timestamp}")
    print(f"{'#'*60}")
    
    # 加载模型
    print(f"\nLoading model {model_name}...")
    mdl, tok, device = load_model(model_name)
    print(f"  Loaded on {device}")
    
    # 运行所有实验
    results = {}
    
    results["P246"] = p246_mlp_fit_L0_L1(mdl, tok, device, model_name)
    
    results["P247"] = p247_jacobian_modification(mdl, tok, device, model_name)
    
    results["P248"] = p248_context_modulation_trace(mdl, tok, device, model_name)
    
    results["P249"] = p249_causal_equation_predictability(mdl, tok, device, model_name)
    
    results["P250"] = p250_cross_model_nonlinearity(mdl, tok, device, model_name)
    
    # 保存结果
    out_file = OUT_DIR / f"phase40_{model_name}_{timestamp}.json"
    
    # 转换为可序列化格式
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'#'*60}")
    print(f"Phase XL Complete! Results saved to {out_file}")
    print(f"{'#'*60}")
    
    # 释放模型
    del mdl
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
