"""
CCL-O(256): W_U特征值谱分析 — 修正W_U⊥的误解
==================================================
核心问题: 之前用200维SVD近似的"W_U⊥"是虚假的!
  → W_U几乎满秩(2546/2560), 真正的W_U⊥只有14维
  → 之前的"75-94%在W_U⊥"实际上是"在W_U低能量方向"

正确问题: 语法角色信息在W_U的哪个特征值频段?
  → W_U^T @ W_U的特征值谱: 从16045到4.78
  → 高特征值方向: W_U对这些方向"放大"很大
  → 低特征值方向: W_U对这些方向"放大"很小
  → 语法角色编码是否偏好低特征值方向?

实验:
  Exp1: ★★★★★ 语法角色方向在W_U特征值谱中的投影
    → 将探针方向分解到W_U的特征向量上
    → 分析: 语法角色方向的能量分布 vs W_U特征值
    → 如果语法角色在低特征值方向, W_U"看不到"它们

  Exp2: ★★★★★ 不同特征值频段的分类能力
    → 在W_U的不同特征值频段(hidden states投影)中分类
    → 高频段(>median): W_U"看得到"的方向
    → 低频段(<median): W_U"看不太到"的方向
    → 极低频段(<1%): 之前误称为"W_U⊥"的方向

  Exp3: ★★★★★ 不同频段操控的logits效率
    → 在高/中/低特征值方向上分别操控
    → 测量每个方向的"logits效率": Δlogits / 扰动范数
    → 预测: 低特征值方向的logits效率更低
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
from scipy.sparse.linalg import svds

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode, get_W_U


# ===== 数据集 =====
EXTENDED_DATA = {
    "nsubj": {
        "sentences": [
            "The king ruled the kingdom", "The doctor treated the patient",
            "The artist painted the portrait", "The soldier defended the castle",
            "The cat sat on the mat", "The dog ran through the park",
            "The bird sang a beautiful song", "The child played with the toys",
            "The student read the textbook", "The teacher explained the lesson",
            "The woman drove the car", "The man fixed the roof",
            "The girl sang a song", "The boy kicked the ball",
            "The president signed the bill", "The chef cooked the meal",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog", "bird", "child",
            "student", "teacher", "woman", "man", "girl", "boy", "president", "chef",
        ],
    },
    "dobj": {
        "sentences": [
            "They crowned the king yesterday", "She visited the doctor recently",
            "He admired the artist greatly", "We honored the soldier today",
            "She chased the cat away", "He found the dog outside",
            "They watched the bird closely", "We helped the child today",
            "I praised the student loudly", "You thanked the teacher warmly",
            "The police arrested the man quickly", "The company hired the woman recently",
            "The coach trained the girl daily", "The teacher praised the boy warmly",
            "The nation elected the president fairly", "The customer tipped the chef generously",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog", "bird", "child",
            "student", "teacher", "man", "woman", "girl", "boy", "president", "chef",
        ],
    },
    "amod": {
        "sentences": [
            "The brave king fought hard", "The kind doctor helped many",
            "The creative artist worked well", "The strong soldier marched far",
            "The beautiful cat sat quietly", "The large dog ran swiftly",
            "The small bird sang softly", "The young child played happily",
            "The bright student read carefully", "The wise teacher explained clearly",
            "The old woman walked slowly", "The tall man stood quietly",
            "The little girl smiled sweetly", "The smart boy answered quickly",
            "The powerful president decided firmly", "The skilled chef cooked perfectly",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "beautiful", "large", "small", "young",
            "bright", "wise", "old", "tall", "little", "smart", "powerful", "skilled",
        ],
    },
    "advmod": {
        "sentences": [
            "The king ruled wisely forever", "The doctor worked carefully always",
            "The artist painted beautifully daily", "The soldier fought bravely there",
            "The cat ran quickly home", "The dog barked loudly today",
            "The bird sang softly outside", "The child played happily inside",
            "The student read carefully alone", "The teacher spoke clearly again",
            "The woman drove slowly home", "The man spoke quietly now",
            "The girl laughed happily then", "The boy ran fast away",
            "The president spoke firmly today", "The chef worked quickly then",
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely", "quickly", "loudly",
            "softly", "happily", "carefully", "clearly", "slowly", "quietly",
            "happily", "fast", "firmly", "quickly",
        ],
    },
}

ROLE_NAMES = ["nsubj", "dobj", "amod", "advmod"]

FUNCTION_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
    'or', 'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these',
    'those', 'it', 'its', 'he', 'she', 'they', 'them', 'we', 'us', 'me',
    'my', 'your', 'his', 'her', 'their', 'our', 'what', 'which', 'who',
    'whom', 'whose', ',', '.', '!', '?', ';', ':', "'", '"', '-',
    "'s", "'t", "'re", "'ve", "'ll", "'d", "'m",
}


def find_token_index(tokens, word):
    word_lower = word.lower()
    for i, tok in enumerate(tokens):
        if word_lower in tok.lower() or tok.lower().startswith(word_lower[:3]):
            return i
    return None


def safe_decode(tokenizer, token_id):
    try:
        result = tokenizer.decode([token_id])
        if result is None:
            return f"<tok_{token_id}>"
        return result
    except Exception:
        return f"<tok_{token_id}>"


def collect_hidden_states(model, tokenizer, device, layer_idx=-1):
    """收集所有语法角色的hidden states"""
    all_h = []
    all_labels = []
    
    layers = get_layers(model)
    target_layer = layers[layer_idx] if layer_idx >= 0 else layers[-1]
    
    for role_idx, role in enumerate(ROLE_NAMES):
        data = EXTENDED_DATA[role]
        for sent, target_word in zip(data["sentences"], data["target_words"]):
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
            all_labels.append(role_idx)
    
    return np.array(all_h), np.array(all_labels)


def compute_W_U_eigenspectrum(W_U):
    """计算W_U的完整特征值谱和特征向量"""
    d_model = W_U.shape[1]
    W_U_f64 = W_U.astype(np.float64)
    
    print(f"  计算W_U^T @ W_U ({d_model}x{d_model})...")
    WtW = W_U_f64.T @ W_U_f64
    
    eigenvalues, eigenvectors = np.linalg.eigh(WtW)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    
    return eigenvalues, eigenvectors


# ===== Exp1: 探针方向在W_U特征值谱中的投影 =====
def exp1_probe_spectrum_projection(model, tokenizer, device):
    """分析语法角色探针方向在W_U特征值谱中的投影分布"""
    print("\n" + "="*70)
    print("Exp1: 探针方向在W_U特征值谱中的投影 ★★★★★")
    print("="*70)
    
    # 收集hidden states
    print("  收集hidden states...")
    H, labels = collect_hidden_states(model, tokenizer, device)
    print(f"  收集到 {len(H)} 个样本")
    
    # 训练探针
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)
    probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    probe.fit(H_scaled, labels)
    
    # 探针方向
    probe_dirs = {}
    for ri, role in enumerate(ROLE_NAMES):
        w = probe.coef_[ri]
        probe_dirs[role] = w / np.linalg.norm(w)
    
    # 计算W_U特征谱
    W_U = get_W_U(model)
    eigenvalues, eigenvectors = compute_W_U_eigenspectrum(W_U)
    
    # 特征值频段定义
    total_energy = np.sum(eigenvalues)
    cum_energy = np.cumsum(eigenvalues) / total_energy
    
    print(f"\n  W_U特征值谱概况:")
    print(f"    最大: {eigenvalues[0]:.1f}, 最小: {eigenvalues[-1]:.2f}")
    print(f"    条件数: {eigenvalues[0]/max(eigenvalues[-1],1e-10):.0f}")
    
    # 定义频段
    bands = {
        'top1%': (0, int(0.01 * len(eigenvalues))),        # top 1% (26维)
        'top10%': (0, int(0.10 * len(eigenvalues))),       # top 10% (256维)
        'top50%': (0, int(0.50 * len(eigenvalues))),       # top 50% (1280维)
        'bottom50%': (int(0.50 * len(eigenvalues)), len(eigenvalues)),  # bottom 50%
        'bottom10%': (int(0.90 * len(eigenvalues)), len(eigenvalues)),  # bottom 10%
        'bottom1%': (int(0.99 * len(eigenvalues)), len(eigenvalues)),   # bottom 1%
    }
    
    results = {}
    
    # 分析每个探针方向在特征值频段中的投影能量
    print(f"\n  探针方向在W_U特征值频段中的投影能量比:")
    print(f"  {'Role':>8s} " + " ".join(f"{band:>10s}" for band in bands))
    
    for role in ROLE_NAMES:
        direction = probe_dirs[role]
        dir_norm_sq = np.dot(direction, direction)
        
        # 在每个特征向量上的投影
        proj_coeffs = eigenvectors.T @ direction  # [d_model]
        proj_energy = proj_coeffs ** 2  # 每个特征向量上的投影能量
        
        band_energies = {}
        for band_name, (start, end) in bands.items():
            band_energy = np.sum(proj_energy[start:end]) / dir_norm_sq
            band_energies[band_name] = float(band_energy)
        
        row = f"  {role:>8s} " + " ".join(f"{band_energies[b]:10.4f}" for b in bands)
        print(row)
        results[role] = band_energies
    
    # ★ 关键分析: 探针方向的加权投影
    # W_U对方向v的"放大倍数" = sqrt(v^T W_U^T W_U v) / ||v||
    # = sqrt(sum_i lambda_i * (v·e_i)^2) / ||v||
    # 即: 探针方向的"有效特征值" = sum_i lambda_i * proj_energy_i / ||v||^2
    
    print(f"\n  探针方向的W_U'有效特征值' (加权平均):")
    for role in ROLE_NAMES:
        direction = probe_dirs[role]
        proj_coeffs = eigenvectors.T @ direction
        proj_energy = proj_coeffs ** 2
        
        weighted_eigenvalue = np.sum(eigenvalues * proj_energy) / np.sum(proj_energy)
        median_eigenvalue = np.median(eigenvalues)
        ratio = weighted_eigenvalue / median_eigenvalue
        
        print(f"    {role:>8s}: weighted_eigenvalue={weighted_eigenvalue:.2f}, "
              f"median={median_eigenvalue:.2f}, ratio={ratio:.2f}")
        results[role + '_weighted_eigen'] = float(weighted_eigenvalue)
        results[role + '_eigen_ratio'] = float(ratio)
    
    # 分析质心位移方向
    print(f"\n  质心位移方向的W_U'有效特征值':")
    centers = {}
    for ri, role in enumerate(ROLE_NAMES):
        mask = labels == ri
        centers[role] = H[mask].mean(axis=0)
    
    overall_center = H.mean(axis=0)
    displacements = {}
    for role in ROLE_NAMES:
        displacements[role] = centers[role] - overall_center
    
    for role in ROLE_NAMES:
        d = displacements[role]
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            continue
        
        d_hat = d / d_norm
        proj_coeffs = eigenvectors.T @ d_hat
        proj_energy = proj_coeffs ** 2
        
        weighted_eigenvalue = np.sum(eigenvalues * proj_energy) / np.sum(proj_energy)
        ratio = weighted_eigenvalue / np.median(eigenvalues)
        
        print(f"    {role:>8s}: weighted_eigenvalue={weighted_eigenvalue:.2f}, "
              f"ratio_to_median={ratio:.2f}")
        results[role + '_displacement_weighted_eigen'] = float(weighted_eigenvalue)
        results[role + '_displacement_ratio'] = float(ratio)
    
    results['eigenvalue_spectrum_stats'] = {
        'max': float(eigenvalues[0]),
        'min': float(eigenvalues[-1]),
        'median': float(np.median(eigenvalues)),
        'condition_number': float(eigenvalues[0] / max(eigenvalues[-1], 1e-10)),
    }
    
    return results


# ===== Exp2: 不同特征值频段的分类能力 =====
def exp2_band_classification(model, tokenizer, device):
    """在不同W_U特征值频段中分类语法角色"""
    print("\n" + "="*70)
    print("Exp2: 不同特征值频段的分类能力 ★★★★★")
    print("="*70)
    
    # 收集hidden states
    print("  收集hidden states...")
    H, labels = collect_hidden_states(model, tokenizer, device)
    print(f"  收集到 {len(H)} 个样本, d_model={H.shape[1]}")
    
    # 计算W_U特征谱
    W_U = get_W_U(model)
    eigenvalues, eigenvectors = compute_W_U_eigenspectrum(W_U)
    
    d_model = H.shape[1]
    
    # 定义频段(按特征值大小分)
    bands = {
        'top10': (0, d_model // 10),                    # top 10% 特征值方向 (256维)
        '10-25': (d_model // 10, d_model // 4),          # 10-25%
        '25-50': (d_model // 4, d_model // 2),           # 25-50%
        '50-75': (d_model // 2, 3 * d_model // 4),       # 50-75%
        '75-90': (3 * d_model // 4, 9 * d_model // 10),  # 75-90%
        'bottom10': (9 * d_model // 10, d_model),         # bottom 10% (256维)
    }
    
    results = {}
    
    print(f"\n  各频段特征值范围和分类准确率:")
    print(f"  {'Band':>10s} {'Dim':>5s} {'λ_range':>20s} {'CV':>8s}")
    
    for band_name, (start, end) in bands.items():
        dim = end - start
        
        # 投影到该频段
        U_band = eigenvectors[:, start:end]  # [d_model, dim]
        H_band = H @ U_band  # [N, dim]
        
        # 分类
        probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
        cv = cross_val_score(probe, H_band, labels, cv=5, scoring='accuracy')
        
        lambda_range = f"[{eigenvalues[start]:.1f}, {eigenvalues[end-1]:.1f}]"
        print(f"  {band_name:>10s} {dim:5d} {lambda_range:>20s} {cv.mean():8.4f}")
        
        results[band_name] = {
            'dim': dim,
            'lambda_start': float(eigenvalues[start]),
            'lambda_end': float(eigenvalues[end-1]),
            'cv_mean': float(cv.mean()),
            'cv_std': float(cv.std()),
        }
    
    # ★ 关键对比: W_U高能方向 vs 低能方向
    # 如果语法角色在低能方向, 则低能方向的分类应更好
    
    # 合并高能和低能
    U_high = eigenvectors[:, :d_model//2]  # top 50%
    U_low = eigenvectors[:, d_model//2:]   # bottom 50%
    
    H_high = H @ U_high
    H_low = H @ U_low
    
    probe_high = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    probe_low = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    
    cv_high = cross_val_score(probe_high, H_high, labels, cv=5, scoring='accuracy')
    cv_low = cross_val_score(probe_low, H_low, labels, cv=5, scoring='accuracy')
    
    print(f"\n  ★ 高能50%方向: CV={cv_high.mean():.4f}")
    print(f"  ★ 低能50%方向: CV={cv_low.mean():.4f}")
    print(f"  ★ 比值: low/high = {cv_low.mean()/cv_high.mean():.2f}")
    
    # 更细致: 10个等分频段
    print(f"\n  10等分频段分类准确率:")
    n_bins = 10
    bin_size = d_model // n_bins
    bin_results = []
    
    for bi in range(n_bins):
        start = bi * bin_size
        end = min((bi + 1) * bin_size, d_model)
        U_bin = eigenvectors[:, start:end]
        H_bin = H @ U_bin
        
        probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
        cv = cross_val_score(probe, H_bin, labels, cv=5, scoring='accuracy')
        
        print(f"    Bin {bi+1} ({start:4d}-{end:4d}): λ=[{eigenvalues[start]:.1f}, {eigenvalues[end-1]:.1f}], CV={cv.mean():.4f}")
        bin_results.append({
            'bin': bi + 1,
            'start': start,
            'end': end,
            'lambda_start': float(eigenvalues[start]),
            'lambda_end': float(eigenvalues[end-1]),
            'cv_mean': float(cv.mean()),
        })
    
    results['high50_cv'] = float(cv_high.mean())
    results['low50_cv'] = float(cv_low.mean())
    results['low_high_ratio'] = float(cv_low.mean() / cv_high.mean())
    results['bin_results'] = bin_results
    
    return results


# ===== Exp3: 不同频段操控的logits效率 =====
def exp3_band_manipulation_efficiency(model, tokenizer, device):
    """在不同W_U特征值频段上操控, 测量logits效率"""
    print("\n" + "="*70)
    print("Exp3: 不同频段操控的logits效率 ★★★★★")
    print("="*70)
    
    # 收集hidden states
    print("  收集hidden states...")
    H, labels = collect_hidden_states(model, tokenizer, device)
    
    # 训练探针
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)
    probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    probe.fit(H_scaled, labels)
    
    probe_dirs = {}
    for ri, role in enumerate(ROLE_NAMES):
        w = probe.coef_[ri]
        probe_dirs[role] = w / np.linalg.norm(w)
    
    # 计算W_U特征谱
    W_U = get_W_U(model)
    eigenvalues, eigenvectors = compute_W_U_eigenspectrum(W_U)
    
    d_model = H.shape[1]
    
    # 定义频段
    bands = {
        'top10': (0, d_model // 10),
        'mid40': (d_model // 10, d_model // 2),
        'low40': (d_model // 2, 9 * d_model // 10),
        'bottom10': (9 * d_model // 10, d_model),
    }
    
    # 获取final norm
    model_norm = None
    if hasattr(model, "model"):
        for attr in ["norm", "final_layernorm"]:
            if hasattr(model.model, attr):
                norm_obj = getattr(model.model, attr)
                if hasattr(norm_obj, "weight"):
                    model_norm = norm_obj.weight.detach().cpu().float().numpy()
                    break
    
    # 操控测试
    manipulation_tests = [
        ("The king ruled the kingdom", "king", "nsubj", "dobj"),
        ("The cat sat on the mat", "cat", "nsubj", "advmod"),
        ("The brave king fought hard", "brave", "amod", "nsubj"),
    ]
    
    alpha = 0.2
    
    results = {}
    
    print(f"\n  不同频段操控的logits效率 (alpha={alpha}):")
    print(f"  {'Case':>25s} " + " ".join(f"{band:>12s}" for band in bands))
    
    for sent, target, src_role, tgt_role in manipulation_tests:
        # 获取base hidden state
        layers_list = get_layers(model)
        last_layer = layers_list[-1]
        
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens_list, target)
        if dep_idx is None:
            continue
        
        # 获取base logits
        with torch.no_grad():
            output_base = model(**toks)
        logits_base = output_base.logits[0, dep_idx].float().cpu().numpy()
        
        # 获取base hidden state
        captured = {}
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured['h'] = output[0].detach().float().cpu().numpy()
            else:
                captured['h'] = output.detach().float().cpu().numpy()
        
        h_handle = last_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(**toks)
        h_handle.remove()
        
        if 'h' not in captured:
            continue
        
        h_vec = captured['h'][0, dep_idx, :]
        h_norm = float(np.linalg.norm(h_vec))
        
        # 探针方向差
        delta_dir = probe_dirs[tgt_role] - probe_dirs[src_role]
        delta_dir = delta_dir / max(np.linalg.norm(delta_dir), 1e-10)
        
        case_key = f"{src_role}→{tgt_role}"
        case_results = {}
        
        for band_name, (start, end) in bands.items():
            # 将操控方向投影到该频段
            U_band = eigenvectors[:, start:end]
            delta_band = U_band @ (U_band.T @ delta_dir)
            delta_band_norm = np.linalg.norm(delta_band)
            
            if delta_band_norm < 1e-10:
                case_results[band_name] = {
                    'delta_band_norm': 0,
                    'logits_change': 0,
                    'efficiency': 0,
                }
                continue
            
            # 扰动
            delta_h = alpha * h_norm * delta_band
            
            # 线性预测: W_U @ delta_h
            linear_logits_change = float(np.linalg.norm(W_U @ delta_h))
            
            # 实际模型forward
            def make_modify_hook(delta, pos_idx):
                def hook_fn2(module, input, output):
                    if isinstance(output, tuple):
                        h_out = output[0].detach().clone()
                        h_out[0, pos_idx, :] += torch.tensor(delta, dtype=h_out.dtype, device=h_out.device)
                        return (h_out,) + output[1:]
                    return output
                return hook_fn2
            
            hook_handle = last_layer.register_forward_hook(make_modify_hook(delta_h, dep_idx))
            with torch.no_grad():
                output_mod = model(**toks)
            hook_handle.remove()
            
            logits_mod = output_mod.logits[0, dep_idx].float().cpu().numpy()
            actual_logits_change = float(np.linalg.norm(logits_mod - logits_base))
            
            # 效率: logits变化 / 扰动范数
            perturb_norm = float(np.linalg.norm(delta_h))
            efficiency = actual_logits_change / max(perturb_norm, 1e-10)
            linear_efficiency = linear_logits_change / max(perturb_norm, 1e-10)
            
            case_results[band_name] = {
                'delta_band_norm': float(delta_band_norm),
                'perturb_norm': perturb_norm,
                'linear_logits_change': linear_logits_change,
                'actual_logits_change': actual_logits_change,
                'efficiency': efficiency,
                'linear_efficiency': linear_efficiency,
                'band_eigenvalue_range': f"[{eigenvalues[start]:.1f}, {eigenvalues[end-1]:.1f}]",
            }
        
        row = f"  {case_key:>25s} "
        for band in bands:
            eff = case_results.get(band, {}).get('efficiency', 0)
            row += f"{eff:12.4f}"
        print(row)
        
        results[case_key] = case_results
    
    # 汇总效率
    print(f"\n  平均logits效率:")
    for band in bands:
        effs = [results[k][band]['efficiency'] for k in results if band in results.get(k, {})]
        lin_effs = [results[k][band]['linear_efficiency'] for k in results if band in results.get(k, {})]
        if effs:
            print(f"    {band:>10s}: actual={np.mean(effs):.4f}, linear={np.mean(lin_effs):.4f}, "
                  f"λ_range={results[list(results.keys())[0]][band]['band_eigenvalue_range']}")
    
    return results


# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3])
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"CCL-O W_U特征值谱分析 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")
    
    try:
        if args.exp == 1:
            results = exp1_probe_spectrum_projection(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_band_classification(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_band_manipulation_efficiency(model, tokenizer, device)
        
        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 
                               f"cclo_exp{args.exp}_{args.model}_results.json")
        
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
                return tuple(convert(v) for v in obj)
            return obj
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {out_path}")
    
    finally:
        release_model(model)
        print(f"  模型已释放")


if __name__ == "__main__":
    main()
