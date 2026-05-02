"""
CCX(360): 随机 vs 训练MLP Jacobian对比分析
===========================================

★★★★★ CCIX核心发现:
  - MLP是方向反射器: Jacobian preservation ≈ -0.5 (深层)
  - MLP不选择性处理语义维度
  - 残差连接 + MLP反射 = 几何保持

★★★★★ CCX核心问题:
  MLP的方向反射是架构自发的(SwiGLU固有的)还是训练习得的?
  
  如果随机MLP也有pres≈-0.5 → 架构自发 (SwiGLU的数学性质)
  如果随机MLP的pres不同 → 训练调整了反射强度

★★★★★ 实验设计:
  Part 1: 随机权重MLP Jacobian
    - 在训练模型提取的MLP输入点上, 用随机权重计算Jacobian
    - 随机权重: 与训练权重相同shape, 正态分布, 相同Frobenius范数
    - 比较preservation: 随机 vs 训练
  
  Part 2: SwiGLU几何效应的数学分析
    - 计算有效变换矩阵: M = W_down @ diag(silu(g) ⊙ silu'(g)) @ W_gate + ...
    - 分析M的特征值分布 (实部/虚部)
    - 特征值实部的符号 → 方向保持/反射
    
  Part 3: 逐步随机化分析
    - 只随机化W_gate: 保留silu的输入分布
    - 只随机化W_up: 保留门控信号
    - 只随机化W_down: 保留中间表示
    - 确定哪个权重矩阵贡献了方向反射

用法:
  python ccx_random_vs_trained_jacobian.py --model qwen3
  python ccx_random_vs_trained_jacobian.py --model glm4
"""

import argparse, os, sys, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from scipy.linalg import svd, eig
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, get_layer_weights, release_model,
    MODEL_CONFIGS
)

TEMP = Path("tests/glm5_temp")


def safe_get_weight(param):
    """安全提取权重 (兼容8bit量化模型)"""
    try:
        # 普通tensor
        w = param.detach().cpu().float().numpy()
        return w
    except (NotImplementedError, RuntimeError):
        # 8bit量化tensor: 需要先dequantize
        try:
            w = param.detach().dequantize().cpu().float().numpy()
            return w
        except Exception:
            # 如果dequantize也失败, 尝试直接用数据
            try:
                w = param.data.cpu().float().numpy()
                return w
            except Exception:
                return None


def safe_get_layer_mlp_weights(layer, d_model, mlp_type):
    """安全提取MLP权重 (兼容8bit量化)"""
    mlp = layer.mlp
    W_up = W_gate = W_down = None
    
    if mlp_type == "split_gate_up":
        if hasattr(mlp, 'up_proj'):
            W_up = safe_get_weight(mlp.up_proj.weight)
        if hasattr(mlp, 'gate_proj'):
            W_gate = safe_get_weight(mlp.gate_proj.weight)
        if hasattr(mlp, 'down_proj'):
            W_down = safe_get_weight(mlp.down_proj.weight)
    elif mlp_type == "merged_gate_up":
        if hasattr(mlp, 'gate_up_proj'):
            W_gate_up = safe_get_weight(mlp.gate_up_proj.weight)
            if W_gate_up is not None:
                W_gate = W_gate_up[:W_gate_up.shape[0]//2]
                W_up = W_gate_up[W_gate_up.shape[0]//2:]
        if hasattr(mlp, 'down_proj'):
            W_down = safe_get_weight(mlp.down_proj.weight)
    
    return W_gate, W_up, W_down

# 复用CCIX的领域定义
ANIMAL50 = {
    "dog": ["dog", "puppy", "hound", "canine", "pooch"],
    "cat": ["cat", "kitten", "feline", "tomcat", "kitty"],
    "wolf": ["wolf", "werewolf", "lupine", "coyote", "jackal"],
    "lion": ["lion", "tiger", "leopard", "cheetah", "panther"],
    "bird": ["bird", "sparrow", "robin", "finch", "wren"],
    "eagle": ["eagle", "hawk", "falcon", "vulture", "osprey"],
    "fish": ["fish", "trout", "salmon", "bass", "perch"],
    "shark": ["shark", "whale", "dolphin", "porpoise", "orca"],
    "snake": ["snake", "serpent", "viper", "cobra", "python"],
    "lizard": ["lizard", "gecko", "iguana", "chameleon", "salamander"],
    "horse": ["horse", "stallion", "mare", "pony", "colt"],
    "cow": ["cow", "cattle", "bull", "ox", "heifer"],
    "pig": ["pig", "swine", "hog", "boar", "sow"],
    "sheep": ["sheep", "lamb", "ram", "ewe", "fleece"],
    "goat": ["goat", "billy", "kid", "nanny", "caprine"],
    "chicken": ["chicken", "hen", "rooster", "cock", "poultry"],
    "duck": ["duck", "drake", "goose", "swan", "mallard"],
    "rabbit": ["rabbit", "bunny", "hare", "lagomorph", "cottontail"],
    "mouse": ["mouse", "rat", "rodent", "vole", "hamster"],
    "bear": ["bear", "grizzly", "polar", "cub", "ursine"],
    "elephant": ["elephant", "pachyderm", "tusk", "mammoth", "trunk"],
    "giraffe": ["giraffe", "ruminant", "hoof", "neck", "savanna"],
    "zebra": ["zebra", "stripes", "equine", "mustang", "striped"],
    "monkey": ["monkey", "ape", "primate", "chimp", "baboon"],
    "gorilla": ["gorilla", "silverback", "simian", "primate", "ape"],
    "penguin": ["penguin", "emperor", "frost", "ice", "flightless"],
    "owl": ["owl", "nocturnal", "hoot", "raptor", "bird"],
    "parrot": ["parrot", "macaw", "cockatoo", "tropical", "feather"],
    "turtle": ["turtle", "tortoise", "shell", "reptile", "terrapin"],
    "crocodile": ["crocodile", "alligator", "caiman", "reptile", "scale"],
    "frog": ["frog", "toad", "amphibian", "tadpole", "ribbit"],
    "butterfly": ["butterfly", "moth", "insect", "caterpillar", "cocoon"],
    "bee": ["bee", "honey", "wasp", "hive", "pollinator"],
    "ant": ["ant", "colony", "insect", "worker", "formic"],
    "spider": ["spider", "arachnid", "web", "silk", "tarantula"],
    "crab": ["crab", "lobster", "crustacean", "claw", "shellfish"],
    "octopus": ["octopus", "squid", "tentacle", "cephalopod", "ink"],
    "jellyfish": ["jellyfish", "tentacle", "sting", "medusa", "plankton"],
    "deer": ["deer", "stag", "doe", "fawn", "buck"],
    "fox": ["fox", "vixen", "canine", "crafty", "pelt"],
    "squirrel": ["squirrel", "chipmunk", "rodent", "acorn", "bushy"],
    "beaver": ["beaver", "dam", "rodent", "lodger", "gnaw"],
    "otter": ["otter", "river", "mustelid", "playful", "swim"],
    "kangaroo": ["kangaroo", "marsupial", "joey", "outback", "hop"],
    "koala": ["koala", "marsupial", "eucalyptus", "bear", "pouch"],
    "panda": ["panda", "bamboo", "bear", "china", "black"],
    "rhino": ["rhino", "rhinoceros", "horn", "pachyderm", "charge"],
    "hippo": ["hippo", "hippopotamus", "river", "africa", "water"],
    "camel": ["camel", "dromedary", "hump", "desert", "arabian"],
    "bat": ["bat", "nocturnal", "wing", "cave", "echolocation"],
    "seal": ["seal", "sea", "flipper", "pinniped", "arctic"],
}


def silu(x):
    """SiLU/Swish activation"""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def silu_derivative(x):
    """SiLU导数"""
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
    return sig * (1.0 + x * (1.0 - sig))


def compute_mlp_jvp(W_gate, W_up, W_down, x, v):
    """计算MLP Jacobian-vector product"""
    g = W_gate @ x
    u = W_up @ x
    silu_g = silu(g)
    silu_prime_g = silu_derivative(g)
    
    W_gate_v = W_gate @ v
    W_up_v = W_up @ v
    
    jf_v = (silu_prime_g * u) * W_gate_v + silu_g * W_up_v
    jmlp_v = W_down @ jf_v
    
    return jmlp_v


def compute_pca(points):
    """PCA"""
    N, d = points.shape
    K = min(N - 1, d)
    mean = points.mean(axis=0)
    centered = points - mean
    U, S, Vt = svd(centered, full_matrices=False)
    scores = U[:, :K] * S[:K]
    return scores, Vt[:K, :], S[:K], mean, K


def collect_mlp_inputs(model, tokenizer, device, model_name, domain_dict, sample_layers):
    """收集各层MLP输入和embedding"""
    layers = get_layers(model)
    embed_layer = model.get_input_embeddings()
    d_model = embed_layer.weight.shape[1]
    
    categories = list(domain_dict.keys())
    N = len(categories)
    
    emb_list = []
    mlp_inputs = {li: [] for li in sample_layers}
    
    for cat_idx, cat in enumerate(categories):
        main_word = domain_dict[cat][0]
        tok_ids = tokenizer.encode(main_word, add_special_tokens=False)
        if len(tok_ids) == 0:
            continue
        tok_id = tok_ids[-1]
        
        with torch.no_grad():
            emb = embed_layer.weight[tok_id].detach().float().cpu().numpy()
        emb_list.append(emb)
        
        input_ids = torch.tensor([[tok_id]], device=device)
        captured = {}
        
        def make_hook(key):
            def hook(module, inp, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook
        
        hooks = []
        for li in sample_layers:
            layer = layers[li]
            for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
                if hasattr(layer, ln_name):
                    ln_mod = getattr(layer, ln_name)
                    hooks.append(ln_mod.register_forward_hook(make_hook(f"LN{li}")))
                    break
        
        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids)
            except Exception as e:
                print(f"  Forward failed for {cat}: {e}")
        
        for h in hooks:
            h.remove()
        
        for li in sample_layers:
            ln_key = f"LN{li}"
            if ln_key in captured:
                mlp_inputs[li].append(captured[ln_key][0, -1, :].numpy())
        
        del captured
        gc.collect()
        
        if cat_idx % 10 == 0:
            print(f"  已收集 {cat_idx+1}/{N} tokens")
    
    emb_matrix = np.array(emb_list)
    for li in sample_layers:
        if len(mlp_inputs[li]) == N:
            mlp_inputs[li] = np.array(mlp_inputs[li])
        else:
            mlp_inputs[li] = None
    
    return emb_matrix, mlp_inputs, N


def randomize_weights(W, seed=None):
    """生成与W相同shape和Frobenius范数的随机矩阵"""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(42)
    
    shape = W.shape
    frob_norm = np.linalg.norm(W, 'fro')
    
    W_rand = rng.randn(*shape).astype(np.float32)
    W_rand = W_rand / np.linalg.norm(W_rand, 'fro') * frob_norm
    
    return W_rand


def analyze_random_vs_trained_jacobian(model, model_name, emb_matrix, mlp_inputs, 
                                         sample_layers, N, n_random_dirs=50, n_random_seeds=5):
    """
    Part 1: 随机 vs 训练MLP Jacobian对比
    
    核心思路:
    - 对每层, 提取训练后的MLP权重 W_gate, W_up, W_down
    - 生成多个随机种子版本的权重 (保持Frobenius范数)
    - 在训练模型的MLP输入点, 计算Jacobian preservation
    - 比较: 训练权重 vs 随机权重的preservation
    """
    print("\n  === Part 1: 随机 vs 训练MLP Jacobian对比 ===")
    
    layers = get_layers(model)
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    
    # Embedding PCA
    emb_centered = emb_matrix - emb_matrix.mean(axis=0)
    _, _, Vt_emb = svd(emb_centered, full_matrices=False)
    K = min(N - 1, d_model)
    V_pca = Vt_emb[:K].T  # [d_model, K]
    
    # 随机方向
    np.random.seed(42)
    random_dirs = np.random.randn(n_random_dirs, d_model).astype(np.float32)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
    
    results = []
    
    for li in sample_layers:
        mlp_input = mlp_inputs.get(li)
        if mlp_input is None or not isinstance(mlp_input, np.ndarray):
            continue
        
        W_gate, W_up, W_down = safe_get_layer_mlp_weights(layers[li], d_model, model_info.mlp_type)
        if W_gate is None or W_up is None or W_down is None:
            print(f"  L{li}: MLP权重不完整, 跳过")
            continue
        
        W_gate = W_gate.astype(np.float32)
        W_up = W_up.astype(np.float32)
        W_down = W_down.astype(np.float32)
        
        x_mean = mlp_input.mean(axis=0).astype(np.float32)
        
        # ===== 训练权重的Jacobian =====
        trained_pres_pca = []
        trained_pres_rand = []
        trained_amp_pca = []
        trained_amp_rand = []
        
        for k in range(min(K, 20)):  # 前20个PCA方向
            v = V_pca[:, k].astype(np.float32)
            jvp = compute_mlp_jvp(W_gate, W_up, W_down, x_mean, v)
            amp = np.linalg.norm(jvp) / max(np.linalg.norm(v), 1e-10)
            cos_pres = np.dot(jvp, v) / max(np.linalg.norm(jvp) * np.linalg.norm(v), 1e-10)
            trained_pres_pca.append(cos_pres)
            trained_amp_pca.append(amp)
        
        for i in range(n_random_dirs):
            v = random_dirs[i]
            jvp = compute_mlp_jvp(W_gate, W_up, W_down, x_mean, v)
            amp = np.linalg.norm(jvp) / max(np.linalg.norm(v), 1e-10)
            cos_pres = np.dot(jvp, v) / max(np.linalg.norm(jvp) * np.linalg.norm(v), 1e-10)
            trained_pres_rand.append(cos_pres)
            trained_amp_rand.append(amp)
        
        # ===== 随机权重的Jacobian (多个种子) =====
        rand_pres_pca_all = []
        rand_pres_rand_all = []
        rand_amp_pca_all = []
        rand_amp_rand_all = []
        
        for seed in range(n_random_seeds):
            W_gate_r = randomize_weights(W_gate, seed=seed * 100)
            W_up_r = randomize_weights(W_up, seed=seed * 100 + 1)
            W_down_r = randomize_weights(W_down, seed=seed * 100 + 2)
            
            pres_pca = []
            pres_rand = []
            amp_pca = []
            amp_rand = []
            
            for k in range(min(K, 20)):
                v = V_pca[:, k].astype(np.float32)
                jvp = compute_mlp_jvp(W_gate_r, W_up_r, W_down_r, x_mean, v)
                amp = np.linalg.norm(jvp) / max(np.linalg.norm(v), 1e-10)
                cos_pres = np.dot(jvp, v) / max(np.linalg.norm(jvp) * np.linalg.norm(v), 1e-10)
                pres_pca.append(cos_pres)
                amp_pca.append(amp)
            
            for i in range(n_random_dirs):
                v = random_dirs[i]
                jvp = compute_mlp_jvp(W_gate_r, W_up_r, W_down_r, x_mean, v)
                amp = np.linalg.norm(jvp) / max(np.linalg.norm(v), 1e-10)
                cos_pres = np.dot(jvp, v) / max(np.linalg.norm(jvp) * np.linalg.norm(v), 1e-10)
                pres_rand.append(cos_pres)
                amp_rand.append(amp)
            
            rand_pres_pca_all.append(np.mean(pres_pca))
            rand_pres_rand_all.append(np.mean(pres_rand))
            rand_amp_pca_all.append(np.mean(amp_pca))
            rand_amp_rand_all.append(np.mean(amp_rand))
        
        result = {
            "layer": li,
            # 训练权重
            "trained_pres_pca": float(np.mean(trained_pres_pca)),
            "trained_pres_rand": float(np.mean(trained_pres_rand)),
            "trained_amp_pca": float(np.mean(trained_amp_pca)),
            "trained_amp_rand": float(np.mean(trained_amp_rand)),
            # 随机权重 (跨种子均值)
            "random_pres_pca": float(np.mean(rand_pres_pca_all)),
            "random_pres_rand": float(np.mean(rand_pres_rand_all)),
            "random_pres_pca_std": float(np.std(rand_pres_pca_all)),
            "random_pres_rand_std": float(np.std(rand_pres_rand_all)),
            "random_amp_pca": float(np.mean(rand_amp_pca_all)),
            "random_amp_rand": float(np.mean(rand_amp_rand_all)),
            # 关键对比
            "pres_diff_trained_vs_random": float(np.mean(trained_pres_pca) - np.mean(rand_pres_pca_all)),
            "pres_ratio_trained_vs_random": float(np.mean(trained_pres_pca) / max(abs(np.mean(rand_pres_pca_all)), 1e-10)),
        }
        results.append(result)
        
        print(f"  L{li:2d}: trained_pres={np.mean(trained_pres_pca):+.4f}, "
              f"random_pres={np.mean(rand_pres_pca_all):+.4f}±{np.std(rand_pres_pca_all):.4f}, "
              f"diff={result['pres_diff_trained_vs_random']:+.4f}")
    
    return results


def analyze_swiglu_eigenvalues(model, model_name, mlp_inputs, sample_layers, N):
    """
    Part 2: SwiGLU有效变换矩阵的特征值分析
    
    核心思路:
    - MLP(x) = W_down @ (silu(W_gate @ x) ⊙ (W_up @ x))
    - 在均值输入x_mean处, 有效变换矩阵:
      J_MLP(x_mean) = W_down @ [diag(silu'(g) ⊙ u) @ W_gate + diag(silu(g)) @ W_up]
      其中 g = W_gate @ x_mean, u = W_up @ x_mean
    - 分析J_MLP的特征值分布
    - 特征值实部 < 0 → 方向反射
    - 特征值实部 > 0 → 方向保持
    """
    print("\n  === Part 2: SwiGLU有效变换矩阵的特征值分析 ===")
    
    layers = get_layers(model)
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    
    results = []
    
    for li in sample_layers:
        mlp_input = mlp_inputs.get(li)
        if mlp_input is None or not isinstance(mlp_input, np.ndarray):
            continue
        
        W_gate, W_up, W_down = safe_get_layer_mlp_weights(layers[li], d_model, model_info.mlp_type)
        if W_gate is None or W_up is None or W_down is None:
            continue
        
        W_gate = W_gate.astype(np.float32)
        W_up = W_up.astype(np.float32)
        W_down = W_down.astype(np.float32)
        
        x_mean = mlp_input.mean(axis=0).astype(np.float32)
        
        g = W_gate @ x_mean
        u = W_up @ x_mean
        silu_g = silu(g)
        silu_prime_g = silu_derivative(g)
        
        # 有效中间对角矩阵
        D1 = silu_prime_g * u  # diag(silu'(g) ⊙ u)
        D2 = silu_g            # diag(silu(g))
        
        # J_MLP = W_down @ (diag(D1) @ W_gate + diag(D2) @ W_up)
        # 但J_MLP是 [d_model, d_model], 太大无法直接计算特征值
        # 改用: 计算J_MLP的SVD, 用前K个奇异值和左/右奇异向量
        # 或者: 对小模型(d_model<4096), 直接计算
        
        # 方法1: 计算J_MLP^T @ J_MLP的特征值 (功率法)
        # 方法2: 对J_MLP @ v 做多次随机投影, 估计特征值分布
        
        # 实际方案: 计算J_MLP的低秩近似
        # J_MLP = W_down @ M, 其中 M = diag(D1) @ W_gate + diag(D2) @ W_up
        # M 是 [intermediate, d_model]
        # W_down 是 [d_model, intermediate]
        
        M = np.diag(D1) @ W_gate + np.diag(D2) @ W_up  # [intermediate, d_model]
        
        # ★ 分析对角元素D1和D2的统计
        d1_pos_frac = np.mean(D1 > 0)
        d1_neg_frac = np.mean(D1 < 0)
        d2_pos_frac = np.mean(D2 > 0)
        d2_neg_frac = np.mean(D2 < 0)
        
        # ★ M的行空间分析
        # M的每一行: m_i = D1[i] * W_gate[i] + D2[i] * W_up[i]
        # 如果D1[i]和D2[i]符号相反 → m_i是两个反向的行的差
        # 如果D1[i]和D2[i]同号 → m_i是两个同向的行的和
        
        same_sign_frac = np.mean((D1 > 0) == (D2 > 0))
        opp_sign_frac = 1.0 - same_sign_frac
        
        # ★ 随机投影估计J_MLP的谱
        n_probes = 200
        np.random.seed(42)
        probes = np.random.randn(n_probes, d_model).astype(np.float32)
        probes = probes / np.linalg.norm(probes, axis=1, keepdims=True)
        
        # J_MLP @ probe
        rayleigh_quotients = []
        preservations = []
        
        for i in range(n_probes):
            v = probes[i]
            jvp = compute_mlp_jvp(W_gate, W_up, W_down, x_mean, v)
            
            # Rayleigh quotient: v^T J v / v^T v
            rq = np.dot(v, jvp)
            rayleigh_quotients.append(rq)
            
            # 方向保持
            cos_pres = np.dot(jvp, v) / max(np.linalg.norm(jvp) * np.linalg.norm(v), 1e-10)
            preservations.append(cos_pres)
        
        rayleigh = np.array(rayleigh_quotients)
        pres_arr = np.array(preservations)
        
        # Rayleigh quotient的符号分布
        rq_pos_frac = np.mean(rayleigh > 0)
        rq_neg_frac = np.mean(rayleigh < 0)
        rq_mean = np.mean(rayleigh)
        rq_std = np.std(rayleigh)
        
        # Preservation的分布
        pres_pos_frac = np.mean(pres_arr > 0)
        pres_neg_frac = np.mean(pres_arr < 0)
        pres_mean = np.mean(pres_arr)
        
        # ★ 关键: 随机MLP的Rayleigh quotient期望值
        # 对随机矩阵A ~ N(0, σ²/d), v^T A v 期望=0 (因为E[a_ij]=0, v_i v_j E[a_ij]=0)
        # 但这里不是线性变换, 有SiLU非线性
        
        # ★ silu(g)的统计: silu对正输入保持正, 对负输入近似0
        # 所以D2 = silu(g): 大部分>0 (silu的输出总≥0)
        # D1 = silu'(g) * u: silu'总是>0, u可以是正或负
        # 所以D1的符号由u决定
        
        result = {
            "layer": li,
            # D1/D2统计
            "D1_pos_frac": float(d1_pos_frac),
            "D1_neg_frac": float(d1_neg_frac),
            "D2_pos_frac": float(d2_pos_frac),
            "D2_neg_frac": float(d2_neg_frac),
            "same_sign_frac": float(same_sign_frac),
            "opp_sign_frac": float(opp_sign_frac),
            # M统计
            "M_frob_norm": float(np.linalg.norm(M, 'fro')),
            "D1_mean": float(np.mean(D1)),
            "D2_mean": float(np.mean(D2)),
            "D1_std": float(np.std(D1)),
            "D2_std": float(np.std(D2)),
            # Rayleigh quotient
            "rq_pos_frac": float(rq_pos_frac),
            "rq_neg_frac": float(rq_neg_frac),
            "rq_mean": float(rq_mean),
            "rq_std": float(rq_std),
            # Preservation
            "pres_pos_frac": float(pres_pos_frac),
            "pres_neg_frac": float(pres_neg_frac),
            "pres_mean": float(pres_mean),
        }
        results.append(result)
        
        print(f"  L{li:2d}: pres_mean={pres_mean:+.4f}, pres_pos={pres_pos_frac:.3f}, "
              f"RQ_mean={rq_mean:+.4f}, RQ_pos={rq_pos_frac:.3f}, "
              f"D1+={d1_pos_frac:.3f} D2+={d2_pos_frac:.3f} same={same_sign_frac:.3f}")
    
    return results


def analyze_partial_randomization(model, model_name, mlp_inputs, sample_layers, N, 
                                   n_random_dirs=50, n_seeds=3):
    """
    Part 3: 逐步随机化分析 — 确定哪个权重贡献了方向反射
    
    三种随机化方案:
    A. 只随机化W_gate (保留W_up, W_down)
    B. 只随机化W_up (保留W_gate, W_down)  
    C. 只随机化W_down (保留W_gate, W_up)
    D. 全部随机化
    E. 训练权重 (对照)
    """
    print("\n  === Part 3: 逐步随机化分析 ===")
    
    layers = get_layers(model)
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    
    np.random.seed(42)
    random_dirs = np.random.randn(n_random_dirs, d_model).astype(np.float32)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
    
    results = []
    
    for li in sample_layers:
        mlp_input = mlp_inputs.get(li)
        if mlp_input is None or not isinstance(mlp_input, np.ndarray):
            continue
        
        W_gate, W_up, W_down = safe_get_layer_mlp_weights(layers[li], d_model, model_info.mlp_type)
        if W_gate is None or W_up is None or W_down is None:
            continue
        
        W_gate = W_gate.astype(np.float32)
        W_up = W_up.astype(np.float32)
        W_down = W_down.astype(np.float32)
        
        x_mean = mlp_input.mean(axis=0).astype(np.float32)
        
        # 五种方案
        schemes = {
            "trained": (W_gate, W_up, W_down),
        }
        
        # 对每种随机化方案, 用多个种子
        for seed in range(n_seeds):
            W_gate_r = randomize_weights(W_gate, seed=seed * 100)
            W_up_r = randomize_weights(W_up, seed=seed * 100 + 1)
            W_down_r = randomize_weights(W_down, seed=seed * 100 + 2)
            
            schemes[f"rand_gate_s{seed}"] = (W_gate_r, W_up, W_down)
            schemes[f"rand_up_s{seed}"] = (W_gate, W_up_r, W_down)
            schemes[f"rand_down_s{seed}"] = (W_gate, W_up, W_down_r)
            schemes[f"rand_all_s{seed}"] = (W_gate_r, W_up_r, W_down_r)
        
        # 计算每种方案的preservation
        scheme_pres = {}
        
        for scheme_name, (Wg, Wu, Wd) in schemes.items():
            pres_list = []
            for i in range(n_random_dirs):
                v = random_dirs[i]
                jvp = compute_mlp_jvp(Wg, Wu, Wd, x_mean, v)
                cos_pres = np.dot(jvp, v) / max(np.linalg.norm(jvp) * np.linalg.norm(v), 1e-10)
                pres_list.append(cos_pres)
            scheme_pres[scheme_name] = np.mean(pres_list)
        
        # 按方案类型汇总
        trained_pres = scheme_pres["trained"]
        
        rand_gate_pres = np.mean([scheme_pres[k] for k in scheme_pres if k.startswith("rand_gate_")])
        rand_up_pres = np.mean([scheme_pres[k] for k in scheme_pres if k.startswith("rand_up_")])
        rand_down_pres = np.mean([scheme_pres[k] for k in scheme_pres if k.startswith("rand_down_")])
        rand_all_pres = np.mean([scheme_pres[k] for k in scheme_pres if k.startswith("rand_all_")])
        
        result = {
            "layer": li,
            "trained_pres": float(trained_pres),
            "rand_gate_pres": float(rand_gate_pres),
            "rand_up_pres": float(rand_up_pres),
            "rand_down_pres": float(rand_down_pres),
            "rand_all_pres": float(rand_all_pres),
            # 哪个随机化最改变preservation?
            "delta_gate": float(trained_pres - rand_gate_pres),
            "delta_up": float(trained_pres - rand_up_pres),
            "delta_down": float(trained_pres - rand_down_pres),
            "delta_all": float(trained_pres - rand_all_pres),
        }
        results.append(result)
        
        print(f"  L{li:2d}: trained={trained_pres:+.4f}, "
              f"r_gate={rand_gate_pres:+.4f}(Δ={trained_pres-rand_gate_pres:+.4f}), "
              f"r_up={rand_up_pres:+.4f}(Δ={trained_pres-rand_up_pres:+.4f}), "
              f"r_down={rand_down_pres:+.4f}(Δ={trained_pres-rand_down_pres:+.4f}), "
              f"r_all={rand_all_pres:+.4f}(Δ={trained_pres-rand_all_pres:+.4f})")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4"])
    parser.add_argument("--analysis", type=str, default="all",
                       choices=["all", "random_jacobian", "eigenvalue", "partial_random"])
    args = parser.parse_args()
    
    model_name = args.model
    analysis = args.analysis
    
    print(f"\n{'='*70}")
    print(f"CCX: 随机 vs 训练MLP Jacobian对比分析 — {model_name}")
    print(f"{'='*70}")
    
    # 加载模型: GLM4用8bit量化避免OOM
    if model_name == "glm4":
        from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
        cfg = MODEL_CONFIGS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
        model.eval()
        device = next(model.parameters()).device
        print(f"  [CCX] {model_name} loaded with 8bit quantization, device={device}")
    else:
        model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  d_model={model_info.d_model}, n_layers={model_info.n_layers}")
    
    n_layers = model_info.n_layers
    
    # 采样层 (与CCIX一致)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    print(f"  采样层: {sample_layers}")
    
    # 收集MLP输入数据
    print(f"\n  收集MLP输入数据...")
    emb_matrix, mlp_inputs, N = collect_mlp_inputs(
        model, tokenizer, device, model_name, ANIMAL50, sample_layers
    )
    print(f"  收集完成: N={N}, d_model={emb_matrix.shape[1]}")
    
    all_results = {"model": model_name, "d_model": model_info.d_model, "n_layers": n_layers}
    
    # Part 1: 随机 vs 训练Jacobian
    if analysis in ["all", "random_jacobian"]:
        random_jacobian_results = analyze_random_vs_trained_jacobian(
            model, model_name, emb_matrix, mlp_inputs, sample_layers, N
        )
        all_results["random_jacobian"] = random_jacobian_results
    
    # Part 2: SwiGLU特征值分析
    if analysis in ["all", "eigenvalue"]:
        eigenvalue_results = analyze_swiglu_eigenvalues(
            model, model_name, mlp_inputs, sample_layers, N
        )
        all_results["swiglu_eigenvalue"] = eigenvalue_results
    
    # Part 3: 逐步随机化
    if analysis in ["all", "partial_random"]:
        partial_results = analyze_partial_randomization(
            model, model_name, mlp_inputs, sample_layers, N
        )
        all_results["partial_random"] = partial_results
    
    # 保存结果
    output_path = TEMP / f"ccx_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")
    
    # ===== 汇总分析 =====
    print(f"\n{'='*70}")
    print(f"CCX 汇总分析: {model_name}")
    print(f"{'='*70}")
    
    # Part 1 汇总
    if "random_jacobian" in all_results:
        print(f"\n--- Part 1: 随机 vs 训练Jacobian ---")
        print(f"  {'Layer':>5s}  {'Trained':>8s}  {'Random':>8s}  {'Δ(t-r)':>8s}  {'Ratio':>6s}  {'判断':>6s}")
        for r in all_results["random_jacobian"]:
            judgment = "训练!" if abs(r['pres_diff_trained_vs_random']) > 0.1 else "架构?"
            print(f"  L{r['layer']:3d}  {r['trained_pres_pca']:+8.4f}  "
                  f"{r['random_pres_pca']:+8.4f}  {r['pres_diff_trained_vs_random']:+8.4f}  "
                  f"{r['pres_ratio_trained_vs_random']:6.2f}  {judgment:>6s}")
    
    # Part 2 汇总
    if "swiglu_eigenvalue" in all_results:
        print(f"\n--- Part 2: SwiGLU几何分析 ---")
        print(f"  {'Layer':>5s}  {'pres_m':>7s}  {'pres+':>6s}  {'pres-':>6s}  "
              f"{'RQ_m':>7s}  {'RQ+':>6s}  {'D1+':>5s}  {'D2+':>5s}  {'same':>5s}")
        for r in all_results["swiglu_eigenvalue"]:
            print(f"  L{r['layer']:3d}  {r['pres_mean']:+7.4f}  {r['pres_pos_frac']:6.3f}  "
                  f"{r['pres_neg_frac']:6.3f}  {r['rq_mean']:+7.4f}  {r['rq_pos_frac']:6.3f}  "
                  f"{r['D1_pos_frac']:5.3f}  {r['D2_pos_frac']:5.3f}  {r['same_sign_frac']:5.3f}")
    
    # Part 3 汇总
    if "partial_random" in all_results:
        print(f"\n--- Part 3: 逐步随机化 ---")
        print(f"  {'Layer':>5s}  {'Trained':>8s}  {'r_gate':>8s}  {'r_up':>8s}  "
              f"{'r_down':>8s}  {'r_all':>8s}  {'max_Δ':>6s}  {'来源':>8s}")
        for r in all_results["partial_random"]:
            deltas = {
                "gate": abs(r['delta_gate']),
                "up": abs(r['delta_up']),
                "down": abs(r['delta_down']),
            }
            max_source = max(deltas, key=deltas.get)
            max_delta = deltas[max_source]
            print(f"  L{r['layer']:3d}  {r['trained_pres']:+8.4f}  "
                  f"{r['rand_gate_pres']:+8.4f}  {r['rand_up_pres']:+8.4f}  "
                  f"{r['rand_down_pres']:+8.4f}  {r['rand_all_pres']:+8.4f}  "
                  f"{max_delta:+6.4f}  {max_source:>8s}")
    
    # 核心判断
    print(f"\n{'='*70}")
    print("★★★★★ CCX核心判断")
    print(f"{'='*70}")
    
    if "random_jacobian" in all_results:
        trained_mid = [r['trained_pres_pca'] for r in all_results["random_jacobian"] 
                       if 3 <= r['layer'] <= n_layers * 0.7]
        random_mid = [r['random_pres_pca'] for r in all_results["random_jacobian"]
                      if 3 <= r['layer'] <= n_layers * 0.7]
        
        if trained_mid and random_mid:
            t_mean = np.mean(trained_mid)
            r_mean = np.mean(random_mid)
            print(f"\n  中层(L3-L{n_layers*7//10})preservation:")
            print(f"    训练权重: {t_mean:+.4f}")
            print(f"    随机权重: {r_mean:+.4f}")
            print(f"    差值: {t_mean - r_mean:+.4f}")
            
            if abs(t_mean - r_mean) < 0.05:
                print(f"\n  ★★★★★ 结论: pres≈{t_mean:+.2f}是架构自发的!")
                print(f"    随机MLP和训练MLP的preservation几乎相同")
                print(f"    → SwiGLU的数学结构决定了方向反射")
                print(f"    → 训练不改变反射的方向, 只调整强度")
            elif abs(t_mean - r_mean) > 0.15:
                print(f"\n  ★★★★★ 结论: pres差异显著, 训练调整了反射!")
                print(f"    训练使pres从{r_mean:+.2f}变为{t_mean:+.2f}")
                print(f"    → 训练确实改变了MLP的几何行为")
            else:
                print(f"\n  ★★★★ 结论: 部分架构自发, 部分训练调整")
                print(f"    差异={t_mean-r_mean:+.4f}, 不大不小")
                print(f"    → SwiGLU提供了基底反射, 训练在此基础上微调")
    
    if "partial_random" in all_results:
        print(f"\n  逐步随机化 — 哪个权重最关键?")
        gate_deltas = [abs(r['delta_gate']) for r in all_results["partial_random"]]
        up_deltas = [abs(r['delta_up']) for r in all_results["partial_random"]]
        down_deltas = [abs(r['delta_down']) for r in all_results["partial_random"]]
        
        print(f"    gate随机化的平均Δ: {np.mean(gate_deltas):.4f}")
        print(f"    up随机化的平均Δ:   {np.mean(up_deltas):.4f}")
        print(f"    down随机化的平均Δ:  {np.mean(down_deltas):.4f}")
        
        max_delta = max(np.mean(gate_deltas), np.mean(up_deltas), np.mean(down_deltas))
        if max_delta == np.mean(gate_deltas):
            print(f"    → W_gate最关键! gate控制了silu的输入分布")
        elif max_delta == np.mean(up_deltas):
            print(f"    → W_up最关键! up控制了信息传递")
        else:
            print(f"    → W_down最关键! down控制了输出投影方向")
    
    release_model(model)
    print(f"\nCCX {model_name} 完成!")


if __name__ == "__main__":
    main()
