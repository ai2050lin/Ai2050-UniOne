"""
CCXI(361): W_gate特征值/奇异值分析 — 方向反射的线性代数根源
============================================================

★★★★★ CCX核心发现:
  - MLP方向反射是训练习得的, 不是架构自发的!
  - 随机MLP的preservation≈0
  - W_gate是最关键的权重
  - 训练使W_gate的输出g大多数为负

★★★★★ CCXI核心问题:
  W_gate如何产生方向反射? 其线性代数根源是什么?

★★★★★ 实验设计:
  Part 1: W_gate的奇异值分布 (训练 vs 随机)
    - SVD: W_gate = U @ diag(S) @ Vt
    - 比较训练和随机W_gate的奇异值分布
    - 奇异值衰减率 → 秩结构

  Part 2: W_gate @ x_mean的分布分析
    - g = W_gate @ x_mean 的分量分布
    - 训练后为什么g大多数为负?
    - g的分布 vs W_gate行向量与x_mean的内积分布

  Part 3: W_gate的有效反射矩阵
    - 计算 J_eff = W_down @ diag(D1) @ W_gate (T1主导项)
    - 分析J_eff的特征值分布
    - 特征值实部 < 0 → 反射的根源

  Part 4: 语义方向在W_gate中的行为
    - 对语义方向(PCA) v, 计算 W_gate @ v
    - 比较 W_gate @ v 与 v 的关系
    - W_gate是否选择性翻转语义方向?

用法:
  python ccxi_gate_eigenvalue.py --model qwen3
  python ccxi_gate_eigenvalue.py --model glm4
"""

import argparse, os, sys, json, gc, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from scipy.linalg import svd
from scipy.stats import pearsonr

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model,
    MODEL_CONFIGS
)

TEMP = Path("tests/glm5_temp")

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


def safe_get_weight(param):
    """安全提取权重"""
    try:
        return param.detach().cpu().float().numpy()
    except (NotImplementedError, RuntimeError):
        try:
            return param.detach().dequantize().cpu().float().numpy()
        except:
            try:
                return param.data.cpu().float().numpy()
            except:
                return None


def safe_get_mlp_weights(layer, mlp_type):
    """安全提取MLP权重"""
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


def randomize_weights(W, seed=42):
    """生成相同shape和Frobenius范数的随机矩阵"""
    rng = np.random.RandomState(seed)
    shape = W.shape
    frob_norm = np.linalg.norm(W, 'fro')
    W_rand = rng.randn(*shape).astype(np.float32)
    W_rand = W_rand / np.linalg.norm(W_rand, 'fro') * frob_norm
    return W_rand


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def silu_derivative(x):
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
    return sig * (1.0 + x * (1.0 - sig))


def collect_mlp_inputs(model, tokenizer, device, model_name, domain_dict, sample_layers):
    """收集各层MLP输入"""
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


def analyze_gate_svd(model, model_name, sample_layers):
    """
    Part 1: W_gate的奇异值分布 (训练 vs 随机)
    """
    print("\n  === Part 1: W_gate奇异值分布分析 ===")
    
    layers = get_layers(model)
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    
    results = []
    
    for li in sample_layers:
        W_gate, W_up, W_down = safe_get_mlp_weights(layers[li], model_info.mlp_type)
        if W_gate is None:
            continue
        
        W_gate = W_gate.astype(np.float32)
        
        # 训练W_gate的SVD
        U_t, S_t, Vt_t = svd(W_gate, full_matrices=False)
        
        # 随机W_gate的SVD (3个种子)
        S_random_list = []
        for seed in range(3):
            W_r = randomize_weights(W_gate, seed=seed*100)
            _, S_r, _ = svd(W_r, full_matrices=False)
            S_random_list.append(S_r)
        S_random_avg = np.mean(S_random_list, axis=0)
        
        # 关键统计
        # 1. 奇异值衰减率
        n_sv = min(50, len(S_t))
        sv_ratio_10 = S_t[0] / max(S_t[9], 1e-10)  # 第1个/第10个
        sv_ratio_50 = S_t[0] / max(S_t[min(49, n_sv-1)], 1e-10)
        
        # 2. 有效秩
        total_energy = np.sum(S_t**2)
        cumulative = np.cumsum(S_t**2) / total_energy
        eff_rank_90 = np.searchsorted(cumulative, 0.9) + 1
        eff_rank_99 = np.searchsorted(cumulative, 0.99) + 1
        
        # 3. 随机SVD的对比
        random_ratio_10 = S_random_avg[0] / max(S_random_avg[9], 1e-10)
        random_total = np.sum(S_random_avg**2)
        random_cum = np.cumsum(S_random_avg**2) / random_total
        random_eff_rank_90 = np.searchsorted(random_cum, 0.9) + 1
        
        # 4. V_t (右奇异向量) 的分析
        # V_t的行是W_gate的右奇异向量
        # 这些向量定义了W_gate的输入空间结构
        
        result = {
            "layer": li,
            "gate_shape": list(W_gate.shape),
            # 训练SVD
            "sv_top5": [float(s) for s in S_t[:5]],
            "sv_ratio_10": float(sv_ratio_10),
            "sv_ratio_50": float(sv_ratio_50),
            "eff_rank_90": int(eff_rank_90),
            "eff_rank_99": int(eff_rank_99),
            # 随机SVD
            "random_sv_top5": [float(s) for s in S_random_avg[:5]],
            "random_sv_ratio_10": float(random_ratio_10),
            "random_eff_rank_90": int(random_eff_rank_90),
            # 对比
            "sv_ratio_trained_vs_random": float(sv_ratio_10 / max(random_ratio_10, 1e-10)),
            "eff_rank_ratio": float(eff_rank_90 / max(random_eff_rank_90, 1e-10)),
        }
        results.append(result)
        
        print(f"  L{li:2d}: S_top5={[f'{s:.1f}' for s in S_t[:5]]}, "
              f"ratio10={sv_ratio_10:.1f}(r={random_ratio_10:.1f}), "
              f"eff_rank90={eff_rank_90}(r={random_eff_rank_90}), "
              f"ratio_t/r={sv_ratio_10/random_ratio_10:.2f}")
    
    return results


def analyze_gate_output_distribution(model, model_name, mlp_inputs, sample_layers, N):
    """
    Part 2: W_gate @ x_mean的分布分析
    为什么训练后g大多数为负?
    """
    print("\n  === Part 2: W_gate输出分布分析 ===")
    
    layers = get_layers(model)
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    
    results = []
    
    for li in sample_layers:
        mlp_input = mlp_inputs.get(li)
        if mlp_input is None or not isinstance(mlp_input, np.ndarray):
            continue
        
        W_gate, W_up, W_down = safe_get_mlp_weights(layers[li], model_info.mlp_type)
        if W_gate is None:
            continue
        
        W_gate = W_gate.astype(np.float32)
        x_mean = mlp_input.mean(axis=0).astype(np.float32)
        
        # g = W_gate @ x_mean
        g = W_gate @ x_mean
        
        # 训练后的g分布
        g_neg_frac = np.mean(g < 0)
        g_mean = np.mean(g)
        g_std = np.std(g)
        g_median = np.median(g)
        g_skew = float(np.mean(((g - g_mean) / max(g_std, 1e-10))**3))
        
        # 随机W_gate的g分布
        g_random_stats = []
        for seed in range(3):
            W_r = randomize_weights(W_gate, seed=seed*100)
            g_r = W_r @ x_mean
            g_random_stats.append({
                "neg_frac": np.mean(g_r < 0),
                "mean": np.mean(g_r),
                "std": np.std(g_r),
            })
        
        # 分析W_gate的行向量与x_mean的内积
        # g[i] = W_gate[i] @ x_mean
        # W_gate的每一行与x_mean的点积
        
        # ★ 关键: W_gate的行向量均值偏移
        W_gate_row_means = W_gate.mean(axis=1)  # [intermediate]
        W_gate_row_norms = np.linalg.norm(W_gate, axis=1)  # [intermediate]
        
        # x_mean的范数
        x_mean_norm = np.linalg.norm(x_mean)
        
        # 行均值与g的关系
        # g[i] = W_gate[i] @ x_mean ≈ ||W_gate[i]|| * ||x_mean|| * cos(theta_i)
        # 如果cos(theta_i)多为负 → g多为负
        cos_angles = g / (W_gate_row_norms * x_mean_norm + 1e-10)
        cos_neg_frac = np.mean(cos_angles < 0)
        
        # ★ W_gate行向量与x_mean的对齐分析
        # 随机矩阵: cos(theta)的期望=0 → g的neg_frac≈0.5
        # 训练矩阵: cos(theta)的期望<0 → g的neg_frac>0.5
        
        result = {
            "layer": li,
            # g的统计 (训练)
            "g_neg_frac": float(g_neg_frac),
            "g_mean": float(g_mean),
            "g_std": float(g_std),
            "g_median": float(g_median),
            "g_skew": float(g_skew),
            # g的统计 (随机)
            "g_random_neg_frac": float(np.mean([s["neg_frac"] for s in g_random_stats])),
            "g_random_mean": float(np.mean([s["mean"] for s in g_random_stats])),
            # 行向量与x_mean的对齐
            "cos_neg_frac": float(cos_neg_frac),
            "x_mean_norm": float(x_mean_norm),
            # ★ 关键: 训练使W_gate的行向量与x_mean反向对齐
            "alignment_shift": float(g_neg_frac - np.mean([s["neg_frac"] for s in g_random_stats])),
        }
        results.append(result)
        
        print(f"  L{li:2d}: g_neg={g_neg_frac:.3f}(r={np.mean([s['neg_frac'] for s in g_random_stats]):.3f}), "
              f"g_mean={g_mean:.4f}, cos_neg={cos_neg_frac:.3f}, "
              f"shift={g_neg_frac - np.mean([s['neg_frac'] for s in g_random_stats]):+.3f}")
    
    return results


def analyze_effective_reflection_matrix(model, model_name, mlp_inputs, sample_layers, N):
    """
    Part 3: 有效反射矩阵的特征值分析
    
    T1 = W_down @ diag(silu'(g)⊙u) @ W_gate
    当silu(g)≈0时, T1主导J_MLP
    """
    print("\n  === Part 3: 有效反射矩阵分析 ===")
    
    layers = get_layers(model)
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    
    results = []
    
    for li in sample_layers:
        mlp_input = mlp_inputs.get(li)
        if mlp_input is None or not isinstance(mlp_input, np.ndarray):
            continue
        
        W_gate, W_up, W_down = safe_get_mlp_weights(layers[li], model_info.mlp_type)
        if W_gate is None or W_up is None or W_down is None:
            continue
        
        W_gate = W_gate.astype(np.float32)
        W_up = W_up.astype(np.float32)
        W_down = W_down.astype(np.float32)
        
        x_mean = mlp_input.mean(axis=0).astype(np.float32)
        
        g = W_gate @ x_mean
        u = W_up @ x_mean
        silu_prime_g = silu_derivative(g)
        
        # T1主导项: J_eff = W_down @ diag(silu'(g)⊙u) @ W_gate
        # 这个矩阵是 [d_model, d_model], 太大无法直接SVD
        # 但可以分析: diag(D1) @ W_gate 的奇异值
        
        D1 = silu_prime_g * u  # [intermediate]
        
        # ★ 方法: 通过随机投影估计J_eff的谱
        n_probes = 300
        np.random.seed(42)
        probes = np.random.randn(n_probes, d_model).astype(np.float32)
        probes = probes / np.linalg.norm(probes, axis=1, keepdims=True)
        
        rayleigh_list = []
        pres_list = []
        
        for i in range(n_probes):
            v = probes[i]
            # J_eff @ v = W_down @ (D1 ⊙ (W_gate @ v))
            Wgv = W_gate @ v
            Jv = W_down @ (D1 * Wgv)
            
            # Rayleigh quotient
            rq = np.dot(v, Jv)
            rayleigh_list.append(rq)
            
            # Direction preservation
            cos_pres = np.dot(Jv, v) / max(np.linalg.norm(Jv) * np.linalg.norm(v), 1e-10)
            pres_list.append(cos_pres)
        
        rq_arr = np.array(rayleigh_list)
        pres_arr = np.array(pres_list)
        
        # ★ 随机版本: 用随机D1 (从正态采样, 与D1同范数)
        D1_norm = np.linalg.norm(D1)
        D1_random_list = []
        for seed in range(3):
            rng = np.random.RandomState(seed)
            D1_r = rng.randn(len(D1)).astype(np.float32)
            D1_r = D1_r / np.linalg.norm(D1_r) * D1_norm
            
            pres_r_list = []
            for i in range(min(100, n_probes)):
                v = probes[i]
                Wgv = W_gate @ v
                Jv = W_down @ (D1_r * Wgv)
                cos_pres = np.dot(Jv, v) / max(np.linalg.norm(Jv) * np.linalg.norm(v), 1e-10)
                pres_r_list.append(cos_pres)
            D1_random_list.append(np.mean(pres_r_list))
        
        # ★ T2项分析: W_down @ diag(silu(g)) @ W_up
        silu_g = silu(g)
        D2 = silu_g
        
        pres_t2_list = []
        for i in range(min(100, n_probes)):
            v = probes[i]
            Wuv = W_up @ v
            Jv_t2 = W_down @ (D2 * Wuv)
            cos_pres = np.dot(Jv_t2, v) / max(np.linalg.norm(Jv_t2) * np.linalg.norm(v), 1e-10)
            pres_t2_list.append(cos_pres)
        
        result = {
            "layer": li,
            # T1 (J_eff) 的Rayleigh统计
            "T1_rq_mean": float(np.mean(rq_arr)),
            "T1_rq_std": float(np.std(rq_arr)),
            "T1_rq_pos_frac": float(np.mean(rq_arr > 0)),
            "T1_pres_mean": float(np.mean(pres_arr)),
            "T1_pres_pos_frac": float(np.mean(pres_arr > 0)),
            # 随机D1
            "T1_random_pres": float(np.mean(D1_random_list)),
            # T2的preservation
            "T2_pres_mean": float(np.mean(pres_t2_list)),
            # D1统计
            "D1_neg_frac": float(np.mean(D1 < 0)),
            "D1_mean": float(np.mean(D1)),
            "D1_std": float(np.std(D1)),
            # D2统计
            "D2_norm_frac": float(np.linalg.norm(D2) / max(np.linalg.norm(D1), 1e-10)),
        }
        results.append(result)
        
        print(f"  L{li:2d}: T1_pres={np.mean(pres_arr):+.4f}(rD1={np.mean(D1_random_list):+.4f}), "
              f"T1_RQ_pos={np.mean(rq_arr>0):.3f}, T2_pres={np.mean(pres_t2_list):+.4f}, "
              f"D2/D1_norm={result['D2_norm_frac']:.3f}")
    
    return results


def analyze_semantic_direction_behavior(model, model_name, emb_matrix, mlp_inputs, 
                                         sample_layers, N):
    """
    Part 4: 语义方向在W_gate中的行为
    """
    print("\n  === Part 4: 语义方向在W_gate中的行为 ===")
    
    layers = get_layers(model)
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    
    # PCA
    emb_centered = emb_matrix - emb_matrix.mean(axis=0)
    _, _, Vt_emb = svd(emb_centered, full_matrices=False)
    K = min(N - 1, d_model)
    V_pca = Vt_emb[:K].T  # [d_model, K]
    
    results = []
    
    for li in sample_layers:
        mlp_input = mlp_inputs.get(li)
        if mlp_input is None or not isinstance(mlp_input, np.ndarray):
            continue
        
        W_gate, W_up, W_down = safe_get_mlp_weights(layers[li], model_info.mlp_type)
        if W_gate is None:
            continue
        
        W_gate = W_gate.astype(np.float32)
        x_mean = mlp_input.mean(axis=0).astype(np.float32)
        
        # 对PCA方向分析 W_gate @ v_k
        n_dirs = min(K, 20)
        
        # W_gate @ v_k 的方向与 v_k 的关系
        gate_pres_pca = []  # cos(W_gate@v, v) — 注意W_gate是[intermediate, d_model]
        # 这里不能直接算preservation因为维度不同
        # 改为: 分析W_gate @ v_k 在W_gate行空间中的分布
        
        # ★ 更有意义: v_k → W_gate @ v_k → silu'(g)⊙(W_gate@v_k) → W_down @ (...)
        # 即完整JVP
        def _compute_jvp(W_gate, W_up, W_down, x, v):
            """计算MLP Jacobian-vector product"""
            g = W_gate @ x
            u = W_up @ x
            sig_g = silu(g)
            silu_prime_g = silu_derivative(g)
            Wgv = W_gate @ v
            Wuv = W_up @ v
            jvp = W_down @ (silu_prime_g * u * Wgv + sig_g * Wuv)
            return jvp
        
        pres_pca = []
        amp_pca = []
        pres_rand = []
        amp_rand = []
        
        np.random.seed(42)
        random_dirs = np.random.randn(50, d_model).astype(np.float32)
        random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
        
        if W_up is not None and W_down is not None:
            W_up = W_up.astype(np.float32)
            W_down = W_down.astype(np.float32)
            
            for k in range(n_dirs):
                v = V_pca[:, k].astype(np.float32)
                jvp = _compute_jvp(W_gate, W_up, W_down, x_mean, v)
                amp = np.linalg.norm(jvp) / max(np.linalg.norm(v), 1e-10)
                cos_pres = np.dot(jvp, v) / max(np.linalg.norm(jvp) * np.linalg.norm(v), 1e-10)
                pres_pca.append(cos_pres)
                amp_pca.append(amp)
            
            for i in range(50):
                v = random_dirs[i]
                jvp = _compute_jvp(W_gate, W_up, W_down, x_mean, v)
                amp = np.linalg.norm(jvp) / max(np.linalg.norm(v), 1e-10)
                cos_pres = np.dot(jvp, v) / max(np.linalg.norm(jvp) * np.linalg.norm(v), 1e-10)
                pres_rand.append(cos_pres)
                amp_rand.append(amp)
        
        # ★ W_gate单独的"反射性": W_gate @ v 与 W_gate 的右奇异向量
        # 如果W_gate的右奇异向量(输入方向)与PCA方向对齐 → 选择性
        
        # 计算 W_gate 的右奇异向量与PCA方向的余弦
        U_g, S_g, Vt_g = svd(W_gate, full_matrices=False)
        # Vt_g[:n_dirs] 是前n_dirs个右奇异向量 [n_dirs, d_model]
        
        alignment_pca_vs_gate = []
        for k in range(min(n_dirs, Vt_g.shape[0])):
            v_gate = Vt_g[k]  # [d_model]
            # 与前K个PCA方向的最大对齐
            cos_vals = np.abs(V_pca[:, :n_dirs].T @ v_gate)
            alignment_pca_vs_gate.append(float(np.max(cos_vals)))
        
        # 随机W_gate的对齐 (基线)
        W_gate_r = randomize_weights(W_gate, seed=42)
        _, _, Vt_r = svd(W_gate_r, full_matrices=False)
        alignment_random = []
        for k in range(min(n_dirs, Vt_r.shape[0])):
            v_r = Vt_r[k]
            cos_vals = np.abs(V_pca[:, :n_dirs].T @ v_r)
            alignment_random.append(float(np.max(cos_vals)))
        
        result = {
            "layer": li,
            # JVP preservation
            "jvp_pres_pca_mean": float(np.mean(pres_pca)) if pres_pca else None,
            "jvp_pres_rand_mean": float(np.mean(pres_rand)) if pres_rand else None,
            "jvp_amp_pca_mean": float(np.mean(amp_pca)) if amp_pca else None,
            "jvp_amp_rand_mean": float(np.mean(amp_rand)) if amp_rand else None,
            # W_gate右奇异向量与PCA的对齐
            "gate_alignment_mean": float(np.mean(alignment_pca_vs_gate)),
            "gate_alignment_top5": [float(a) for a in alignment_pca_vs_gate[:5]],
            "random_alignment_mean": float(np.mean(alignment_random)),
            "alignment_enhancement": float(np.mean(alignment_pca_vs_gate) / max(np.mean(alignment_random), 1e-10)),
        }
        results.append(result)
        
        pres_pca_str = f"{np.mean(pres_pca):+.4f}" if pres_pca else "N/A"
        pres_rand_str = f"{np.mean(pres_rand):+.4f}" if pres_rand else "N/A"
        print(f"  L{li:2d}: JVP_pres_pca={pres_pca_str}, JVP_pres_rand={pres_rand_str}, "
              f"gate_align={np.mean(alignment_pca_vs_gate):.3f}(r={np.mean(alignment_random):.3f}), "
              f"enhance={np.mean(alignment_pca_vs_gate)/max(np.mean(alignment_random),1e-10):.2f}x")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4"])
    parser.add_argument("--analysis", type=str, default="all",
                       choices=["all", "svd", "distribution", "reflection", "semantic"])
    args = parser.parse_args()
    
    model_name = args.model
    analysis = args.analysis
    
    print(f"\n{'='*70}")
    print(f"CCXI: W_gate特征值/奇异值分析 — {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
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
        print(f"  [CCXI] {model_name} loaded with 8bit, device={device}")
    else:
        model, tokenizer, device = load_model(model_name)
    
    model_info = get_model_info(model, model_name)
    print(f"  d_model={model_info.d_model}, n_layers={model_info.n_layers}")
    
    n_layers = model_info.n_layers
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    print(f"  采样层: {sample_layers}")
    
    # 收集MLP输入
    print(f"\n  收集MLP输入数据...")
    emb_matrix, mlp_inputs, N = collect_mlp_inputs(
        model, tokenizer, device, model_name, ANIMAL50, sample_layers
    )
    print(f"  收集完成: N={N}, d_model={emb_matrix.shape[1]}")
    
    all_results = {"model": model_name, "d_model": model_info.d_model, "n_layers": n_layers}
    
    # Part 1
    if analysis in ["all", "svd"]:
        svd_results = analyze_gate_svd(model, model_name, sample_layers)
        all_results["gate_svd"] = svd_results
    
    # Part 2
    if analysis in ["all", "distribution"]:
        dist_results = analyze_gate_output_distribution(
            model, model_name, mlp_inputs, sample_layers, N
        )
        all_results["gate_distribution"] = dist_results
    
    # Part 3
    if analysis in ["all", "reflection"]:
        refl_results = analyze_effective_reflection_matrix(
            model, model_name, mlp_inputs, sample_layers, N
        )
        all_results["effective_reflection"] = refl_results
    
    # Part 4
    if analysis in ["all", "semantic"]:
        sem_results = analyze_semantic_direction_behavior(
            model, model_name, emb_matrix, mlp_inputs, sample_layers, N
        )
        all_results["semantic_direction"] = sem_results
    
    # 保存结果
    output_path = TEMP / f"ccxi_{model_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")
    
    # 汇总
    print(f"\n{'='*70}")
    print(f"CCXI 汇总: {model_name}")
    print(f"{'='*70}")
    
    if "gate_svd" in all_results:
        print("\n--- Part 1: W_gate SVD ---")
        for r in all_results["gate_svd"]:
            print(f"  L{r['layer']:3d}: S_top={r['sv_top5'][:3]}, "
                  f"ratio10={r['sv_ratio_10']:.1f}(r={r['random_sv_ratio_10']:.1f}), "
                  f"eff90={r['eff_rank_90']}(r={r['random_eff_rank_90']}), "
                  f"t/r={r['sv_ratio_trained_vs_random']:.2f}")
    
    if "gate_distribution" in all_results:
        print("\n--- Part 2: W_gate输出分布 ---")
        for r in all_results["gate_distribution"]:
            print(f"  L{r['layer']:3d}: g_neg={r['g_neg_frac']:.3f}(r={r['g_random_neg_frac']:.3f}), "
                  f"cos_neg={r['cos_neg_frac']:.3f}, shift={r['alignment_shift']:+.3f}")
    
    if "effective_reflection" in all_results:
        print("\n--- Part 3: 有效反射矩阵 ---")
        for r in all_results["effective_reflection"]:
            print(f"  L{r['layer']:3d}: T1_pres={r['T1_pres_mean']:+.4f}(rD1={r['T1_random_pres']:+.4f}), "
                  f"T2_pres={r['T2_pres_mean']:+.4f}, D2/D1={r['D2_norm_frac']:.3f}")
    
    if "semantic_direction" in all_results:
        print("\n--- Part 4: 语义方向 ---")
        for r in all_results["semantic_direction"]:
            pp = r.get('jvp_pres_pca_mean', 'N/A')
            pr = r.get('jvp_pres_rand_mean', 'N/A')
            ga = r.get('gate_alignment_mean', 0)
            ra = r.get('random_alignment_mean', 0)
            enh = r.get('alignment_enhancement', 0)
            print(f"  L{r['layer']:3d}: JVP_pca={pp}, JVP_rand={pr}, "
                  f"align={ga:.3f}(r={ra:.3f}), enhance={enh:.2f}x")
    
    release_model(model)
    print(f"\nCCXI {model_name} 完成!")


if __name__ == "__main__":
    main()
