"""
Phase XLIV-P260/261/262/263/264: FFN非线性机制破解
====================================================

Phase XLI-XLIII核心结论:
  1. 编码双轴定理: 方向编码(骨架) + 稀疏激活模式(肌肉)
  2. G项占组合编码70-94%, 是FFN非线性输出
  3. 两模型FFN策略根本不同: Qwen3=冗余编码, DS7B=正交分解+路由放大

五大实验:
  P260: G项的FFN逐神经元分解
    - 对每个FFN神经元, 计算gate_i, post_i, out_i
    - 分析哪些神经元的gate值受"苹果vs香蕉"影响最大
    - 分析这些神经元的输出方向与G项方向的关系
    - 预期: G项由少数关键神经元的非线性激活产生

  P261: 符号回归搜索G项的数学形式
    - 不预设函数形式, 用多项式+条件函数搜索G ≈ f(B_fruit, A_red, A_sweet)
    - 包含: 线性、二次、三次、交互项、条件门控、分段线性
    - 评估R2和cos指标

  P262: 条件计算分析
    - 同一FFN层, 不同输入x的条件响应函数FFN(x)
    - 水果vs动物、同族不同词(苹果vs香蕉)、修饰vs无修饰
    - FFN作为条件映射的特性

  P263: 注意力-FFN精确交互
    - 先做Attn, 分析Attn选择了哪些上下文信息
    - 然后做FFN, 分析FFN如何处理Attn的输出
    - 上下文信息如何条件化FFN的gate选择

  P264: FFN策略等价性分析
    - Qwen3/DS7B两种FFN策略在LCS层面是否等价
    - 参数空间中是否存在连续路径

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免OOM)
数据规模: 5家族(60词) + 3类属性(36词) + 30个组合 + 7个prompt模板
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys, gc, time, json, argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "phase_xliv_log.txt"

class Logger:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w', encoding='utf-8', buffering=1)
    def log(self, msg):
        ts = time.strftime('%H:%M:%S')
        self.f.write(f"{ts} {msg}\n")
        self.f.flush()
        print(f"  [{ts}] {msg}")
    def close(self): self.f.close()

L = Logger(LOG_FILE)

def get_model_path(model_name):
    paths = {
        "qwen3": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
        "deepseek7b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    }
    return paths.get(model_name)

def load_model(model_name):
    p = get_model_path(model_name)
    p_abs = os.path.abspath(p)
    tok = AutoTokenizer.from_pretrained(
        p_abs, trust_remote_code=True,
        local_files_only=True,
        use_fast=False
    )
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        p_abs, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager", device_map="cpu"
    )
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")
    mdl.eval()
    device = next(mdl.parameters()).device
    return mdl, tok, device

# ===================== 增强版数据集 =====================
STIMULI = {
    # 5个语义家族, 每个12个成员
    "fruit_family": ["apple","banana","pear","orange","grape","mango","strawberry","watermelon","cherry","peach","lemon","lime"],
    "animal_family": ["cat","dog","rabbit","horse","lion","eagle","elephant","dolphin","tiger","bear","fox","wolf"],
    "vehicle_family": ["car","bus","train","plane","boat","bicycle","truck","helicopter","motorcycle","ship","subway","taxi"],
    "furniture_family": ["chair","table","desk","sofa","bed","cabinet","shelf","bench","stool","dresser","couch","armchair"],
    "tool_family": ["hammer","wrench","screwdriver","saw","drill","pliers","knife","scissors","shovel","rake","axe","chisel"],
    # 3类属性
    "color_attrs": ["red","green","yellow","orange","brown","white","blue","black","pink","purple","gray","gold"],
    "taste_attrs": ["sweet","sour","bitter","salty","crisp","soft","spicy","fresh","tart","savory","rich","mild"],
    "size_attrs": ["big","small","tall","short","long","wide","thin","thick","heavy","light","huge","tiny"],
    # 组合词 (更多数据量)
    "fruit_color_combos": [
        "red apple","green apple","yellow banana","orange orange","green pear",
        "red grape","yellow mango","red cherry","pink peach","yellow lemon",
        "red strawberry","green watermelon","purple grape","brown pear","gold mango",
    ],
    "fruit_taste_combos": [
        "sweet apple","sour apple","sweet banana","sour orange","sweet pear",
        "bitter grape","sweet mango","sweet cherry","tart lemon","sweet peach",
        "crisp apple","soft banana","fresh strawberry","rich mango","mild pear",
    ],
    "animal_color_combos": [
        "brown cat","white dog","brown rabbit","black horse","golden eagle",
        "white cat","black dog","orange tiger","brown bear","red fox",
        "gray wolf","black cat","white rabbit","brown horse","golden dog",
    ],
    "animal_size_combos": [
        "big elephant","small cat","tall horse","tiny rabbit","huge lion",
        "big dog","small fox","long snake","heavy bear","light eagle",
    ],
    "vehicle_color_combos": [
        "red car","blue bus","white plane","black truck","green bicycle",
        "yellow taxi","gray ship","brown boat",
    ],
    "furniture_size_combos": [
        "big table","small chair","long desk","tall cabinet","wide sofa",
        "heavy bed","light shelf",
    ],
}

# 7个prompt模板
PROMPT_TEMPLATES = [
    "The {word} is",
    "A {word} can be",
    "This {word} has",
    "I saw a {word}",
    "The {word} was",
    "My {word} is",
    "That {word} looks",
]

# 测试三元组 (P258/P260用)
TEST_TRIPLES = [
    ("apple","red","red apple"), ("apple","sweet","sweet apple"),
    ("apple","green","green apple"), ("banana","yellow","yellow banana"),
    ("banana","sweet","sweet banana"), ("pear","green","green pear"),
    ("cat","brown","brown cat"), ("dog","white","white dog"),
    ("car","red","red car"), ("apple","fresh","fresh apple"),
    ("horse","black","black horse"), ("apple","sour","sour apple"),
    ("cherry","red","red cherry"), ("grape","purple","purple grape"),
    ("mango","yellow","yellow mango"), ("lemon","tart","tart lemon"),
    ("peach","sweet","sweet peach"), ("orange","sour","sour orange"),
    ("strawberry","sweet","sweet strawberry"), ("apple","crisp","crisp apple"),
    ("banana","soft","soft banana"), ("pear","fresh","fresh pear"),
    ("elephant","big","big elephant"), ("cat","small","small cat"),
    ("horse","tall","tall horse"), ("table","big","big table"),
    ("car","blue","blue car"), ("dog","black","black dog"),
    ("tiger","orange","orange tiger"), ("bear","brown","brown bear"),
]


# ==================== 数据收集 ====================

def collect_all_data(mdl, tok, device):
    """收集所有hidden states + FFN中间激活"""
    n_layers = mdl.config.num_hidden_layers
    
    all_single = sorted(set(
        STIMULI["fruit_family"] + STIMULI["animal_family"] + 
        STIMULI["vehicle_family"] + STIMULI["furniture_family"] + STIMULI["tool_family"] +
        STIMULI["color_attrs"] + STIMULI["taste_attrs"] + STIMULI["size_attrs"]
    ))
    all_combos = sorted(set(
        STIMULI["fruit_color_combos"] + STIMULI["fruit_taste_combos"] + 
        STIMULI["animal_color_combos"] + STIMULI["animal_size_combos"] +
        STIMULI["vehicle_color_combos"] + STIMULI["furniture_size_combos"]
    ))
    
    L.log(f"收集: {len(all_single)}单字 + {len(all_combos)}组合词")
    
    single_hs = {}
    for i, word in enumerate(all_single):
        avg_hs = None
        for template in PROMPT_TEMPLATES:
            prompt = template.replace("{word}", word)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl(**inputs, output_hidden_states=True)
            hs = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
            avg_hs = hs if avg_hs is None else avg_hs + hs
            del out
        single_hs[word] = avg_hs / len(PROMPT_TEMPLATES)
        if (i+1) % 20 == 0:
            L.log(f"  单字 [{i+1}/{len(all_single)}]")
    
    combo_hs = {}
    for i, combo in enumerate(all_combos):
        avg_hs = None
        for template in PROMPT_TEMPLATES:
            prompt = template.replace("{word}", combo)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl(**inputs, output_hidden_states=True)
            hs = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
            avg_hs = hs if avg_hs is None else avg_hs + hs
            del out
        combo_hs[combo] = avg_hs / len(PROMPT_TEMPLATES)
        if (i+1) % 10 == 0:
            L.log(f"  组合 [{i+1}/{len(all_combos)}]")
    
    gc.collect()
    return single_hs, combo_hs


def extract_ffn_weights(mdl, key_layers=None):
    """提取指定层的FFN权重矩阵, 支持多种架构:
    - Llama/Qwen: gate_proj + up_proj + down_proj
    - GLM4: gate_up_proj(融合) + down_proj
    - TransformerLens: W_gate + W_in + W_out
    
    key_layers: 只提取这些层, None则提取全部
    """
    ffn_weights = {}
    layers = None
    if hasattr(mdl, 'model') and hasattr(mdl.model, 'layers'):
        layers = mdl.model.layers
    elif hasattr(mdl, 'transformer') and hasattr(mdl.transformer, 'h'):
        layers = mdl.transformer.h
    elif hasattr(mdl, 'transformer') and hasattr(mdl.transformer, 'encoder') and hasattr(mdl.transformer.encoder, 'layers'):
        layers = mdl.transformer.encoder.layers
    
    if layers is None:
        L.log("警告: 无法找到模型层!")
        return ffn_weights
    
    n_total = len(layers)
    target_layers = set(key_layers) if key_layers else set(range(n_total))
    
    for li, layer in enumerate(layers):
        if li not in target_layers:
            continue
        mlp = layer.mlp if hasattr(layer, 'mlp') else (layer.ffn if hasattr(layer, 'ffn') else None)
        if mlp is None:
            continue
        wd = {}
        
        # 标准Llama格式: gate_proj + up_proj + down_proj
        if hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj'):
            wd['W_gate'] = mlp.gate_proj.weight.detach().float().cpu().T  # [d_model, d_mlp]
            wd['W_in'] = mlp.up_proj.weight.detach().float().cpu().T
            wd['W_out'] = mlp.down_proj.weight.detach().float().cpu().T
        # GLM4融合格式: gate_up_proj + down_proj
        elif hasattr(mlp, 'gate_up_proj') and hasattr(mlp, 'down_proj'):
            W_fused = mlp.gate_up_proj.weight.detach().float().cpu()  # [2*d_mlp, d_model]
            d_mlp_half = W_fused.shape[0] // 2
            wd['W_gate'] = W_fused[:d_mlp_half].T.clone()  # [d_model, d_mlp]
            wd['W_in'] = W_fused[d_mlp_half:].T.clone()    # [d_model, d_mlp]
            del W_fused  # 立即释放融合权重
            wd['W_out'] = mlp.down_proj.weight.detach().float().cpu().T  # [d_mlp, d_model]
        # TransformerLens格式
        elif hasattr(mlp, 'W_gate'):
            wd['W_gate'] = mlp.W_gate.detach().float().cpu()
            wd['W_in'] = mlp.W_in.detach().float().cpu()
            wd['W_out'] = mlp.W_out.detach().float().cpu()
        else:
            # 尝试其他常见名称
            found = False
            for gate_name in ['gate_proj', 'w1', 'W_gate']:
                for in_name in ['up_proj', 'w3', 'W_in']:
                    for out_name in ['down_proj', 'w2', 'W_out']:
                        if hasattr(mlp, gate_name) and hasattr(mlp, in_name) and hasattr(mlp, out_name):
                            g = getattr(mlp, gate_name)
                            i = getattr(mlp, in_name)
                            o = getattr(mlp, out_name)
                            if hasattr(g, 'weight'):
                                wd['W_gate'] = g.weight.detach().float().cpu().T
                                wd['W_in'] = i.weight.detach().float().cpu().T
                                wd['W_out'] = o.weight.detach().float().cpu().T
                            else:
                                wd['W_gate'] = g.detach().float().cpu()
                                wd['W_in'] = i.detach().float().cpu()
                                wd['W_out'] = o.detach().float().cpu()
                            found = True
                            break
                    if found: break
                if found: break
            if not found:
                L.log(f"  L{li}: 无法识别MLP格式")
                for name, _ in mlp.named_children():
                    L.log(f"    mlp child: {name}")
                continue
        
        if wd:
            ffn_weights[li] = wd
        L.log(f"  L{li} 权重提取完成, W_gate={wd.get('W_gate',torch.zeros(1)).shape}")
    
    gc.collect()
    L.log(f"提取了 {len(ffn_weights)} 层FFN权重")
    return ffn_weights


def get_act_fn(model_name):
    """获取激活函数 - 不同模型可能使用不同的激活"""
    if model_name in ["deepseek7b", "glm4"]:
        return F.silu  # SiLU/Swish
    return F.gelu  # Qwen3使用GeLU


# ==================== P260: G项的FFN逐神经元分解 ====================

def p260_ffn_neuron_decomposition(ffn_weights, single_hs, combo_hs, n_layers, d_model, d_mlp, act_fn):
    """
    P260: G项的FFN逐神经元分解
    - 对每个FFN神经元i, 计算gate_i, post_i, out_i
    - G = h(combo, l+1) - h(noun, l+1)
    - 分解G到每个FFN神经元的贡献
    """
    L.log("=== P260: G项的FFN逐神经元分解 ===")
    
    results = {"per_triple": [], "per_layer_summary": []}
    key_layers = sorted(set([0,1,2,3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]))
    
    for ti, (noun, attr, combo) in enumerate(TEST_TRIPLES):
        if noun not in single_hs or combo not in combo_hs:
            continue
        
        td = {"noun": noun, "attr": attr, "combo": combo, "per_layer": []}
        
        for li in key_layers:
            if li not in ffn_weights:
                continue
            wd = ffn_weights[li]
            W_gate = wd['W_gate']
            W_in = wd['W_in']
            W_out = wd['W_out']
            
            h_noun = single_hs[noun][li]
            h_combo = combo_hs[combo][li]
            
            # FFN逐神经元计算
            gate_pre_n = torch.matmul(h_noun, W_gate)
            gate_pre_c = torch.matmul(h_combo, W_gate)
            gate_n = act_fn(gate_pre_n)
            gate_c = act_fn(gate_pre_c)
            in_pre_n = torch.matmul(h_noun, W_in)
            in_pre_c = torch.matmul(h_combo, W_in)
            post_n = gate_n * in_pre_n
            post_c = gate_c * in_pre_c
            
            # 每个神经元对delta_ffn的贡献
            delta_post = post_c - post_n  # [d_mlp]
            # ffn_out_delta = sum_j delta_post[j] * W_out[j, :]
            # 所以每个神经元的贡献 = delta_post[j] * W_out[j, :]
            
            # G = h(combo, l+1) - h(noun, l+1)
            G = combo_hs[combo][li+1] - single_hs[noun][li+1]
            G_norm = G.norm().item()
            
            if G_norm < 1e-8:
                continue
            
            # 每个神经元的贡献方向与G的对齐
            # 对top-50贡献最大的神经元分析
            delta_post_abs = delta_post.abs()
            top_k = min(50, d_mlp)
            top_neurons = delta_post_abs.topk(top_k)
            
            # 这些top神经元的贡献之和
            top_contrib_vec = torch.zeros(d_model)
            for idx in top_neurons.indices:
                top_contrib_vec += delta_post[idx] * W_out[idx]
            
            top_contrib_cos = F.cosine_similarity(top_contrib_vec.unsqueeze(0), G.unsqueeze(0)).item()
            
            # 每个top神经元的gate值变化
            gate_diff = (gate_c - gate_n)
            gate_diff_top = gate_diff[top_neurons.indices]
            
            # 哪些神经元从关闭变为打开?
            switch_on = ((gate_n < 0.1) & (gate_c > 0.5)).sum().item()
            switch_off = ((gate_n > 0.5) & (gate_c < 0.1)).sum().item()
            
            # 逐神经元贡献的G方向对齐
            neuron_G_cos = []
            for idx in top_neurons.indices[:20]:
                contrib = delta_post[idx] * W_out[idx]
                cos = F.cosine_similarity(contrib.unsqueeze(0), G.unsqueeze(0)).item()
                neuron_G_cos.append(cos)
            
            lld = {
                "layer": li,
                "G_norm": round(G_norm, 2),
                "top50_contrib_cos_to_G": round(top_contrib_cos, 4),
                "switch_on_neurons": switch_on,
                "switch_off_neurons": switch_off,
                "gate_diff_top_mean": round(gate_diff_top.mean().item(), 4),
                "gate_diff_top_max": round(gate_diff_top.abs().max().item(), 4),
                "top20_neuron_G_cos_mean": round(np.mean(neuron_G_cos), 4),
                "top20_neuron_G_cos_std": round(np.std(neuron_G_cos), 4),
                "top10_neurons_idx": top_neurons.indices[:10].tolist(),
                "top10_neurons_contrib": [round(v, 4) for v in top_neurons.values[:10].tolist()],
                "top20_neuron_G_cos": [round(c, 4) for c in neuron_G_cos],
                "delta_post_sparsity": round((delta_post_abs > delta_post_abs.max()*0.01).float().mean().item(), 4),
            }
            td["per_layer"].append(lld)
        
        results["per_triple"].append(td)
    
    # 逐层汇总
    for li in key_layers:
        top_coses = []
        switch_ons = []
        switch_offs = []
        gate_diffs = []
        sparsities = []
        for td in results["per_triple"]:
            for lld in td["per_layer"]:
                if lld["layer"] == li:
                    top_coses.append(lld["top50_contrib_cos_to_G"])
                    switch_ons.append(lld["switch_on_neurons"])
                    switch_offs.append(lld["switch_off_neurons"])
                    gate_diffs.append(lld["gate_diff_top_mean"])
                    sparsities.append(lld["delta_post_sparsity"])
        if top_coses:
            results["per_layer_summary"].append({
                "layer": li,
                "avg_top50_cos_to_G": round(np.mean(top_coses), 4),
                "avg_switch_on": round(np.mean(switch_ons), 2),
                "avg_switch_off": round(np.mean(switch_offs), 2),
                "avg_gate_diff": round(np.mean(gate_diffs), 4),
                "avg_sparsity": round(np.mean(sparsities), 4),
            })
    
    L.log(f"P260完成: {len(results['per_triple'])}个三元组, {len(results['per_layer_summary'])}层汇总")
    return results


# ==================== P261: 符号回归搜索G项的数学形式 ====================

def p261_symbolic_regression_G(ffn_weights, single_hs, combo_hs, n_layers, d_model, act_fn):
    """
    P261: 符号回归搜索G项的数学形式
    - 输入: 水果骨干方向(B_fruit), 属性方向(A_red, A_sweet等)
    - 输出: G = h(combo) - h(noun) - delta_linear
    - 搜索: 线性、二次、三次、交互项、条件门控、分段线性
    """
    L.log("=== P261: 符号回归搜索G项的数学形式 ===")
    
    results = {"per_model": [], "global_summary": {}}
    
    fruit_words = [w for w in STIMULI["fruit_family"] if w in single_hs]
    animal_words = [w for w in STIMULI["animal_family"] if w in single_hs]
    
    if not fruit_words or not animal_words:
        L.log("缺少必要词汇数据, 跳过P261")
        return results
    
    key_layers = sorted(set([1, 2, 3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]))
    
    for li in key_layers:
        if li not in ffn_weights:
            continue
        wd = ffn_weights[li]
        W_gate = wd['W_gate']
        W_in = wd['W_in']
        W_out = wd['W_out']
        
        ld = {"layer": li}
        
        # 计算语义方向
        fruit_mean = torch.stack([single_hs[w][li+1] for w in fruit_words]).mean(0)
        animal_mean = torch.stack([single_hs[w][li+1] for w in animal_words]).mean(0)
        B_fruit = F.normalize((fruit_mean - animal_mean).unsqueeze(0), dim=1).squeeze(0)
        
        # 属性方向 (用颜色属性)
        color_words = [w for w in STIMULI["color_attrs"] if w in single_hs]
        if color_words:
            color_mean = torch.stack([single_hs[w][li+1] for w in color_words]).mean(0)
            # red方向 = h(red) - color_mean
            A_red = single_hs["red"][li+1] - color_mean if "red" in single_hs else None
            A_green = single_hs["green"][li+1] - color_mean if "green" in single_hs else None
        else:
            A_red = None
            A_green = None
        
        # 对每个测试三元组, 拟合G
        all_r2 = {"linear": [], "quadratic": [], "cubic": [], "interaction": [], 
                   "gated": [], "piecewise": []}
        all_cos = {"linear": [], "quadratic": [], "cubic": [], "interaction": [],
                    "gated": [], "piecewise": []}
        
        for noun, attr, combo in TEST_TRIPLES[:15]:
            if noun not in single_hs or combo not in combo_hs:
                continue
            if attr not in single_hs:
                continue
            
            G = combo_hs[combo][li+1] - single_hs[noun][li+1]
            G_norm = G.norm()
            if G_norm < 1e-8:
                continue
            
            # 输入特征
            h_noun = single_hs[noun][li]
            h_attr = single_hs[attr][li+1] - fruit_mean  # 属性方向
            
            # 线性模型: G ≈ α * h_noun + β * B_fruit + γ * h_attr
            X_linear = torch.stack([h_noun, B_fruit, h_attr], dim=1)  # [d_model, 3]
            # 用最小二乘拟合
            try:
                # 在低维空间做拟合 (用前50个PCA成分)
                G_pca = G[:50] if d_model > 50 else G
                X_pca = X_linear[:50, :] if d_model > 50 else X_linear
                
                # Ridge回归
                from sklearn.linear_model import Ridge
                reg = Ridge(alpha=1.0)
                reg.fit(X_pca.numpy(), G_pca.numpy())
                G_pred = torch.tensor(reg.predict(X_pca.numpy()))
                r2 = max(0, r2_score(G_pca.numpy(), G_pred.numpy()))
                cos_val = F.cosine_similarity(G.unsqueeze(0), torch.tensor(reg.predict(X_linear.numpy())).unsqueeze(0)).item()
                all_r2["linear"].append(r2)
                all_cos["linear"].append(cos_val)
            except:
                pass
            
            # 二次模型: G ≈ α * h_noun + β * B_fruit + γ * h_attr + δ * h_noun^2
            # 用gate_pre的平方近似 (因为FFN的非线性来自gate)
            gate_pre = torch.matmul(h_noun, W_gate)  # [d_mlp]
            gate_val = act_fn(gate_pre)
            
            # 条件门控模型: G ≈ Σ_j gate_j(h_noun) * [h_attr · W_in[:, j]] * W_out[j, :]
            # 这是FFN的条件计算
            try:
                in_pre = torch.matmul(h_noun, W_in)
                post = gate_val * in_pre  # [d_mlp]
                ffn_out = torch.matmul(post, W_out)  # [d_model]
                
                # G与FFN输出的cos
                cos_gated = F.cosine_similarity(G.unsqueeze(0), ffn_out.unsqueeze(0)).item()
                all_cos["gated"].append(cos_gated)
                
                # FFN输出能否解释G?
                proj = (G * ffn_out).sum() / (ffn_out.norm()**2 + 1e-8)
                G_pred_gated = proj * ffn_out
                r2_gated = 1 - (G - G_pred_gated).norm()**2 / (G_norm**2 + 1e-8)
                all_r2["gated"].append(max(0, r2_gated.item()))
            except:
                pass
            
            # 分段线性: 用k-means将gate值分为2段, 每段线性
            try:
                active_mask = gate_val > 0.5
                inactive_mask = ~active_mask
                
                if active_mask.sum() > 10 and inactive_mask.sum() > 10:
                    # active神经元的贡献
                    active_post = post * active_mask.float()
                    active_ffn = torch.matmul(active_post, W_out)
                    
                    # inactive神经元的贡献
                    inactive_post = post * inactive_mask.float()
                    inactive_ffn = torch.matmul(inactive_post, W_out)
                    
                    cos_piecewise = F.cosine_similarity(G.unsqueeze(0), active_ffn.unsqueeze(0)).item()
                    all_cos["piecewise"].append(cos_piecewise)
            except:
                pass
        
        # 汇总该层结果
        for model_name in all_r2:
            if all_r2[model_name]:
                ld[f"r2_{model_name}_mean"] = round(np.mean(all_r2[model_name]), 4)
                ld[f"r2_{model_name}_std"] = round(np.std(all_r2[model_name]), 4)
            if all_cos[model_name]:
                ld[f"cos_{model_name}_mean"] = round(np.mean(all_cos[model_name]), 4)
                ld[f"cos_{model_name}_std"] = round(np.std(all_cos[model_name]), 4)
        
        results["per_model"].append(ld)
    
    # 全局总结
    for model_name in ["linear", "gated", "piecewise"]:
        r2s = [ld.get(f"r2_{model_name}_mean", 0) for ld in results["per_model"]]
        coses = [ld.get(f"cos_{model_name}_mean", 0) for ld in results["per_model"]]
        if r2s:
            results["global_summary"][f"{model_name}_r2_layers"] = round(np.mean(r2s), 4)
            results["global_summary"][f"{model_name}_cos_layers"] = round(np.mean(coses), 4)
    
    L.log(f"P261完成: {len(results['per_model'])}层, 总结: {results['global_summary']}")
    return results


# ==================== P262: 条件计算分析 ====================

def p262_conditional_computation(ffn_weights, single_hs, n_layers, d_model, d_mlp, act_fn):
    """
    P262: 条件计算分析
    - 同一FFN层, 不同输入x的条件响应函数FFN(x)
    - 水果vs动物、同族不同词(苹果vs香蕉)、修饰vs无修饰
    """
    L.log("=== P262: 条件计算分析 ===")
    
    results = {"per_layer": [], "global_summary": {}}
    key_layers = sorted(set([0,1,2,3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]))
    
    fruit_words = [w for w in STIMULI["fruit_family"] if w in single_hs]
    animal_words = [w for w in STIMULI["animal_family"] if w in single_hs]
    vehicle_words = [w for w in STIMULI["vehicle_family"] if w in single_hs]
    color_words = [w for w in STIMULI["color_attrs"] if w in single_hs]
    taste_words = [w for w in STIMULI["taste_attrs"] if w in single_hs]
    
    for li in key_layers:
        if li not in ffn_weights:
            continue
        wd = ffn_weights[li]
        W_gate = wd['W_gate']
        W_in = wd['W_in']
        W_out = wd['W_out']
        
        ld = {"layer": li}
        
        # 1. 同族不同词的FFN响应差异
        # 苹果vs香蕉: gate激活的重叠度
        fruit_gate_patterns = {}
        for word in fruit_words:
            h = single_hs[word][li]
            gate_pre = torch.matmul(h, W_gate)
            gate_val = act_fn(gate_pre)
            fruit_gate_patterns[word] = gate_val
        
        # 2. 异族词的FFN响应差异
        animal_gate_patterns = {}
        for word in animal_words[:8]:
            h = single_hs[word][li]
            gate_pre = torch.matmul(h, W_gate)
            gate_val = act_fn(gate_pre)
            animal_gate_patterns[word] = gate_val
        
        # 3. 属性词的FFN响应
        attr_gate_patterns = {}
        for word in color_words[:8] + taste_words[:8]:
            if word in single_hs:
                h = single_hs[word][li]
                gate_pre = torch.matmul(h, W_gate)
                gate_val = act_fn(gate_pre)
                attr_gate_patterns[word] = gate_val
        
        # 计算各种重叠度
        # 同族内gate重叠
        fruit_cos_list = []
        fruit_words_list = list(fruit_gate_patterns.keys())
        for i in range(min(8, len(fruit_words_list))):
            for j in range(i+1, min(8, len(fruit_words_list))):
                cos = F.cosine_similarity(
                    fruit_gate_patterns[fruit_words_list[i]].unsqueeze(0),
                    fruit_gate_patterns[fruit_words_list[j]].unsqueeze(0)
                ).item()
                fruit_cos_list.append(cos)
        
        # 异族gate重叠 (水果vs动物)
        cross_cos_list = []
        for fw in list(fruit_gate_patterns.keys())[:6]:
            for aw in list(animal_gate_patterns.keys())[:6]:
                cos = F.cosine_similarity(
                    fruit_gate_patterns[fw].unsqueeze(0),
                    animal_gate_patterns[aw].unsqueeze(0)
                ).item()
                cross_cos_list.append(cos)
        
        # 属性vs名词gate重叠
        attr_noun_cos_list = []
        for aw in list(attr_gate_patterns.keys())[:6]:
            for fw in list(fruit_gate_patterns.keys())[:6]:
                cos = F.cosine_similarity(
                    attr_gate_patterns[aw].unsqueeze(0),
                    fruit_gate_patterns[fw].unsqueeze(0)
                ).item()
                attr_noun_cos_list.append(cos)
        
        ld["intra_fruit_gate_cos_mean"] = round(np.mean(fruit_cos_list), 4) if fruit_cos_list else 0
        ld["intra_fruit_gate_cos_std"] = round(np.std(fruit_cos_list), 4) if fruit_cos_list else 0
        ld["cross_family_gate_cos_mean"] = round(np.mean(cross_cos_list), 4) if cross_cos_list else 0
        ld["attr_noun_gate_cos_mean"] = round(np.mean(attr_noun_cos_list), 4) if attr_noun_cos_list else 0
        
        # 4. gate激活的稀疏度分析
        # 对每个词, 多少比例的神经元被激活(gate>0.5)?
        fruit_active_ratios = []
        for word, gate in fruit_gate_patterns.items():
            active = (gate > 0.5).float().mean().item()
            fruit_active_ratios.append(active)
        
        animal_active_ratios = []
        for word, gate in animal_gate_patterns.items():
            active = (gate > 0.5).float().mean().item()
            animal_active_ratios.append(active)
        
        attr_active_ratios = []
        for word, gate in attr_gate_patterns.items():
            active = (gate > 0.5).float().mean().item()
            attr_active_ratios.append(active)
        
        ld["fruit_gate_active_ratio_mean"] = round(np.mean(fruit_active_ratios), 4)
        ld["animal_gate_active_ratio_mean"] = round(np.mean(animal_active_ratios), 4)
        ld["attr_gate_active_ratio_mean"] = round(np.mean(attr_active_ratios), 4)
        
        # 5. 条件FFN输出的方向
        # 计算FFN(h_fruit) - FFN(h_animal)的方向
        if fruit_words and animal_words:
            fruit_h = single_hs[fruit_words[0]][li]
            animal_h = single_hs[animal_words[0]][li]
            
            ffn_fruit = torch.matmul(act_fn(torch.matmul(fruit_h, W_gate)) * torch.matmul(fruit_h, W_in), W_out)
            ffn_animal = torch.matmul(act_fn(torch.matmul(animal_h, W_gate)) * torch.matmul(animal_h, W_in), W_out)
            
            delta_ffn = ffn_fruit - ffn_animal
            ld["ffn_direction_fruit_animal_norm"] = round(delta_ffn.norm().item(), 2)
        
        # 6. gate的输入依赖性 — gate_i对h的哪个维度最敏感?
        # 用gate权重的范数分析
        gate_col_norms = W_gate.norm(dim=0)  # [d_mlp] 每个神经元的key方向范数
        gate_row_norms = W_gate.norm(dim=1)  # [d_model] 每个输入维度对所有神经元的总影响
        
        ld["gate_key_norm_cv"] = round((gate_col_norms.std() / gate_col_norms.mean()).item(), 4)
        ld["gate_input_dim_cv"] = round((gate_row_norms.std() / gate_row_norms.mean()).item(), 4)
        
        # 7. 同族不同词的FFN输出差异
        if len(fruit_words) >= 3:
            ffn_outputs = []
            for word in fruit_words[:6]:
                h = single_hs[word][li]
                gate = act_fn(torch.matmul(h, W_gate))
                post = gate * torch.matmul(h, W_in)
                ffn_out = torch.matmul(post, W_out)
                ffn_outputs.append(ffn_out)
            
            # 水果内FFN输出的cos
            intra_ffn_cos = []
            for i in range(len(ffn_outputs)):
                for j in range(i+1, len(ffn_outputs)):
                    cos = F.cosine_similarity(ffn_outputs[i].unsqueeze(0), ffn_outputs[j].unsqueeze(0)).item()
                    intra_ffn_cos.append(cos)
            
            ld["intra_fruit_ffn_out_cos_mean"] = round(np.mean(intra_ffn_cos), 4) if intra_ffn_cos else 0
            ld["intra_fruit_ffn_out_cos_std"] = round(np.std(intra_ffn_cos), 4) if intra_ffn_cos else 0
        
        results["per_layer"].append(ld)
    
    # 全局总结
    if results["per_layer"]:
        intra_gates = [ld["intra_fruit_gate_cos_mean"] for ld in results["per_layer"]]
        cross_gates = [ld["cross_family_gate_cos_mean"] for ld in results["per_layer"]]
        attr_gates = [ld["attr_noun_gate_cos_mean"] for ld in results["per_layer"]]
        results["global_summary"] = {
            "intra_family_gate_cos_avg": round(np.mean(intra_gates), 4),
            "cross_family_gate_cos_avg": round(np.mean(cross_gates), 4),
            "attr_noun_gate_cos_avg": round(np.mean(attr_gates), 4),
            "gate_discrimination_ratio": round(np.mean(intra_gates) / (np.mean(cross_gates) + 1e-8), 4),
        }
    
    L.log(f"P262完成: {len(results['per_layer'])}层, 总结: {results['global_summary']}")
    return results


# ==================== P263: 注意力-FFN精确交互 ====================

def p263_attn_ffn_interaction(mdl, tok, device, single_hs, ffn_weights, n_layers, d_model, d_mlp, act_fn):
    """
    P263: 注意力-FFN精确交互
    - 收集attn_output + ffn_output
    - 分析Attn输出如何条件化FFN的gate
    - 关键: Attn选择了哪些上下文信息, FFN如何处理
    """
    L.log("=== P263: 注意力-FFN精确交互 ===")
    
    results = {"per_sentence": [], "per_layer_summary": []}
    key_layers = sorted(set([0,1,2,3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]))
    
    # 用"red apple"这样的句子, 分析Attn和FFN的交互
    test_sentences = [
        "The red apple is",
        "The green apple is",
        "The sweet apple is",
        "The big elephant is",
        "The brown cat is",
        "The apple is",
        "The banana is",
        "The cat is",
    ]
    
    for si, sentence in enumerate(test_sentences):
        sd = {"sentence": sentence, "per_layer": []}
        
        inputs = tok(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**inputs, output_hidden_states=True, output_attentions=True)
        
        hs = out.hidden_states
        attns = out.attentions
        
        # 获取最后一个token的hidden state
        last_token_hs = [h[0, -1].float().cpu() for h in hs]
        
        for li in key_layers:
            if li not in ffn_weights:
                continue
            wd = ffn_weights[li]
            W_gate = wd['W_gate']
            W_in = wd['W_in']
            W_out = wd['W_out']
            
            # Attn输出 = hs[li+1] - hs[li] - ffn_out
            # 实际: h_{l+1} = h_l + attn_out + ffn_out (+ RMSNorm)
            # 近似: attn_out ≈ hs[li+1] - hs[li] - ffn_out
            
            h_in = last_token_hs[li]  # FFN输入
            gate = act_fn(torch.matmul(h_in, W_gate))
            post = gate * torch.matmul(h_in, W_in)
            ffn_out = torch.matmul(post, W_out)
            
            h_out = last_token_hs[li+1]
            delta_h = h_out - h_in
            attn_out_approx = delta_h - ffn_out  # 近似的attn输出
            
            # Attn对FFN gate的影响
            # 如果没有Attn, gate会是什么? (用h_in直接计算)
            # 如果有Attn, gate是什么? (用h_in + attn_out计算)
            h_with_attn = h_in + attn_out_approx
            gate_with_attn = act_fn(torch.matmul(h_with_attn, W_gate))
            
            # gate变化
            gate_diff = gate_with_attn - gate
            gate_diff_norm = gate_diff.norm().item()
            gate_diff_cos = F.cosine_similarity(gate.unsqueeze(0), gate_with_attn.unsqueeze(0)).item()
            
            # Attn改变了哪些神经元的gate?
            switch_on = ((gate < 0.1) & (gate_with_attn > 0.5)).sum().item()
            switch_off = ((gate > 0.5) & (gate_with_attn < 0.1)).sum().item()
            
            # Attn输出的方向
            attn_out_norm = attn_out_approx.norm().item()
            ffn_out_norm = ffn_out.norm().item()
            
            # Attn vs FFN的方向对齐
            attn_ffn_cos = F.cosine_similarity(attn_out_approx.unsqueeze(0), ffn_out.unsqueeze(0)).item() if attn_out_norm > 1e-8 and ffn_out_norm > 1e-8 else 0
            
            # 注意力权重 (如果有)
            attn_weights = None
            if attns and li < len(attns):
                attn_weights = attns[li][0, -1].float().cpu()  # 最后一头最后一token
            
            lld = {
                "layer": li,
                "attn_out_norm": round(attn_out_norm, 2),
                "ffn_out_norm": round(ffn_out_norm, 2),
                "attn_ffn_cos": round(attn_ffn_cos, 4),
                "gate_change_norm": round(gate_diff_norm, 4),
                "gate_change_cos": round(gate_diff_cos, 4),
                "attn_switch_on": switch_on,
                "attn_switch_off": switch_off,
            }
            sd["per_layer"].append(lld)
        
        results["per_sentence"].append(sd)
        del out, hs, attns
    
    # 逐层汇总
    for li in key_layers:
        attn_norms = []
        ffn_norms = []
        attn_ffn_coses = []
        gate_changes = []
        for sd in results["per_sentence"]:
            for lld in sd["per_layer"]:
                if lld["layer"] == li:
                    attn_norms.append(lld["attn_out_norm"])
                    ffn_norms.append(lld["ffn_out_norm"])
                    attn_ffn_coses.append(lld["attn_ffn_cos"])
                    gate_changes.append(lld["gate_change_cos"])
        if attn_norms:
            results["per_layer_summary"].append({
                "layer": li,
                "avg_attn_norm": round(np.mean(attn_norms), 2),
                "avg_ffn_norm": round(np.mean(ffn_norms), 2),
                "avg_attn_ffn_cos": round(np.mean(attn_ffn_coses), 4),
                "avg_gate_change_cos": round(np.mean(gate_changes), 4),
            })
    
    L.log(f"P263完成: {len(results['per_sentence'])}个句子, {len(results['per_layer_summary'])}层汇总")
    return results


# ==================== P264: FFN策略等价性分析 ====================

def p264_ffn_strategy_equivalence(all_model_results, ffn_weights, single_hs, n_layers, d_model, d_mlp, act_fn, model_name):
    """
    P264: FFN策略等价性分析
    - 对当前模型, 分析其FFN策略的特征
    - 与其他模型的结果对比
    """
    L.log("=== P264: FFN策略等价性分析 ===")
    
    results = {"per_layer": [], "strategy_profile": {}}
    key_layers = sorted(set([0,1,2,3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]))
    
    fruit_words = [w for w in STIMULI["fruit_family"] if w in single_hs]
    animal_words = [w for w in STIMULI["animal_family"] if w in single_hs]
    
    if not fruit_words or not animal_words:
        L.log("缺少必要词汇数据, 跳过P264")
        return results
    
    for li in key_layers:
        if li not in ffn_weights:
            continue
        wd = ffn_weights[li]
        W_gate = wd['W_gate']
        W_in = wd['W_in']
        W_out = wd['W_out']
        
        ld = {"layer": li}
        
        # 1. Gate-In重叠度 (Qwen3高, DS7B低)
        n_sample = min(500, W_gate.shape[1])
        gate_cols = F.normalize(W_gate[:, :n_sample], dim=0)
        in_cols = F.normalize(W_in[:, :n_sample], dim=0)
        gate_in_cos = (gate_cols * in_cols).sum(0)
        ld["gate_in_cos_mean"] = round(gate_in_cos.mean().item(), 4)
        ld["gate_in_cos_std"] = round(gate_in_cos.std().item(), 4)
        
        # 2. W_out方差集中度 (Qwen3高, DS7B低)
        row_norms = W_out.norm(dim=1)
        total_norm_sq = (row_norms**2).sum()
        top50_sq = torch.topk(row_norms**2, min(50, len(row_norms))).values.sum()
        top500_sq = torch.topk(row_norms**2, min(500, len(row_norms))).values.sum()
        ld["W_out_cumvar_50"] = round((top50_sq / total_norm_sq).item(), 4)
        ld["W_out_cumvar_500"] = round((top500_sq / total_norm_sq).item(), 4)
        
        # 3. FFN输出范数随层的变化
        fruit_mean = torch.stack([single_hs[w][li] for w in fruit_words]).mean(0)
        gate_f = act_fn(torch.matmul(fruit_mean, W_gate))
        post_f = gate_f * torch.matmul(fruit_mean, W_in)
        ffn_out_f = torch.matmul(post_f, W_out)
        ld["ffn_out_norm_fruit"] = round(ffn_out_f.norm().item(), 2)
        
        # 4. G项与FFN输出的对齐
        apple_h = single_hs.get("apple")
        if apple_h is not None:
            G = apple_h[li+1] - fruit_mean  # 这里用li+1的hidden state
            G_norm = G.norm()
            if G_norm > 1e-8:
                cos_G_ffn = F.cosine_similarity(G.unsqueeze(0), ffn_out_f.unsqueeze(0)).item()
                ld["cos_G_ffn"] = round(cos_G_ffn, 4)
            else:
                ld["cos_G_ffn"] = 0
        
        # 5. Gate稀疏度
        gate_sparsity = (gate_f > 0.5).float().mean().item()
        ld["gate_sparsity"] = round(gate_sparsity, 4)
        
        # 6. 策略分类指标
        # 冗余编码策略: gate_in_cos高 + W_out集中度高
        # 正交分解策略: gate_in_cos低 + W_out集中度低 + 最后层路由放大
        ld["strategy_redundancy_score"] = round(ld["gate_in_cos_mean"] * ld["W_out_cumvar_500"], 4)
        ld["strategy_orthogonality_score"] = round((1 - abs(ld["gate_in_cos_mean"])) * (1 - ld["W_out_cumvar_500"]), 4)
        
        results["per_layer"].append(ld)
    
    # 策略画像
    if results["per_layer"]:
        avg_gate_in = np.mean([ld["gate_in_cos_mean"] for ld in results["per_layer"]])
        avg_cumvar500 = np.mean([ld["W_out_cumvar_500"] for ld in results["per_layer"]])
        last_ffn_norms = [ld["ffn_out_norm_fruit"] for ld in results["per_layer"] if ld["layer"] >= n_layers-3]
        avg_last_ffn = np.mean(last_ffn_norms) if last_ffn_norms else 0
        
        # 早期层的ffn范数
        early_ffn_norms = [ld["ffn_out_norm_fruit"] for ld in results["per_layer"] if ld["layer"] <= 3]
        avg_early_ffn = np.mean(early_ffn_norms) if early_ffn_norms else 0
        
        amplification = avg_last_ffn / (avg_early_ffn + 1e-8)
        
        results["strategy_profile"] = {
            "model": model_name,
            "avg_gate_in_cos": round(avg_gate_in, 4),
            "avg_W_out_cumvar500": round(avg_cumvar500, 4),
            "avg_last_ffn_norm": round(avg_last_ffn, 2),
            "avg_early_ffn_norm": round(avg_early_ffn, 2),
            "ffn_amplification_ratio": round(amplification, 2),
            "strategy_type": "redundant" if avg_gate_in > 0.5 else "orthogonal",
        }
    
    L.log(f"P264完成: {len(results['per_layer'])}层, 策略={results['strategy_profile'].get('strategy_type','?')}")
    return results


# ==================== 主函数 ====================

def run_model(model_name):
    """运行单个模型的所有实验"""
    L.log(f"\n{'='*60}")
    L.log(f"模型: {model_name}")
    L.log(f"{'='*60}")
    
    t0 = time.time()
    
    # 加载模型
    L.log("加载模型...")
    mdl, tok, device = load_model(model_name)
    n_layers = mdl.config.num_hidden_layers
    d_model = mdl.config.hidden_size
    d_mlp = None
    layers = None
    if hasattr(mdl, 'model') and hasattr(mdl.model, 'layers'):
        layers = mdl.model.layers
    elif hasattr(mdl, 'transformer') and hasattr(mdl.transformer, 'h'):
        layers = mdl.transformer.h
    elif hasattr(mdl, 'transformer') and hasattr(mdl.transformer, 'encoder') and hasattr(mdl.transformer.encoder, 'layers'):
        layers = mdl.transformer.encoder.layers
    
    if layers is not None:
        mlp = layers[0].mlp if hasattr(layers[0], 'mlp') else (layers[0].ffn if hasattr(layers[0], 'ffn') else None)
        if mlp is not None:
            # 检测融合格式
            if hasattr(mlp, 'gate_up_proj'):
                d_mlp = mlp.gate_up_proj.weight.shape[0] // 2
            elif hasattr(mlp, 'gate_proj'):
                d_mlp = mlp.gate_proj.weight.shape[0]
            elif hasattr(mlp, 'W_gate'):
                d_mlp = mlp.W_gate.shape[1]
            else:
                for name, param in mlp.named_parameters():
                    if 'weight' in name and param.shape[0] > d_model:
                        d_mlp = param.shape[0]
                        break
    
    act_fn = get_act_fn(model_name)
    
    L.log(f"结构: {n_layers}L, d={d_model}, d_mlp={d_mlp}, act={act_fn.__name__}")
    
    # 收集数据
    L.log("数据收集...")
    single_hs, combo_hs = collect_all_data(mdl, tok, device)
    
    # 提取FFN权重 (只提取关键层以节省内存)
    key_layers = sorted(set([0,1,2,3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]))
    L.log(f"FFN权重提取 (关键层: {key_layers})...")
    ffn_weights = extract_ffn_weights(mdl, key_layers=key_layers)
    
    # P260: G项的FFN逐神经元分解
    L.log("P260: G项的FFN逐神经元分解...")
    p260_results = p260_ffn_neuron_decomposition(ffn_weights, single_hs, combo_hs, n_layers, d_model, d_mlp, act_fn)
    
    # P261: 符号回归搜索G项的数学形式
    L.log("P261: 符号回归搜索G项...")
    p261_results = p261_symbolic_regression_G(ffn_weights, single_hs, combo_hs, n_layers, d_model, act_fn)
    
    # P262: 条件计算分析
    L.log("P262: 条件计算分析...")
    p262_results = p262_conditional_computation(ffn_weights, single_hs, n_layers, d_model, d_mlp, act_fn)
    
    # P263: 注意力-FFN精确交互 (需要模型)
    L.log("P263: 注意力-FFN精确交互...")
    p263_results = p263_attn_ffn_interaction(mdl, tok, device, single_hs, ffn_weights, n_layers, d_model, d_mlp, act_fn)
    
    # P264: FFN策略等价性分析
    L.log("P264: FFN策略等价性分析...")
    p264_results = p264_ffn_strategy_equivalence({}, ffn_weights, single_hs, n_layers, d_model, d_mlp, act_fn, model_name)
    
    # 释放模型
    L.log("释放模型...")
    del mdl, tok, layers
    gc.collect()
    torch.cuda.empty_cache()
    
    # 汇总结果
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "d_mlp": d_mlp,
        "act_fn": act_fn.__name__,
        "p260_ffn_neuron_decomposition": p260_results,
        "p261_symbolic_regression_G": p261_results,
        "p262_conditional_computation": p262_results,
        "p263_attn_ffn_interaction": p263_results,
        "p264_ffn_strategy_equivalence": p264_results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    
    # 保存
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        return obj
    
    all_results = sanitize(all_results)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"phase_xliv_p260_264_{model_name}_{ts}.json"
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    L.log(f"结果保存: {out_path}")
    L.log(f"总用时: {round(time.time() - t0, 1)}秒")
    
    # 打印关键摘要
    print_summary(all_results)
    
    return all_results


def print_summary(results):
    """打印关键发现摘要"""
    model = results["model"]
    L.log(f"\n{'='*60}")
    L.log(f"  {model} 关键发现摘要")
    L.log(f"  结构: {results['n_layers']}L, d={results['d_model']}, d_mlp={results['d_mlp']}, act={results['act_fn']}")
    L.log(f"{'='*60}")
    
    # P260 摘要
    p260 = results["p260_ffn_neuron_decomposition"]
    L.log("\nP260: G项的FFN逐神经元分解")
    for ls in p260.get("per_layer_summary", []):
        L.log(f"  L{ls['layer']}: top50_cos_to_G={ls['avg_top50_cos_to_G']}, "
              f"switch_on={ls['avg_switch_on']}, gate_diff={ls['avg_gate_diff']}")
    
    # P261 摘要
    p261 = results["p261_symbolic_regression_G"]
    L.log("\nP261: 符号回归搜索G项")
    gs = p261.get("global_summary", {})
    for k, v in gs.items():
        L.log(f"  {k}: {v}")
    
    # P262 摘要
    p262 = results["p262_conditional_computation"]
    L.log("\nP262: 条件计算分析")
    gs262 = p262.get("global_summary", {})
    for k, v in gs262.items():
        L.log(f"  {k}: {v}")
    
    # P263 摘要
    p263 = results["p263_attn_ffn_interaction"]
    L.log("\nP263: 注意力-FFN交互")
    for ls in p263.get("per_layer_summary", []):
        L.log(f"  L{ls['layer']}: attn_norm={ls['avg_attn_norm']}, ffn_norm={ls['avg_ffn_norm']}, "
              f"attn_ffn_cos={ls['avg_attn_ffn_cos']}, gate_change={ls['avg_gate_change_cos']}")
    
    # P264 摘要
    p264 = results["p264_ffn_strategy_equivalence"]
    L.log("\nP264: FFN策略等价性")
    sp = p264.get("strategy_profile", {})
    for k, v in sp.items():
        L.log(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", 
                       choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    results = run_model(args.model)
    L.log("ALL DONE!")
