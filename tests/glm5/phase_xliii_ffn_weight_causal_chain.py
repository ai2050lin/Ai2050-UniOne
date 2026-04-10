"""
Phase XLIII-P257/258/259: FFN权重级因果链追踪
================================================

P253-256核心结论: 苹果编码=水果骨干方向+微调偏置(极小)+苹果特异神经元集合(少量)
                  G项=FFN非线性输出, G残差比88-100%, G增长在最后2-3层
关键缺口: 权重空间仅5%覆盖, 不知道FFN权重如何产生G项

三大实验:
  P257: FFN权重矩阵全景分析 — W_gate/W_in/W_out SVD + key方向 + 与h的对齐
        核心问题: FFN权重的几何结构是什么? key方向与语义方向的关系?
  
  P258: FFN因果链精确分解 — 逐层分解 h→W_gate→act→W_in→post→W_out→ffn_out
        核心问题: G项 = ffn_out(combo) - ffn_out(noun), 这个差值由哪些权重成分贡献?
  
  P259: 苹果-水果编码的权重级来源 — 哪些FFN神经元负责水果/苹果的区分?
        核心问题: 苹果特异神经元(PhaseXLII发现的16-40个)对应的W_gate/W_in/W_out行向量有什么特征?

实验模型: qwen3 -> deepseek7b -> glm4 (串行, 避免OOM)
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
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_model_path(model_name):
    paths = {
        "qwen3": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "deepseek7b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "glm4": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
    }
    return paths.get(model_name)

def load_model(model_name):
    p = get_model_path(model_name)
    p_abs = os.path.abspath(p)
    tok = AutoTokenizer.from_pretrained(
        p_abs, trust_remote_code=True,
        local_files_only=True,       # 纯本地加载, 避免网络超时
        use_fast=False               # slow tokenizer更兼容
    )
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        p_abs, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True,       # 纯本地加载
        low_cpu_mem_usage=True,      # 分片加载, 减少CPU RAM峰值(os error 1455修复)
        attn_implementation="eager", device_map="cpu"
    )
    if torch.cuda.is_available():    # GPU可用性检测
        mdl = mdl.to("cuda")
    mdl.eval()
    device = next(mdl.parameters()).device
    return mdl, tok, device

STIMULI = {
    "fruit_family": ["apple","banana","pear","orange","grape","mango","strawberry","watermelon","cherry","peach","lemon","lime"],
    "animal_family": ["cat","dog","rabbit","horse","lion","eagle","elephant","dolphin","tiger","bear","fox","wolf"],
    "vehicle_family": ["car","bus","train","plane","boat","bicycle","truck","helicopter","motorcycle","ship","subway","taxi"],
    "furniture_family": ["chair","table","desk","sofa","bed","cabinet","shelf","bench","stool","dresser","couch","armchair"],
    "tool_family": ["hammer","wrench","screwdriver","saw","drill","pliers","knife","scissors","shovel","rake","axe","chisel"],
    "color_attrs": ["red","green","yellow","orange","brown","white","blue","black","pink","purple","gray","gold"],
    "taste_attrs": ["sweet","sour","bitter","salty","crisp","soft","spicy","fresh","tart","savory","rich","mild"],
    "size_attrs": ["big","small","tall","short","long","wide","thin","thick","heavy","light","huge","tiny"],
    "fruit_color_combos": ["red apple","green apple","yellow banana","orange orange","green pear","red grape","yellow mango","red cherry","pink peach","yellow lemon"],
    "fruit_taste_combos": ["sweet apple","sour apple","sweet banana","sour orange","sweet pear","bitter grape","sweet mango","sweet cherry","tart lemon","sweet peach"],
    "animal_color_combos": ["brown cat","white dog","brown rabbit","black horse","golden eagle","white cat","black dog","orange tiger","brown bear","red fox"],
}

PROMPT_TEMPLATES = ["The {word} is", "A {word} can be", "This {word} has"]

TEST_TRIPLES = [
    ("apple","red","red apple"), ("apple","sweet","sweet apple"),
    ("apple","green","green apple"), ("banana","yellow","yellow banana"),
    ("banana","sweet","sweet banana"), ("pear","green","green pear"),
    ("cat","brown","brown cat"), ("dog","white","white dog"),
    ("car","red","red car"), ("car","fast","fast car"),
    ("apple","fresh","fresh apple"), ("horse","black","black horse"),
    ("apple","sour","sour apple"), ("cherry","red","red cherry"),
    ("grape","purple","purple grape"), ("mango","yellow","yellow mango"),
    ("lemon","tart","tart lemon"), ("peach","sweet","sweet peach"),
    ("orange","sour","sour orange"), ("strawberry","sweet","sweet strawberry"),
]


# ==================== 数据收集 ====================

def collect_all_data(mdl, tok, device):
    """一次性收集所有hidden states + FFN中间激活"""
    n_layers = mdl.config.num_hidden_layers
    
    all_single = sorted(set(
        STIMULI["fruit_family"] + STIMULI["animal_family"] + 
        STIMULI["vehicle_family"] + STIMULI["furniture_family"] + STIMULI["tool_family"] +
        STIMULI["color_attrs"] + STIMULI["taste_attrs"] + STIMULI["size_attrs"]
    ))
    all_combos = sorted(set(
        STIMULI["fruit_color_combos"] + STIMULI["fruit_taste_combos"] + STIMULI["animal_color_combos"]
    ))
    
    print(f"  收集: {len(all_single)}单字 + {len(all_combos)}组合词")
    
    single_hs = {}
    for i, word in enumerate(all_single):
        if i % 20 == 0:
            print(f"    单字 [{i+1}/{len(all_single)}]...")
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
    
    combo_hs = {}
    for i, combo in enumerate(all_combos):
        if i % 10 == 0:
            print(f"    组合 [{i+1}/{len(all_combos)}]...")
        inputs = tok(f"The {combo}", return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**inputs, output_hidden_states=True)
        combo_hs[combo] = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
        del out
    
    gc.collect()
    return single_hs, combo_hs


def extract_ffn_weights(mdl, device):
    """提取所有层的FFN权重矩阵, 统一转换为标准格式:
    W_gate: [d_model, d_mlp]  (gate投影矩阵)
    W_in:   [d_model, d_mlp]  (value/up投影矩阵)
    W_out:  [d_mlp, d_model]  (output/down投影矩阵)
    """
    ffn_weights = {}
    
    layers = None
    if hasattr(mdl, 'model') and hasattr(mdl.model, 'layers'):
        layers = mdl.model.layers
    elif hasattr(mdl, 'transformer') and hasattr(mdl.transformer, 'h'):
        layers = mdl.transformer.h
    
    if layers is None:
        print("  警告: 无法找到模型层!")
        return ffn_weights
    
    for li, layer in enumerate(layers):
        mlp = layer.mlp if hasattr(layer, 'mlp') else layer.ffn
        
        wd = {'type': 'gated'}
        
        # TransformerLens格式: W_gate [d_model, d_mlp]
        if hasattr(mlp, 'W_gate'):
            wd['W_gate'] = mlp.W_gate.detach().float().cpu()
            wd['W_in'] = mlp.W_in.detach().float().cpu()
            wd['W_out'] = mlp.W_out.detach().float().cpu()
            wd['b_in'] = mlp.b_in.detach().float().cpu()
            wd['b_out'] = mlp.b_out.detach().float().cpu()
        # HuggingFace格式: gate_proj.weight [d_mlp, d_model] 需要转置
        elif hasattr(mlp, 'gate_proj'):
            wd['W_gate'] = mlp.gate_proj.weight.detach().float().cpu().T  # [d_model, d_mlp]
            wd['W_in'] = mlp.up_proj.weight.detach().float().cpu().T      # [d_model, d_mlp]
            wd['W_out'] = mlp.down_proj.weight.detach().float().cpu().T   # [d_mlp, d_model]
            wd['b_in'] = torch.zeros(wd['W_gate'].shape[1])               # [d_mlp]
            wd['b_out'] = torch.zeros(wd['W_out'].shape[1])               # [d_model]
        # 其他HF格式
        else:
            print(f"    L{li}: 无法识别MLP格式, 尝试自动检测...")
            for name, param in mlp.named_parameters():
                if 'weight' in name:
                    print(f"      {name}: shape={param.shape}")
            # 尝试常见名称
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
                print(f"    L{li}: 跳过, 无法提取权重")
                continue
        
        ffn_weights[li] = wd
    
    print(f"  提取了 {len(ffn_weights)} 层FFN权重")
    for li in [0, len(ffn_weights)//2, len(ffn_weights)-1]:
        if li in ffn_weights:
            wd = ffn_weights[li]
            print(f"    L{li} W_gate={wd['W_gate'].shape}, W_in={wd['W_in'].shape}, W_out={wd['W_out'].shape}")
    
    return ffn_weights


# ==================== P257: FFN权重矩阵全景分析 ====================

def p257_ffn_weight_analysis(ffn_weights, single_hs, n_layers, d_model, d_mlp):
    """
    P257: FFN权重矩阵全景分析
    - W_gate/W_in/W_out SVD分解
    - Key方向(=FFN神经元的输入方向)与h的对齐
    - 权重矩阵的几何结构
    """
    print(f"\n  P257: FFN权重矩阵全景分析")
    
    results = {"per_layer": [], "global_summary": {}}
    
    # 选取关键层: L0, L1, 中间层, 最后2层
    key_layers = sorted(set([0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]))
    
    for li in key_layers:
        if li not in ffn_weights:
            continue
        wd = ffn_weights[li]
        if 'W_gate' not in wd:
            continue
        
        W_gate = wd['W_gate']  # [d_model, d_mlp]
        W_in = wd['W_in']      # [d_model, d_mlp]
        W_out = wd['W_out']    # [d_mlp, d_model]
        
        print(f"    分析L{li}...")
        ld = {"layer": li}
        
        # 1. SVD分解
        # W_gate的列向量(=FFN神经元key方向)的SVD
        Ug, Sg, Vg = torch.linalg.svd(W_gate, full_matrices=False)
        Ui, Si, Vi = torch.linalg.svd(W_in, full_matrices=False)
        Uo, So, Vo = torch.linalg.svd(W_out, full_matrices=False)
        
        # 奇异值分布
        ld['W_gate_svd_top10'] = Sg[:10].tolist()
        ld['W_gate_svd_tail10'] = Sg[-10:].tolist()
        ld['W_in_svd_top10'] = Si[:10].tolist()
        ld['W_out_svd_top10'] = So[:10].tolist()
        
        # 累积方差比
        total_gate = Sg.sum().item()
        total_in = Si.sum().item()
        total_out = So.sum().item()
        for k in [10, 50, 100, 500]:
            ld[f'W_gate_cumvar_{k}'] = round(Sg[:k].sum().item() / total_gate, 4) if k <= len(Sg) else None
            ld[f'W_in_cumvar_{k}'] = round(Si[:k].sum().item() / total_in, 4) if k <= len(Si) else None
            ld[f'W_out_cumvar_{k}'] = round(So[:k].sum().item() / total_out, 4) if k <= len(So) else None
        
        # 2. W_gate列向量(=FFN key方向)之间的正交性
        # 采样100个key方向计算pairwise cos
        n_sample = min(200, W_gate.shape[1])
        keys = W_gate[:, :n_sample].T  # [n_sample, d_model]
        keys_norm = F.normalize(keys, dim=1)
        cos_matrix = torch.mm(keys_norm, keys_norm.T)
        # 去掉对角线
        mask = ~torch.eye(n_sample, dtype=torch.bool)
        off_diag = cos_matrix[mask]
        ld['key_pairwise_cos_mean'] = round(off_diag.mean().item(), 4)
        ld['key_pairwise_cos_std'] = round(off_diag.std().item(), 4)
        ld['key_pairwise_cos_max'] = round(off_diag.abs().max().item(), 4)
        
        # 3. W_gate vs W_in 对齐 — gate和value的关系
        # 对每个FFN神经元, gate方向和value方向的cos
        gate_cols = F.normalize(W_gate, dim=0)  # [d_model, d_mlp]
        in_cols = F.normalize(W_in, dim=0)       # [d_model, d_mlp]
        gate_in_cos = (gate_cols * in_cols).sum(0)  # [d_mlp]
        ld['gate_in_cos_mean'] = round(gate_in_cos.mean().item(), 4)
        ld['gate_in_cos_std'] = round(gate_in_cos.std().item(), 4)
        
        # 4. W_out行向量与h方向的对齐
        # 取h的方向(苹果, 水果骨干, 动物骨干)
        if "apple" in single_hs:
            apple_dir = F.normalize(single_hs["apple"][li+1].unsqueeze(0), dim=1)  # [1, d_model]
        else:
            apple_dir = None
        
        fruit_words = [w for w in STIMULI["fruit_family"] if w in single_hs]
        animal_words = [w for w in STIMULI["animal_family"] if w in single_hs]
        
        if fruit_words and animal_words:
            # 用li+1的hidden state (FFN输出后的状态) 计算语义方向
            fruit_mean = torch.stack([single_hs[w][li+1] for w in fruit_words]).mean(0)
            animal_mean = torch.stack([single_hs[w][li+1] for w in animal_words]).mean(0)
            fruit_dir = F.normalize((fruit_mean - animal_mean).unsqueeze(0), dim=1)
            
            # W_out行向量与水果方向的cos
            out_rows = F.normalize(W_out, dim=1)  # [d_mlp, d_model]
            fruit_out_cos = torch.mm(out_rows, fruit_dir.T).squeeze()  # [d_mlp]
            ld['fruit_dir_Wout_cos_top10'] = fruit_out_cos.abs().topk(10).values.tolist()
            ld['fruit_dir_Wout_cos_top10_idx'] = fruit_out_cos.abs().topk(10).indices.tolist()
            ld['fruit_dir_Wout_cos_mean'] = round(fruit_out_cos.mean().item(), 4)
            ld['fruit_dir_Wout_cos_max'] = round(fruit_out_cos.abs().max().item(), 4)
            
            if apple_dir is not None:
                apple_out_cos = torch.mm(out_rows, apple_dir.T).squeeze()
                ld['apple_dir_Wout_cos_max'] = round(apple_out_cos.abs().max().item(), 4)
        
        # 5. W_gate行向量与语义方向的对齐
        # W_gate的每一行 = d_model的一个维度对所有FFN神经元的贡献
        # 如果某个维度的gate权重特别大, 说明该维度对FFN激活很重要
        gate_row_norms = W_gate.norm(dim=1)  # [d_model]
        ld['gate_row_norm_top10_idx'] = gate_row_norms.topk(10).indices.tolist()
        ld['gate_row_norm_cv'] = round((gate_row_norms.std() / gate_row_norms.mean()).item(), 4)
        
        # 6. FFN key方向与LM Head行方向的关系
        # 检查W_gate的列向量是否与W_lm的行向量对齐
        # (已在PhaseXXXIV发现W_lm行间cos=0.087, 这里检查FFN key是否有类似结构)
        ld['d_mlp'] = W_gate.shape[1]
        
        results["per_layer"].append(ld)
    
    # 全局总结
    if len(results["per_layer"]) >= 2:
        first = results["per_layer"][0]
        last = results["per_layer"][-1]
        results["global_summary"] = {
            "key_cos_early_vs_late": [
                first.get('key_pairwise_cos_mean', -1),
                last.get('key_pairwise_cos_mean', -1)
            ],
            "gate_in_cos_early_vs_late": [
                first.get('gate_in_cos_mean', -1),
                last.get('gate_in_cos_mean', -1)
            ],
            "gate_row_norm_cv_early_vs_late": [
                first.get('gate_row_norm_cv', -1),
                last.get('gate_row_norm_cv', -1)
            ]
        }
    
    print(f"    P257完成: 分析了{len(results['per_layer'])}层")
    return results


# ==================== P258: FFN因果链精确分解 ====================

def p258_ffn_causal_chain(ffn_weights, single_hs, combo_hs, n_layers, d_model, d_mlp):
    """
    P258: FFN因果链精确分解 (纯权重计算, 不需要模型前向传播)
    
    核心思路:
      h_{l+1} = h_l + attn_out_l + ffn_out_l  (残差连接)
      delta_h = h(combo, l+1) - h(noun, l+1)
      
      FFN因果链:
        gate_pre = h_l @ W_gate     [d_mlp]
        in_pre = h_l @ W_in         [d_mlp]  
        post = gelu(gate_pre) * in_pre + b_in  [d_mlp]
        ffn_out = post @ W_out + b_out  [d_model]
      
      G = h(combo) - h(noun)
      delta_ffn = ffn_out(combo) - ffn_out(noun)
      ffn对G的贡献比 = delta_ffn / G
      
    注意: hidden_states[li] = embedding层+各层输出
          ffn_weights[li] 对应的是第li层的FFN
          第li层的FFN输入是 hidden_states[li] (上一层的输出)
    """
    print(f"\n  P258: FFN因果链精确分解")
    
    results = {"per_triple": [], "per_layer_summary": []}
    
    key_layers = sorted(set([0, 1, 2, 3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]))
    act_fn = torch.nn.functional.gelu
    
    for ti, (noun, attr, combo) in enumerate(TEST_TRIPLES):
        if noun not in single_hs or combo not in combo_hs:
            continue
        
        print(f"    三元组 [{ti+1}/{len(TEST_TRIPLES)}]: {noun}+{attr}={combo}")
        td = {"noun": noun, "attr": attr, "combo": combo, "per_layer": []}
        
        for li in key_layers:
            if li not in ffn_weights:
                continue
            wd = ffn_weights[li]
            if 'W_gate' not in wd:
                continue
            
            W_gate = wd['W_gate']  # [d_model, d_mlp] CPU
            W_in = wd['W_in']
            W_out = wd['W_out']    # [d_mlp, d_model]
            b_in = wd.get('b_in', torch.zeros(d_mlp))
            b_out = wd.get('b_out', torch.zeros(d_model))
            
            # 输入到FFN的h = 上一层的输出
            # hidden_states[li] = 第li层输入 = 第(li-1)层输出(或embedding)
            h_noun = single_hs[noun][li]    # [d_model] — 输入到第li层
            h_combo = combo_hs[combo][li]   # [d_model]
            
            # FFN因果链分解
            gate_pre_n = torch.matmul(h_noun, W_gate)     # [d_mlp]
            gate_pre_c = torch.matmul(h_combo, W_gate)
            in_pre_n = torch.matmul(h_noun, W_in)
            in_pre_c = torch.matmul(h_combo, W_in)
            post_n = act_fn(gate_pre_n) * in_pre_n + b_in
            post_c = act_fn(gate_pre_c) * in_pre_c + b_in
            ffn_out_n = torch.matmul(post_n, W_out) + b_out  # [d_model]
            ffn_out_c = torch.matmul(post_c, W_out) + b_out
            
            # G = h(combo, l+1) - h(noun, l+1) — 下一层的差
            h_next_n = single_hs[noun][li+1]
            h_next_c = combo_hs[combo][li+1]
            G = h_next_c - h_next_n
            
            # delta_ffn = ffn_out(combo) - ffn_out(noun)
            delta_ffn = ffn_out_c - ffn_out_n
            
            # ffn对G的贡献
            ffn_G_cos = F.cosine_similarity(delta_ffn.unsqueeze(0), G.unsqueeze(0)).item()
            ffn_G_proj = ((delta_ffn * G).sum() / (G.norm()**2 + 1e-8)).item()
            ffn_norm_ratio = (delta_ffn.norm() / (G.norm() + 1e-8)).item()
            
            # delta_post分析 — 哪些FFN神经元贡献最大
            delta_post = post_c - post_n
            delta_post_abs = delta_post.abs()
            top_contrib = delta_post_abs.topk(min(50, d_mlp))
            
            # gate激活分析
            gate_active_n = (gate_pre_n > 0).sum().item()
            gate_active_c = (gate_pre_c > 0).sum().item()
            gate_overlap = ((gate_pre_n > 0) & (gate_pre_c > 0)).sum().item()
            
            # gate差异
            delta_gate = gate_pre_c - gate_pre_n
            top_gate_diff = delta_gate.abs().topk(min(20, d_mlp))
            
            # post稀疏度
            sparsity_threshold = delta_post_abs.max() * 0.01
            delta_post_sparse = (delta_post_abs > sparsity_threshold).float().mean().item()
            
            # 线性近似误差: 如果用gate_pre_n近似gate_pre_c, 误差多大?
            gate_lin_err = F.cosine_similarity(gate_pre_n.unsqueeze(0), gate_pre_c.unsqueeze(0)).item()
            
            lld = {
                "layer": li,
                "ffn_G_cos": round(ffn_G_cos, 4),
                "ffn_G_proj": round(ffn_G_proj, 4),
                "ffn_norm_ratio": round(ffn_norm_ratio, 4),
                "gate_active_noun": gate_active_n,
                "gate_active_combo": gate_active_c,
                "gate_active_overlap": gate_overlap,
                "gate_overlap_ratio": round(gate_overlap / (max(gate_active_n, gate_active_c) + 1e-8), 4),
                "top10_contrib_neurons_idx": top_contrib.indices[:10].tolist(),
                "top10_contrib_neurons_val": [round(v, 4) for v in top_contrib.values[:10].tolist()],
                "top10_gate_diff_idx": top_gate_diff.indices[:10].tolist(),
                "top10_gate_diff_val": [round(v, 4) for v in top_gate_diff.values[:10].tolist()],
                "delta_ffn_norm": round(delta_ffn.norm().item(), 2),
                "G_norm": round(G.norm().item(), 2),
                "delta_post_sparsity": round(delta_post_sparse, 4),
                "gate_lin_similarity": round(gate_lin_err, 4),
            }
            td["per_layer"].append(lld)
        
        results["per_triple"].append(td)
    
    # 逐层汇总
    for li in key_layers:
        ffn_G_coses = []
        ffn_norm_ratios = []
        gate_overlaps = []
        gate_sims = []
        
        for td in results["per_triple"]:
            for lld in td["per_layer"]:
                if lld["layer"] == li:
                    ffn_G_coses.append(lld["ffn_G_cos"])
                    ffn_norm_ratios.append(lld["ffn_norm_ratio"])
                    gate_overlaps.append(lld["gate_overlap_ratio"])
                    gate_sims.append(lld["gate_lin_similarity"])
        
        if ffn_G_coses:
            results["per_layer_summary"].append({
                "layer": li,
                "avg_ffn_G_cos": round(np.mean(ffn_G_coses), 4),
                "avg_ffn_norm_ratio": round(np.mean(ffn_norm_ratios), 4),
                "avg_gate_overlap": round(np.mean(gate_overlaps), 4),
                "avg_gate_lin_sim": round(np.mean(gate_sims), 4),
            })
    
    print(f"    P258完成: 分析了{len(results['per_triple'])}个三元组, {len(results['per_layer_summary'])}层汇总")
    return results


# ==================== P259: 苹果-水果编码的权重级来源 ====================

def p259_weight_level_source(ffn_weights, single_hs, n_layers, d_model, d_mlp):
    """
    P259: 苹果-水果编码的权重级来源
    核心问题: 
      - PhaseXLII发现的苹果特异神经元(16-40个), 对应的W_gate/W_in/W_out行/列有什么特征?
      - 水果骨干方向是否由W_out的某些行向量(=FFN输出方向)贡献?
      - W_gate中是否存在"水果检测器"——对水果输入特别敏感的列?
    """
    print(f"\n  P259: 苹果-水果编码的权重级来源")
    
    results = {"per_layer": [], "global_findings": {}}
    
    key_layers = sorted(set([0, 1, 2, 3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]))
    
    fruit_words = [w for w in STIMULI["fruit_family"] if w in single_hs]
    animal_words = [w for w in STIMULI["animal_family"] if w in single_hs]
    vehicle_words = [w for w in STIMULI["vehicle_family"] if w in single_hs]
    non_fruit_words = animal_words + vehicle_words
    
    if not fruit_words or not non_fruit_words:
        print("    缺少必要词汇数据, 跳过P259")
        return results
    
    for li in key_layers:
        if li not in ffn_weights:
            continue
        wd = ffn_weights[li]
        if 'W_gate' not in wd:
            continue
        
        W_gate = wd['W_gate']  # [d_model, d_mlp]
        W_in = wd['W_in']
        W_out = wd['W_out']   # [d_mlp, d_model]
        
        print(f"    分析L{li}...")
        ld = {"layer": li}
        
        # 1. "水果检测器"分析
        # 对每个FFN神经元, 计算: 如果输入是水果方向的h, gate值多大?
        # gate_pre_j = h_fruit · W_gate[:, j]  vs  gate_pre_j = h_animal · W_gate[:, j]
        # 水果方向
        fruit_mean_h = torch.stack([single_hs[w][li+1] for w in fruit_words]).mean(0)  # [d_model]
        non_fruit_mean_h = torch.stack([single_hs[w][li+1] for w in non_fruit_words]).mean(0)
        
        fruit_dir = fruit_mean_h - non_fruit_mean_h
        fruit_dir_norm = F.normalize(fruit_dir.unsqueeze(0), dim=1)  # [1, d_model]
        
        # W_gate的每一列 = FFN神经元的key方向
        # 对每个FFN神经元j, gate_pre_j = h · W_gate[:, j]
        # 水果方向上gate_pre = fruit_dir · W_gate[:, j]
        gate_fruit_proj = torch.matmul(fruit_dir_norm, W_gate).squeeze(0)  # [d_mlp]
        
        # top水果敏感FFN神经元
        top_fruit_sensitive = gate_fruit_proj.abs().topk(min(50, d_mlp))
        ld['top20_fruit_sensitive_idx'] = top_fruit_sensitive.indices[:20].tolist()
        ld['top20_fruit_sensitive_val'] = [round(v, 4) for v in top_fruit_sensitive.values[:20].tolist()]
        
        # 2. "苹果检测器"分析
        if "apple" in single_hs:
            apple_h = single_hs["apple"][li+1]
            other_fruit_h = torch.stack([single_hs[w][li+1] for w in fruit_words if w != "apple"]).mean(0)
            apple_dir = F.normalize((apple_h - other_fruit_h).unsqueeze(0), dim=1)
            
            gate_apple_proj = torch.matmul(apple_dir, W_gate).squeeze(0)
            top_apple_sensitive = gate_apple_proj.abs().topk(min(50, d_mlp))
            ld['top20_apple_sensitive_idx'] = top_apple_sensitive.indices[:20].tolist()
            ld['top20_apple_sensitive_val'] = [round(v, 4) for v in top_apple_sensitive.values[:20].tolist()]
            
            # 水果检测器与苹果检测器的重叠
            fruit_set = set(top_fruit_sensitive.indices[:50].tolist())
            apple_set = set(top_apple_sensitive.indices[:50].tolist())
            ld['fruit_apple_detector_overlap'] = len(fruit_set & apple_set)
            ld['fruit_apple_detector_overlap_ratio'] = round(len(fruit_set & apple_set) / 50, 4)
        
        # 3. W_out行向量的语义聚类
        # W_out的第j行 = FFN神经元j的输出方向
        # 如果某些FFN神经元的输出方向与水果骨干方向对齐, 它们就是"水果输出器"
        W_out_norm = F.normalize(W_out, dim=1)  # [d_mlp, d_model]
        
        fruit_output_cos = torch.matmul(W_out_norm, fruit_dir_norm.T).squeeze()  # [d_mlp]
        top_fruit_output = fruit_output_cos.abs().topk(min(50, d_mlp))
        ld['top20_fruit_output_idx'] = top_fruit_output.indices[:20].tolist()
        ld['top20_fruit_output_val'] = [round(v, 4) for v in top_fruit_output.values[:20].tolist()]
        
        # 4. "水果检测器"和"水果输出器"是否是同一批神经元?
        if 'top20_fruit_sensitive_idx' in ld:
            detector_set = set(ld['top20_fruit_sensitive_idx'][:50] if len(ld['top20_fruit_sensitive_idx']) >= 50 else ld['top20_fruit_sensitive_idx'])
            output_set = set(top_fruit_output.indices[:50].tolist())
            ld['detector_output_overlap'] = len(detector_set & output_set)
        
        # 5. W_gate vs W_in: 水果敏感神经元的gate和value方向关系
        # 如果gate检测水果, value应该输出水果方向
        for k in [10, 50]:
            top_k_idx = top_fruit_sensitive.indices[:k]
            # 这些神经元的W_in列向量与水果方向的cos
            in_dirs = W_in[:, top_k_idx]  # [d_model, k]
            in_fruit_cos = F.cosine_similarity(
                in_dirs.T,  # [k, d_model]
                fruit_dir_norm.expand(k, -1),  # [k, d_model]
            )
            ld[f'fruit_detector_in_fruit_cos_top{k}_mean'] = round(in_fruit_cos.mean().item(), 4)
            
            # 这些神经元的W_out行向量与水果方向的cos
            out_dirs = W_out[top_k_idx]  # [k, d_model]
            out_fruit_cos = F.cosine_similarity(
                F.normalize(out_dirs, dim=1),
                fruit_dir_norm.expand(k, -1),
            )
            ld[f'fruit_detector_out_fruit_cos_top{k}_mean'] = round(out_fruit_cos.mean().item(), 4)
        
        # 6. 跨层水果检测器的稳定性
        ld['gate_fruit_proj_mean'] = round(gate_fruit_proj.mean().item(), 6)
        ld['gate_fruit_proj_std'] = round(gate_fruit_proj.std().item(), 6)
        ld['gate_fruit_proj_max'] = round(gate_fruit_proj.abs().max().item(), 4)
        
        results["per_layer"].append(ld)
    
    # 全局分析: 跨层稳定的水果检测器
    if len(results["per_layer"]) >= 2:
        all_fruit_detectors = {}
        all_fruit_outputs = {}
        for ld in results["per_layer"]:
            for idx in ld.get('top20_fruit_sensitive_idx', []):
                all_fruit_detectors[idx] = all_fruit_detectors.get(idx, 0) + 1
            for idx in ld.get('top20_fruit_output_idx', []):
                all_fruit_outputs[idx] = all_fruit_outputs.get(idx, 0) + 1
        
        stable_detectors = sorted(all_fruit_detectors.items(), key=lambda x: -x[1])
        stable_detectors = [(idx, cnt) for idx, cnt in stable_detectors if cnt >= 3]
        stable_outputs = sorted(all_fruit_outputs.items(), key=lambda x: -x[1])
        stable_outputs = [(idx, cnt) for idx, cnt in stable_outputs if cnt >= 3]
        
        results["global_findings"] = {
            "stable_fruit_detectors": stable_detectors[:20],
            "stable_fruit_outputs": stable_outputs[:20],
            "detector_output_stable_overlap": len(
                set([x[0] for x in stable_detectors]) & set([x[0] for x in stable_outputs])
            ),
        }
    
    print(f"    P259完成: 分析了{len(results['per_layer'])}层")
    return results


# ==================== 主函数 ====================

def run_model(model_name):
    """运行单个模型的所有实验"""
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    # 加载模型
    print("  加载模型...")
    mdl, tok, device = load_model(model_name)
    n_layers = mdl.config.num_hidden_layers
    d_model = mdl.config.hidden_size
    # 推断d_mlp
    d_mlp = None
    layers = mdl.model.layers if hasattr(mdl, 'model') and hasattr(mdl.model, 'layers') else None
    if layers is not None:
        mlp = layers[0].mlp
        if hasattr(mlp, 'W_gate'):
            d_mlp = mlp.W_gate.shape[1]
        elif hasattr(mlp, 'W_in'):
            d_mlp = mlp.W_in.shape[1]
        else:
            for name, param in mlp.named_parameters():
                if 'weight' in name and param.shape[0] > d_model:
                    d_mlp = param.shape[0]
                    break
    
    print(f"  模型结构: {n_layers}层, d_model={d_model}, d_mlp={d_mlp}")
    
    # 收集数据
    print("\n  === 数据收集 ===")
    single_hs, combo_hs = collect_all_data(mdl, tok, device)
    
    # 提取FFN权重
    print("\n  === FFN权重提取 ===")
    ffn_weights = extract_ffn_weights(mdl, device)
    
    # 释放模型(仅P258需要模型)
    # P257和P259只需要权重和hidden states, 不需要模型
    # P258需要模型做前向传播
    
    # P257: FFN权重矩阵全景分析
    print("\n  === P257: FFN权重矩阵全景分析 ===")
    p257_results = p257_ffn_weight_analysis(ffn_weights, single_hs, n_layers, d_model, d_mlp)
    
    # P258: FFN因果链精确分解(需要模型)
    print("\n  === P258: FFN因果链精确分解 ===")
    p258_results = p258_ffn_causal_chain(ffn_weights, single_hs, combo_hs, n_layers, d_model, d_mlp)
    
    # P259: 苹果-水果编码的权重级来源
    print("\n  === P259: 苹果-水果编码的权重级来源 ===")
    p259_results = p259_weight_level_source(ffn_weights, single_hs, n_layers, d_model, d_mlp)
    
    # 释放模型
    print("\n  释放模型...")
    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()
    
    # 汇总结果
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "d_mlp": d_mlp,
        "p257_ffn_weight_analysis": p257_results,
        "p258_ffn_causal_chain": p258_results,
        "p259_weight_level_source": p259_results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    
    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"phase_xliii_p257_259_{model_name}_{ts}.json"
    
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
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n  结果保存: {out_path}")
    print(f"  总用时: {round(time.time() - t0, 1)}秒")
    
    return all_results


def print_summary(all_results):
    """打印关键发现摘要"""
    model = all_results["model"]
    n_layers = all_results["n_layers"]
    d_model = all_results["d_model"]
    d_mlp = all_results["d_mlp"]
    
    print(f"\n{'='*60}")
    print(f"  {model} 关键发现摘要")
    print(f"  结构: {n_layers}L, d={d_model}, d_mlp={d_mlp}")
    print(f"{'='*60}")
    
    # P257 摘要
    p257 = all_results["p257_ffn_weight_analysis"]
    print(f"\n  P257: FFN权重矩阵全景分析")
    for ld in p257["per_layer"]:
        li = ld["layer"]
        print(f"    L{li}: key_pairwise_cos={ld.get('key_pairwise_cos_mean','?')}, "
              f"gate_in_cos={ld.get('gate_in_cos_mean','?')}, "
              f"gate_row_norm_cv={ld.get('gate_row_norm_cv','?')}")
        if 'fruit_dir_Wout_cos_max' in ld:
            print(f"         fruit_dir→Wout_cos_max={ld['fruit_dir_Wout_cos_max']}")
    
    gs = p257.get("global_summary", {})
    if gs:
        print(f"    全局: key_cos={gs.get('key_cos_early_vs_late','?')}, "
              f"gate_in_cos={gs.get('gate_in_cos_early_vs_late','?')}")
    
    # P258 摘要
    p258 = all_results["p258_ffn_causal_chain"]
    print(f"\n  P258: FFN因果链精确分解")
    for ls in p258["per_layer_summary"]:
        print(f"    L{ls['layer']}: ffn_G_cos={ls['avg_ffn_G_cos']}, "
              f"ffn_norm_ratio={ls['avg_ffn_norm_ratio']}, "
              f"gate_overlap={ls['avg_gate_overlap']}, "
              f"gate_sim={ls.get('avg_gate_lin_sim','?')}")
    
    # P259 摘要
    p259 = all_results["p259_weight_level_source"]
    print(f"\n  P259: 苹果-水果编码的权重级来源")
    gf = p259.get("global_findings", {})
    print(f"    跨层稳定水果检测器: {len(gf.get('stable_fruit_detectors', []))}个")
    print(f"    跨层稳定水果输出器: {len(gf.get('stable_fruit_outputs', []))}个")
    print(f"    检测器-输出器重叠: {gf.get('detector_output_stable_overlap', '?')}")
    
    for ld in p259["per_layer"][:3]:
        li = ld["layer"]
        print(f"    L{li}: fruit_detector_in_fruit_cos={ld.get('fruit_detector_in_fruit_cos_top10_mean','?')}, "
              f"fruit_detector_out_fruit_cos={ld.get('fruit_detector_out_fruit_cos_top10_mean','?')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()
    
    results = run_model(args.model)
    print_summary(results)
