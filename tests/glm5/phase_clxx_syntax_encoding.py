"""
Phase CLXX: 语法关系编码 — 从静态子空间到动态激活动力学
============================================================

CLXIX确认了W_U的子空间正交结构: 风格⊥语义≈80-86°, 风格⊥语法≈68-84°
但这是"静态结构"——只分析了权重矩阵W_U, 没有分析"动态过程"。

Phase CLXX目标: 分析h在正交子空间间的动态流动, 揭示语法/推理的编码机制

P731: 词对(w1,w2)的关系编码结构
  - W_E[w1]和W_E[w2]的内积 → 语义相关性
  - W_U[w1]和W_U[w2]的内积 → 语法兼容性  
  - 两者是否独立? 语义相关≠语法兼容?
  - 关系向量: W_U[w1]-W_U[w2] 在各子空间的投影

P732: 句式模板的权重指纹
  - 分析不同句式下各层的激活模式差异
  - 句式 = 特定层的特定子空间激活模式?
  - 方法: 对比 "The cat sits" vs "Does the cat sit?" 的h在各子空间的投影

P733: 语法层级在Transformer层中的对应
  - 段落结构 → 哪些层编码段落边界?
  - 从句嵌入 → 哪些层编码嵌套深度?
  - 词序约束 → 哪些层编码词序规则?
  - 方法: 用Hansard语料中的嵌套从句, 分析各层h在语法子空间的投影

核心方法论转变:
  CLXIX: 分析 W_U (静态权重) → 子空间结构
  CLXX:  分析 h(t) (动态激活) → 在子空间间的流动
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import json
import time
import argparse
import gc
from collections import defaultdict
from scipy import stats
from sklearn.decomposition import PCA

from model_utils import (
    load_model, get_layers, get_model_info,
    get_W_U, release_model, get_sample_layers
)


def to_numpy(tensor_or_array):
    """统一转换为numpy float32数组"""
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    return tensor_or_array.detach().cpu().numpy().astype(np.float32)


# ============================================================
# P731: 词对关系编码
# ============================================================

# 定义词对: 语义相关但语法不同 / 语法兼容但语义不同 / 两者都相关
WORD_PAIRS = {
    # 语义相关, 语法兼容 (同义词/近义词)
    'synonym': [
        ('big', 'large'), ('small', 'tiny'), ('fast', 'quick'),
        ('happy', 'glad'), ('sad', 'unhappy'), ('smart', 'clever'),
        ('beautiful', 'pretty'), ('strong', 'powerful'),
        ('begin', 'start'), ('end', 'finish'),
    ],
    # 语义相关, 语法不兼容 (不同词类)
    'derivational': [
        ('run', 'runner'), ('teach', 'teacher'), ('write', 'writer'),
        ('act', 'action'), ('create', 'creation'), ('decide', 'decision'),
        ('happy', 'happiness'), ('strong', 'strength'),
        ('beautiful', 'beauty'), ('quick', 'quickly'),
    ],
    # 语义不相关, 语法兼容 (同词类)
    'same_pos': [
        ('cat', 'table'), ('run', 'think'), ('big', 'red'),
        ('the', 'a'), ('in', 'on'), ('and', 'but'),
        ('king', 'mountain'), ('water', 'idea'),
        ('always', 'never'), ('here', 'there'),
    ],
    # 语义相反, 语法兼容 (反义词)
    'antonym': [
        ('big', 'small'), ('hot', 'cold'), ('good', 'bad'),
        ('up', 'down'), ('in', 'out'), ('on', 'off'),
        ('light', 'dark'), ('fast', 'slow'),
        ('love', 'hate'), ('war', 'peace'),
    ],
    # 语法搭配 (高频共现)
    'collocation': [
        ('the', 'cat'), ('big', 'house'), ('run', 'fast'),
        ('make', 'decision'), ('take', 'time'),
        ('heavy', 'rain'), ('strong', 'wind'),
        ('deep', 'water'), ('high', 'mountain'),
        ('dark', 'night'), ('bright', 'sun'),
    ],
}


def P731_word_pair_relations(model, tokenizer, device, model_info, model_name):
    """
    P731: 词对关系编码 — 分析语义相关vs语法兼容在W_U和W_E中的分离
    """
    print("\n=== P731: 词对关系编码 ===")
    
    W_U = to_numpy(get_W_U(model))
    d = W_U.shape[1]
    
    # 获取W_E (embedding矩阵)
    W_E = None
    try:
        if hasattr(model, 'W_E'):
            W_E = to_numpy(model.W_E)
        elif hasattr(model, 'embed'):
            W_E = to_numpy(model.embed.W_E)
    except:
        pass
    
    # tokenize词对
    def get_word_vec(word, W_matrix, tokenizer):
        """获取词在W矩阵中的向量"""
        for prefix in ['', ' ', '▁']:
            try:
                ids = tokenizer.encode(prefix + word, add_special_tokens=False)
                if len(ids) == 1 and ids[0] < W_matrix.shape[0]:
                    return W_matrix[ids[0]], ids[0]
            except:
                pass
        return None, None
    
    results = {}
    
    for pair_type, pairs in WORD_PAIRS.items():
        type_results = {
            'W_U_cos': [],
            'W_U_diff_norm': [],
            'W_E_cos': [],
        }
        
        for w1, w2 in pairs:
            v1_u, id1 = get_word_vec(w1, W_U, tokenizer)
            v2_u, id2 = get_word_vec(w2, W_U, tokenizer)
            
            if v1_u is None or v2_u is None:
                continue
            
            # W_U中的余弦相似度 → 语法兼容性
            cos_u = np.dot(v1_u, v2_u) / (np.linalg.norm(v1_u) * np.linalg.norm(v2_u) + 1e-10)
            type_results['W_U_cos'].append(float(cos_u))
            
            # 差向量范数 → 编码距离
            diff_norm = np.linalg.norm(v1_u - v2_u)
            type_results['W_U_diff_norm'].append(float(diff_norm))
            
            # W_E中的余弦相似度 → 语义相关性
            if W_E is not None:
                v1_e, _ = get_word_vec(w1, W_E, tokenizer)
                v2_e, _ = get_word_vec(w2, W_E, tokenizer)
                if v1_e is not None and v2_e is not None:
                    cos_e = np.dot(v1_e, v2_e) / (np.linalg.norm(v1_e) * np.linalg.norm(v2_e) + 1e-10)
                    type_results['W_E_cos'].append(float(cos_e))
        
        # 汇总
        n_pairs = len(type_results['W_U_cos'])
        if n_pairs > 0:
            results[pair_type] = {
                'n_pairs': n_pairs,
                'W_U_cos_mean': float(np.mean(type_results['W_U_cos'])),
                'W_U_cos_std': float(np.std(type_results['W_U_cos'])),
                'W_U_diff_norm_mean': float(np.mean(type_results['W_U_diff_norm'])),
                'W_E_cos_mean': float(np.mean(type_results['W_E_cos'])) if type_results['W_E_cos'] else None,
                'W_E_cos_std': float(np.std(type_results['W_E_cos'])) if len(type_results['W_E_cos']) > 1 else None,
            }
            print(f"  {pair_type}: n={n_pairs}, W_U_cos={results[pair_type]['W_U_cos_mean']:.4f}, "
                  f"W_E_cos={results[pair_type].get('W_E_cos_mean', 'N/A')}")
    
    # 关键分析: 语义相关 vs 语法兼容是否独立?
    print("\n  语义vs语法的独立性分析:")
    
    # 如果语义(W_E_cos)和语法(W_U_cos)独立, 则:
    # synonym(语义高)和same_pos(语法高)的W_U_cos应该不同
    syn_wu = results.get('synonym', {}).get('W_U_cos_mean', 0)
    same_wu = results.get('same_pos', {}).get('W_U_cos_mean', 0)
    deriv_wu = results.get('derivational', {}).get('W_U_cos_mean', 0)
    anto_wu = results.get('antonym', {}).get('W_U_cos_mean', 0)
    coll_wu = results.get('collocation', {}).get('W_U_cos_mean', 0)
    
    syn_we = results.get('synonym', {}).get('W_E_cos_mean', 0)
    same_we = results.get('same_pos', {}).get('W_E_cos_mean', 0)
    
    syn_we_str = f"{syn_we:.4f}" if syn_we is not None else "N/A"
    same_we_str = f"{same_we:.4f}" if same_we is not None else "N/A"
    print(f"    同义词 W_U_cos={syn_wu:.4f}, W_E_cos={syn_we_str}")
    print(f"    同词类 W_U_cos={same_wu:.4f}, W_E_cos={same_we_str}")
    print(f"    派生词 W_U_cos={deriv_wu:.4f}")
    print(f"    反义词 W_U_cos={anto_wu:.4f}")
    print(f"    搭配词 W_U_cos={coll_wu:.4f}")
    
    # 差向量分析: 差向量指向什么方向?
    print("\n  差向量方向分析:")
    
    # 收集所有词对的差向量
    all_diffs = {}
    for pair_type, pairs in WORD_PAIRS.items():
        diffs = []
        for w1, w2 in pairs:
            v1, _ = get_word_vec(w1, W_U, tokenizer)
            v2, _ = get_word_vec(w2, W_U, tokenizer)
            if v1 is not None and v2 is not None:
                diffs.append(v1 - v2)
        if diffs:
            all_diffs[pair_type] = np.array(diffs)
    
    # 差向量的范数分布
    for ptype, diffs in all_diffs.items():
        norms = np.linalg.norm(diffs, axis=1)
        print(f"    {ptype}: diff_norm mean={norms.mean():.2f}, std={norms.std():.2f}")
    
    results['summary'] = {
        'synonym_W_U_cos': syn_wu,
        'same_pos_W_U_cos': same_wu,
        'derivational_W_U_cos': deriv_wu,
        'antonym_W_U_cos': anto_wu,
        'collocation_W_U_cos': coll_wu,
        'conclusion': '语义和语法在W_U中部分分离' if abs(syn_wu - same_wu) > 0.02 else '语义和语法在W_U中混合',
    }
    
    print("\n=== P731 完成 ===")
    return results


# ============================================================
# P732: 句式模板的权重指纹
# ============================================================

# 定义不同句式模板
SENTENCE_TEMPLATES = {
    'declarative': [  # 陈述句
        "The cat sits on the mat.",
        "A dog runs in the park.",
        "The sun shines brightly.",
        "Birds fly across the sky.",
        "Water flows down the river.",
    ],
    'interrogative': [  # 疑问句
        "Does the cat sit on the mat?",
        "Is a dog running in the park?",
        "Will the sun shine brightly?",
        "Do birds fly across the sky?",
        "Can water flow down the river?",
    ],
    'negative': [  # 否定句
        "The cat does not sit on the mat.",
        "A dog is not running in the park.",
        "The sun does not shine brightly.",
        "Birds do not fly across the sky.",
        "Water cannot flow down the river.",
    ],
    'passive': [  # 被动句
        "The mat is sat on by the cat.",
        "The park is run in by a dog.",
        "Brightly is shone by the sun.",
        "The sky is flown across by birds.",
        "The river is flowed down by water.",
    ],
    'conditional': [  # 条件句
        "If the cat sits on the mat, it will sleep.",
        "If a dog runs in the park, it will be happy.",
        "If the sun shines, we will go outside.",
        "If birds fly, they will find food.",
        "If water flows, the river will grow.",
    ],
    'relative_clause': [  # 关系从句
        "The cat that sits on the mat is black.",
        "The dog which runs in the park is friendly.",
        "The sun that shines brightly is hot.",
        "The birds that fly across the sky are free.",
        "The water which flows down the river is cold.",
    ],
}


def P732_sentence_templates(model, tokenizer, device, model_info, model_name):
    """
    P732: 句式模板的权重指纹
    - 不同句式在各层的激活模式差异
    - 句式 = 特定层的特定子空间激活模式?
    """
    print("\n=== P732: 句式模板的权重指纹 ===")
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 获取W_U的子空间 (复用CLXIX的方法)
    W_U = to_numpy(get_W_U(model))
    
    # 用W_U的PCA定义子空间
    n_pc = 20
    pca = PCA(n_components=min(n_pc, W_U.shape[1]))
    pca.fit(W_U)
    pc_components = pca.components_  # [n_pc, d]
    
    results = {}
    
    # 对每种句式, 获取各层的h, 分析在子空间的投影
    for template_name, sentences in SENTENCE_TEMPLATES.items():
        print(f"\n  分析句式: {template_name}...")
        
        template_projections = []  # [n_sentences, n_layers, n_pc]
        
        for sent in sentences:
            # tokenize
            tokens = tokenizer.encode(sent, return_tensors='pt').to(device)
            
            # 前向传播, 获取各层残差流
            with torch.no_grad():
                try:
                    logits, cache = model.run_with_cache(tokens)
                except Exception as e:
                    print(f"    前向传播失败: {e}")
                    continue
            
            # 获取各层的残差流 (最后一个token位置)
            last_pos = tokens.shape[1] - 1
            layer_projections = []
            
            for layer_idx in range(min(n_layers, 36)):  # 最多分析36层
                hook_name = f'blocks.{layer_idx}.hook_resid_post'
                if hook_name in cache:
                    h = to_numpy(cache[hook_name][0, last_pos])  # [d_model]
                else:
                    continue
                
                # 投影到W_U的PCA子空间
                proj = h @ pc_components.T  # [n_pc]
                layer_projections.append(proj)
            
            if layer_projections:
                template_projections.append(np.array(layer_projections))
        
        if not template_projections:
            results[template_name] = {'error': 'no valid sentences'}
            continue
        
        # 平均各句子的投影
        avg_proj = np.mean(template_projections, axis=0)  # [n_layers, n_pc]
        
        results[template_name] = {
            'n_sentences': len(template_projections),
            'n_layers': avg_proj.shape[0],
            'proj_shape': list(avg_proj.shape),
            # 各层在各PC上的平均投影(绝对值)
            'layer_pc_energy': [
                {
                    'layer': l,
                    'total_energy': float(np.sum(avg_proj[l]**2)),
                    'pc0_energy': float(avg_proj[l, 0]**2),
                    'pc1_energy': float(avg_proj[l, min(1, avg_proj.shape[1]-1)]**2),
                    'top_pc': int(np.argmax(np.abs(avg_proj[l]))),
                }
                for l in range(avg_proj.shape[0])
            ],
            # 最后3层的PC分布
            'final_layers_pc': [
                {
                    'layer': l,
                    'pc_distribution': np.abs(avg_proj[l]).tolist()[:10],
                }
                for l in range(max(0, avg_proj.shape[0]-3), avg_proj.shape[0])
            ],
        }
        
        # 释放cache
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    
    # 句式间差异分析
    print("\n  句式间差异分析:")
    
    # 比较不同句式在最后3层的PC分布
    template_final_projs = {}
    for tname, tdata in results.items():
        if 'error' in tdata:
            continue
        # 获取最后3层的平均PC分布
        final_layers = tdata.get('final_layers_pc', [])
        if final_layers:
            last_layer_pc = final_layers[-1].get('pc_distribution', [])
            template_final_projs[tname] = last_layer_pc
    
    if len(template_final_projs) > 1:
        # 计算句式间的PC分布差异
        tnames = list(template_final_projs.keys())
        pairwise_diff = {}
        for i in range(len(tnames)):
            for j in range(i+1, len(tnames)):
                p1 = np.array(template_final_projs[tnames[i]])
                p2 = np.array(template_final_projs[tnames[j]])
                if len(p1) > 0 and len(p2) > 0:
                    min_len = min(len(p1), len(p2))
                    cos_sim = np.dot(p1[:min_len], p2[:min_len]) / (
                        np.linalg.norm(p1[:min_len]) * np.linalg.norm(p2[:min_len]) + 1e-10)
                    pairwise_diff[f"{tnames[i]}_vs_{tnames[j]}"] = float(cos_sim)
        
        results['pairwise_similarity'] = pairwise_diff
        
        # 找出最不同和最相似的句式对
        if pairwise_diff:
            sorted_pairs = sorted(pairwise_diff.items(), key=lambda x: x[1])
            print(f"    最不同的句式对: {sorted_pairs[0][0]} (cos={sorted_pairs[0][1]:.4f})")
            print(f"    最相似的句式对: {sorted_pairs[-1][0]} (cos={sorted_pairs[-1][1]:.4f})")
    
    print("\n=== P732 完成 ===")
    return results


# ============================================================
# P733: 语法层级在Transformer层中的对应
# ============================================================

# 定义不同嵌套深度的句子
NESTING_SENTENCES = {
    'depth_0': [  # 简单句, 无嵌套
        "The cat sleeps.",
        "Dogs bark loudly.",
        "Rain falls gently.",
        "Birds sing beautifully.",
        "Time passes quickly.",
    ],
    'depth_1': [  # 一级嵌套(1个从句)
        "The cat that is black sleeps.",
        "Dogs which are loyal bark loudly.",
        "Rain that comes from clouds falls gently.",
        "Birds that migrate south sing beautifully.",
        "Time which waits for no one passes quickly.",
    ],
    'depth_2': [  # 二级嵌套(2个从句)
        "The cat that is black which has green eyes sleeps.",
        "Dogs which are loyal that guard the house bark loudly.",
        "Rain that comes from clouds which are dark falls gently.",
    ],
    'depth_3': [  # 三级嵌套(3个从句)
        "The cat that is black which has green eyes that sparkle sleeps.",
    ],
    'conjunction': [  # 并列结构
        "The cat sleeps and the dog barks.",
        "Birds sing and fish swim.",
        "Rain falls and wind blows.",
        "Time passes and life goes on.",
        "Water flows and fire burns.",
    ],
    'subordination': [  # 主从结构
        "The cat sleeps because it is tired.",
        "Dogs bark when they see strangers.",
        "Rain falls if clouds are heavy.",
        "Birds sing while the sun shines.",
        "Time passes before we know it.",
    ],
}


def P733_syntax_hierarchy(model, tokenizer, device, model_info, model_name):
    """
    P733: 语法层级在Transformer层中的对应
    - 嵌套深度 → 哪些层编码嵌套?
    - 并列vs主从 → 哪些层区分结构类型?
    """
    print("\n=== P733: 语法层级在Transformer层中的对应 ===")
    
    n_layers = model_info.n_layers
    W_U = to_numpy(get_W_U(model))
    
    # PCA子空间
    n_pc = 20
    pca = PCA(n_components=min(n_pc, W_U.shape[1]))
    pca.fit(W_U)
    pc_components = pca.components_
    
    results = {}
    
    for depth_name, sentences in NESTING_SENTENCES.items():
        print(f"\n  分析嵌套深度: {depth_name}...")
        
        depth_projections = []
        
        for sent in sentences:
            tokens = tokenizer.encode(sent, return_tensors='pt').to(device)
            
            with torch.no_grad():
                try:
                    logits, cache = model.run_with_cache(tokens)
                except Exception as e:
                    continue
            
            # 获取各层残差流 - 关注关键位置
            # 对于嵌套从句, 关注关系代词(that/which/who)位置
            last_pos = tokens.shape[1] - 1
            layer_projections = []
            
            for layer_idx in range(min(n_layers, 36)):
                hook_name = f'blocks.{layer_idx}.hook_resid_post'
                if hook_name in cache:
                    h = to_numpy(cache[hook_name][0, last_pos])
                    proj = h @ pc_components.T
                    layer_projections.append(proj)
            
            if layer_projections:
                depth_projections.append(np.array(layer_projections))
            
            del cache
            gc.collect()
            torch.cuda.empty_cache()
        
        if not depth_projections:
            results[depth_name] = {'error': 'no valid sentences'}
            continue
        
        avg_proj = np.mean(depth_projections, axis=0)  # [n_layers, n_pc]
        
        # 分析各层的"语法信号强度"
        # 定义: 语法信号 = h在语法相关PC上的投影
        # 简单假设: PC1-5承载语法信息(基于CLXIX的发现)
        
        results[depth_name] = {
            'n_sentences': len(depth_projections),
            'n_layers': avg_proj.shape[0],
            # 各层的总能量
            'layer_energy': [
                float(np.sum(avg_proj[l]**2))
                for l in range(avg_proj.shape[0])
            ],
            # 各层的PC1-5能量占比
            'syntax_energy_ratio': [
                float(np.sum(avg_proj[l, :5]**2) / (np.sum(avg_proj[l]**2) + 1e-10))
                for l in range(avg_proj.shape[0])
            ],
        }
    
    # 嵌套深度 vs 语法信号的关系
    print("\n  嵌套深度 vs 语法信号:")
    
    depth_energy = {}
    for depth_name, data in results.items():
        if 'error' in data:
            continue
        # 中间层(1/3到2/3)的语法信号
        n_l = data['n_layers']
        mid_start = n_l // 3
        mid_end = 2 * n_l // 3
        mid_syntax = np.mean(data['syntax_energy_ratio'][mid_start:mid_end]) if data['syntax_energy_ratio'] else 0
        depth_energy[depth_name] = mid_syntax
        print(f"    {depth_name}: 中间层语法信号={mid_syntax:.4f}")
    
    # 层级编码假说: 深度嵌套 → 更深的层参与
    # 检验: depth_3的语法信号是否比depth_0在更深层更强?
    
    results['depth_analysis'] = {
        'mid_layer_syntax_signal': depth_energy,
        'conclusion': '嵌套越深, 中间层语法信号越强' if depth_energy.get('depth_3', 0) > depth_energy.get('depth_0', 0) else '嵌套深度与中间层语法信号无单调关系',
    }
    
    # 并列vs主从的层差异
    print("\n  并列vs主从的层差异:")
    conj_data = results.get('conjunction', {})
    sub_data = results.get('subordination', {})
    
    if 'layer_energy' in conj_data and 'layer_energy' in sub_data:
        conj_energy = np.array(conj_data['layer_energy'])
        sub_energy = np.array(sub_data['layer_energy'])
        min_len = min(len(conj_energy), len(sub_energy))
        diff = sub_energy[:min_len] - conj_energy[:min_len]
        
        # 找到差异最大的层
        max_diff_layer = int(np.argmax(np.abs(diff)))
        print(f"    差异最大的层: L{max_diff_layer} (diff={diff[max_diff_layer]:.2f})")
        
        results['conj_vs_sub'] = {
            'max_diff_layer': max_diff_layer,
            'max_diff_value': float(diff[max_diff_layer]),
            'layer_diffs': diff.tolist(),
        }
    
    print("\n=== P733 完成 ===")
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['qwen3', 'deepseek7b', 'glm4'])
    args = parser.parse_args()
    model_name = args.model
    
    print(f"\n{'='*60}")
    print(f"Phase CLXX: 语法关系编码 — {model_name}")
    print(f"{'='*60}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    model.tokenizer = tokenizer
    
    results = {}
    
    # P731 (不需要cache, 只需W_U和W_E)
    try:
        results["P731"] = P731_word_pair_relations(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P731失败: {e}")
        import traceback
        traceback.print_exc()
        results["P731"] = {"error": str(e)}
    
    # P732 (需要run_with_cache)
    try:
        results["P732"] = P732_sentence_templates(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P732失败: {e}")
        import traceback
        traceback.print_exc()
        results["P732"] = {"error": str(e)}
    
    # P733 (需要run_with_cache)
    try:
        results["P733"] = P733_syntax_hierarchy(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P733失败: {e}")
        import traceback
        traceback.print_exc()
        results["P733"] = {"error": str(e)}
    
    # 保存结果
    save_dir = f'results/phase_clxx'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{model_name}_results.json'
    
    results['model_info'] = model_info
    results['model_name'] = model_name
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存到: {save_path}")
    
    # 释放GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
