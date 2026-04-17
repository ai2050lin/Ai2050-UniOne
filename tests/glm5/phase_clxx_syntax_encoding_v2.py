"""
Phase CLXX v2: 语法关系编码 — 使用register_forward_hook替代run_with_cache
============================================================
P732: 句式模板的权重指纹 — 不同句式在各层的激活模式差异
P733: 语法层级在Transformer层中的对应 — 嵌套深度→层编码

关键改进: 使用register_forward_hook获取各层残差流, 兼容HuggingFace模型
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import gc
import numpy as np
import torch
from sklearn.decomposition import PCA
from pathlib import Path

from model_utils import load_model, get_model_info, get_W_U, get_layers


def to_numpy(tensor_or_array):
    """统一转换为numpy float32数组"""
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    return tensor_or_array.detach().cpu().float().numpy().astype(np.float32)


def get_residual_streams(model, tokenizer, sentences, device, n_layers):
    """
    使用register_forward_hook获取各层残差流
    
    Returns:
        dict: {sentence_idx: {layer_idx: h_vector}}
    """
    all_residuals = {}
    
    for sent_idx, sent in enumerate(sentences):
        tokens = tokenizer.encode(sent, return_tensors='pt').to(device)
        
        # 注册hook获取各层残差流
        layer_outputs = {}
        hooks = []
        
        layers = get_layers(model)
        
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output可能是tuple, 第一个元素是hidden_states
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                layer_outputs[layer_idx] = hidden.detach()
            return hook_fn
        
        for layer_idx, layer in enumerate(layers[:n_layers]):
            h = layer.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)
        
        # 前向传播
        with torch.no_grad():
            try:
                outputs = model(tokens)
            except Exception as e:
                print(f"    前向传播失败: {e}")
                for h in hooks:
                    h.remove()
                continue
        
        # 移除hook
        for h in hooks:
            h.remove()
        
        # 提取各层最后一个token的残差流
        sent_residuals = {}
        last_pos = tokens.shape[1] - 1
        for layer_idx, hidden in layer_outputs.items():
            if len(hidden.shape) == 3:  # [batch, seq, dim]
                h_vec = to_numpy(hidden[0, last_pos])  # [d_model]
            elif len(hidden.shape) == 2:  # [seq, dim]
                h_vec = to_numpy(hidden[last_pos])
            else:
                continue
            sent_residuals[layer_idx] = h_vec
        
        all_residuals[sent_idx] = sent_residuals
        
        # 释放内存
        del layer_outputs
        gc.collect()
        torch.cuda.empty_cache()
    
    return all_residuals


# ============================================================
# 句式模板定义
# ============================================================

SENTENCE_TEMPLATES = {
    'declarative': [
        "The cat sits on the mat.",
        "A dog runs in the park.",
        "The sun shines brightly.",
        "Birds fly across the sky.",
        "Water flows down the river.",
    ],
    'interrogative': [
        "Does the cat sit on the mat?",
        "Is a dog running in the park?",
        "Will the sun shine brightly?",
        "Do birds fly across the sky?",
        "Can water flow down the river?",
    ],
    'negative': [
        "The cat does not sit on the mat.",
        "A dog is not running in the park.",
        "The sun does not shine brightly.",
        "Birds do not fly across the sky.",
        "Water cannot flow down the river.",
    ],
    'passive': [
        "The mat is sat on by the cat.",
        "The park is run in by a dog.",
        "Brightly is shone by the sun.",
        "The sky is flown across by birds.",
        "The river is flowed down by water.",
    ],
    'conditional': [
        "If the cat sits on the mat, it will sleep.",
        "If a dog runs in the park, it will be happy.",
        "If the sun shines, we will go outside.",
        "If birds fly, they will find food.",
        "If water flows, the river will grow.",
    ],
    'relative_clause': [
        "The cat that sits on the mat is black.",
        "The dog which runs in the park is friendly.",
        "The sun that shines brightly is hot.",
        "The birds that fly across the sky are free.",
        "The water which flows down the river is cold.",
    ],
}

# 嵌套深度句子
NESTING_SENTENCES = {
    'depth_0': [
        "The cat sleeps.",
        "Dogs bark loudly.",
        "Rain falls gently.",
        "Birds sing beautifully.",
        "Time passes quickly.",
    ],
    'depth_1': [
        "The cat that is black sleeps.",
        "Dogs which are loyal bark loudly.",
        "Rain that comes from clouds falls gently.",
        "Birds that migrate south sing beautifully.",
        "Time which waits for no one passes quickly.",
    ],
    'depth_2': [
        "The cat that is black which has green eyes sleeps.",
        "Dogs which are loyal that guard the house bark loudly.",
        "Rain that comes from clouds which are dark falls gently.",
    ],
    'conjunction': [
        "The cat sleeps and the dog barks.",
        "Birds sing and fish swim.",
        "Rain falls and wind blows.",
        "Time passes and life goes on.",
        "Water flows and fire burns.",
    ],
    'subordination': [
        "The cat sleeps because it is tired.",
        "Dogs bark when they see strangers.",
        "Rain falls if clouds are heavy.",
        "Birds sing while the sun shines.",
        "Time passes before we know it.",
    ],
}


# ============================================================
# P732: 句式模板的权重指纹
# ============================================================

def P732_sentence_templates(model, tokenizer, device, model_info, model_name):
    """
    P732: 句式模板的权重指纹
    - 不同句式在各层的激活模式差异
    - 句式 = 特定层的特定子空间激活模式?
    """
    print("\n=== P732: 句式模板的权重指纹 ===")
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 获取W_U的子空间
    W_U = to_numpy(get_W_U(model))
    
    n_pc = 20
    pca = PCA(n_components=min(n_pc, W_U.shape[1]))
    pca.fit(W_U)
    pc_components = pca.components_  # [n_pc, d]
    
    results = {}
    
    for template_name, sentences in SENTENCE_TEMPLATES.items():
        print(f"\n  分析句式: {template_name}...")
        
        # 获取各层残差流
        all_residuals = get_residual_streams(model, tokenizer, sentences, device, min(n_layers, 36))
        
        if not all_residuals:
            results[template_name] = {'error': 'no valid sentences'}
            continue
        
        # 计算各句子的投影
        template_projections = []
        for sent_idx in range(len(sentences)):
            if sent_idx not in all_residuals:
                continue
            layer_projs = []
            for layer_idx in sorted(all_residuals[sent_idx].keys()):
                h = all_residuals[sent_idx][layer_idx]
                proj = h @ pc_components.T  # [n_pc]
                layer_projs.append(proj)
            if layer_projs:
                template_projections.append(np.array(layer_projs))
        
        if not template_projections:
            results[template_name] = {'error': 'no valid projections'}
            continue
        
        # 平均各句子的投影
        avg_proj = np.mean(template_projections, axis=0)  # [n_layers, n_pc]
        
        results[template_name] = {
            'n_sentences': len(template_projections),
            'n_layers': avg_proj.shape[0],
            'proj_shape': list(avg_proj.shape),
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
            'final_layers_pc': [
                {
                    'layer': l,
                    'pc_distribution': np.abs(avg_proj[l]).tolist()[:10],
                }
                for l in range(max(0, avg_proj.shape[0]-3), avg_proj.shape[0])
            ],
        }
        
        # 释放内存
        del all_residuals
        gc.collect()
        torch.cuda.empty_cache()
    
    # 句式间差异分析
    print("\n  句式间差异分析:")
    
    template_final_projs = {}
    for tname, tdata in results.items():
        if 'error' in tdata:
            continue
        final_layers = tdata.get('final_layers_pc', [])
        if final_layers:
            last_layer_pc = final_layers[-1].get('pc_distribution', [])
            template_final_projs[tname] = last_layer_pc
    
    if len(template_final_projs) > 1:
        tnames = list(template_final_projs.keys())
        pairwise_diff = {}
        for i in range(len(tnames)):
            for j in range(i+1, len(tnames)):
                p1 = np.array(template_final_projs[tnames[i]])
                p2 = np.array(template_final_projs[tnames[j]])
                if len(p1) > 0 and len(p2) > 0:
                    min_len = min(len(p1), len(p2))
                    norm1 = np.linalg.norm(p1[:min_len])
                    norm2 = np.linalg.norm(p2[:min_len])
                    if norm1 > 1e-10 and norm2 > 1e-10:
                        cos_sim = np.dot(p1[:min_len], p2[:min_len]) / (norm1 * norm2)
                        pairwise_diff[f"{tnames[i]}_vs_{tnames[j]}"] = float(cos_sim)
        
        results['pairwise_similarity'] = pairwise_diff
        
        if pairwise_diff:
            sorted_pairs = sorted(pairwise_diff.items(), key=lambda x: x[1])
            print(f"    最不同的句式对: {sorted_pairs[0][0]} (cos={sorted_pairs[0][1]:.4f})")
            print(f"    最相似的句式对: {sorted_pairs[-1][0]} (cos={sorted_pairs[-1][1]:.4f})")
    
    # 分析: 不同句式在不同层的差异
    print("\n  各句式能量分布(前5层/中5层/后5层):")
    for tname, tdata in results.items():
        if 'error' in tdata or 'layer_pc_energy' not in tdata:
            continue
        energies = [d['total_energy'] for d in tdata['layer_pc_energy']]
        n_e = len(energies)
        if n_e >= 15:
            early = np.mean(energies[:5])
            mid = np.mean(energies[n_e//2-2:n_e//2+3])
            late = np.mean(energies[-5:])
            print(f"    {tname}: 早期={early:.2f}, 中期={mid:.2f}, 晚期={late:.2f}")
    
    results['conclusion'] = '句式差异主要在晚期层体现' if len(template_final_projs) > 2 else '数据不足'
    
    print("\n=== P732 完成 ===")
    return results


# ============================================================
# P733: 语法层级在Transformer层中的对应
# ============================================================

def P733_syntax_hierarchy(model, tokenizer, device, model_info, model_name):
    """
    P733: 语法层级在Transformer层中的对应
    - 嵌套深度 → 哪些层编码嵌套?
    - 并列vs主从 → 哪些层区分结构类型?
    """
    print("\n=== P733: 语法层级在Transformer层中的对应 ===")
    
    n_layers = model_info.n_layers
    W_U = to_numpy(get_W_U(model))
    
    n_pc = 20
    pca = PCA(n_components=min(n_pc, W_U.shape[1]))
    pca.fit(W_U)
    pc_components = pca.components_
    
    results = {}
    
    for depth_name, sentences in NESTING_SENTENCES.items():
        print(f"\n  分析嵌套深度: {depth_name}...")
        
        all_residuals = get_residual_streams(model, tokenizer, sentences, device, min(n_layers, 36))
        
        if not all_residuals:
            results[depth_name] = {'error': 'no valid sentences'}
            continue
        
        depth_projections = []
        for sent_idx in range(len(sentences)):
            if sent_idx not in all_residuals:
                continue
            layer_projs = []
            for layer_idx in sorted(all_residuals[sent_idx].keys()):
                h = all_residuals[sent_idx][layer_idx]
                proj = h @ pc_components.T
                layer_projs.append(proj)
            if layer_projs:
                depth_projections.append(np.array(layer_projs))
        
        if not depth_projections:
            results[depth_name] = {'error': 'no valid projections'}
            continue
        
        avg_proj = np.mean(depth_projections, axis=0)  # [n_layers, n_pc]
        
        results[depth_name] = {
            'n_sentences': len(depth_projections),
            'n_layers': avg_proj.shape[0],
            'layer_energy': [
                float(np.sum(avg_proj[l]**2))
                for l in range(avg_proj.shape[0])
            ],
            'syntax_energy_ratio': [
                float(np.sum(avg_proj[l, :5]**2) / (np.sum(avg_proj[l]**2) + 1e-10))
                for l in range(avg_proj.shape[0])
            ],
        }
        
        del all_residuals
        gc.collect()
        torch.cuda.empty_cache()
    
    # 嵌套深度 vs 语法信号
    print("\n  嵌套深度 vs 语法信号:")
    
    depth_energy = {}
    for depth_name, data in results.items():
        if 'error' in data:
            continue
        n_l = data['n_layers']
        mid_start = n_l // 3
        mid_end = 2 * n_l // 3
        mid_syntax = np.mean(data['syntax_energy_ratio'][mid_start:mid_end]) if data['syntax_energy_ratio'] else 0
        depth_energy[depth_name] = mid_syntax
        print(f"    {depth_name}: 中间层语法信号={mid_syntax:.4f}")
    
    # 深度编码假说检验
    depth_order = ['depth_0', 'depth_1', 'depth_2']
    available_depths = [d for d in depth_order if d in depth_energy]
    
    if len(available_depths) >= 2:
        values = [depth_energy[d] for d in available_depths]
        is_monotonic = all(values[i] <= values[i+1] for i in range(len(values)-1))
        conclusion = '嵌套越深, 中间层语法信号越强' if is_monotonic else '嵌套深度与中间层语法信号无单调关系'
    else:
        conclusion = '数据不足'
    
    print(f"  结论: {conclusion}")
    
    results['depth_analysis'] = {
        'mid_layer_syntax_signal': depth_energy,
        'conclusion': conclusion,
    }
    
    # 并列vs主从
    print("\n  并列vs主从的层差异:")
    conj_data = results.get('conjunction', {})
    sub_data = results.get('subordination', {})
    
    if 'layer_energy' in conj_data and 'layer_energy' in sub_data:
        conj_energy = np.array(conj_data['layer_energy'])
        sub_energy = np.array(sub_data['layer_energy'])
        min_len = min(len(conj_energy), len(sub_energy))
        diff = sub_energy[:min_len] - conj_energy[:min_len]
        
        max_diff_layer = int(np.argmax(np.abs(diff)))
        print(f"    差异最大的层: L{max_diff_layer} (diff={diff[max_diff_layer]:.2f})")
        
        # 哪种结构在中间层能量更大?
        mid = min_len // 2
        if sub_energy[mid] > conj_energy[mid]:
            print(f"    主从结构在中间层(L{mid})能量更大: {sub_energy[mid]:.2f} vs {conj_energy[mid]:.2f}")
        else:
            print(f"    并列结构在中间层(L{mid})能量更大: {conj_energy[mid]:.2f} vs {sub_energy[mid]:.2f}")
        
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
    print(f"Phase CLXX v2: 语法关系编码 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    results = {}
    
    # P732
    try:
        results["P732"] = P732_sentence_templates(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P732失败: {e}")
        import traceback
        traceback.print_exc()
        results["P732"] = {"error": str(e)}
    
    # P733
    try:
        results["P733"] = P733_syntax_hierarchy(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P733失败: {e}")
        import traceback
        traceback.print_exc()
        results["P733"] = {"error": str(e)}
    
    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 保存结果
    output_dir = Path(f"results/phase_clxx")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_v2_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
