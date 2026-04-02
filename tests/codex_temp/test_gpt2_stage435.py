#!/usr/bin/env python3
"""
Stage435简化测试 - GPT-2快速验证
目标：验证脚本是否正常工作
"""

import sys
import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformer_lens import HookedTransformer

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# 简化测试词库 - 每个词性5个单词（快速测试）
TEST_WORDS_SIMPLE = {
    "名词": ["苹果", "桌子", "电脑", "手机", "汽车"],
    "动词": ["吃", "喝", "睡", "走", "跑"],
    "形容词": ["大", "小", "好", "坏", "美"],
    "副词": ["很", "非常", "快速", "缓慢", "总是"],
    "代词": ["我", "你", "他", "这", "那"],
    "介词": ["在", "从", "到", "对", "给"]
}

def main():
    print("\n" + "="*60)
    print("Stage435简化测试: GPT-2快速验证")
    print("="*60)
    
    try:
        print("\n加载GPT-2模型...")
        model = HookedTransformer.from_pretrained("gpt2", device='cuda')
        print(f"✓ 模型加载成功!")
        print(f"  - 层数: {model.cfg.n_layers}")
        print(f"  - 隐藏层大小: {model.cfg.d_model}")
        print(f"  - MLP神经元总数: {model.cfg.n_layers * model.cfg.d_mlp}")
        
        num_layers = model.cfg.n_layers
        d_mlp = model.cfg.d_mlp
        
        # 测试一个词性
        pos_tag = "名词"
        words = TEST_WORDS_SIMPLE[pos_tag]
        
        print(f"\n提取神经元: {pos_tag}")
        print(f"单词数: {len(words)}")
        print(f"单词: {words}")
        
        all_activations = []
        for word_idx, word in enumerate(words):
            try:
                tokens = model.to_tokens(word)
                
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens)
                
                # 提取MLP后激活
                neuron_activations = []
                for layer_idx in range(num_layers):
                    act = cache[f"blocks.{layer_idx}.mlp.hook_post"][0, -1, :]
                    neuron_activations.append(act.cpu().numpy())
                
                merged_activations = np.concatenate(neuron_activations)
                all_activations.append(merged_activations)
                
                print(f"  单词 {word_idx+1}: {word} - 完成")
                
                del cache
                clear_gpu_memory()
                
            except Exception as e:
                print(f"  单词 {word_idx+1}: {word} - 失败: {e}")
                all_activations.append(np.zeros(num_layers * d_mlp))
        
        all_activations = np.array(all_activations)
        avg_activations = np.mean(all_activations, axis=0)
        
        # 提取top 10神经元
        top_k = 10
        top_indices = np.argsort(avg_activations)[-top_k:][::-1]
        
        top_neurons = []
        for idx in top_indices:
            layer_idx = idx // d_mlp
            neuron_idx = idx % d_mlp
            top_neurons.append({
                'layer': int(layer_idx),
                'neuron_idx': int(neuron_idx),
                'avg_activation': float(avg_activations[idx])
            })
        
        print(f"\nTop {top_k}神经元:")
        for i, neuron in enumerate(top_neurons):
            print(f"  {i+1}. Layer {neuron['layer']}, Neuron {neuron['neuron_idx']}: {neuron['avg_activation']:.4f}")
        
        # 保存结果
        result = {
            'model_name': 'gpt2',
            'pos_tag': pos_tag,
            'words': words,
            'top_neurons': top_neurons,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        output_file = "tests/codex_temp/test_gpt2_stage435_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 测试完成!")
        print(f"  - 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
