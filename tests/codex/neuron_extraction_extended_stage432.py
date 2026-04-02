"""
Stage432: 扩展词库的神经元提取
每个词性测试30个单词，提高统计显著性
"""
import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import gc
import sys

# 添加TransformerLens路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 输出目录
OUTPUT_DIR = Path("D:/develop/TransformerLens-main/tests/codex_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型配置（使用本地模型路径）
MODELS = {
    "qwen3_4b": {
        "name": "Qwen/Qwen3-4B",
        "local_path": "D:\\develop\\model\\hub\\models--Qwen--Qwen3-4B\\snapshots\\1cfa9a7208912126459214e8b04321603b3df60c",
        "num_layers": 36,
        "hidden_size": 2560,
        "description": "Qwen3-4B模型"
    },
    "deepseek_7b": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "local_path": "D:\\develop\\model\\hub\\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\\snapshots\\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "num_layers": 28,
        "hidden_size": 3584,
        "description": "DeepSeek-7B模型"
    }
}

# 扩展的词性测试词库（每个词性30个单词）
POS_WORDS = {
    "noun": {
        "description": "名词",
        "words": [
            # 水果
            "apple", "banana", "orange", "grape", "mango",
            # 动物
            "dog", "cat", "bird", "fish", "tiger",
            # 自然
            "tree", "flower", "mountain", "river", "ocean",
            # 人工物品
            "car", "house", "book", "computer", "phone",
            # 抽象概念
            "love", "freedom", "justice", "truth", "beauty",
            # 人物
            "teacher", "doctor", "student", "child", "parent"
        ]
    },
    "adjective": {
        "description": "形容词",
        "words": [
            # 外观
            "beautiful", "ugly", "big", "small", "tall",
            # 尺寸
            "short", "long", "wide", "narrow", "thick",
            # 速度
            "fast", "slow", "quick", "rapid", "swift",
            # 温度
            "hot", "cold", "warm", "cool", "freezing",
            # 情感
            "happy", "sad", "angry", "calm", "excited",
            # 颜色
            "red", "blue", "green", "yellow", "purple"
        ]
    },
    "verb": {
        "description": "动词",
        "words": [
            # 移动
            "run", "walk", "jump", "swim", "fly",
            # 基本动作
            "eat", "drink", "sleep", "think", "speak",
            # 感知
            "write", "read", "listen", "watch", "feel",
            # 创造
            "make", "build", "create", "destroy", "change",
            # 交流
            "say", "tell", "ask", "answer", "explain",
            # 思维
            "understand", "believe", "know", "remember", "forget"
        ]
    },
    "adverb": {
        "description": "副词",
        "words": [
            # 速度
            "quickly", "slowly", "rapidly", "gradually", "suddenly",
            # 方式
            "carefully", "carelessly", "happily", "sadly", "angrily",
            # 频率
            "always", "never", "often", "rarely", "sometimes",
            # 程度
            "very", "quite", "extremely", "fairly", "rather",
            # 时间
            "now", "then", "today", "yesterday", "tomorrow",
            # 地点
            "here", "there", "everywhere", "nowhere", "somewhere"
        ]
    },
    "pronoun": {
        "description": "代词",
        "words": [
            # 主格
            "I", "you", "he", "she", "it",
            # 复数主格
            "we", "they",
            # 宾格
            "me", "him", "her", "us", "them",
            # 指示代词
            "this", "that", "these", "those",
            # 疑问代词
            "who", "which", "what", "whose", "whom",
            # 不定代词
            "someone", "anyone", "everyone", "nobody", "something"
        ]
    },
    "preposition": {
        "description": "介词",
        "words": [
            # 基本介词
            "in", "on", "at", "to", "from",
            # 伴随
            "with", "by", "for", "of", "about",
            # 方向
            "into", "through", "over", "under", "between",
            # 位置
            "among", "behind", "beside", "above", "below",
            # 时间
            "during", "before", "after", "since", "until",
            # 其他
            "across", "along", "around", "toward", "against"
        ]
    }
}

def clear_gpu_memory():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU内存已清理")

def load_model_single(model_key):
    """单模型加载"""
    print(f"\n{'='*80}")
    print(f"加载模型: {model_key}")
    print(f"{'='*80}")
    
    clear_gpu_memory()
    
    model_config = MODELS[model_key]
    print(f"模型名称: {model_config['name']}")
    print(f"描述: {model_config['description']}")
    print(f"层数: {model_config['num_layers']}")
    print(f"隐藏层大小: {model_config['hidden_size']}")
    
    # 设置离线环境
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"\n开始加载模型...")
        
        # 优先使用本地路径
        if model_config.get('local_path'):
            model_path = model_config['local_path']
            print(f"使用本地路径: {model_path}")
        else:
            model_path = model_config['name']
            print(f"使用模型名称: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        
        print(f"[OK] 模型加载成功")
        
        return model, tokenizer, model_config
        
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        return None, None, None

def extract_neuron_activations(model, tokenizer, model_config, model_key):
    """提取神经元激活"""
    print(f"\n{'='*80}")
    print(f"提取神经元激活: {model_key}")
    print(f"{'='*80}")
    
    results = {}
    
    for pos, pos_data in POS_WORDS.items():
        print(f"\n处理词性: {pos} ({pos_data['description']}) - {len(pos_data['words'])}个单词")
        
        word_activations = {}
        
        for word_idx, word in enumerate(pos_data['words'], 1):
            print(f"  [{word_idx}/{len(pos_data['words'])}] 测试单词: {word}")
            
            try:
                # Tokenize
                inputs = tokenizer(word, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Forward pass with hidden states
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                # 提取隐藏状态
                hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_size)
                
                # 计算每层的激活强度（跳过embedding层，从第1个transformer层开始）
                layer_activations = []
                for layer_idx, hidden_state in enumerate(hidden_states[1:], start=0):  # 从第1个transformer层开始，索引从0开始
                    # 取最后一个token的激活
                    last_token_hidden = hidden_state[0, -1, :].cpu().numpy()
                    
                    # 计算激活强度（L2范数，使用float64避免溢出）
                    last_token_hidden_f64 = last_token_hidden.astype(np.float64)
                    activation_norm = np.linalg.norm(last_token_hidden_f64)
                    
                    # 找到激活最强的前10个神经元
                    top_indices = np.argsort(np.abs(last_token_hidden))[-10:][::-1]
                    top_values = last_token_hidden[top_indices]
                    
                    layer_activations.append({
                        "layer_idx": layer_idx,
                        "activation_norm": float(activation_norm),
                        "top_neurons": [int(x) for x in top_indices],
                        "top_values": [float(x) for x in top_values]
                    })
                
                word_activations[word] = {
                    "layer_activations": layer_activations
                }
                
                # 清理内存
                del outputs, hidden_states
                clear_gpu_memory()
                
            except Exception as e:
                print(f"    [WARN] 单词处理失败: {e}")
                word_activations[word] = {"error": str(e)}
        
        # 计算该词性的关键神经元
        key_neurons = identify_key_neurons(word_activations, model_config['num_layers'])
        
        results[pos] = {
            "description": pos_data['description'],
            "word_count": len(word_activations),
            "word_activations": word_activations,
            "key_neurons": key_neurons
        }
    
    return results

def identify_key_neurons(word_activations, num_layers):
    """识别关键神经元"""
    print(f"  识别关键神经元...")
    
    # 统计每层每神经元的激活频率
    neuron_activation_count = {}
    
    for word, word_data in word_activations.items():
        if "error" in word_data:
            continue
        
        for layer_data in word_data["layer_activations"]:
            layer_idx = layer_data["layer_idx"]
            top_neurons = layer_data["top_neurons"]
            
            for neuron_idx in top_neurons:
                key = (layer_idx, neuron_idx)
                if key not in neuron_activation_count:
                    neuron_activation_count[key] = 0
                neuron_activation_count[key] += 1
    
    # 按激活频率排序
    sorted_neurons = sorted(
        neuron_activation_count.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # 取前100个关键神经元
    top_100_neurons = [
        {
            "layer": int(layer_idx),
            "neuron": int(neuron_idx),
            "activation_count": int(count),
            "normalized_layer": float(layer_idx / (num_layers - 1))
        }
        for (layer_idx, neuron_idx), count in sorted_neurons[:100]
    ]
    
    # 统计层分布
    layer_distribution = {}
    for layer_idx in range(num_layers):
        layer_distribution[layer_idx] = 0
    
    for (layer_idx, _), count in sorted_neurons[:100]:
        layer_distribution[layer_idx] += count
    
    # 计算质心层
    total_count = sum(layer_distribution.values())
    if total_count > 0:
        weighted_center = sum(
            layer_idx * count for layer_idx, count in layer_distribution.items()
        ) / total_count
    else:
        weighted_center = 0
    
    return {
        "top_100_neurons": top_100_neurons,
        "layer_distribution": layer_distribution,
        "weighted_center": float(weighted_center),
        "total_activation_count": int(total_count)
    }

def save_results(model_key, results, model_config):
    """保存结果"""
    print(f"\n保存结果: {model_key}")
    
    output_data = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": f"neuron_extraction_extended_{model_key}_stage432",
        "timestamp": datetime.now().isoformat(),
        "model_key": model_key,
        "model_config": model_config,
        "word_count_per_pos": 30,  # 每个词性30个单词
        "results": results
    }
    
    output_file = OUTPUT_DIR / f"neuron_extraction_extended_{model_key}_stage432.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] 结果已保存: {output_file}")
    
    return output_file

def test_single_model(model_key):
    """测试单个模型"""
    print(f"\n{'='*80}")
    print(f"开始测试模型: {model_key}")
    print(f"{'='*80}")
    
    # 加载模型
    model, tokenizer, model_config = load_model_single(model_key)
    
    if model is None:
        print(f"[ERROR] 跳过模型: {model_key}")
        return None
    
    # 提取神经元激活
    results = extract_neuron_activations(model, tokenizer, model_config, model_key)
    
    # 保存结果
    output_file = save_results(model_key, results, model_config)
    
    # 卸载模型
    del model
    del tokenizer
    clear_gpu_memory()
    
    print(f"\n[OK] 完成模型测试: {model_key}")
    
    return output_file

def main():
    """主函数"""
    print("="*80)
    print("Stage432: 扩展词库的神经元提取")
    print("="*80)
    print("\n提示: 每个词性测试30个单词，提高统计显著性")
    print("\n可用模型:")
    for key, config in MODELS.items():
        print(f"  - {key}: {config['description']}")
    
    # 测试模型列表（按顺序测试，每次只加载一个模型）
    models_to_test = ["qwen3_4b", "deepseek_7b"]  # 测试Qwen3-4B和DeepSeek-7B
    
    output_files = {}
    
    for model_key in models_to_test:
        print(f"\n\n{'#'*80}")
        print(f"# 测试模型: {model_key}")
        print(f"{'#'*80}")
        
        try:
            output_file = test_single_model(model_key)
            if output_file:
                output_files[model_key] = str(output_file)
        except Exception as e:
            print(f"[ERROR] 模型测试失败 {model_key}: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存汇总结果
    summary_file = OUTPUT_DIR / "neuron_extraction_extended_summary_stage432.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "schema_version": "agi_research_result.v1",
            "timestamp": datetime.now().isoformat(),
            "word_count_per_pos": 30,
            "models_tested": list(output_files.keys()),
            "output_files": output_files
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n{'='*80}")
    print("Stage432完成!")
    print(f"{'='*80}")
    print(f"\n测试完成的模型: {list(output_files.keys())}")
    print(f"汇总文件: {summary_file}")
    
    return output_files

if __name__ == "__main__":
    main()
