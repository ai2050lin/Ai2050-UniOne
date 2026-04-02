# -*- coding: utf-8 -*-
"""
Stage448: DeepSeek-7B跨模型验证实验
验证Qwen3-4B上发现的编码机制是否在DeepSeek-7B上成立
"""
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import os

print("=" * 60)
print("Stage448: DeepSeek-7B Cross-Model Validation")
print("=" * 60)

# ============================================================
# 配置
# ============================================================
print("\n" + "=" * 60)
print("Configuration")
print("=" * 60)

# 词库（英文名词、动词、形容词等）
WORD_SETS = {
    "noun": [
        "apple", "river", "mountain", "ocean", "forest", "city", "village", "house", "car", "tree",
        "book", "table", "chair", "phone", "computer", "flower", "bird", "fish", "dog", "cat",
        "sun", "moon", "star", "cloud", "rain", "snow", "wind", "fire", "earth", "sky"
    ],
    "verb": [
        "run", "walk", "eat", "drink", "sleep", "think", "speak", "write", "read", "play",
        "work", "build", "create", "destroy", "find", "lose", "give", "take", "see", "hear",
        "love", "hate", "fear", "hope", "wish", "start", "stop", "move", "turn", "open"
    ],
    "adjective": [
        "happy", "sad", "big", "small", "tall", "short", "young", "old", "new", "ancient",
        "beautiful", "ugly", "fast", "slow", "strong", "weak", "rich", "poor", "hot", "cold",
        "bright", "dark", "loud", "quiet", "clean", "dirty", "full", "empty", "safe", "dangerous"
    ],
    "adverb": [
        "quickly", "slowly", "happily", "sadly", "well", "badly", "very", "quite", "almost", "nearly",
        "always", "never", "often", "sometimes", "rarely", "here", "there", "now", "then", "always",
        "together", "alone", "away", "back", "forth", "up", "down", "in", "out", "about"
    ],
    "pronoun": [
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "her", "its", "our", "their", "mine",
        "yours", "hers", "ours", "theirs", "this", "that", "these", "those", "who", "what"
    ],
    "preposition": [
        "in", "on", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "over", "under", "again", "further", "then", "once", "here", "there", "when", "where"
    ]
}

# 模型配置
MODELS = {
    "deepseek_7b": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "local_path": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "num_layers": 28,
        "hidden_size": 3584,
        "description": "DeepSeek-7B (Qwen蒸馏版)"
    }
}

print(f"Models to test: {list(MODELS.keys())}")
print(f"Words per POS: {len(WORD_SETS['noun'])}")

# ============================================================
# 加载模型
# ============================================================
print("\n" + "=" * 60)
print("Loading Model")
print("=" * 60)

import torch
from transformer_lens import HookedTransformer

def load_model(model_key, model_info):
    """加载模型"""
    print(f"\nLoading {model_key}...")
    print(f"  Name: {model_info['name']}")
    print(f"  Local: {model_info['local_path']}")

    try:
        # 检查本地路径是否存在
        if os.path.exists(model_info['local_path']):
            print(f"  Using local path...")
            model = HookedTransformer.from_pretrained(
                model_info['local_path'],
                device='cuda',
                dtype=torch.float16,
                default_padding_side='right',
                trust_remote_code=True
            )
        else:
            print(f"  Using HuggingFace (downloading if needed)...")
            model = HookedTransformer.from_pretrained(
                model_info['name'],
                device='cuda',
                dtype=torch.float16,
                default_padding_side='right',
                trust_remote_code=True
            )

        print(f"  [OK] Model loaded successfully!")
        print(f"  Layers: {model.cfg.n_layers}")
        print(f"  Hidden size: {model.cfg.d_model}")
        return model

    except Exception as e:
        print(f"  [FAIL] Error loading model: {e}")
        return None

# 加载模型
model = load_model("deepseek_7b", MODELS["deepseek_7b"])

if model is None:
    print("\n[ERROR] Model loading failed. Cannot proceed.")
    print("Will create analysis based on simulated/similar model data instead.")
    model = None
else:
    print(f"\nModel config: {model.cfg.n_layers} layers, {model.cfg.d_model} hidden")

# ============================================================
# 神经元提取函数
# ============================================================
def extract_neurons(model, word, pos_tag, top_k=10):
    """提取单词激活的神经元"""
    try:
        # Tokenize
        tokens = model.to_tokens(word, prepend_bos=False)
        tokens = tokens.to(model.device)

        # 前向传播
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

        # 提取每层的激活神经元
        layer_activations = []
        for layer_idx in range(model.cfg.n_layers):
            # 获取该层的激活
            layer_name = f"blocks.{layer_idx}.hook_mlp_out"
            if layer_name in cache:
                activation = cache[layer_name]
            else:
                # 尝试其他可能的名称
                layer_name = f"blocks.{layer_idx}.hook_mlp_post"
                if layer_name in cache:
                    activation = cache[layer_name]
                else:
                    continue

            # 计算激活范数
            activation_norm = activation.norm().item()

            # 获取top神经元索引
            flat_activation = activation.flatten()
            if len(flat_activation) > top_k:
                top_values, top_indices = torch.topk(flat_activation, top_k)
                top_neurons = top_indices.tolist()
                top_vals = top_values.tolist()
            else:
                top_neurons = list(range(len(flat_activation)))
                top_vals = flat_activation.tolist()

            layer_activations.append({
                "layer_idx": layer_idx,
                "activation_norm": float(activation_norm),
                "top_neurons": top_neurons,
                "top_values": top_vals
            })

        return {
            "word": word,
            "pos": pos_tag,
            "layer_activations": layer_activations,
            "success": True
        }

    except Exception as e:
        return {
            "word": word,
            "pos": pos_tag,
            "error": str(e),
            "success": False
        }

# ============================================================
# 提取神经元激活（如果模型加载成功）
# ============================================================
if model is not None:
    print("\n" + "=" * 60)
    print("Extracting Neuron Activations")
    print("=" * 60)

    results = {}
    total_words = sum(len(words) for words in WORD_SETS.values())
    processed = 0

    for pos_tag, words in WORD_SETS.items():
        print(f"\nProcessing {pos_tag}...")
        pos_results = {}
        for i, word in enumerate(words):
            result = extract_neurons(model, word, pos_tag)
            pos_results[word] = result
            processed += 1
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(words)}")
        results[pos_tag] = {
            "description": pos_tag,
            "word_count": len(words),
            "word_activations": pos_results
        }

    print(f"\n[OK] Extracted activations for {processed} words")

    # 保存结果
    output_file = r"d:\develop\TransformerLens-main\tests\codex_temp\deepseek7b_neuron_activation_stage448.json"

    output_data = {
        "schema_version": "agi_research_result.v1",
        "experiment_id": "deepseek7b_neuron_activation_stage448",
        "timestamp": datetime.now().isoformat(),
        "model_key": "deepseek_7b",
        "model_config": {
            "name": MODELS["deepseek_7b"]["name"],
            "num_layers": MODELS["deepseek_7b"]["num_layers"],
            "hidden_size": MODELS["deepseek_7b"]["hidden_size"],
        },
        "word_count_per_pos": len(WORD_SETS["noun"]),
        "results": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Results saved to: {output_file}")

else:
    print("\n[INFO] Model not loaded. Will use fallback analysis.")

# ============================================================
# 备用分析：如果模型未加载，使用理论模型进行对比分析
# ============================================================
print("\n" + "=" * 60)
print("Running Comparative Analysis")
print("=" * 60)

# 加载Qwen3-4B的结果作为参考
print("\nLoading Qwen3-4B results as reference...")
try:
    with open(r"d:\develop\TransformerLens-main\tests\codex_temp\neuron_extraction_extended_qwen3_4b_stage432.json", "r", encoding="utf-8") as f:
        qwen_results = json.load(f)
    print("[OK] Qwen3-4B data loaded")
except Exception as e:
    print(f"[FAIL] Error loading Qwen3-4B data: {e}")
    qwen_results = None

# 加载DeepSeek数据（如果存在）
print("\nLoading DeepSeek-7B data...")
try:
    with open(r"d:\develop\TransformerLens-main\tests\codex_temp\deepseek7b_neuron_activation_stage448.json", "r", encoding="utf-8") as f:
        deepseek_results = json.load(f)
    print("[OK] DeepSeek-7B data loaded")
except Exception as e:
    print(f"[FAIL] DeepSeek-7B data not found: {e}")
    deepseek_results = None

# ============================================================
# 对比分析
# ============================================================
print("\n" + "=" * 60)
print("Cross-Model Comparison Analysis")
print("=" * 60)

if qwen_results is not None:
    print("\n--- Qwen3-4B Baseline ---")
    print(f"Model: Qwen/Qwen3-4B")
    print(f"Layers: {qwen_results['model_config']['num_layers']}")
    print(f"Hidden size: {qwen_results['model_config']['hidden_size']}")
    print(f"Words per POS: {qwen_results['word_count_per_pos']}")

    # 分析Qwen3-4B的激活模式
    qwen_neurons = defaultdict(int)
    qwen_layer_activation = defaultdict(lambda: defaultdict(int))

    for pos_tag, pos_data in qwen_results['results'].items():
        if 'word_activations' in pos_data:
            for word, word_data in pos_data['word_activations'].items():
                if 'layer_activations' in word_data:
                    for layer_act in word_data['layer_activations']:
                        layer_idx = layer_act['layer_idx']
                        if 'top_neurons' in layer_act:
                            neuron_count = len(layer_act['top_neurons'])
                            qwen_neurons[pos_tag] += neuron_count
                            qwen_layer_activation[layer_idx][pos_tag] += neuron_count

    print(f"\nQwen3-4B neuron activations by POS:")
    for pos, count in sorted(qwen_neurons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pos}: {count}")

if deepseek_results is not None:
    print("\n--- DeepSeek-7B Results ---")
    print(f"Model: {deepseek_results['model_key']}")
    print(f"Layers: {deepseek_results['model_config']['num_layers']}")
    print(f"Hidden size: {deepseek_results['model_config']['hidden_size']}")

    # 分析DeepSeek的激活模式
    ds_neurons = defaultdict(int)
    ds_layer_activation = defaultdict(lambda: defaultdict(int))

    for pos_tag, pos_data in deepseek_results['results'].items():
        if 'word_activations' in pos_data:
            for word, word_data in pos_data['word_activations'].items():
                if 'layer_activations' in word_data:
                    for layer_act in word_data['layer_activations']:
                        layer_idx = layer_act['layer_idx']
                        if 'top_neurons' in layer_act:
                            neuron_count = len(layer_act['top_neurons'])
                            ds_neurons[pos_tag] += neuron_count
                            ds_layer_activation[layer_idx][pos_tag] += neuron_count

    print(f"\nDeepSeek-7B neuron activations by POS:")
    for pos, count in sorted(ds_neurons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pos}: {count}")

# ============================================================
# 理论预测对比
# ============================================================
print("\n" + "=" * 60)
print("Theoretical Predictions vs Observations")
print("=" * 60)

print("""
ENCODING MECHANISMS TO VALIDATE:

1. DISTRIBUTED REPRESENTATION
   Prediction: Most neurons have MID specificity
   Status: [TO BE VALIDATED BY DEEPSEEK]

2. POS-SEPARATED COMPONENTS (NMF)
   Prediction: 6 components, one per POS
   Status: [TO BE VALIDATED BY DEEPSEEK]

3. HUB INTEGRATION
   Prediction: Hub neurons are 100% multifunctional
   Status: [TO BE VALIDATED BY DEEPSEEK]

4. LAYER CONTINUITY
   Prediction: Adjacent layers correlated (r>0.9)
   Status: [TO BE VALIDATED BY DEEPSEEK]

5. FUNCTIONAL MODULES (k=15-20)
   Prediction: Multiple specialized modules
   Status: [TO BE VALIDATED BY DEEPSEEK]

6. UNIVERSAL PATHWAY (PC1)
   Prediction: PC1 explains 80%+ variance
   Status: [TO BE VALIDATED BY DEEPSEEK]
""")

# ============================================================
# 保存对比结果
# ============================================================
print("\n" + "=" * 60)
print("Saving Comparison Results")
print("=" * 60)

comparison_results = {
    "experiment_id": "deepseek_cross_model_validation_stage448",
    "timestamp": datetime.now().isoformat(),
    "models_compared": {
        "qwen3_4b": {
            "status": "reference" if qwen_results else "not_loaded",
            "layers": qwen_results['model_config']['num_layers'] if qwen_results else None,
            "hidden_size": qwen_results['model_config']['hidden_size'] if qwen_results else None
        },
        "deepseek_7b": {
            "status": "loaded" if deepseek_results else "not_loaded",
            "layers": deepseek_results['model_config']['num_layers'] if deepseek_results else None,
            "hidden_size": deepseek_results['model_config']['hidden_size'] if deepseek_results else None
        }
    },
    "predicted_mechanisms": [
        "distributed_representation",
        "pos_separated_components",
        "hub_integration",
        "layer_continuity",
        "functional_modules",
        "universal_pathway"
    ],
    "validation_status": "pending_deepseek_analysis"
}

output_file = r"d:\develop\TransformerLens-main\tests\codex_temp\cross_model_validation_stage448.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(comparison_results, f, indent=2, ensure_ascii=False)

print(f"[OK] Comparison results saved to: {output_file}")

print("\n" + "=" * 60)
print("Stage448 Status")
print("=" * 60)

if deepseek_results is not None:
    print("\n[OK] DeepSeek-7B data extracted successfully!")
    print("Next step: Run NMF/PCA analysis on DeepSeek-7B data")
    print("          to validate encoding mechanisms.")
else:
    print("\n[WARNING] DeepSeek-7B model could not be loaded.")
    print("Possible reasons:")
    print("  1. Network connection issue (cannot download model)")
    print("  2. Model not in TransformerLens supported list")
    print("  3. CUDA memory issue")
    print("\nPlease check:")
    print("  1. Network connection")
    print("  2. Try loading model with different settings")
    print("  3. Use GPT-2 as fallback validation")

print("\n" + "=" * 60)
print("Stage448 Completed")
print("=" * 60)
