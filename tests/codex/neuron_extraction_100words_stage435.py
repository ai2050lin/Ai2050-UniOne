"""
Stage435: 扩展神经元提取 - 每个词性100个单词
目标：提供更强的统计显著性，进行更深入的神经元级别分析

测试词量：每个词性100个单词（共600个单词）
分析维度：
1. 神经元激活频率分布
2. 神经元特异性分析
3. 神经元激活强度分布
4. 神经元共激活模式
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

# 模型配置（使用官方支持的模型）
MODELS = {
    "gpt2": {
        "name": "gpt2",
        "local_path": None,
        "num_layers": 12,
        "hidden_size": 768,
        "description": "GPT-2模型（小型，用于快速测试）"
    },
    "gpt2_medium": {
        "name": "gpt2-medium",
        "local_path": None,
        "num_layers": 24,
        "hidden_size": 1024,
        "description": "GPT-2-Medium模型"
    }
}

# 扩展的词性测试词库（每个词性100个单词）
POS_WORDS = {
    "noun": {
        "description": "名词",
        "words": [
            # 水果
            "apple", "banana", "orange", "grape", "mango",
            "pear", "peach", "cherry", "strawberry", "blueberry",
            "watermelon", "melon", "pineapple", "coconut", "kiwi",
            # 动物
            "dog", "cat", "bird", "fish", "tiger",
            "lion", "elephant", "giraffe", "zebra", "monkey",
            "rabbit", "horse", "cow", "pig", "sheep",
            # 自然
            "tree", "flower", "mountain", "river", "ocean",
            "lake", "forest", "desert", "sky", "sun",
            "moon", "star", "cloud", "rain", "snow",
            # 人工物品
            "car", "house", "book", "computer", "phone",
            "table", "chair", "bed", "sofa", "desk",
            "lamp", "television", "refrigerator", "stove", "oven",
            # 抽象概念
            "love", "freedom", "justice", "truth", "beauty",
            "peace", "hope", "dream", "knowledge", "wisdom",
            "happiness", "success", "failure", "time", "space",
            # 人物
            "teacher", "doctor", "student", "child", "parent",
            "friend", "enemy", "stranger", "neighbor", "colleague",
            "brother", "sister", "husband", "wife", "grandparent",
            # 职业
            "engineer", "scientist", "artist", "musician", "writer",
            "lawyer", "soldier", "police", "firefighter", "pilot",
            # 食物
            "bread", "rice", "pasta", "pizza", "burger",
            "salad", "soup", "sandwich", "cake", "cookie",
            # 城市与地点
            "city", "village", "country", "island", "continent",
            "park", "garden", "beach", "mountain", "valley"
        ]
    },
    "adjective": {
        "description": "形容词",
        "words": [
            # 外观
            "beautiful", "ugly", "pretty", "attractive", "handsome",
            "plain", "gorgeous", "lovely", "cute", "elegant",
            # 尺寸
            "big", "small", "large", "tiny", "huge",
            "tall", "short", "long", "wide", "narrow",
            "thick", "thin", "deep", "shallow", "high",
            # 速度
            "fast", "slow", "quick", "rapid", "swift",
            "speedy", "gradual", "steady", "constant", "sudden",
            # 温度
            "hot", "cold", "warm", "cool", "freezing",
            "boiling", "chilly", "freezing", "mild", "moderate",
            # 情感
            "happy", "sad", "angry", "calm", "excited",
            "worried", "fearful", "brave", "scared", "nervous",
            "proud", "ashamed", "confident", "shy", "bold",
            # 颜色
            "red", "blue", "green", "yellow", "purple",
            "orange", "pink", "brown", "black", "white",
            # 质量
            "good", "bad", "excellent", "poor", "great",
            "terrible", "wonderful", "awful", "perfect", "flawed",
            "important", "useless", "valuable", "worthless", "precious",
            # 其他
            "new", "old", "young", "modern", "ancient",
            "fresh", "stale", "clean", "dirty", "pure",
            "simple", "complex", "easy", "difficult", "hard",
            "soft", "hard", "rough", "smooth", "sharp",
            "dull", "bright", "dark", "light", "heavy",
            "light", "strong", "weak", "powerful", "helpless"
        ]
    },
    "verb": {
        "description": "动词",
        "words": [
            # 移动
            "run", "walk", "jump", "swim", "fly",
            "climb", "crawl", "hop", "dance", "march",
            # 基本动作
            "eat", "drink", "sleep", "think", "speak",
            "listen", "watch", "smell", "taste", "touch",
            # 感知
            "write", "read", "listen", "watch", "feel",
            "see", "hear", "notice", "observe", "perceive",
            # 创造
            "make", "build", "create", "destroy", "change",
            "invent", "design", "produce", "manufacture", "construct",
            # 交流
            "say", "tell", "ask", "answer", "explain",
            "describe", "discuss", "argue", "agree", "disagree",
            # 思维
            "understand", "believe", "know", "remember", "forget",
            "learn", "teach", "study", "practice", "improve",
            # 工作
            "work", "play", "rest", "relax", "exercise",
            "train", "practice", "compete", "win", "lose",
            # 情感
            "love", "like", "hate", "dislike", "enjoy",
            "appreciate", "prefer", "want", "need", "desire",
            # 其他
            "start", "stop", "begin", "end", "finish",
            "continue", "pause", "resume", "complete", "accomplish",
            "give", "take", "receive", "offer", "accept",
            "buy", "sell", "pay", "cost", "spend",
            "help", "support", "assist", "serve", "protect",
            "attack", "defend", "fight", "struggle", "surrender"
        ]
    },
    "adverb": {
        "description": "副词",
        "words": [
            # 速度
            "quickly", "slowly", "rapidly", "gradually", "suddenly",
            "swiftly", "hastily", "steadily", "constantly", "instantly",
            # 方式
            "carefully", "carelessly", "happily", "sadly", "angrily",
            "calmly", "loudly", "quietly", "softly", "roughly",
            # 频率
            "always", "never", "often", "rarely", "sometimes",
            "usually", "frequently", "occasionally", "regularly", "constantly",
            # 程度
            "very", "extremely", "quite", "rather", "fairly",
            "somewhat", "highly", "deeply", "fully", "completely",
            # 时间
            "yesterday", "today", "tomorrow", "now", "soon",
            "later", "earlier", "before", "after", "already",
            # 地点
            "here", "there", "everywhere", "nowhere", "somewhere",
            "outside", "inside", "nearby", "far", "close",
            # 肯定/否定
            "yes", "no", "maybe", "perhaps", "possibly",
            "certainly", "definitely", "absolutely", "surely", "really",
            # 其他
            "easily", "hardly", "simply", "clearly", "obviously",
            "finally", "initially", "firstly", "secondly", "lastly"
        ]
    },
    "pronoun": {
        "description": "代词",
        "words": [
            "I", "you", "he", "she", "it",
            "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his",
            "her", "its", "our", "their", "mine",
            "yours", "hers", "ours", "theirs", "this",
            "that", "these", "those", "who", "whom",
            "whose", "which", "what", "where", "when",
            "why", "how", "myself", "yourself", "himself",
            "herself", "itself", "ourselves", "themselves", "anyone",
            "everyone", "someone", "no one", "anybody", "everybody",
            "somebody", "nobody", "anything", "everything", "something",
            "nothing", "any", "each", "every", "either",
            "neither", "both", "few", "many", "several",
            "some", "all", "none", "most", "such",
            "same", "other", "another", "one", "two"
        ]
    },
    "preposition": {
        "description": "介词",
        "words": [
            "in", "on", "at", "to", "from",
            "with", "by", "for", "of", "about",
            "between", "among", "under", "over", "above",
            "below", "through", "across", "around", "behind",
            "before", "after", "during", "since", "until",
            "against", "along", "beside", "beyond", "despite",
            "except", "inside", "into", "near", "outside",
            "past", "throughout", "toward", "underneath", "upon",
            "within", "without", "according", "regarding", "concerning",
            "following", "including", "unlike", "via", "versus"
        ]
    }
}

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def extract_neuron_features(model, words, pos_tag, word_count_per_pos=100):
    """
    提取神经元的多种特征
    
    返回: 包含单词激活信息的字典
    """
    print(f"\n处理词性: {pos_tag}")
    print(f"单词数: {len(words)}")
    
    num_layers = model.cfg.n_layers
    d_mlp = model.cfg.d_mlp
    
    word_activations = {}
    
    for word_idx, word in enumerate(words):
        if (word_idx + 1) % 10 == 0:
            print(f"  进度: {word_idx+1}/{len(words)}")
        
        try:
            # Tokenize
            tokens = model.to_tokens(word)
            
            # Forward pass
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            
            # 提取每一层的激活和top神经元
            layer_activations = []
            for layer_idx in range(num_layers):
                act = cache[f"blocks.{layer_idx}.mlp.hook_post"][0, -1, :].cpu().numpy()
                activation_norm = float(np.linalg.norm(act))
                
                # 提取top 10神经元
                top_indices = np.argsort(np.abs(act))[-10:][::-1]
                top_values = [float(act[idx]) for idx in top_indices]
                
                layer_activations.append({
                    "layer_idx": layer_idx,
                    "activation_norm": activation_norm,
                    "top_neurons": [int(idx) for idx in top_indices],
                    "top_values": top_values
                })
            
            word_activations[word] = {
                "layer_activations": layer_activations
            }
            
            # 清理
            del cache
            clear_gpu_memory()
            
        except Exception as e:
            print(f"  处理单词 '{word}' 时出错: {e}")
            word_activations[word] = None
    
    return word_activations

def analyze_word_activations(word_activations, num_layers, d_mlp):
    """
    分析单词激活数据，提取神经元级别的特征
    """
    # 统计每个神经元在多少个单词中被激活（在top 10中）
    neuron_activation_count = np.zeros((num_layers, d_mlp))
    neuron_activation_values = []
    
    for word, activations in word_activations.items():
        if activations is None:
            continue
        
        for layer_act in activations["layer_activations"]:
            layer_idx = layer_act["layer_idx"]
            for neuron_idx in layer_act["top_neurons"]:
                neuron_activation_count[layer_idx, neuron_idx] += 1
                neuron_activation_values.append({
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "word": word,
                    "value": layer_act["top_values"][layer_act["top_neurons"].index(neuron_idx)]
                })
    
    # 提取最活跃的神经元（激活次数最多）
    total_words = len([w for w, a in word_activations.items() if a is not None])
    
    # 扁平化并排序
    flat_indices = np.argsort(neuron_activation_count.flatten())[-100:][::-1]
    
    top_neurons = []
    for flat_idx in flat_indices:
        layer_idx = flat_idx // d_mlp
        neuron_idx = flat_idx % d_mlp
        count = int(neuron_activation_count[layer_idx, neuron_idx])
        frequency = count / total_words
        
        top_neurons.append({
            "layer": int(layer_idx),
            "neuron_idx": int(neuron_idx),
            "activation_count": count,
            "activation_frequency": frequency
        })
    
    return {
        "total_words": total_words,
        "top_100_neurons": top_neurons,
        "neuron_activation_values": neuron_activation_values[:1000]  # 限制大小
    }

def main():
    print("\n" + "="*60)
    print("Stage435: 扩展神经元提取 - 每个词性100个单词")
    print("="*60)
    
    # 测试模型（只测试GPT-2）
    models_to_test = ["gpt2"]
    
    for model_key in models_to_test:
        model_config = MODELS[model_key]
        
        print(f"\n{'='*60}")
        print(f"加载模型: {model_key}")
        print(f"{'='*60}")
        print(f"模型名称: {model_config['name']}")
        print(f"本地路径: {model_config['local_path']}")
        print(f"层数: {model_config['num_layers']}")
        print(f"隐藏层大小: {model_config['hidden_size']}")
        
        # 加载模型
        from transformer_lens import HookedTransformer
        
        # 从官方模型加载
        model = HookedTransformer.from_pretrained(
            model_config['name'],
            device='cuda',
            dtype=torch.float16,
            default_padding_side='right'
        )
        
        num_layers = model.cfg.n_layers
        d_mlp = model.cfg.d_mlp
        
        print(f"\n模型加载成功!")
        print(f"实际层数: {num_layers}")
        print(f"实际MLP大小: {d_mlp}")
        print(f"总神经元数: {num_layers * d_mlp}")
        
        # 存储结果
        results = {
            "model_key": model_key,
            "model_config": model_config,
            "word_count_per_pos": 100,
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # 处理每个词性
        for pos_tag, pos_data in POS_WORDS.items():
            print(f"\n{'='*60}")
            print(f"处理词性: {pos_tag} ({pos_data['description']})")
            print(f"{'='*60}")
            
            words = pos_data['words'][:100]  # 只取前100个
            print(f"单词数: {len(words)}")
            print(f"示例单词: {words[:5]}")
            
            try:
                # 提取神经元特征
                word_activations = extract_neuron_features(model, words, pos_tag)
                
                # 分析激活数据
                analysis = analyze_word_activations(word_activations, num_layers, d_mlp)
                
                results["results"][pos_tag] = {
                    "description": pos_data['description'],
                    "word_count": len(words),
                    "word_activations": word_activations,
                    "analysis": analysis
                }
                
                print(f"\n完成!")
                print(f"  - 处理单词数: {analysis['total_words']}")
                if analysis['top_100_neurons']:
                    print(f"  - 最活跃神经元: Layer {analysis['top_100_neurons'][0]['layer']}, Neuron {analysis['top_100_neurons'][0]['neuron_idx']}")
                    print(f"  - 激活频率: {analysis['top_100_neurons'][0]['activation_frequency']:.2%}")
                
            except Exception as e:
                print(f"\n处理失败: {e}")
                import traceback
                traceback.print_exc()
                results["results"][pos_tag] = None
        
        # 保存结果
        output_file = OUTPUT_DIR / f"neuron_extraction_100words_{model_key}_stage435.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"结果已保存: {output_file}")
        print(f"{'='*60}")
        
        # 清理模型
        del model
        clear_gpu_memory()
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)

if __name__ == "__main__":
    main()
