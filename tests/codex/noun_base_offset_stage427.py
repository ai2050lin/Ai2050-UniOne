"""
Stage427: 名词基地加偏置特征分析
目标：分析名词的基地（base）和偏置（offset）特征，对比前部层、中部层、后部层的名词区别

任务：
1. 准备名词数据集（水果类、动物类、自然类、人工物品类、抽象概念类等）
2. 模拟名词在各层的激活模式
3. 识别基地神经元（base neurons）：所有名词共享的神经元
4. 识别偏置神经元（offset neurons）：特定名词独有的神经元
5. 分析前部层、中部层、后部层的基地和偏置分布
6. 对比不同名词类别的基地和偏置特征
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 名词数据集（按类别组织）
NOUN_DATA = {
    "fruit": {
        "description": "水果类名词",
        "words": ["apple", "banana", "orange", "grape", "strawberry", "watermelon", "mango", "peach", "pear", "cherry"]
    },
    "animal": {
        "description": "动物类名词",
        "words": ["dog", "cat", "bird", "fish", "lion", "tiger", "elephant", "horse", "rabbit", "monkey"]
    },
    "nature": {
        "description": "自然类名词",
        "words": ["tree", "flower", "mountain", "river", "ocean", "sky", "sun", "moon", "star", "cloud"]
    },
    "artifact": {
        "description": "人工物品类名词",
        "words": ["car", "house", "book", "computer", "phone", "table", "chair", "door", "window", "road"]
    },
    "abstract": {
        "description": "抽象概念类名词",
        "words": ["time", "love", "hope", "fear", "joy", "anger", "peace", "war", "life", "death"]
    },
    "role": {
        "description": "社会角色类名词",
        "words": ["teacher", "doctor", "engineer", "artist", "scientist", "writer", "musician", "chef", "lawyer", "pilot"]
    },
    "food": {
        "description": "食物类名词",
        "words": ["bread", "rice", "pasta", "soup", "salad", "pizza", "burger", "cake", "cookie", "ice cream"]
    },
    "body": {
        "description": "身体部位类名词",
        "words": ["head", "hand", "foot", "eye", "ear", "nose", "mouth", "arm", "leg", "heart"]
    },
    "location": {
        "description": "地点类名词",
        "words": ["city", "country", "school", "hospital", "airport", "station", "park", "garden", "forest", "beach"]
    }
}

# 模型配置
MODEL_CONFIG = {
    "num_layers": 40,  # Qwen3-4B的层数
    "hidden_size": 2048,
    "activation_rate": 0.15,  # 基地激活率（15%）
    "offset_rate": 0.05,  # 偏置激活率（5%）
    "layer_bias": {
        "noun": (0.6, 0.8),  # 名词在后部层激活
    }
}


class NounBaseOffsetAnalyzer:
    """名词基地加偏置分析器"""
    
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.num_layers = model_config["num_layers"]
        self.hidden_size = model_config["hidden_size"]
        
    def simulate_noun_activation(self, noun: str, category: str) -> np.ndarray:
        """
        模拟名词激活
        
        参数:
            noun: 名词
            category: 名词类别
        
        返回:
            (num_layers, hidden_size) 的激活矩阵
        """
        # 设置随机种子（基于名词）
        np.random.seed(sum(ord(c) for c in noun) % (2**32))
        
        # 初始化激活矩阵
        activations = np.zeros((self.num_layers, self.hidden_size))
        
        # 1. 基地神经元（base neurons）
        # 所有名词共享的神经元（同一类别内）
        # 使用类别的随机种子
        category_seed = sum(ord(c) for c in category) % (2**32)
        np.random.seed(category_seed)
        base_neurons = np.random.choice(
            self.hidden_size, 
            int(self.model_config["activation_rate"] * self.hidden_size), 
            replace=False
        )
        
        # 2. 偏置神经元（offset neurons）
        # 特定名词独有的神经元
        np.random.seed(sum(ord(c) for c in noun) % (2**32))
        offset_neurons = np.random.choice(
            self.hidden_size, 
            int(self.model_config["offset_rate"] * self.hidden_size), 
            replace=False
        )
        
        # 3. 层偏向（layer bias）
        # 名词在后部层更活跃
        bias_min, bias_max = self.model_config["layer_bias"]["noun"]
        min_layer = int(bias_min * self.num_layers)
        max_layer = int(bias_max * self.num_layers)
        
        # 4. 生成激活
        for layer in range(self.num_layers):
            # 计算该层的激活强度
            if min_layer <= layer <= max_layer:
                # 在偏向范围内，激活强度更高
                intensity = 1.0
            else:
                # 在偏向范围外，激活强度更低
                intensity = 0.3
            
            # 基地神经元激活
            activations[layer, base_neurons] = np.random.uniform(0.5, 1.0, len(base_neurons)) * intensity
            
            # 偏置神经元激活
            activations[layer, offset_neurons] = np.random.uniform(0.5, 1.0, len(offset_neurons)) * intensity
        
        return activations
    
    def identify_base_neurons(self, activations_list: List[np.ndarray], threshold: float = 0.5) -> Set[int]:
        """
        识别基地神经元（在所有名词中都激活的神经元）
        
        参数:
            activations_list: 多个名词的激活矩阵列表
            threshold: 激活阈值
        
        返回:
            基地神经元集合
        """
        # 收集每个名词的激活神经元
        active_sets = []
        for activations in activations_list:
            # 找到所有层中激活的神经元
            active_neurons = set()
            for layer in range(activations.shape[0]):
                layer_active = np.where(activations[layer] > threshold)[0]
                active_neurons.update(layer_active)
            active_sets.append(active_neurons)
        
        # 找到在所有名词中都激活的神经元
        base_neurons = set.intersection(*active_sets)
        
        return base_neurons
    
    def identify_offset_neurons(self, activations: np.ndarray, base_neurons: Set[int], threshold: float = 0.5) -> Set[int]:
        """
        识别偏置神经元（特定名词独有的神经元）
        
        参数:
            activations: 单个名词的激活矩阵
            base_neurons: 基地神经元集合
            threshold: 激活阈值
        
        返回:
            偏置神经元集合
        """
        # 找到所有激活的神经元
        active_neurons = set()
        for layer in range(activations.shape[0]):
            layer_active = np.where(activations[layer] > threshold)[0]
            active_neurons.update(layer_active)
        
        # 偏置神经元 = 激活神经元 - 基地神经元
        offset_neurons = active_neurons - base_neurons
        
        return offset_neurons
    
    def analyze_layer_distribution(self, activations: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        分析层分布
        
        参数:
            activations: (num_layers, hidden_size) 的激活矩阵
            threshold: 激活阈值
        
        返回:
            层分布统计字典
        """
        # 计算每层的激活数量
        layer_activations = []
        for layer in range(activations.shape[0]):
            active_count = np.sum(activations[layer] > threshold)
            layer_activations.append(active_count)
        
        layer_activations = np.array(layer_activations)
        
        # 计算前中后部分的比例
        early_end = self.num_layers // 3
        middle_end = 2 * self.num_layers // 3
        
        total_activations = np.sum(layer_activations)
        if total_activations > 0:
            early_ratio = np.sum(layer_activations[:early_end]) / total_activations
            middle_ratio = np.sum(layer_activations[early_end:middle_end]) / total_activations
            late_ratio = np.sum(layer_activations[middle_end:]) / total_activations
        else:
            early_ratio = middle_ratio = late_ratio = 0.0
        
        # 找到最大激活层
        max_layer = int(np.argmax(layer_activations))
        
        return {
            "layer_activations": layer_activations.tolist(),
            "early_ratio": float(early_ratio),
            "middle_ratio": float(middle_ratio),
            "late_ratio": float(late_ratio),
            "max_layer": max_layer,
            "total_activations": float(total_activations)
        }
    
    def analyze_noun(self, noun: str, category: str, activations: np.ndarray, base_neurons: Set[int]) -> Dict:
        """
        分析单个名词的基地和偏置特征
        
        参数:
            noun: 名词
            category: 类别
            activations: 激活矩阵
            base_neurons: 基地神经元集合
        
        返回:
            名词分析结果字典
        """
        # 识别偏置神经元
        offset_neurons = self.identify_offset_neurons(activations, base_neurons)
        
        # 分析层分布
        layer_distribution = self.analyze_layer_distribution(activations)
        
        # 分析基地神经元的层分布
        base_activations = np.zeros((self.num_layers, self.hidden_size))
        for layer in range(self.num_layers):
            for neuron in base_neurons:
                base_activations[layer, neuron] = activations[layer, neuron]
        base_layer_distribution = self.analyze_layer_distribution(base_activations)
        
        # 分析偏置神经元的层分布
        offset_activations = np.zeros((self.num_layers, self.hidden_size))
        for layer in range(self.num_layers):
            for neuron in offset_neurons:
                offset_activations[layer, neuron] = activations[layer, neuron]
        offset_layer_distribution = self.analyze_layer_distribution(offset_activations)
        
        return {
            "noun": noun,
            "category": category,
            "num_base_neurons": int(len(base_neurons)),
            "num_offset_neurons": int(len(offset_neurons)),
            "base_ratio": float(len(base_neurons) / self.hidden_size),
            "offset_ratio": float(len(offset_neurons) / self.hidden_size),
            "layer_distribution": layer_distribution,
            "base_layer_distribution": base_layer_distribution,
            "offset_layer_distribution": offset_layer_distribution,
            "base_neurons": sorted([int(x) for x in list(base_neurons)])[:100],  # 只保存前100个
            "offset_neurons": sorted([int(x) for x in list(offset_neurons)])[:100]  # 只保存前100个
        }
    
    def analyze_category(self, category: str, nouns: List[str]) -> Dict:
        """
        分析一个名词类别的基地和偏置特征
        
        参数:
            category: 类别名称
            nouns: 名词列表
        
        返回:
            类别分析结果字典
        """
        print(f"  分析类别: {category} ({len(nouns)} 个名词)")
        
        # 1. 模拟所有名词的激活
        activations_list = []
        for noun in nouns:
            activations = self.simulate_noun_activation(noun, category)
            activations_list.append(activations)
        
        # 2. 识别基地神经元（类别内所有名词共享）
        base_neurons = self.identify_base_neurons(activations_list)
        print(f"    基地神经元数量: {len(base_neurons)} ({len(base_neurons)/self.hidden_size*100:.2f}%)")
        
        # 3. 分析每个名词
        noun_results = []
        for noun, activations in zip(nouns, activations_list):
            noun_result = self.analyze_noun(noun, category, activations, base_neurons)
            noun_results.append(noun_result)
        
        # 4. 聚合统计
        base_ratios = [r["base_ratio"] for r in noun_results]
        offset_ratios = [r["offset_ratio"] for r in noun_results]
        
        avg_result = {
            "category": category,
            "num_nouns": int(len(nouns)),
            "num_base_neurons": int(len(base_neurons)),
            "base_ratio": float(len(base_neurons) / self.hidden_size),
            "avg_offset_neurons": float(np.mean([r["num_offset_neurons"] for r in noun_results])),
            "avg_offset_ratio": float(np.mean(offset_ratios)),
            "base_ratio_std": float(np.std(base_ratios)),
            "offset_ratio_std": float(np.std(offset_ratios)),
            "noun_results": noun_results
        }
        
        print(f"    平均偏置神经元: {avg_result['avg_offset_neurons']:.1f} ({avg_result['avg_offset_ratio']*100:.2f}%)")
        
        return avg_result
    
    def compare_layers(self, category_results: Dict) -> Dict:
        """
        对比前部层、中部层、后部层的名词特征
        
        参数:
            category_results: 类别分析结果
        
        返回:
            层对比结果字典
        """
        early_end = self.num_layers // 3
        middle_end = 2 * self.num_layers // 3
        
        results = {
            "early_layer": {"range": f"0-{early_end-1}", "categories": {}},
            "middle_layer": {"range": f"{early_end}-{middle_end-1}", "categories": {}},
            "late_layer": {"range": f"{middle_end}-{self.num_layers-1}", "categories": {}}
        }
        
        for category, cat_result in category_results.items():
            # 前部层
            early_base = 0
            early_offset = 0
            for noun_result in cat_result["noun_results"]:
                early_base += noun_result["base_layer_distribution"]["early_ratio"]
                early_offset += noun_result["offset_layer_distribution"]["early_ratio"]
            results["early_layer"]["categories"][category] = {
                "avg_base_ratio": early_base / cat_result["num_nouns"],
                "avg_offset_ratio": early_offset / cat_result["num_nouns"]
            }
            
            # 中部层
            middle_base = 0
            middle_offset = 0
            for noun_result in cat_result["noun_results"]:
                middle_base += noun_result["base_layer_distribution"]["middle_ratio"]
                middle_offset += noun_result["offset_layer_distribution"]["middle_ratio"]
            results["middle_layer"]["categories"][category] = {
                "avg_base_ratio": middle_base / cat_result["num_nouns"],
                "avg_offset_ratio": middle_offset / cat_result["num_nouns"]
            }
            
            # 后部层
            late_base = 0
            late_offset = 0
            for noun_result in cat_result["noun_results"]:
                late_base += noun_result["base_layer_distribution"]["late_ratio"]
                late_offset += noun_result["offset_layer_distribution"]["late_ratio"]
            results["late_layer"]["categories"][category] = {
                "avg_base_ratio": late_base / cat_result["num_nouns"],
                "avg_offset_ratio": late_offset / cat_result["num_nouns"]
            }
        
        return results


def main():
    """主函数"""
    print("="*70)
    print("Stage427: 名词基地加偏置特征分析")
    print("="*70)
    
    # 结果字典
    results = {
        "metadata": {
            "stage": 427,
            "description": "名词基地加偏置特征分析",
            "timestamp": "2026-03-30",
            "num_categories": len(NOUN_DATA),
            "total_nouns": sum(len(data["words"]) for data in NOUN_DATA.values())
        },
        "model_config": MODEL_CONFIG,
        "category_analysis": {},
        "layer_comparison": {},
        "cross_category_comparison": {}
    }
    
    # 创建分析器
    analyzer = NounBaseOffsetAnalyzer(MODEL_CONFIG)
    
    # 对每个类别进行分析
    print(f"\n{'='*70}")
    print("类别分析")
    print(f"{'='*70}")
    
    for category, data in NOUN_DATA.items():
        print(f"\n{category} ({data['description']}):")
        cat_result = analyzer.analyze_category(category, data["words"])
        results["category_analysis"][category] = cat_result
    
    # 层对比分析
    print(f"\n{'='*70}")
    print("层对比分析")
    print(f"{'='*70}")
    
    layer_comparison = analyzer.compare_layers(results["category_analysis"])
    results["layer_comparison"] = layer_comparison
    
    # 打印层对比结果
    for layer_name, layer_data in layer_comparison.items():
        print(f"\n{layer_name} ({layer_data['range']}):")
        for category, cat_data in layer_data["categories"].items():
            print(f"  {category}: 基地={cat_data['avg_base_ratio']*100:.2f}%, 偏置={cat_data['avg_offset_ratio']*100:.2f}%")
    
    # 跨类别对比
    print(f"\n{'='*70}")
    print("跨类别对比")
    print(f"{'='*70}")
    
    for category, cat_result in results["category_analysis"].items():
        print(f"\n{category}:")
        print(f"  基地神经元: {cat_result['num_base_neurons']} ({cat_result['base_ratio']*100:.2f}%)")
        print(f"  平均偏置神经元: {cat_result['avg_offset_neurons']:.1f} ({cat_result['avg_offset_ratio']*100:.2f}%)")
    
    # 基地神经元大小对比
    print(f"\n{'='*70}")
    print("基地神经元大小对比")
    print(f"{'='*70}")
    
    base_sizes = [(cat, results["category_analysis"][cat]["num_base_neurons"]) 
                  for cat in results["category_analysis"]]
    base_sizes.sort(key=lambda x: x[1], reverse=True)
    
    for cat, size in base_sizes:
        print(f"  {cat}: {size} ({results['category_analysis'][cat]['base_ratio']*100:.2f}%)")
    
    # 偏置神经元大小对比
    print(f"\n{'='*70}")
    print("偏置神经元大小对比")
    print(f"{'='*70}")
    
    offset_sizes = [(cat, results["category_analysis"][cat]["avg_offset_neurons"]) 
                    for cat in results["category_analysis"]]
    offset_sizes.sort(key=lambda x: x[1], reverse=True)
    
    for cat, size in offset_sizes:
        print(f"  {cat}: {size:.1f} ({results['category_analysis'][cat]['avg_offset_ratio']*100:.2f}%)")
    
    # 保存结果
    output_file = Path(__file__).parent / "noun_base_offset_stage427.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 结果已保存到: {output_file}")
    
    print("\n" + "="*70)
    print("Stage427完成")
    print("="*70)


if __name__ == "__main__":
    main()
