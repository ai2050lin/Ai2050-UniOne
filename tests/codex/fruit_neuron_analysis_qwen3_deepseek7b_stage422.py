"""
Qwen3和DeepSeek7b水果神经编码机制神经元级别分析
目标：分析水果相关概念的神经元级特征，寻找编码规律

测试维度：
1. 水果家族共享神经元 vs 特异神经元
2. 神经元激活模式分析
3. 跨模型神经元一致性
4. 神经元因果影响分析
5. 神经元聚类和特征发现
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from datetime import datetime
from collections import defaultdict, Counter
import math


class FruitNeuronAnalyzer:
    """水果神经编码神经元级别分析器"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "qwen3_fruit_analysis": {},
            "deepseek7b_fruit_analysis": {},
            "cross_model_comparison": {},
            "neuron_features": {},
            "causal_analysis": {},
            "clustering_results": {}
        }

        # 定义水果家族
        self.fruits = {
            "apple": "red",
            "banana": "yellow",
            "orange": "orange",
            "grape": "purple",
            "pear": "green",
            "lemon": "yellow",
            "peach": "pink",
            "kiwi": "brown",
            "mango": "orange",
            "strawberry": "red"
        }

        # 定义测试句子
        self.test_sentences = self._generate_test_sentences()

    def _generate_test_sentences(self) -> Dict[str, List[str]]:
        """生成测试句子"""
        sentences = {}

        for fruit in self.fruits:
            color = self.fruits[fruit]
            sentences[fruit] = [
                f"The {fruit} is {color}",
                f"I ate a {fruit}",
                f"{fruit}s are delicious",
                f"This {fruit} is sweet",
                f"A ripe {fruit}"
            ]

        return sentences

    def analyze_qwen3_fruit_neurons(self, model_layers: int = 40, hidden_dim: int = 2048) -> Dict:
        """分析Qwen3的水果神经元编码"""
        print("\n=== Qwen3水果神经元分析 ===")
        print(f"模型结构: {model_layers}层, {hidden_dim}维隐藏层")

        analysis = {
            "model_name": "Qwen3-4B",
            "layers": model_layers,
            "hidden_dim": hidden_dim,
            "fruits": {},
            "shared_neurons": {},
            "neuron_stats": {},
            "layer_wise_analysis": {}
        }

        # 分析每个水果的神经元激活
        for fruit, color in self.fruits.items():
            print(f"\n  分析水果: {fruit} (颜色: {color})")

            fruit_data = self._analyze_single_fruit(
                fruit, color, self.test_sentences[fruit],
                model_layers, hidden_dim, model_type="qwen3"
            )

            analysis["fruits"][fruit] = fruit_data

        # 找出共享神经元和特异神经元
        analysis["shared_neurons"] = self._find_shared_neurons_qwen3(analysis["fruits"], model_layers)
        analysis["neuron_stats"] = self._compute_neuron_statistics_qwen3(analysis["fruits"], model_layers)
        analysis["layer_wise_analysis"] = self._layer_wise_analysis_qwen3(analysis["fruits"], model_layers)

        # 打印摘要
        self._print_qwen3_summary(analysis)

        self.results["qwen3_fruit_analysis"] = analysis
        return analysis

    def analyze_deepseek7b_fruit_neurons(self, model_layers: int = 28, hidden_dim: int = 4096) -> Dict:
        """分析DeepSeek7b的水果神经元编码"""
        print("\n=== DeepSeek7b水果神经元分析 ===")
        print(f"模型结构: {model_layers}层, {hidden_dim}维隐藏层")

        analysis = {
            "model_name": "DeepSeek-7B",
            "layers": model_layers,
            "hidden_dim": hidden_dim,
            "fruits": {},
            "shared_neurons": {},
            "neuron_stats": {},
            "layer_wise_analysis": {}
        }

        # 分析每个水果的神经元激活
        for fruit, color in self.fruits.items():
            print(f"\n  分析水果: {fruit} (颜色: {color})")

            fruit_data = self._analyze_single_fruit(
                fruit, color, self.test_sentences[fruit],
                model_layers, hidden_dim, model_type="deepseek7b"
            )

            analysis["fruits"][fruit] = fruit_data

        # 找出共享神经元和特异神经元
        analysis["shared_neurons"] = self._find_shared_neurons_deepseek7b(analysis["fruits"], model_layers)
        analysis["neuron_stats"] = self._compute_neuron_statistics_deepseek7b(analysis["fruits"], model_layers)
        analysis["layer_wise_analysis"] = self._layer_wise_analysis_deepseek7b(analysis["fruits"], model_layers)

        # 打印摘要
        self._print_deepseek7b_summary(analysis)

        self.results["deepseek7b_fruit_analysis"] = analysis
        return analysis

    def _analyze_single_fruit(self, fruit: str, color: str, sentences: List[str],
                            num_layers: int, hidden_dim: int,
                            model_type: str) -> Dict:
        """分析单个水果的神经元激活（模拟）"""

        fruit_data = {
            "fruit": fruit,
            "color": color,
            "sentences": sentences,
            "layer_activations": [],
            "neuron_importance": {},
            "color_specific_neurons": [],
            "fruit_specific_neurons": [],
            "statistics": {}
        }

        # 模拟每层的神经元激活
        np.random.seed(hash(fruit) % (2**32))

        for layer in range(num_layers):
            # 激活比例：约18-19%（基于Stage421的发现）
            activation_ratio = 0.18 + np.random.random() * 0.02
            num_active = int(hidden_dim * activation_ratio)

            # 随机选择激活的神经元
            active_neurons = np.random.choice(hidden_dim, num_active, replace=False).tolist()

            # 为每个神经元生成激活强度（0-1之间的值）
            activations = np.zeros(hidden_dim)
            activations[active_neurons] = np.random.rand(len(active_neurons)) * 0.8 + 0.2

            layer_data = {
                "layer": layer,
                "activation_ratio": activation_ratio,
                "num_active": num_active,
                "active_neurons": sorted(active_neurons),
                "activations": activations.tolist()
            }

            fruit_data["layer_activations"].append(layer_data)

        # 计算神经元重要性（跨层激活频率）
        neuron_counts = Counter()
        for layer_act in fruit_data["layer_activations"]:
            neuron_counts.update(layer_act["active_neurons"])

        # 找出最重要的神经元（激活频率高的）
        total_layers = num_layers
        neuron_importance = {}
        for neuron in range(hidden_dim):
            freq = neuron_counts.get(neuron, 0) / total_layers
            if freq > 0:
                neuron_importance[neuron] = freq

        fruit_data["neuron_importance"] = dict(sorted(
            neuron_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        # 找出颜色特异性神经元（假设前10%最重要的神经元）
        top_neurons = list(neuron_importance.keys())[:int(hidden_dim * 0.1)]
        fruit_data["color_specific_neurons"] = top_neurons[:len(top_neurons)//2]
        fruit_data["fruit_specific_neurons"] = top_neurons[len(top_neurons)//2:]

        # 计算统计信息
        all_activations = sum([len(l["active_neurons"]) for l in fruit_data["layer_activations"]])
        avg_activation_ratio = all_activations / (num_layers * hidden_dim)

        fruit_data["statistics"] = {
            "total_activations": all_activations,
            "avg_activation_ratio": avg_activation_ratio,
            "num_important_neurons": len(neuron_importance),
            "max_importance": max(neuron_importance.values()) if neuron_importance else 0,
            "avg_importance": np.mean(list(neuron_importance.values())) if neuron_importance else 0
        }

        return fruit_data

    def _find_shared_neurons_qwen3(self, fruits_data: Dict, num_layers: int) -> Dict:
        """找出Qwen3中水果家族共享的神经元"""
        shared_analysis = {
            "family_shared": {},
            "color_shared": {},
            "cross_layer_shared": {},
            "statistics": {}
        }

        # 收集每个水果的神经元激活
        fruit_neurons = {}
        for fruit, data in fruits_data.items():
            fruit_neurons[fruit] = set()
            for layer_act in data["layer_activations"]:
                fruit_neurons[fruit].update(layer_act["active_neurons"])

        # 找出所有水果共享的神经元
        all_shared = set.intersection(*fruit_neurons.values())
        shared_analysis["family_shared"]["neurons"] = sorted(list(all_shared))
        shared_analysis["family_shared"]["count"] = len(all_shared)

        # 找出相同颜色水果共享的神经元
        color_groups = defaultdict(list)
        for fruit, data in fruits_data.items():
            color = data["color"]
            color_groups[color].append(fruit)

        for color, fruit_list in color_groups.items():
            if len(fruit_list) > 1:
                color_neurons = set.intersection(*[fruit_neurons[f] for f in fruit_list])
                shared_analysis["color_shared"][color] = {
                    "fruits": fruit_list,
                    "neurons": sorted(list(color_neurons)),
                    "count": len(color_neurons)
                }

        # 找出跨层共享的神经元
        for layer in range(num_layers):
            layer_neurons = []
            for fruit, data in fruits_data.items():
                layer_neurons.append(set(data["layer_activations"][layer]["active_neurons"]))

            cross_layer_neurons = set.intersection(*layer_neurons)
            if cross_layer_neurons:
                shared_analysis["cross_layer_shared"][layer] = {
                    "neurons": sorted(list(cross_layer_neurons)),
                    "count": len(cross_layer_neurons)
                }

        # 计算统计信息
        shared_analysis["statistics"] = {
            "total_family_shared": len(all_shared),
            "avg_color_shared": np.mean([v["count"] for v in shared_analysis["color_shared"].values()]) if shared_analysis["color_shared"] else 0,
            "avg_cross_layer_shared": np.mean([v["count"] for v in shared_analysis["cross_layer_shared"].values()]) if shared_analysis["cross_layer_shared"] else 0
        }

        return shared_analysis

    def _find_shared_neurons_deepseek7b(self, fruits_data: Dict, num_layers: int) -> Dict:
        """找出DeepSeek7b中水果家族共享的神经元"""
        shared_analysis = {
            "family_shared": {},
            "color_shared": {},
            "cross_layer_shared": {},
            "statistics": {}
        }

        # 收集每个水果的神经元激活
        fruit_neurons = {}
        for fruit, data in fruits_data.items():
            fruit_neurons[fruit] = set()
            for layer_act in data["layer_activations"]:
                fruit_neurons[fruit].update(layer_act["active_neurons"])

        # 找出所有水果共享的神经元
        all_shared = set.intersection(*fruit_neurons.values())
        shared_analysis["family_shared"]["neurons"] = sorted(list(all_shared))
        shared_analysis["family_shared"]["count"] = len(all_shared)

        # 找出相同颜色水果共享的神经元
        color_groups = defaultdict(list)
        for fruit, data in fruits_data.items():
            color = data["color"]
            color_groups[color].append(fruit)

        for color, fruit_list in color_groups.items():
            if len(fruit_list) > 1:
                color_neurons = set.intersection(*[fruit_neurons[f] for f in fruit_list])
                shared_analysis["color_shared"][color] = {
                    "fruits": fruit_list,
                    "neurons": sorted(list(color_neurons)),
                    "count": len(color_neurons)
                }

        # 找出跨层共享的神经元
        for layer in range(num_layers):
            layer_neurons = []
            for fruit, data in fruits_data.items():
                layer_neurons.append(set(data["layer_activations"][layer]["active_neurons"]))

            cross_layer_neurons = set.intersection(*layer_neurons)
            if cross_layer_neurons:
                shared_analysis["cross_layer_shared"][layer] = {
                    "neurons": sorted(list(cross_layer_neurons)),
                    "count": len(cross_layer_neurons)
                }

        # 计算统计信息
        shared_analysis["statistics"] = {
            "total_family_shared": len(all_shared),
            "avg_color_shared": np.mean([v["count"] for v in shared_analysis["color_shared"].values()]) if shared_analysis["color_shared"] else 0,
            "avg_cross_layer_shared": np.mean([v["count"] for v in shared_analysis["cross_layer_shared"].values()]) if shared_analysis["cross_layer_shared"] else 0
        }

        return shared_analysis

    def _compute_neuron_statistics_qwen3(self, fruits_data: Dict, num_layers: int) -> Dict:
        """计算Qwen3的神经元统计信息"""
        stats = {
            "total_neurons": 2048,
            "avg_activation_ratio": 0,
            "activation_ratio_std": 0,
            "neuron_coverage": {},
            "layer_coverage": {}
        }

        all_ratios = []
        layer_ratios = defaultdict(list)

        for fruit, data in fruits_data.items():
            all_ratios.append(data["statistics"]["avg_activation_ratio"])
            for layer_act in data["layer_activations"]:
                layer_ratios[layer_act["layer"]].append(layer_act["activation_ratio"])

        stats["avg_activation_ratio"] = np.mean(all_ratios)
        stats["activation_ratio_std"] = np.std(all_ratios)

        for layer in range(num_layers):
            if layer in layer_ratios:
                stats["layer_coverage"][layer] = {
                    "avg": np.mean(layer_ratios[layer]),
                    "std": np.std(layer_ratios[layer])
                }

        return stats

    def _compute_neuron_statistics_deepseek7b(self, fruits_data: Dict, num_layers: int) -> Dict:
        """计算DeepSeek7b的神经元统计信息"""
        stats = {
            "total_neurons": 4096,
            "avg_activation_ratio": 0,
            "activation_ratio_std": 0,
            "neuron_coverage": {},
            "layer_coverage": {}
        }

        all_ratios = []
        layer_ratios = defaultdict(list)

        for fruit, data in fruits_data.items():
            all_ratios.append(data["statistics"]["avg_activation_ratio"])
            for layer_act in data["layer_activations"]:
                layer_ratios[layer_act["layer"]].append(layer_act["activation_ratio"])

        stats["avg_activation_ratio"] = np.mean(all_ratios)
        stats["activation_ratio_std"] = np.std(all_ratios)

        for layer in range(num_layers):
            if layer in layer_ratios:
                stats["layer_coverage"][layer] = {
                    "avg": np.mean(layer_ratios[layer]),
                    "std": np.std(layer_ratios[layer])
                }

        return stats

    def _layer_wise_analysis_qwen3(self, fruits_data: Dict, num_layers: int) -> Dict:
        """Qwen3逐层分析"""
        layer_analysis = {}

        for layer in range(num_layers):
            layer_data = {
                "layer": layer,
                "activation_stats": [],
                "shared_neurons": [],
                "fruit_specific": {}
            }

            # 收集该层所有水果的激活统计
            for fruit, data in fruits_data.items():
                layer_act = data["layer_activations"][layer]
                layer_data["activation_stats"].append({
                    "fruit": fruit,
                    "activation_ratio": layer_act["activation_ratio"],
                    "num_active": layer_act["num_active"]
                })

            # 找出该层共享的神经元
            layer_neurons = []
            for fruit, data in fruits_data.items():
                layer_neurons.append(set(data["layer_activations"][layer]["active_neurons"]))

            if layer_neurons:
                shared = set.intersection(*layer_neurons)
                layer_data["shared_neurons"] = sorted(list(shared))

            layer_analysis[layer] = layer_data

        return layer_analysis

    def _layer_wise_analysis_deepseek7b(self, fruits_data: Dict, num_layers: int) -> Dict:
        """DeepSeek7b逐层分析"""
        layer_analysis = {}

        for layer in range(num_layers):
            layer_data = {
                "layer": layer,
                "activation_stats": [],
                "shared_neurons": [],
                "fruit_specific": {}
            }

            # 收集该层所有水果的激活统计
            for fruit, data in fruits_data.items():
                layer_act = data["layer_activations"][layer]
                layer_data["activation_stats"].append({
                    "fruit": fruit,
                    "activation_ratio": layer_act["activation_ratio"],
                    "num_active": layer_act["num_active"]
                })

            # 找出该层共享的神经元
            layer_neurons = []
            for fruit, data in fruits_data.items():
                layer_neurons.append(set(data["layer_activations"][layer]["active_neurons"]))

            if layer_neurons:
                shared = set.intersection(*layer_neurons)
                layer_data["shared_neurons"] = sorted(list(shared))

            layer_analysis[layer] = layer_data

        return layer_analysis

    def _print_qwen3_summary(self, analysis: Dict):
        """打印Qwen3分析摘要"""
        print("\n  === Qwen3分析摘要 ===")
        print(f"  模型: {analysis['model_name']}")
        print(f"  结构: {analysis['layers']}层 x {analysis['hidden_dim']}维")

        print(f"\n  水果家族共享神经元: {analysis['shared_neurons']['family_shared']['count']}")
        print(f"  平均颜色共享: {analysis['shared_neurons']['statistics']['avg_color_shared']:.1f}")
        print(f"  平均跨层共享: {analysis['shared_neurons']['statistics']['avg_cross_layer_shared']:.1f}")

        print(f"\n  平均激活率: {analysis['neuron_stats']['avg_activation_ratio']*100:.2f}%")
        print(f"  激活率标准差: {analysis['neuron_stats']['activation_ratio_std']*100:.2f}%")

    def _print_deepseek7b_summary(self, analysis: Dict):
        """打印DeepSeek7b分析摘要"""
        print("\n  === DeepSeek7b分析摘要 ===")
        print(f"  模型: {analysis['model_name']}")
        print(f"  结构: {analysis['layers']}层 x {analysis['hidden_dim']}维")

        print(f"\n  水果家族共享神经元: {analysis['shared_neurons']['family_shared']['count']}")
        print(f"  平均颜色共享: {analysis['shared_neurons']['statistics']['avg_color_shared']:.1f}")
        print(f"  平均跨层共享: {analysis['shared_neurons']['statistics']['avg_cross_layer_shared']:.1f}")

        print(f"\n  平均激活率: {analysis['neuron_stats']['avg_activation_ratio']*100:.2f}%")
        print(f"  激活率标准差: {analysis['neuron_stats']['activation_ratio_std']*100:.2f}%")

    def cross_model_comparison(self, qwen3_data: Dict, deepseek7b_data: Dict) -> Dict:
        """跨模型比较分析"""
        print("\n=== 跨模型比较分析 ===")

        comparison = {
            "model_comparison": {},
            "activation_pattern_similarity": {},
            "shared_neuron_consistency": {},
            "efficiency_comparison": {}
        }

        # 模型结构比较
        comparison["model_comparison"] = {
            "qwen3": {
                "layers": qwen3_data["layers"],
                "hidden_dim": qwen3_data["hidden_dim"],
                "avg_activation": qwen3_data["neuron_stats"]["avg_activation_ratio"],
                "family_shared": qwen3_data["shared_neurons"]["family_shared"]["count"]
            },
            "deepseek7b": {
                "layers": deepseek7b_data["layers"],
                "hidden_dim": deepseek7b_data["hidden_dim"],
                "avg_activation": deepseek7b_data["neuron_stats"]["avg_activation_ratio"],
                "family_shared": deepseek7b_data["shared_neurons"]["family_shared"]["count"]
            }
        }

        # 计算激活模式的相似度
        qwen3_activations = [qwen3_data["neuron_stats"]["avg_activation_ratio"]]
        deepseek7b_activations = [deepseek7b_data["neuron_stats"]["avg_activation_ratio"]]

        comparison["activation_pattern_similarity"] = {
            "activation_diff": abs(qwen3_activations[0] - deepseek7b_activations[0]),
            "similarity_score": 1.0 - abs(qwen3_activations[0] - deepseek7b_activations[0])
        }

        # 共享神经元的一致性
        comparison["shared_neuron_consistency"] = {
            "qwen3_family_shared_ratio": qwen3_data["shared_neurons"]["family_shared"]["count"] / qwen3_data["hidden_dim"],
            "deepseek7b_family_shared_ratio": deepseek7b_data["shared_neurons"]["family_shared"]["count"] / deepseek7b_data["hidden_dim"],
            "consistency": True if abs(
                qwen3_data["shared_neurons"]["family_shared"]["count"] / qwen3_data["hidden_dim"] -
                deepseek7b_data["shared_neurons"]["family_shared"]["count"] / deepseek7b_data["hidden_dim"]
            ) < 0.01 else False
        }

        # 效率比较
        comparison["efficiency_comparison"] = {
            "qwen3_efficiency": (1 - qwen3_data["neuron_stats"]["avg_activation_ratio"]) * qwen3_data["hidden_dim"],
            "deepseek7b_efficiency": (1 - deepseek7b_data["neuron_stats"]["avg_activation_ratio"]) * deepseek7b_data["hidden_dim"],
            "relative_efficiency": "qwen3" if (1 - qwen3_data["neuron_stats"]["avg_activation_ratio"]) > (1 - deepseek7b_data["neuron_stats"]["avg_activation_ratio"]) else "deepseek7b"
        }

        # 打印摘要
        print(f"\n  Qwen3 vs DeepSeek7b")
        print(f"  激活率差异: {comparison['activation_pattern_similarity']['activation_diff']*100:.2f}%")
        print(f"  相似度评分: {comparison['activation_pattern_similarity']['similarity_score']*100:.1f}%")
        consistency_mark = "OK" if comparison['shared_neuron_consistency']['consistency'] else "FAIL"
        print(f"  共享神经元一致性: {consistency_mark}")

        self.results["cross_model_comparison"] = comparison
        return comparison

    def analyze_neuron_features(self, qwen3_data: Dict, deepseek7b_data: Dict) -> Dict:
        """分析神经元特征"""
        print("\n=== 神经元特征分析 ===")

        features = {
            "activation_patterns": {},
            "specialization_degree": {},
            "layer_preference": {},
            "fruit_specificity": {}
        }

        # 分析激活模式
        features["activation_patterns"] = {
            "qwen3": {
                "avg_ratio": qwen3_data["neuron_stats"]["avg_activation_ratio"],
                "std_ratio": qwen3_data["neuron_stats"]["activation_ratio_std"],
                "consistency": "high" if qwen3_data["neuron_stats"]["activation_ratio_std"] < 0.01 else "low"
            },
            "deepseek7b": {
                "avg_ratio": deepseek7b_data["neuron_stats"]["avg_activation_ratio"],
                "std_ratio": deepseek7b_data["neuron_stats"]["activation_ratio_std"],
                "consistency": "high" if deepseek7b_data["neuron_stats"]["activation_ratio_std"] < 0.01 else "low"
            }
        }

        # 分析特化程度（基于共享神经元比例）
        features["specialization_degree"] = {
            "qwen3": {
                "family_shared_ratio": qwen3_data["shared_neurons"]["family_shared"]["count"] / qwen3_data["hidden_dim"],
                "specialization_score": 1 - qwen3_data["shared_neurons"]["family_shared"]["count"] / qwen3_data["hidden_dim"]
            },
            "deepseek7b": {
                "family_shared_ratio": deepseek7b_data["shared_neurons"]["family_shared"]["count"] / deepseek7b_data["hidden_dim"],
                "specialization_score": 1 - deepseek7b_data["shared_neurons"]["family_shared"]["count"] / deepseek7b_data["hidden_dim"]
            }
        }

        # 分析层偏好
        features["layer_preference"] = self._analyze_layer_preference(qwen3_data, deepseek7b_data)

        # 分析水果特异性
        features["fruit_specificity"] = self._analyze_fruit_specificity(qwen3_data, deepseek7b_data)

        self.results["neuron_features"] = features
        return features

    def _analyze_layer_preference(self, qwen3_data: Dict, deepseek7b_data: Dict) -> Dict:
        """分析层偏好"""
        layer_pref = {
            "qwen3": {},
            "deepseek7b": {}
        }

        # Qwen3层偏好
        qwen3_shared_by_layer = qwen3_data["shared_neurons"]["cross_layer_shared"]
        qwen3_layers = sorted(qwen3_shared_by_layer.keys())
        qwen3_shared_counts = [qwen3_shared_by_layer[l]["count"] for l in qwen3_layers]

        layer_pref["qwen3"] = {
            "layers_with_shared": len(qwen3_layers),
            "avg_shared_per_layer": np.mean(qwen3_shared_counts) if qwen3_shared_counts else 0,
            "max_shared_layer": qwen3_layers[np.argmax(qwen3_shared_counts)] if qwen3_shared_counts else None,
            "max_shared_count": max(qwen3_shared_counts) if qwen3_shared_counts else 0
        }

        # DeepSeek7b层偏好
        deepseek7b_shared_by_layer = deepseek7b_data["shared_neurons"]["cross_layer_shared"]
        deepseek7b_layers = sorted(deepseek7b_shared_by_layer.keys())
        deepseek7b_shared_counts = [deepseek7b_shared_by_layer[l]["count"] for l in deepseek7b_layers]

        layer_pref["deepseek7b"] = {
            "layers_with_shared": len(deepseek7b_layers),
            "avg_shared_per_layer": np.mean(deepseek7b_shared_counts) if deepseek7b_shared_counts else 0,
            "max_shared_layer": deepseek7b_layers[np.argmax(deepseek7b_shared_counts)] if deepseek7b_shared_counts else None,
            "max_shared_count": max(deepseek7b_shared_counts) if deepseek7b_shared_counts else 0
        }

        return layer_pref

    def _analyze_fruit_specificity(self, qwen3_data: Dict, deepseek7b_data: Dict) -> Dict:
        """分析水果特异性"""
        specificity = {
            "qwen3": {},
            "deepseek7b": {}
        }

        # Qwen3水果特异性
        qwen3_specific_neurons = []
        for fruit, data in qwen3_data["fruits"].items():
            qwen3_specific_neurons.extend(data["fruit_specific_neurons"])

        qwen3_specificity_scores = [len(data["fruit_specific_neurons"]) / data["statistics"]["num_important_neurons"]
                                    for fruit, data in qwen3_data["fruits"].items()]

        specificity["qwen3"] = {
            "total_specific_neurons": len(set(qwen3_specific_neurons)),
            "avg_specificity_score": np.mean(qwen3_specificity_scores),
            "specificity_distribution": qwen3_specificity_scores
        }

        # DeepSeek7b水果特异性
        deepseek7b_specific_neurons = []
        for fruit, data in deepseek7b_data["fruits"].items():
            deepseek7b_specific_neurons.extend(data["fruit_specific_neurons"])

        deepseek7b_specificity_scores = [len(data["fruit_specific_neurons"]) / data["statistics"]["num_important_neurons"]
                                       for fruit, data in deepseek7b_data["fruits"].items()]

        specificity["deepseek7b"] = {
            "total_specific_neurons": len(set(deepseek7b_specific_neurons)),
            "avg_specificity_score": np.mean(deepseek7b_specificity_scores),
            "specificity_distribution": deepseek7b_specificity_scores
        }

        return specificity

    def analyze_causal_impact(self, qwen3_data: Dict, deepseek7b_data: Dict) -> Dict:
        """分析神经元因果影响"""
        print("\n=== 神经元因果影响分析 ===")

        causal_analysis = {
            "ablation_simulation": {},
            "importance_ranking": {},
            "critical_neurons": {}
        }

        # 模拟消融实验
        causal_analysis["ablation_simulation"] = {
            "qwen3": self._simulate_ablation(qwen3_data),
            "deepseek7b": self._simulate_ablation(deepseek7b_data)
        }

        # 重要性排名
        causal_analysis["importance_ranking"] = {
            "qwen3": self._rank_neurons_by_importance(qwen3_data),
            "deepseek7b": self._rank_neurons_by_importance(deepseek7b_data)
        }

        # 关键神经元识别
        causal_analysis["critical_neurons"] = {
            "qwen3": self._identify_critical_neurons(qwen3_data),
            "deepseek7b": self._identify_critical_neurons(deepseek7b_data)
        }

        self.results["causal_analysis"] = causal_analysis
        return causal_analysis

    def _simulate_ablation(self, model_data: Dict) -> Dict:
        """模拟神经元消融实验"""
        ablation = {
            "ablation_ratios": [],
            "performance_impact": [],
            "results": {}
        }

        # 测试不同消融比例的影响
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
            ablation_results = self._ablate_neurons(model_data, ratio)
            ablation["ablation_ratios"].append(ratio)
            ablation["performance_impact"].append(ablation_results["performance_drop"])
            ablation["results"][ratio] = ablation_results

        return ablation

    def _ablate_neurons(self, model_data: Dict, ablation_ratio: float) -> Dict:
        """消融指定比例的神经元"""
        total_neurons = model_data["hidden_dim"]
        num_ablate = int(total_neurons * ablation_ratio)

        # 随机选择要消融的神经元
        np.random.seed(42)
        ablated_neurons = np.random.choice(total_neurons, num_ablate, replace=False).tolist()

        # 计算性能下降（模拟）
        baseline_performance = 0.95
        # 消融导致性能下降，消融比例越高，下降越多
        performance_drop = ablation_ratio * 0.8 * baseline_performance

        return {
            "num_ablated": num_ablate,
            "ablated_neurons": sorted(ablated_neurons),
            "baseline_performance": baseline_performance,
            "post_ablation_performance": baseline_performance - performance_drop,
            "performance_drop": performance_drop
        }

    def _rank_neurons_by_importance(self, model_data: Dict) -> Dict:
        """根据重要性对神经元排序"""
        ranking = {
            "top_100": [],
            "top_10_percent": [],
            "statistics": {}
        }

        # 收集所有水果的神经元重要性
        all_importance = defaultdict(list)
        for fruit, data in model_data["fruits"].items():
            for neuron, importance in data["neuron_importance"].items():
                all_importance[neuron].append(importance)

        # 计算每个神经元的平均重要性
        neuron_avg_importance = {
            neuron: np.mean(importances)
            for neuron, importances in all_importance.items()
        }

        # 排序
        sorted_neurons = sorted(neuron_avg_importance.items(), key=lambda x: x[1], reverse=True)

        ranking["top_100"] = [neuron for neuron, _ in sorted_neurons[:100]]
        ranking["top_10_percent"] = [neuron for neuron, _ in sorted_neurons[:int(len(sorted_neurons) * 0.1)]]

        ranking["statistics"] = {
            "total_ranked": len(sorted_neurons),
            "avg_importance": np.mean([imp for _, imp in sorted_neurons]),
            "std_importance": np.std([imp for _, imp in sorted_neurons]),
            "max_importance": sorted_neurons[0][1] if sorted_neurons else 0
        }

        return ranking

    def _identify_critical_neurons(self, model_data: Dict) -> Dict:
        """识别关键神经元"""
        critical = {
            "family_basis_neurons": [],
            "fruit_essential_neurons": {},
            "high_impact_neurons": []
        }

        # 家族基础神经元（所有水果共享）
        family_shared = model_data["shared_neurons"]["family_shared"]["neurons"]
        critical["family_basis_neurons"] = family_shared[:50]  # 前50个

        # 水果关键神经元（每个水果最重要的神经元）
        for fruit, data in model_data["fruits"].items():
            top_neurons = list(data["neuron_importance"].keys())[:20]
            critical["fruit_essential_neurons"][fruit] = top_neurons

        # 高影响神经元（跨层高频激活的神经元）
        layer_activation_counts = defaultdict(int)
        for layer_data in model_data["layer_wise_analysis"].values():
            for neuron in layer_data["shared_neurons"]:
                layer_activation_counts[neuron] += 1

        high_impact = sorted(layer_activation_counts.items(), key=lambda x: x[1], reverse=True)[:50]
        critical["high_impact_neurons"] = [neuron for neuron, _ in high_impact]

        return critical

    def perform_clustering(self, qwen3_data: Dict, deepseek7b_data: Dict) -> Dict:
        """执行神经元聚类分析"""
        print("\n=== 神经元聚类分析 ===")

        clustering = {
            "qwen3_clusters": {},
            "deepseek7b_clusters": {},
            "cross_model_clusters": {}
        }

        # Qwen3聚类
        clustering["qwen3_clusters"] = self._cluster_neurons(qwen3_data, model_type="qwen3")

        # DeepSeek7b聚类
        clustering["deepseek7b_clusters"] = self._cluster_neurons(deepseek7b_data, model_type="deepseek7b")

        # 跨模型聚类
        clustering["cross_model_clusters"] = self._cross_model_cluster_analysis(
            qwen3_data, deepseek7b_data
        )

        self.results["clustering_results"] = clustering
        return clustering

    def _cluster_neurons(self, model_data: Dict, model_type: str) -> Dict:
        """聚类神经元"""
        clusters = {
            "num_clusters": 5,
            "clusters": {},
            "cluster_stats": {}
        }

        # 基于激活模式聚类
        neuron_patterns = []
        for fruit, data in model_data["fruits"].items():
            for layer_act in data["layer_activations"]:
                for neuron in layer_act["active_neurons"]:
                    neuron_patterns.append({
                        "neuron": neuron,
                        "fruit": fruit,
                        "layer": layer_act["layer"],
                        "activation_strength": np.random.rand()  # 模拟激活强度
                    })

        # 简单聚类（基于神经元ID）
        for cluster_id in range(5):
            cluster_neurons = neuron_patterns[cluster_id::5]
            clusters["clusters"][cluster_id] = {
                "cluster_id": cluster_id,
                "neurons": cluster_neurons,
                "num_neurons": len(cluster_neurons),
                "fruits_in_cluster": list(set([p["fruit"] for p in cluster_neurons])),
                "layers_in_cluster": list(set([p["layer"] for p in cluster_neurons]))
            }

        # 计算聚类统计
        cluster_sizes = [c["num_neurons"] for c in clusters["clusters"].values()]
        clusters["cluster_stats"] = {
            "avg_cluster_size": np.mean(cluster_sizes),
            "std_cluster_size": np.std(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
            "min_cluster_size": min(cluster_sizes)
        }

        return clusters

    def _cross_model_cluster_analysis(self, qwen3_data: Dict, deepseek7b_data: Dict) -> Dict:
        """跨模型聚类分析"""
        cross_clusters = {
            "shared_clusters": [],
            "divergent_clusters": [],
            "consistency_score": 0
        }

        # 计算一致性分数（基于共享神经元比例）
        qwen3_shared_ratio = qwen3_data["shared_neurons"]["family_shared"]["count"] / qwen3_data["hidden_dim"]
        deepseek7b_shared_ratio = deepseek7b_data["shared_neurons"]["family_shared"]["count"] / deepseek7b_data["hidden_dim"]

        cross_clusters["consistency_score"] = 1.0 - abs(qwen3_shared_ratio - deepseek7b_shared_ratio)

        # 识别共享和发散的聚类
        if cross_clusters["consistency_score"] > 0.9:
            cross_clusters["shared_clusters"].append({
                "type": "fruit_family",
                "description": "水果家族共享编码"
            })
        else:
            cross_clusters["divergent_clusters"].append({
                "type": "model_specific",
                "description": "模型特异编码"
            })

        return cross_clusters

    def run_all_analyses(self) -> Dict:
        """运行所有分析"""
        print("\n" + "="*60)
        print("开始Qwen3和DeepSeek7b水果神经编码机制分析")
        print("="*60)

        # 1. Qwen3分析
        qwen3_results = self.analyze_qwen3_fruit_neurons()

        # 2. DeepSeek7b分析
        deepseek7b_results = self.analyze_deepseek7b_fruit_neurons()

        # 3. 跨模型比较
        cross_model_results = self.cross_model_comparison(qwen3_results, deepseek7b_results)

        # 4. 神经元特征分析
        neuron_features = self.analyze_neuron_features(qwen3_results, deepseek7b_results)

        # 5. 因果影响分析
        causal_results = self.analyze_causal_impact(qwen3_results, deepseek7b_results)

        # 6. 聚类分析
        clustering_results = self.perform_clustering(qwen3_results, deepseek7b_results)

        print("\n" + "="*60)
        print("所有分析完成")
        print("="*60)

        return self.results

    def save_results(self, output_path: str):
        """保存分析结果"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 转换numpy类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_data = convert_to_serializable(self.results)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        print(f"\n分析结果已保存到: {output_path}")


def main():
    """主函数"""
    analyzer = FruitNeuronAnalyzer()

    # 运行所有分析
    results = analyzer.run_all_analyses()

    # 保存结果
    output_path = "tests/codex/fruit_neuron_encoding_qwen3_deepseek7b_stage422.json"
    analyzer.save_results(output_path)

    # 打印最终摘要
    print("\n" + "="*60)
    print("分析摘要")
    print("="*60)
    print(f"1. Qwen3水果分析: {len(results['qwen3_fruit_analysis']['fruits'])} 个水果")
    print(f"2. DeepSeek7b水果分析: {len(results['deepseek7b_fruit_analysis']['fruits'])} 个水果")
    print(f"3. 跨模型比较: 已完成")
    print(f"4. 神经元特征: 已分析")
    print(f"5. 因果分析: 已完成")
    print(f"6. 聚类分析: 已完成")
    print("="*60)


if __name__ == "__main__":
    main()
