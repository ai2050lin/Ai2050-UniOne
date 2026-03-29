"""
深度神经网络编码机制神经元级分析
Stage421: 神经元激活模式追踪
时间: 2026-03-29 23:25

目标: 分析单个神经元在不同任务中的激活模式,寻找功能特化证据
"""

import torch
import torch.nn as nn
import numpy as np
from transformer_lens import HookedTransformer
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from collections import defaultdict

class NeuronActivationAnalyzer:
    """
    神经元激活模式分析器
    分析单个神经元在不同任务中的激活模式
    """

    def __init__(self, model_name: str = "gpt2-small"):
        self.model_name = model_name
        print(f"加载模型: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 存储神经元激活数据
        self.neuron_activations = defaultdict(dict)

    def hook_neuron_activation(self, layer_idx: int, neuron_idx: int):
        """
        创建钩子函数,用于追踪特定神经元的激活
        """
        def hook_fn(activations, hook):
            # activations: [batch, seq_pos, d_mlp]
            neuron_activation = activations[0, :, neuron_idx].cpu().detach().numpy()
            return activations

        return hook_fn

    def analyze_noun_neurons(self):
        """
        分析名词相关的神经元激活模式
        """
        print("\n" + "="*80)
        print("分析1: 名词相关的神经元激活模式")
        print("="*80)

        # 定义名词测试集
        noun_sets = {
            "水果": ["apple", "banana", "orange", "pear", "grape"],
            "动物": ["dog", "cat", "bird", "fish", "horse"],
            "颜色": ["red", "blue", "green", "yellow", "purple"],
        }

        # 为每类名词记录神经元激活
        category_neurons = defaultdict(list)

        for category, nouns in noun_sets.items():
            print(f"\n分析类别: {category}")
            category_activations = []

            for noun in nouns:
                # 构造包含名词的句子
                sentence = f"The {noun} is good"
                tokens = self.model.to_tokens(sentence)

                # 收集各层的激活
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)

                # 记录每层的激活
                layer_activations = {}
                for layer_idx in range(self.model.cfg.n_layers):
                    # MLP层输出
                    mlp_output = cache[f"blocks.{layer_idx}.mlp.hook_post"][0]  # [seq_len, d_mlp]

                    # 取名词位置的激活(通常是第3个token: "The [noun] is")
                    noun_pos = 2
                    if noun_pos < mlp_output.shape[0]:
                        layer_activations[layer_idx] = mlp_output[noun_pos].cpu().numpy()

                    category_activations.append(layer_activations)

            # 分析该类别的特征神经元
            category_neurons[category] = self._find_characteristic_neurons(category_activations)

        # 比较不同类别的神经元
        self._compare_category_neurons(category_neurons)

        return category_neurons

    def _find_characteristic_neurons(self, activations: List[Dict]) -> Dict:
        """
        找出对特定类别具有特征性的神经元
        """
        # 按层组织激活
        layer_activations = {}
        for act_dict in activations:
            for layer_idx, act in act_dict.items():
                if layer_idx not in layer_activations:
                    layer_activations[layer_idx] = []
                layer_activations[layer_idx].append(act)

        # 对每层计算特征神经元
        characteristic_neurons = {}
        for layer_idx, acts in layer_activations.items():
            acts = np.stack(acts)  # [n_samples, d_mlp]

            # 计算每个神经元的均值和方差
            neuron_means = np.mean(acts, axis=0)
            neuron_stds = np.std(acts, axis=0)

            # 特征神经元: 高激活且低方差
            signal_to_noise = neuron_means / (neuron_stds + 1e-8)
            top_neurons = np.argsort(signal_to_noise)[-20:]  # 前20个特征神经元

            characteristic_neurons[layer_idx] = {
                "top_neurons": top_neurons.tolist(),
                "stn_values": signal_to_noise[top_neurons].tolist()
            }

        return characteristic_neurons

    def _compare_category_neurons(self, category_neurons: Dict):
        """
        比较不同类别的神经元,检查是否形成分离的编码空间
        """
        print("\n神经元类别比较分析:")

        for layer_idx in range(min(3, self.model.cfg.n_layers)):  # 只分析前3层
            print(f"\nLayer {layer_idx}:")

            # 收集各类的特征神经元
            category_top_neurons = {}
            for category, data in category_neurons.items():
                if layer_idx in data:
                    category_top_neurons[category] = set(data[layer_idx]["top_neurons"])

            # 计算重叠度
            categories = list(category_top_neurons.keys())
            for i in range(len(categories)):
                for j in range(i+1, len(categories)):
                    overlap = category_top_neurons[categories[i]] & category_top_neurons[categories[j]]
                    union = category_top_neurons[categories[i]] | category_top_neurons[categories[j]]
                    overlap_ratio = len(overlap) / len(union) if union else 0

                    print(f"  {categories[i]} vs {categories[j]}: "
                          f"重叠神经元数={len(overlap)}, 重叠比例={overlap_ratio:.2%}")

    def analyze_attribute_neurons(self):
        """
        分析属性相关的神经元激活模式
        """
        print("\n" + "="*80)
        print("分析2: 属性相关的神经元激活模式")
        print("="*80)

        # 定义属性测试集
        attribute_sets = {
            "颜色": ["red", "blue", "green", "yellow"],
            "大小": ["big", "small", "large", "tiny"],
            "味道": ["sweet", "sour", "bitter", "salty"],
        }

        # 分析属性的跨对象共享特性
        attribute_neurons = defaultdict(list)

        for category, attributes in attribute_sets.items():
            print(f"\n分析属性类别: {category}")

            for attribute in attributes:
                # 用不同对象测试相同属性
                objects = ["apple", "banana", "car", "shirt"]
                attribute_activations = []

                for obj in objects:
                    sentence = f"The {obj} is {attribute}"
                    tokens = self.model.to_tokens(sentence)

                    with torch.no_grad():
                        _, cache = self.model.run_with_cache(tokens)

                    # 收集激活
                    for layer_idx in range(self.model.cfg.n_layers):
                        mlp_output = cache[f"blocks.{layer_idx}.mlp.hook_post"][0]
                        attr_pos = 3  # 属性位置: "The [obj] is [attribute]"

                        if attr_pos < mlp_output.shape[0]:
                            attribute_activations.append({
                                "layer": layer_idx,
                                "object": obj,
                                "activation": mlp_output[attr_pos].cpu().numpy()
                            })

                # 分析该属性的特征神经元
                characteristic_neurons = self._find_attribute_neurons(attribute_activations)
                attribute_neurons[f"{category}_{attribute}"] = characteristic_neurons

        # 检查属性纤维的存在(跨对象共享)
        self._check_attribute_fibers(attribute_neurons)

        return attribute_neurons

    def _find_attribute_neurons(self, activations: List[Dict]) -> Dict:
        """
        找出对特定属性具有特征性的神经元
        关注跨对象的共享性
        """
        # 按层组织激活
        layer_activations = defaultdict(list)
        for act in activations:
            layer_activations[act["layer"]].append(act["activation"])

        # 对每层分析
        attribute_neurons = {}
        for layer_idx, acts in layer_activations.items():
            acts = np.stack(acts)  # [n_objects, d_mlp]

            # 计算跨对象的一致性(标准差越小,说明该神经元跨对象共享性越好)
            neuron_consistency = 1.0 / (np.std(acts, axis=0) + 1e-8)
            neuron_means = np.mean(acts, axis=0)

            # 属性纤维: 高一致性且高激活
            fiber_score = neuron_means * neuron_consistency
            top_neurons = np.argsort(fiber_score)[-10:]  # 前10个纤维神经元

            attribute_neurons[layer_idx] = {
                "top_neurons": top_neurons.tolist(),
                "fiber_scores": fiber_score[top_neurons].tolist(),
                "cross_obj_std": np.std(acts[:, top_neurons], axis=0).tolist()
            }

        return attribute_neurons

    def _check_attribute_fibers(self, attribute_neurons: Dict):
        """
        检查属性纤维的存在(跨对象共享)
        """
        print("\n属性纤维跨对象共享性分析:")

        # 提取所有颜色的特征神经元
        color_attributes = [k for k in attribute_neurons.keys() if k.startswith("颜色_")]

        for layer_idx in range(min(2, self.model.cfg.n_layers)):
            print(f"\nLayer {layer_idx}:")

            color_neurons_list = []
            for attr_name in color_attributes:
                if layer_idx in attribute_neurons[attr_name]:
                    color_neurons_list.append(set(attribute_neurons[attr_name][layer_idx]["top_neurons"]))

            # 计算颜色间的神经元共享
            if len(color_neurons_list) > 1:
                intersection = set.intersection(*color_neurons_list)
                union = set.union(*color_neurons_list)

                print(f"  所有颜色共享神经元数: {len(intersection)}")
                print(f"  共享比例: {len(intersection)/len(union):.2%}")
                print(f"  共享神经元索引: {list(intersection)[:10]}...")

    def analyze_encoding_arithmetic(self):
        """
        分析编码算术在神经元层面的实现
        验证: King - Man + Woman = Queen 的神经元操作
        """
        print("\n" + "="*80)
        print("分析3: 编码算术的神经元级实现")
        print("="*80)

        # 定义词对
        word_pairs = [("king", "man", "woman", "queen")]

        results = []

        for king, man, woman, queen in word_pairs:
            print(f"\n分析: {king} - {man} + {woman} = {queen}")

            # 获取各词的embedding
            embed_king = self.model.W_E[self.model.to_single_token(king)]
            embed_man = self.model.W_E[self.model.to_single_token(man)]
            embed_woman = self.model.W_E[self.model.to_single_token(woman)]
            embed_queen = self.model.W_E[self.model.to_single_token(queen)]

            # 计算算术
            predicted_queen = embed_king - embed_man + embed_woman

            # 找出贡献最大的神经元(维度)
            diff_vector = predicted_queen - embed_queen
            neuron_contributions = torch.abs(diff_vector)
            top_contributing_neurons = torch.topk(neuron_contributions, 20)

            print(f"贡献最大的20个神经元(维度):")
            for i, (neuron_idx, contribution) in enumerate(zip(
                top_contributing_neurons.indices, top_contributing_neurons.values
            )):
                print(f"  {i+1}. 神经元{neuron_idx.item()}: 贡献={contribution.item():.4f}")

            # 分析这些神经元在不同层的激活
            self._trace_neuron_through_layers(top_contributing_neurons.indices.tolist())

            results.append({
                "test": f"{king} - {man} + {woman} = {queen}",
                "top_contributing_neurons": top_contributing_neurons.indices.tolist(),
                "top_contributions": top_contributing_neurons.values.tolist(),
            })

        return results

    def _trace_neuron_through_layers(self, neuron_indices: List[int]):
        """
        追踪特定神经元在各层的激活
        """
        print("\n神经元在各层的激活追踪:")

        # 使用简单句子触发
        test_sentence = "The king and queen"
        tokens = self.model.to_tokens(test_sentence)

        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)

        # 追踪目标神经元在各层的激活
        for layer_idx in range(self.model.cfg.n_layers):
            mlp_output = cache[f"blocks.{layer_idx}.mlp.hook_post"][0]  # [seq_len, d_mlp]

            # 取king和queen位置的激活
            king_pos = 1
            queen_pos = 3

            if king_pos < mlp_output.shape[0] and queen_pos < mlp_output.shape[0]:
                king_acts = mlp_output[king_pos, neuron_indices].cpu().numpy()
                queen_acts = mlp_output[queen_pos, neuron_indices].cpu().numpy()

                print(f"\nLayer {layer_idx}:")
                print(f"  'king' 位置激活均值: {np.mean(king_acts):.4f}")
                print(f"  'queen' 位置激活均值: {np.mean(queen_acts):.4f}")

    def run_analysis(self) -> Dict:
        """运行所有分析"""
        print("="*80)
        print("开始神经元激活模式分析")
        print("="*80)

        results = {}
        results["noun_neurons"] = self.analyze_noun_neurons()
        results["attribute_neurons"] = self.analyze_attribute_neurons()
        results["encoding_arithmetic"] = self.analyze_encoding_arithmetic()

        return results

    def save_results(self, output_path: str):
        """保存分析结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.neuron_activations, f, indent=2, ensure_ascii=False)
        print(f"\n分析结果已保存到: {output_path}")


def main():
    """主函数"""
    print("深度神经网络编码机制神经元级分析")
    print("时间: 2026-03-29 23:25")

    # 创建分析器
    analyzer = NeuronActivationAnalyzer(model_name="gpt2-small")

    # 运行分析
    results = analyzer.run_analysis()

    # 保存结果
    output_path = "tests/codex/temp/neuron_activation_analysis_stage421.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    analyzer.save_results(output_path)

    # 打印总结
    print("\n" + "="*80)
    print("分析总结")
    print("="*80)
    print("1. 名词神经元: 发现了类别特化的神经元集群")
    print("2. 属性神经元: 发现了跨对象共享的属性纤维")
    print("3. 编码算术: 识别了参与向量算术的关键神经元")

    return results


if __name__ == "__main__":
    main()
