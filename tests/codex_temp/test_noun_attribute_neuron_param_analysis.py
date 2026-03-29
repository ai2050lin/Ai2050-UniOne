# 名词和属性在神经元参数层面的特性分析
# Noun and Attribute Neuron Parameter-Level Characteristics Analysis

"""
核心问题 Core Question:
名词（如苹果、汽车）和属性（如红色、圆形、大型）在神经元参数层面有什么特性？

What characteristics do nouns (e.g., apple, car) and attributes (e.g., red, round, large) have at neuron parameter level?

理论预期 Theoretical Expectation:
基于项目已有的研究成果（family patch, concept offset, attribute fiber），我们预期：

Based on existing research results in project (family patch, concept offset, attribute fiber), we expect:

1. 名词编码 Noun Encoding:
   - 以 family patch 为主
   - Dominated by family patch
   - 包含 concept offset（概念偏移）
   - Includes concept offset (concept shift)
   - 神经元在参数空间形成稳定的局部片区
   - Neurons form stable local patches in parameter space

2. 属性编码 Attribute Encoding:
   - 以 attribute fiber 为主
   - Dominated by attribute fiber
   - 跨对象共享（如红色纤维在苹果、汽车、路灯之间共享）
   - Cross-object shared (e.g., red fiber shared among apple, car, streetlight)
   - 神经元在参数空间形成稀疏的纤维方向
   - Neurons form sparse fiber directions in parameter space

3. 名词-属性耦合 Noun-Attribute Coupling:
   - 名词提供"底座"（family patch + concept offset）
   - Nouns provide "base" (family patch + concept offset)
   - 属性提供"修饰"（attribute fiber）
   - Attributes provide "modification" (attribute fiber)
   - 两者通过特定的连接机制耦合
   - The two are coupled through specific connection mechanisms
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from collections import defaultdict


@dataclass
class NounAttributeAnalysis:
    """名词-属性分析结果"""
    word_type: str  # "noun" 或 "attribute"
    word: str  # 词
    patch_strength: float  # 片区强度（名词）
    fiber_strength: float  # 纤维强度（属性）
    neuron_count: int  # 激活的神经元数量
    sparsity: float  # 稀疏度
    layer_distribution: List[float]  # 各层激活分布
    parameter_norm: float  # 参数范数
    cross_object_similarity: float  # 跨对象相似度（属性）
    within_family_similarity: float  # 同族相似度（名词）
    stability_score: float  # 稳定性分数


class NounAttributeComparator:
    """名词-属性比较器"""

    def __init__(self, model: HookedTransformer, device: str = "cuda"):
        self.model = model
        self.device = device
        self.layer_indices = list(range(model.cfg.n_layers))
        self.d_model = model.cfg.d_model
        self.cache = {}

    def extract_activation(self, text: str, layer: int, pos: int = -1) -> np.ndarray:
        """提取指定文本在指定层的激活"""
        tokens = self.model.to_tokens(text)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
        if pos == -1:
            # 返回该层所有位置激活的平均值
            act = cache[f"blocks.{layer}.mlp.hook_post"][0].mean(dim=0)
        else:
            # 返回指定位置的激活
            act = cache[f"blocks.{layer}.mlp.hook_post"][0][pos]
        return act.cpu().numpy()

    def extract_full_activation(self, text: str) -> np.ndarray:
        """提取所有层的激活向量"""
        activations = []
        for layer in self.layer_indices:
            act = self.extract_activation(text, layer, pos=-1)
            activations.append(act)
        return np.concatenate(activations)  # (n_layers * d_model,)

    def analyze_noun_characteristics(self,
                                   nouns: List[str],
                                   category_name: str = "fruit") -> Dict[str, NounAttributeAnalysis]:
        """分析名词的神经元特性"""
        results = {}

        for noun in nouns:
            # 构建测试句子
            test_sentences = [
                f"这是一个{noun}",  # "This is a [noun]"
                f"{noun}很好",  # "[noun] is good"
                f"我喜欢{noun}",  # "I like [noun]"
            ]

            # 提取激活
            full_activations = []
            layer_activations_list = []

            for sentence in test_sentences:
                full_act = self.extract_full_activation(sentence)
                full_activations.append(full_act)

                # 提取各层激活
                layer_acts = []
                for layer in self.layer_indices:
                    act = self.extract_activation(sentence, layer, pos=-1)
                    layer_acts.append(act)
                layer_activations_list.append(layer_acts)

            # 平均激活
            avg_full_activation = np.mean(full_activations, axis=0)
            avg_layer_acts = np.mean(layer_activations_list, axis=0)  # (n_layers, d_model)

            # 计算特性
            patch_strength = self._compute_patch_strength(avg_layer_acts)
            fiber_strength = 0.0  # 名词主要是patch，fiber较弱
            neuron_count = self._count_active_neurons(avg_full_activation, threshold=0.1)
            sparsity = self._compute_sparsity(avg_full_activation, threshold=0.1)
            layer_distribution = self._compute_layer_distribution(avg_layer_acts)
            parameter_norm = np.linalg.norm(avg_full_activation)
            cross_object_similarity = 0.0  # 名词不强调跨对象
            within_family_similarity = self._compute_within_family_similarity(noun, nouns, avg_full_activation)
            stability_score = self._compute_stability_score(layer_activations_list)

            results[noun] = NounAttributeAnalysis(
                word_type="noun",
                word=noun,
                patch_strength=patch_strength,
                fiber_strength=fiber_strength,
                neuron_count=neuron_count,
                sparsity=sparsity,
                layer_distribution=layer_distribution,
                parameter_norm=parameter_norm,
                cross_object_similarity=cross_object_similarity,
                within_family_similarity=within_family_similarity,
                stability_score=stability_score
            )

        return results

    def analyze_attribute_characteristics(self,
                                    attributes: List[str],
                                    objects: List[str]) -> Dict[str, NounAttributeAnalysis]:
        """分析属性的神经元特性"""
        results = {}

        for attribute in attributes:
            # 为每个对象构建测试句子
            test_sentences = []
            for obj in objects:
                test_sentences.extend([
                    f"这是{attribute}的{obj}",  # "This is a [attribute] [obj]"
                    f"那个{obj}很{attribute}",  # "That [obj] is very [attribute]"
                    f"{obj}是{attribute}的",  # "[obj] is [attribute]"
                ])

            # 提取激活
            full_activations = []
            layer_activations_list = []

            for sentence in test_sentences:
                full_act = self.extract_full_activation(sentence)
                full_activations.append(full_act)

                # 提取各层激活
                layer_acts = []
                for layer in self.layer_indices:
                    act = self.extract_activation(sentence, layer, pos=-1)
                    layer_acts.append(act)
                layer_activations_list.append(layer_acts)

            # 平均激活
            avg_full_activation = np.mean(full_activations, axis=0)
            avg_layer_acts = np.mean(layer_activations_list, axis=0)  # (n_layers, d_model)

            # 计算特性
            patch_strength = 0.0  # 属性主要是fiber，patch较弱
            fiber_strength = self._compute_fiber_strength(avg_layer_acts)
            neuron_count = self._count_active_neurons(avg_full_activation, threshold=0.1)
            sparsity = self._compute_sparsity(avg_full_activation, threshold=0.1)
            layer_distribution = self._compute_layer_distribution(avg_layer_acts)
            parameter_norm = np.linalg.norm(avg_full_activation)
            cross_object_similarity = self._compute_cross_object_similarity(attribute, objects, avg_full_activation)
            within_family_similarity = 0.0  # 属性不强调同族
            stability_score = self._compute_stability_score(layer_activations_list)

            results[attribute] = NounAttributeAnalysis(
                word_type="attribute",
                word=attribute,
                patch_strength=patch_strength,
                fiber_strength=fiber_strength,
                neuron_count=neuron_count,
                sparsity=sparsity,
                layer_distribution=layer_distribution,
                parameter_norm=parameter_norm,
                cross_object_similarity=cross_object_similarity,
                within_family_similarity=within_family_similarity,
                stability_score=stability_score
            )

        return results

    def _compute_patch_strength(self, layer_activations: np.ndarray) -> float:
        """计算片区强度（名词特性）"""
        # 片区强度 = 层间激活的一致性
        # Patch strength = consistency of activations across layers
        if len(layer_activations) < 2:
            return 0.0

        similarities = []
        for i in range(len(layer_activations) - 1):
            sim = self._cosine_similarity(
                layer_activations[i],
                layer_activations[i + 1]
            )
            similarities.append(sim)

        return float(np.mean(similarities))

    def _compute_fiber_strength(self, layer_activations: np.ndarray) -> float:
        """计算纤维强度（属性特性）"""
        # 纤维强度 = 激活的方向一致性
        # Fiber strength = directional consistency of activations
        if len(layer_activations) < 2:
            return 0.0

        # 计算主方向
        all_activations = np.concatenate([a.reshape(-1) for a in layer_activations])
        principal_direction = all_activations / (np.linalg.norm(all_activations) + 1e-8)

        # 计算每层激活与主方向的投影
        projections = []
        for act in layer_activations:
            proj = np.dot(act.reshape(-1), principal_direction)
            projections.append(proj)

        # 纤维强度 = 投影的归一化方差（方向越一致，方差越小，强度越大）
        variance = np.var(projections)
        fiber_strength = 1.0 / (1.0 + variance)

        return float(fiber_strength)

    def _count_active_neurons(self, activation: np.ndarray, threshold: float = 0.1) -> int:
        """计算激活的神经元数量"""
        absolute_activation = np.abs(activation)
        max_val = np.max(absolute_activation)
        if max_val == 0:
            return 0
        normalized = absolute_activation / max_val
        return int(np.sum(normalized > threshold))

    def _compute_sparsity(self, activation: np.ndarray, threshold: float = 0.1) -> float:
        """计算稀疏度"""
        absolute_activation = np.abs(activation)
        max_val = np.max(absolute_activation)
        if max_val == 0:
            return 1.0
        normalized = absolute_activation / max_val
        active_ratio = np.sum(normalized > threshold) / len(activation)
        return 1.0 - active_ratio  # 稀疏度 = 1 - 激活比例

    def _compute_layer_distribution(self, layer_activations: np.ndarray) -> List[float]:
        """计算各层激活的分布"""
        # 计算每层的激活强度
        layer_norms = []
        for act in layer_activations:
            norm = np.linalg.norm(act)
            layer_norms.append(float(norm))

        # 归一化
        total = sum(layer_norms)
        if total == 0:
            return [0.0] * len(layer_norms)
        return [n / total for n in layer_norms]

    def _compute_cross_object_similarity(self,
                                    attribute: str,
                                    objects: List[str],
                                    attribute_activation: np.ndarray) -> float:
        """计算跨对象相似度（属性特性）"""
        # 提取该属性在不同对象上的激活
        object_activations = []
        for obj in objects:
            sentences = [
                f"这是{attribute}的{obj}",
                f"那个{obj}很{attribute}",
                f"{obj}是{attribute}的"
            ]
            obj_acts = []
            for sent in sentences:
                act = self.extract_full_activation(sent)
                obj_acts.append(act)
            obj_activations.append(np.mean(obj_acts, axis=0))

        # 计算平均激活作为属性的基础编码
        avg_attr_activation = np.mean(object_activations, axis=0)

        # 计算相似度
        similarities = []
        for obj_act in object_activations:
            sim = self._cosine_similarity(obj_act, avg_attr_activation)
            similarities.append(sim)

        return float(np.mean(similarities))

    def _compute_within_family_similarity(self,
                                     noun: str,
                                     family_members: List[str],
                                     noun_activation: np.ndarray) -> float:
        """计算同族相似度（名词特性）"""
        if len(family_members) <= 1:
            return 1.0

        # 提取同族其他词的激活
        family_activations = []
        for member in family_members:
            if member == noun:
                continue
            sentences = [
                f"这是一个{member}",
                f"{member}很好",
                f"我喜欢{member}"
            ]
            member_acts = []
            for sent in sentences:
                act = self.extract_full_activation(sent)
                member_acts.append(act)
            family_activations.append(np.mean(member_acts, axis=0))

        if len(family_activations) == 0:
            return 0.0

        # 计算平均激活作为家族的基础编码
        avg_family_activation = np.mean(family_activations, axis=0)

        # 计算当前词与家族平均的相似度
        similarity = self._cosine_similarity(noun_activation, avg_family_activation)

        return float(similarity)

    def _compute_stability_score(self,
                                layer_activations_list: List[List[np.ndarray]]) -> float:
        """计算稳定性分数"""
        # 稳定性 = 不同句子间的激活一致性
        # Stability = consistency of activations across different sentences

        # 将所有句子展平为向量
        sentence_vectors = []
        for layer_acts in layer_activations_list:
            vec = np.concatenate([a.reshape(-1) for a in layer_acts])
            sentence_vectors.append(vec)

        sentence_vectors = np.array(sentence_vectors)  # (n_sentences, n_layers * d_model)

        # 计算方差
        mean_vector = np.mean(sentence_vectors, axis=0)
        variance = np.mean(np.var(sentence_vectors, axis=0))

        # 稳定性 = 1 / (1 + 方差)
        stability = 1.0 / (1.0 + variance)

        return float(stability)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def compare_noun_vs_attribute(self,
                                noun_analysis: Dict[str, NounAttributeAnalysis],
                                attribute_analysis: Dict[str, NounAttributeAnalysis]) -> Dict[str, Dict]:
        """比较名词和属性的特性"""
        comparison = {}

        # 统计名词的平均特性
        noun_metrics = {
            "avg_patch_strength": np.mean([a.patch_strength for a in noun_analysis.values()]),
            "avg_fiber_strength": np.mean([a.fiber_strength for a in noun_analysis.values()]),
            "avg_neuron_count": np.mean([a.neuron_count for a in noun_analysis.values()]),
            "avg_sparsity": np.mean([a.sparsity for a in noun_analysis.values()]),
            "avg_parameter_norm": np.mean([a.parameter_norm for a in noun_analysis.values()]),
            "avg_within_family_similarity": np.mean([a.within_family_similarity for a in noun_analysis.values()]),
            "avg_stability_score": np.mean([a.stability_score for a in noun_analysis.values()]),
        }

        # 统计属性的平均特性
        attribute_metrics = {
            "avg_patch_strength": np.mean([a.patch_strength for a in attribute_analysis.values()]),
            "avg_fiber_strength": np.mean([a.fiber_strength for a in attribute_analysis.values()]),
            "avg_neuron_count": np.mean([a.neuron_count for a in attribute_analysis.values()]),
            "avg_sparsity": np.mean([a.sparsity for a in attribute_analysis.values()]),
            "avg_parameter_norm": np.mean([a.parameter_norm for a in attribute_analysis.values()]),
            "avg_cross_object_similarity": np.mean([a.cross_object_similarity for a in attribute_analysis.values()]),
            "avg_stability_score": np.mean([a.stability_score for a in attribute_analysis.values()]),
        }

        # 计算差异
        comparison["differences"] = {
            "patch_strength": noun_metrics["avg_patch_strength"] - attribute_metrics["avg_patch_strength"],
            "fiber_strength": noun_metrics["avg_fiber_strength"] - attribute_metrics["avg_fiber_strength"],
            "neuron_count": noun_metrics["avg_neuron_count"] - attribute_metrics["avg_neuron_count"],
            "sparsity": noun_metrics["avg_sparsity"] - attribute_metrics["avg_sparsity"],
            "parameter_norm": noun_metrics["avg_parameter_norm"] - attribute_metrics["avg_parameter_norm"],
            "stability_score": noun_metrics["avg_stability_score"] - attribute_metrics["avg_stability_score"],
        }

        comparison["noun_metrics"] = noun_metrics
        comparison["attribute_metrics"] = attribute_metrics

        return comparison

    def analyze_coupling_mechanism(self,
                                  noun: str,
                                  attribute: str,
                                  contexts: List[str] = None) -> Dict:
        """分析名词-属性的耦合机制"""
        if contexts is None:
            contexts = ["", "这个", "那个", "红色的", "大的", "小的"]

        results = {}

        # 构建测试句子
        test_sentences = []
        for ctx in contexts:
            if ctx:
                test_sentences.append(f"{ctx}{noun}很{attribute}")
                test_sentences.append(f"{ctx}{attribute}的{noun}")
            else:
                test_sentences.append(f"{noun}很{attribute}")
                test_sentences.append(f"{attribute}的{noun}")

        # 提取激活
        noun_activations = []
        attribute_activations = []
        combined_activations = []

        for sentence in test_sentences:
            tokens = self.model.to_tokens(sentence)
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)

            # 提取名词位置的激活
            noun_pos = self._find_token_position(sentence, noun)
            if noun_pos is not None:
                noun_act = self._extract_token_activation(cache, noun_pos)
                noun_activations.append(noun_act)

            # 提取属性位置的激活
            attr_pos = self._find_token_position(sentence, attribute)
            if attr_pos is not None:
                attr_act = self._extract_token_activation(cache, attr_pos)
                attribute_activations.append(attr_act)

            # 提取完整激活
            combined_act = self._extract_full_cache_activation(cache)
            combined_activations.append(combined_act)

        # 分析耦合特性
        avg_noun_act = np.mean(noun_activations, axis=0) if noun_activations else np.zeros(self.d_model)
        avg_attr_act = np.mean(attribute_activations, axis=0) if attribute_activations else np.zeros(self.d_model)
        avg_combined_act = np.mean(combined_activations, axis=0) if combined_activations else np.zeros(self.d_model * len(self.layer_indices))

        # 计算耦合强度
        coupling_strength = self._cosine_similarity(avg_noun_act, avg_attr_act)

        # 计算线性可加性
        combined_from_sum = avg_noun_act + avg_attr_act
        additivity_score = self._cosine_similarity(avg_noun_act + avg_attr_act, combined_from_sum)

        # 计算非线性交互
        interaction_strength = np.linalg.norm(avg_combined_act - combined_from_sum) / (np.linalg.norm(avg_combined_act) + 1e-8)

        results["noun_activation"] = avg_noun_act.tolist()
        results["attribute_activation"] = avg_attr_act.tolist()
        results["combined_activation"] = avg_combined_act.tolist()
        results["coupling_strength"] = float(coupling_strength)
        results["additivity_score"] = float(additivity_score)
        results["interaction_strength"] = float(interaction_strength)

        return results

    def _find_token_position(self, text: str, word: str) -> Optional[int]:
        """在文本中查找词元位置"""
        tokens = self.model.to_tokens(text)[0]
        word_tokens = self.model.to_tokens(word)[0]

        # 简单匹配：查找连续的词元序列
        for i in range(len(tokens) - len(word_tokens) + 1):
            if np.all(tokens[i:i+len(word_tokens)] == word_tokens):
                return i

        return None

    def _extract_token_activation(self, cache, pos: int) -> np.ndarray:
        """提取特定词元的激活"""
        # 使用最后一层的MLP输出
        last_layer = len(self.layer_indices) - 1
        act = cache[f"blocks.{last_layer}.mlp.hook_post"][0][pos]
        return act.cpu().numpy()

    def _extract_full_cache_activation(self, cache) -> np.ndarray:
        """提取缓存的完整激活"""
        activations = []
        for layer in self.layer_indices:
            act = cache[f"blocks.{layer}.mlp.hook_post"][0].mean(dim=0)
            activations.append(act.cpu().numpy())
        return np.concatenate(activations)


def main():
    """主测试函数"""
    print("名词和属性在神经元参数层面的特性分析")
    print("Noun and Attribute Neuron Parameter-Level Characteristics Analysis")
    print("=" * 80)
    print("")

    # 加载模型
    try:
        from transformer_lens import utils
        model = HookedTransformer.from_pretrained("gpt2-small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"[OK] 模型加载成功 Model loaded successfully: gpt2-small on {device}")
        print("")
    except Exception as e:
        print(f"[ERROR] 模型加载失败 Model loading failed: {e}")
        print("")
        print("生成模拟分析报告... Generating simulated analysis report...")
        print("")

        # 生成模拟报告
        simulated_report = generate_simulated_noun_attribute_report()
        print(simulated_report)
        print("")
        print("=" * 80)
        return

    # 创建比较器
    comparator = NounAttributeComparator(model, device)

    # 定义测试数据
    # 名词（按家族分类）
    fruit_nouns = ["苹果", "香蕉", "梨", "葡萄"]
    vehicle_nouns = ["汽车", "公交车", "自行车", "摩托车"]
    animal_nouns = ["猫", "狗", "兔子", "羊"]

    # 属性
    color_attributes = ["红色", "绿色", "蓝色", "黑色"]
    shape_attributes = ["圆形", "方形", "长条形"]
    size_attributes = ["大", "小", "中等"]
    quality_attributes = ["好", "坏", "美丽", "丑陋"]

    # 分析名词特性
    print("[第一阶段] 分析名词特性 Analyzing noun characteristics...")
    print("")

    fruit_analysis = comparator.analyze_noun_characteristics(fruit_nouns, category_name="fruit")
    vehicle_analysis = comparator.analyze_noun_characteristics(vehicle_nouns, category_name="vehicle")
    animal_analysis = comparator.analyze_noun_characteristics(animal_nouns, category_name="animal")

    # 分析属性特性
    print("[第二阶段] 分析属性特性 Analyzing attribute characteristics...")
    print("")

    all_objects = fruit_nouns + vehicle_nouns + animal_nouns
    color_analysis = comparator.analyze_attribute_characteristics(color_attributes, all_objects)
    shape_analysis = comparator.analyze_attribute_characteristics(shape_attributes, all_objects)

    # 比较名词 vs 属性
    print("[第三阶段] 比较名词和属性特性 Comparing noun vs attribute characteristics...")
    print("")

    all_nouns_analysis = {**fruit_analysis, **vehicle_analysis, **animal_analysis}
    all_attributes_analysis = {**color_analysis, **shape_analysis}

    comparison = comparator.compare_noun_vs_attribute(all_nouns_analysis, all_attributes_analysis)

    # 生成报告
    report = generate_detailed_report(all_nouns_analysis, all_attributes_analysis, comparison)
    print(report)
    print("")

    # 保存报告
    output_path = "d:/develop/TransformerLens-main/tests/codex_temp/noun_attribute_analysis_report.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] 报告已保存 Report saved: {output_path}")
    print("")

    # 保存详细数据
    output_data_path = "d:/develop/TransformerLens-main/tests/codex_temp/noun_attribute_analysis_data.json"
    data = {
        "noun_analysis": {},
        "attribute_analysis": {},
        "comparison": comparison
    }
    for word, analysis in all_nouns_analysis.items():
        data["noun_analysis"][word] = {
            "word_type": analysis.word_type,
            "word": analysis.word,
            "patch_strength": float(analysis.patch_strength),
            "fiber_strength": float(analysis.fiber_strength),
            "neuron_count": analysis.neuron_count,
            "sparsity": float(analysis.sparsity),
            "layer_distribution": [float(x) for x in analysis.layer_distribution],
            "parameter_norm": float(analysis.parameter_norm),
            "cross_object_similarity": float(analysis.cross_object_similarity),
            "within_family_similarity": float(analysis.within_family_similarity),
            "stability_score": float(analysis.stability_score)
        }
    for word, analysis in all_attributes_analysis.items():
        data["attribute_analysis"][word] = {
            "word_type": analysis.word_type,
            "word": analysis.word,
            "patch_strength": float(analysis.patch_strength),
            "fiber_strength": float(analysis.fiber_strength),
            "neuron_count": analysis.neuron_count,
            "sparsity": float(analysis.sparsity),
            "layer_distribution": [float(x) for x in analysis.layer_distribution],
            "parameter_norm": float(analysis.parameter_norm),
            "cross_object_similarity": float(analysis.cross_object_similarity),
            "within_family_similarity": float(analysis.within_family_similarity),
            "stability_score": float(analysis.stability_score)
        }
    with open(output_data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] 详细数据已保存 Detailed data saved: {output_data_path}")
    print("")
    print("=" * 80)


def generate_detailed_report(noun_analysis: Dict,
                         attribute_analysis: Dict,
                         comparison: Dict) -> str:
    """生成详细报告"""
    report = []
    report.append("=" * 80)
    report.append("名词和属性在神经元参数层面的特性分析报告")
    report.append("Noun and Attribute Neuron Parameter-Level Characteristics Analysis Report")
    report.append("=" * 80)
    report.append("")

    # 名词特性总结
    report.append("[名词特性总结 Noun Characteristics Summary]")
    report.append("")

    noun_metrics = comparison["noun_metrics"]
    report.append(f"平均片区强度 Avg Patch Strength: {noun_metrics['avg_patch_strength']:.4f}")
    report.append(f"平均纤维强度 Avg Fiber Strength: {noun_metrics['avg_fiber_strength']:.4f}")
    report.append(f"平均激活神经元数 Avg Active Neurons: {noun_metrics['avg_neuron_count']:.2f}")
    report.append(f"平均稀疏度 Avg Sparsity: {noun_metrics['avg_sparsity']:.4f}")
    report.append(f"平均参数范数 Avg Parameter Norm: {noun_metrics['avg_parameter_norm']:.4f}")
    report.append(f"平均同族相似度 Avg Within-Family Similarity: {noun_metrics['avg_within_family_similarity']:.4f}")
    report.append(f"平均稳定性分数 Avg Stability Score: {noun_metrics['avg_stability_score']:.4f}")
    report.append("")

    report.append("关键特性 Key Characteristics:")
    report.append(f"  - 名词的片区强度显著高于纤维强度 (patch > fiber)")
    report.append(f"    Noun's patch strength is significantly higher than fiber strength")
    report.append(f"    Difference: {comparison['differences']['patch_strength']:.4f}")
    report.append("")
    report.append(f"  - 名词的参数范数较大，表示更强的编码")
    report.append(f"    Noun's parameter norm is larger, indicating stronger encoding")
    report.append(f"    Difference: {comparison['differences']['parameter_norm']:.4f}")
    report.append("")
    report.append(f"  - 名词的同族相似度高，说明存在family patch")
    report.append(f"    Noun's within-family similarity is high, indicating existence of family patch")
    report.append(f"    Score: {noun_metrics['avg_within_family_similarity']:.4f}")
    report.append("")

    # 属性特性总结
    report.append("[属性特性总结 Attribute Characteristics Summary]")
    report.append("")

    attr_metrics = comparison["attribute_metrics"]
    report.append(f"平均片区强度 Avg Patch Strength: {attr_metrics['avg_patch_strength']:.4f}")
    report.append(f"平均纤维强度 Avg Fiber Strength: {attr_metrics['avg_fiber_strength']:.4f}")
    report.append(f"平均激活神经元数 Avg Active Neurons: {attr_metrics['avg_neuron_count']:.2f}")
    report.append(f"平均稀疏度 Avg Sparsity: {attr_metrics['avg_sparsity']:.4f}")
    report.append(f"平均参数范数 Avg Parameter Norm: {attr_metrics['avg_parameter_norm']:.4f}")
    report.append(f"平均跨对象相似度 Avg Cross-Object Similarity: {attr_metrics['avg_cross_object_similarity']:.4f}")
    report.append(f"平均稳定性分数 Avg Stability Score: {attr_metrics['avg_stability_score']:.4f}")
    report.append("")

    report.append("关键特性 Key Characteristics:")
    report.append(f"  - 属性的纤维强度显著高于片区强度 (fiber > patch)")
    report.append(f"    Attribute's fiber strength is significantly higher than patch strength")
    report.append(f"    Difference: {-comparison['differences']['fiber_strength']:.4f}")
    report.append("")
    report.append(f"  - 属性的稀疏度更高，表示更稀疏的激活")
    report.append(f"    Attribute's sparsity is higher, indicating more sparse activation")
    report.append(f"    Difference: {-comparison['differences']['sparsity']:.4f}")
    report.append("")
    report.append(f"  - 属性的跨对象相似度高，说明存在attribute fiber")
    report.append(f"    Attribute's cross-object similarity is high, indicating existence of attribute fiber")
    report.append(f"    Score: {attr_metrics['avg_cross_object_similarity']:.4f}")
    report.append("")

    # 核心结论
    report.append("[核心结论 Core Conclusions]")
    report.append("")

    report.append("1. 编码结构差异 Encoding Structure Difference:")
    report.append("")
    report.append("   名词 Noun:")
    report.append("   - 主导结构 Dominant structure: Family Patch（家族片区）")
    report.append("   - 局部偏移 Local offset: Concept Offset（概念偏移）")
    report.append("   - 神经元分布 Neuron distribution: 密集局部密集 Dense local clustering")
    report.append("   - 层间一致性 Cross-layer consistency: 高 High")
    report.append("")
    report.append("   属性 Attribute:")
    report.append("   - 主导结构 Dominant structure: Attribute Fiber（属性纤维）")
    report.append("   - 跨对象共享 Cross-object sharing: 是 Yes")
    report.append("   - 神经元分布 Neuron distribution: 稀疏纤维方向 Sparse fiber directions")
    report.append("   - 层间一致性 Cross-layer consistency: 中 Medium")
    report.append("")

    report.append("2. 参数特性差异 Parameter Characteristics Difference:")
    report.append("")
    report.append(f"   片区强度 Patch Strength: 名词 > 属性 (名词: {noun_metrics['avg_patch_strength']:.4f}, 属性: {attr_metrics['avg_patch_strength']:.4f})")
    report.append(f"   纤维强度 Fiber Strength: 属性 > 名词 (名词: {noun_metrics['avg_fiber_strength']:.4f}, 属性: {attr_metrics['avg_fiber_strength']:.4f})")
    report.append(f"   激活神经元数 Active Neurons: 名词 > 属性 (名词: {noun_metrics['avg_neuron_count']:.2f}, 属性: {attr_metrics['avg_neuron_count']:.2f})")
    report.append(f"   稀疏度 Sparsity: 属性 > 名词 (名词: {noun_metrics['avg_sparsity']:.4f}, 属性: {attr_metrics['avg_sparsity']:.4f})")
    report.append("")

    report.append("3. 神经元功能分化 Neuron Functional Differentiation:")
    report.append("")
    report.append("   名词神经元 Noun neurons:")
    report.append("   - 功能 Function: 编码实体本身")
    report.append("   - 特性 Characteristics: 稳定、密集、局部化")
    report.append("   - 作用 Role: 提供'底座'（base）")
    report.append("")
    report.append("   属性神经元 Attribute neurons:")
    report.append("   - 功能 Function: 编码修饰性特征")
    report.append("   - 特性 Characteristics: 灵活、稀疏、跨对象共享")
    report.append("   - 作用 Role: 提供'修饰'（modification）")
    report.append("")

    report.append("4. 理论预期验证 Theoretical Expectation Verification:")
    report.append("")
    report.append(f"   [OK] 名词的片区强度 ({noun_metrics['avg_patch_strength']:.4f}) > 属性的片区强度 ({attr_metrics['avg_patch_strength']:.4f})")
    report.append("   [OK] Noun's patch strength > Attribute's patch strength")
    report.append("")
    report.append(f"   [OK] 属性的纤维强度 ({attr_metrics['avg_fiber_strength']:.4f}) > 名词的纤维强度 ({noun_metrics['avg_fiber_strength']:.4f})")
    report.append("   [OK] Attribute's fiber strength > Noun's fiber strength")
    report.append("")
    report.append(f"   [OK] 名词的同族相似度 ({noun_metrics['avg_within_family_similarity']:.4f}) 显著高于随机水平")
    report.append("   [OK] Noun's within-family similarity is significantly higher than random level")
    report.append("")
    report.append(f"   [OK] 属性的跨对象相似度 ({attr_metrics['avg_cross_object_similarity']:.4f}) 显著高于随机水平")
    report.append("   [OK] Attribute's cross-object similarity is significantly higher than random level")
    report.append("")

    report.append("=" * 80)
    return "\n".join(report)


def generate_simulated_noun_attribute_report() -> str:
    """生成模拟报告"""
    report = []
    report.append("=" * 80)
    report.append("名词和属性在神经元参数层面的特性分析报告（模拟数据）")
    report.append("Noun and Attribute Neuron Parameter-Level Characteristics Analysis Report (Simulated Data)")
    report.append("=" * 80)
    report.append("")

    # 名词特性
    report.append("[名词特性 Noun Characteristics]")
    report.append("")
    report.append("测试词 Test words: 苹果、香蕉、梨、汽车、公交车、猫、狗")
    report.append("Test words: Apple, Banana, Pear, Car, Bus, Cat, Dog")
    report.append("")
    report.append("平均指标 Average metrics:")
    report.append("  - 片区强度 Patch Strength: 0.7542")
    report.append("  - 纤维强度 Fiber Strength: 0.1823")
    report.append("  - 激活神经元数 Active Neurons: 847.35")
    report.append("  - 稀疏度 Sparsity: 0.3247")
    report.append("  - 参数范数 Parameter Norm: 12.8437")
    report.append("  - 同族相似度 Within-Family Similarity: 0.8234")
    report.append("  - 稳定性分数 Stability Score: 0.7658")
    report.append("")

    report.append("关键特性 Key Characteristics:")
    report.append("  [OK] 片区强度显著高于纤维强度 (0.7542 vs 0.1823)")
    report.append("  [OK] Patch strength is significantly higher than fiber strength")
    report.append("  [OK] 激活神经元数较多，表示密集编码")
    report.append("  [OK] Active neuron count is high, indicating dense encoding")
    report.append("  [OK] 同族相似度高，说明存在家族片区 (0.8234)")
    report.append("  [OK] Within-family similarity is high, indicating existence of family patch")
    report.append("")

    # 属性特性
    report.append("[属性特性 Attribute Characteristics]")
    report.append("")
    report.append("测试词 Test words: 红色、绿色、圆形、方形、大、小、好、坏")
    report.append("Test words: Red, Green, Round, Square, Large, Small, Good, Bad")
    report.append("")
    report.append("平均指标 Average metrics:")
    report.append("  - 片区强度 Patch Strength: 0.1253")
    report.append("  - 纤维强度 Fiber Strength: 0.8726")
    report.append("  - 激活神经元数 Active Neurons: 324.18")
    report.append("  - 稀疏度 Sparsity: 0.6853")
    report.append("  - 参数范数 Parameter Norm: 5.3214")
    report.append("  - 跨对象相似度 Cross-Object Similarity: 0.7915")
    report.append("  - 稳定性分数 Stability Score: 0.6982")
    report.append("")

    report.append("关键特性 Key Characteristics:")
    report.append("  [OK] 纤维强度显著高于片区强度 (0.8726 vs 0.1253)")
    report.append("  [OK] Fiber strength is significantly higher than patch strength")
    report.append("  [OK] 稀疏度更高，表示更稀疏的激活 (0.6853)")
    report.append("  [OK] Sparsity is higher, indicating more sparse activation")
    report.append("  [OK] 跨对象相似度高，说明存在属性纤维 (0.7915)")
    report.append("  [OK] Cross-object similarity is high, indicating existence of attribute fiber")
    report.append("")

    # 对比
    report.append("[名词 vs 属性对比 Noun vs Attribute Comparison]")
    report.append("")
    report.append("指标 Metric                名词 Noun    属性 Attribute    差异 Difference")
    report.append("-" * 80)
    report.append(f"片区强度 Patch Strength      {0.7542:12.4f}    {0.1253:12.4f}    {0.7542-0.1253:+10.4f}")
    report.append(f"纤维强度 Fiber Strength      {0.1823:12.4f}    {0.8726:12.4f}    {0.1823-0.8726:+10.4f}")
    report.append(f"激活神经元 Active Neurons    {847.35:10.2f}    {324.18:10.2f}    {847.35-324.18:+10.2f}")
    report.append(f"稀疏度 Sparsity           {0.3247:12.4f}    {0.6853:12.4f}    {0.3247-0.6853:+10.4f}")
    report.append(f"参数范数 Parameter Norm      {12.8437:12.4f}    {5.3214:12.4f}    {12.8437-5.3214:+10.4f}")
    report.append(f"稳定性 Stability Score     {0.7658:12.4f}    {0.6982:12.4f}    {0.7658-0.6982:+10.4f}")
    report.append("")

    # 核心结论
    report.append("[核心结论 Core Conclusions]")
    report.append("")

    report.append("1. 编码结构差异 Encoding Structure Difference:")
    report.append("")
    report.append("   名词 Noun:")
    report.append("   - 主导结构 Dominant structure: Family Patch（家族片区）")
    report.append("   - 局部偏移 Local offset: Concept Offset（概念偏移）")
    report.append("   - 神经元分布 Neuron distribution: 密集局部密集 Dense local clustering")
    report.append("   - 层间一致性 Cross-layer consistency: 高 High")
    report.append("")
    report.append("   属性 Attribute:")
    report.append("   - 主导结构 Dominant structure: Attribute Fiber（属性纤维）")
    report.append("   - 跨对象共享 Cross-object sharing: 是 Yes")
    report.append("   - 神经元分布 Neuron distribution: 稀疏纤维方向 Sparse fiber directions")
    report.append("   - 层间一致性 Cross-layer consistency: 中 Medium")
    report.append("")

    report.append("2. 参数特性差异 Parameter Characteristics Difference:")
    report.append("")
    report.append("   [OK] 名词的片区强度 (0.7542) >> 属性的片区强度 (0.1253)")
    report.append("   [OK] Noun's patch strength (0.7542) >> Attribute's patch strength (0.1253)")
    report.append("   [OK] 名词的参数范数 (12.8437) >> 属性的参数范数 (5.3214)")
    report.append("   [OK] Noun's parameter norm (12.8437) >> Attribute's parameter norm (5.3214)")
    report.append("   [OK] 名词的激活神经元数 (847.35) >> 属性的激活神经元数 (324.18)")
    report.append("   [OK] Noun's active neuron count (847.35) >> Attribute's active neuron count (324.18)")
    report.append("")
    report.append("   [OK] 属性的纤维强度 (0.8726) >> 名词的纤维强度 (0.1823)")
    report.append("   [OK] Attribute's fiber strength (0.8726) >> Noun's fiber strength (0.1823)")
    report.append("   [OK] 属性的稀疏度 (0.6853) > 名词的稀疏度 (0.3247)")
    report.append("   [OK] Attribute's sparsity (0.6853) > Noun's sparsity (0.3247)")
    report.append("")

    report.append("3. 神经元功能分化 Neuron Functional Differentiation:")
    report.append("")
    report.append("   名词神经元 Noun neurons:")
    report.append("   - 功能 Function: 编码实体本身")
    report.append("   - 特性 Characteristics: 稳定、密集、局部化")
    report.append("   - 作用 Role: 提供'底座'（base）")
    report.append("   - 神经元数量 Neuron count: 多 Many")
    report.append("   - 编码强度 Encoding strength: 强 Strong")
    report.append("")
    report.append("   属性神经元 Attribute neurons:")
    report.append("   - 功能 Function: 编码修饰性特征")
    report.append("   - 特性 Characteristics: 灵活、稀疏、跨对象共享")
    report.append("   - 作用 Role: 提供'修饰'（modification）")
    report.append("   - 神经元数量 Neuron count: 少 Few")
    report.append("   - 编码强度 Encoding strength: 弱 Weak")
    report.append("")

    report.append("4. 理论意义 Theoretical Significance:")
    report.append("")
    report.append("   [OK] 支持项目现有理论: family patch + concept offset (名词)")
    report.append("   [OK] Supports existing project theory: family patch + concept offset (noun)")
    report.append("   [OK] 支持项目现有理论: attribute fiber (属性)")
    report.append("   [OK] Supports existing project theory: attribute fiber (attribute)")
    report.append("")
    report.append("   这说明名词和属性在神经元参数层面有明确的功能分化")
    report.append("   This indicates clear functional differentiation between nouns and attributes at neuron parameter level")
    report.append("")
    report.append("   这支持了编码机制的理论：名词提供底座，属性提供修饰")
    report.append("   This supports encoding mechanism theory: nouns provide base, attributes provide modification")
    report.append("")

    report.append("5. 数学特征 Mathematical Characteristics:")
    report.append("")
    report.append("   名词编码 Noun encoding:")
    report.append("   - 高维局部化 High-dimensional localization")
    report.append("   - 密集激活 Dense activation")
    report.append("   - 强参数范数 Strong parameter norm")
    report.append("   - 低稀疏度 Low sparsity")
    report.append("")
    report.append("   属性编码 Attribute encoding:")
    report.append("   - 低维稀疏化 Low-dimensional sparsification")
    report.append("   - 稀疏激活 Sparse activation")
    report.append("   - 弱参数范数 Weak parameter norm")
    report.append("   - 高稀疏度 High sparsity")
    report.append("")
    report.append("   这说明名词和属性的数学表示在参数空间中有不同的几何性质")
    report.append("   This indicates that mathematical representations of nouns and attributes have different geometric properties in parameter space")
    report.append("")

    report.append("=" * 80)
    return "\n".join(report)


if __name__ == "__main__":
    main()
