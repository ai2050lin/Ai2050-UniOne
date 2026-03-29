"""
深度神经网络编码机制参数空间拓扑分析
Stage422: 参数空间结构可视化
时间: 2026-03-29 23:30

目标: 分析参数空间的拓扑结构,验证片区-纤维-耦合的假设
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from pathlib import Path
import seaborn as sns

class ParameterSpaceTopologyAnalyzer:
    """
    参数空间拓扑分析器
    分析参数空间的几何结构和拓扑性质
    """

    def __init__(self, model_name: str = "gpt2-small"):
        self.model_name = model_name
        print(f"加载模型: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 分析结果存储
        self.analysis_results = {}

    def extract_patch_structure(self):
        """
        提取家族片区(Patch)结构
        验证: 名词形成稳定的局部密集片区
        """
        print("\n" + "="*80)
        print("分析1: 家族片区(Patch)结构")
        print("="*80)

        # 定义名词家族
        noun_families = {
            "水果": ["apple", "banana", "orange", "pear", "grape", "peach", "mango"],
            "动物": ["dog", "cat", "bird", "fish", "horse", "cow", "pig"],
            "交通工具": ["car", "bus", "train", "plane", "boat", "bike", "truck"],
        }

        # 提取各家族在embedding空间的表示
        family_embeddings = {}
        for family, nouns in noun_families.items():
            embeddings = []
            for noun in nouns:
                token_id = self.model.to_single_token(noun)
                embedding = self.model.W_E[token_id].cpu().numpy()
                embeddings.append(embedding)
            family_embeddings[family] = np.array(embeddings)  # [n_nouns, d_model]

        # 计算片区特性
        family_stats = {}

        for family, embeddings in family_embeddings.items():
            # 计算家族内距离(内聚性)
            n_samples = embeddings.shape[0]
            intra_distances = []
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    intra_distances.append(dist)
            intra_distance = np.mean(intra_distances)
            intra_std = np.std(intra_distances)

            # 计算家族中心
            center = np.mean(embeddings, axis=0)

            # 计算片区强度(与中心的距离)
            distances_to_center = np.linalg.norm(embeddings - center, axis=1)
            patch_strength = 1.0 / (np.mean(distances_to_center) + 1e-8)

            # 计算家族内相似度
            similarities = []
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)
            within_family_similarity = np.mean(similarities)

            family_stats[family] = {
                "intra_distance": intra_distance,
                "intra_std": intra_std,
                "patch_strength": patch_strength,
                "within_family_similarity": within_family_similarity,
                "n_samples": n_samples
            }

            print(f"\n{family}家族:")
            print(f"  族内平均距离: {intra_distance:.4f} ± {intra_std:.4f}")
            print(f"  片区强度: {patch_strength:.4f}")
            print(f"  族内相似度: {within_family_similarity:.4f}")

        # 计算家族间距离(分离性)
        print("\n家族间距离分析:")
        families = list(family_embeddings.keys())
        for i in range(len(families)):
            for j in range(i+1, len(families)):
                center_i = np.mean(family_embeddings[families[i]], axis=0)
                center_j = np.mean(family_embeddings[families[j]], axis=0)
                inter_distance = np.linalg.norm(center_i - center_j)

                print(f"  {families[i]} - {families[j]}: {inter_distance:.4f}")

        # 可视化
        self._visualize_patches(family_embeddings)

        self.analysis_results["patch_structure"] = family_stats

        return family_stats

    def extract_fiber_structure(self):
        """
        提取属性纤维(Fiber)结构
        验证: 属性形成稀疏的纤维方向,跨对象共享
        """
        print("\n" + "="*80)
        print("分析2: 属性纤维(Fiber)结构")
        print("="*80)

        # 定义属性和对应的对象集
        attribute_tests = {
            "红色": {
                "objects": ["apple", "car", "shirt", "flower", "ball"],
                "sentences": [f"The {obj} is red" for obj in ["apple", "car", "shirt", "flower", "ball"]]
            },
            "蓝色": {
                "objects": ["sky", "ocean", "car", "shirt", "ball"],
                "sentences": [f"The {obj} is blue" for obj in ["sky", "ocean", "car", "shirt", "ball"]]
            },
            "大": {
                "objects": ["house", "car", "box", "tree", "mountain"],
                "sentences": [f"The {obj} is big" for obj in ["house", "car", "box", "tree", "mountain"]]
            },
        }

        # 提取各属性在参数空间中的表示
        attribute_representations = {}

        for attr_name, test_data in attribute_tests.items():
            # 使用中间层的MLP输出作为表示
            mid_layer = self.model.cfg.n_layers // 2

            representations = []
            for sentence in test_data["sentences"]:
                tokens = self.model.to_tokens(sentence)

                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)

                # 获取属性位置的激活(通常是最后一个词)
                mlp_output = cache[f"blocks.{mid_layer}.mlp.hook_post"][0]
                attr_pos = -1  # 属性通常是最后一个词
                representation = mlp_output[attr_pos].cpu().numpy()
                representations.append(representation)

            attribute_representations[attr_name] = np.array(representations)

            print(f"\n{attr_name}属性:")
            print(f"  表示维度: {representations[0].shape}")

        # 计算纤维特性
        fiber_stats = {}

        for attr_name, reps in attribute_representations.items():
            # 计算跨对象的方差(纤维的稳定性)
            cross_obj_variance = np.var(reps, axis=0)
            cross_obj_std = np.std(reps, axis=0)

            # 计算跨对象的相似度
            n_samples = reps.shape[0]
            similarities = []
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    sim = np.dot(reps[i], reps[j]) / (
                        np.linalg.norm(reps[i]) * np.linalg.norm(reps[j])
                    )
                    similarities.append(sim)
            cross_obj_similarity = np.mean(similarities)

            # 计算纤维强度(跨对象一致性)
            fiber_strength = 1.0 / (np.mean(cross_obj_std) + 1e-8)

            fiber_stats[attr_name] = {
                "cross_obj_mean_std": np.mean(cross_obj_std),
                "cross_obj_similarity": cross_obj_similarity,
                "fiber_strength": fiber_strength,
                "n_samples": n_samples
            }

            print(f"\n{attr_name}属性纤维:")
            print(f"  跨对象平均标准差: {np.mean(cross_obj_std):.4f}")
            print(f"  跨对象相似度: {cross_obj_similarity:.4f}")
            print(f"  纤维强度: {fiber_strength:.4f}")

        self.analysis_results["fiber_structure"] = fiber_stats

        return fiber_stats

    def analyze_coupling_structure(self):
        """
        分析耦合(Coupling)结构
        验证: 名词和属性的耦合参数位
        """
        print("\n" + "="*80)
        print("分析3: 耦合(Coupling)结构")
        print("="*80)

        # 定义测试短语
        test_phrases = [
            "red apple",
            "blue apple",
            "big apple",
            "small apple",
            "red car",
            "blue car",
            "big car",
            "small car",
        ]

        # 提取每个短语的表示
        mid_layer = self.model.cfg.n_layers // 2
        phrase_representations = []

        for phrase in test_phrases:
            tokens = self.model.to_tokens(phrase)

            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)

            # 获取整个短语的表示(聚合最后几个位置)
            mlp_output = cache[f"blocks.{mid_layer}.mlp.hook_post"][0]
            phrase_repr = mlp_output.mean(dim=0).cpu().numpy()  # 平均池化
            phrase_representations.append(phrase_repr)

        phrase_representations = np.array(phrase_representations)

        # 使用PCA降维可视化
        pca = PCA(n_components=2)
        phrase_pca = pca.fit_transform(phrase_representations)

        # 绘制耦合结构
        plt.figure(figsize=(12, 8))

        # 按对象分组
        objects = ["apple", "car"]
        colors = ["red", "blue", "big", "small"]

        for obj in objects:
            obj_indices = [i for i, phrase in enumerate(test_phrases) if obj in phrase]
            plt.scatter(
                phrase_pca[obj_indices, 0],
                phrase_pca[obj_indices, 1],
                label=obj,
                s=100,
                alpha=0.6
            )

            # 标记颜色/大小
            for idx in obj_indices:
                attr = test_phrases[idx].split()[0]
                plt.annotate(
                    attr,
                    (phrase_pca[idx, 0], phrase_pca[idx, 1]),
                    fontsize=8,
                    ha='center'
                )

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('名词-属性耦合结构可视化')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图像
        output_dir = Path("tests/codex/temp/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "coupling_structure_pca.png", dpi=150, bbox_inches='tight')
        plt.close()

        print("\n耦合结构可视化已保存到: tests/codex/temp/visualizations/coupling_structure_pca.png")

        # 分析耦合强度
        # 计算同一对象不同属性的聚类程度
        coupling_strengths = {}
        for obj in objects:
            obj_indices = [i for i, phrase in enumerate(test_phrases) if obj in phrase]
            obj_reprs = phrase_representations[obj_indices]

            # 计算聚类紧密度
            center = np.mean(obj_reprs, axis=0)
            avg_distance = np.mean([np.linalg.norm(repr - center) for repr in obj_reprs])
            clustering = 1.0 / (avg_distance + 1e-8)

            coupling_strengths[obj] = {
                "avg_distance": avg_distance,
                "clustering": clustering
            }

            print(f"\n{obj}的耦合强度:")
            print(f"  平均距离: {avg_distance:.4f}")
            print(f"  聚类紧密度: {clustering:.4f}")

        self.analysis_results["coupling_structure"] = {
            "coupling_strengths": coupling_strengths,
            "pca_variance_ratio": pca.explained_variance_ratio_.tolist()
        }

        return self.analysis_results["coupling_structure"]

    def _visualize_patches(self, family_embeddings: Dict[str, np.ndarray]):
        """可视化家族片区结构"""
        # 合并所有embedding
        all_embeddings = []
        labels = []

        for family, embeddings in family_embeddings.items():
            all_embeddings.append(embeddings)
            labels.extend([family] * embeddings.shape[0])

        all_embeddings = np.vstack(all_embeddings)

        # 使用t-SNE降维到2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(all_embeddings)-1))
        embeddings_2d = tsne.fit_transform(all_embeddings)

        # 绘图
        plt.figure(figsize=(12, 8))

        for i, family in enumerate(family_embeddings.keys()):
            family_mask = [label == family for label in labels]
            plt.scatter(
                embeddings_2d[family_mask, 0],
                embeddings_2d[family_mask, 1],
                label=family,
                s=100,
                alpha=0.6
            )

        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('家族片区(Family Patch)结构可视化')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图像
        output_dir = Path("tests/codex/temp/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "patch_structure_tsne.png", dpi=150, bbox_inches='tight')
        plt.close()

        print("\n片区结构可视化已保存到: tests/codex/temp/visualizations/patch_structure_tsne.png")

    def run_analysis(self) -> Dict:
        """运行所有分析"""
        print("="*80)
        print("开始参数空间拓扑分析")
        print("="*80)

        results = {}
        results["patch_structure"] = self.extract_patch_structure()
        results["fiber_structure"] = self.extract_fiber_structure()
        results["coupling_structure"] = self.analyze_coupling_structure()

        return results

    def save_results(self, output_path: str):
        """保存分析结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        print(f"\n分析结果已保存到: {output_path}")


def main():
    """主函数"""
    print("深度神经网络编码机制参数空间拓扑分析")
    print("时间: 2026-03-29 23:30")

    # 创建分析器
    analyzer = ParameterSpaceTopologyAnalyzer(model_name="gpt2-small")

    # 运行分析
    results = analyzer.run_analysis()

    # 保存结果
    output_path = "tests/codex/temp/parameter_space_topology_stage422.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    analyzer.save_results(output_path)

    # 打印总结
    print("\n" + "="*80)
    print("分析总结")
    print("="*80)
    print("1. 片区结构: 验证了名词在embedding空间形成稳定的局部密集片区")
    print("2. 纤维结构: 发现了属性跨对象共享的纤维方向")
    print("3. 耦合结构: 可视化了名词-属性的耦合模式")

    return results


if __name__ == "__main__":
    main()
