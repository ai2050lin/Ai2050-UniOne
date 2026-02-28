"""
四特性评估器
============

评估DNN编码的四个核心特性:
1. 高维抽象: 类内距离 / 类间距离
2. 低维精确: 低维探针准确率
3. 特异性: 概念正交性
4. 系统性: 类比推理能力
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 高维抽象参数
    abstraction_categories: Dict[str, List[str]] = None
    
    # 低维精确参数
    probe_dims: List[int] = None
    
    # 特异性参数
    specificity_concepts: List[str] = None
    
    # 系统性参数
    analogy_pairs: List[Tuple[str, str, str, str]] = None
    
    def __post_init__(self):
        if self.abstraction_categories is None:
            self.abstraction_categories = {
                "动物": ["狗", "猫", "鸟", "鱼", "马", "牛", "羊", "猪"],
                "颜色": ["红", "蓝", "绿", "黄", "黑", "白", "紫", "橙"],
                "数字": ["一", "二", "三", "四", "五", "六", "七", "八"],
            }
        
        if self.probe_dims is None:
            self.probe_dims = [4, 8, 16, 32]
        
        if self.specificity_concepts is None:
            self.specificity_concepts = [
                "苹果", "汽车", "房子", "电脑", "音乐", "数学",
                "快乐", "悲伤", "大", "小", "快", "慢"
            ]
        
        if self.analogy_pairs is None:
            self.analogy_pairs = [
                ("国王", "女王", "男人", "女人"),
                ("巴黎", "法国", "伦敦", "英国"),
                ("大", "小", "高", "矮"),
                ("快", "慢", "早", "晚"),
            ]


class FourPropertiesEvaluator:
    """
    四特性评估器
    
    评估DNN内部编码的核心特性
    """
    
    def __init__(
        self,
        model,
        config: Optional[EvaluationConfig] = None
    ):
        """
        Args:
            model: Transformer模型
            config: 评估配置
        """
        self.model = model
        self.config = config or EvaluationConfig()
        
        # 缓存激活
        self._activation_cache = {}
    
    def get_activation(
        self,
        text: str,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        获取文本的激活向量
        
        Args:
            text: 输入文本
            layer_idx: 层索引 (-1表示最后一层)
        
        Returns:
            activation: [d_model]
        """
        if layer_idx == -1:
            layer_idx = self.model.cfg.n_layers - 1
        
        cache_key = f"{text}_{layer_idx}"
        if cache_key in self._activation_cache:
            return self._activation_cache[cache_key]
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                text,
                names_filter=lambda x: f"blocks.{layer_idx}.hook_resid_post" in x
            )
            activation = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1, :].cpu()
        
        self._activation_cache[cache_key] = activation
        return activation
    
    def evaluate_all(
        self,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        评估所有四特性
        
        Args:
            layer_indices: 要评估的层列表
        
        Returns:
            完整评估结果
        """
        if layer_indices is None:
            layer_indices = [self.model.cfg.n_layers // 2]  # 中间层
        
        results = {
            "config": {
                "probe_dims": self.config.probe_dims,
                "categories": list(self.config.abstraction_categories.keys()),
            },
            "layers": {}
        }
        
        for layer_idx in layer_indices:
            logging.info(f"\n=== Evaluating Layer {layer_idx} ===")
            
            layer_results = {}
            
            # 1. 高维抽象
            logging.info("  Evaluating High-Dim Abstraction...")
            layer_results["abstraction"] = self.eval_high_dim_abstraction(layer_idx)
            
            # 2. 低维精确
            logging.info("  Evaluating Low-Dim Precision...")
            layer_results["precision"] = self.eval_low_dim_precision(layer_idx)
            
            # 3. 特异性
            logging.info("  Evaluating Specificity...")
            layer_results["specificity"] = self.eval_specificity(layer_idx)
            
            # 4. 系统性
            logging.info("  Evaluating Systematicity...")
            layer_results["systematicity"] = self.eval_systematicity(layer_idx)
            
            # 综合评分
            layer_results["overall_score"] = self._compute_overall_score(layer_results)
            
            results["layers"][layer_idx] = layer_results
        
        return results
    
    def eval_high_dim_abstraction(
        self,
        layer_idx: int
    ) -> Dict[str, Any]:
        """
        高维抽象评估
        
        指标: 类内距离 / 类间距离
        
        解释:
        - 高比率 (>2.0): 良好的抽象能力
        - 低比率 (<1.5): 抽象能力不足
        
        物理意义:
        - 类内距离小: 同类概念编码相似
        - 类间距离大: 不同类概念编码差异大
        """
        categories = self.config.abstraction_categories
        
        # 收集各类别的激活
        category_activations = {}
        for cat_name, words in categories.items():
            activations = []
            for word in words:
                act = self.get_activation(word, layer_idx)
                activations.append(act)
            category_activations[cat_name] = torch.stack(activations)
        
        # 计算类内距离
        intra_dists = []
        for cat_name, acts in category_activations.items():
            # 计算类别内两两距离
            dists = torch.cdist(acts, acts)
            # 排除对角线 (自己到自己的距离)
            mask = ~torch.eye(len(acts), dtype=torch.bool)
            intra_dists.append(dists[mask].mean().item())
        
        avg_intra = np.mean(intra_dists)
        
        # 计算类间距离
        inter_dists = []
        cat_names = list(category_activations.keys())
        for i, cat1 in enumerate(cat_names):
            for cat2 in cat_names[i+1:]:
                acts1 = category_activations[cat1]
                acts2 = category_activations[cat2]
                
                # 计算两个类别之间的距离
                dists = torch.cdist(acts1, acts2)
                inter_dists.append(dists.mean().item())
        
        avg_inter = np.mean(inter_dists)
        
        # 计算比率
        ratio = avg_inter / (avg_intra + 1e-8)
        
        return {
            "intra_distance": float(avg_intra),
            "inter_distance": float(avg_inter),
            "ratio": float(ratio),
            "passed": ratio > 2.0,
            "interpretation": self._interpret_abstraction(ratio)
        }
    
    def _interpret_abstraction(self, ratio: float) -> str:
        """解释抽象性指标"""
        if ratio > 3.0:
            return "优秀的抽象能力: 类别分离度高"
        elif ratio > 2.0:
            return "良好的抽象能力: 类别区分明显"
        elif ratio > 1.5:
            return "中等抽象能力: 类别有部分重叠"
        else:
            return "抽象能力不足: 类别区分度低"
    
    def eval_low_dim_precision(
        self,
        layer_idx: int,
        task: str = "category_classification"
    ) -> Dict[str, Any]:
        """
        低维精确评估
        
        指标: 在低维投影后，线性探针的分类准确率
        
        解释:
        - 高准确率 (>90% @ k=8): 精确编码
        - 中等准确率 (70-90%): 部分精确
        - 低准确率 (<70%): 编码不够精确
        
        物理意义:
        - 高准确率意味着激活向量包含足够信息
        - 低维就能提取，说明编码高效
        """
        # 准备数据
        if task == "category_classification":
            X, y = self._prepare_category_data(layer_idx)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        if len(X) == 0:
            return {"error": "No data available"}
        
        # 转换为numpy
        X_np = X.cpu().numpy()
        y_np = np.array(y)
        
        results = {}
        
        for k_dim in self.config.probe_dims:
            # PCA降维
            pca = PCA(n_components=min(k_dim, X_np.shape[1], X_np.shape[0]))
            X_reduced = pca.fit_transform(X_np)
            
            # 训练线性探针
            X_train, X_test, y_train, y_test = train_test_split(
                X_reduced, y_np, test_size=0.2, random_state=42
            )
            
            probe = LogisticRegression(max_iter=1000)
            probe.fit(X_train, y_train)
            
            accuracy = probe.score(X_test, y_test)
            
            results[f"k={k_dim}"] = float(accuracy)
        
        # 判断是否通过
        passed = results.get("k=8", 0) > 0.9
        
        return {
            "accuracies": results,
            "best_accuracy": max(results.values()),
            "passed": passed,
            "interpretation": self._interpret_precision(results)
        }
    
    def _prepare_category_data(
        self,
        layer_idx: int
    ) -> Tuple[torch.Tensor, List[int]]:
        """准备分类数据"""
        X = []
        y = []
        
        categories = self.config.abstraction_categories
        
        for label, (cat_name, words) in enumerate(categories.items()):
            for word in words:
                act = self.get_activation(word, layer_idx)
                X.append(act)
                y.append(label)
        
        return torch.stack(X), y
    
    def _interpret_precision(self, results: Dict[str, float]) -> str:
        """解释精确性指标"""
        k8_acc = results.get("k=8", 0)
        
        if k8_acc > 0.95:
            return "极高精确度: 低维即可完美分类"
        elif k8_acc > 0.9:
            return "高精确度: 低维分类准确"
        elif k8_acc > 0.8:
            return "中等精确度: 需要稍高维度"
        else:
            return "低精确度: 编码不够高效"
    
    def eval_specificity(
        self,
        layer_idx: int
    ) -> Dict[str, Any]:
        """
        特异性评估
        
        指标: 不同概念编码的正交性
        
        解释:
        - 高正交性 (>0.7): 不同概念编码清晰区分
        - 中等正交性 (0.5-0.7): 有一定区分度
        - 低正交性 (<0.5): 概念编码混淆
        
        物理意义:
        - 正交性高 = 特异性强
        - 每个概念有独特的编码模式
        """
        concepts = self.config.specificity_concepts
        
        # 获取所有概念的激活
        activations = []
        for concept in concepts:
            act = self.get_activation(concept, layer_idx)
            activations.append(act)
        
        activations = torch.stack(activations)
        
        # 归一化
        norms = torch.norm(activations, dim=1, keepdim=True)
        activations_norm = activations / (norms + 1e-10)
        
        # 计算内积矩阵
        inner_products = activations_norm @ activations_norm.T
        
        # 提取上三角 (不含对角线)
        upper_tri = inner_products[
            torch.triu_indices(len(concepts), len(concepts), offset=1)
        ]
        
        # 正交性 = 1 - |平均内积|
        orthogonality = 1.0 - torch.abs(upper_tri).mean().item()
        
        return {
            "num_concepts": len(concepts),
            "orthogonality": float(orthogonality),
            "mean_inner_product": float(torch.abs(upper_tri).mean()),
            "max_inner_product": float(torch.abs(upper_tri).max()),
            "passed": orthogonality > 0.7,
            "interpretation": self._interpret_specificity(orthogonality)
        }
    
    def _interpret_specificity(self, orthogonality: float) -> str:
        """解释特异性指标"""
        if orthogonality > 0.8:
            return "极高特异性: 概念编码完全正交"
        elif orthogonality > 0.7:
            return "高特异性: 概念编码清晰区分"
        elif orthogonality > 0.5:
            return "中等特异性: 概念有一定区分"
        else:
            return "低特异性: 概念编码混淆"
    
    def eval_systematicity(
        self,
        layer_idx: int
    ) -> Dict[str, Any]:
        """
        系统性评估
        
        指标: 类比推理准确率
        
        任务: A:B :: C:? 
        方法: vec(D) ≈ vec(C) + vec(B) - vec(A)
        
        解释:
        - 高准确率 (>70%): 良好的系统性
        - 中等准确率 (50-70%): 部分系统性
        - 低准确率 (<50%): 系统性不足
        
        物理意义:
        - 系统性 = 统一的操作规则
        - 相同的关系用相同的向量方向编码
        """
        analogy_pairs = self.config.analogy_pairs
        
        correct = 0
        total = len(analogy_pairs)
        
        for a, b, c, d_true in analogy_pairs:
            # 获取向量
            vec_a = self.get_activation(a, layer_idx)
            vec_b = self.get_activation(b, layer_idx)
            vec_c = self.get_activation(c, layer_idx)
            
            # 计算预测向量
            vec_d_pred = vec_c + vec_b - vec_a
            
            # 在概念列表中找最近邻
            concepts = self.config.specificity_concepts
            best_match = None
            best_sim = -float('inf')
            
            for concept in concepts:
                vec_concept = self.get_activation(concept, layer_idx)
                sim = torch.cosine_similarity(
                    vec_d_pred.unsqueeze(0),
                    vec_concept.unsqueeze(0)
                ).item()
                
                if sim > best_sim:
                    best_sim = sim
                    best_match = concept
            
            if best_match == d_true:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": float(accuracy),
            "correct": correct,
            "total": total,
            "passed": accuracy > 0.7,
            "interpretation": self._interpret_systematicity(accuracy)
        }
    
    def _interpret_systematicity(self, accuracy: float) -> str:
        """解释系统性指标"""
        if accuracy > 0.8:
            return "高系统性: 类比推理能力强"
        elif accuracy > 0.7:
            return "良好系统性: 类比推理基本成功"
        elif accuracy > 0.5:
            return "中等系统性: 部分类比可行"
        else:
            return "低系统性: 类比推理困难"
    
    def _compute_overall_score(
        self,
        layer_results: Dict[str, Any]
    ) -> float:
        """
        计算综合评分
        
        权重:
        - 高维抽象: 0.25
        - 低维精确: 0.25
        - 特异性: 0.25
        - 系统性: 0.25
        """
        scores = []
        
        # 高维抽象
        abstraction_ratio = layer_results["abstraction"]["ratio"]
        abstraction_score = min(1.0, abstraction_ratio / 3.0)  # 3.0以上满分
        scores.append(abstraction_score)
        
        # 低维精确
        precision_acc = layer_results["precision"].get("accuracies", {}).get("k=8", 0)
        scores.append(precision_acc)
        
        # 特异性
        specificity = layer_results["specificity"]["orthogonality"]
        scores.append(specificity)
        
        # 系统性
        systematicity = layer_results["systematicity"]["accuracy"]
        scores.append(systematicity)
        
        # 加权平均
        return float(np.mean(scores))


# ============================================================
# 测试代码
# ============================================================

def test_four_properties():
    """测试四特性评估"""
    import sys
    sys.path.insert(0, "d:/ai2050/TransformerLens-Project")
    
    from transformer_lens import HookedTransformer
    
    # 加载模型
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small")
    
    # 创建评估器
    evaluator = FourPropertiesEvaluator(model)
    
    # 评估
    print("\nRunning four properties evaluation...")
    results = evaluator.evaluate_all(layer_indices=[6, 11])
    
    # 打印结果
    print("\n=== Results ===")
    for layer_idx, layer_results in results["layers"].items():
        print(f"\n--- Layer {layer_idx} ---")
        print(f"Overall Score: {layer_results['overall_score']:.3f}")
        
        print(f"\n1. High-Dim Abstraction:")
        print(f"   Ratio: {layer_results['abstraction']['ratio']:.2f}")
        print(f"   {layer_results['abstraction']['interpretation']}")
        
        print(f"\n2. Low-Dim Precision:")
        for k, acc in layer_results['precision']['accuracies'].items():
            print(f"   {k}: {acc:.2%}")
        print(f"   {layer_results['precision']['interpretation']}")
        
        print(f"\n3. Specificity:")
        print(f"   Orthogonality: {layer_results['specificity']['orthogonality']:.3f}")
        print(f"   {layer_results['specificity']['interpretation']}")
        
        print(f"\n4. Systematicity:")
        print(f"   Accuracy: {layer_results['systematicity']['accuracy']:.2%}")
        print(f"   {layer_results['systematicity']['interpretation']}")
    
    return results


if __name__ == "__main__":
    test_four_properties()
