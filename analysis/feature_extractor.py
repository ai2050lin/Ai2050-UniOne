"""
特征提取器
==========

从训练好的DNN中提取特征编码

核心方法:
1. 激活提取: 从每层提取激活向量
2. 稀疏自编码器: 提取稀疏特征
3. 探针分析: 测试特征可线性提取性
4. 内在维度: 测量流形复杂度
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class ExtractionConfig:
    """特征提取配置"""
    # 模型参数
    model_name: str = "gpt2-small"
    target_layers: List[int] = None  # 默认所有层
    
    # 稀疏自编码器参数
    sae_latent_dim: int = 4096  # SAE潜在维度 (过完备)
    sae_sparsity_penalty: float = 0.01
    
    # 采样参数
    batch_size: int = 32
    num_samples: int = 1000
    
    # 保存路径
    output_dir: str = "results/feature_extraction"
    
    def __post_init__(self):
        if self.target_layers is None:
            self.target_layers = [0, 3, 6, 9, 11]  # 默认采样这些层


class SparseAutoencoder(nn.Module):
    """
    稀疏自编码器 (Sparse Autoencoder, SAE)
    
    目的: 从激活向量中提取稀疏特征
    
    数学形式:
    encode: h = ReLU(W_e @ x + b_e)
    decode: x̂ = W_d @ h + b_d
    loss = ||x - x̂||² + λ * ||h||₁
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        sparsity_penalty: float = 0.01
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_penalty = sparsity_penalty
        
        # 编码器
        self.encoder = nn.Linear(input_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # 解码器权重初始化为编码器转置
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码"""
        return torch.relu(self.encoder(x))
    
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """解码"""
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        h = self.encode(x)
        x_recon = self.decode(h)
        return x_recon, h
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """计算损失"""
        # 重建损失
        recon_loss = torch.mean((x - x_recon) ** 2)
        
        # 稀疏损失 (L1)
        sparsity_loss = torch.mean(torch.abs(h))
        
        # 总损失
        total_loss = recon_loss + self.sparsity_penalty * sparsity_loss
        
        return total_loss, recon_loss, sparsity_loss
    
    def train_sae(
        self,
        activations: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str = "cuda"
    ) -> Dict[str, List[float]]:
        """
        训练SAE
        
        Args:
            activations: [N, input_dim] 激活矩阵
            epochs: 训练轮数
            lr: 学习率
            batch_size: 批次大小
        
        Returns:
            训练历史
        """
        self.to(device)
        activations = activations.to(device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        history = {
            "total_loss": [],
            "recon_loss": [],
            "sparsity_loss": []
        }
        
        N = activations.shape[0]
        
        for epoch in range(epochs):
            # 打乱数据
            perm = torch.randperm(N)
            activations = activations[perm]
            
            epoch_losses = []
            
            for i in range(0, N, batch_size):
                batch = activations[i:i+batch_size]
                
                # 前向
                x_recon, h = self.forward(batch)
                
                # 计算损失
                total_loss, recon_loss, sparsity_loss = self.compute_loss(batch, x_recon, h)
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            # 记录
            history["total_loss"].append(np.mean(epoch_losses))
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: loss = {np.mean(epoch_losses):.4f}")
        
        return history
    
    def get_features(self) -> torch.Tensor:
        """
        获取学到的特征
        
        Returns:
            features: [latent_dim, input_dim] 特征矩阵
        """
        # decoder.weight shape: [input_dim, latent_dim], need transpose
        return self.decoder.weight.data.T  # 每行是一个特征


class FeatureExtractor:
    """
    特征提取器
    
    从训练好的模型中提取特征编码
    """
    
    def __init__(
        self,
        model,
        config: Optional[ExtractionConfig] = None
    ):
        """
        Args:
            model: Transformer模型 (HookedTransformer)
            config: 提取配置
        """
        self.model = model
        self.config = config or ExtractionConfig()
        
        # 存储结果
        self.activations = {}
        self.sae_models = {}
        self.features = {}
    
    def extract_activations(
        self,
        texts: List[str],
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        提取激活向量
        
        Args:
            texts: 文本列表
            layer_idx: 目标层 (None则提取所有目标层)
        
        Returns:
            activations: [N, d_model] 或 Dict[int, [N, d_model]]
        """
        self.model.eval()
        
        if layer_idx is not None:
            # 提取单层
            return self._extract_single_layer(texts, layer_idx)
        else:
            # 提取所有目标层
            all_activations = {}
            for layer in self.config.target_layers:
                all_activations[layer] = self._extract_single_layer(texts, layer)
            return all_activations
    
    def _extract_single_layer(
        self,
        texts: List[str],
        layer_idx: int
    ) -> torch.Tensor:
        """提取单层激活"""
        activations = []
        
        with torch.no_grad():
            for text in texts:
                try:
                    # 运行模型并获取缓存
                    _, cache = self.model.run_with_cache(
                        text,
                        names_filter=lambda x: f"blocks.{layer_idx}.hook_resid_post" in x
                    )
                    
                    # 提取最后一个token的激活
                    act = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1, :]
                    activations.append(act.cpu())
                    
                except Exception as e:
                    logging.warning(f"Error extracting from '{text[:50]}...': {e}")
                    continue
        
        if not activations:
            return torch.zeros(0, self.model.cfg.d_model)
        
        return torch.stack(activations)
    
    def train_sparse_autoencoder(
        self,
        activations: torch.Tensor,
        layer_idx: int,
        epochs: int = 100
    ) -> SparseAutoencoder:
        """
        为特定层训练稀疏自编码器
        
        Args:
            activations: [N, d_model] 激活矩阵
            layer_idx: 层索引
            epochs: 训练轮数
        
        Returns:
            训练好的SAE
        """
        input_dim = activations.shape[1]
        
        logging.info(f"Training SAE for layer {layer_idx}...")
        logging.info(f"  Input dim: {input_dim}")
        logging.info(f"  Latent dim: {self.config.sae_latent_dim}")
        logging.info(f"  Samples: {activations.shape[0]}")
        
        sae = SparseAutoencoder(
            input_dim=input_dim,
            latent_dim=self.config.sae_latent_dim,
            sparsity_penalty=self.config.sae_sparsity_penalty
        )
        
        # 训练
        history = sae.train_sae(
            activations,
            epochs=epochs,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 存储
        self.sae_models[layer_idx] = sae
        self.features[layer_idx] = sae.get_features()
        
        return sae
    
    def compute_intrinsic_dimension(
        self,
        activations: torch.Tensor,
        method: str = "mle"
    ) -> float:
        """
        计算内在维度 (Intrinsic Dimension, ID)
        
        ID反映了激活流形的复杂度
        
        Args:
            activations: [N, d] 激活矩阵
            method: 计算方法
        
        Returns:
            intrinsic_dim: 内在维度
        """
        # 转换为numpy
        X = activations.cpu().numpy()
        
        if method == "mle":
            # 最大似然估计
            # 参考: Levina & Bickel (2004)
            return self._id_mle(X)
        elif method == "pca":
            # PCA方法: 解释95%方差所需的维度
            pca = PCA()
            pca.fit(X)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            return np.searchsorted(cumsum, 0.95) + 1
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _id_mle(self, X: np.ndarray) -> float:
        """MLE方法估计内在维度"""
        from sklearn.neighbors import NearestNeighbors
        
        n_neighbors = min(20, X.shape[0] - 1)
        
        # 计算k近邻距离
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # 只使用最近的邻居
        distances = distances[:, 1:]  # 排除自己
        
        # MLE估计
        # d = 1 / (mean(log(r_k/r_{k-1})))
        ratios = distances[:, 1:] / (distances[:, :-1] + 1e-10)
        log_ratios = np.log(ratios + 1e-10)
        
        id_estimate = 1.0 / (np.mean(log_ratios) + 1e-10)
        
        return float(id_estimate)
    
    def analyze_feature_sparsity(
        self,
        features: torch.Tensor,
        threshold: float = 0.01
    ) -> Dict[str, float]:
        """
        分析特征的稀疏性
        
        Args:
            features: [latent_dim, input_dim] 特征矩阵
            threshold: 稀疏阈值
        
        Returns:
            稀疏性指标
        """
        features_np = features.cpu().numpy()
        
        # 1. L0稀疏度: 非零元素比例
        l0_sparsity = np.mean(np.abs(features_np) > threshold)
        
        # 2. L1稀疏度: 平均绝对值
        l1_sparsity = np.mean(np.abs(features_np))
        
        # 3. Gini系数: 不均匀程度
        gini = self._compute_gini(np.abs(features_np).flatten())
        
        return {
            "l0_sparsity": l0_sparsity,
            "l1_sparsity": l1_sparsity,
            "gini_coefficient": gini
        }
    
    def _compute_gini(self, x: np.ndarray) -> float:
        """计算Gini系数"""
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x) + 1e-10)
    
    def analyze_feature_orthogonality(
        self,
        features: torch.Tensor
    ) -> Dict[str, float]:
        """
        分析特征的正交性
        
        Args:
            features: [latent_dim, input_dim] 特征矩阵
        
        Returns:
            正交性指标
        """
        features_np = features.cpu().numpy()
        
        # 计算特征向量之间的内积
        # 归一化
        norms = np.linalg.norm(features_np, axis=1, keepdims=True)
        features_norm = features_np / (norms + 1e-10)
        
        # 内积矩阵
        inner_products = features_norm @ features_norm.T
        
        # 提取上三角 (不含对角线)
        upper_tri = inner_products[np.triu_indices_from(inner_products, k=1)]
        
        return {
            "mean_inner_product": float(np.mean(np.abs(upper_tri))),
            "std_inner_product": float(np.std(upper_tri)),
            "max_inner_product": float(np.max(np.abs(upper_tri))),
            "orthogonality_score": 1.0 - float(np.mean(np.abs(upper_tri)))
        }
    
    def run_full_extraction(
        self,
        texts: List[str],
        train_sae: bool = True,
        epochs: int = 100
    ) -> Dict[str, Any]:
        """
        运行完整的特征提取流程
        
        Args:
            texts: 文本列表
            train_sae: 是否训练SAE
            epochs: SAE训练轮数
        
        Returns:
            完整分析结果
        """
        results = {
            "config": {
                "model_name": self.config.model_name,
                "target_layers": self.config.target_layers,
                "sae_latent_dim": self.config.sae_latent_dim
            },
            "layers": {}
        }
        
        # 1. 提取所有层的激活
        logging.info("Step 1: Extracting activations...")
        all_activations = self.extract_activations(texts)
        
        for layer_idx, activations in all_activations.items():
            logging.info(f"\n=== Processing Layer {layer_idx} ===")
            
            layer_results = {
                "num_samples": activations.shape[0],
                "activation_dim": activations.shape[1]
            }
            
            # 2. 计算内在维度
            logging.info("  Computing intrinsic dimension...")
            layer_results["intrinsic_dimension"] = self.compute_intrinsic_dimension(activations)
            
            # 3. 训练SAE
            if train_sae:
                logging.info("  Training Sparse Autoencoder...")
                sae = self.train_sparse_autoencoder(activations, layer_idx, epochs)
                
                # 4. 分析特征
                features = self.features[layer_idx]
                
                logging.info("  Analyzing feature sparsity...")
                layer_results["sparsity"] = self.analyze_feature_sparsity(features)
                
                logging.info("  Analyzing feature orthogonality...")
                layer_results["orthogonality"] = self.analyze_feature_orthogonality(features)
            
            results["layers"][layer_idx] = layer_results
        
        # 5. 跨层分析
        logging.info("\n=== Cross-Layer Analysis ===")
        results["cross_layer"] = self._cross_layer_analysis()
        
        return results
    
    def _cross_layer_analysis(self) -> Dict[str, Any]:
        """跨层分析"""
        if len(self.features) < 2:
            return {}
        
        # 计算不同层特征的相关性
        layers = sorted(self.features.keys())
        correlations = {}
        
        for i, layer1 in enumerate(layers):
            for layer2 in layers[i+1:]:
                f1 = self.features[layer1].cpu().numpy()
                f2 = self.features[layer2].cpu().numpy()
                
                # CCA或简单相关
                # 这里简化为特征均值的相关
                corr = np.corrcoef(f1.mean(axis=1), f2.mean(axis=1))[0, 1]
                correlations[f"{layer1}-{layer2}"] = float(corr)
        
        return {
            "layer_correlations": correlations
        }


# ============================================================
# 测试代码
# ============================================================

def test_feature_extractor():
    """测试特征提取器"""
    import sys
    sys.path.insert(0, "d:/ai2050/TransformerLens-Project")
    
    from transformer_lens import HookedTransformer
    
    # 加载模型
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small")
    
    # 创建配置
    config = ExtractionConfig(
        model_name="gpt2-small",
        target_layers=[0, 6, 11],
        sae_latent_dim=2048,
        num_samples=100
    )
    
    # 创建提取器
    extractor = FeatureExtractor(model, config)
    
    # 准备测试文本
    texts = [
        "The cat sat on the mat.",
        "A dog is running in the park.",
        "Mathematics is the language of nature.",
        "The quick brown fox jumps over the lazy dog.",
    ] * 25  # 100 samples
    
    # 运行提取
    print("\nRunning feature extraction...")
    results = extractor.run_full_extraction(texts, train_sae=True, epochs=50)
    
    # 打印结果
    print("\n=== Results ===")
    for layer_idx, layer_results in results["layers"].items():
        print(f"\nLayer {layer_idx}:")
        print(f"  Intrinsic Dimension: {layer_results.get('intrinsic_dimension', 'N/A'):.2f}")
        if "sparsity" in layer_results:
            print(f"  L0 Sparsity: {layer_results['sparsity']['l0_sparsity']:.4f}")
            print(f"  Orthogonality: {layer_results['orthogonality']['orthogonality_score']:.4f}")
    
    return results


if __name__ == "__main__":
    test_feature_extractor()
