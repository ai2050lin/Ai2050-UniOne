"""
特征涌现追踪实验

目标: 追踪神经网络如何从信号流中提取特征

方法:
1. 小规模训练: 训练一个小型GPT模型
2. 检查点追踪: 每100步保存激活模式
3. 涌现检测: 检测特征何时出现
4. 模式分析: 分析涌现规律

运行时间: 约30分钟（CPU）/ 5分钟（GPU）
"""

import os
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time
from tqdm import tqdm


@dataclass
class EmergenceExperimentConfig:
    """特征涌现实验配置"""
    # 模型配置
    vocab_size: int = 1000  # 小词表加速
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 256
    max_seq_len: int = 32
    
    # 训练配置
    total_steps: int = 5000
    save_interval: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # 追踪配置
    track_layers: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    concept_probes: List[str] = field(default_factory=lambda: [
        "cat", "dog", "run", "jump", "red", "blue",
        "happy", "sad", "big", "small", "good", "bad"
    ])
    
    # 输出配置
    output_dir: str = "results/feature_emergence"


class SimpleTransformer(nn.Module):
    """
    简化版Transformer用于特征涌现追踪
    
    特点:
    - 参数量小，训练快
    - 可以访问中间层激活
    - 用于验证方法论
    """
    
    def __init__(self, config: EmergenceExperimentConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer块
        self.layers = nn.ModuleList([
            self._make_layer() for _ in range(config.n_layers)
        ])
        
        # 输出层
        self.ln_final = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 存储中间激活
        self.activations: Dict[int, torch.Tensor] = {}
        self._hooks = []
    
    def _make_layer(self):
        """创建单个Transformer层"""
        return nn.ModuleDict({
            'ln1': nn.LayerNorm(self.config.d_model),
            'attn': nn.MultiheadAttention(
                self.config.d_model, 
                self.config.n_heads,
                batch_first=True
            ),
            'ln2': nn.LayerNorm(self.config.d_model),
            'mlp': nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_mlp),
                nn.GELU(),
                nn.Linear(self.config.d_mlp, self.config.d_model)
            )
        })
    
    def register_hooks(self):
        """注册hook以捕获中间激活"""
        def make_hook(layer_idx):
            def hook(module, input, output):
                self.activations[layer_idx] = output.detach()
            return hook
        
        for i, layer in enumerate(self.layers):
            self._hooks.append(
                layer['ln2'].register_forward_hook(make_hook(i))
            )
    
    def remove_hooks(self):
        """移除hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def forward(self, x):
        """前向传播"""
        seq_len = x.shape[1]
        
        # 嵌入
        x = self.embed(x) + self.pos_embed(torch.arange(seq_len, device=x.device))
        
        # Transformer层
        for layer in self.layers:
            # 注意力
            x_norm = layer['ln1'](x)
            attn_out, _ = layer['attn'](x_norm, x_norm, x_norm, need_weights=False)
            x = x + attn_out
            
            # MLP
            x_norm = layer['ln2'](x)
            mlp_out = layer['mlp'](x_norm)
            x = x + mlp_out
        
        # 输出
        x = self.ln_final(x)
        logits = self.unembed(x)
        
        return logits
    
    def get_layer_activations(self, x):
        """获取各层激活"""
        self.activations = {}
        self.register_hooks()
        _ = self.forward(x)
        self.remove_hooks()
        return dict(self.activations)


class SyntheticDataset(Dataset):
    """
    合成数据集用于训练
    
    生成简单的语言模式，便于追踪特征涌现
    """
    
    def __init__(self, vocab_size: int = 1000, seq_len: int = 32, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # 生成数据
        self.data = self._generate_data()
    
    def _generate_data(self):
        """生成合成数据"""
        # 创建简单的模式
        patterns = [
            # A -> B 模式
            lambda: torch.cat([
                torch.randint(0, 100, (self.seq_len // 2,)),
                torch.randint(100, 200, (self.seq_len // 2,))
            ]),
            # 重复模式
            lambda: torch.cat([
                base := torch.randint(0, 100, (self.seq_len // 4,)),
                base,
                torch.randint(200, 300, (self.seq_len // 2,))
            ]),
            # 递增模式
            lambda: torch.arange(0, self.seq_len) % 100 + torch.randint(0, 10, (1,)).item(),
        ]
        
        data = []
        for _ in range(self.num_samples):
            pattern_fn = patterns[np.random.randint(len(patterns))]
            data.append(pattern_fn())
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return x, y


class FeatureEmergenceTracker:
    """
    特征涌现追踪器
    
    核心问题: 神经网络如何从信号流中提取特征？
    
    方法:
    1. 在训练过程中定期检查点
    2. 分析激活模式的演化
    3. 检测特征涌现事件
    """
    
    def __init__(self, config: EmergenceExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储涌现历史
        self.emergence_history: Dict[int, Dict[str, Any]] = {}
        self.feature_trajectory: Dict[str, List[float]] = defaultdict(list)
        
        # 特征检测器状态
        self.feature_detectors: Dict[int, Dict[str, Any]] = defaultdict(dict)
        
    def track_step(
        self,
        model: SimpleTransformer,
        step: int,
        dataloader: DataLoader,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        在训练步骤追踪特征涌现
        
        关键指标:
        1. 激活聚类质量: 特征是否形成清晰簇？
        2. 激活稀疏度: 特征是否稀疏？
        3. 特征分离度: 不同概念是否分离？
        4. 特征稳定性: 特征是否稳定？
        """
        model.eval()
        results = {
            "step": step,
            "layers": {},
            "emergence_events": [],
            "feature_metrics": {}
        }
        
        # 收集一批激活
        all_activations: Dict[int, List[torch.Tensor]] = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                if batch_idx >= 5:  # 只采样5批
                    break
                
                x = x.to(device)
                activations = model.get_layer_activations(x)
                
                for layer_idx, act in activations.items():
                    all_activations[layer_idx].append(act)
        
        # 分析每层激活
        for layer_idx in self.config.track_layers:
            if layer_idx not in all_activations:
                continue
            
            layer_acts = torch.cat(all_activations[layer_idx], dim=0)
            layer_acts = layer_acts.view(-1, layer_acts.shape[-1])  # [N, d_model]
            
            # 计算特征指标
            metrics = self._compute_feature_metrics(layer_acts, layer_idx, step)
            results["layers"][layer_idx] = metrics
            
            # 检测涌现事件
            emergence = self._detect_emergence(metrics, layer_idx, step)
            if emergence:
                results["emergence_events"].extend(emergence)
        
        # 存储历史
        self.emergence_history[step] = results
        
        model.train()
        return results
    
    def _compute_feature_metrics(
        self,
        activations: torch.Tensor,
        layer_idx: int,
        step: int
    ) -> Dict[str, Any]:
        """
        计算特征涌现指标
        
        指标:
        1. 稀疏度: 激活中有多少接近零
        2. 聚类性: 激活是否形成簇
        3. 分离度: 簇之间的距离
        4. 熵: 激活分布的熵
        """
        metrics = {}
        
        # 1. 稀疏度
        threshold = activations.abs().mean() * 0.1
        sparsity = (activations.abs() < threshold).float().mean().item()
        metrics["sparsity"] = sparsity
        
        # 2. 激活范数
        norm = activations.norm(dim=1).mean().item()
        metrics["activation_norm"] = norm
        
        # 3. 聚类性（使用简化的方法）
        try:
            # 随机采样计算距离
            n_samples = min(1000, activations.shape[0])
            indices = torch.randperm(activations.shape[0])[:n_samples]
            sampled = activations[indices]
            
            # 计算样本间距离
            dists = torch.cdist(sampled[:100], sampled[:100])
            mean_dist = dists.mean().item()
            std_dist = dists.std().item()
            
            metrics["mean_pairwise_dist"] = mean_dist
            metrics["std_pairwise_dist"] = std_dist
            
            # 聚类指标（距离的变异系数）
            cv = std_dist / (mean_dist + 1e-8)
            metrics["clustering_cv"] = cv
            
        except Exception as e:
            metrics["clustering_error"] = str(e)
        
        # 4. 激活熵
        act_flat = activations.flatten()
        hist = torch.histc(act_flat, bins=50)
        hist = hist / hist.sum()
        entropy = -(hist * torch.log2(hist + 1e-8)).sum().item()
        metrics["activation_entropy"] = entropy
        
        # 5. 特征维度（有效秩）
        try:
            cov = activations.T @ activations / activations.shape[0]
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 1e-6]
            if len(eigenvalues) > 0:
                effective_rank = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
                metrics["effective_rank"] = effective_rank.item()
        except:
            pass
        
        return metrics
    
    def _detect_emergence(
        self,
        metrics: Dict[str, Any],
        layer_idx: int,
        step: int
    ) -> List[Dict[str, Any]]:
        """
        检测特征涌现事件
        
        涌现条件:
        1. 稀疏度首次超过阈值
        2. 聚类性首次超过阈值
        3. 有效秩首次增加
        """
        events = []
        
        # 获取前一次状态
        prev_state = self.feature_detectors[layer_idx]
        
        # 检测稀疏度涌现
        sparsity_threshold = 0.3
        current_sparsity = metrics.get("sparsity", 0)
        prev_sparsity = prev_state.get("sparsity", 0)
        
        if current_sparsity > sparsity_threshold and prev_sparsity <= sparsity_threshold:
            events.append({
                "type": "sparsity_emergence",
                "layer": layer_idx,
                "step": step,
                "value": current_sparsity,
                "message": f"Layer {layer_idx} 稀疏度首次超过{sparsity_threshold:.0%}"
            })
        
        # 检测聚类涌现
        clustering_threshold = 0.5
        current_cv = metrics.get("clustering_cv", 0)
        prev_cv = prev_state.get("clustering_cv", 0)
        
        if current_cv > clustering_threshold and prev_cv <= clustering_threshold:
            events.append({
                "type": "clustering_emergence",
                "layer": layer_idx,
                "step": step,
                "value": current_cv,
                "message": f"Layer {layer_idx} 激活开始形成聚类"
            })
        
        # 检测有效秩涌现
        rank_threshold = 5
        current_rank = metrics.get("effective_rank", 0)
        prev_rank = prev_state.get("effective_rank", 0)
        
        if current_rank > rank_threshold and prev_rank <= rank_threshold:
            events.append({
                "type": "dimension_emergence",
                "layer": layer_idx,
                "step": step,
                "value": current_rank,
                "message": f"Layer {layer_idx} 特征维度涌现"
            })
        
        # 更新状态
        self.feature_detectors[layer_idx] = metrics.copy()
        
        return events
    
    def analyze_emergence_pattern(self) -> Dict[str, Any]:
        """
        分析涌现模式
        
        关键问题:
        1. 哪些层先出现特征？
        2. 稀疏性和聚类性的涌现顺序？
        3. 涌现是否稳定？
        """
        if not self.emergence_history:
            return {"error": "No history"}
        
        analysis = {
            "emergence_timeline": [],
            "layer_emergence_order": [],
            "metric_evolution": {},
            "key_findings": []
        }
        
        # 1. 提取涌现事件时间线
        for step in sorted(self.emergence_history.keys()):
            for event in self.emergence_history[step].get("emergence_events", []):
                analysis["emergence_timeline"].append({
                    "step": step,
                    **event
                })
        
        # 2. 分析各层的涌现顺序
        layer_first_emergence = {}
        for event in analysis["emergence_timeline"]:
            layer = event["layer"]
            step = event["step"]
            if layer not in layer_first_emergence:
                layer_first_emergence[layer] = step
        
        analysis["layer_emergence_order"] = sorted(
            layer_first_emergence.items(),
            key=lambda x: x[1]
        )
        
        # 3. 分析指标演化
        for layer in self.config.track_layers:
            evolution = {
                "sparsity": [],
                "activation_norm": [],
                "effective_rank": []
            }
            
            for step in sorted(self.emergence_history.keys()):
                layer_data = self.emergence_history[step].get("layers", {}).get(layer, {})
                for key in evolution:
                    if key in layer_data:
                        evolution[key].append((step, layer_data[key]))
            
            analysis["metric_evolution"][layer] = evolution
        
        # 4. 提取关键发现
        findings = []
        
        if analysis["layer_emergence_order"]:
            first_layer, first_step = analysis["layer_emergence_order"][0]
            findings.append(f"特征最先在Layer {first_layer}涌现（Step {first_step}）")
        
        # 检查是否浅层先于深层
        layer_order = [l for l, _ in analysis["layer_emergence_order"]]
        if layer_order == sorted(layer_order):
            findings.append("涌现顺序: 浅层 → 深层（符合预期）")
        elif layer_order == sorted(layer_order, reverse=True):
            findings.append("涌现顺序: 深层 → 浅层（需进一步分析）")
        
        # 检查稀疏度演化
        if analysis["metric_evolution"]:
            layer_0_sparsity = analysis["metric_evolution"].get(0, {}).get("sparsity", [])
            if layer_0_sparsity:
                initial = layer_0_sparsity[0][1] if layer_0_sparsity else 0
                final = layer_0_sparsity[-1][1] if layer_0_sparsity else 0
                findings.append(f"Layer 0 稀疏度: {initial:.2%} → {final:.2%}")
        
        analysis["key_findings"] = findings
        
        return analysis
    
    def save_results(self):
        """保存追踪结果"""
        results = {
            "config": {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "total_steps": self.config.total_steps,
                "save_interval": self.config.save_interval
            },
            "emergence_history": {
                str(k): v for k, v in self.emergence_history.items()
            },
            "analysis": self.analyze_emergence_pattern()
        }
        
        output_path = self.output_dir / "feature_emergence_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Results saved to {output_path}")


def train_and_track():
    """
    训练模型并追踪特征涌现
    
    这是核心实验！
    """
    print("=" * 70)
    print("特征涌现追踪实验")
    print("=" * 70)
    print()
    
    # 配置
    config = EmergenceExperimentConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=4,
        total_steps=3000,  # 减少步数加速
        save_interval=100,
        batch_size=64
    )
    
    print(f"模型配置: {config.n_layers}层, {config.d_model}维")
    print(f"训练配置: {config.total_steps}步, 每{config.save_interval}步追踪")
    print()
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print()
    
    # 创建模型
    model = SimpleTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    print()
    
    # 创建数据
    dataset = SyntheticDataset(
        vocab_size=config.vocab_size,
        seq_len=config.max_seq_len,
        num_samples=10000
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    print(f"数据集大小: {len(dataset):,} 样本")
    print()
    
    # 创建追踪器
    tracker = FeatureEmergenceTracker(config)
    
    # 训练设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print("开始训练...")
    print("-" * 70)
    
    start_time = time.time()
    step = 0
    running_loss = 0
    
    progress_bar = tqdm(range(config.total_steps), desc="Training")
    
    for epoch in range(1000):  # 足够大的epoch数
        for batch_x, batch_y in dataloader:
            if step >= config.total_steps:
                break
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 前向传播
            logits = model(batch_x)
            loss = criterion(
                logits.view(-1, config.vocab_size),
                batch_y.view(-1)
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 追踪特征涌现
            if step % config.save_interval == 0:
                # 评估
                model.eval()
                track_results = tracker.track_step(model, step, dataloader, device)
                
                # 打印进度
                avg_loss = running_loss / config.save_interval
                emergence = track_results.get("emergence_events", [])
                
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "emergence": len(emergence)
                })
                
                if emergence:
                    print(f"\n  Step {step}: 发现 {len(emergence)} 个涌现事件")
                    for e in emergence:
                        print(f"    - {e['message']}")
                
                running_loss = 0
                model.train()
            
            step += 1
            progress_bar.update(1)
        
        if step >= config.total_steps:
            break
    
    progress_bar.close()
    elapsed = time.time() - start_time
    print(f"\n训练完成! 用时: {elapsed:.1f}秒")
    print()
    
    # 分析涌现模式
    print("-" * 70)
    print("分析涌现模式...")
    analysis = tracker.analyze_emergence_pattern()
    
    print("\n关键发现:")
    for finding in analysis.get("key_findings", []):
        print(f"  - {finding}")
    
    # 保存结果
    tracker.save_results()
    
    # 打印涌现时间线
    print("\n涌现事件时间线:")
    timeline = analysis.get("emergence_timeline", [])
    if timeline:
        for event in timeline[:10]:  # 只显示前10个
            print(f"  Step {event['step']}: Layer {event['layer']} - {event['type']}")
        if len(timeline) > 10:
            print(f"  ... 还有 {len(timeline) - 10} 个事件")
    else:
        print("  未检测到明显的涌现事件")
        print("  提示: 可能需要调整涌现阈值或增加训练步数")
    
    print()
    print("=" * 70)
    print("实验完成!")
    print("=" * 70)
    
    return tracker, model


if __name__ == "__main__":
    tracker, model = train_and_track()
