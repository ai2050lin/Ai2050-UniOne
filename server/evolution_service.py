"""
Evolution Service (Phase XXXIII) - Ricci 睡眠演化调度引擎
==========================================================

将 OptimizedRicciFlowService 与 MotherEngineService 权重桥接，
实现 AGI 的"离线逻辑平滑"与"顿悟 (Grokking)"机制。

核心流程:
  1. 从 Mother Engine 提取当前 L1/L2 权重矩阵
  2. 初始化优化版 Ricci 流形
  3. 执行分层/自适应演化（模拟"睡眠"）
  4. 将平滑后的权重写回 Mother Engine
  5. 验证生成质量的提升

Author: AGI Research Team
Date: 2026-02-23
Version: 1.0
"""

import asyncio
import time
import torch
import numpy as np
from typing import Any, Dict, List, Optional


class EvolutionService:
    """
    Phase XXXIII: AGI 睡眠演化调度引擎
    
    连接 Mother Engine 的物理权重与 Ricci Flow 几何优化器，
    实现自动化的神经流形平滑。
    """
    
    def __init__(self):
        self.is_evolving = False
        self.evolution_progress = 0.0
        self.current_curvature = 0.0
        self.curvature_history: List[float] = []
        self.evolution_log: List[Dict] = []
        self.total_sleep_cycles = 0
        self.last_sleep_time = None
        self.pre_sleep_curvature = None
        self.post_sleep_curvature = None
        
    def get_status(self) -> Dict[str, Any]:
        """获取演化引擎的完整状态"""
        return {
            "is_evolving": self.is_evolving,
            "progress": round(self.evolution_progress, 1),
            "current_curvature": round(self.current_curvature, 6),
            "total_sleep_cycles": self.total_sleep_cycles,
            "last_sleep_time": self.last_sleep_time,
            "curvature_history": self.curvature_history[-50:],  # 最近50个数据点
            "pre_sleep_curvature": self.pre_sleep_curvature,
            "post_sleep_curvature": self.post_sleep_curvature,
            "curvature_reduction_pct": self._calc_reduction_pct(),
        }
    
    def _calc_reduction_pct(self) -> Optional[float]:
        if self.pre_sleep_curvature and self.post_sleep_curvature and self.pre_sleep_curvature > 0:
            return round((1 - self.post_sleep_curvature / self.pre_sleep_curvature) * 100, 2)
        return None
        
    async def enter_sleep(
        self,
        mother_engine,
        iterations: int = 20,
        alpha: float = 0.01,
        mode: str = "adaptive"
    ) -> Dict[str, Any]:
        """
        让 Mother Engine 进入"睡眠"状态，执行 Ricci Flow 曲率平滑。
        
        Args:
            mother_engine: MotherEngineService 实例
            iterations: 演化迭代次数
            alpha: 演化步长
            mode: 演化模式 (adaptive/uniform)
        """
        if self.is_evolving:
            return {"status": "error", "message": "Already in sleep cycle"}
        
        if not mother_engine.is_loaded:
            return {"status": "error", "message": "Mother Engine not loaded. Cannot enter sleep."}
        
        self.is_evolving = True
        self.evolution_progress = 0.0
        self.curvature_history = []
        start_time = time.time()
        
        try:
            # === Phase 1: 提取当前物理权重 ===
            # 使用 L1 感受器权重 W_receptors [vocab, dim] 作为演化目标
            W = mother_engine.W_receptors.clone().detach().float()
            print(f"[Evolution] Extracted L1 Weight Matrix: {W.shape}")
            
            # === Phase 2: 构建邻域图 ===
            # 为了避免在超大词表上计算，采样活跃区域
            vocab_size = W.shape[0]
            sample_size = min(500, vocab_size)
            
            # 按 L2 范数选取最活跃的嵌入向量
            norms = torch.norm(W, dim=1)
            _, top_indices = torch.topk(norms, sample_size)
            W_sample = W[top_indices]
            
            print(f"[Evolution] Sampled {sample_size} active embeddings for Ricci analysis")
            
            # === Phase 3: 计算初始曲率 ===
            neighbor_k = min(15, sample_size - 1)
            
            # 构建 k-NN 邻域
            dists = torch.cdist(W_sample, W_sample)
            _, knn_indices = torch.topk(dists, neighbor_k + 1, largest=False)
            knn_indices = knn_indices[:, 1:]  # 排除自身
            
            # 初始曲率估计
            initial_curvatures = self._estimate_curvatures(W_sample, knn_indices, neighbor_k)
            self.pre_sleep_curvature = initial_curvatures.mean().item()
            self.current_curvature = self.pre_sleep_curvature
            self.curvature_history.append(self.current_curvature)
            
            print(f"[Evolution] Initial Mean Curvature: {self.pre_sleep_curvature:.6f}")
            
            # === Phase 4: Ricci Flow 迭代演化 ===
            for i in range(iterations):
                # 计算每个采样点的局部 Ricci 曲率
                curvatures = self._estimate_curvatures(W_sample, knn_indices, neighbor_k)
                
                # 自适应步长
                if mode == "adaptive":
                    # 高曲率区域用更大的步长
                    curvature_weights = torch.abs(curvatures)
                    curvature_weights = curvature_weights / (curvature_weights.max() + 1e-8)
                    effective_alpha = alpha * (0.5 + curvature_weights)
                else:
                    effective_alpha = torch.full((sample_size,), alpha)
                
                # 演化更新: 向局部均值场偏移（受曲率加权）
                for j in range(sample_size):
                    neighbors = knn_indices[j]
                    local_mean = W_sample[neighbors].mean(dim=0)
                    pull = (local_mean - W_sample[j]) * effective_alpha[j].item()
                    W_sample[j] += pull
                
                # 更新邻域（每5步重算一次）
                if (i + 1) % 5 == 0:
                    dists = torch.cdist(W_sample, W_sample)
                    _, knn_indices = torch.topk(dists, neighbor_k + 1, largest=False)
                    knn_indices = knn_indices[:, 1:]
                
                # 记录当前曲率
                self.current_curvature = curvatures.abs().mean().item()
                self.curvature_history.append(self.current_curvature)
                self.evolution_progress = (i + 1) / iterations * 100
                
                self.evolution_log.append({
                    "cycle": self.total_sleep_cycles,
                    "step": i + 1,
                    "curvature": self.current_curvature,
                    "alpha": alpha if mode != "adaptive" else effective_alpha.mean().item()
                })
                
                print(f"[Evolution] Sleep Step {i+1}/{iterations} | Curvature: {self.current_curvature:.6f} | Progress: {self.evolution_progress:.0f}%")
                
                await asyncio.sleep(0.01)  # 异步让步
            
            # === Phase 5: 写回平滑后的权重 ===
            W[top_indices] = W_sample
            mother_engine.W_receptors = W.to(mother_engine.device)
            
            self.post_sleep_curvature = self.current_curvature
            self.total_sleep_cycles += 1
            self.last_sleep_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            elapsed = time.time() - start_time
            
            print(f"[Evolution] Sleep cycle complete. Curvature: {self.pre_sleep_curvature:.6f} -> {self.post_sleep_curvature:.6f}")
            print(f"[Evolution] Reduction: {self._calc_reduction_pct()}% in {elapsed:.1f}s")
            
            return {
                "status": "success",
                "sleep_cycle": self.total_sleep_cycles,
                "pre_curvature": round(self.pre_sleep_curvature, 6),
                "post_curvature": round(self.post_sleep_curvature, 6),
                "reduction_pct": self._calc_reduction_pct(),
                "iterations": iterations,
                "elapsed_seconds": round(elapsed, 2),
                "curvature_history": self.curvature_history,
            }
            
        except Exception as e:
            print(f"[Evolution] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
        finally:
            self.is_evolving = False
            self.evolution_progress = 100.0
    
    def _estimate_curvatures(
        self,
        points: torch.Tensor,
        knn_indices: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """
        估计每个采样点的离散 Ricci 曲率。
        
        使用 Ollivier-Ricci 曲率的简化近似:
        κ(x) ≈ 1 - (mean_neighbor_dist / mean_global_dist)
        
        高曲率 = 局部过度拥挤或拉伸 = "逻辑死结"
        """
        n = points.shape[0]
        curvatures = torch.zeros(n)
        
        global_mean_dist = torch.cdist(points, points).mean().item()
        
        for i in range(n):
            neighbors = knn_indices[i]
            local_dists = torch.norm(points[neighbors] - points[i], dim=1)
            local_mean = local_dists.mean().item()
            
            # Ollivier-Ricci 近似
            if global_mean_dist > 0:
                curvatures[i] = 1.0 - (local_mean / global_mean_dist)
            
        return curvatures
    
    def get_curvature_chart_data(self) -> Dict[str, Any]:
        """获取用于前端图表渲染的曲率数据"""
        return {
            "history": self.curvature_history,
            "total_steps": len(self.curvature_history),
            "current": self.current_curvature,
            "min": min(self.curvature_history) if self.curvature_history else 0,
            "max": max(self.curvature_history) if self.curvature_history else 0,
        }


# 全局单例
evolution_service = EvolutionService()
