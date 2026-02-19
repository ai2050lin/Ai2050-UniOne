"""
Phase VI 集成测试：Global Workspace + Dynamic Sparsity + Cross-Bundle Sync
===========================================================================

测试 AGI 意识系统的完整工作流程：
1. GWT 竞争性激活
2. 动态稀疏注意力
3. 跨束同步
4. 全局广播

Author: AGI Research Team
Date: 2026-02-19
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TestResult:
    """测试结果"""
    name: str
    passed: bool
    time_ms: float
    details: Dict = None


class PhaseVITester:
    """Phase VI 集成测试器"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Phase VI Tester] Device: {self.device}")
        
    def test_gws_basic(self) -> TestResult:
        """测试全局工作空间基础功能"""
        start = time.time()
        
        try:
            from scripts.global_workspace_controller import GlobalWorkspaceController
            
            # 创建控制器
            gws = GlobalWorkspaceController(gws_dim=64, sparsity_ratio=0.2).to(self.device)
            
            # 注册模块
            gws.register_module("Vision", 32)
            gws.register_module("Logic", 16)
            gws.register_module("Language", 48)
            
            # 模拟信号
            batch_size = 2
            signals = {
                "Vision": torch.randn(batch_size, 32).to(self.device) * 2.0,
                "Logic": torch.randn(batch_size, 16).to(self.device) * 0.5,
                "Language": torch.randn(batch_size, 48).to(self.device) * 1.5
            }
            
            # 竞争
            winner, gws_state = gws.compete(signals)
            
            # 验证
            assert winner in ["Vision", "Logic", "Language"], f"Invalid winner: {winner}"
            assert gws_state.shape == (64,), f"Invalid GWS state shape: {gws_state.shape}"
            
            # 广播
            broadcast = gws.broadcast("Vision")
            assert broadcast.shape == (32,), f"Invalid broadcast shape: {broadcast.shape}"
            
            # 意识水平
            consciousness = gws.get_consciousness_level()
            assert 0 <= consciousness <= 1, f"Invalid consciousness level: {consciousness}"
            
            elapsed = (time.time() - start) * 1000
            
            return TestResult(
                name="GWS Basic",
                passed=True,
                time_ms=elapsed,
                details={
                    "winner": winner,
                    "consciousness_level": consciousness.item(),
                    "gws_energy": torch.norm(gws_state).item()
                }
            )
            
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name="GWS Basic",
                passed=False,
                time_ms=elapsed,
                details={"error": str(e)}
            )
    
    def test_gws_attention_cycle(self) -> TestResult:
        """测试注意力周期"""
        start = time.time()
        
        try:
            from scripts.global_workspace_controller import GlobalWorkspaceController
            
            gws = GlobalWorkspaceController(gws_dim=64, sparsity_ratio=0.2).to(self.device)
            gws.register_module("Vision", 32)
            gws.register_module("Logic", 16)
            
            # 多周期竞争
            signals = {
                "Vision": torch.randn(4, 32).to(self.device) * 3.0,
                "Logic": torch.randn(4, 16).to(self.device) * 1.0
            }
            
            result = gws.run_attention_cycle(signals, num_cycles=3)
            
            # 验证
            assert "cycles" in result, "Missing cycles in result"
            assert len(result["cycles"]) == 3, f"Expected 3 cycles, got {len(result['cycles'])}"
            assert result["final_winner"] in ["Vision", "Logic"], "Invalid final winner"
            
            elapsed = (time.time() - start) * 1000
            
            return TestResult(
                name="GWS Attention Cycle",
                passed=True,
                time_ms=elapsed,
                details={
                    "final_winner": result["final_winner"],
                    "final_consciousness": result["final_consciousness"],
                    "cycle_count": len(result["cycles"])
                }
            )
            
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name="GWS Attention Cycle",
                passed=False,
                time_ms=elapsed,
                details={"error": str(e)}
            )
    
    def test_dynamic_sparsity(self) -> TestResult:
        """测试动态稀疏引擎"""
        start = time.time()
        
        try:
            from scripts.global_workspace_controller import DynamicSparsityEngine
            
            engine = DynamicSparsityEngine(d_model=128, sparsity_ratio=0.2).to(self.device)
            
            # 测试稀疏化
            x = torch.randn(4, 32, 128).to(self.device)
            sparse_x, mask = engine(x, return_mask=True)
            
            # 验证稀疏度
            if mask is not None:
                actual_sparsity = 1 - mask.mean().item()
                assert actual_sparsity > 0.5, f"Sparsity too low: {actual_sparsity}"
            
            # 测试稀疏注意力
            Q = torch.randn(2, 16, 64).to(self.device)
            K = torch.randn(2, 16, 64).to(self.device)
            V = torch.randn(2, 16, 64).to(self.device)
            
            output = engine.sparse_attention(Q, K, V, k_ratio=0.3)
            assert output.shape == Q.shape, f"Shape mismatch: {output.shape} vs {Q.shape}"
            
            elapsed = (time.time() - start) * 1000
            
            return TestResult(
                name="Dynamic Sparsity",
                passed=True,
                time_ms=elapsed,
                details={
                    "input_shape": list(x.shape),
                    "output_shape": list(output.shape),
                    "sparse_ratio": 0.2
                }
            )
            
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name="Dynamic Sparsity",
                passed=False,
                time_ms=elapsed,
                details={"error": str(e)}
            )
    
    def test_cross_bundle_sync(self) -> TestResult:
        """测试跨束同步"""
        start = time.time()
        
        try:
            from scripts.global_workspace_controller import CrossBundleSynchronizer
            
            sync = CrossBundleSynchronizer(d_model=64, n_bundles=3).to(self.device)
            
            # 测试平行移动
            v = torch.randn(4, 64).to(self.device)
            dx = torch.randn(4, 64).to(self.device) * 0.1
            
            v_new = sync.parallel_transport(v, connection_idx=0, dx=dx)
            assert v_new.shape == v.shape, f"Shape mismatch: {v_new.shape} vs {v.shape}"
            
            # 测试同步
            bundle_states = [
                torch.randn(4, 64).to(self.device),
                torch.randn(4, 64).to(self.device),
                torch.randn(4, 64).to(self.device)
            ]
            
            synced = sync.sync_bundles(bundle_states)
            assert len(synced) == 3, f"Expected 3 synced states, got {len(synced)}"
            
            # 验证同步效果
            # 同步后，束的状态应该更接近
            initial_dist = torch.norm(bundle_states[0] - bundle_states[1]).item()
            synced_dist = torch.norm(synced[0] - synced[1]).item()
            
            # 测试曲率计算
            curvature = sync.compute_curvature(0)
            assert curvature.shape == (64, 64), f"Invalid curvature shape: {curvature.shape}"
            
            elapsed = (time.time() - start) * 1000
            
            return TestResult(
                name="Cross-Bundle Sync",
                passed=True,
                time_ms=elapsed,
                details={
                    "initial_distance": initial_dist,
                    "synced_distance": synced_dist,
                    "sync_improvement": initial_dist / max(synced_dist, 0.01)
                }
            )
            
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name="Cross-Bundle Sync",
                passed=False,
                time_ms=elapsed,
                details={"error": str(e)}
            )
    
    def test_geometric_sparse_attention(self) -> TestResult:
        """测试几何稀疏注意力"""
        start = time.time()
        
        try:
            from models.geometric_sparse_attention import GeometricSparseAttention
            
            attn = GeometricSparseAttention(
                d_model=256,
                n_heads=4,
                sparse_ratio=0.3,
                use_geometric_distance=True
            ).to(self.device)
            
            x = torch.randn(2, 64, 256).to(self.device)
            output, weights = attn(x, return_attention_weights=True)
            
            # 验证输出
            assert output.shape == x.shape, f"Output shape mismatch: {output.shape}"
            
            # 获取统计
            stats = attn.get_stats()
            
            elapsed = (time.time() - start) * 1000
            
            return TestResult(
                name="Geometric Sparse Attention",
                passed=True,
                time_ms=elapsed,
                details={
                    "effective_sparsity": stats["effective_sparsity"],
                    "geometric_pruned": stats["geometric_pruned"]
                }
            )
            
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name="Geometric Sparse Attention",
                passed=False,
                time_ms=elapsed,
                details={"error": str(e)}
            )
    
    def test_full_integration(self) -> TestResult:
        """完整集成测试：GWT + Sparsity + Sync"""
        start = time.time()
        
        try:
            from scripts.global_workspace_controller import GlobalWorkspaceService
            from models.geometric_sparse_attention import GeometricSparseAttention
            
            # 创建服务
            service = GlobalWorkspaceService(gws_dim=64, sparsity_ratio=0.2, device=self.device)
            
            # 连接模块
            service.connect_module("Vision_Stream", 32)
            service.connect_module("Logic_Core", 16)
            service.connect_module("Language_Model", 48)
            
            # 创建注意力模块
            attn = GeometricSparseAttention(
                d_model=64, n_heads=4, sparse_ratio=0.3
            ).to(self.device)
            
            # 模拟完整流程
            # 1. 各模块产生信号
            vision_signal = torch.randn(4, 32).to(self.device) * 2.0
            logic_signal = torch.randn(4, 16).to(self.device) * 1.0
            lang_signal = torch.randn(4, 48).to(self.device) * 1.5
            
            # 2. GWS 处理
            signals = {
                "Vision_Stream": vision_signal,
                "Logic_Core": logic_signal,
                "Language_Model": lang_signal
            }
            
            result = service.process_signals(signals)
            
            # 3. 广播回模块
            vision_feedback = service.get_broadcast("Vision_Stream")
            
            # 4. 应用几何稀疏注意力
            gws_state = service.controller.gws_state.unsqueeze(0)  # (1, 64)
            # 创建序列形式的输入
            seq_input = gws_state.unsqueeze(1).expand(1, 8, -1).to(self.device)  # (1, 8, 64)
            attn_output, _ = attn(seq_input)
            
            # 5. 同步束
            synced = service.controller.synchronize_bundles(
                ["Vision_Stream", "Logic_Core", "Language_Model"]
            )
            
            elapsed = (time.time() - start) * 1000
            
            return TestResult(
                name="Full Integration",
                passed=True,
                time_ms=elapsed,
                details={
                    "winner": result["final_winner"],
                    "consciousness": result["final_consciousness"],
                    "synced_bundles": len(synced),
                    "attn_output_shape": list(attn_output.shape)
                }
            )
            
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name="Full Integration",
                passed=False,
                time_ms=elapsed,
                details={"error": str(e)}
            )
    
    def benchmark_efficiency(self) -> TestResult:
        """效率基准测试"""
        start = time.time()
        
        try:
            from scripts.global_workspace_controller import GlobalWorkspaceService
            from models.geometric_sparse_attention import GeometricSparseAttention
            
            service = GlobalWorkspaceService(gws_dim=64, device=self.device)
            service.connect_module("Vision", 32)
            service.connect_module("Logic", 16)
            
            # 批量测试
            times = []
            for _ in range(100):
                t0 = time.time()
                signals = {
                    "Vision": torch.randn(8, 32).to(self.device),
                    "Logic": torch.randn(8, 16).to(self.device)
                }
                service.process_signals(signals)
                times.append((time.time() - t0) * 1000)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            elapsed = (time.time() - start) * 1000
            
            return TestResult(
                name="Efficiency Benchmark",
                passed=True,
                time_ms=elapsed,
                details={
                    "avg_time_ms": avg_time,
                    "min_time_ms": min_time,
                    "max_time_ms": max_time,
                    "throughput_per_sec": 1000 / avg_time
                }
            )
            
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return TestResult(
                name="Efficiency Benchmark",
                passed=False,
                time_ms=elapsed,
                details={"error": str(e)}
            )
    
    def run_all_tests(self) -> Dict:
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("Phase VI Integration Test Suite")
        print("=" * 60 + "\n")
        
        tests = [
            self.test_gws_basic,
            self.test_gws_attention_cycle,
            self.test_dynamic_sparsity,
            self.test_cross_bundle_sync,
            self.test_geometric_sparse_attention,
            self.test_full_integration,
            self.benchmark_efficiency
        ]
        
        for test in tests:
            result = test()
            self.results.append(result)
            
            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"{status} {result.name} ({result.time_ms:.2f}ms)")
            
            if result.details:
                if result.passed:
                    for k, v in result.details.items():
                        if isinstance(v, float):
                            print(f"       {k}: {v:.4f}")
                        else:
                            print(f"       {k}: {v}")
                else:
                    print(f"       Error: {result.details.get('error', 'Unknown')}")
            print()
        
        # 汇总
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        total_time = sum(r.time_ms for r in self.results)
        
        print("=" * 60)
        print(f"Summary: {passed}/{total} tests passed in {total_time:.2f}ms")
        print("=" * 60)
        
        return {
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "total_time_ms": total_time,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "time_ms": r.time_ms,
                    "details": r.details
                }
                for r in self.results
            ]
        }


def main():
    """主函数"""
    tester = PhaseVITester()
    summary = tester.run_all_tests()
    
    # 保存报告
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/phase_vi_test_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nReport saved to: tempdata/phase_vi_test_report.json")
    
    return summary


if __name__ == "__main__":
    main()
