"""
NFBT 效率优化基准测试 (Efficiency Optimization Benchmark)
========================================================

测试内容：
1. 曲率计算性能（向量化 vs 原始）
2. Ricci Flow 演化性能（分层 vs 均匀）
3. 注意力计算性能（稀疏 vs 稠密）
4. 内存占用对比
5. 端到端推理延迟

Author: AGI Research Team
Date: 2026-02-18
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    time_seconds: float
    memory_mb: float
    speedup: float = 1.0
    details: Dict = None


def measure_memory(func, *args, **kwargs) -> tuple:
    """测量函数执行的内存和时间"""
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    start_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    end_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    return result, end_time - start_time, max(0, end_mem - start_mem)


def benchmark_curvature_computation(N: int = 500, D: int = 64, d: int = 4) -> Dict[str, BenchmarkResult]:
    """
    曲率计算性能测试
    
    对比：
    - 原始版本：嵌套循环
    - 优化版本：向量化 einsum
    """
    print("\n" + "="*60)
    print("曲率计算性能测试")
    print(f"参数: N={N}, D={D}, d={d}")
    print("="*60)
    
    from scripts.riemannian_geometry import RiemannianManifold
    from scripts.riemannian_geometry_optimized import OptimizedRiemannianManifold
    
    data = torch.randn(N, D)
    
    # 原始版本
    print("\n[原始版本]")
    start = time.time()
    manifold_orig = RiemannianManifold(data)
    orig_results = []
    for idx in range(min(50, N)):
        g = manifold_orig.compute_metric_tensor(idx)
        gamma = manifold_orig.compute_christoffel_symbols(idx)
        R = manifold_orig.compute_riemann_curvature(idx)
        orig_results.append((g, gamma, R))
    orig_time = time.time() - start
    
    print(f"  时间: {orig_time:.4f}s")
    
    # 优化版本
    print("\n[优化版本]")
    start = time.time()
    manifold_opt = OptimizedRiemannianManifold(data, intrinsic_dim=d, use_cache=True)
    
    # 批量计算
    indices = list(range(min(50, N)))
    ricci, scalar, metric = manifold_opt.compute_curvatures_batch(indices)
    opt_time = time.time() - start
    
    print(f"  时间: {opt_time:.4f}s")
    print(f"  加速比: {orig_time/opt_time:.2f}x")
    print(f"  缓存大小: {manifold_opt.cache.get_size()}")
    
    return {
        "original": BenchmarkResult("原始曲率计算", orig_time, 0),
        "optimized": BenchmarkResult("优化曲率计算", opt_time, 0, orig_time/opt_time)
    }


def benchmark_ricci_flow(N: int = 500, D: int = 64, iterations: int = 20) -> Dict[str, BenchmarkResult]:
    """
    Ricci Flow 演化性能测试
    
    对比：
    - 均匀演化
    - 分层演化
    - 自适应演化
    """
    print("\n" + "="*60)
    print("Ricci Flow 演化性能测试")
    print(f"参数: N={N}, D={D}, iterations={iterations}")
    print("="*60)
    
    import asyncio
    from server.ricci_flow_service_optimized import OptimizedRicciFlowService
    
    results = {}
    
    # 均匀演化
    print("\n[均匀演化模式]")
    data = torch.randn(N, D)
    service = OptimizedRicciFlowService()
    service.initialize(data)
    
    start = time.time()
    asyncio.run(service.run_evolution_step(iterations, mode="uniform"))
    uniform_time = time.time() - start
    uniform_result = service.get_statistics()
    
    print(f"  时间: {uniform_time:.4f}s")
    print(f"  最终曲率: {uniform_result['current_curvature']:.6f}")
    results["uniform"] = BenchmarkResult("均匀演化", uniform_time, 0, details=uniform_result)
    
    # 分层演化
    print("\n[分层演化模式]")
    data = torch.randn(N, D)
    service = OptimizedRicciFlowService()
    service.initialize(data)
    
    start = time.time()
    asyncio.run(service.run_evolution_step(iterations, mode="hierarchical"))
    hier_time = time.time() - start
    hier_result = service.get_statistics()
    
    print(f"  时间: {hier_time:.4f}s")
    print(f"  最终曲率: {hier_result['current_curvature']:.6f}")
    print(f"  加速比: {uniform_time/hier_time:.2f}x")
    results["hierarchical"] = BenchmarkResult(
        "分层演化", hier_time, 0, 
        speedup=uniform_time/hier_time, 
        details=hier_result
    )
    
    # 自适应演化
    print("\n[自适应演化模式]")
    data = torch.randn(N, D)
    service = OptimizedRicciFlowService()
    service.initialize(data)
    
    start = time.time()
    asyncio.run(service.run_evolution_step(iterations, mode="adaptive"))
    adapt_time = time.time() - start
    adapt_result = service.get_statistics()
    
    print(f"  时间: {adapt_time:.4f}s")
    print(f"  最终曲率: {adapt_result['current_curvature']:.6f}")
    print(f"  加速比: {uniform_time/adapt_time:.2f}x")
    results["adaptive"] = BenchmarkResult(
        "自适应演化", adapt_time, 0,
        speedup=uniform_time/adapt_time,
        details=adapt_result
    )
    
    return results


def benchmark_attention(
    batch_size: int = 4,
    seq_len: int = 256,
    d_model: int = 256,
    n_heads: int = 4,
    num_iterations: int = 100
) -> Dict[str, BenchmarkResult]:
    """
    Attention computation performance test
    
    Compare:
    - Standard attention (O(N^2))
    - Geometric sparse attention (O(N*k))
    """
    print("\n" + "="*60)
    print("注意力计算性能测试")
    print(f"参数: batch={batch_size}, seq_len={seq_len}, d_model={d_model}")
    print("="*60)
    
    from models.geometric_sparse_attention import GeometricSparseAttention
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 标准注意力
    print("\n[标准注意力]")
    standard_attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    
    # 预热
    for _ in range(10):
        standard_attn(x, x, x)
    
    start = time.time()
    for _ in range(num_iterations):
        standard_out, _ = standard_attn(x, x, x)
    standard_time = time.time() - start
    
    print(f"  时间({num_iterations}次): {standard_time:.4f}s")
    print(f"  计算复杂度: O(N²) = {seq_len * seq_len}")
    
    # 几何稀疏注意力
    print("\n[几何稀疏注意力]")
    geo_attn = GeometricSparseAttention(d_model, n_heads, sparse_ratio=0.3)
    
    # 预热
    for _ in range(10):
        geo_attn(x)
    
    start = time.time()
    for _ in range(num_iterations):
        sparse_out, _ = geo_attn(x)
    sparse_time = time.time() - start
    
    stats = geo_attn.get_stats()
    
    print(f"  时间({num_iterations}次): {sparse_time:.4f}s")
    print(f"  加速比: {standard_time/sparse_time:.2f}x")
    print(f"  有效稀疏度: {stats['effective_sparsity']:.4f}")
    print(f"  几何过滤率: {stats['geometric_pruned']:.2%}")
    
    return {
        "standard": BenchmarkResult("标准注意力", standard_time, 0),
        "sparse": BenchmarkResult("稀疏注意力", sparse_time, 0, standard_time/sparse_time, stats)
    }


def benchmark_end_to_end(N: int = 500, seq_len: int = 128) -> Dict[str, Any]:
    """
    端到端性能测试
    
    模拟完整的推理流程：
    1. 嵌入获取
    2. 几何分析
    3. 注意力计算
    4. Ricci Flow（可选）
    """
    print("\n" + "="*60)
    print("端到端性能测试")
    print(f"参数: N={N}, seq_len={seq_len}")
    print("="*60)
    
    from models.geometric_sparse_attention import GeometricSparseAttention
    from scripts.riemannian_geometry_optimized import OptimizedRiemannianManifold
    
    d_model = 128
    
    # 模拟嵌入
    embeddings = torch.randn(N, d_model)
    input_seq = torch.randn(1, seq_len, d_model)
    
    total_start = time.time()
    
    # 步骤1: 几何分析
    step1_start = time.time()
    manifold = OptimizedRiemannianManifold(embeddings, use_cache=True)
    indices = list(range(min(50, N)))
    ricci, scalar, _ = manifold.compute_curvatures_batch(indices)
    step1_time = time.time() - step1_start
    print(f"\n[步骤1] 几何分析: {step1_time:.4f}s")
    
    # 步骤2: 注意力计算
    step2_start = time.time()
    attn = GeometricSparseAttention(d_model, sparse_ratio=0.3)
    output, _ = attn(input_seq)
    step2_time = time.time() - step2_start
    print(f"[步骤2] 注意力计算: {step2_time:.4f}s")
    
    # 步骤3: Ricci Flow（单步）
    step3_start = time.time()
    result = manifold.ricci_flow_step(sample_size=50)
    step3_time = time.time() - step3_start
    print(f"[步骤3] Ricci Flow单步: {step3_time:.4f}s")
    
    total_time = time.time() - total_start
    print(f"\n[总计] 端到端时间: {total_time:.4f}s")
    
    return {
        "geometric_analysis": step1_time,
        "attention": step2_time,
        "ricci_flow_step": step3_time,
        "total": total_time
    }


def generate_report(results: Dict[str, Any]) -> str:
    """生成测试报告"""
    report = []
    report.append("\n" + "="*60)
    report.append("NFBT 效率优化报告")
    report.append("="*60)
    report.append(f"\n测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 汇总性能提升
    report.append("\n## 性能提升汇总")
    report.append("-"*40)
    
    if "curvature" in results:
        r = results["curvature"]
        if "optimized" in r:
            report.append(f"曲率计算加速: {r['optimized'].speedup:.2f}x")
    
    if "ricci_flow" in results:
        r = results["ricci_flow"]
        if "hierarchical" in r:
            report.append(f"Ricci Flow加速: {r['hierarchical'].speedup:.2f}x")
    
    if "attention" in results:
        r = results["attention"]
        if "sparse" in r:
            report.append(f"注意力计算加速: {r['sparse'].speedup:.2f}x")
    
    return "\n".join(report)


def run_all_benchmarks():
    """运行所有基准测试"""
    print("="*60)
    print("NFBT 效率优化基准测试套件")
    print("="*60)
    
    results = {}
    
    # 1. 曲率计算测试
    try:
        results["curvature"] = benchmark_curvature_computation(N=300, D=64, d=4)
    except Exception as e:
        print(f"曲率计算测试失败: {e}")
    
    # 2. Ricci Flow 测试
    try:
        results["ricci_flow"] = benchmark_ricci_flow(N=300, D=64, iterations=10)
    except Exception as e:
        print(f"Ricci Flow测试失败: {e}")
    
    # 3. 注意力测试
    try:
        results["attention"] = benchmark_attention(
            batch_size=4, seq_len=128, d_model=128, n_heads=4
        )
    except Exception as e:
        print(f"注意力测试失败: {e}")
    
    # 4. 端到端测试
    try:
        results["end_to_end"] = benchmark_end_to_end(N=300, seq_len=64)
    except Exception as e:
        print(f"端到端测试失败: {e}")
    
    # 生成报告
    report = generate_report(results)
    print(report)
    
    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
