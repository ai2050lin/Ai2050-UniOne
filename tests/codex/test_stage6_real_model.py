# -*- coding: utf-8 -*-
"""
阶段6真实模型验证脚本
在本地真实LLM模型上验证统一参数级编码理论

支持模型:
- Qwen/Qwen2.5-0.5B (或Qwen3-4B)
- DeepSeek-7B (或DeepSeek-R1)
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import os

# 尝试导入transformers
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] transformers未安装，尝试使用torch直接加载...")

@dataclass
class TestResults:
    """测试结果数据类"""
    test_name: str
    passed: bool
    metrics: Dict[str, float]
    details: str = ""

class RealModelParametricEncodingTester:
    """真实模型参数级编码测试器"""

    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """初始化测试器"""
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        print(f"[INFO] 初始化真实模型测试器")
        print(f"[INFO] 模型名称: {model_name}")
        print(f"[INFO] 设备: {device}")

    def load_model(self):
        """加载真实模型"""
        print(f"\n[INFO] 正在加载模型: {self.model_name}")
        
        if not TRANSFORMERS_AVAILABLE:
            print("[ERROR] transformers库未安装")
            print("[ERROR] 请运行: pip install transformers")
            return False

        try:
            # 加载tokenizer
            print("[INFO] 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载模型
            print("[INFO] 加载模型权重...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            self.model.eval()
            print(f"[INFO] 模型加载成功!")
            
            # 打印模型信息
            print(f"\n[MODEL INFO]")
            print(f"  模型类型: {type(self.model).__name__}")
            print(f"  参数量: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
            print(f"  层数: {len(list(self.model.modules()))}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            print(f"[ERROR] 请检查模型路径是否正确")
            return False

    def extract_hidden_states(self, text: str, layer_indices: List[int] = None) -> Dict[str, torch.Tensor]:
        """
        提取文本的隐藏状态
        返回指定层的隐藏状态
        """
        if layer_indices is None:
            # 默认获取所有层
            layer_indices = list(range(len(list(self.model.model.layers))))

        # 准备输入
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 前向传播并提取隐藏状态
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # 提取指定层的隐藏状态
        hidden_states = {}
        for layer_idx in layer_indices:
            if layer_idx < len(outputs.hidden_states):
                hidden_states[f"layer_{layer_idx}"] = outputs.hidden_states[layer_idx]

        return hidden_states

    def compute_concept_encoding(self, concept: str, context: str = "") -> np.ndarray:
        """计算概念的编码（使用最后一层的隐藏状态）"""
        text = f"{context}{concept}" if context else concept
        
        hidden_states = self.extract_hidden_states(text)
        
        # 获取最后一层的隐藏状态
        last_layer_key = sorted(hidden_states.keys())[-1]
        hidden_state = hidden_states[last_layer_key]
        
        # 平均池化得到概念向量
        concept_vector = hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        return concept_vector

    def test_family_patch_mechanism(self, concepts: List[str], family: str) -> TestResults:
        """测试1: Family Patch机制（家族基+概念偏移）"""
        print(f"\n{'='*60}")
        print(f"  测试1: Family Patch机制 - {family}家族")
        print(f"{'='*60}\n")

        if self.model is None:
            print("[SKIP] 模型未加载")
            return TestResults(
                test_name=f"Family Patch机制 - {family}",
                passed=False,
                metrics={},
                details="模型未加载"
            )

        print(f"测试概念: {concepts}")

        # 计算每个概念的编码
        concept_encodings = {}
        for concept in concepts:
            encoding = self.compute_concept_encoding(concept)
            concept_encodings[concept] = encoding
            print(f"  - {concept}: 形状={encoding.shape}")

        # 计算家族基（所有概念编码的平均）
        family_base = np.mean(list(concept_encodings.values()), axis=0)

        # 计算每个概念的offset
        concept_offsets = {
            concept: encoding - family_base
            for concept, encoding in concept_encodings.items()
        }

        # 计算指标
        # 1. 共享基比例（概念编码与家族基的余弦相似度）
        shared_base_ratios = []
        for concept, encoding in concept_encodings.items():
            similarity = np.dot(encoding, family_base) / (
                np.linalg.norm(encoding) * np.linalg.norm(family_base) + 1e-6
            )
            shared_base_ratios.append(similarity)

        mean_shared_ratio = float(np.mean(shared_base_ratios))

        # 2. Offset稀疏度（offset中接近零的比例）
        offset_sparsities = []
        for offset in concept_offsets.values():
            sparsity = np.mean(np.abs(offset) < 0.01)
            offset_sparsities.append(sparsity)

        mean_offset_sparsity = float(np.mean(offset_sparsities))

        # 3. 同家族概念相似度（应该高于不同家族）
        intra_family_similarities = []
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if i < j:
                    sim = np.dot(concept_encodings[c1], concept_encodings[c2]) / (
                        np.linalg.norm(concept_encodings[c1]) * np.linalg.norm(concept_encodings[c2]) + 1e-6
                    )
                    intra_family_similarities.append(sim)

        mean_intra_similarity = float(np.mean(intra_family_similarities))

        metrics = {
            "shared_base_ratio": mean_shared_ratio,
            "offset_sparsity": mean_offset_sparsity,
            "intra_family_similarity": mean_intra_similarity,
        }

        # 阈值
        passed = (
            mean_shared_ratio > 0.5 and
            mean_offset_sparsity > 0.3 and
            mean_intra_similarity > 0.5
        )

        passed_result = self.print_result(
            f"Family Patch机制 - {family}",
            passed,
            metrics
        )

        details = (
            f"测试概念数: {len(concepts)}\n"
            f"共享基维度: {family_base.shape[0]}\n"
            f"- 概概念共享基比例: {[f'{c}: {s:.4f}' for c, s in zip(concepts, shared_base_ratios)]}\n"
        )

        return TestResults(
            test_name=f"Family Patch机制 - {family}",
            passed=passed_result,
            metrics=metrics,
            details=details
        )

    def test_multidimension_encoding(self, texts: List[Dict[str, str]]) -> TestResults:
        """
        测试2: 多维度编码
        比较同一内容在不同风格/逻辑/语法下的编码
        """
        print(f"\n{'='*60}")
        print(f"  测试2: 多维度编码")
        print(f"{'='*60}\n")

        if self.model is None:
            print("[SKIP] 模型未加载")
            return TestResults(
                test_name="多维度编码",
                passed=False,
                metrics={},
                details="模型未加载"
            )

        print("测试文本:")
        for i, item in enumerate(texts):
            print(f"  {i+1}. {item['label']}: {item['text']}")

        # 计算每个文本的编码
        text_encodings = {
            item['label']: self.compute_concept_encoding(item['text'])
            for item in texts
        }

        # 计算维度可分离性
        # 同一维度（相同风格）的文本应该更相似
        separability_scores = []

        # 比较不同维度的差异
        labels = list(text_encodings.keys())
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                sim = np.dot(text_encodings[labels[i]], text_encodings[labels[j]]) / (
                    np.linalg.norm(text_encodings[labels[i]]) * np.linalg.norm(text_encodings[labels[j]]) + 1e-6
                )
                separability_scores.append(sim)

        mean_similarity = float(np.mean(separability_scores))
        std_similarity = float(np.std(separability_scores))

        # 计算编码多样性（标准差越大越好）
        encoding_diversity = std_similarity

        metrics = {
            "mean_similarity": mean_similarity,
            "encoding_diversity": encoding_diversity,
        }

        # 阈值
        passed = encoding_diversity > 0.05

        passed_result = self.print_result(
            "多维度编码",
            passed,
            metrics
        )

        details = f"测试文本数: {len(texts)}\n"

        return TestResults(
            test_name="多维度编码",
            passed=passed_result,
            metrics=metrics,
            details=details
        )

    def test_dynamic_encoding(self, texts: List[str]) -> TestResults:
        """
        测试3: 动态编码
        跟踪同一概念在不同上下文中的编码变化
        """
        print(f"\n{'='*60}")
        print(f"  测试3: 动态编码")
        print(f"{'='*60}\n")

        if self.model is None:
            print("[SKIP] 模型未加载")
            return TestResults(
                test_name="动态编码",
                passed=False,
                metrics={},
                details="模型未加载"
            )

        print("测试上下文变化:")
        for i, text in enumerate(texts):
            print(f"  {i+1}. {text}")

        # 计算每个文本的编码
        context_encodings = [self.compute_concept_encoding(text) for text in texts]

        # 计算编码变化
        changes = []
        for i in range(len(context_encodings)-1):
            change = np.linalg.norm(context_encodings[i+1] - context_encodings[i])
            changes.append(change)

        mean_change = float(np.mean(changes))
        std_change = float(np.std(changes))

        # 计算连续性（变化应该平滑）
        if len(changes) >= 2:
            accelerations = [abs(changes[i+1] - changes[i]) for i in range(len(changes)-1)]
            mean_acceleration = float(np.mean(accelerations))
        else:
            mean_acceleration = 0.0

        # 连续性 = 1 / (1 + 加速度)
        continuity = 1.0 / (1.0 + mean_acceleration)

        metrics = {
            "mean_encoding_change": mean_change,
            "encoding_continuity": continuity,
        }

        # 阈值
        passed = continuity > 0.5

        passed_result = self.print_result(
            "动态编码",
            passed,
            metrics
        )

        details = f"测试文本数: {len(texts)}\n"

        return TestResults(
            test_name="动态编码",
            passed=passed_result,
            metrics=metrics,
            details=details
        )

    def print_result(self, test_name: str, passed: bool, metrics: Dict[str, float]):
        """打印测试结果"""
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")

        for metric_name, metric_value in metrics.items():
            print(f"    - {metric_name}: {metric_value:.4f}")

        return passed

    def run_all_tests(self) -> Dict:
        """运行所有测试"""
        print(f"\n{'='*60}")
        print(f"  阶段6: 真实模型参数级编码验证")
        print(f"  模型: {self.model_name}")
        print(f"{'='*60}")

        all_results = []

        # 测试1: Family Patch机制 - 水果家族
        result1 = self.test_family_patch_mechanism(
            concepts=["苹果", "香蕉", "橙子"],
            family="水果"
        )
        all_results.append(result1)

        # 测试2: Family Patch机制 - 动物家族
        result2 = self.test_family_patch_mechanism(
            concepts=["狗", "猫", "兔子"],
            family="动物"
        )
        all_results.append(result2)

        # 测试3: 多维度编码
        result3 = self.test_multidimension_encoding([
            {"label": "聊天风格", "text": "这个苹果很好吃，你尝尝？"},
            {"label": "论文风格", "text": "苹果作为一种重要的水果，在营养学研究中具有重要意义。"},
            {"label": "技术风格", "text": "苹果公司的产品在技术创新方面具有独特的优势。"},
        ])
        all_results.append(result3)

        # 测试4: 动态编码
        result4 = self.test_dynamic_encoding([
            "苹果是一种水果",
            "苹果是红色的，味道很甜",
            "苹果含有丰富的维生素，对健康有益",
            "苹果可以生吃，也可以做成果汁",
        ])
        all_results.append(result4)

        # 总结
        self.print_section("测试总结")

        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.passed)

        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {total_tests - passed_tests}")
        print(f"通过率: {passed_tests/total_tests*100:.2f}%")

        # 详细信息
        for result in all_results:
            print(f"\n{result.test_name}:")
            print(result.details)

        return {
            "model_name": self.model_name,
            "device": self.device,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": passed_tests / total_tests,
            "results": [asdict(r) for r in all_results],
        }

    def print_section(self, title: str):
        """打印分隔线"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")

def main():
    """主函数"""
    # 检测可用的模型
    print("[INFO] 检测本地模型...")
    
    # 检查常见的模型路径
    model_candidates = []
    
    # 尝试Qwen3-4B
    qwen3_paths = [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen-7B",
        "Qwen/Qwen1.5-7B",
    ]
    for path in qwen3_paths:
        model_candidates.append(("Qwen", path))
    
    # 尝试DeepSeek
    deepseek_paths = [
        "deepseek-ai/deepseek-llm-7b",
        "deepseek-ai/deepseek-coder-6.7b",
    ]
    for path in deepseek_paths:
        model_candidates.append(("DeepSeek", path))
    
    print(f"[INFO] 可用的模型候选:")
    for i, (name, path) in enumerate(model_candidates):
        print(f"  {i+1}. {name}: {path}")
    
    # 选择模型（这里默认第一个）
    if not model_candidates:
        print("[ERROR] 未找到可用的模型")
        print("[ERROR] 请指定模型路径或检查HuggingFace连接")
        return
    
    selected_model_name, selected_model_path = model_candidates[0]
    print(f"\n[INFO] 选择模型: {selected_model_name} ({selected_model_path})")
    
    # 创建测试器
    tester = RealModelParametricEncodingTester(
        model_name=selected_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 加载模型
    if not tester.load_model():
        print("[ERROR] 模型加载失败，退出测试")
        return
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 保存结果
    output_dir = "tempdata"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"stage6_real_model_test_{timestamp}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[INFO] 结果已保存到: {output_file}")
    
    return results

if __name__ == "__main__":
    main()
