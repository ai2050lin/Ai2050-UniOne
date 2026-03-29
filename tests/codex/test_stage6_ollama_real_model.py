# -*- coding: utf-8 -*-
"""
阶段6真实模型验证脚本 - 使用Ollama API
在本地真实LLM模型上验证统一参数级编码理论

使用方法:
1. 确保ollama服务正在运行
2. 安装必要的模型: ollama pull qwen2.5:0.5b
3. 运行: python tests/codex/test_stage6_ollama_real_model.py
"""

import requests
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import sys

# ==================== Ollama配置 ====================
OLLAMA_API_URL = "http://localhost:11434/api"

# ==================== 测试阈值 ====================
THRESHOLDS = {
    # Family Patch机制阈值
    "shared_base_ratio": 0.5,      # 共享基比例
    "offset_sparsity": 0.3,          # offset稀疏度
    "intra_family_similarity": 0.5,  # 同家族相似度
    
    # 多维度编码阈值
    "encoding_diversity": 0.05,     # 编码多样性
    
    # 动态编码阈值
    "encoding_continuity": 0.5,      # 编码连续性
}

@dataclass
class TestResults:
    """测试结果数据类"""
    test_name: str
    passed: bool
    metrics: Dict[str, float]
    details: str = ""

class OllamaModelTester:
    """使用Ollama API的真实模型测试器"""

    def __init__(self, model_name: str = "qwen2.5:0.5b"):
        """初始化测试器"""
        self.model_name = model_name
        self.api_url = OLLAMA_API_URL
        
        print(f"\n{'='*60}")
        print(f"  初始化Ollama真实模型测试器")
        print(f"{'='*60}")
        print(f"  模型名称: {model_name}")
        print(f"  API地址: {self.api_url}")
        print(f"{'='*60}\n")

    def check_connection(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m.get('name', '') for m in models]
                print(f"[INFO] Ollama服务正常")
                print(f"[INFO] 可用模型: {available_models}")
                
                if self.model_name in available_models:
                    print(f"[INFO] 模型 '{self.model_name}' 已安装")
                    return True
                else:
                    print(f"[WARNING] 模型 '{self.model_name}' 未安装")
                    print(f"[HINT] 运行: ollama pull {self.model_name}")
                    return False
            else:
                print(f"[ERROR] Ollama服务响应异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] 无法连接到Ollama服务: {e}")
            print(f"[HINT] 请确保ollama服务正在运行")
            print(f"[HINT] 运行: ollama serve")
            return False

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        生成文本的embedding
        使用Ollama API的embeddings端点
        """
        try:
            payload = {
                "model": self.model_name,
                "input": text
            }
            
            response = requests.post(
                f"{self.api_url}/embeddings",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data.get('embedding', [])
                return np.array(embedding, dtype=np.float32)
            else:
                print(f"[ERROR] 生成embedding失败: {response.status_code}")
                print(f"[ERROR] 响应: {response.text}")
                return None
        except Exception as e:
            print(f"[ERROR] 生成embedding异常: {e}")
            return None

    def compute_concept_encoding(self, concept: str, context: str = "") -> np.ndarray:
        """
        计算概念的编码（使用embedding）
        """
        text = f"{context}{concept}" if context else concept
        
        embedding = self.generate_embedding(text)
        
        if embedding is None:
            raise Exception("无法生成embedding")
        
        return embedding

    def test_family_patch_mechanism(self, concepts: List[str], family: str) -> TestResults:
        """测试1: Family Patch机制（家族基+概念偏移）"""
        print(f"\n{'='*60}")
        print(f"  测试1: Family Patch机制 - {family}家族")
        print(f"{'='*60}\n")

        print(f"测试概念: {concepts}\n")

        # 计算每个概念的编码
        concept_encodings = {}
        for concept in concepts:
            try:
                encoding = self.compute_concept_encoding(concept)
                concept_encodings[concept] = encoding
                print(f"  ✓ {concept}: 形状={encoding.shape}, 范数={np.linalg.norm(encoding):.4f}")
            except Exception as e:
                print(f"  ✗ {concept}: 失败 - {e}")
                continue

        if not concept_encodings:
            print("[ERROR] 所有概念编码都失败")
            return TestResults(
                test_name=f"Family Patch机制 - {family}",
                passed=False,
                metrics={},
                details="所有概念编码都失败"
            )

        # 计算家族基（所有概念编码的平均）
        family_base = np.mean(list(concept_encodings.values()), axis=0)

        # 计算每个概念的offset
        concept_offsets = {
            concept: encoding - family_base
            for concept, encoding in concept_encodings.items()
        }

        # 计算指标
        # 1. 共享基比例（概念编码与家族基的余弦相似度）
        print(f"\n[指标计算]")
        shared_base_ratios = []
        for concept, encoding in concept_encodings.items():
            similarity = np.dot(encoding, family_base) / (
                np.linalg.norm(encoding) * np.linalg.norm(family_base) + 1e-6
            )
            shared_base_ratios.append(similarity)
            print(f"  - {concept} 共享基比例: {similarity:.4f}")

        mean_shared_ratio = float(np.mean(shared_base_ratios))

        # 2. Offset稀疏度（offset中接近零的比例）
        offset_sparsities = []
        for concept, offset in concept_offsets.items():
            sparsity = np.mean(np.abs(offset) < 0.1)
            offset_sparsities.append(sparsity)
            print(f"  - {concept} offset稀疏度: {sparsity:.4f}")

        mean_offset_sparsity = float(np.mean(offset_sparsities))

        # 3. 同家族概念相似度（应该高于不同家族）
        intra_family_similarities = []
        concept_names = list(concept_encodings.keys())
        for i, c1 in enumerate(concept_names):
            for j, c2 in enumerate(concept_names):
                if i < j:
                    sim = np.dot(concept_encodings[c1], concept_encodings[c2]) / (
                        np.linalg.norm(concept_encodings[c1]) * np.linalg.norm(concept_encodings[c2]) + 1e-6
                    )
                    intra_family_similarities.append(sim)
                    print(f"  - {c1} <-> {c2}: {sim:.4f}")

        mean_intra_similarity = float(np.mean(intra_family_similarities)) if intra_family_similarities else 0.0

        metrics = {
            "shared_base_ratio": mean_shared_ratio,
            "offset_sparsity": mean_offset_sparsity,
            "intra_family_similarity": mean_intra_similarity,
        }

        # 判断是否通过
        passed = (
            mean_shared_ratio >= THRESHOLDS["shared_base_ratio"] and
            mean_offset_sparsity >= THRESHOLDS["offset_sparsity"] and
            mean_intra_similarity >= THRESHOLDS["intra_family_similarity"]
        )

        passed_result = self.print_result(
            f"Family Patch机制 - {family}",
            passed,
            metrics
        )

        details = (
            f"测试概念数: {len(concepts)}\n"
            f"共享基维度: {family_base.shape[0]}\n"
            f"- 概念共享基比例: {[f'{c}: {s:.4f}' for c, s in zip(concept_names, shared_base_ratios)]}\n"
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

        print("测试文本:")
        for i, item in enumerate(texts):
            print(f"  {i+1}. {item['label']}: {item['text']}")
        print()

        # 计算每个文本的编码
        text_encodings = {}
        for item in texts:
            try:
                encoding = self.compute_concept_encoding(item['text'])
                text_encodings[item['label']] = encoding
                print(f"  ✓ {item['label']}: 范数={np.linalg.norm(encoding):.4f}")
            except Exception as e:
                print(f"  ✗ {item['label']}: 失败 - {e}")

        if not text_encodings:
            print("[ERROR] 所有文本编码都失败")
            return TestResults(
                test_name="多维度编码",
                passed=False,
                metrics={},
                details="所有文本编码都失败"
            )

        # 计算维度可分离性
        # 比较不同维度的差异
        separability_scores = []
        labels = list(text_encodings.keys())
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                sim = np.dot(text_encodings[labels[i]], text_encodings[labels[j]]) / (
                    np.linalg.norm(text_encodings[labels[i]]) * np.linalg.norm(text_encodings[labels[j]]) + 1e-6
                )
                separability_scores.append(sim)

        mean_similarity = float(np.mean(separability_scores))
        std_similarity = float(np.std(separability_scores))

        # 编码多样性（标准差越大，说明不同维度的编码差异越大）
        encoding_diversity = std_similarity

        print(f"\n[指标计算]")
        print(f"  - 平均相似度: {mean_similarity:.4f}")
        print(f"  - 编码多样性: {encoding_diversity:.4f}")

        metrics = {
            "mean_similarity": mean_similarity,
            "encoding_diversity": encoding_diversity,
        }

        # 判断是否通过
        passed = encoding_diversity >= THRESHOLDS["encoding_diversity"]

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

        print("测试上下文变化:")
        for i, text in enumerate(texts):
            print(f"  {i+1}. {text}")
        print()

        # 计算每个文本的编码
        context_encodings = []
        for i, text in enumerate(texts):
            try:
                encoding = self.compute_concept_encoding(text)
                context_encodings.append(encoding)
                print(f"  ✓ 文本{i+1}: 范数={np.linalg.norm(encoding):.4f}")
            except Exception as e:
                print(f"  ✗ 文本{i+1}: 失败 - {e}")

        if len(context_encodings) < 2:
            print("[ERROR] 需要至少2个有效编码")
            return TestResults(
                test_name="动态编码",
                passed=False,
                metrics={},
                details="需要至少2个有效编码"
            )

        # 计算编码变化
        changes = []
        print(f"\n[指标计算]")
        for i in range(len(context_encodings)-1):
            change = np.linalg.norm(context_encodings[i+1] - context_encodings[i])
            changes.append(change)
            print(f"  - 步骤{i+1} -> {i+2}: {change:.4f}")

        mean_change = float(np.mean(changes))
        std_change = float(np.std(changes))

        # 计算连续性（变化应该平滑）
        if len(changes) >= 2:
            accelerations = [abs(changes[i+1] - changes[i]) for i in range(len(changes)-1)]
            mean_acceleration = float(np.mean(accelerations))
            print(f"  - 平均加速度: {mean_acceleration:.4f}")
        else:
            mean_acceleration = 0.0

        # 连续性 = 1 / (1 + 加速度)
        continuity = 1.0 / (1.0 + mean_acceleration)

        print(f"  - 编码连续性: {continuity:.4f}")

        metrics = {
            "mean_encoding_change": mean_change,
            "encoding_continuity": continuity,
        }

        # 判断是否通过
        passed = continuity >= THRESHOLDS["encoding_continuity"]

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
        print(f"\n{status} {test_name}")
        
        for metric_name, metric_value in metrics.items():
            # 查找对应的阈值
            threshold = THRESHOLDS.get(metric_name, None)
            if threshold is not None:
                threshold_str = f" (阈值: {threshold})"
            else:
                threshold_str = ""
            
            print(f"    - {metric_name}: {metric_value:.4f}{threshold_str}")

        return passed

    def run_all_tests(self) -> Dict:
        """运行所有测试"""
        print(f"\n{'='*60}")
        print(f"  阶段6: 真实模型参数级编码验证（Ollama）")
        print(f"  模型: {self.model_name}")
        print(f"{'='*60}\n")

        all_results = []

        # 测试1: Family Patch机制 - 水果家族
        result1 = self.test_family_patch_mechanism(
            concepts=["苹果", "香蕉", "橙子", "葡萄"],
            family="水果"
        )
        all_results.append(result1)

        # 测试2: Family Patch机制 - 动物家族
        result2 = self.test_family_patch_mechanism(
            concepts=["狗", "猫", "兔子", "鸟"],
            family="动物"
        )
        all_results.append(result2)

        # 测试3: 多维度编码
        result3 = self.test_multidimension_encoding([
            {"label": "聊天风格", "text": "这个苹果很好吃，你尝尝？"},
            {"label": "论文风格", "text": "苹果作为一种重要的水果，在营养学研究中具有重要意义。"},
            {"label": "技术风格", "text": "苹果公司的产品在技术创新方面具有独特的优势。"},
            {"label": "诗歌风格", "text": "红苹果挂枝头，秋风送果香。"},
        ])
        all_results.append(result3)

        # 测试4: 动态编码
        result4 = self.test_dynamic_encoding([
            "苹果是一种水果",
            "苹果是红色的，味道很甜",
            "苹果含有丰富的维生素，对健康有益",
            "苹果可以生吃，也可以做成果汁或苹果派",
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
            "model_type": "Ollama",
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
    print(f"\n{'='*60}")
    print(f"  阶段6: 真实模型参数级编码验证（Ollama）")
    print(f"{'='*60}\n")

    # 检查Ollama服务
    print(f"[INFO] 检查Ollama服务...")
    
    tester = OllamaModelTester()
    
    if not tester.check_connection():
        print(f"\n[ERROR] 无法连接到Ollama服务")
        print(f"[HINT] 请确保ollama服务正在运行:")
        print(f"[HINT]   1. 打开新的终端窗口")
        print(f"[HINT]   2. 运行: ollama serve")
        print(f"[HINT]   3. 安装模型: ollama pull qwen2.5:0.5b")
        return None
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 保存结果
    if results:
        output_dir = "tempdata"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"stage6_ollama_test_{timestamp}.json")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[INFO] 结果已保存到: {output_file}")
    
    return results

if __name__ == "__main__":
    main()
