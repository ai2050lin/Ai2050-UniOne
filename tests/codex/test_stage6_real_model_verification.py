# -*- coding: utf-8 -*-
"""
阶段6真实模型验证脚本
使用本地真实LLM模型验证统一参数级编码理论

使用方法:
1. 将模型放在本地路径（如 ~/models/Qwen3-4B, ~/models/DeepSeek-7B）
2. 修改 MODEL_CONFIGS 中的路径
3. 运行: python tests/codex/test_stage6_real_model_verification.py
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import sys

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 尝试导入transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ==================== 模型配置 ====================
# 请根据您的实际路径修改以下配置
MODEL_CONFIGS = {
    "Qwen3-4B": {
        "path": os.path.expanduser("~/models/Qwen3-4B"),  # 修改为您的实际路径
        "type": "qwen",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "DeepSeek-7B": {
        "path": os.path.expanduser("~/models/DeepSeek-7B"),  # 修改为您的实际路径
        "type": "deepseek",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
}

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

class RealModelTester:
    """真实模型测试器"""

    def __init__(self, model_name: str, model_path: str, device: str = "cpu"):
        """初始化测试器"""
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
        print(f"\n{'='*60}")
        print(f"  初始化真实模型测试器")
        print(f"{'='*60}")
        print(f"  模型名称: {model_name}")
        print(f"  模型路径: {model_path}")
        print(f"  设备: {device}")
        print(f"{'='*60}\n")

    def load_model(self):
        """加载真实模型"""
        print(f"[INFO] 正在加载模型: {self.model_name}")
        print(f"[INFO] 模型路径: {self.model_path}")
        
        if not TRANSFORMERS_AVAILABLE:
            print("[ERROR] transformers库未安装")
            print("[ERROR] 请运行: pip install transformers")
            return False

        # 检查模型路径是否存在
        if not os.path.exists(self.model_path):
            print(f"[ERROR] 模型路径不存在: {self.model_path}")
            print(f"[ERROR] 请检查路径或修改MODEL_CONFIGS")
            return False

        try:
            # 加载tokenizer
            print("[INFO] 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载模型
            print("[INFO] 加载模型权重...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            print(f"[INFO] 模型加载成功!\n")
            
            # 打印模型信息
            self._print_model_info()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _print_model_info(self):
        """打印模型信息"""
        print(f"[MODEL INFO]")
        print(f"  模型类型: {type(self.model).__name__}")
        print(f"  参数量: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        print(f"  训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e9:.2f}B")
        
        # 获取层数
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'layers'):
            num_layers = len(self.model.layers)
        else:
            num_layers = "未知"
        print(f"  层数: {num_layers}")
        
        # 获取隐藏维度
        if hasattr(self.model, 'config'):
            hidden_size = getattr(self.model.config, 'hidden_size', '未知')
            print(f"  隐藏维度: {hidden_size}")
        
        print()

    def extract_hidden_states(self, text: str, layer_indices: List[int] = None) -> Dict[str, torch.Tensor]:
        """
        提取文本的隐藏状态
        返回指定层的隐藏状态
        """
        if layer_indices is None:
            # 默认获取所有层
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                layer_indices = list(range(len(self.model.model.layers)))
            elif hasattr(self.model, 'layers'):
                layer_indices = list(range(len(self.model.layers)))
            else:
                layer_indices = [0]

        # 准备输入
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
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

    def compute_concept_encoding(self, concept: str, context: str = "", layer: int = -1) -> np.ndarray:
        """
        计算概念的编码
        
        Args:
            concept: 概念文本
            context: 上下文
            layer: 使用的层索引（-1表示最后一层）
        
        Returns:
            概念向量 (numpy数组)
        """
        text = f"{context}{concept}" if context else concept
        
        hidden_states = self.extract_hidden_states(text)
        
        # 获取指定层的隐藏状态
        if layer == -1:
            layer_key = sorted(hidden_states.keys())[-1]
        else:
            layer_key = f"layer_{layer}"
        
        if layer_key not in hidden_states:
            print(f"[WARNING] 层 {layer} 不存在，使用最后一层")
            layer_key = sorted(hidden_states.keys())[-1]
        
        hidden_state = hidden_states[layer_key]
        
        # 平均池化得到概念向量
        # shape: (batch_size, seq_len, hidden_size)
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
            sparsity = np.mean(np.abs(offset) < 0.1)  # 使用较大的阈值
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
        print(f"  阶段6: 真实模型参数级编码验证")
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
            "model_path": self.model_path,
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

def find_model_in_common_paths():
    """在常见路径中查找模型"""
    common_paths = [
        "~/models",
        "~/huggingface/models",
        "./models",
        "../models",
        "/mnt/models",
    ]
    
    found_models = []
    
    for base_path in common_paths:
        expanded_path = os.path.expanduser(base_path)
        if os.path.exists(expanded_path):
            # 查找模型目录
            for item in os.listdir(expanded_path):
                model_path = os.path.join(expanded_path, item)
                if os.path.isdir(model_path):
                    # 检查是否包含模型文件
                    model_files = os.listdir(model_path)
                    if any(f.endswith(('.bin', '.safetensors', '.pt')) for f in model_files):
                        found_models.append((item, model_path))
    
    return found_models

def main():
    """主函数"""
    print(f"\n{'='*60}")
    print(f"  阶段6: 真实模型参数级编码验证")
    print(f"{'='*60}\n")

    # 1. 检测设备
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[INFO] 检测到CUDA设备")
        print(f"[INFO] GPU数量: {torch.cuda.device_count()}")
        print(f"[INFO] GPU名称: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"[INFO] 未检测到CUDA设备，使用CPU")

    # 2. 查找可用的模型
    print(f"\n[INFO] 查找本地模型...")
    
    found_models = find_model_in_common_paths()
    
    if found_models:
        print(f"[INFO] 找到以下模型:")
        for i, (name, path) in enumerate(found_models):
            print(f"  {i+1}. {name}: {path}")
    else:
        print(f"[WARNING] 未在常见路径找到模型")
        print(f"[INFO] 请手动配置MODEL_CONFIGS中的路径")
        print(f"[INFO] 常见路径:")
        for path in ["~/models", "~/huggingface/models", "./models"]:
            print(f"      - {os.path.expanduser(path)}")

    # 3. 选择模型
    if found_models:
        selected_name, selected_path = found_models[0]
        print(f"\n[INFO] 自动选择模型: {selected_name}")
    else:
        # 尝试使用配置文件中的第一个模型
        print(f"\n[INFO] 尝试使用配置文件中的模型...")
        selected_name = list(MODEL_CONFIGS.keys())[0]
        selected_path = MODEL_CONFIGS[selected_name]["path"]
        device = MODEL_CONFIGS[selected_name]["device"]
        print(f"[INFO] 选择模型: {selected_name}")
        print(f"[INFO] 模型路径: {selected_path}")

    # 4. 创建测试器
    tester = RealModelTester(
        model_name=selected_name,
        model_path=selected_path,
        device=device
    )
    
    # 5. 加载模型
    if not tester.load_model():
        print(f"\n[ERROR] 模型加载失败")
        print(f"[ERROR] 请检查:")
        print(f"      1. 模型路径是否正确")
        print(f"      2. 模型文件是否完整")
        print(f"      3. transformers库是否安装")
        print(f"[INFO] 您可以修改脚本中的MODEL_CONFIGS来指定正确的模型路径")
        return None
    
    # 6. 运行所有测试
    results = tester.run_all_tests()
    
    # 7. 保存结果
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
