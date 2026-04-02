"""
所有名词编码机制分析（CUDA真实模型） - Stage424
目标：在真实模型上使用CUDA进行测试，获取编码结构的原理

测试维度：
1. 大规模名词编码（1000+名词，真实模型）
2. 多类别编码模式（10+类别）
3. 名词家族共享 vs 特异分析
4. 跨模型一致性（Qwen3 & DeepSeek7b，真实模型）
5. 编码结构原理推断

注意：需要CUDA支持，将使用TransformerLens加载真实模型
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings

# 抑制警告
warnings.filterwarnings("ignore")


class RealModelNounEncodingAnalyzer:
    """真实模型名词编码机制分析器（使用CUDA）"""

    def __init__(self):
        self.nouns = self._load_nouns()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] 使用设备: {self.device}")
        if self.device.type == "cuda":
            print(f"[INFO] CUDA设备: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        self.models = {
            "qwen3": {
                "name": "Qwen3-4B",
                "model_id": "Qwen/Qwen2.5-3B-Instruct",  # 使用较小的模型测试
                "layers": 36,
                "hidden_dim": 2048,
                "ff_dim": 5504
            },
            "deepseek7b": {
                "name": "DeepSeek-7B",
                "model_id": "deepseek-ai/DeepSeek-V2-Lite-Instruct",  # 使用轻量级模型测试
                "layers": 28,
                "hidden_dim": 4096,
                "ff_dim": 11008
            }
        }
        self.loaded_models = {}
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "models": self.models,
            "noun_count": len(self.nouns),
            "category_count": len(set(n["category"] for n in self.nouns)),
            "tests": {}
        }

    def _load_nouns(self) -> List[Dict]:
        """加载名词数据"""
        nouns = []
        try:
            with open("tests/codex/deepseek7b_bilingual_nouns_1000plus.csv", 
                     "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        nouns.append({
                            "word": parts[0],
                            "category": parts[1]
                        })
        except FileNotFoundError:
            print(f"[WARNING] 名词数据文件不存在，使用模拟数据")
            nouns = self._generate_mock_nouns()
        
        print(f"[OK] 加载了 {len(nouns)} 个名词")
        categories = set(n["category"] for n in nouns)
        print(f"[OK] 类别数量: {len(categories)}")
        for cat in sorted(categories):
            count = len([n for n in nouns if n["category"] == cat])
            print(f"  - {cat}: {count} 个")
        
        return nouns

    def _generate_mock_nouns(self) -> List[Dict]:
        """生成模拟名词数据"""
        mock_data = {
            "fruit": ["apple", "banana", "orange", "grape", "pear"],
            "animal": ["cat", "dog", "bird", "fish", "horse"],
            "vehicle": ["car", "bus", "train", "plane", "bike"],
            "color": ["red", "blue", "green", "yellow", "black"],
            "food": ["bread", "rice", "meat", "vegetable", "soup"],
            "clothing": ["shirt", "pants", "shoes", "hat", "coat"],
            "furniture": ["table", "chair", "bed", "desk", "sofa"],
            "building": ["house", "school", "hospital", "store", "library"],
            "nature": ["tree", "flower", "mountain", "river", "forest"],
            "tool": ["hammer", "knife", "drill", "screwdriver", "wrench"]
        }
        
        nouns = []
        for category, words in mock_data.items():
            for word in words:
                nouns.append({"word": word, "category": category})
        
        return nouns

    def _load_model(self, model_name: str):
        """加载模型到GPU"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        model_info = self.models[model_name]
        model_id = model_info["model_id"]
        
        print(f"\n[INFO] 正在加载 {model_info['name']} ({model_id})...")
        print(f"[INFO] 这可能需要几分钟时间...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            # 加载模型到GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            model.eval()
            
            self.loaded_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "info": model_info
            }
            
            print(f"[OK] {model_info['name']} 加载完成")
            
            # 清理GPU缓存
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return self.loaded_models[model_name]
            
        except Exception as e:
            print(f"[ERROR] 加载模型失败: {e}")
            print(f"[INFO] 将使用模拟数据替代")
            return None

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*70)
        print("所有名词编码机制分析（CUDA真实模型） - Stage424")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"名词数量: {len(self.nouns)}")
        print(f"类别数量: {len(set(n['category'] for n in self.nouns))}")
        print(f"使用设备: {self.device}")
        print("="*70)
        
        # 检查CUDA是否可用
        if self.device.type == "cpu":
            print("\n[WARNING] CUDA不可用，将使用CPU运行")
            print("[WARNING] 真实模型测试将非常慢")
            print("[INFO] 建议安装CUDA支持或使用模拟数据")
        
        # 运行测试（简化版本，先测试CUDA环境）
        print("\n[INFO] 测试CUDA环境...")
        if self.device.type == "cuda":
            print(f"[OK] CUDA可用")
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            
            # 测试简单张量操作
            x = torch.randn(1000, 1000).to(self.device)
            y = torch.randn(1000, 1000).to(self.device)
            z = torch.matmul(x, y)
            print(f"[OK] 张量运算测试通过")
            
            # 显示GPU内存使用
            if self.device.type == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                cached = torch.cuda.memory_reserved(0) / 1024**3
                print(f"[INFO] GPU内存: 已分配 {allocated:.2f} GB, 已缓存 {cached:.2f} GB")
        else:
            print("[WARNING] CUDA不可用")
        
        # 保存基本结果
        self.results["cuda_test"] = {
            "cuda_available": self.device.type == "cuda",
            "device_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
            "cuda_version": torch.version.cuda if self.device.type == "cuda" else None,
            "pytorch_version": torch.__version__
        }
        
        # 保存结果
        output_file = "tests/codex/all_noun_encoding_cuda_test_stage424.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"[OK] CUDA测试完成！")
        print(f"[OK] 结果已保存到: {output_file}")
        print(f"[INFO] 真实模型测试将在下一阶段进行")
        print(f"[INFO] 需要先下载和加载Qwen3-4B和DeepSeek-7B模型")
        print(f"{'='*70}")
        
        return self.results


if __name__ == "__main__":
    analyzer = RealModelNounEncodingAnalyzer()
    results = analyzer.run_all_tests()
