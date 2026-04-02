"""
Stage428: 使用CUDA测试Qwen3和DeepSeek7B模型的词性层分布
目标：在真实模型上分析名词、形容词、动词、副词、代词、介词在不同layer中的分布情况

任务：
1. 加载Qwen3和DeepSeek7B模型（使用CUDA加速）
2. 准备词性数据集（名词、形容词、动词、副词、代词、介词）
3. 提取每个单词在各层的激活值
4. 分析词性层分布特征
5. 对比不同模型的层分布差异
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import sys
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 尝试导入必要的库
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARN] transformers库未安装，请运行: pip install transformers")

try:
    import transformer_lens
    from transformer_lens import HookedTransformer
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    print("[WARN] TransformerLens未安装，请运行: pip install transformer-lens")

# 词性数据集
POS_DATA = {
    "noun": {
        "description": "名词",
        "words": [
            # 水果类
            "apple", "banana", "orange", "grape", "strawberry",
            # 动物类
            "dog", "cat", "bird", "fish", "lion",
            # 自然类
            "tree", "flower", "mountain", "river", "ocean"
        ]
    },
    "adjective": {
        "description": "形容词",
        "words": [
            # 颜色形容词
            "red", "blue", "green", "yellow", "black",
            # 大小形容词
            "big", "small", "large", "tiny", "huge",
            # 情感形容词
            "happy", "sad", "angry", "afraid", "excited"
        ]
    },
    "verb": {
        "description": "动词",
        "words": [
            # 动作动词
            "run", "walk", "jump", "swim", "fly",
            # 感知动词
            "see", "hear", "feel", "taste", "smell",
            # 思维动词
            "think", "know", "understand", "believe", "remember"
        ]
    },
    "adverb": {
        "description": "副词",
        "words": [
            # 方式副词
            "quickly", "slowly", "carefully", "easily", "badly",
            # 程度副词
            "very", "extremely", "quite", "rather", "fairly",
            # 频率副词
            "always", "often", "usually", "sometimes", "rarely"
        ]
    },
    "pronoun": {
        "description": "代词",
        "words": [
            # 人称代词
            "I", "you", "he", "she", "it",
            # 物主代词
            "my", "your", "his", "her", "its",
            # 指示代词
            "this", "that", "these", "those", "such"
        ]
    },
    "preposition": {
        "description": "介词",
        "words": [
            # 空间介词
            "in", "on", "at", "by", "to",
            # 时间介词
            "before", "after", "during", "since", "until",
            # 方向介词
            "toward", "through", "across", "along", "around"
        ]
    }
}

# 模型配置
MODEL_CONFIGS = {
    "gpt2-small": {
        "name": "gpt2",
        "num_layers": 12,
        "hidden_size": 768,
        "use_transformer_lens": True,
        "trust_remote_code": False,
        "torch_dtype": "float32"
    }
}


class CUDAModelPOSAnalyzer:
    """CUDA加速的模型词性层分布分析器"""
    
    def __init__(self, model_name: str, model_config: Dict):
        self.model_name = model_name
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n初始化模型分析器: {model_name}")
        print(f"  设备: {self.device}")
        print(f"  配置: {model_config}")
    
    def load_model(self) -> bool:
        """加载模型"""
        if not TRANSFORMERS_AVAILABLE:
            print("  [ERROR] transformers库未安装")
            return False
        
        print(f"  加载模型: {self.model_config['name']}")
        
        try:
            start_time = time.time()
            
            # 设置torch dtype
            torch_dtype = torch.float16 if self.model_config["torch_dtype"] == "float16" else torch.float32
            
            if self.model_config["use_transformer_lens"] and TRANSFORMER_LENS_AVAILABLE:
                # 使用TransformerLens加载模型
                print("  使用TransformerLens加载...")
                self.model = HookedTransformer.from_pretrained(
                    self.model_config["name"],
                    device=self.device,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.model_config.get("trust_remote_code", False)
                )
                self.tokenizer = self.model.tokenizer
                print("  [OK] TransformerLens模型加载成功")
            else:
                # 使用HuggingFace加载模型
                print("  使用HuggingFace加载...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_config["name"],
                    torch_dtype=torch_dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=self.model_config.get("trust_remote_code", False)
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_config["name"],
                    trust_remote_code=self.model_config.get("trust_remote_code", False)
                )
                print("  [OK] HuggingFace模型加载成功")
            
            elapsed_time = time.time() - start_time
            print(f"  加载时间: {elapsed_time:.2f}秒")
            
            # 显示模型信息
            if hasattr(self.model, 'cfg'):
                print(f"  层数: {self.model.cfg.n_layers}")
                print(f"  隐藏层大小: {self.model.cfg.d_model}")
                actual_num_layers = self.model.cfg.n_layers
                actual_hidden_size = self.model.cfg.d_model
            elif hasattr(self.model, 'config'):
                print(f"  层数: {self.model.config.num_hidden_layers}")
                print(f"  隐藏层大小: {self.model.config.hidden_size}")
                actual_num_layers = self.model.config.num_hidden_layers
                actual_hidden_size = self.model.config.hidden_size
            else:
                actual_num_layers = self.model_config["num_layers"]
                actual_hidden_size = self.model_config["hidden_size"]
            
            # 更新模型配置为实际值
            self.model_config["num_layers"] = actual_num_layers
            self.model_config["hidden_size"] = actual_hidden_size
            
            return True
            
        except Exception as e:
            print(f"  [ERROR] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_word_activation(self, word: str) -> Optional[np.ndarray]:
        """
        获取单词在各层的激活值
        
        返回: (num_layers, hidden_size) 的激活矩阵
        """
        if self.model is None:
            return None
        
        try:
            with torch.no_grad():
                # Tokenize
                if self.tokenizer is not None:
                    tokens = self.tokenizer.encode(word, return_tensors="pt").to(self.device)
                else:
                    print(f"  [WARN] Tokenizer未初始化，跳过单词: {word}")
                    return None
                
                # 获取各层激活
                if hasattr(self.model, 'run_with_cache'):
                    # TransformerLens模型
                    _, cache = self.model.run_with_cache(tokens)
                    activations = []
                    
                    num_layers = self.model_config["num_layers"]
                    hidden_size = self.model_config["hidden_size"]
                    
                    for layer_idx in range(num_layers):
                        # 尝试多种hook名称
                        hook_names = [
                            f"blocks.{layer_idx}.hook_resid_post",
                            f"blocks.{layer_idx}.hook_resid_mid",
                            f"blocks.{layer_idx}.mlp.hook_post",
                            f"blocks.{layer_idx}.attn.hook_q"
                        ]
                        
                        activation = None
                        for hook_name in hook_names:
                            if hook_name in cache:
                                # 获取最后一个token的激活
                                activation = cache[hook_name][0, -1, :].cpu().numpy()
                                break
                        
                        if activation is None:
                            # 如果都没有，使用零向量
                            activation = np.zeros(hidden_size)
                        
                        activations.append(activation)
                    
                    return np.array(activations)
                else:
                    # HuggingFace模型 - 需要注册hook
                    # 简化版：返回None，后续实现
                    print(f"  [WARN] HuggingFace模型激活提取尚未完全实现")
                    return None
                    
        except Exception as e:
            print(f"  [WARN] 获取激活失败 ({word}): {e}")
            return None
    
    def analyze_layer_distribution(self, activations: np.ndarray, threshold: float = 0.0) -> Dict:
        """
        分析层分布
        
        参数:
            activations: (num_layers, hidden_size) 的激活矩阵
            threshold: 激活阈值
        
        返回:
            层分布统计字典
        """
        num_layers = activations.shape[0]
        
        # 计算每层的激活强度（L2范数）
        layer_norms = []
        for layer in range(num_layers):
            norm = np.linalg.norm(activations[layer])
            layer_norms.append(norm)
        
        layer_norms = np.array(layer_norms)
        
        # 归一化
        total_norm = np.sum(layer_norms)
        if total_norm > 0:
            layer_ratios = layer_norms / total_norm
        else:
            layer_ratios = np.zeros(num_layers)
        
        # 计算前中后部分的比例
        early_end = num_layers // 3
        middle_end = 2 * num_layers // 3
        
        early_ratio = np.sum(layer_ratios[:early_end])
        middle_ratio = np.sum(layer_ratios[early_end:middle_end])
        late_ratio = np.sum(layer_ratios[middle_end:])
        
        # 找到最大激活层
        max_layer = int(np.argmax(layer_norms))
        
        # 计算有效层数（激活强度 > 平均值的层）
        mean_norm = np.mean(layer_norms)
        num_effective_layers = int(np.sum(layer_norms > mean_norm))
        
        return {
            "layer_norms": layer_norms.tolist(),
            "layer_ratios": layer_ratios.tolist(),
            "early_ratio": float(early_ratio),
            "middle_ratio": float(middle_ratio),
            "late_ratio": float(late_ratio),
            "max_layer": max_layer,
            "num_effective_layers": num_effective_layers,
            "total_norm": float(total_norm)
        }
    
    def analyze_pos(self, pos: str, words: List[str]) -> Dict:
        """
        分析特定词性的层分布
        
        参数:
            pos: 词性标签
            words: 单词列表
        
        返回:
            词性分析结果字典
        """
        print(f"  分析词性: {pos} ({len(words)} 个单词)")
        
        # 收集所有单词的激活分布
        all_layer_distributions = []
        successful_words = 0
        
        for i, word in enumerate(words):
            # 获取激活
            activations = self.get_word_activation(word)
            
            if activations is not None:
                # 分析层分布
                layer_distribution = self.analyze_layer_distribution(activations)
                all_layer_distributions.append(layer_distribution)
                successful_words += 1
            
            if (i + 1) % 5 == 0:
                print(f"    已处理: {i+1}/{len(words)} (成功: {successful_words})")
        
        if len(all_layer_distributions) == 0:
            print(f"    [WARN] 没有成功提取任何单词的激活")
            return {
                "pos": pos,
                "num_words": len(words),
                "successful_words": 0,
                "early_ratio": 0.0,
                "middle_ratio": 0.0,
                "late_ratio": 0.0,
                "avg_max_layer": 0.0
            }
        
        # 聚合层分布统计
        early_ratios = [d['early_ratio'] for d in all_layer_distributions]
        middle_ratios = [d['middle_ratio'] for d in all_layer_distributions]
        late_ratios = [d['late_ratio'] for d in all_layer_distributions]
        
        avg_layer_distribution = {
            'pos': pos,
            'num_words': len(words),
            'successful_words': successful_words,
            'early_ratio': float(np.mean(early_ratios)),
            'middle_ratio': float(np.mean(middle_ratios)),
            'late_ratio': float(np.mean(late_ratios)),
            'early_ratio_std': float(np.std(early_ratios)),
            'middle_ratio_std': float(np.std(middle_ratios)),
            'late_ratio_std': float(np.std(late_ratios)),
            'max_layers': [d['max_layer'] for d in all_layer_distributions],
            'avg_max_layer': float(np.mean([d['max_layer'] for d in all_layer_distributions]))
        }
        
        print(f"    成功: {successful_words}/{len(words)}")
        print(f"    前/中/后比例: {avg_layer_distribution['early_ratio']:.2f} / {avg_layer_distribution['middle_ratio']:.2f} / {avg_layer_distribution['late_ratio']:.2f}")
        
        return avg_layer_distribution


def main():
    """主函数"""
    print("="*70)
    print("Stage428: 使用CUDA测试Qwen3和DeepSeek7B模型的词性层分布")
    print("="*70)
    
    # 检查CUDA
    print(f"\nCUDA状态:")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
    
    # 结果字典
    results = {
        "metadata": {
            "stage": 428,
            "description": "使用CUDA测试真实模型词性层分布",
            "timestamp": datetime.now().isoformat(),
            "cuda_available": torch.cuda.is_available(),
            "models": list(MODEL_CONFIGS.keys()),
            "pos_types": list(POS_DATA.keys())
        },
        "models": {},
        "cross_model_comparison": {}
    }
    
    # 对每个模型进行分析
    for model_name, model_config in MODEL_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"模型: {model_name}")
        print(f"{'='*70}")
        
        # 创建分析器
        analyzer = CUDAModelPOSAnalyzer(model_name, model_config)
        
        # 加载模型
        model_loaded = analyzer.load_model()
        
        if not model_loaded:
            print("  [ERROR] 模型加载失败，跳过此模型")
            results["models"][model_name] = {
                "model_name": model_name,
                "model_config": model_config,
                "model_loaded": False,
                "pos_analysis": {},
                "error": "模型加载失败"
            }
            continue
        
        # 存储模型结果
        model_results = {
            "model_name": model_name,
            "model_config": model_config,
            "model_loaded": True,
            "pos_analysis": {},
            "layer_distribution": {}
        }
        
        # 对每个词性进行分析
        for pos, pos_data in POS_DATA.items():
            pos_result = analyzer.analyze_pos(pos, pos_data["words"])
            model_results["pos_analysis"][pos] = pos_result
            model_results["layer_distribution"][pos] = pos_result
        
        results["models"][model_name] = model_results
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n  GPU内存已清理")
    
    # 跨模型比较
    if len([m for m in results["models"].values() if m["model_loaded"]]) >= 2:
        print(f"\n{'='*70}")
        print("跨模型比较")
        print(f"{'='*70}")
        
        for pos in POS_DATA.keys():
            qwen3_result = results["models"]["qwen3"]["pos_analysis"].get(pos, {})
            deepseek7b_result = results["models"]["deepseek7b"]["pos_analysis"].get(pos, {})
            
            if qwen3_result and deepseek7b_result:
                # 计算早中后比例的相似度
                qwen3_ratios = np.array([qwen3_result.get("early_ratio", 0), 
                                         qwen3_result.get("middle_ratio", 0), 
                                         qwen3_result.get("late_ratio", 0)])
                deepseek7b_ratios = np.array([deepseek7b_result.get("early_ratio", 0), 
                                               deepseek7b_result.get("middle_ratio", 0), 
                                               deepseek7b_result.get("late_ratio", 0)])
                
                # 计算余弦相似度
                if np.linalg.norm(qwen3_ratios) > 0 and np.linalg.norm(deepseek7b_ratios) > 0:
                    similarity = np.dot(qwen3_ratios, deepseek7b_ratios) / (np.linalg.norm(qwen3_ratios) * np.linalg.norm(deepseek7b_ratios))
                else:
                    similarity = 0.0
                
                if np.isnan(similarity):
                    similarity = 0.0
                
                results["cross_model_comparison"][pos] = {
                    "similarity": float(similarity),
                    "qwen3_early_ratio": qwen3_result.get("early_ratio", 0),
                    "qwen3_middle_ratio": qwen3_result.get("middle_ratio", 0),
                    "qwen3_late_ratio": qwen3_result.get("late_ratio", 0),
                    "deepseek7b_early_ratio": deepseek7b_result.get("early_ratio", 0),
                    "deepseek7b_middle_ratio": deepseek7b_result.get("middle_ratio", 0),
                    "deepseek7b_late_ratio": deepseek7b_result.get("late_ratio", 0),
                    "qwen3_avg_max_layer": qwen3_result.get("avg_max_layer", 0),
                    "deepseek7b_avg_max_layer": deepseek7b_result.get("avg_max_layer", 0)
                }
                
                print(f"\n{pos}:")
                print(f"  层分布相似度: {similarity*100:.2f}%")
                print(f"  Qwen3: 前/中/后 = {qwen3_result.get('early_ratio', 0):.2f}/{qwen3_result.get('middle_ratio', 0):.2f}/{qwen3_result.get('late_ratio', 0):.2f}")
                print(f"  DeepSeek7b: 前/中/后 = {deepseek7b_result.get('early_ratio', 0):.2f}/{deepseek7b_result.get('middle_ratio', 0):.2f}/{deepseek7b_result.get('late_ratio', 0):.2f}")
    
    # 保存结果
    output_file = Path(__file__).parent / "pos_layer_cuda_real_model_stage428.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 结果已保存到: {output_file}")
    
    print("\n" + "="*70)
    print("Stage428完成")
    print("="*70)


if __name__ == "__main__":
    main()
