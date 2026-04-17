"""
Phase CXC-ProBE: GLM-4-9B-Chat-HF 4-bit量化探针测试
======================================================
目的: 验证GLM-4-9B在4-bit量化下是否可以正常运行探针测试

测试内容:
1. 模型加载 (4-bit NF4量化, bitsandbytes)
2. 基本前向传播 + hooks 捕获层输出
3. 功能差异探针: happy vs sad, cat vs dog
4. 残差流范数增长 + cos相似度衰减
5. 与Qwen3-4B对比

运行条件:
- 需要先下载GLM-4-9B-Chat-HF模型 (18GB FP16, 在线4-bit量化)
- bitsandbytes已安装 (0.49.2)
- RTX 5070 12GB显存
"""

import sys
import os
import time
import json
import torch
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.insert(0, r'd:\Ai2050\TransformerLens-Project')

# ===== 配置 =====
GLM4_MODEL_PATH = r'D:\develop\model\hub\models--zai-org--glm-4-9b-chat-hf\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf'
QWEN3_MODEL_PATH = r'D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c'
RESULTS_DIR = r'd:\Ai2050\TransformerLens-Project\results\phase_glm4_4bit_probe'

# 测试句对
TEST_PAIRS = {
    "emotion": {
        "pos": "She is very happy today",
        "neg": "She is very sad today",
    },
    "animal": {
        "pos": "The cat sat on the mat",
        "neg": "The dog sat on the mat",
    },
    "logic": {
        "pos": "Because it rained heavily, the ground is completely wet and flooded",
        "neg": "The ground is wet and flooded, although it did not rain at all",
    },
}

def load_model_4bit(model_path, model_name="glm4"):
    """加载4-bit量化模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading {model_name} with 4-bit quantization...")
    t0 = time.time()
    
    # 4-bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",        # NF4: 最优4-bit量化
        bnb_4bit_use_double_quant=True,    # 双量化进一步节省显存
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    t1 = time.time()
    mem_alloc = torch.cuda.memory_allocated(0) / 1e9
    mem_reserved = torch.cuda.memory_reserved(0) / 1e9
    
    print(f"  Loaded in {t1-t0:.1f}s")
    print(f"  GPU allocated: {mem_alloc:.2f} GB")
    print(f"  GPU reserved: {mem_reserved:.2f} GB")
    print(f"  Model class: {type(model).__name__}")
    
    return model, tokenizer


def load_model_fp16(model_path, model_name="qwen3"):
    """加载FP16模型(用于对比)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading {model_name} (FP16)...")
    t0 = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    t1 = time.time()
    mem_alloc = torch.cuda.memory_allocated(0) / 1e9
    print(f"  Loaded in {t1-t0:.1f}s, GPU: {mem_alloc:.2f} GB")
    
    return model, tokenizer


def get_layers(model):
    """获取transformer层列表"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


def run_probe_test(model, tokenizer, model_name, test_pairs):
    """
    运行探针测试: 提取各层残差流, 计算功能差异
    
    Returns:
        dict: {pair_name: {layer_idx: {cos, norm_pos, norm_neg, delta_norm}}}
    """
    layers = get_layers(model)
    n_layers = len(layers)
    device = next(model.parameters()).device
    
    # 采样层
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    sample_layers = sorted(set(sample_layers))
    
    print(f"\n  Model: {model_name}, Layers: {n_layers}, Sample layers: {sample_layers}")
    
    results = {}
    
    for pair_name, pair in test_pairs.items():
        pos_sent = pair["pos"]
        neg_sent = pair["neg"]
        
        print(f"\n  Pair: {pair_name}")
        print(f"    POS: {pos_sent}")
        print(f"    NEG: {neg_sent}")
        
        # 注册hooks
        captured = {}
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu().numpy()
                else:
                    captured[key] = output.detach().float().cpu().numpy()
            return hook
        
        hooks = []
        for li in sample_layers:
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
        
        # 前向传播 - 正例
        captured.clear()
        inputs_pos = tokenizer(pos_sent, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            model(**inputs_pos)
        pos_outputs = {k: v.copy() for k, v in captured.items()}
        
        # 前向传播 - 反例
        captured.clear()
        inputs_neg = tokenizer(neg_sent, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            model(**inputs_neg)
        neg_outputs = {k: v.copy() for k, v in captured.items()}
        
        # 移除hooks
        for h in hooks:
            h.remove()
        
        # 计算功能差异指标
        pair_results = {}
        for li in sample_layers:
            key = f"L{li}"
            if key not in pos_outputs or key not in neg_outputs:
                continue
            
            pos_h = pos_outputs[key]  # [1, seq_len, d_model]
            neg_h = neg_outputs[key]
            
            # 取last token的隐藏状态
            pos_vec = pos_h[0, -1, :]  # [d_model]
            neg_vec = neg_h[0, -1, :]
            
            # 计算指标
            norm_pos = float(np.linalg.norm(pos_vec))
            norm_neg = float(np.linalg.norm(neg_vec))
            delta = pos_vec - neg_vec
            delta_norm = float(np.linalg.norm(delta))
            
            # 余弦相似度
            cos_val = 0.0
            if norm_pos > 1e-10 and norm_neg > 1e-10:
                cos_val = float(np.dot(pos_vec, neg_vec) / (norm_pos * norm_neg))
            
            pair_results[li] = {
                "cos": round(cos_val, 6),
                "norm_pos": round(norm_pos, 2),
                "norm_neg": round(norm_neg, 2),
                "delta_norm": round(delta_norm, 2),
            }
            
            print(f"    L{li}: cos={cos_val:.6f}, norm_pos={norm_pos:.2f}, "
                  f"norm_neg={norm_neg:.2f}, delta_norm={delta_norm:.2f}")
        
        results[pair_name] = pair_results
    
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 70)
    print("Phase CXC-PROBE: GLM-4-9B-Chat-HF 4-bit量化探针测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 检查模型文件是否存在
    if not os.path.exists(GLM4_MODEL_PATH):
        print(f"\n[ERROR] GLM-4-9B-Chat-HF model not found at:")
        print(f"  {GLM4_MODEL_PATH}")
        print(f"\nPlease download the model first. Run:")
        print(f"  python tests/glm5_temp/download_glm4.py")
        print(f"\nOr wait for the background download to complete.")
        
        # 检查下载进度
        blobs_dir = os.path.join(os.path.dirname(GLM4_MODEL_PATH), '..', '..', 'blobs')
        blobs_dir = os.path.normpath(blobs_dir)
        if os.path.exists(blobs_dir):
            total = sum(os.path.getsize(os.path.join(blobs_dir, f)) 
                       for f in os.listdir(blobs_dir))
            print(f"\nDownload progress: {total/1e9:.2f} GB / ~18 GB ({total/18e9*100:.1f}%)")
        
        return
    
    all_results = {}
    
    # ===== 测试1: GLM-4-9B-Chat-HF 4-bit =====
    print("\n" + "=" * 70)
    print("Test 1: GLM-4-9B-Chat-HF (4-bit NF4)")
    print("=" * 70)
    
    try:
        model_glm4, tok_glm4 = load_model_4bit(GLM4_MODEL_PATH, "glm4-9b-chat-hf")
        
        # 模型基本信息
        n_layers = len(get_layers(model_glm4))
        d_model = model_glm4.get_input_embeddings().weight.shape[1]
        print(f"\n  GLM-4-9B: n_layers={n_layers}, d_model={d_model}")
        
        # 运行探针测试
        glm4_results = run_probe_test(model_glm4, tok_glm4, "glm4-4bit", TEST_PAIRS)
        all_results["glm4_4bit"] = glm4_results
        
        # 释放模型
        del model_glm4
        torch.cuda.empty_cache()
        print(f"\n  GLM-4 released. GPU free: {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB")
        
    except Exception as e:
        print(f"\n  [ERROR] GLM-4-9B test failed: {e}")
        import traceback
        traceback.print_exc()
        all_results["glm4_4bit_error"] = str(e)
    
    # ===== 测试2: Qwen3-4B FP16 (对比) =====
    print("\n" + "=" * 70)
    print("Test 2: Qwen3-4B (FP16, 对比)")
    print("=" * 70)
    
    try:
        model_qwen3, tok_qwen3 = load_model_fp16(QWEN3_MODEL_PATH, "qwen3-4b")
        
        n_layers = len(get_layers(model_qwen3))
        d_model = model_qwen3.get_input_embeddings().weight.shape[1]
        print(f"\n  Qwen3-4B: n_layers={n_layers}, d_model={d_model}")
        
        qwen3_results = run_probe_test(model_qwen3, tok_qwen3, "qwen3-fp16", TEST_PAIRS)
        all_results["qwen3_fp16"] = qwen3_results
        
        del model_qwen3
        torch.cuda.empty_cache()
        print(f"\n  Qwen3 released. GPU free: {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB")
        
    except Exception as e:
        print(f"\n  [ERROR] Qwen3-4B test failed: {e}")
        import traceback
        traceback.print_exc()
        all_results["qwen3_fp16_error"] = str(e)
    
    # ===== 保存结果 =====
    results_file = os.path.join(RESULTS_DIR, "probe_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_file}")
    
    # ===== 结果分析 =====
    print("\n" + "=" * 70)
    print("结果分析")
    print("=" * 70)
    
    for model_key in ["glm4_4bit", "qwen3_fp16"]:
        if model_key not in all_results or model_key.endswith("_error"):
            print(f"\n  {model_key}: 测试失败")
            continue
        
        model_res = all_results[model_key]
        print(f"\n  {model_key}:")
        
        for pair_name, pair_res in model_res.items():
            print(f"    {pair_name}:")
            layers_sorted = sorted(pair_res.keys())
            for li in layers_sorted:
                r = pair_res[li]
                print(f"      L{li}: cos={r['cos']:.6f}, delta_norm={r['delta_norm']:.2f}")
    
    # 跨模型对比
    if "glm4_4bit" in all_results and "qwen3_fp16" in all_results:
        print("\n  跨模型对比 (emotion pair, L_last):")
        glm4_emotion = all_results["glm4_4bit"].get("emotion", {})
        qwen3_emotion = all_results["qwen3_fp16"].get("emotion", {})
        
        glm4_last = max(glm4_emotion.keys()) if glm4_emotion else None
        qwen3_last = max(qwen3_emotion.keys()) if qwen3_emotion else None
        
        if glm4_last and qwen3_last:
            print(f"    GLM4 L{glm4_last}: cos={glm4_emotion[glm4_last]['cos']:.6f}, "
                  f"delta={glm4_emotion[glm4_last]['delta_norm']:.2f}")
            print(f"    Qwen3 L{qwen3_last}: cos={qwen3_emotion[qwen3_last]['cos']:.6f}, "
                  f"delta={qwen3_emotion[qwen3_last]['delta_norm']:.2f}")
    
    print(f"\n测试完成! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
