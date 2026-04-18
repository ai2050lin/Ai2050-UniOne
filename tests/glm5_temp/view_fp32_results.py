"""查看FP32测试结果并与FP16对比"""
import json
import numpy as np
from pathlib import Path

def safe_str(s):
    """安全转字符串,避免编码问题"""
    try:
        return str(s).encode('ascii', errors='replace').decode('ascii')
    except:
        return repr(s)

FP16_DIR = Path("d:/Ai2050/TransformerLens-Project/results/clix_adversarial_balance")
FP32_DIR = Path("d:/Ai2050/TransformerLens-Project/results/clix_adversarial_balance_fp32")

WORDS = ["apple", "banana", "cat", "dog", "run", "red", "the", "is", "beautiful", "mountain"]

for model_key in ["qwen3"]:  # 先只看已完成的
    fp32_path = FP32_DIR / f"clix_fp32_{model_key}_results.json"
    fp16_path = FP16_DIR / f"clix_{model_key}_results.json"
    
    if not fp32_path.exists():
        print(f"{model_key} FP32: not found")
        continue
    
    with open(fp32_path, encoding='utf-8') as f:
        fp32 = json.load(f)
    
    fp16 = None
    if fp16_path.exists():
        with open(fp16_path, encoding='utf-8') as f:
            fp16 = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"{model_key} FP32 Results")
    print(f"{'='*80}")
    
    r32 = fp32["results"]
    
    for word in WORDS:
        if word not in r32:
            continue
        d = r32[word]
        print(f"\n{word}:")
        print(f"  final_logit={d['final_logit']:.4f}, cumG={d['cum_G']:.1f}, cumA={d['cum_A']:.1f}")
        print(f"  G/A_ratio={d['G_A_ratio']:.4f}")
        print(f"  global_balance={d['global_balance']:.6f}")
        print(f"  global_cos_GA={d['global_cos_GA']:.6f}")
        print(f"  mean_cos_GA={d['mean_cos_GA']:.6f}")
        print(f"  same={d['same_sign_count']}, opp={d['opposite_sign_count']}")
        
        iv = d["interventions"]
        print(f"  normal:  prob={iv['normal']['prob']:.8f}")
        print(f"  only_A:  prob={iv['only_A']['prob']:.8f} (d={iv['only_A']['prob_delta']:+.8f})")
        print(f"  only_G:  prob={iv['only_G']['prob']:.8f} (d={iv['only_G']['prob_delta']:+.8f})")
        print(f"  flip_G:  prob={iv['flip_G']['prob']:.8f} (d={iv['flip_G']['prob_delta']:+.8f})")
        print(f"  amp_G:   prob={iv['amplify_G']['prob']:.8f} (d={iv['amplify_G']['prob_delta']:+.8f})")
        print(f"  last3:   prob={iv['last3_only']['prob']:.2e} (d={iv['last3_only']['prob_delta']:+.2e})")
    
    # FP32 vs FP16对比
    if fp16:
        r16 = fp16["results"]
        print(f"\n{'='*80}")
        print(f"{model_key}: FP32 vs FP16 Diff")
        print(f"{'='*80}")
        print(f"{'Word':<12} {'Metric':<20} {'FP16':>14} {'FP32':>14} {'Diff%':>10}")
        print("-"*74)
        
        for word in WORDS:
            if word not in r16 or word not in r32:
                continue
            f16 = r16[word]
            f32 = r32[word]
            
            for metric in ["final_logit", "global_balance", "global_cos_GA", "G_A_ratio", 
                          "mean_balance_ratio", "mean_cos_GA"]:
                v16 = f16.get(metric, 0)
                v32 = f32.get(metric, 0)
                diff_pct = (v32 - v16) / abs(v16) * 100 if abs(v16) > 1e-10 else 0
                print(f"{word:<12} {metric:<20} {v16:>14.6f} {v32:>14.6f} {diff_pct:>+10.3f}%")
            
            # 干预概率对比
            for iv_name in ["only_A", "only_G", "flip_G", "amplify_G"]:
                v16 = f16["interventions"][iv_name]["prob"]
                v32 = f32["interventions"][iv_name]["prob"]
                if abs(v16) > 1e-15:
                    diff_pct = (v32 - v16) / abs(v16) * 100
                else:
                    diff_pct = 0
                print(f"{word:<12} prob_{iv_name:<14} {v16:>14.2e} {v32:>14.2e} {diff_pct:>+10.1f}%")
