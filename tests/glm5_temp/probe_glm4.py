"""探测单个模型结构 - 简化版，只探查GLM4"""
import torch
import sys
import traceback

def probe_glm4():
    path = "D:/develop/model/hub/models--zai-org--GLM-4-9B-Chat-HF/snapshots/8599336fc6c125203efb2360bfaf4c80eef1d1bf"
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float16, device_map="auto",
        local_files_only=True, low_cpu_mem_usage=True,
    )
    
    lines = []
    lines.append(f"GLM4 class: {type(model).__name__}")
    
    # Check model.model structure
    if hasattr(model, "model"):
        mm = model.model
        lines.append(f"model.model class: {type(mm).__name__}")
        attrs = [a for a in dir(mm) if not a.startswith("_")]
        lines.append(f"model.model attrs: {attrs[:40]}")
        
        # Try to find layers
        for attr in ["layers", "encoder", "h", "transformer"]:
            if hasattr(mm, attr):
                obj = getattr(mm, attr)
                lines.append(f"  model.model.{attr}: type={type(obj).__name__}, len={len(obj)}")
                if len(obj) > 0:
                    l0 = obj[0]
                    lines.append(f"  Layer0 class: {type(l0).__name__}")
                    l0_attrs = [a for a in dir(l0) if not a.startswith("_")]
                    lines.append(f"  Layer0 attrs: {l0_attrs[:40]}")
                    
                    # self_attn
                    if hasattr(l0, "self_attn"):
                        sa = l0.self_attn
                        lines.append(f"  self_attn: {type(sa).__name__}")
                        for w in ["q_proj", "k_proj", "v_proj", "o_proj", "query_key_value"]:
                            if hasattr(sa, w):
                                lines.append(f"    {w}: {getattr(sa, w).weight.shape}")
                    
                    # mlp
                    if hasattr(l0, "mlp"):
                        mlp = l0.mlp
                        lines.append(f"  mlp: {type(mlp).__name__}")
                        for w in ["up_proj", "down_proj", "gate_proj", "dense_h_to_4h", "dense_4h_to_h"]:
                            if hasattr(mlp, w):
                                lines.append(f"    {w}: {getattr(mlp, w).weight.shape}")
    
    # lm_head
    if hasattr(model, "lm_head"):
        lines.append(f"lm_head: {model.lm_head.weight.shape}")
    
    embed = model.get_input_embeddings()
    lines.append(f"embed: {embed.weight.shape}")
    
    result = "\n".join(lines)
    
    with open("tests/glm5_temp/glm4_probe.txt", "w", encoding="utf-8") as f:
        f.write(result)
    
    del model
    torch.cuda.empty_cache()
    return result

if __name__ == "__main__":
    try:
        result = probe_glm4()
        print("SUCCESS - saved to tests/glm5_temp/glm4_probe.txt")
    except Exception as e:
        with open("tests/glm5_temp/glm4_probe_error.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"ERROR: {e}")
