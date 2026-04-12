"""探测单个模型结构"""
import torch
import sys
import os

def probe_model(name, path):
    print(f"Loading {name}...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="auto",
        local_files_only=True, low_cpu_mem_usage=True,
    )
    
    out_lines = []
    out_lines.append(f"Model: {name}")
    out_lines.append(f"  class: {type(model).__name__}")
    
    # Find layers
    layers = None
    layer_path = ""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        layer_path = "model.model.layers"
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        layer_path = "transformer.h"
    
    if layers is None:
        out_lines.append("  ERROR: Cannot find layers!")
        if hasattr(model, "model"):
            attrs = [a for a in dir(model.model) if not a.startswith("_")]
            out_lines.append(f"  model.model attrs: {attrs[:30]}")
        result = "\n".join(out_lines)
        print(result)
        with open("tests/glm5_temp/probe_result.txt", "a", encoding="utf-8") as f:
            f.write(result + "\n\n")
        del model
        torch.cuda.empty_cache()
        return
    
    out_lines.append(f"  layers path: {layer_path}")
    out_lines.append(f"  num layers: {len(layers)}")
    
    layer0 = layers[0]
    out_lines.append(f"  Layer0 class: {type(layer0).__name__}")
    
    # Self-attention
    if hasattr(layer0, "self_attn"):
        sa = layer0.self_attn
        out_lines.append(f"  self_attn class: {type(sa).__name__}")
        for w in ["q_proj", "k_proj", "v_proj", "o_proj", "query_key_value"]:
            if hasattr(sa, w):
                shape = tuple(getattr(sa, w).weight.shape)
                out_lines.append(f"    {w}: {shape}")
    
    # MLP
    if hasattr(layer0, "mlp"):
        mlp = layer0.mlp
        out_lines.append(f"  mlp class: {type(mlp).__name__}")
        for w in ["up_proj", "down_proj", "gate_proj", "dense_h_to_4h", "dense_4h_to_h"]:
            if hasattr(mlp, w):
                shape = tuple(getattr(mlp, w).weight.shape)
                out_lines.append(f"    {w}: {shape}")
    
    # LayerNorm
    for ln in ["input_layernorm", "layernorm", "ln_1", "post_attention_layernorm", "ln_2"]:
        if hasattr(layer0, ln):
            out_lines.append(f"    {ln}: {type(getattr(layer0, ln)).__name__}")
    
    # lm_head
    if hasattr(model, "lm_head"):
        out_lines.append(f"  lm_head shape: {model.lm_head.weight.shape}")
    
    embed = model.get_input_embeddings()
    out_lines.append(f"  embed shape: {embed.weight.shape}")
    
    if hasattr(model, "lm_head"):
        tied = torch.equal(model.lm_head.weight.data_ptr(), embed.weight.data_ptr())
        out_lines.append(f"  lm_head tied with embed: {tied}")
    
    result = "\n".join(out_lines)
    print(result)
    with open("tests/glm5_temp/probe_result.txt", "a", encoding="utf-8") as f:
        f.write(result + "\n\n")
    
    del model
    torch.cuda.empty_cache()
    print(f"{name} done, GPU released.")

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "glm4"
    
    MODELS = {
        "glm4": "D:/develop/model/hub/models--zai-org--GLM-4-9B-Chat-HF/snapshots/8599336fc6c125203efb2360bfaf4c80eef1d1bf",
        "deepseek7b": "D:/develop/model/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    }
    
    if model_name in MODELS:
        probe_model(model_name, MODELS[model_name])
    else:
        print(f"Unknown model: {model_name}")
