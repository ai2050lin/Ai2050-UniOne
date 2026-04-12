"""探测模型结构，生成兼容性报告"""
import torch
import sys
from transformers import AutoModelForCausalLM

MODELS = {
    "glm4": "D:/develop/model/hub/models--zai-org--GLM-4-9B-Chat-HF/snapshots/8599336fc6c125203efb2360bfaf4c80eef1d1bf",
    "deepseek7b": "D:/develop/model/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
}

out = open("tests/glm5_temp/model_structure_report.txt", "w", encoding="utf-8")

for name, path in MODELS.items():
    out.write(f"\n{'='*60}\n")
    out.write(f"Model: {name}\n")
    out.write(f"{'='*60}\n")
    sys.stdout.write(f"Loading {name}...\n")
    sys.stdout.flush()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto",
            local_files_only=True, low_cpu_mem_usage=True,
        )
        out.write(f"  class: {type(model).__name__}\n")
        
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
            out.write("  ERROR: Cannot find layers!\n")
            # Try to list model.model attrs
            if hasattr(model, "model"):
                attrs = [a for a in dir(model.model) if not a.startswith("_")]
                out.write(f"  model.model attrs: {attrs[:30]}\n")
            del model
            torch.cuda.empty_cache()
            continue
        
        out.write(f"  layers path: {layer_path}\n")
        out.write(f"  num layers: {len(layers)}\n")
        
        # Layer 0 structure
        layer0 = layers[0]
        out.write(f"  Layer0 class: {type(layer0).__name__}\n")
        
        # Self-attention
        if hasattr(layer0, "self_attn"):
            sa = layer0.self_attn
            out.write(f"  self_attn class: {type(sa).__name__}\n")
            for w in ["q_proj", "k_proj", "v_proj", "o_proj", "query_key_value"]:
                if hasattr(sa, w):
                    shape = tuple(getattr(sa, w).weight.shape)
                    out.write(f"    {w}: {shape}\n")
        
        # MLP
        if hasattr(layer0, "mlp"):
            mlp = layer0.mlp
            out.write(f"  mlp class: {type(mlp).__name__}\n")
            for w in ["up_proj", "down_proj", "gate_proj", "dense_h_to_4h", "dense_4h_to_h"]:
                if hasattr(mlp, w):
                    shape = tuple(getattr(mlp, w).weight.shape)
                    out.write(f"    {w}: {shape}\n")
        
        # LayerNorm
        for ln in ["input_layernorm", "layernorm", "ln_1", "post_attention_layernorm", "ln_2"]:
            if hasattr(layer0, ln):
                out.write(f"    {ln}: {type(getattr(layer0, ln)).__name__}\n")
        
        # lm_head
        if hasattr(model, "lm_head"):
            out.write(f"  lm_head shape: {model.lm_head.weight.shape}\n")
        
        # embed
        embed = model.get_input_embeddings()
        out.write(f"  embed shape: {embed.weight.shape}\n")
        
        # Check if lm_head and embed are tied
        if hasattr(model, "lm_head"):
            tied = torch.equal(model.lm_head.weight.data_ptr(), embed.weight.data_ptr())
            out.write(f"  lm_head tied with embed: {tied}\n")
        
        del model
        torch.cuda.empty_cache()
        out.write(f"  GPU memory released\n")
        sys.stdout.write(f"{name} done.\n")
        sys.stdout.flush()
        
    except Exception as e:
        out.write(f"  ERROR: {e}\n")
        import traceback
        out.write(traceback.format_exc())
        torch.cuda.empty_cache()
        sys.stdout.write(f"{name} FAILED: {e}\n")
        sys.stdout.flush()

out.write("\nDone!\n")
out.close()
print("Report saved to tests/glm5_temp/model_structure_report.txt")
