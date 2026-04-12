"""Minimal test to check GLM4 model structure"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"
print("Loading model...", flush=True)
try:
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    print(f"Tokenizer loaded: vocab_size={len(tokenizer)}", flush=True)
    
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    print("Model loaded successfully", flush=True)
    
    # Check structure
    print(f"Model type: {type(model)}", flush=True)
    print(f"Has model.model: {hasattr(model, 'model')}", flush=True)
    if hasattr(model, 'model'):
        print(f"model.model type: {type(model.model)}", flush=True)
        print(f"Has model.model.layers: {hasattr(model.model, 'layers')}", flush=True)
        if hasattr(model.model, 'layers'):
            print(f"n_layers: {len(model.model.layers)}", flush=True)
        print(f"model.model dir: {[x for x in dir(model.model) if not x.startswith('_')][:20]}", flush=True)
    
    # Test get_input_embeddings
    embed = model.get_input_embeddings()
    print(f"Embed type: {type(embed)}", flush=True)
    print(f"Embed shape: {embed.weight.shape}", flush=True)
    
    # Test forward
    inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(inputs["input_ids"], output_hidden_states=True)
    print(f"Forward pass OK, n_hidden_states: {len(out.hidden_states)}", flush=True)
    
    # Test hook
    def make_hs_hook(store, layer_idx):
        def hook_fn(module, args):
            x = args[0] if isinstance(args, tuple) else args
            store[layer_idx] = x[0, -1, :].detach().cpu().float().numpy()
            return args
        return hook_fn
    
    stored = {}
    if hasattr(model.model, 'layers'):
        hook = model.model.layers[0].register_forward_pre_hook(make_hs_hook(stored, 0))
        with torch.no_grad():
            _ = model(inputs["input_ids"])
        hook.remove()
        print(f"Hook test OK, stored shape: {stored[0].shape if 0 in stored else 'FAILED'}", flush=True)
    
    del model
    torch.cuda.empty_cache()
    print("ALL TESTS PASSED", flush=True)
    
except Exception as e:
    import traceback
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
