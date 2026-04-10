"""DeepSeek7B 诊断脚本 — 排查加载和运行问题"""
import torch
import os, sys, time, traceback

# 强制flush
import functools
print = functools.partial(print, flush=True)

model_path = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"

print("=" * 60)
print("DeepSeek7B 诊断脚本")
print("=" * 60)

# 1. 基础环境检查
print("\n[1] 基础环境检查")
print(f"  Python: {sys.version}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"  VRAM free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")

# 2. 模型文件检查
print("\n[2] 模型文件检查")
if os.path.isdir(model_path):
    print(f"  模型目录存在: {model_path}")
    for f in os.listdir(model_path):
        fpath = os.path.join(model_path, f)
        if os.path.isfile(fpath):
            size_gb = os.path.getsize(fpath) / 1e9
            print(f"  {f}: {size_gb:.2f} GB")
else:
    print(f"  错误: 模型目录不存在!")
    sys.exit(1)

# 3. 加载tokenizer
print("\n[3] 加载Tokenizer")
try:
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  OK! vocab_size={tok.vocab_size}, 耗时={time.time()-t0:.1f}s")
except Exception as e:
    print(f"  失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 4. 方式A: device_map="cpu" -> .to("cuda") (Phase XLIII方式)
print("\n[4] 方式A: device_map='cpu' -> .to('cuda')")
try:
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    
    torch.cuda.empty_cache()
    gc_start = torch.cuda.memory_allocated(0)
    
    print("  [4.1] 加载到CPU...")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", device_map="cpu"
    )
    print(f"  [4.2] CPU加载完成, 耗时={time.time()-t0:.1f}s")
    
    # 检查CPU内存
    import psutil
    mem = psutil.virtual_memory()
    print(f"  [4.3] CPU RAM: {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB used ({mem.percent}%)")
    
    print("  [4.4] 移动到GPU...")
    t1 = time.time()
    mdl = mdl.to("cuda")
    mdl.eval()
    print(f"  [4.5] GPU移动完成, 耗时={time.time()-t1:.1f}s")
    
    gpu_mem = torch.cuda.memory_allocated(0) / 1e9
    print(f"  [4.6] GPU VRAM: {gpu_mem:.2f} GB")
    
    # 简单前向传播
    print("  [4.7] 前向传播测试...")
    inputs = tok("The cat sat", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = mdl(**inputs, output_hidden_states=True)
    print(f"  [4.8] 前向传播OK, n_hs={len(out.hidden_states)}")
    
    # FFN权重提取测试
    print("  [4.9] FFN权重提取测试...")
    layers = mdl.model.layers
    mlp0 = layers[0].mlp
    has_gate = hasattr(mlp0, 'gate_proj')
    has_W_gate = hasattr(mlp0, 'W_gate')
    print(f"  [4.10] MLP结构: has_gate_proj={has_gate}, has_W_gate={has_W_gate}")
    if has_gate:
        print(f"    gate_proj.weight: {mlp0.gate_proj.weight.shape}")
        print(f"    up_proj.weight: {mlp0.up_proj.weight.shape}")
        print(f"    down_proj.weight: {mlp0.down_proj.weight.shape}")
    
    del mdl, out
    torch.cuda.empty_cache()
    print("  [4.11] 模型释放完成")
    
except Exception as e:
    print(f"  失败: {e}")
    traceback.print_exc()
    try:
        del mdl
    except:
        pass
    torch.cuda.empty_cache()

# 5. 方式B: device_map="auto" (stage745方式)
print("\n[5] 方式B: device_map='auto'")
try:
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    
    torch.cuda.empty_cache()
    
    print("  [5.1] 加载模型(device_map=auto)...")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", device_map="auto"
    )
    mdl.eval()
    print(f"  [5.2] 加载完成, 耗时={time.time()-t0:.1f}s")
    
    gpu_mem = torch.cuda.memory_allocated(0) / 1e9
    print(f"  [5.3] GPU VRAM: {gpu_mem:.2f} GB")
    
    device = next(mdl.parameters()).device
    print(f"  [5.4] 模型设备: {device}")
    
    # 简单前向传播
    print("  [5.5] 前向传播测试...")
    inputs = tok("The cat sat", return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl(**inputs, output_hidden_states=True)
    print(f"  [5.6] 前向传播OK, n_hs={len(out.hidden_states)}")
    
    del mdl, out
    torch.cuda.empty_cache()
    print("  [5.7] 模型释放完成")
    
except Exception as e:
    print(f"  失败: {e}")
    traceback.print_exc()
    try:
        del mdl
    except:
        pass
    torch.cuda.empty_cache()

# 6. 内存峰值测试 (模拟Phase XLIII的数据收集)
print("\n[6] 内存峰值测试 (模拟数据收集)")
try:
    from transformers import AutoModelForCausalLM
    
    torch.cuda.empty_cache()
    
    print("  [6.1] 加载模型...")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", device_map="auto"
    )
    mdl.eval()
    device = next(mdl.parameters()).device
    
    # 模拟Phase XLIII的数据收集: 逐词收集hidden states
    test_words = ["apple", "banana", "cat", "dog", "car", "red", "sweet", "big"]
    templates = ["The {word} is", "A {word} can be", "This {word} has"]
    
    print(f"  [6.2] 收集{len(test_words)}词×{len(templates)}模板的hidden states...")
    t0 = time.time()
    
    all_hs = {}
    for i, word in enumerate(test_words):
        for template in templates:
            prompt = template.replace("{word}", word)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl(**inputs, output_hidden_states=True)
            hs = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
            if word not in all_hs:
                all_hs[word] = hs
            else:
                all_hs[word] = all_hs[word] + hs
            del out
        
        all_hs[word] = all_hs[word] / len(templates)
        
        gpu_mem = torch.cuda.memory_allocated(0) / 1e9
        cpu_mem = psutil.virtual_memory()
        print(f"    [{i+1}/{len(test_words)}] {word}: GPU={gpu_mem:.2f}GB, CPU={cpu_mem.percent}%")
    
    print(f"  [6.3] 数据收集完成, 耗时={time.time()-t0:.1f}s")
    
    # 模拟FFN权重提取
    print("  [6.4] FFN权重提取...")
    layers = mdl.model.layers
    for li in [0, len(layers)//2, len(layers)-1]:
        mlp = layers[li].mlp
        if hasattr(mlp, 'gate_proj'):
            W_gate = mlp.gate_proj.weight.detach().float().cpu()
            W_in = mlp.up_proj.weight.detach().float().cpu()
            W_out = mlp.down_proj.weight.detach().float().cpu()
            print(f"    L{li}: W_gate={W_gate.shape}, W_in={W_in.shape}, W_out={W_out.shape}")
            del W_gate, W_in, W_out
    
    del mdl, all_hs
    torch.cuda.empty_cache()
    print("  [6.5] 清理完成")
    
except Exception as e:
    print(f"  失败: {e}")
    traceback.print_exc()
    try:
        del mdl
    except:
        pass
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("诊断完成!")
print("=" * 60)
