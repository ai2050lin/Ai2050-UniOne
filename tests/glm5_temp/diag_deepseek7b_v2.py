"""DeepSeek7B v2 诊断 — 对比GPT5成功方式 vs GLM5失败方式"""
import torch
import os, sys, time, gc, traceback, psutil
import functools
print = functools.partial(print, flush=True)

model_path = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"

print("=" * 60)
print("DeepSeek7B v2 诊断: 对比GPT5成功方式 vs GLM5失败方式")
print("=" * 60)

# 1. 基础环境
print("\n[1] 基础环境")
print(f"  Python: {sys.version}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM total: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB")
    print(f"  VRAM free: {(torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
mem = psutil.virtual_memory()
print(f"  CPU RAM: {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB ({mem.percent}%)")

# 2. GLM5方式 (phase_xliii脚本): device_map="cpu" -> .to("cuda")
print("\n[2] GLM5方式: device_map='cpu' -> .to('cuda')")
print("  问题: CPU加载需要大量RAM(约28GB), 页面文件可能不足")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    mem_before = psutil.virtual_memory()
    print(f"  [2.1] 加载前 RAM: {mem_before.used/1e9:.1f}/{mem_before.total/1e9:.1f} GB ({mem_before.percent}%)")
    
    t0 = time.time()
    print("  [2.2] 加载到CPU...")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", device_map="cpu"
    )
    t1 = time.time()
    mem_after_cpu = psutil.virtual_memory()
    print(f"  [2.3] CPU加载完成: 耗时={t1-t0:.1f}s, RAM: {mem_after_cpu.used/1e9:.1f}GB ({mem_after_cpu.percent}%)")
    
    print("  [2.4] 移动到GPU...")
    mdl = mdl.to("cuda")
    mdl.eval()
    t2 = time.time()
    gpu_mem = torch.cuda.memory_allocated(0) / 1e9
    print(f"  [2.5] GPU移动完成: 耗时={t2-t1:.1f}s, GPU VRAM: {gpu_mem:.2f}GB")
    
    # 简单推理
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    inputs = tok("The cat sat", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = mdl(**inputs, output_hidden_states=True)
    print(f"  [2.6] 推理OK: n_hs={len(out.hidden_states)}, L0 shape={out.hidden_states[0].shape}")
    
    del mdl, out, inputs
    torch.cuda.empty_cache()
    gc.collect()
    print("  [2.7] 清理完成")
    
except Exception as e:
    print(f"  失败: {e}")
    traceback.print_exc()
    try: del mdl
    except: pass
    torch.cuda.empty_cache()
    gc.collect()

# 3. GPT5方式 (stage573脚本): device_map="cpu" + low_cpu_mem_usage=True
print("\n[3] GPT5方式: device_map='cpu' + low_cpu_mem_usage=True")
print("  优势: low_cpu_mem_usage减少CPU RAM峰值, 避免页面文件溢出")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    mem_before = psutil.virtual_memory()
    print(f"  [3.1] 加载前 RAM: {mem_before.used/1e9:.1f}/{mem_before.total/1e9:.1f} GB ({mem_before.percent}%)")
    
    t0 = time.time()
    print("  [3.2] 加载到CPU (low_cpu_mem_usage=True)...")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True,
        trust_remote_code=True, low_cpu_mem_usage=True,
        dtype=torch.bfloat16, attn_implementation="eager",
        device_map="cpu"
    )
    t1 = time.time()
    mem_after_cpu = psutil.virtual_memory()
    print(f"  [3.3] CPU加载完成: 耗时={t1-t0:.1f}s, RAM: {mem_after_cpu.used/1e9:.1f}GB ({mem_after_cpu.percent}%)")
    
    print("  [3.4] 移动到GPU...")
    mdl = mdl.to("cuda")
    mdl.eval()
    t2 = time.time()
    gpu_mem = torch.cuda.memory_allocated(0) / 1e9
    print(f"  [3.5] GPU移动完成: 耗时={t2-t1:.1f}s, GPU VRAM: {gpu_mem:.2f}GB")
    
    # 简单推理
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    inputs = tok("The cat sat", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = mdl(**inputs, output_hidden_states=True)
    print(f"  [3.6] 推理OK: n_hs={len(out.hidden_states)}, L0 shape={out.hidden_states[0].shape}")
    
    # FFN权重提取测试
    print("  [3.7] FFN权重提取测试...")
    layers = mdl.model.layers
    mlp0 = layers[0].mlp
    print(f"    MLP属性: {[a for a in dir(mlp0) if not a.startswith('_')]}")
    if hasattr(mlp0, 'gate_proj'):
        print(f"    gate_proj.weight: {mlp0.gate_proj.weight.shape}")
        print(f"    up_proj.weight: {mlp0.up_proj.weight.shape}")
        print(f"    down_proj.weight: {mlp0.down_proj.weight.shape}")
    
    # 模拟Phase XLIII的大规模数据收集
    print("  [3.8] 模拟Phase XLIII数据收集(60词×7模板)...")
    words = ["apple","banana","cat","dog","car","red","sweet","big"] * 8  # 64词
    templates = ["The {w} is", "A {w} can be", "This {w} has"]
    t3 = time.time()
    for i, w in enumerate(words[:16]):  # 先测16词
        for t in templates:
            prompt = t.replace("{w}", w)
            inputs = tok(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = mdl(**inputs, output_hidden_states=True, use_cache=False)
            hs = [h[0, -1].float().cpu() for h in out.hidden_states]
            del out, inputs, hs
        if (i+1) % 4 == 0:
            gpu_m = torch.cuda.memory_allocated(0) / 1e9
            print(f"    [{i+1}/16] GPU={gpu_m:.2f}GB")
    
    t4 = time.time()
    print(f"  [3.9] 16词收集完成: 耗时={t4-t3:.1f}s")
    
    del mdl, out
    torch.cuda.empty_cache()
    gc.collect()
    print("  [3.10] 清理完成")
    
except Exception as e:
    print(f"  失败: {e}")
    traceback.print_exc()
    try: del mdl
    except: pass
    torch.cuda.empty_cache()
    gc.collect()

# 4. device_map="auto" 测试
print("\n[4] device_map='auto' 测试")
print("  注意: 04-09 02:40记录显示此方式可能导致ACCESS_VIOLATION")
try:
    from transformers import AutoModelForCausalLM
    torch.cuda.empty_cache()
    gc.collect()
    
    print("  [4.1] 加载模型(device_map=auto)...")
    t0 = time.time()
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", device_map="auto"
    )
    mdl.eval()
    t1 = time.time()
    gpu_mem = torch.cuda.memory_allocated(0) / 1e9
    device = next(mdl.parameters()).device
    print(f"  [4.2] 加载完成: 耗时={t1-t0:.1f}s, GPU={gpu_mem:.2f}GB, device={device}")
    
    inputs = tok("The cat sat", return_tensors="pt").to(device)
    with torch.no_grad():
        out = mdl(**inputs, output_hidden_states=True)
    print(f"  [4.3] 推理OK: n_hs={len(out.hidden_states)}")
    
    del mdl, out
    torch.cuda.empty_cache()
    gc.collect()
    
except Exception as e:
    print(f"  失败(符合预期): {e}")
    traceback.print_exc()
    try: del mdl
    except: pass
    torch.cuda.empty_cache()
    gc.collect()

# 5. 总结
print("\n" + "=" * 60)
print("诊断总结:")
print("=" * 60)
print("""
GLM5脚本(phase_xliii) vs GPT5脚本(stage573) 关键差异:

1. low_cpu_mem_usage:
   - GLM5: 未使用 → CPU加载时RAM峰值高(~28GB), 容易触发os error 1455
   - GPT5: 使用low_cpu_mem_usage=True → 峰值降低, 使用分片加载
   
2. local_files_only:
   - GLM5: 未使用 → 可能触发网络请求, 增加不确定性
   - GPT5: 使用local_files_only=True → 纯本地加载, 更稳定
   
3. use_fast (tokenizer):
   - GLM5: 未指定(默认True) → 可能使用fast tokenizer
   - GPT5: use_fast=False → 使用slow tokenizer, 更兼容
   
4. 命令超时:
   - GLM5: CPU加载+GPU迁移约需2-3分钟, Phase XLIII总运行可能超过10分钟
   - GPT5: stage573同样方式, 但命令不被跳过
   
根本原因推断:
   1. 主要: 缺少low_cpu_mem_usage=True导致CPU RAM峰值过高
   2. 次要: 命令运行时间过长被系统跳过(不是模型本身的问题)
   3. 验证: GPT5侧的deepseek7b完全正常, 模型本身无问题
""")
