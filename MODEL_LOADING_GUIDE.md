# 🔧 模型加载问题解决方案

## 问题
服务器在加载大模型（Qwen3-4B）后崩溃，原因是内存不足（Out of Memory）。

## 解决方案

### 1. 改进的模型加载器（已实施）

#### 功能特性
- **自动回退机制**: 如果大模型加载失败，自动切换到 gpt2-small
- **半精度支持**: 使用 float16 减少内存占用（~50%）
- **内存清理**: 加载后自动清理未使用的内存
- **GPU检测**: 自动检测并使用GPU（如果可用）

#### 使用方法
编辑 `server.py` 第53-54行配置：

```python
model_name = "Qwen/Qwen3-4B"  # 修改这里选择模型
use_half_precision = True     # True = float16, False = float32
```

### 2. 推荐配置

#### 对于大模型 (Qwen3-4B)
```python
model_name = "Qwen/Qwen3-4B"
use_half_precision = True  # 必需！减少内存使用
```

**最低要求**:
- GPU: 16GB VRAM (推荐 24GB+)
- CPU + RAM: 32GB+ 系统内存（不推荐，会很慢）

#### 对于小模型 (gpt2-small)  
```python
model_name = "gpt2-small"
use_half_precision = False  # fp32 即可
```

**最低要求**:
- GPU: 2GB VRAM
- CPU + RAM: 4GB 系统内存

### 3. 内存检查

#### Windows 快速检查
```powershell
# 查看系统可用内存
Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory

# 查看 GPU 内存（如果有NVIDIA GPU）
nvidia-smi
```

#### Python 内存监控
```python
import psutil
import torch

# 系统内存
memory = psutil.virtual_memory()
print(f"Available: {memory.available / 1024**3:.1f} GB")

# GPU 内存（如果可用）
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### 4. 崩溃时的应对

如果服务器仍然崩溃：

1. **降级方法 1**: 使用更小的模型
   ```python
   model_name = "gpt2-small"  # 或 "gpt2-medium"
   ```

2. **降级方法 2**: 启用梯度检查点（更慢但省内存）
   ```python
   model = transformer_lens.HookedTransformer.from_pretrained(
       model_name,
       trust_remote_code=True,
       torch_dtype=torch.float16,
       use_cache=False  # 禁用KV cache
   )
   ```

3. **降级方法 3**: 使用CPU + 限制批次大小
   - 在分析时一次只处理一个token

### 5. 错误日志位置

如果崩溃，检查以下日志：
- Windows事件查看器: Applications & Services Logs > Python
- PowerShell 输出
- `server_log.txt`（如果使用了重定向）

### 6. 当前状态

✓ 服务器运行中 (PID 21844)
✓ 使用 gpt2-small（安全配置）
✓ 监听端口 8888

如需切换到 Qwen3-4B，确保：
1. 有足够的内存（16GB+ GPU 或 32GB+ RAM）
2. `use_half_precision = True`
3. 重启服务器以应用更改
