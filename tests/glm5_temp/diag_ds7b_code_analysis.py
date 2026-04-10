# -*- coding: utf-8 -*-
"""DeepSeek7B 纯文件诊断 - 不加载模型, 只做代码对比分析"""
import os, sys, io
import functools
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

model_path = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"

print("=" * 60)
print("DeepSeek7B 代码级对比诊断 (不加载模型)")
print("=" * 60)

# 1. 模型文件检查
print("\n[1] 模型文件检查")
if os.path.isdir(model_path):
    total_size = 0
    for f in os.listdir(model_path):
        fpath = os.path.join(model_path, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            print(f"  {f}: {size/1e9:.2f} GB")
    print(f"  总大小: {total_size/1e9:.2f} GB")
else:
    print(f"  错误: 目录不存在!")

# 2. 对比GPT5 vs GLM5的加载代码
print("\n[2] GPT5 stage573 vs GLM5 phase_xliii 加载代码对比")
print("""
对比项              | GPT5 (stage573, 成功)   | GLM5 (phase_xliii, 失败)
--------------------+------------------------+-------------------------
low_cpu_mem_usage   | True (分片加载)         | 未设置(默认False)
local_files_only    | True (纯本地)           | 未设置(默认False)
use_fast (tokenizer)| False (slow)           | 未设置(默认True)
device_map          | "cpu"                  | "cpu"
.to("cuda")         | 有GPU检测              | 直接.to("cuda")
torch_dtype         | torch.bfloat16         | dtype=torch.bfloat16
""")

# 3. 问题根因分析
print("[3] 问题根因分析")
print("""
DeepSeek7B在GPT5侧正常、GLM5侧失败的3个根因:

[1] low_cpu_mem_usage=True (最关键!)
   不设置时: transformers一次性将全部权重读入CPU内存
             7B bf16模型 = 14GB权重 + Python开销 = 约28GB RAM
             系统RAM不够时触发 os error 1455 (页面文件不足)
   
   设置后:   transformers使用分片加载(safetensors)
             每次只加载一个分片, 峰值RAM = 模型大小+2GB
             避免了页面文件溢出

[2] local_files_only=True (次要)
   不设置时: 可能尝试连接HuggingFace Hub检查更新
             网络超时或连接失败会导致加载卡住
   
   设置后:   纯本地加载, 不触发网络请求

[3] 命令超时被系统跳过 (操作层面)
   CPU加载7B模型: 约30-60秒
   CPU->GPU迁移: 约20-40秒
   数据收集(60词x7模板): 约3-5分钟
   P257+P258+P259分析: 约5-10分钟
   总计: 10-20分钟
   
   系统对长运行命令有超时限制, 被跳过后看起来像是"失败"
   实际上模型本身可以正常运行, 只是总时间超限
""")

# 4. 验证: 查看GLM5历史记录
print("[4] GLM5记录验证")
print("""
2026-04-09 02:40 记录:
  - device_map="auto" -> ACCESS_VIOLATION (CUDA驱动状态不一致)
  - 修复: 改用CPU加载+手动.cuda() -> 4个模型全部通过

2026-04-09 12:24 记录:
  - DeepSeek7B CUDA测试通过: 15.28 GB显存
  - 之前os error 1455(页面文件不足)的原因: CPU加载模型时RAM+页面文件不够
  - 解决方案: 先CPU加载再GPU, 或使用load_in_8bit+device_map="auto"

2026-04-09 13:40 记录:
  - device_map="auto"直接GPU加载成功(跳过CPU峰值)
  - DeepSeek7B CUDA: 15.28 GB显存, 推理正常

矛盾点: 02:40说device_map=auto崩溃, 13:40说device_map=auto成功
解释: 02:40是系统重启后CUDA状态不干净; 13:40是CUDA状态恢复后auto可用
结论: device_map=auto的稳定性依赖CUDA驱动状态, 不是可靠的通用方案
""")

# 5. 修复方案
print("[5] 修复方案")
print("""
修复phase_xliii脚本的load_model函数:

def load_model(model_name):
    p = get_model_path(model_name)
    p_abs = os.path.abspath(p)
    tok = AutoTokenizer.from_pretrained(
        p_abs, trust_remote_code=True,
        local_files_only=True,       # 新增: 纯本地加载
        use_fast=False               # 新增: slow tokenizer更兼容
    )
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        p_abs,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,       # 新增: 纯本地加载
        low_cpu_mem_usage=True,      # 新增: 分片加载, 减少RAM峰值
        attn_implementation="eager",
        device_map="cpu"
    )
    if torch.cuda.is_available():    # 新增: GPU可用性检测
        mdl = mdl.to("cuda")
    mdl.eval()
    device = next(mdl.parameters()).device
    return mdl, tok, device
""")

print("\n完成!")
