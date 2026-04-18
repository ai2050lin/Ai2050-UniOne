"""
通过ModelScope国内源下载 DeepSeek-R1-Distill-Qwen-7B
预计大小: ~15GB, 预计时间: ~2小时 (@2MB/s)
"""
from modelscope import snapshot_download
import time

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
cache_dir = "D:/develop/model/hub/modelscope_cache"

print(f"开始下载 {model_id}")
print(f"缓存目录: {cache_dir}")
print(f"预计大小: ~15GB, 预计时间: ~2小时")
t0 = time.time()

model_dir = snapshot_download(
    model_id,
    cache_dir=cache_dir,
)

elapsed = time.time() - t0
print(f"\n下载完成! 耗时: {elapsed/60:.1f}分钟")
print(f"模型路径: {model_dir}")
