"""
下载GLM-4-9B-Chat-HF模型
使用huggingface_hub的snapshot_download
预计大小: ~18GB
"""
import os
import sys
import time
from huggingface_hub import snapshot_download

model_id = 'zai-org/glm-4-9b-chat-hf'
cache_dir = r'D:\develop\model\hub'

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting download: {model_id}")
print(f"Cache dir: {cache_dir}")
print("Estimated size: ~18GB")
print("This may take a while depending on network speed...")
sys.stdout.flush()

try:
    path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        resume_download=True,
    )
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Download completed!")
    print(f"Model path: {path}")
except Exception as e:
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Download failed: {e}")
    sys.exit(1)
