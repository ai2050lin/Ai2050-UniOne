"""
直接下载GLM-4-9B-Chat-HF模型文件
使用huggingface_hub + hf_transfer加速
"""
import os
import sys
import time

# 启用hf_transfer加速
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

REPO_ID = "zai-org/glm-4-9b-chat-hf"
LOCAL_DIR = r"D:\develop\model\hub\models--zai-org--glm-4-9b-chat-hf\download"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    print(f"[{time.strftime('%H:%M:%S')}] Downloading with hf_transfer acceleration...")
    print(f"Repo: {REPO_ID}")
    print(f"Local: {LOCAL_DIR}")
    sys.stdout.flush()
    
    t0 = time.time()
    try:
        path = snapshot_download(
            repo_id=REPO_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
        )
        t1 = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] Download completed in {t1-t0:.0f}s!")
        print(f"Path: {path}")
    except Exception as e:
        print(f"\n[{time.strftime('%H:%M:%S')}] Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
