"""
下载GLM-4-9B-Chat-HF模型 - 使用huggingface_hub
优化: 逐个文件下载, 避免并发超时
"""
import os
import sys
import time
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "zai-org/glm-4-9b-chat-hf"
LOCAL_DIR = r"D:\develop\model\hub\models--zai-org--glm-4-9b-chat-hf\download"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting download: {REPO_ID}")
    print(f"Target: {LOCAL_DIR}")
    
    # 列出所有文件
    files = list_repo_files(REPO_ID, repo_type="model")
    print(f"\nTotal files: {len(files)}")
    
    # 按优先级排序: 小文件先下, 大文件后下
    # safetensors文件是模型权重(大), 其他文件是配置(小)
    small_files = []
    large_files = []
    for f in files:
        if f.endswith('.safetensors') or f.endswith('.bin'):
            large_files.append(f)
        else:
            small_files.append(f)
    
    print(f"Small files: {len(small_files)}")
    print(f"Large files: {len(large_files)}")
    for lf in large_files:
        print(f"  {lf}")
    
    # 先下载小文件
    print(f"\n[{time.strftime('%H:%M:%S')}] Downloading small files...")
    for f in small_files:
        target = os.path.join(LOCAL_DIR, f)
        if os.path.exists(target):
            print(f"  SKIP (exists): {f}")
            continue
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
            )
            print(f"  OK: {f}")
        except Exception as e:
            print(f"  FAIL: {f} - {e}")
    
    # 再逐个下载大文件
    print(f"\n[{time.strftime('%H:%M:%S')}] Downloading large files...")
    for i, f in enumerate(large_files):
        target = os.path.join(LOCAL_DIR, f)
        if os.path.exists(target):
            size = os.path.getsize(target)
            print(f"  SKIP (exists, {size/1e9:.2f}GB): {f}")
            continue
        
        print(f"\n  [{i+1}/{len(large_files)}] Downloading: {f}")
        t0 = time.time()
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
            )
            t1 = time.time()
            size = os.path.getsize(target)
            speed = size / (t1 - t0) / 1e6
            print(f"  OK: {f} ({size/1e9:.2f}GB, {speed:.1f}MB/s, {t1-t0:.0f}s)")
        except Exception as e:
            print(f"  FAIL: {f} - {e}")
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Download completed!")

if __name__ == "__main__":
    main()
