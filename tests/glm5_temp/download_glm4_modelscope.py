"""
从ModelScope下载GLM-4-9B-Chat-HF模型
国内源速度更快 (~2MB/s vs 0.5MB/s)
"""
import os
import sys
import time

# 设置ModelScope缓存目录
os.environ["MODELSCOPE_CACHE"] = r"D:\develop\model\hub"

from modelscope import snapshot_download

MODEL_ID = "ZhipuAI/glm-4-9b-chat-hf"
LOCAL_DIR = r"D:\develop\model\hub\models--zai-org--glm-4-9b-chat-hf\download"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    print(f"[{time.strftime('%H:%M:%S')}] Downloading from ModelScope...")
    print(f"Model: {MODEL_ID}")
    sys.stdout.flush()
    
    t0 = time.time()
    try:
        # 使用ModelScope下载
        model_dir = snapshot_download(
            MODEL_ID,
            cache_dir=r"D:\develop\model\hub\modelscope_cache",
        )
        t1 = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] Download completed in {(t1-t0)/60:.1f} minutes!")
        print(f"Model dir: {model_dir}")
        
        # 列出文件
        for f in os.listdir(model_dir):
            fp = os.path.join(model_dir, f)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                if size > 1e6:
                    print("  %.2f GB  %s" % (size/1e9, f))
        
    except Exception as e:
        print(f"\n[{time.strftime('%H:%M:%S')}] Download failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
