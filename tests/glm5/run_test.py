"""用subprocess运行模型测试, 避免PowerShell stderr问题"""
import subprocess, sys, time, os

script = r"d:\develop\TransformerLens-main\tests\glm5\test_model_v3.py"
model = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
outfile = r"d:\develop\TransformerLens-main\tests\glm5_temp\model_test_subprocess.log"

print(f"Starting test for {model}...", flush=True)

with open(outfile, "w", encoding="utf-8") as f:
    proc = subprocess.Popen(
        [sys.executable, script, model],
        stdout=f, stderr=subprocess.STDOUT,
        cwd=r"d:\develop\TransformerLens-main"
    )
    print(f"PID={proc.pid}, log={outfile}", flush=True)
    
    # 等待完成, 最长10分钟
    try:
        proc.wait(timeout=600)
        print(f"Exit code: {proc.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("TIMEOUT - killed", flush=True)

# 读取结果
with open(outfile, "r", encoding="utf-8", errors="replace") as f:
    print(f.read())
