"""运行模型测试并等待完成"""
import subprocess, sys, time

SCRIPT = r"d:\develop\TransformerLens-main\tests\glm5\test_model_v5.py"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
RESULT = r"d:\develop\TransformerLens-main\tests\glm5_temp\model_test_result.txt"

t0 = time.time()
print(f"Testing {MODEL}...", flush=True)

with open(RESULT, "w", encoding="utf-8") as f:
    f.write(f"Starting {MODEL} at {time.strftime('%H:%M:%S')}\n")

proc = subprocess.Popen(
    [sys.executable, SCRIPT, MODEL],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    cwd=r"d:\develop\TransformerLens-main"
)
print(f"PID={proc.pid}", flush=True)

# 循环检查直到进程结束
while proc.poll() is None:
    elapsed = time.time() - t0
    # 每10秒检查一次
    proc.wait(timeout=10)
    break

# 等待进程结束
try:
    rc = proc.wait(timeout=600)
except subprocess.TimeoutExpired:
    proc.kill()
    rc = -1

elapsed = time.time() - t0
print(f"Exit code: {rc}, time: {elapsed:.1f}s", flush=True)

# 读取结果
with open(RESULT, "r", encoding="utf-8", errors="replace") as f:
    print(f.read())
