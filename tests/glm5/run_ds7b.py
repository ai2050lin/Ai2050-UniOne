"""用subprocess运行测试, stderr和stdout全到文件"""
import subprocess, sys, time

LOG = r"d:\develop\TransformerLens-main\tests\glm5_temp\ds7b_test_output.txt"
SCRIPT = r"d:\develop\TransformerLens-main\tests\glm5\test_ds7b_direct.py"

t0 = time.time()
with open(LOG, "w", encoding="utf-8") as f:
    proc = subprocess.Popen(
        [sys.executable, SCRIPT],
        stdout=f, stderr=subprocess.STDOUT,
        cwd=r"d:\develop\TransformerLens-main"
    )
    print(f"PID={proc.pid}, waiting...")
    try:
        rc = proc.wait(timeout=300)
        print(f"Exit code: {rc} (0xC0000005={hex(rc)} if negative)")
    except subprocess.TimeoutExpired:
        proc.kill()
        print("TIMEOUT killed")

elapsed = time.time() - t0
print(f"Time: {elapsed:.1f}s")

# 读结果
with open(LOG, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()
    print(content if content else "(empty log)")
