"""运行DS7B测试并写入日志文件"""
import subprocess
import sys
import time

log_file = r"d:\develop\TransformerLens-main\tests\glm5_temp\test_ds7b_run.log"

with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"Starting DS7B test at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.flush()

proc = subprocess.Popen(
    [sys.executable, r"d:\develop\TransformerLens-main\tests\glm5_temp\test_deepseek7b.py"],
    stdout=open(log_file, "a", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    cwd=r"d:\develop\TransformerLens-main",
)

print(f"Process started: PID={proc.pid}")
print(f"Log file: {log_file}")
print("Model loading may take 3-5 minutes...")
