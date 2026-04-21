"""Run Phase CCIX with output logged to file."""
import sys
import os

# Redirect stdout/stderr to file
log_path = "tests/glm5_temp/ccix_run.log"
sys.stdout = open(log_path, "w", buffering=1)
sys.stderr = sys.stdout

print(f"Starting at {os.popen('echo %time%').read().strip()}")

# Now import and run
from importlib import import_module
sys.argv = ["phase_ccix_v2.py", "--model", "deepseek7b", "--n_pairs", "200"]

# Directly execute the main script
exec(open("tests/glm5/phase_ccix_v2.py").read())
