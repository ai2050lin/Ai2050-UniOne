"""CCLXXXX DS7B only - restart from failed model"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxx_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

# Import everything from the main script
sys.path.insert(0, r"d:\Ai2050\TransformerLens-Project")
from tests.glm5.cclxxxx_centroid_trajectory import run_model
import gc

log("=== Restarting DS7B ===")
run_model("deepseek7b")
gc.collect()
log("=== DS7B done ===")
