"""Run CCLXXXV for deepseek7b only"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxv_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

# Import the run_model function from the main script
sys.path.insert(0, r"d:\Ai2050\TransformerLens-Project")
from tests.glm5.cclxxxv_multi_category_hierarchy import run_model
import gc, time

log("=== Starting DS7B only run ===")
run_model("deepseek7b")
gc.collect()
log("=== DS7B only run done ===")
