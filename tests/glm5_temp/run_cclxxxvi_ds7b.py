"""Run CCLXXXVI for deepseek7b only"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Fix the resume check to use correct name
LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxvi_log.txt"

# Check if already completed
with open(LOG, 'r', encoding='utf-8') as f:
    if "deepseek7b COMPLETE" in f.read():
        print("deepseek7b already completed, skipping")
        sys.exit(0)

# Patch the main script to run only deepseek7b
sys.path.insert(0, r"d:\Ai2050\TransformerLens-Project")

# Import the run_model function from the main script
import importlib.util
spec = importlib.util.spec_from_file_location("cclxxxvi", 
    r"d:\Ai2050\TransformerLens-Project\tests\glm5\cclxxxvi_boundary_geometry.py")
mod = importlib.util.module_from_spec(spec)

# Override the models list
import tests.glm5.cclxxxvi_boundary_geometry as cclxxxvi
cclxxxvi.run_model('deepseek7b')
print("deepseek7b done")
