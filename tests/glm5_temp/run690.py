import sys
sys.path.insert(0, r"d:\develop\TransformerLens-main")
import torch
import sklearn
print("torch OK:", torch.__version__)
print("sklearn OK")
try:
    exec(open(r"d:\develop\TransformerLens-main\tests\glm5\stage690_cross_model_basis_alignment.py").read())
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
