"""Quick script: Measure W_down effective rank for a model"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, get_layer_weights
import numpy as np, torch, gc
from scipy.sparse.linalg import svds
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
args = parser.parse_args()
mn = args.model

model, tokenizer, device = load_model(mn)
mi = get_model_info(model, mn)
layers = get_layers(model)
dm = mi.d_model
ni = mi.intermediate_size
nl = mi.n_layers
print(f'\n{mn}: d_model={dm}, n_inter={ni}')

for li in [0, nl//2, nl-1]:
    lw = get_layer_weights(layers[li], dm, mi.mlp_type)
    W = lw.W_down  # [d_model, n_inter]
    k = min(dm - 2, 500)
    U, s, Vt = svds(W.astype(np.float32), k=k)
    s = s[np.argsort(-s)]
    t = np.sum(s**2)
    c = np.cumsum(s**2) / t
    r90 = int(np.searchsorted(c, 0.90) + 1)
    r95 = int(np.searchsorted(c, 0.95) + 1)
    print(f'  L{li}: r90={r90}, r95={r95}, ratio90={r90/ni:.4f} ({r90}/{ni})')

del model
torch.cuda.empty_cache()
gc.collect()
