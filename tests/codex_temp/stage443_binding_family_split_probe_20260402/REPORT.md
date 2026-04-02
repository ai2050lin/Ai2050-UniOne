# stage443_binding_family_split_probe

- model_key: deepseek7b
- model_name: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- used_cuda: True
- batch_size: 1

## color
- ok: False
- error: CUDA error: unknown error
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


## taste
- ok: True
- candidate_probe: {'attention_head_ok': True, 'mlp_neuron_ok': True, 'probe_errors': {}}
- subset: ['H:25:8']
- binding_drop: 0.012207
- heldout_binding_drop: 0.016276
- utility: 0.009277
- mixed_support: False

## size
- ok: True
- candidate_probe: {'attention_head_ok': True, 'mlp_neuron_ok': True, 'probe_errors': {}}
- subset: ['H:25:12', 'N:26:17120', 'H:23:11', 'H:23:3', 'H:27:19']
- binding_drop: 0.064331
- heldout_binding_drop: 0.052083
- utility: 0.027222
- mixed_support: True

