# stage444_qwen3_binding_failure_boundary

- model_key: qwen3
- model_name: Qwen/Qwen3-4B
- used_cuda: True
- batch_size: 1
- max_heads: 16
- max_neurons: 16

## color
- baseline_search_ok: True
- baseline_heldout_ok: True
- head_probe: {'tested_count': 16, 'success_count': 0, 'success_ids': [], 'first_error': {'candidate': 'H:33:0', 'error': 'CUDA error: unknown error\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n'}}
- neuron_probe: {'tested_count': 16, 'success_count': 0, 'success_ids': [], 'first_error': {'candidate': 'N:27:995', 'error': 'CUDA error: unknown error\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n'}}

## taste
- baseline_search_ok: False
- baseline_search_error: CUDA error: unknown error
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

- baseline_heldout_ok: False
- head_probe: None
- neuron_probe: None

## size
- baseline_search_ok: False
- baseline_search_error: CUDA error: unknown error
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

- baseline_heldout_ok: False
- head_probe: None
- neuron_probe: None

