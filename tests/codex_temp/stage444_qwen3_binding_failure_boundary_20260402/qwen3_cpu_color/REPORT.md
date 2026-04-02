# stage444_qwen3_binding_failure_boundary

- model_key: qwen3
- model_name: Qwen/Qwen3-4B
- used_cuda: False
- batch_size: 1
- max_heads: 8
- max_neurons: 8

## color
- baseline_search_ok: True
- baseline_heldout_ok: True
- head_probe: {'tested_count': 8, 'success_count': 8, 'success_ids': ['H:33:0', 'H:33:1', 'H:33:2', 'H:33:3', 'H:33:4', 'H:33:5', 'H:33:6', 'H:33:7'], 'first_error': None}
- neuron_probe: {'tested_count': 8, 'success_count': 8, 'success_ids': ['N:27:995', 'N:34:2409', 'N:32:9371', 'N:35:4525', 'N:24:191', 'N:33:4032', 'N:33:749', 'N:31:9411'], 'first_error': None}

