# stage444_qwen3_binding_failure_boundary

- model_key: qwen3
- model_name: Qwen/Qwen3-4B
- used_cuda: True
- batch_size: 1
- max_heads: 4
- max_neurons: 4

## color
- baseline_search_ok: True
- baseline_heldout_ok: True
- head_probe: {'tested_count': 4, 'success_count': 4, 'success_ids': ['H:33:0', 'H:33:1', 'H:33:2', 'H:33:3'], 'first_error': None}
- neuron_probe: {'tested_count': 4, 'success_count': 4, 'success_ids': ['N:27:995', 'N:34:2409', 'N:32:9371', 'N:35:4525'], 'first_error': None}

## taste
- baseline_search_ok: True
- baseline_heldout_ok: True
- head_probe: {'tested_count': 4, 'success_count': 4, 'success_ids': ['H:33:0', 'H:33:1', 'H:33:2', 'H:33:3'], 'first_error': None}
- neuron_probe: {'tested_count': 4, 'success_count': 4, 'success_ids': ['N:27:995', 'N:34:2409', 'N:32:9371', 'N:35:4525'], 'first_error': None}

## size
- baseline_search_ok: True
- baseline_heldout_ok: True
- head_probe: {'tested_count': 4, 'success_count': 4, 'success_ids': ['H:33:0', 'H:33:1', 'H:33:2', 'H:33:3'], 'first_error': None}
- neuron_probe: {'tested_count': 4, 'success_count': 4, 'success_ids': ['N:27:995', 'N:34:2409', 'N:32:9371', 'N:35:4525'], 'first_error': None}

