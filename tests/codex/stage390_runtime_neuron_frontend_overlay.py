from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "codex_temp" / f"stage390_runtime_neuron_frontend_overlay_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "Stage390",
        "title": "前端实时神经元叠加层",
        "component": "frontend/src/ParameterEncoding3D.jsx",
        "overlay_component": "LiveRuntimeFlow",
        "data_source": "/api/runtime/neuron_flow",
        "display_elements": [
            "runtime neuron nodes",
            "runtime links",
            "target token caption",
        ],
        "runtime_node_info_fields": [
            "label",
            "activation_abs",
            "activation_value",
            "layer_index",
            "token_index",
            "dim_index",
            "hook_name",
        ],
        "integration_mode": [
            "shared_carrier_3d",
            "bias_deflection_3d",
            "layerwise_amplification_3d",
            "multispace_operator_3d",
            "cross_model_compare_3d",
        ],
        "status": "implemented",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
