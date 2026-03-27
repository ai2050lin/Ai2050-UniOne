from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "codex_temp" / f"stage389_runtime_neuron_flow_endpoint_{datetime.now().strftime('%Y%m%d')}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "Stage389",
        "title": "实时前向抓流接口",
        "endpoint": "/api/runtime/neuron_flow",
        "method": "GET",
        "query_fields": ["mode", "prompt", "top_k", "token_index"],
        "response_fields": [
            "mode",
            "model_name",
            "prompt",
            "top_k",
            "token_index",
            "target_token",
            "token_count",
            "layer_count",
            "nodes",
            "links",
        ],
        "node_fields": [
            "id",
            "label",
            "layer_index",
            "token_index",
            "token",
            "dim_index",
            "activation_value",
            "activation_abs",
            "hook_name",
            "topk_rank",
            "position",
        ],
        "link_fields": [
            "id",
            "from",
            "to",
            "from_layer",
            "to_layer",
            "rank",
            "strength",
        ],
        "model_backend": "server/server.py 中全局 HookedTransformer",
        "runtime_source": "run_with_cache 抓取 blocks.{layer}.hook_resid_post",
        "status": "implemented",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
