from pathlib import Path
import json

root = Path(r'd:\develop\TransformerLens-main')
text = (root / 'frontend/src/blueprint/AppleNeuron3DTab.jsx').read_text(encoding='utf-8')
def has(*patterns):
    return any(p in text for p in patterns)
summary = {
    'basic_summary_present': has('当前基础摘要', '\\u5f53\\u524d\\u57fa\\u7840\\u6458\\u8981'),
    'basic_concept_browser_present': has('基础名词 / 概念浏览', '\\u57fa\\u7840\\u540d\\u8bcd / \\u6982\\u5ff5\\u6d4f\\u89c8'),
    'algorithm_explanation_present': has('算法显示说明', '\\u7b97\\u6cd5\\u663e\\u793a\\u8bf4\\u660e'),
    'algorithm_entry_present': has('算法入口', '\\u7b97\\u6cd5\\u5165\\u53e3'),
    'concept_core_button_present': has('显示苹果概念核', '\\u663e\\u793a\\u82f9\\u679c\\u6982\\u5ff5\\u6838'),
    'encoding_layer_button_present': has('显示静态编码层', '\\u663e\\u793a\\u9759\\u6001\\u7f16\\u7801\\u5c42'),
    'runtime_chain_button_present': has('显示运行链路', '\\u663e\\u793a\\u8fd0\\u884c\\u94fe\\u8def'),
}
out_dir = root / 'tests/codex_temp/stage424_dnn_panel_system_restructure_20260326'
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
