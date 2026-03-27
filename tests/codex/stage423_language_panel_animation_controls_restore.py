from pathlib import Path
import json

root = Path(r'd:\develop\TransformerLens-main')
panel = (root / 'frontend/src/components/LanguageResearchControlPanel.jsx').read_text(encoding='utf-8')
tab = (root / 'frontend/src/blueprint/AppleNeuron3DTab.jsx').read_text(encoding='utf-8')
summary = {
    'language_panel_animation_controls_restored': ('\\u5f00\\u59cb\\u52a8\\u753b' in panel and '\\u7ed3\\u675f\\u52a8\\u753b' in panel and '\\u91cd\\u65b0\\u64ad\\u653e' in panel),
    'language_panel_runtime_status_present': '\\u5f53\\u524d\\u72b6\\u6001' in panel,
    'apple_tab_runtime_card_removed': ('Layer \\u9010\\u5c42\\u52a8\\u753b' not in tab),
}
out_dir = root / 'tests/codex_temp/stage423_language_panel_animation_controls_restore_20260326'
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
