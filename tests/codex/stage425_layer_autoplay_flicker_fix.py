from pathlib import Path
import json

root = Path(r'd:\develop\TransformerLens-main')
text = (root / 'frontend/src/blueprint/AppleNeuron3DTab.jsx').read_text(encoding='utf-8')
summary = {
    'basic_runtime_default_stopped': 'const [basicRuntimePlaying, setBasicRuntimePlaying] = useState(false);' in text,
    'basic_runtime_no_autoplay_on_layer_change': 'setBasicRuntimePlaying(false);' in text and 'setBasicRuntimeStep(1);' in text,
}
out_dir = root / 'tests/codex_temp/stage425_layer_autoplay_flicker_fix_20260326'
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
