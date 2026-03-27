from pathlib import Path
import json

root = Path(__file__).resolve().parents[2]
panel = (root / "frontend/src/components/LanguageResearchControlPanel.jsx").read_text(encoding="utf-8")
apple = (root / "frontend/src/blueprint/AppleNeuron3DTab.jsx").read_text(encoding="utf-8")

summary = {
    "encoding_scene_removed": ("<span>??????</span>" not in panel and "<span>?????????</span>" not in panel),
    "structure_overlay_removed": ("<span>????</span>" not in panel and "<span>??????</span>" not in panel),
    "linkage_summary_removed": ("<span>??????</span>" not in panel and "<span>?????????</span>" not in panel),
    "five_layer_block_kept": ("RESEARCH_LAYERS.map" in panel),
    "layer_animation_controls_restored": ("handleBasicRuntimeStart" in apple and "handleBasicRuntimeStop" in apple and "handleBasicRuntimeReplay" in apple),
    "algorithm_buttons_restored": ("setLanguageFocus?.((prev) => ({ ...prev, researchLayer: 'static_encoding' }))" in apple and "mechanism_chain: true" in apple),
}

out_dir = root / "tests/codex_temp/stage421_panel_module_cleanup_and_animation_restore_20260326"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
