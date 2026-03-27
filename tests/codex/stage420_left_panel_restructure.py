from pathlib import Path


def main() -> None:
    apple = Path("frontend/src/blueprint/AppleNeuron3DTab.jsx").read_text(encoding="utf-8")
    control = Path("frontend/src/components/LanguageResearchControlPanel.jsx").read_text(encoding="utf-8")

    asset_idx = apple.find("基础信息")
    dnn_idx = apple.find("DNN 分层显示")
    legacy_idx = control.find("高级分析工具")

    result = {
        "apple_basic_info_found": asset_idx != -1,
        "apple_dnn_found": dnn_idx != -1,
        "basic_before_dnn": asset_idx != -1 and dnn_idx != -1 and asset_idx < dnn_idx,
        "legacy_panel_removed": legacy_idx == -1,
    }
    print(result)


if __name__ == "__main__":
    main()
