from pathlib import Path


def main() -> None:
    path = Path("frontend/src/blueprint/AppleNeuron3DTab.jsx")
    text = path.read_text(encoding="utf-8")

    start_marker = "{assetPanelTab === 'manual' ? ("
    end_marker = ") : ("
    start = text.find(start_marker)
    end = text.find(end_marker, start)

    if start == -1 or end == -1:
        raise RuntimeError("未找到手动输入面板区间")

    block = text[start:end]

    forbidden_markers = [
        "当前理论对象",
        "当前选中的 3D 节点",
        "苹果相关神经元信息",
        "当前已接入数据",
        "当前实体统计",
        "当前编码链数据索引",
        "当前编码链统计",
    ]

    remaining = [item for item in forbidden_markers if item in block]

    print(
        {
            "manual_block_found": True,
            "removed_marker_count": len(forbidden_markers) - len(remaining),
            "remaining_markers": remaining,
            "status": "ok" if not remaining else "has_remaining",
        }
    )


if __name__ == "__main__":
    main()
