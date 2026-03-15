from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research" / "gpt5" / "docs" / "AGI_GPT5_ICSPB.md"


def test_agi_gpt5_icspb_doc_refresh_block():
    text = DOC.read_text(encoding="utf-8")

    assert "最后更新：2026-03-15" in text
    assert "### 3.4 当前 DNN 数学提取的真实进度" in text
    assert "### 5.5 当前系统级精确编码候选定理" in text
    assert "system_parametric_score = 0.7282" in text
    assert "exact_system_closure_score = 0.3424" in text
    assert "family-to-specific exact closure + successor exact closure" in text
    assert "DNN 侧系统级参数原理理解度" in text


if __name__ == "__main__":
    test_agi_gpt5_icspb_doc_refresh_block()
    print("ok")
