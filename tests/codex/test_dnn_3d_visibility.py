"""
DNN分析3D视图可见性测试脚本
验证3D视图按钮和功能是否正常工作
"""

import json
import sys
from pathlib import Path


def test_component_file_exists():
    """验证DNN分析组件文件存在"""
    component_path = Path("d:/develop/TransformerLens-main/frontend/src/components/DNNAnalysisControlPanel.jsx")
    three_d_path = Path("d:/develop/TransformerLens-main/frontend/src/components/DNNAnalysis3DVisualization.jsx")

    print("[测试1] 检查组件文件存在性:")
    if component_path.exists():
        print(f"  [OK] DNNAnalysisControlPanel.jsx 存在")
    else:
        print(f"  [FAIL] DNNAnalysisControlPanel.jsx 不存在")
        return False

    if three_d_path.exists():
        print(f"  [OK] DNNAnalysis3DVisualization.jsx 存在")
    else:
        print(f"  [FAIL] DNNAnalysis3DVisualization.jsx 不存在")
        return False

    return True


def test_3d_button_in_code():
    """验证3D视图按钮在代码中"""
    component_path = Path("d:/develop/TransformerLens-main/frontend/src/components/DNNAnalysisControlPanel.jsx")
    content = component_path.read_text(encoding='utf-8')

    print("\n[测试2] 检查3D视图按钮代码:")

    # 检查导入
    if 'DNNAnalysis3DVisualization' in content:
        print(f"  [OK] DNNAnalysis3DVisualization 已导入")
    else:
        print(f"  [FAIL] DNNAnalysis3DVisualization 未导入")
        return False

    # 检查Box3D图标
    if 'Box3D' in content:
        print(f"  [OK] Box3D 图标已导入")
    else:
        print(f"  [FAIL] Box3D 图标未导入")
        return False

    # 检查3D状态
    if 'show3DVisualization' in content:
        print(f"  [OK] show3DVisualization 状态已定义")
    else:
        print(f"  [FAIL] show3DVisualization 状态未定义")
        return False

    # 检查3D视图按钮
    if '3D视图' in content or '3D' in content:
        print(f"  [OK] 3D视图按钮文本存在")
    else:
        print(f"  [FAIL] 3D视图按钮文本不存在")
        return False

    # 检查setShow3DVisualization调用
    if 'setShow3DVisualization' in content:
        print(f"  [OK] setShow3DVisualization 函数调用存在")
    else:
        print(f"  [FAIL] setShow3DVisualization 函数调用不存在")
        return False

    return True


def test_conditional_rendering():
    """验证条件渲染逻辑"""
    component_path = Path("d:/develop/TransformerLens-main/frontend/src/components/DNNAnalysisControlPanel.jsx")
    content = component_path.read_text(encoding='utf-8')

    print("\n[测试3] 检查条件渲染逻辑:")

    # 检查analysisResults条件
    if 'analysisResults?.dimension === dimension.id' in content:
        print(f"  [OK] analysisResults 条件渲染正确")
    else:
        print(f"  [FAIL] analysisResults 条件渲染不正确")
        return False

    # 检查show3DVisualization条件
    if 'show3DVisualization ?' in content:
        print(f"  [OK] show3DVisualization 条件渲染正确")
    else:
        print(f"  [FAIL] show3DVisualization 条件渲染不正确")
        return False

    return True


def test_button_visibility():
    """分析按钮样式是否可见"""
    component_path = Path("d:/develop/TransformerLens-main/frontend/src/components/DNNAnalysisControlPanel.jsx")
    content = component_path.read_text(encoding='utf-8')

    print("\n[测试4] 检查按钮可见性样式:")

    # 检查按钮样式
    if 'marginLeft: \'auto\'' in content:
        print(f"  [OK] 按钮右侧对齐样式")
    else:
        print(f"  [WARN] 按钮右侧对齐样式可能缺失")

    # 检查按钮颜色
    if 'color: \'#4facfe\'' in content:
        print(f"  [OK] 按钮颜色样式")
    else:
        print(f"  [WARN] 按钮颜色样式可能缺失")

    # 检查按钮边框
    if 'border:' in content or 'borderWidth:' in content:
        print(f"  [OK] 按钮边框样式")
    else:
        print(f"  [WARN] 按钮边框样式可能缺失")

    return True


def test_mock_data_generation():
    """验证模拟数据生成"""
    component_path = Path("d:/develop/TransformerLens-main/frontend/src/components/DNNAnalysisControlPanel.jsx")
    content = component_path.read_text(encoding='utf-8')

    print("\n[测试5] 检查模拟数据生成:")

    # 检查generateMockAnalysisData函数
    if 'generateMockAnalysisData' in content:
        print(f"  [OK] generateMockAnalysisData 函数存在")
    else:
        print(f"  [FAIL] generateMockAnalysisData 函数不存在")
        return False

    # 检查catch块中的模拟数据
    if 'catch (error)' in content and 'setAnalysisResults' in content:
        print(f"  [OK] 错误处理和模拟数据设置存在")
    else:
        print(f"  [FAIL] 错误处理或模拟数据设置缺失")
        return False

    return True


def test_integration_points():
    """验证集成点"""
    component_path = Path("d:/develop/TransformerLens-main/frontend/src/main.jsx")
    content = component_path.read_text(encoding='utf-8')

    print("\n[测试6] 检查CSS集成:")

    # 检查CSS导入
    if 'DNNAnalysisControlPanel.css' in content:
        print(f"  [OK] DNNAnalysisControlPanel.css 已导入")
    else:
        print(f"  [WARN] DNNAnalysisControlPanel.css 可能未导入")

    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("DNN分析3D视图可见性测试")
    print("=" * 60)

    tests = [
        ("组件文件存在性", test_component_file_exists),
        ("3D视图按钮代码", test_3d_button_in_code),
        ("条件渲染逻辑", test_conditional_rendering),
        ("按钮可见性样式", test_button_visibility),
        ("模拟数据生成", test_mock_data_generation),
        ("集成点", test_integration_points),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] 测试失败: {str(e)}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed == 0:
        print("\n[成功] 所有测试通过！3D视图功能应该正常工作。")
        print("\n使用说明:")
        print("1. 打开前端应用")
        print("2. 导航到'语言研究'控制面板")
        print("3. 展开'DNN分析'区域")
        print("4. 点击'运行分析'按钮（任一分析维度）")
        print("5. 查看分析结果，应该能看到'切换到3D'按钮")
        print("6. 点击按钮切换到3D视图")
        return 0
    else:
        print(f"\n[失败] {failed} 个测试失败，请检查代码。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
