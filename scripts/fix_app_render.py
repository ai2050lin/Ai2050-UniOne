"""
修复 App.jsx: 导入 FiberNetPanel 并替换 FiberNet Lab 渲染区，
使 FiberNetPanel（六频道实验室）成为 FiberNet Lab 的实际显示组件。
"""

filepath = r"d:\develop\TransformerLens-main\frontend\src\App.jsx"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# 1. 检查是否已经导入了 FiberNetPanel
if "import FiberNetPanel" not in content:
    # 在 MotherEnginePanel import 之后添加
    old_import = "import { MotherEnginePanel } from './components/MotherEnginePanel';"
    new_import = old_import + "\nimport FiberNetPanel from './components/FiberNetPanel';"
    content = content.replace(old_import, new_import)
    print("Added FiberNetPanel import")
else:
    print("FiberNetPanel import already exists")

# 2. 替换 FiberNet Lab 渲染区
old_render = """            {/* FiberNet Lab Content */}
            {inputPanelTab === 'fibernet' && (
              <div className="animate-fade-in" style={{ height: '100%' }}>
                <FiberNetV2Demo t={t} />
              </div>
            )}"""

new_render = """            {/* FiberNet Lab Content - Phase XXXIV Unified Lab */}
            {inputPanelTab === 'fibernet' && (
              <div className="animate-fade-in" style={{ height: '100%', overflowY: 'auto' }}>
                <FiberNetPanel />
              </div>
            )}"""

if old_render in content:
    content = content.replace(old_render, new_render)
    print("Replaced FiberNet Lab render with FiberNetPanel")
else:
    print("WARNING: Could not find exact render block, trying alternative match...")
    # 尝试更宽松的匹配
    if "<FiberNetV2Demo t={t} />" in content and "inputPanelTab === 'fibernet'" in content:
        content = content.replace("<FiberNetV2Demo t={t} />", "<FiberNetPanel />")
        print("Replaced FiberNetV2Demo reference with FiberNetPanel (simple replace)")
    else:
        print("ERROR: Could not find FiberNet Lab render block in any form")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

# 验证
with open(filepath, "r", encoding="utf-8") as f:
    verify = f.read()

has_import = "import FiberNetPanel" in verify
has_render = "<FiberNetPanel" in verify
print(f"\nVerification:")
print(f"  FiberNetPanel imported: {has_import}")
print(f"  FiberNetPanel rendered: {has_render}")
print(f"Done!")
