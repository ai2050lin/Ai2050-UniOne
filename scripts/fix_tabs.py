"""最终修复：删除重复的 {[ 行"""

filepath = r"d:\develop\TransformerLens-main\frontend\src\components\FiberNetPanel.jsx"

with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 找到连续的两行 {[ 并删除第一行
i = 0
while i < len(lines) - 1:
    if lines[i].strip().startswith("{[") and lines[i+1].strip().startswith("{["):
        print(f"Removing duplicate '{{[' at line {i+1}")
        del lines[i]
        continue
    i += 1

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(lines)

# 验证
with open(filepath, "r", encoding="utf-8") as f:
    lines2 = f.readlines()
for j, line in enumerate(lines2, 1):
    if "evolution" in line or "observer" in line:
        if "id:" in line:
            print(f"L{j}: {line.rstrip()}")

print("Done.")
