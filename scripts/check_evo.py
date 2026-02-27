filepath = r"d:\develop\TransformerLens-main\frontend\src\components\FiberNetPanel.jsx"
with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()
for i, line in enumerate(lines, 1):
    if "evolution" in line.lower() or "moon" in line.lower():
        print(f"L{i}: {line.rstrip()}")
