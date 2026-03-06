import os

paths_to_check = [
    r'C:\Users\chris\.cache\huggingface\hub',
    r'C:\Users\chris\.ollama\models',
    r'd:\develop\TransformerLens-main'
]

extensions = ['.safetensors', '.bin', '.pt', '.pth', '.gguf', '.onnx', '.h5', '.ckpt']

results = []

for base_path in paths_to_check:
    if not os.path.exists(base_path):
        continue
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(full_path)
                    if size > 10 * 1024 * 1024:  # > 10MB
                        results.append((size, full_path))
                except Exception:
                    pass

results.sort(reverse=True)

print("Found models (Top 20 largest):")
for size, path in results[:20]:
    print(f"{size / (1024**3):.2f} GB - {path}")

if not results:
    print("No large model files found in common directories.")
