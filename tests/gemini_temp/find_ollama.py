import os

path = r'C:\Users\chris\.ollama\models\blobs'
if not os.path.exists(path):
    print("Ollama models directory not found.")
else:
    blobs = []
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            if sz > 10 * 1024 * 1024:
                blobs.append((sz, fp))
    
    blobs.sort(reverse=True)
    for sz, p in blobs[:10]:
        print(f"{sz / (1024**3):.2f} GB - Ollama model: {f}")
