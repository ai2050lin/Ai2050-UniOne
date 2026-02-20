"""
WikiText-2 数据预处理脚本 (v2 - 修复版)
功能：从 GitHub 直接下载 Raw 文本，避免不稳定 Zip 链接。
"""
import os, requests
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata")
WIKI_RAW_URL = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"

def setup_wikitext():
    os.makedirs(DATA_DIR, exist_ok=True)
    wiki_txt = os.path.join(DATA_DIR, "wiki_train.txt")
    if not os.path.exists(wiki_txt):
        print(f"[*] 下载 WikiText-2 (Raw) from {WIKI_RAW_URL}...")
        r = requests.get(WIKI_RAW_URL)
        if r.status_code == 200:
            with open(wiki_txt, "w", encoding="utf-8") as f:
                f.write(r.text)
        else:
            raise Exception(f"下载失败: {r.status_code}")
    
    with open(wiki_txt, "r", encoding="utf-8") as f:
        return f.read()

class SimpleBPETokenizer:
    def __init__(self):
        self.encoder = {chr(i): i for i in range(256)}
        
    def encode(self, text):
        # 字节级编码方案
        return list(text.encode('utf-8', errors='ignore'))

def main():
    try:
        text = setup_wikitext()
        print(f"数据量: {len(text)/1e6:.2f} MB")
        
        tokenizer = SimpleBPETokenizer()
        tokens = tokenizer.encode(text) 
        print(f"Token 数量: {len(tokens)}")
        
        save_path = os.path.join(DATA_DIR, "wiki_v4_3.npy")
        np.save(save_path, np.array(tokens, dtype=np.uint16))
        print(f"[+] 预处理完成，保存至: {save_path}")
    except Exception as e:
        print(f"[!] 错误: {e}")

if __name__ == "__main__":
    main()
