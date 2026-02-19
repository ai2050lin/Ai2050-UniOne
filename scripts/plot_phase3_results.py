"""
FiberNet Phase 3 结果可视化
读取 phase3_wiki_results.json 并绘制 Loss 和 Intrinsic Dimension (ID) 曲线。
"""
import os, json, matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata")
JSON_PATH = os.path.join(DATA_DIR, "phase3_wiki_results.json")
SAVE_PATH = os.path.join(DATA_DIR, "phase3_plot.png")

def main():
    if not os.path.exists(JSON_PATH):
        print(f"数据文件不存在: {JSON_PATH}")
        return

    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    
    epochs = [d['ep'] for d in data]
    losses = [d['loss'] for d in data]
    ids = [d['id'] for d in data]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, losses, color=color, marker='o', label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Intrinsic Dimension (ID)', color=color)
    ax2.plot(epochs, ids, color=color, marker='x', linestyle='--', label='ID')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('FiberNet WikiText Scaling: Loss vs Intrinsic Dimension')
    fig.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"图表已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    main()
