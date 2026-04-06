#!/usr/bin/env python3
"""
P45: Training Dynamics of Direction Convergence (Stage691)

Core questions:
1. Does L0 alignment start at 1.0 from random initialization?
2. When does the "semantic basis" emerge during training?
3. How does L0->Final rotation evolve during training?
4. Is the high L0 alignment due to embedding convergence or architecture?

Method: Train tiny GPT-2 (d=256, 6 layers) and track at steps 1/5/10/25/50/100/200/310
Extract basis at L0 and final layer, measure alignment and rotation.

Usage: python tests/glm5/stage691_training_dynamics_basis.py
"""
import time, math, gc, json, pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.decomposition import PCA

TEXTS = [
    "The cat sat on the mat.", "The dog chased the ball.", "Birds fly south.",
    "Paris is the capital.", "Tokyo is large.", "The Amazon river.",
    "She folded the paper.", "The orchestra played well.",
    "His writing was elegant.", "The painting was detailed.",
    "If it rains then wet.", "She studied hard today.",
    "The boy fell down.", "Although tired she worked.",
    "The quick fox jumps.", "She has been working.",
    "They went to market.", "Report was on time.",
    "Yesterday it rained.", "She will finish soon.",
    "Project was completed.", "He arrived before.",
    "Two plus two is four.", "Derivative of x squared.",
    "DNA has instructions.", "Gravity pulls objects.",
    "Neural nets learn.", "Equation solved step by step.",
    "Experiment was consistent.", "Data supported hypothesis.",
    "A red apple fruit.", "The bank by river.",
    "Spring flowers bloom.", "The match was close.",
]


class TinyTextDataset(Dataset):
    def __init__(self, texts, char2id, max_len=32):
        self.data = []
        for text in texts:
            ids = [char2id.get(c, 2) for c in text]  # 2=<unk>
            ids = ids[:max_len]
            ids = ids + [1] * (max_len - len(ids))  # 1=<eos>
            self.data.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TinyGPT2(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6, max_len=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True, activation='gelu')
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits, h


def extract_basis(hidden_states_list):
    """Extract basis direction from list of (B, T, D) tensors, using last token"""
    directions = []
    for hs in hidden_states_list:
        last = hs[0, -1, :].float().detach().cpu()
        norm = torch.norm(last).item()
        if norm > 1e-10:
            directions.append(last / norm)

    if len(directions) < 2:
        return None

    mat = torch.stack(directions)
    mean_dir = mat.mean(dim=0)
    mean_norm = torch.norm(mean_dir).item()
    if mean_norm < 1e-10:
        return None
    mean_dir = mean_dir / mean_norm

    alignments = [F.cosine_similarity(d.unsqueeze(0), mean_dir.unsqueeze(0)).item() for d in directions]

    pca = PCA(n_components=min(5, len(directions) - 1))
    pca.fit(mat.numpy())

    return {
        "mean_alignment": float(np.mean(alignments)),
        "std_alignment": float(np.std(alignments)) if len(alignments) > 1 else 0,
        "pca": pca.explained_variance_ratio_.tolist(),
        "mean_dir": mean_dir.numpy(),
    }


def measure_at_step(model, texts, char2id, device):
    """Measure basis at L0 and final layer"""
    model.eval()
    l0_hiddens = []
    final_hiddens = []

    with torch.no_grad():
        for text in texts:
            ids = [char2id.get(c, 2) for c in text]
            x = torch.tensor([ids], dtype=torch.long, device=device)

            # Manual forward to get intermediate states
            B, T = x.shape
            pos = torch.arange(T, device=device).unsqueeze(0)
            h = model.tok_emb(x) + model.pos_emb(pos)

            # L0 hidden state (after embedding)
            l0_hiddens.append(h.clone())

            mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
            for layer in model.layers:
                h = layer(h, src_mask=mask, is_causal=True)
            h = model.ln_f(h)

            # Final hidden state
            final_hiddens.append(h.clone())

    l0_basis = extract_basis(l0_hiddens)
    final_basis = extract_basis(final_hiddens)

    # L0 vs Final cross-cos
    cross_cos = None
    if l0_basis is not None and final_basis is not None:
        d1 = torch.tensor(l0_basis["mean_dir"])
        d2 = torch.tensor(final_basis["mean_dir"])
        cross_cos = F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item()

    return {
        "l0": {"alignment": l0_basis["mean_alignment"] if l0_basis else None,
                "pca": l0_basis["pca"] if l0_basis else None},
        "final": {"alignment": final_basis["mean_alignment"] if final_basis else None,
                  "pca": final_basis["pca"] if final_basis else None},
        "l0_final_cos": cross_cos,
    }


def main():
    print("=" * 60)
    print("  P45: Training Dynamics of Direction Convergence (Stage691)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # Build simple tokenizer (character-level or small vocab)
    # Use a tiny vocab to avoid CUDA indexing issues
    chars = sorted(set("".join(TEXTS)))
    char2id = {c: i+3 for i, c in enumerate(chars)}
    char2id["<pad>"] = 0
    char2id["<eos>"] = 1
    char2id["<unk>"] = 2
    id2char = {v: k for k, v in char2id.items()}
    vocab_size = len(char2id)
    print(f"  vocab_size: {vocab_size} (character-level)")
    print(f"  texts: {len(TEXTS)}")

    # Create dataset
    dataset = TinyTextDataset(TEXTS, char2id, max_len=32)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Create model
    model = TinyGPT2(vocab_size, d_model=256, n_heads=4, n_layers=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    steps_to_measure = [0, 1, 5, 10, 25, 50, 100, 200, 310]
    results = {}
    t_total = time.time()

    step = 0
    measured_steps = set()

    for epoch in range(100):
        if step >= 310:
            break
        for batch in loader:
            if step >= 310:
                break

            # Measure at specific steps
            if step in steps_to_measure and step not in measured_steps:
                r = measure_at_step(model, TEXTS, char2id, device)
                results[str(step)] = r
                measured_steps.add(step)
                l0_a = r["l0"]["alignment"]
                f_a = r["final"]["alignment"]
                l0_pca1 = r["l0"]["pca"][0] if r["l0"]["pca"] else 0
                f_pca1 = r["final"]["pca"][0] if r["final"]["pca"] else 0
                cross = r["l0_final_cos"]
                print(f"  step {step:4d}: L0_align={l0_a:.4f}, Final_align={f_a:.4f}, "
                      f"L0_PCA1={l0_pca1:.4f}, F_PCA1={f_pca1:.4f}, L0-F cos={cross:+.4f}")

            # Training step
            model.train()
            batch = batch.to(device)
            logits, _ = model(batch)

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

            # Handle NaN
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

        if step % 100 == 0:
            print(f"  epoch {epoch}, step {step}, loss={loss.item():.4f}")

    # Final measurement
    if 310 not in measured_steps:
        r = measure_at_step(model, TEXTS, char2id, device)
        results["310"] = r
        l0_a = r["l0"]["alignment"]
        f_a = r["final"]["alignment"]
        cross = r["l0_final_cos"]
        print(f"  step {310:4d}: L0_align={l0_a:.4f}, Final_align={f_a:.4f}, L0-F cos={cross:+.4f}")

    elapsed = time.time() - t_total
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    # Analysis
    print("\n" + "=" * 60)
    print("  ANALYSIS")
    print("=" * 60)

    print("\n  A: L0 Alignment Across Training")
    for s in sorted(results.keys(), key=int):
        r = results[s]
        a = r["l0"]["alignment"]
        pca1 = r["l0"]["pca"][0] if r["l0"]["pca"] else 0
        print(f"  step {s:>4s}: alignment={a:.4f}, PCA1={pca1:.4f}")

    print("\n  B: L0-Final Cross Cosine Across Training")
    for s in sorted(results.keys(), key=int):
        r = results[s]
        c = r["l0_final_cos"]
        if c is not None:
            angle = math.degrees(math.acos(min(abs(c), 1.0)))
            print(f"  step {s:>4s}: cos={c:+.4f}, angle={angle:.1f} deg")

    print("\n  C: Final Layer Alignment Across Training")
    for s in sorted(results.keys(), key=int):
        r = results[s]
        a = r["final"]["alignment"]
        pca1 = r["final"]["pca"][0] if r["final"]["pca"] else 0
        print(f"  step {s:>4s}: alignment={a:.4f}, PCA1={pca1:.4f}")

    # Key question: Does L0 alignment start high?
    step0_l0 = results.get("0", {}).get("l0", {}).get("alignment", 0)
    print(f"\n  D: INV-347 Test: Does L0 alignment start high from random init?")
    print(f"  Step 0 L0 alignment: {step0_l0:.4f}")
    if step0_l0 > 0.9:
        print(f"  -> YES: L0 alignment is high even at random initialization")
        print(f"  -> This confirms L0 alignment is an ARCHITECTURE property, not a trained feature")
        print(f"  -> The 'semantic basis' is NOT a learned structure")
    elif step0_l0 > 0.5:
        print(f"  -> PARTIAL: L0 alignment is moderate at init, increases with training")
    else:
        print(f"  -> NO: L0 alignment is low at init, emerges during training")

    # Save
    out_path = pathlib.Path(r"d:\develop\TransformerLens-main\tests\glm5_temp\stage691_training_basis_20260406_2200\summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "elapsed": elapsed}, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved to: {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
