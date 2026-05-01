"""
CCXVIII(318): 注意力头级别perturb — DS7B悖论的解答?
CCXIV发现DS7B定向perturb=0%, 但K1=98.4%。
假设: 控制通过attention head路由, 不通过residual线性映射。
本实验: 在attn head级别perturb, 看DS7B是否可控。

用法:
  python ccxviii_attn_head_perturb.py --model qwen3
  python ccxviii_attn_head_perturb.py --model glm4
  python ccxviii_attn_head_perturb.py --model deepseek7b
"""
import argparse, os, sys, json
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict
from scipy import stats

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxviii_attn_head_perturb_log.txt"

HAB_WORDS = {
    "land": ["dog","cat","lion","tiger","horse","cow","sheep","rabbit","fox","deer"],
    "ocean": ["whale","shark","dolphin","octopus","salmon","turtle","crab","seal","squid","lobster"],
    "sky": ["eagle","hawk","owl","parrot","crow","sparrow","swallow","falcon","pigeon","robin"],
}

HAB_TOKENS = {
    "land": ["land","ground","earth","field","forest","plains","jungle","savanna"],
    "ocean": ["ocean","sea","water","river","lake","deep","marine","coastal"],
    "sky": ["sky","air","trees","mountains","heights","clouds","nests","branches"],
}

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def get_tok_ids(tokenizer, words):
    ids = []
    for w in words:
        t = tokenizer.encode(" " + w, add_special_tokens=False)
        if t:
            ids.append((w, t[0]))
    return ids

def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    # Get n_heads from config (different models use different attr names)
    cfg = model.config
    n_heads = getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_heads', d_model // getattr(cfg, 'head_dim', 64)))
    d_head = d_model // n_heads

    all_words, word_hab = [], {}
    for h, ws in HAB_WORDS.items():
        all_words.extend(ws)
        for w in ws: word_hab[w] = h

    test_layers = sorted(set([n_layers//2, 3*n_layers//4, n_layers-3, n_layers-2, n_layers-1]))

    log(f"\n{'='*70}\nCCXVIII(318): Attn Head Perturb - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, d_head={d_head}")
    log(f"{'='*70}")

    results = {}

    # Step 1: 找habitat控制head — 扫描每个head的habitat区分力
    log("\n--- Step 1: 扫描habitat控制head ---")
    scan_words = ["dog","whale","eagle","cat","shark","hawk","lion","dolphin","owl","salmon"]
    head_hab_f = {}

    for li in test_layers:
        layer = layers[li]
        head_outs = {}

        for word in scan_words:
            prompt = f"The {word} lives in the"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            lp = toks.input_ids.shape[1] - 1
            cap = [None]
            def mk_hook(c):
                def hook(m, inp, out):
                    c[0] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
                return hook
            h = layer.self_attn.register_forward_hook(mk_hook(cap))
            with torch.no_grad(): _ = model(**toks)
            h.remove()
            if cap[0] is not None:
                head_outs[word] = cap[0][0, lp, :].numpy().reshape(n_heads, d_head)

        for hi in range(n_heads):
            groups = {"land":[], "ocean":[], "sky":[]}
            for word in scan_words:
                if word in head_outs:
                    hab = word_hab.get(word,"")
                    if hab in groups:
                        groups[hab].append(np.linalg.norm(head_outs[word][hi]))
            if all(len(v)>=2 for v in groups.values()):
                f, _ = stats.f_oneway(groups["land"], groups["ocean"], groups["sky"])
                if not np.isnan(f):
                    head_hab_f[(li,hi)] = f

        lh = sorted([(h,f) for (l,h),f in head_hab_f.items() if l==li], key=lambda x:-x[1])
        log(f"  L{li} top-3: {[(h,f'{f:.1f}') for h,f in lh[:3]]}")

    # Step 2: 计算per-head habitat方向 + perturb
    log("\n--- Step 2: Per-head habitat perturb ---")

    hab_tok_ids = {h: get_tok_ids(tokenizer, ws) for h, ws in HAB_TOKENS.items()}

    for li in test_layers:
        lh = sorted([(h,f) for (l,h),f in head_hab_f.items() if l==li], key=lambda x:-x[1])
        if not lh: continue
        top_hi, top_f = lh[0]  # 最强head
        layer = layers[li]

        log(f"\n  === L{li} Head{top_hi} (F={top_f:.1f}) ===")

        # 计算per-head habitat方向
        hab_means = {}
        for hab in ["land","ocean","sky"]:
            vecs = []
            for word in HAB_WORDS[hab][:5]:
                prompt = f"The {word} lives in the"
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                lp = toks.input_ids.shape[1] - 1
                cap = [None]
                def mk_hook(c):
                    def hook(m, inp, out):
                        c[0] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
                    return hook
                h = layer.self_attn.register_forward_hook(mk_hook(cap))
                with torch.no_grad(): _ = model(**toks)
                h.remove()
                if cap[0] is not None:
                    vecs.append(cap[0][0, lp, :].numpy().reshape(n_heads, d_head)[top_hi])
            if vecs:
                hab_means[hab] = np.mean(vecs, axis=0)

        if len(hab_means) < 3: continue
        overall_mean = np.mean(list(hab_means.values()), axis=0)

        hab_dirs = {}
        for hab in ["land","ocean","sky"]:
            d = hab_means[hab] - overall_mean
            n = np.linalg.norm(d)
            hab_dirs[hab] = d/n if n > 1e-10 else d

        # 对每个目标栖息地做perturb
        for target_hab in ["land","ocean","sky"]:
            direction = hab_dirs[target_hab]
            for alpha in [2.0, 4.0]:
                succ, total = 0, 0

                for word in all_words[:15]:
                    prompt = f"The {word} lives in the"
                    toks = tokenizer(prompt, return_tensors="pt").to(device)
                    lp = toks.input_ids.shape[1] - 1

                    # Baseline
                    with torch.no_grad(): base_out = model(**toks)
                    base_logits = base_out.logits[0,-1,:].float().cpu().numpy()

                    # Head perturb: hook attn output, 对特定head加perturb
                    perturb = alpha * direction  # [d_head]
                    done = [False]
                    def mk_perturb_hook(pv, hi, nh, dh, dn, lpos):
                        def hook(m, inp, out):
                            if dn[0]: return out
                            dn[0] = True
                            o = out[0] if isinstance(out, tuple) else out
                            new_o = o.clone()
                            # 拆分heads, 对特定head加perturb, 再合并
                            head_view = new_o[0, lpos, :].view(nh, dh)
                            pt = torch.tensor(pv, dtype=head_view.dtype, device=o.device)
                            head_view[hi, :] += pt
                            if isinstance(out, tuple):
                                return (new_o,) + out[1:]
                            return new_o
                        return hook

                    hh = layer.self_attn.register_forward_hook(
                        mk_perturb_hook(perturb, top_hi, n_heads, d_head, done, lp))
                    with torch.no_grad(): pert_out = model(**toks)
                    pert_logits = pert_out.logits[0,-1,:].float().cpu().numpy()
                    hh.remove()

                    delta = pert_logits - base_logits
                    target_shift = np.mean([delta[tid] for _,tid in hab_tok_ids[target_hab] if tid<len(delta)])
                    other_shifts = []
                    for hab2 in ["land","ocean","sky"]:
                        if hab2 != target_hab:
                            other_shifts.append(np.mean([delta[tid] for _,tid in hab_tok_ids[hab2] if tid<len(delta)]))
                    other_shift = np.mean(other_shifts) if other_shifts else 0

                    total += 1
                    if target_shift > other_shift + 0.1:
                        succ += 1

                rate = succ / max(total, 1)
                log(f"    {target_hab} alpha={alpha}: 成功率={rate:.2%} ({succ}/{total})")

                k = f"head_perturb_L{li}_H{top_hi}_{target_hab}_a{alpha}"
                results[k] = {"layer":li, "head":top_hi, "target_hab":target_hab,
                              "alpha":alpha, "success_rate":float(rate),
                              "success_count":succ, "total":total}

    # Step 3: 对比residual perturb vs head perturb (只用最强head)
    log("\n--- Step 3: Residual vs Head perturb对比 ---")
    li_last = n_layers - 1
    lh = sorted([(h,f) for (l,h),f in head_hab_f.items() if l==li_last], key=lambda x:-x[1])
    if lh:
        top_hi, _ = lh[0]
        # 收集head perturb结果
        head_rates = []
        for hab in ["land","ocean","sky"]:
            k = f"head_perturb_L{li_last}_H{top_hi}_{hab}_a4.0"
            if k in results:
                head_rates.append(results[k]["success_rate"])

        log(f"  Head perturb (alpha=4) 平均成功率: {np.mean(head_rates):.2%}" if head_rates else "  No head perturb data")
        log(f"  (CCXIV Residual perturb DS7B=0%, GLM4=100%)")

        results["comparison"] = {
            "model": model_name,
            "best_head": top_hi,
            "head_perturb_mean_rate": float(np.mean(head_rates)) if head_rates else 0,
            "residual_perturb_note": "See CCXIV results",
        }

    out = TEMP / f"ccxviii_attn_head_perturb_{model_name}.json"
    with open(out, "w") as f:
        json.dump({"model":model_name,"d_model":d_model,"n_layers":n_layers,
                    "n_heads":n_heads,"results":results}, f, indent=2, default=str)
    log(f"\n保存: {out}")
    release_model(model)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3","glm4","deepseek7b"])
    TEMP.mkdir(parents=True, exist_ok=True)
    run(parser.parse_args().model)
