"""
CCXIX(319): 直接比较推理测试 — 传递性是否线性?
======================================================================
CCXVI的传递性测试有方法缺陷: "比较语境增量"≠"推理方向"。
本实验: 直接用"A is bigger than B" vs "A is bigger than C"的残差差
来定义"A>B"方向, 正确测试传递性。

核心问题:
  1. 直接比较方向是否可传递? vec(A>B) + vec(B>C) ≈ vec(A>C)?
  2. 比较方向是否线性? 即"A>B"方向与B无关?
  3. 3维语义流形的推理几何?

设计:
  - 固定A, 变B: "A is bigger than B1" vs "A is bigger than B2"
  - 差 = B的语义方向(B1 vs B2)在比较语境中的投影
  - 固定B, 变A: "A1 is bigger than B" vs "A2 is bigger than B"
  - 差 = A的语义方向(A1 vs A2)在比较语境中的投影
  - 传递性: vec(A>C) ≈ vec(A>B) + vec(B>C)?

用法:
  python ccxix_direct_comparison.py --model qwen3
  python ccxix_direct_comparison.py --model glm4
  python ccxix_direct_comparison.py --model deepseek7b
"""
import argparse, os, sys, json
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy import stats

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxix_direct_comparison_log.txt"

# 比较词对: (大, 小) — size维度
SIZE_PAIRS = [
    ("elephant", "mouse"), ("whale", "fish"), ("horse", "cat"),
    ("lion", "rabbit"), ("eagle", "sparrow"), ("bear", "fox"),
    ("cow", "chicken"), ("shark", "crab"), ("tiger", "rat"),
    ("mountain", "hill"), ("tree", "bush"), ("house", "shed"),
]

# weight维度
WEIGHT_PAIRS = [
    ("iron", "feather"), ("rock", "leaf"), ("steel", "paper"),
    ("gold", "cotton"), ("lead", "silk"), ("stone", "grass"),
    ("concrete", "foam"), ("brick", "straw"), ("copper", "wool"),
    ("marble", "petal"), ("silver", "dust"), ("bronze", "thimble"),
]

# speed维度
SPEED_PAIRS = [
    ("cheetah", "turtle"), ("falcon", "snail"), ("horse", "slug"),
    ("rocket", "cart"), ("jet", "boat"), ("leopard", "worm"),
    ("eagle", "ant"), ("tiger", "sloth"), ("deer", "beetle"),
    ("train", "wheelbarrow"), ("car", "bicycle"), ("plane", "canoe"),
]

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def collect_resid(model, tokenizer, device, layers, prompt, test_layers):
    """收集各层残差"""
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    lp = toks.input_ids.shape[1] - 1
    cap = {}
    def mk(k):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[k] = o[0, lp, :].detach().float().cpu().numpy()
        return hook
    hooks = [layers[li].register_forward_hook(mk(f"L{li}")) for li in test_layers]
    with torch.no_grad(): _ = model(**toks)
    for h in hooks: h.remove()
    return cap

def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model

    test_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]))

    log(f"\n{'='*70}\nCCXIX(319): 直接比较推理 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")

    results = {}

    # ===== Part 1: 直接比较方向 =====
    # "A is bigger than B" 的残差 — 这就是 vec(A>B)
    # 传递性测试: vec(A>B) + vec(B>C) 是否≈ vec(A>C)?

    for dim_name, pairs in [("size", SIZE_PAIRS), ("weight", WEIGHT_PAIRS), ("speed", SPEED_PAIRS)]:
        log(f"\n=== 维度: {dim_name} ({len(pairs)}对) ===")

        # 收集所有比较残差
        comp_resids = {}  # {(A,B): {li: vec}}
        for A, B in pairs:
            prompt = f"{A} is {dim_name}r than {B}" if dim_name != "size" else f"{A} is bigger than {B}"
            if dim_name == "weight":
                prompt = f"{A} is heavier than {B}"
            elif dim_name == "speed":
                prompt = f"{A} is faster than {B}"
            else:
                prompt = f"{A} is bigger than {B}"
            cap = collect_resid(model, tokenizer, device, layers, prompt, test_layers)
            comp_resids[(A,B)] = cap

        # Part 1a: 传递性线性度
        # 找三元组 (A,B,C) = (pairs[i][0], pairs[i][1], pairs[j][1])
        # vec(A>B) + vec(B>C) ≈ vec(A>C)?
        log("\n  --- 1a: 传递性线性度 ---")

        transitivity_cos = {li: [] for li in test_layers}

        for i in range(len(pairs)):
            for j in range(len(pairs)):
                if i == j: continue
                A, B1 = pairs[i]
                A2, B2 = pairs[j]
                # 需要 A>B1 和 A>B2 来定义 "A比B1 vs A比B2"
                # 更好: 需要 B1>B2 来测传递性
                pass

        # 更直接的传递性测试:
        # 选3个词 A>B>C, 测 vec(A>B) + vec(B>C) vs vec(A>C)
        # 用同一维度的pairs构造三元组
        triples = []
        for i in range(min(6, len(pairs))):
            for j in range(min(6, len(pairs))):
                if i == j: continue
                A, B = pairs[i][0], pairs[i][1]  # A > B
                # B > C? 找B是大的pair
                for k in range(min(6, len(pairs))):
                    if pairs[k][0] == B:  # B > C
                        C = pairs[k][1]
                        triples.append((A, B, C))

        # 如果没找到自然三元组, 用人工构造
        if dim_name == "size" and not triples:
            # elephant > horse > cat > mouse
            triples = [
                ("elephant", "horse", "cat"),
                ("elephant", "horse", "mouse"),
                ("horse", "cat", "mouse"),
                ("whale", "shark", "fish"),
                ("whale", "dolphin", "crab"),
                ("lion", "dog", "rabbit"),
            ]
        elif dim_name == "weight" and not triples:
            triples = [
                ("iron", "rock", "leaf"),
                ("steel", "stone", "paper"),
                ("gold", "copper", "wool"),
            ]
        elif dim_name == "speed" and not triples:
            triples = [
                ("cheetah", "horse", "turtle"),
                ("falcon", "deer", "snail"),
                ("rocket", "car", "cart"),
            ]

        log(f"  三元组数: {len(triples)}")

        # 收集三元组残差
        for A, B, C in triples[:8]:
            if dim_name == "size":
                p_ab = f"{A} is bigger than {B}"
                p_bc = f"{B} is bigger than {C}"
                p_ac = f"{A} is bigger than {C}"
            elif dim_name == "weight":
                p_ab = f"{A} is heavier than {B}"
                p_bc = f"{B} is heavier than {C}"
                p_ac = f"{A} is heavier than {C}"
            else:
                p_ab = f"{A} is faster than {B}"
                p_bc = f"{B} is faster than {C}"
                p_ac = f"{A} is faster than {C}"

            r_ab = collect_resid(model, tokenizer, device, layers, p_ab, test_layers)
            r_bc = collect_resid(model, tokenizer, device, layers, p_bc, test_layers)
            r_ac = collect_resid(model, tokenizer, device, layers, p_ac, test_layers)

            for li in test_layers:
                v_ab = r_ab.get(f"L{li}")
                v_bc = r_bc.get(f"L{li}")
                v_ac = r_ac.get(f"L{li}")
                if v_ab is None or v_bc is None or v_ac is None:
                    continue

                # 传递性: cos(vec(AB)+vec(BC), vec(AC))
                v_sum = v_ab + v_bc
                norm_sum = np.linalg.norm(v_sum)
                norm_ac = np.linalg.norm(v_ac)
                if norm_sum > 1e-10 and norm_ac > 1e-10:
                    cos_trans = float(np.dot(v_sum, v_ac) / (norm_sum * norm_ac))
                    transitivity_cos[li].append(cos_trans)

        for li in test_layers:
            vals = transitivity_cos[li]
            if vals:
                log(f"  L{li}: 传递性cos mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")
                k = f"transitivity_{dim_name}_L{li}"
                results[k] = {"dim":dim_name, "layer":li,
                              "mean_cos":float(np.mean(vals)),
                              "std_cos":float(np.std(vals)),
                              "n_triples":len(vals)}

        # Part 1b: 比较方向是否独立于B?
        # "A is bigger than B1" vs "A is bigger than B2" → 差 = f(A, B1, B2)
        # 如果比较方向独立于B, 那么差应该≈0
        log("\n  --- 1b: 比较方向对B的依赖 ---")

        b_dependence = {li: [] for li in test_layers}

        for A, _ in pairs[:6]:
            # 收集 "A is bigger than B" 对不同B
            B_list = [p[1] for p in pairs[:6] if p[0] != A][:4]
            if len(B_list) < 2: continue

            resids_by_B = {}
            for B in B_list:
                if dim_name == "size":
                    prompt = f"{A} is bigger than {B}"
                elif dim_name == "weight":
                    prompt = f"{A} is heavier than {B}"
                else:
                    prompt = f"{A} is faster than {B}"
                resids_by_B[B] = collect_resid(model, tokenizer, device, layers, prompt, test_layers)

            # 两两比较: 不同B的残差差异
            B_keys = list(resids_by_B.keys())
            for i in range(len(B_keys)):
                for j in range(i+1, len(B_keys)):
                    for li in test_layers:
                        v1 = resids_by_B[B_keys[i]].get(f"L{li}")
                        v2 = resids_by_B[B_keys[j]].get(f"L{li}")
                        if v1 is not None and v2 is not None:
                            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                            if n1 > 1e-10 and n2 > 1e-10:
                                cos = float(np.dot(v1, v2) / (n1 * n2))
                                b_dependence[li].append(cos)

        for li in test_layers:
            vals = b_dependence[li]
            if vals:
                log(f"  L{li}: B-dependence cos mean={np.mean(vals):.3f} (1=独立于B, 0=依赖B)")
                k = f"b_dependence_{dim_name}_L{li}"
                results[k] = {"dim":dim_name, "layer":li,
                              "mean_cos":float(np.mean(vals)),
                              "n_pairs":len(vals)}

        # Part 1c: 比较方向是否独立于A?
        # "A1 is bigger than B" vs "A2 is bigger than B" → 差 = f(A1, A2, B)
        log("\n  --- 1c: 比较方向对A的依赖 ---")

        a_dependence = {li: [] for li in test_layers}

        for _, B in pairs[:6]:
            A_list = [p[0] for p in pairs[:6] if p[1] != B][:4]
            if len(A_list) < 2: continue

            resids_by_A = {}
            for A in A_list:
                if dim_name == "size":
                    prompt = f"{A} is bigger than {B}"
                elif dim_name == "weight":
                    prompt = f"{A} is heavier than {B}"
                else:
                    prompt = f"{A} is faster than {B}"
                resids_by_A[A] = collect_resid(model, tokenizer, device, layers, prompt, test_layers)

            A_keys = list(resids_by_A.keys())
            for i in range(len(A_keys)):
                for j in range(i+1, len(A_keys)):
                    for li in test_layers:
                        v1 = resids_by_A[A_keys[i]].get(f"L{li}")
                        v2 = resids_by_A[A_keys[j]].get(f"L{li}")
                        if v1 is not None and v2 is not None:
                            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                            if n1 > 1e-10 and n2 > 1e-10:
                                cos = float(np.dot(v1, v2) / (n1 * n2))
                                a_dependence[li].append(cos)

        for li in test_layers:
            vals = a_dependence[li]
            if vals:
                log(f"  L{li}: A-dependence cos mean={np.mean(vals):.3f} (1=独立于A, 0=依赖A)")
                k = f"a_dependence_{dim_name}_L{li}"
                results[k] = {"dim":dim_name, "layer":li,
                              "mean_cos":float(np.mean(vals)),
                              "n_pairs":len(vals)}

    # ===== Part 2: 跨维度推理一致性 =====
    log("\n--- Part 2: 跨维度推理一致性 ---")

    # 同一对词, 不同比较维度: "A is bigger than B" vs "A is faster than B"
    # 如果推理是统一的, 那么残差应该高度相关
    cross_dim_cos = {li: [] for li in test_layers}

    cross_pairs = [("elephant","mouse"), ("whale","fish"), ("horse","cat"),
                   ("lion","rabbit"), ("eagle","sparrow"), ("bear","fox")]

    for A, B in cross_pairs:
        r_big = collect_resid(model, tokenizer, device, layers,
                              f"{A} is bigger than {B}", test_layers)
        r_fast = collect_resid(model, tokenizer, device, layers,
                               f"{A} is faster than {B}", test_layers)
        r_heavy = collect_resid(model, tokenizer, device, layers,
                                f"{A} is heavier than {B}", test_layers)

        for li in test_layers:
            vb = r_big.get(f"L{li}")
            vf = r_fast.get(f"L{li}")
            vh = r_heavy.get(f"L{li}")
            if vb is not None and vf is not None:
                n1, n2 = np.linalg.norm(vb), np.linalg.norm(vf)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos = float(np.dot(vb, vf) / (n1 * n2))
                    cross_dim_cos[li].append(cos)

    for li in test_layers:
        vals = cross_dim_cos[li]
        if vals:
            log(f"  L{li}: 跨维度cos(bigger,faster) mean={np.mean(vals):.3f}")
            k = f"cross_dim_L{li}"
            results[k] = {"layer":li, "mean_cos":float(np.mean(vals)),
                          "n_pairs":len(vals)}

    # 保存
    out = TEMP / f"ccxix_direct_comparison_{model_name}.json"
    with open(out, "w") as f:
        json.dump({"model":model_name,"d_model":d_model,"n_layers":n_layers,
                    "results":results}, f, indent=2, default=str)
    log(f"\n保存: {out}")
    release_model(model)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3","glm4","deepseek7b"])
    TEMP.mkdir(parents=True, exist_ok=True)
    run(parser.parse_args().model)
