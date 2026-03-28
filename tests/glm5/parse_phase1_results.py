"""解析Phase 1结果"""
import json
import ast
import sys
sys.stdout.reconfigure(encoding='utf-8')

def sfloat(v):
    """Safe float conversion."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try: return float(v)
        except: return 0.0
    return 0.0

def sfloat_list(v):
    """Safe float list conversion."""
    if isinstance(v, str):
        try:
            p = ast.literal_eval(v)
            if isinstance(p, (list, tuple)):
                return [sfloat(x) for x in p]
        except: pass
        return []
    if isinstance(v, (list, tuple)):
        return [sfloat(x) for x in v]
    return []

with open('d:/ai2050/TransformerLens-Project/tests/glm5/phase1_results/phase1_comprehensive_20260328_093908.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=== MODELS ANALYZED ===')
for m in data['models']:
    md = data['models'][m]
    if 'error' in md:
        print(f'  {m}: ERROR')
        continue
    cfg = md.get('config', {})
    print(f'  {m}: {cfg.get("num_layers")}L, {cfg.get("hidden_size")}D, {cfg.get("num_heads")}H, {cfg.get("num_parameters",0):,} params')

for model_name in data['models']:
    md = data['models'][model_name]
    if 'error' in md:
        continue
    
    print(f'\n{"="*60}')
    print(f'  {model_name.upper()} RESULTS')
    print(f'{"="*60}')
    
    # 1. Embedding
    emb = md.get('embedding', {})
    print(f'\n--- 1. EMBEDDING ---')
    print(f'  Shape: {emb.get("token_embed_shape")}')
    sv = sfloat_list(emb.get('singular_values_top20', []))
    if sv:
        print(f'  Top 5 singular values: {[f"{x:.1f}" for x in sv[:5]]}')
    cv = sfloat_list(emb.get('singular_values_cumvar_top20', []))
    if cv:
        print(f'  CumVar(PC1-5): {[f"{x:.4f}" for x in cv[:5]]}')
        print(f'  -> Top 5 PCs explain {cv[4]*100:.1f}% of variance')
    euc = sfloat(emb.get('embed_unembed_cosine_similarity', 0))
    print(f'  Embed-Unembed cosine: {euc:.6f}')
    eu_diff = sfloat(emb.get('embed_unembed_diff_relative', 0))
    if eu_diff > 0:
        print(f'  Embed-Unembed diff (relative): {eu_diff:.6f}')
    clust = emb.get('embedding_clustering', {})
    if clust:
        ratio = sfloat(clust.get('intra_over_inter_ratio', 0))
        if ratio > 0:
            print(f'  Clustering intra/inter ratio: {ratio:.4f}')
            if ratio < 1:
                print(f'    -> Categories ARE clustered (intra < inter)')
            else:
                print(f'    -> Categories are NOT well separated')
    
    tc = emb.get('top_components', [])
    if tc:
        for c in tc[:3]:
            ev = sfloat(c.get('explained_variance', 0))
            print(f'  PC{c.get("component","?")} (var={ev*100:.1f}%):')
            print(f'    Top tokens: {[t.encode("ascii","replace").decode() for t in c.get("top_tokens",[])[:5]]}')
            print(f'    Bot tokens: {[t.encode("ascii","replace").decode() for t in c.get("bottom_tokens",[])[:5]]}')

    # 2. Residual geometry
    rg = md.get('residual_geometry', {})
    print(f'\n--- 2. RESIDUAL STREAM GEOMETRY ---')
    for case_name, case_data in rg.items():
        if case_name == 'pairwise_analysis':
            continue
        ld = case_data.get('layer_data', [])
        if ld:
            norms = [sfloat(d['mean_norm']) for d in ld]
            ranks = [sfloat(d['effective_rank']) for d in ld]
            dims = [sfloat(d['effective_dim']) for d in ld]
            print(f'  [{case_name}]')
            if norms[0] > 0:
                print(f'    Norm: {norms[0]:.2f} -> {norms[-1]:.2f} (x{norms[-1]/norms[0]:.2f})')
            print(f'    EffRank: {ranks[0]:.1f} -> {ranks[-1]:.1f}')
            print(f'    EffDim: {dims[0]:.1f} -> {dims[-1]:.1f}')

    pw = rg.get('pairwise_analysis', {})
    if pw:
        print(f'  Pairwise attention sensitivity:')
        for pname, pd in pw.items():
            print(f'    {pname}: max_diff_layer={pd.get("max_diff_layer")}, mean={sfloat(pd.get("mean_diff",0)):.6f}')

    # 3. Attention
    attn = md.get('attention_routing', {})
    hfs = attn.get('head_function_summary', {})
    if hfs:
        print(f'\n--- 3. ATTENTION HEAD ROUTING ---')
        func_counts = {}
        for hid, hd in hfs.items():
            f = hd.get('dominant_function', 'unknown')
            func_counts[f] = func_counts.get(f, 0) + 1
        print(f'  Head function distribution ({len(hfs)} heads):')
        for f, c in sorted(func_counts.items(), key=lambda x: -x[1]):
            print(f'    {f}: {c} heads')
        
        for func in ['prev_token', 'induction', 'remote_dep']:
            top_heads = sorted(
                [(h, v) for h, v in hfs.items() if v.get('dominant_function') == func],
                key=lambda x: sfloat(x[1].get(func, 0)), reverse=True
            )[:3]
            if top_heads:
                print(f'  Top {func} heads:')
                for h, v in top_heads:
                    print(f'    {h}: score={sfloat(v.get(func, 0)):.4f}')

    for key, val in attn.items():
        if isinstance(val, dict) and 'layer_diffs' in val:
            diffs = sfloat_list(val.get('layer_diffs', []))
            if diffs:
                print(f'  Sentence pair [{key[:40]}]:')
                print(f'    mean_diff={sfloat(val.get("mean_diff",0)):.6f}, max_layer={val.get("max_diff_layer")}')

    # 4. FFN
    ffn = md.get('ffn_transformations', {})
    print(f'\n--- 4. FFN TRANSFORMATIONS ---')
    ratios = ffn.get('transform_ratios', {})
    counts = ffn.get('transform_counts', {})
    total = sfloat(ffn.get('total_neurons_analyzed', 0))
    print(f'  Neurons analyzed: {int(total)}')
    for t, r in sorted(ratios.items(), key=lambda x: -sfloat(x[1])):
        print(f'    {t}: {sfloat(r)*100:.1f}% ({counts.get(t,0)} neurons)')

    # 5. Computation dependency
    cd = md.get('computation_dependency', {})
    print(f'\n--- 5. COMPUTATION DEPENDENCY ---')
    lds = cd.get('layer_distance_summary', [])
    ewd = cd.get('embedding_word_distances', [])
    if lds:
        if ewd:
            ed = [sfloat(d.get('embed_dist',0)) for d in ewd]
            print(f'  Embedding mean word-pair dist: {sum(ed)/len(ed):.4f}')
        print(f'  Layer-by-layer mean word-pair distances:')
        for ld in lds:
            d = sfloat(ld['mean_dist'])
            bar = '#' * max(1, int(d * 2))
            print(f'    L{ld["layer"]:2d}: dist={d:.4f} std={sfloat(ld["std_dist"]):.4f} cos={sfloat(ld["mean_cosine"]):.4f} {bar}')
        
        if len(lds) >= 2:
            trend = sfloat(lds[-1]['mean_dist']) - sfloat(lds[0]['mean_dist'])
            print(f'  Distance trend L0->L{lds[-1]["layer"]}: {trend:+.4f}')
            if trend > 0:
                print(f'  -> Semantic SEPARATION increases through layers')
            else:
                print(f'  -> Semantic SEPARATION decreases (convergence)')
