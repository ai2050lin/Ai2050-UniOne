import json

for model_name in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'tests/glm5_temp/ccxlii_discriminative_{model_name}.json','r',encoding='utf-8') as f:
        data = json.load(f)
    print(f'\n=== {model_name} ===')
    rb = data['random_baseline_subspace']
    mr = data['model_results']
    for domain, dr in mr.items():
        N = dr['N']
        rb_n = rb.get(str(N), {})
        rb_fit = rb_n.get('fit_r2', {})
        rb_eu = rb_n.get('edge_uniformity', {})
        rb_ad = rb_n.get('angle_deviation', {})
        
        s_fit = dr['best_fit_r2_Nm1']
        s_eu = dr['best_edge_uni_Nm1']
        s_ad = dr['best_angle_dev_Nm1']
        
        fit_p95 = rb_fit.get('p95', 0)
        eu_p95 = rb_eu.get('p95', 0)
        ad_p5 = rb_ad.get('p5', 999)
        
        above_p95 = s_fit > fit_p95
        eu_above = s_eu > eu_p95
        ad_below = s_ad < ad_p5
        
        fit_mark = ">>>" if above_p95 else "<<<"
        eu_mark = ">>>" if eu_above else "<<<"
        ad_mark = "<<<" if ad_below else ">>>"
        
        n_strong = sum([above_p95, eu_above, ad_below])
        verdict = ["X 无", "* 弱", "** 中", "*** 强"][n_strong]
        
        print(f"  {domain:>15} N={N}: fit_r2={s_fit:.4f}(r{fit_p95:.4f}){fit_mark} "
              f"eu={s_eu:.4f}(r{eu_p95:.4f}){eu_mark} "
              f"ad={s_ad:.2f}(r{ad_p5:.2f}){ad_mark} "
              f"= {verdict}")
