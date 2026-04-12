"""Extract Phase LXXIX results summary"""
import json, glob

for model in ['qwen3', 'glm4', 'deepseek7b']:
    files = glob.glob(f'tests/glm5_temp/phase_lxxix_p396_398_{model}_*.json')
    if not files:
        print(f'{model}: NO FILE')
        continue
    f = files[-1]
    data = json.load(open(f, encoding='utf-8'))
    print(f'\n===== {model} ({f}) =====')
    
    # P396
    if 'p396' in data:
        p = data['p396']
        if 'integral_results' in p:
            print('  P396:')
            for dim, r in p['integral_results'].items():
                adl = r['actual_delta_logit']
                cp = r['cumsum_proj_final']
                cw = r['cumsum_proj_weighted_final']
                lp = r['last_proj']
                print(f'    {dim}: actual_dlogit={adl:.3f}, cumsum_proj={cp:.2f}, cumsum_w={cw:.2f}, last_proj={lp:.2f}')
            if 'metric_r2' in p:
                for m, v in p['metric_r2'].items():
                    r2 = v['r2']
                    print(f'    R2({m})={r2:.4f}')
        else:
            print(f'  P396: ERROR - {str(p)[:100]}')
    else:
        print('  P396: MISSING')
    
    # P397
    if 'p397' in data:
        p = data['p397']
        if 'patching_results' in p:
            print('  P397:')
            for dim, r in p['patching_results'].items():
                kl = r['max_effect_layer']
                md = r['max_effect_dlogit']
                dl = r['dlogit_l0']
                fs = r['first_significant_layer']
                print(f'    {dim}: key_layer=L{kl}, max_dlogit={md:.3f}, L0_dlogit={dl:.3f}, first>50%=L{fs}')
        else:
            print(f'  P397: ERROR - {str(p)[:100]}')
    else:
        print('  P397: MISSING')
    
    # P398
    if 'p398' in data:
        p = data['p398']
        if 'lcs' in p:
            lcs = p['lcs']
            v = lcs['V_lang']
            r = lcs['R_rotate']
            c = lcs['C_compete']
            m = lcs['M_map']
            print('  P398 LCS:')
            er = v['effective_rank']
            ac = v['avg_cos_between_dims']
            mc = v['max_cos_between_dims']
            print(f'    V_lang: rank={er}, avg_cos={ac:.4f}, max_cos={mc:.4f}')
            for dim, rd in r.items():
                g = rd['gamma']
                hl = rd['half_life_layers']
                cm = rd['cos_at_mid_layer']
                cl = rd['cos_at_last_layer']
                print(f'    R_rotate({dim}): gamma={g:.4f}, half_life={hl:.1f}, cos_mid={cm:.3f}, cos_last={cl:.3f}')
            for pair, cd in c.items():
                norm = cd['interaction_norm']
                d1 = cd['interact_dim1']
                d2 = cd['interact_dim2']
                print(f'    C_compete({pair}): norm={norm:.1f}, d1={d1:.3f}, d2={d2:.3f}')
            shape = m['W_lm_shape']
            cond = m['condition_number']
            print(f'    M_map: shape={shape}, cond_num={cond:.1e}')
        else:
            print(f'  P398: ERROR - {str(p)[:100]}')
    else:
        print('  P398: MISSING')
