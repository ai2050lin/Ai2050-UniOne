import json
for model in ['qwen3', 'glm4']:
    with open(f'tests/glm5_temp/ccxxxix_simplex_rigorous_{model}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    geom = data.get('geometry', {})
    print(f'\n=== {model.upper()} best geometry ===')
    for domain in ['habitat', 'emotion', 'occupation']:
        domain_geom = geom.get(domain, {})
        sqi_prof = data['sqi_profiles'].get(domain, {})
        best_lkey = max(sqi_prof.keys(), key=lambda k: sqi_prof[k]['sqi'])
        g = domain_geom.get(best_lkey, {})
        if g:
            print(f"  {domain} {best_lkey}: fit_r2={g.get('simplex_fit_r2',0):.4f}, "
                  f"edge_uni={g.get('edge_uniformity',0):.3f}, "
                  f"angle_dev={g.get('angle_deviation',0):.1f}, "
                  f"cv_edge={g.get('cv_edge',0):.3f}, "
                  f"cv_radius={g.get('cv_radius',0):.3f}")
        else:
            print(f"  {domain} {best_lkey}: no geometry (n_sep<2)")
