import json
with open('tests/glm5_temp/ccxxxix_simplex_rigorous_qwen3.json','r',encoding='utf-8') as f:
    data = json.load(f)
geom = data.get('geometry', {})
for domain in ['habitat','emotion','occupation']:
    dg = geom.get(domain, {})
    if not dg:
        print(f'{domain}: no geometry data')
        continue
    for lk in sorted(dg.keys()):
        v = dg[lk]
        if v is None:
            print(f'{domain} {lk}: None')
            continue
        fr2 = v.get('simplex_fit_r2', -1)
        eu = v.get('edge_uniformity', 0)
        ad = v.get('angle_deviation', 0)
        ndim = v.get('n_dim', 0)
        print(f'{domain} {lk}: fit_r2={fr2:.6f}, edge_uni={eu:.4f}, angle_dev={ad:.2f}, n_dim={ndim}')
