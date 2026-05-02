import json
data = json.load(open('tests/glm5_temp/ccix_glm4_results.json', encoding='utf-8'))
print('CCIX GLM4 Jacobian (non-quantized):')
for r in data.get('jacobian_animal50', []):
    li = r['layer']
    pp = r['pca_pres_mean']
    rp = r['rand_pres_mean']
    print(f'  L{li:3d}  pca_pres={pp:+.4f}  rand_pres={rp:+.4f}')
