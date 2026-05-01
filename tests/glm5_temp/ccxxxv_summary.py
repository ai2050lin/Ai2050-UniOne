"""临时脚本: 汇总CCXXXV三个模型的关键结果"""
import json

for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = f'tests/glm5_temp/ccxxxv_residual_patching_{model}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'\n{"="*60}')
    print(f' {model.upper()} (n_layers={data["n_layers"]}, d_model={data["d_model"]})')
    print(f'{"="*60}')
    
    # Part 1: Residual patching summary
    rp = data.get('residual_patching', {})
    summary = rp.get('summary', [])
    trans = rp.get('transition_point', {})
    print(f'\n--- Residual Patching Summary ---')
    if trans:
        print(f'  Max change layer: L{trans["max_change_layer"]}, value={trans["max_change_value"]}')
        print(f'  First significant layer: L{trans.get("first_significant_layer", "N/A")}')
    
    # Print layer-by-layer change_ratio
    print('  Layer | comp_change | last_change')
    for entry in summary:
        print(f'  L{entry["layer"]:3d} | {entry["mean_change_comp"]:+.4f} | {entry["mean_change_last"]:+.4f}')
    
    # Part 2: Semantic extreme
    sem = data.get('semantic_extreme', {})
    print(f'\n--- Semantic Dimension Results ---')
    for key, val in sorted(sem.items()):
        print(f'  {key}: n_sep={val["n_separating_PCs"]}, expected_N-1={val["expected_dim_simplex"]}')
    
    # Part 3: Geometry
    geo = data.get('geometry_2d', {})
    if geo:
        print(f'\n--- 2D Geometry (L{geo["analysis_layer"]}) ---')
        print(f'  Ideal angle sep: {geo["ideal_angular_sep"]} deg')
        print(f'  Mean angle sep: {geo["mean_angular_sep"]} deg')
        print(f'  Angular uniformity: {geo["angular_uniformity"]}')
        print(f'  n_separating_6class: {geo["n_separating_PCs_6class"]} (expected N-1={geo["expected_dim_simplex"]})')
        for hab in geo.get('sorted_by_angle', []):
            g = geo['habitat_geometry'][hab]
            print(f'  {hab:12s}: center=({g["center_2d"][0]:+.2f},{g["center_2d"][1]:+.2f}), '
                  f'angle={g["angle_degrees"]:+.1f} deg, radius={g["radius_2d"]:.3f}')
