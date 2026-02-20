import json, os
p = "d:/develop/TransformerLens-main/tempdata/phase3_wiki_results.json"
if os.path.exists(p):
    d = json.load(open(p))
    # Update Ep 12
    found=False
    for x in d:
        if x['ep'] == 12:
            x['id'] = 23.8478
            found=True
    if not found:
        d.append({'ep': 12, 'loss': 0.8816, 'id': 23.8478})
    print(f"Patched: {d[-1]}")
    json.dump(d, open(p, 'w'))
print("Done")
