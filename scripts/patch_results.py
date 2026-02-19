import json, os
p = "d:/develop/TransformerLens-main/tempdata/phase3_wiki_results.json"
if os.path.exists(p):
    d = json.load(open(p))
    print(f"Old: {d}")
    # Update Ep 6
    found=False
    for x in d:
        if x['ep'] == 6:
            x['id'] = 20.9591
            found=True
    if not found:
        d.append({'ep': 6, 'loss': 0.9539, 'id': 20.9591}) # Estimate loss from log
    print(f"New: {d}")
    json.dump(d, open(p, 'w'))
print("Done")
