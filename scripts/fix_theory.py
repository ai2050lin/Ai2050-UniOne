"""修复: 在 Energy Spikes Tab 的 flex-1 space-y-4 之前插入原理科普区"""

filepath = r"d:\develop\TransformerLens-main\frontend\src\components\FiberNetPanel.jsx"

with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 找到 Energy Spikes Tab 内的 "flex-1 space-y-4" 行 (line ~331)
theory_block = '''                  <div className="p-4 bg-emerald-950/30 rounded-xl border border-emerald-800/30 mb-5">
                    <div className="flex items-center gap-2 mb-2">
                      <Info className="h-4 w-4 text-emerald-400" />
                      <span className="text-[11px] font-black text-emerald-400 uppercase tracking-wider">Pure Algebraic Reverse-Projection Brain</span>
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      This is a real physical language engine. It generates text <strong className="text-slate-200">without</strong> any Transformer Attention or Softmax. No W_q, W_k, W_v probability computations at all.
                    </p>
                    <div className="mt-2 space-y-1 text-[10px] text-slate-500">
                      <div className="flex gap-2"><span className="text-emerald-400 font-mono font-bold">1.</span> Convert input into local synaptic discharge (+1.0)</div>
                      <div className="flex gap-2"><span className="text-emerald-400 font-mono font-bold">2.</span> Energy flows down the topological gravity graph (P_topology), collapsing to the next token</div>
                    </div>
                  </div>
'''

inserted = False
for i, line in enumerate(lines):
    # 找到 Energy Spikes Tab 区域内的 "flex-1 space-y-4"
    if 'flex-1 space-y-4' in line and not inserted:
        # 确认这是 Energy Spikes Tab (前面应该有 border-l-emerald)
        context = ''.join(lines[max(0,i-5):i])
        if 'emerald' in context:
            # 在这行之前插入原理区
            lines.insert(i, theory_block)
            inserted = True
            print(f"Theory section inserted before line {i+1}")
            break

if not inserted:
    print("WARNING: Could not find insertion point for theory section")
else:
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    # 验证
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"Theory section found: {'Reverse-Projection Brain' in content}")
    print(f"Total lines: {len(content.splitlines())}")

print("Done.")
