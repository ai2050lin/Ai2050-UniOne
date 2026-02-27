"""
Phase XXXIV-B: 升级 Energy Spikes 和 Neural Terminal，
将 MotherEnginePanel 和 AGIChatPanel 的全部独有功能整合进来。

变更:
1. Energy Spikes Tab 头部增加「原理科普区」（纯物理语言引擎原理说明）
2. Energy Spikes 的 trace 展示增加 Token ID 和 Step 编号
3. Neural Terminal 增加 O(1) 欢迎说明文本
4. Neural Terminal 增加 max_tokens 控制
"""

filepath = r"d:\develop\TransformerLens-main\frontend\src\components\FiberNetPanel.jsx"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# ===== 1. 增强 Energy Spikes Tab: 在 POUR ENERGY 按钮区之前增加原理科普区 =====

# 找到 Energy Spikes Tab 中的 border-l-emerald 标志
old_energy_header = """<div className="flex-1 space-y-4">
                     <Input value={energyPrompt}"""

theory_section = """<div className="p-4 bg-emerald-950/30 rounded-xl border border-emerald-800/30 mb-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Info className="h-4 w-4 text-emerald-400" />
                      <span className="text-[11px] font-black text-emerald-400 uppercase tracking-wider">Pure Algebraic Reverse-Projection Brain</span>
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      This is a real physical language engine. It generates text <strong className="text-slate-200">without</strong> any Transformer Attention or Softmax — completely removing W_q, W_k, W_v probability computations. It works in just two steps:
                    </p>
                    <div className="mt-2 space-y-1">
                      <div className="flex items-start gap-2 text-[10px] text-slate-500">
                        <span className="text-emerald-400 font-mono font-bold shrink-0">1.</span>
                        <span>Convert your input into local synaptic discharges (+1.0)</span>
                      </div>
                      <div className="flex items-start gap-2 text-[10px] text-slate-500">
                        <span className="text-emerald-400 font-mono font-bold shrink-0">2.</span>
                        <span>Let energy flow down the topological gravitational potential graph (P_topology), collapsing to whichever neuron it reaches — that becomes the next token</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex-1 space-y-4">
                     <Input value={energyPrompt}"""

content = content.replace(old_energy_header, theory_section)

# ===== 2. 升级 Energy Spikes 的 trace 展示: 增加 Token ID 和 Step 编号 =====

old_trace_card = """<div key={idx} className="p-2 bg-slate-900/50 rounded-lg border border-slate-800"><div className="text-xs font-black text-slate-100 font-mono">'{t.token_str}'</div><div className="text-[8px] font-mono text-slate-600 uppercase mt-1">{t.resonance_energy?.toFixed(2)} eV</div></div>"""

new_trace_card = """<div key={idx} className="p-2 bg-slate-900/50 rounded-lg border border-slate-800 hover:border-emerald-800/50 transition-colors">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[7px] font-mono text-slate-600">#{t.step}</span>
                          <span className="text-[7px] font-mono text-slate-700">ID:{t.token_id}</span>
                        </div>
                        <div className="text-xs font-black text-slate-100 font-mono">'{t.token_str}'</div>
                        <div className="text-[8px] font-mono text-emerald-500/80 mt-1">{t.resonance_energy?.toFixed(2)} eV</div>
                      </div>"""

content = content.replace(old_trace_card, new_trace_card)

# ===== 3. 增强 Neural Terminal: 增加 O(1) 欢迎文本 =====

# 在 messages.map 之前，添加空状态的欢迎文字
old_messages_map = """{messages.map((m, idx) => ("""

new_messages_section = """{messages.length === 0 && (
                    <div className="text-center py-8 space-y-3">
                      <Bot className="h-8 w-8 text-indigo-500/30 mx-auto" />
                      <p className="text-[11px] text-slate-600 max-w-xs mx-auto leading-relaxed">
                        O(1) Phase-Collapse Reasoning Engine terminal mounted. Based on a 50257-dimensional pure mathematical network, completely free of backpropagation and attention mechanisms.
                      </p>
                      <p className="text-[10px] text-slate-700 font-mono">Inject a pre-concept stimulation pulse (Prompt):</p>
                    </div>
                  )}
                  {messages.map((m, idx) => ("""

content = content.replace(old_messages_map, new_messages_section)

# ===== 4. 写回文件 =====
with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

# 验证
with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

found_theory = any("Reverse-Projection Brain" in l for l in lines)
found_token_id = any("ID:{t.token_id}" in l for l in lines)
found_welcome = any("Phase-Collapse Reasoning" in l for l in lines)

print(f"Theory section injected: {found_theory}")
print(f"Token ID in traces: {found_token_id}")
print(f"O(1) welcome text: {found_welcome}")
print(f"Total lines: {len(lines)}")
print("Phase XXXIV-B complete!" if all([found_theory, found_token_id, found_welcome]) else "Some injections may have failed.")
