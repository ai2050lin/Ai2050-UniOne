"""
Phase XXXIV: 一次性注入 AGI Core 和 DNN Probe 两个 Tab
"""
import re

filepath = r"d:\develop\TransformerLens-main\frontend\src\components\FiberNetPanel.jsx"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# ===== 1. 修复导入行：增加 Brain, Microscope, ThumbsUp, ThumbsDown, Layers =====
old_import = "import { Activity, Network, Play, Zap, Send, Bot, User, RefreshCw, Loader2, Cpu, ChevronRight, BarChart2, MessageSquare, Waves, Info, Moon, TrendingDown } from 'lucide-react';"
new_import = "import { Activity, Network, Play, Zap, Send, Bot, User, RefreshCw, Loader2, Cpu, ChevronRight, BarChart2, MessageSquare, Waves, Info, Moon, TrendingDown, Brain, Microscope, ThumbsUp, ThumbsDown, Layers, Heart } from 'lucide-react';"
content = content.replace(old_import, new_import)

# ===== 2. 增加 AGI Core 和 DNN Probe 状态变量 =====
old_state_anchor = "// Evolution State"
new_states = """// AGI Core State
  const [agiCoreData, setAgiCoreData] = useState(null);
  const [isRunningCycle, setIsRunningCycle] = useState(false);

  // DNN Probe State
  const [probeLayer, setProbeLayer] = useState(0);
  const [probeData, setProbeData] = useState(null);
  const [isProbing, setIsProbing] = useState(false);

  // Evolution State"""
content = content.replace(old_state_anchor, new_states)

# ===== 3. Tab 列表增加两个条目 =====
old_tab_entry = "{ id: 'evolution', label: 'Evolution', icon: Moon }"
new_tab_entry = "{ id: 'evolution', label: 'Evolution', icon: Moon },\r\n                 { id: 'agi_core', label: 'AGI Core', icon: Brain },\r\n                 { id: 'dnn_probe', label: 'DNN Probe', icon: Microscope }"
content = content.replace(old_tab_entry, new_tab_entry)

# ===== 4. 在 Evolution Tab 结束后，</CardContent> 之前，插入两个新 Tab 的 UI =====
agi_core_ui = """
          {activeTab === 'agi_core' && (
            <div className="animate-in fade-in slide-in-from-bottom-2 duration-500 space-y-6">
              <div className="bg-slate-950 p-6 rounded-2xl border border-slate-800 border-l-4 border-l-amber-500/50">
                <div className="flex flex-col md:flex-row justify-between items-start gap-6 mb-6">
                  <div className="space-y-2">
                    <div className="flex items-center gap-3">
                      <Brain className="h-5 w-5 text-amber-400" />
                      <span className="text-lg font-black uppercase tracking-wide text-slate-100">AGI Core Engine</span>
                    </div>
                    <p className="text-[11px] text-slate-500 max-w-lg leading-relaxed">Algebraic Unified Consciousness Field. Runs a single conscious cycle through GWS competition, emotion homeostasis, and holographic memory consolidation.</p>
                  </div>
                  <Button
                    onClick={async () => {
                      setIsRunningCycle(true);
                      try {
                        const res = await axios.get(`${API_BASE}/nfb_ra/unified_conscious_field`);
                        setAgiCoreData(res.data);
                      } catch(e) { console.error(e); setAgiCoreData({ error: e.message }); }
                      finally { setIsRunningCycle(false); }
                    }}
                    disabled={isRunningCycle}
                    className="bg-amber-600 hover:bg-amber-700 h-10 px-6 font-black uppercase tracking-wider shrink-0"
                  >
                    {isRunningCycle ? <><Loader2 className="animate-spin mr-2 h-4 w-4" />Running...</> : <><Brain className="mr-2 h-4 w-4" />Run Cycle</>}
                  </Button>
                </div>

                {agiCoreData?.error && (
                  <div className="p-4 bg-red-950/30 border border-red-800/50 rounded-xl text-red-400 text-xs font-mono mb-4">{agiCoreData.error}</div>
                )}

                {agiCoreData?.unified_spectrum && (
                  <>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                      {[
                        { label: 'GWS Winner', value: agiCoreData.unified_spectrum.gws_winner || '---', color: 'amber' },
                        { label: 'Signal Norm', value: agiCoreData.unified_spectrum.signal_norm?.toFixed(3) || '---', color: 'cyan' },
                        { label: 'Energy Save', value: agiCoreData.unified_spectrum.energy_saving || '---', color: 'emerald' },
                        { label: 'Memory Slots', value: agiCoreData.unified_spectrum.memory_slots ?? '---', color: 'violet' },
                      ].map((m, idx) => (
                        <div key={idx} className="p-4 bg-slate-900 rounded-xl border border-slate-800">
                          <div className={`text-[8px] font-black uppercase tracking-widest mb-2 text-${m.color}-400`}>{m.label}</div>
                          <div className="text-lg font-black font-mono text-slate-100 tracking-tighter truncate">{m.value}</div>
                        </div>
                      ))}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                      <div className="p-4 bg-slate-900 rounded-xl border border-slate-800">
                        <div className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-2"><Heart className="h-3 w-3" /> Emotion Homeostasis</div>
                        {agiCoreData.unified_spectrum.emotion && (
                          <div className="space-y-2">
                            {Object.entries(agiCoreData.unified_spectrum.emotion).map(([k, v]) => (
                              <div key={k} className="flex justify-between items-center">
                                <span className="text-[10px] text-slate-400 font-mono uppercase">{k}</span>
                                <span className="text-sm font-black text-slate-200 font-mono">{typeof v === 'number' ? v.toFixed(2) : String(v)}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      <div className="p-4 bg-slate-900 rounded-xl border border-slate-800">
                        <div className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-2"><Layers className="h-3 w-3" /> Active Modules</div>
                        <div className="space-y-2">
                          {(agiCoreData.active_modules || []).map((mod, idx) => (
                            <div key={idx} className="flex items-center gap-2">
                              <div className={`w-2 h-2 rounded-full ${mod === agiCoreData.unified_spectrum.gws_winner ? 'bg-amber-400 animate-pulse' : 'bg-slate-700'}`} />
                              <span className="text-[10px] font-mono text-slate-300">{mod}</span>
                              {mod === agiCoreData.unified_spectrum.gws_winner && <Badge className="text-[7px] bg-amber-600/20 text-amber-400 border-amber-600/30">WINNER</Badge>}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="flex gap-3 items-center p-3 bg-slate-900/50 rounded-xl border border-slate-800">
                      <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">RLMF Feedback</span>
                      <Button size="sm" variant="ghost" className="text-emerald-400 hover:bg-emerald-950"
                        onClick={async () => { try { await axios.post(`${API_BASE}/nfb_ra/inject_feedback?rating=1`); } catch(e){} }}>
                        <ThumbsUp className="h-4 w-4 mr-1" /> Align
                      </Button>
                      <Button size="sm" variant="ghost" className="text-red-400 hover:bg-red-950"
                        onClick={async () => { try { await axios.post(`${API_BASE}/nfb_ra/inject_feedback?rating=-1`); } catch(e){} }}>
                        <ThumbsDown className="h-4 w-4 mr-1" /> Correct
                      </Button>
                      <div className={`ml-auto w-3 h-3 rounded-full ${agiCoreData.unified_spectrum.glow_color === 'amber' ? 'bg-amber-400' : 'bg-indigo-400'} animate-pulse`} />
                    </div>
                  </>
                )}

                {!agiCoreData && !isRunningCycle && (
                  <div className="h-[150px] flex items-center justify-center text-slate-700 text-[10px] font-black uppercase tracking-widest">Click "Run Cycle" to trigger a conscious step</div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'dnn_probe' && (
            <div className="animate-in fade-in slide-in-from-bottom-2 duration-500 space-y-6">
              <div className="bg-slate-950 p-6 rounded-2xl border border-slate-800 border-l-4 border-l-sky-500/50">
                <div className="flex flex-col md:flex-row justify-between items-start gap-6 mb-6">
                  <div className="space-y-2">
                    <div className="flex items-center gap-3">
                      <Microscope className="h-5 w-5 text-sky-400" />
                      <span className="text-lg font-black uppercase tracking-wide text-slate-100">DNN Probe</span>
                    </div>
                    <p className="text-[11px] text-slate-500 max-w-lg leading-relaxed">Physical Large Model Engine. Probes the internal activation topology of the loaded Transformer (GPT-2) across all residual stream layers.</p>
                  </div>
                  <div className="flex items-center gap-4 shrink-0">
                    <div className="flex items-center gap-2 bg-slate-900 p-2 rounded-lg border border-slate-800">
                      <span className="text-[9px] font-black text-slate-500 uppercase">Layer</span>
                      <input type="range" min="0" max="11" value={probeLayer} onChange={e => setProbeLayer(parseInt(e.target.value))} className="w-24 accent-sky-500 cursor-pointer" />
                      <span className="text-xs font-mono text-sky-400 w-4">{probeLayer}</span>
                    </div>
                    <Button
                      onClick={async () => {
                        setIsProbing(true);
                        try {
                          const res = await axios.get(`${API_BASE}/layer_detail/${probeLayer}`);
                          setProbeData(res.data);
                        } catch(e) { console.error(e); setProbeData({ error: e.message }); }
                        finally { setIsProbing(false); }
                      }}
                      disabled={isProbing}
                      className="bg-sky-600 hover:bg-sky-700 h-10 px-6 font-black uppercase tracking-wider"
                    >
                      {isProbing ? <><Loader2 className="animate-spin mr-2 h-4 w-4" />Probing...</> : <><Microscope className="mr-2 h-4 w-4" />Probe</>}
                    </Button>
                  </div>
                </div>

                {probeData?.error && (
                  <div className="p-4 bg-red-950/30 border border-red-800/50 rounded-xl text-red-400 text-xs font-mono mb-4">{probeData.error}</div>
                )}

                {probeData && !probeData.error && (
                  <>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                      {[
                        { label: 'Layer', value: `L${probeData.layer_idx ?? probeLayer}`, color: 'sky' },
                        { label: 'Attn Heads', value: probeData.num_heads ?? probeData.attention_heads ?? '---', color: 'violet' },
                        { label: 'Hidden Dim', value: probeData.hidden_dim ?? probeData.d_model ?? '---', color: 'emerald' },
                        { label: 'Residual Norm', value: typeof probeData.residual_norm === 'number' ? probeData.residual_norm.toFixed(3) : (probeData.mean_activation?.toFixed(3) ?? '---'), color: 'orange' },
                      ].map((m, idx) => (
                        <div key={idx} className="p-4 bg-slate-900 rounded-xl border border-slate-800">
                          <div className={`text-[8px] font-black uppercase tracking-widest mb-2 text-${m.color}-400`}>{m.label}</div>
                          <div className="text-xl font-black font-mono text-slate-100 tracking-tighter">{m.value}</div>
                        </div>
                      ))}
                    </div>

                    {probeData.top_tokens && (
                      <div className="p-4 bg-slate-900 rounded-xl border border-slate-800 mb-4">
                        <div className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-3">Top Activated Tokens</div>
                        <div className="flex flex-wrap gap-2">
                          {probeData.top_tokens.slice(0, 20).map((tok, idx) => (
                            <span key={idx} className="px-2 py-1 bg-sky-950/50 border border-sky-800/30 rounded text-sky-300 text-[10px] font-mono">{typeof tok === 'object' ? tok.token : tok}</span>
                          ))}
                        </div>
                      </div>
                    )}

                    {probeData.attention_pattern && (
                      <div className="p-4 bg-slate-900 rounded-xl border border-slate-800">
                        <div className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-3">Attention Heatmap (Head 0)</div>
                        <div className="flex items-end gap-[1px] h-[80px]">
                          {(Array.isArray(probeData.attention_pattern[0]) ? probeData.attention_pattern[0] : probeData.attention_pattern).slice(0, 50).map((v, idx) => {
                            const h = Math.max(v * 100, 2);
                            return <div key={idx} className="flex-1 bg-sky-500/50 hover:bg-sky-400 rounded-t transition-all" style={{ height: `${h}%` }} title={`${v.toFixed(4)}`} />;
                          })}
                        </div>
                      </div>
                    )}
                  </>
                )}

                {!probeData && !isProbing && (
                  <div className="h-[150px] flex items-center justify-center text-slate-700 text-[10px] font-black uppercase tracking-widest">Select a layer and click "Probe" to inspect activations</div>
                )}
              </div>
            </div>
          )}
"""

# 找到 Evolution Tab 结束后的 </CardContent> 位置
insert_marker = "        </CardContent>"
insert_pos = content.rfind(insert_marker)
if insert_pos < 0:
    print("ERROR: Could not find </CardContent>")
    exit(1)

content = content[:insert_pos] + agi_core_ui + "\n" + insert_marker + content[insert_pos + len(insert_marker):]

# ===== 5. 更新 activeTab 注释 =====
content = content.replace(
    "// observer | energy | chat | evolution",
    "// observer | energy | chat | evolution | agi_core | dnn_probe"
)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("Phase XXXIV: AGI Core + DNN Probe tabs injected successfully!")
print(f"File size: {len(content)} bytes")
