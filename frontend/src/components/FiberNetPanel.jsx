import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Activity, Brain, Bot, ChevronRight, Cpu, Heart, Info, Layers, Loader2,
  MessageSquare, Microscope, Moon, Play, RefreshCw, Send, ThumbsUp,
  ThumbsDown, TrendingDown, User, Waves, Zap
} from 'lucide-react';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

// ======== 样式常量 ========
const S = {
  panel: { padding: 0, color: '#eee', fontFamily: "'Inter', 'Segoe UI', sans-serif", fontSize: 13, lineHeight: 1.5 },
  tabBar: { display: 'flex', gap: 2, padding: 4, background: 'rgba(0,0,0,0.3)', borderRadius: 8, flexWrap: 'wrap' },
  tabBtn: (active) => ({
    padding: '6px 14px', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 11, fontWeight: 700,
    letterSpacing: '0.05em', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 6,
    background: active ? '#4f46e5' : 'transparent', color: active ? '#fff' : '#888',
    transition: 'all 0.2s',
  }),
  card: { background: 'rgba(15,15,25,0.9)', border: '1px solid #333', borderRadius: 10, padding: 16, marginBottom: 12 },
  cardTitle: { fontSize: 10, fontWeight: 800, letterSpacing: '0.15em', textTransform: 'uppercase', color: '#888', marginBottom: 10 },
  btn: (color = '#4f46e5', disabled = false) => ({
    padding: '8px 20px', background: disabled ? '#333' : color, color: '#fff', border: 'none',
    borderRadius: 6, cursor: disabled ? 'not-allowed' : 'pointer', fontWeight: 700, fontSize: 12,
    display: 'flex', alignItems: 'center', gap: 6, opacity: disabled ? 0.6 : 1, transition: 'all 0.2s',
  }),
  input: { flex: 1, padding: '8px 12px', background: '#111', border: '1px solid #444', color: '#fff', borderRadius: 6, outline: 'none', fontSize: 13 },
  metric: (color = '#888') => ({ padding: 12, background: 'rgba(0,0,0,0.3)', borderRadius: 8, border: '1px solid #333' }),
  metricLabel: (color = '#888') => ({ fontSize: 8, fontWeight: 800, letterSpacing: '0.15em', textTransform: 'uppercase', color, marginBottom: 4 }),
  metricValue: { fontSize: 18, fontWeight: 800, fontFamily: 'monospace', color: '#eee' },
  slider: (color = '#4f46e5') => ({ width: '100%', accentColor: color, cursor: 'pointer' }),
  header: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 },
  headerIcon: (color) => ({ background: `${color}22`, padding: 8, borderRadius: 8, display: 'flex' }),
};

const FiberNetPanel = ({ lang = 'zh' }) => {
  const isEn = lang === 'en';
  const [activeTab, setActiveTab] = useState('energy');

  // Energy (Mother Engine) State
  const [energyPrompt, setEnergyPrompt] = useState("The artificial");
  const [energySteps, setEnergySteps] = useState(15);
  const [isEnergyGenerating, setIsEnergyGenerating] = useState(false);
  const [energyResult, setEnergyResult] = useState(null);
  const [renderedTraces, setRenderedTraces] = useState([]);
  const [renderedEnergyText, setRenderedEnergyText] = useState('');

  // Chat State
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [chatStatus, setChatStatus] = useState({ is_ready: false, status_msg: 'Checking...' });

  // Evolution State
  const [evoStatus, setEvoStatus] = useState(null);
  const [isSleeping, setIsSleeping] = useState(false);
  const [sleepIterations, setSleepIterations] = useState(20);

  // AGI Core State
  const [agiCoreData, setAgiCoreData] = useState(null);
  const [isRunningCycle, setIsRunningCycle] = useState(false);

  // DNN Probe State
  const [probeLayer, setProbeLayer] = useState(0);
  const [probeData, setProbeData] = useState(null);
  const [isProbing, setIsProbing] = useState(false);

  // Theory Flow State
  const [inputText, setInputText] = useState('I love her');
  const [tfResult, setTfResult] = useState(null);
  const [tfLoading, setTfLoading] = useState(false);

  const animRef = useRef(null);
  const messagesEndRef = useRef(null);

  // ======== Chat Status Polling ========
  useEffect(() => {
    const check = async () => {
      try { const r = await axios.get(`${API_BASE}/api/agi_chat/status`); setChatStatus(r.data); }
      catch { setChatStatus({ is_ready: false, status_msg: 'Engine Offline' }); }
    };
    check();
    const iv = setInterval(check, 5000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, isTyping]);
  useEffect(() => { return () => { if (animRef.current) clearTimeout(animRef.current); }; }, []);

  // ======== Energy Generate ========
  const handleEnergyGenerate = async () => {
    if (!energyPrompt.trim() || isEnergyGenerating) return;
    setIsEnergyGenerating(true); setEnergyResult(null); setRenderedTraces([]); setRenderedEnergyText('');
    try {
      const r = await axios.post(`${API_BASE}/api/mother-engine/generate`, { prompt: energyPrompt, steps: energySteps });
      if (r.data?.status === 'success') { setEnergyResult(r.data); animateEnergy(r.data.traces); }
    } catch (e) { console.error(e); }
    finally { setIsEnergyGenerating(false); }
  };

  const animateEnergy = (traces) => {
    if (!traces?.length) return; let p = 0;
    const tick = () => {
      if (p < traces.length) {
        const cur = traces[p];
        if (cur) {
          setRenderedTraces(prev => [...prev, cur]);
          setRenderedEnergyText(prev => prev + (cur.token_str || ''));
        }
        p++; animRef.current = setTimeout(tick, 150 + Math.random() * 150);
      }
    }; tick();
  };

  // ======== Chat Send ========
  const handleChatSend = async () => {
    if (!chatInput.trim() || !chatStatus.is_ready) return;
    const msg = chatInput.trim(); setChatInput('');
    setMessages(prev => [...prev, { role: 'user', content: msg }]); setIsTyping(true);
    try {
      const r = await axios.post(`${API_BASE}/api/agi_chat/generate`, { prompt: msg, max_tokens: 30 });
      if (r.data?.generated_text) setMessages(prev => [...prev, { role: 'agi', content: r.data.generated_text }]);
    } catch (e) { setMessages(prev => [...prev, { role: 'sys', content: `Error: ${e.message}` }]); }
    finally { setIsTyping(false); }
  };

  const resetChat = async () => { try { await axios.post(`${API_BASE}/api/agi_chat/reset`); setMessages([]); } catch { } };

  // ======== Theory Flow ========
  const handleInference = async () => {
    setTfLoading(true);
    try {
      const r = await axios.post(`${API_BASE}/fibernet/inference`, { text: inputText, lang: 'en' });
      setTfResult(r.data);
    } catch (e) { console.error(e); }
    finally { setTfLoading(false); }
  };

  // ======== TABS CONFIG (DNN Structure Analysis Style) ========
  const tabGroups = [
    {
      id: 'engines', label: isEn ? 'Engines' : '引擎', icon: Zap,
      items: [
        { id: 'energy', label: isEn ? 'Mother Engine' : '母体引擎', icon: Zap, color: '#00ffcc' },
        { id: 'chat', label: isEn ? 'AGI Chat' : 'AGI 对话', icon: Bot, color: '#4488ff' },
        { id: 'observer', label: isEn ? 'Theory Flow' : '理论推演', icon: Waves, color: '#8b5cf6' },
      ],
    },
    {
      id: 'advanced', label: isEn ? 'Advanced' : '高级', icon: Brain,
      items: [
        { id: 'evolution', label: isEn ? 'Evolution' : '自演化', icon: Moon, color: '#a855f7' },
        { id: 'agi_core', label: isEn ? 'AGI Core' : 'AGI 核心', icon: Brain, color: '#f59e0b' },
        { id: 'dnn_probe', label: isEn ? 'DNN Probe' : 'DNN 探针', icon: Microscope, color: '#0ea5e9' },
      ],
    },
  ];

  return (
    <div style={S.panel}>
      {/* Header */}
      <div style={{ padding: '16px 20px', borderBottom: '1px solid #333', background: 'rgba(0,0,0,0.2)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
          <div style={S.headerIcon('#4f46e5')}><Cpu size={20} color="#818cf8" /></div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 900, letterSpacing: '0.02em' }}>{isEn ? 'FiberNet Integrated Lab' : 'FiberNet 统一实验室'}</div>
            <div style={{ fontSize: 9, color: '#666', fontWeight: 700, letterSpacing: '0.15em', textTransform: 'uppercase' }}>{isEn ? 'Phase XXXIV: Unified AGI Workbench' : '第三十四阶段：统一 AGI 工作台'}</div>
          </div>
        </div>
        <div style={{ padding: '12px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
          {tabGroups.map(group => (
            <div key={group.id} style={{ marginBottom: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, padding: '4px 8px', background: 'rgba(255,255,255,0.03)', borderRadius: 4 }}>
                <group.icon size={14} color="#888" />
                <span style={{ fontSize: 11, fontWeight: 600, color: '#888' }}>{group.label}</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 4 }}>
                {group.items.map(tab => (
                  <button key={tab.id} onClick={() => setActiveTab(tab.id)} title={tab.label} style={{
                    padding: '8px 4px', backgroundColor: activeTab === tab.id ? 'rgba(68, 136, 255, 0.2)' : 'transparent',
                    color: activeTab === tab.id ? '#4488ff' : '#666',
                    border: activeTab === tab.id ? '1px solid rgba(68, 136, 255, 0.4)' : '1px solid transparent',
                    borderRadius: 6, cursor: 'pointer', fontSize: 11, fontWeight: 500,
                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
                    transition: 'all 0.2s',
                  }}>
                    <tab.icon size={14} />
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ padding: 16 }}>

        {/* ==================== TAB: Mother Engine ==================== */}
        {activeTab === 'energy' && (
          <div>
            {/* 理论说明区 */}
            <div style={{ ...S.card, borderLeft: '3px solid #00ffcc', marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                <Info size={14} color="#00ffcc" />
                <span style={{ fontSize: 12, fontWeight: 800, color: '#00ffcc' }}>{isEn ? 'Pure Algebraic Reverse-Projection Brain' : '纯代数·反向投影大脑'}</span>
              </div>
              <p style={{ fontSize: 11, color: '#aaa', lineHeight: 1.6, margin: 0 }}>
                {isEn ? (<>This is a real physical language engine. It generates text <strong style={{ color: '#eee' }}>without</strong> any Transformer Attention or Softmax — no W_q, W_k, W_v at all.<br /><br />1. Convert input into local synaptic discharge (+1.0)<br />2. Energy flows down the topological gravity graph (P_topology), collapsing to whichever neuron — that becomes the next token</>) : (<>这是一个真正的物理语言引擎。它生成文本时<strong style={{ color: '#eee' }}>没有</strong>调用任何 Transformer Attention 或 Softmax，完全去除了 W_q, W_k, W_v 的概率计算。<br /><br />1. 将您的输入转化为局部突触放电 (+1.0)<br />2. 根据拓扑引力势能图谱 (P_topology)，让能量自然滑落坍塌到哪个神经元就蹦出哪个词</>)}
              </p>
              {energyResult?.physics_details && (
                <div style={{ display: 'flex', gap: 10, marginTop: 12 }}>
                  <div style={S.metric()}><div style={S.metricLabel('#88aaff')}>{isEn ? 'Vocab Receptor' : '词表受体'}</div><div style={S.metricValue}>{energyResult.physics_details.vocab} {isEn ? 'D' : '维'}</div></div>
                  <div style={S.metric()}><div style={S.metricLabel('#ff88aa')}>{isEn ? 'Gravity Latent Space' : '引力潜空间'}</div><div style={S.metricValue}>{energyResult.physics_details.represent_dim} {isEn ? 'D' : '维'}</div></div>
                </div>
              )}
            </div>

            {/* 输入与控制 */}
            <div style={S.card}>
              <div style={{ fontSize: 11, color: '#888', marginBottom: 8 }}>{isEn ? 'Prompt Trigger' : '初始点火信标'}</div>
              <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                <input style={S.input} value={energyPrompt} onChange={e => setEnergyPrompt(e.target.value)} disabled={isEnergyGenerating} placeholder={isEn ? 'Enter a word or two to ignite...' : '提供一两个词激发势能...'} />
                <button style={S.btn('#00886644', isEnergyGenerating)} onClick={handleEnergyGenerate} disabled={isEnergyGenerating}>
                  {isEnergyGenerating ? <><Activity size={14} className="animate-spin" /> {isEn ? 'Collapsing...' : '坍塌中...'}</> : <><Play size={14} /> {isEn ? 'Pour Energy' : '倾泻能量'}</>}
                </button>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ fontSize: 11, color: '#666' }}>{isEn ? 'Steps' : '推演级数'}: {energySteps}</span>
                <input type="range" min="1" max="50" value={energySteps} onChange={e => setEnergySteps(parseInt(e.target.value))} style={S.slider('#00ffcc')} />
              </div>
            </div>

            {/* 能量瀑布 */}
            {(energyResult || renderedTraces.length > 0) && (
              <div style={{ ...S.card, padding: 0, overflow: 'hidden' }}>
                <div style={{ padding: '8px 12px', background: '#151520', borderBottom: '1px solid #222', fontSize: 12, color: '#88aaff', display: 'flex', justifyContent: 'space-between' }}>
                  <span>{isEn ? 'High-Dim Energy Mapping Trace (O(1) Matrix Collapse)' : '高纬能量映射轨迹 (O(1) Matrix Collapse)'}</span>
                  <span>{renderedTraces.length}/{energySteps} {isEn ? 'steps' : '步'}</span>
                </div>
                <div style={{ padding: 16 }}>
                  <div style={{ fontSize: 16, fontFamily: 'monospace', lineHeight: 1.6, marginBottom: 16 }}>
                    <span style={{ color: '#00ffcc', fontWeight: 700 }}>{energyPrompt}</span>
                    {renderedEnergyText && <span style={{ color: '#fff', marginLeft: 4 }}>{renderedEnergyText}</span>}
                    {isEnergyGenerating && <span style={{ color: '#00ffcc', marginLeft: 4, animation: 'pulse 1s infinite' }}>_</span>}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {renderedTraces.map((t, idx) => (
                      <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, padding: '4px 8px', background: 'rgba(0,200,180,0.08)', borderRadius: 4, borderLeft: '2px solid #55aaff' }}>
                        <span style={{ color: '#666', width: 24 }}>#{t.step}</span>
                        <ChevronRight size={10} color="#55aaff" />
                        <span style={{ color: '#55aaff', width: 60 }}>ID: {t.token_id}</span>
                        <span style={{ flex: 1, color: '#fff', fontWeight: 700 }}>'{t.token_str}'</span>
                        <span style={{ color: '#ff88aa' }}>{t.resonance_energy?.toFixed(2)} eV</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ==================== TAB: AGI Chat ==================== */}
        {activeTab === 'chat' && (
          <div style={{ ...S.card, padding: 0, display: 'flex', flexDirection: 'column', height: 500, overflow: 'hidden' }}>
            <div style={{ padding: '8px 12px', background: 'rgba(0,0,0,0.3)', borderBottom: '1px solid #333', fontSize: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: chatStatus.is_ready ? '#10b981' : '#f59e0b' }}>
                <Activity size={14} /> {chatStatus.status_msg}
              </div>
              <button onClick={resetChat} style={{ background: 'transparent', border: 'none', color: '#888', cursor: 'pointer' }} title={isEn ? 'Reset' : '重置'}><RefreshCw size={14} /></button>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
              {messages.length === 0 && (
                <div style={{ textAlign: 'center', color: '#666', marginTop: 40, fontSize: 12 }}>
                  <Bot size={32} color="#4488ff33" style={{ margin: '0 auto 12px' }} />
                  <p>{isEn ? 'O(1) phase-collapse reasoning engine terminal mounted.' : 'O(1) 相变坍塌推理引擎交互终端已挂载。'}</p>
                  <p style={{ fontSize: 11, color: '#555' }}>{isEn ? 'Based on a 50257-D pure mathematical network, free of backpropagation and attention.' : '基于 50257 维纯数学网络，完全脱离反向传播与注意力机制。'}</p>
                </div>
              )}
              {messages.map((m, i) => (
                <div key={i} style={{
                  alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '85%',
                  background: m.role === 'user' ? '#4488ff' : (m.role === 'agi' ? 'rgba(255,255,255,0.08)' : 'rgba(255,0,0,0.15)'),
                  padding: '8px 12px', borderRadius: 8, fontSize: 13, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                }}>
                  <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
                    {m.role === 'user' ? <User size={10} /> : <Bot size={10} />}
                    {m.role === 'user' ? (isEn ? 'You' : '你') : (m.role === 'agi' ? 'AGI (O(1))' : (isEn ? 'System' : '系统'))}
                  </div>
                  {m.content}
                </div>
              ))}
              {isTyping && <div style={{ color: '#888', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}><Loader2 size={12} className="animate-spin" /> {isEn ? 'Collapsing Manifold...' : '流形坍塌推演中...'}</div>}
              <div ref={messagesEndRef} />
            </div>
            <div style={{ padding: 12, borderTop: '1px solid #333', display: 'flex', gap: 8 }}>
              <input style={S.input} value={chatInput} onChange={e => setChatInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleChatSend()} placeholder={chatStatus.is_ready ? (isEn ? 'Inject stimulation pulse...' : '注入刺激脉冲...') : (isEn ? 'Waiting...' : '等待连接...')} disabled={!chatStatus.is_ready || isTyping} />
              <button onClick={handleChatSend} disabled={!chatStatus.is_ready || isTyping || !chatInput.trim()} style={S.btn('#4488ff', !chatStatus.is_ready || isTyping)}><Send size={14} /></button>
            </div>
          </div>
        )}

        {/* ==================== TAB: Theory Flow ==================== */}
        {activeTab === 'observer' && (
          <div>
            <div style={S.card}>
              <div style={S.cardTitle}>{isEn ? 'FiberNet Inference' : 'FiberNet 推理'}</div>
              <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                <input style={S.input} value={inputText} onChange={e => setInputText(e.target.value)} placeholder={isEn ? 'Enter text...' : '输入文本...'} />
                <button style={S.btn('#6366f1', tfLoading)} onClick={handleInference} disabled={tfLoading}>
                  {tfLoading ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />} {isEn ? 'Infer' : '推理'}
                </button>
              </div>
              {tfResult && (
                <div>
                  <div style={{ fontSize: 11, color: '#888', marginBottom: 6 }}>{isEn ? 'Tokens' : '词元'}: {tfResult.tokens?.length}</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {tfResult.tokens?.map((t, i) => <span key={i} style={{ padding: '3px 8px', background: '#222', borderRadius: 4, fontSize: 12, color: '#ccc' }}>{t}</span>)}
                    <span style={{ padding: '3px 8px', background: '#4f46e522', borderRadius: 4, fontSize: 12, color: '#818cf8', fontWeight: 700, border: '1px solid #4f46e544' }}>→ {tfResult.next_token}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ==================== TAB: Evolution ==================== */}
        {activeTab === 'evolution' && (
          <div>
            <div style={{ ...S.card, borderLeft: '3px solid #a855f7' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}><Moon size={18} color="#a855f7" /> <span style={{ fontSize: 15, fontWeight: 900 }}>{isEn ? 'Ricci Sleep Engine' : 'Ricci 睡眠引擎'}</span></div>
                  <p style={{ fontSize: 11, color: '#888', maxWidth: 420, margin: 0 }}>{isEn ? 'Triggers offline Ricci Flow evolution to smooth logical contradictions in the manifold.' : '触发离线 Ricci 流演化，平滑流形中的逻辑矛盾。'}</p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#111', padding: '4px 10px', borderRadius: 6, border: '1px solid #333' }}>
                    <span style={{ fontSize: 9, fontWeight: 700, color: '#888' }}>{isEn ? 'Steps' : '步数'}</span>
                    <input type="range" min="5" max="50" value={sleepIterations} onChange={e => setSleepIterations(parseInt(e.target.value))} style={{ ...S.slider('#a855f7'), width: 80 }} disabled={isSleeping} />
                    <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#a855f7', width: 20 }}>{sleepIterations}</span>
                  </div>
                  <button style={S.btn('#7c3aed', isSleeping)} disabled={isSleeping} onClick={async () => {
                    setIsSleeping(true);
                    try { const r = await axios.post(`${API_BASE}/nfb/evolution/ricci?iterations=${sleepIterations}&mode=adaptive`); setEvoStatus(r.data); }
                    catch (e) { console.error(e); } finally { setIsSleeping(false); }
                  }}>
                    {isSleeping ? <><Loader2 size={14} className="animate-spin" /> {isEn ? 'Sleeping...' : '睡眠中...'}</> : <><Moon size={14} /> {isEn ? 'Enter Sleep' : '进入睡眠'}</>}
                  </button>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 16 }}>
                {[
                  { label: isEn ? 'Sleep Cycles' : '睡眠周期', value: evoStatus?.total_sleep_cycles ?? 0, color: '#a855f7' },
                  { label: isEn ? 'Pre-Curvature' : '睡前曲率', value: evoStatus?.pre_sleep_curvature?.toFixed(4) ?? '---', color: '#f97316' },
                  { label: isEn ? 'Post-Curvature' : '睡后曲率', value: evoStatus?.post_sleep_curvature?.toFixed(4) ?? '---', color: '#10b981' },
                  { label: isEn ? 'Reduction' : '降幅', value: evoStatus?.curvature_reduction_pct ? `${evoStatus.curvature_reduction_pct}%` : '---', color: '#06b6d4' },
                ].map((m, i) => (
                  <div key={i} style={S.metric()}><div style={S.metricLabel(m.color)}>{m.label}</div><div style={S.metricValue}>{m.value}</div></div>
                ))}
              </div>

              <div style={{ ...S.card, marginBottom: 0 }}>
                <div style={{ ...S.cardTitle, display: 'flex', justifyContent: 'space-between' }}>
                  <span>{isEn ? 'Curvature Convergence' : '曲率收敛'}</span>
                  <button onClick={async () => { try { const r = await axios.get(`${API_BASE}/api/evolution/chart`); setEvoStatus(p => ({ ...p, curvature_history: r.data.history })); } catch { } }} style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: 10 }}><RefreshCw size={10} /> {isEn ? 'Refresh' : '刷新'}</button>
                </div>
                {evoStatus?.curvature_history?.length > 0 ? (
                  <div style={{ display: 'flex', alignItems: 'flex-end', gap: 1, height: 100 }}>
                    {evoStatus.curvature_history.map((c, i) => {
                      const max = Math.max(...evoStatus.curvature_history);
                      return <div key={i} style={{ flex: 1, background: '#a855f766', borderRadius: '3px 3px 0 0', height: `${max > 0 ? (c / max) * 100 : 5}%`, minHeight: 2 }} title={`Step ${i}: ${c.toFixed(4)}`} />;
                    })}
                  </div>
                ) : <div style={{ height: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#444', fontSize: 10, fontWeight: 700 }}>{isEn ? 'No data yet — click "Enter Sleep"' : '暂无数据 — 点击「进入睡眠」开始'}</div>}
              </div>
            </div>
          </div>
        )}

        {/* ==================== TAB: AGI Core ==================== */}
        {activeTab === 'agi_core' && (
          <div>
            <div style={{ ...S.card, borderLeft: '3px solid #f59e0b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}><Brain size={18} color="#f59e0b" /> <span style={{ fontSize: 15, fontWeight: 900 }}>{isEn ? 'AGI Core Engine' : 'AGI 核心引擎'}</span></div>
                  <p style={{ fontSize: 11, color: '#888', maxWidth: 420, margin: 0 }}>{isEn ? 'Algebraic Unified Consciousness Field. Runs a single conscious cycle through GWS competition, emotion homeostasis, and holographic memory consolidation.' : '代数统一意识场。通过 GWS 竞争、情感稳态和全息记忆固化运行单次意识周期。'}</p>
                </div>
                <button onClick={async () => {
                  setIsRunningCycle(true);
                  try { const r = await axios.get(`${API_BASE}/nfb_ra/unified_conscious_field`); setAgiCoreData(r.data); }
                  catch (e) { setAgiCoreData({ error: e.message }); } finally { setIsRunningCycle(false); }
                }} disabled={isRunningCycle} style={S.btn('#d97706', isRunningCycle)}>
                  {isRunningCycle ? <><Loader2 size={14} className="animate-spin" /> {isEn ? 'Running...' : '运行中...'}</> : <><Brain size={14} /> {isEn ? 'Run Cycle' : '运行周期'}</>}
                </button>
              </div>

              {agiCoreData?.error && <div style={{ padding: 12, background: 'rgba(255,50,50,0.1)', border: '1px solid #882222', borderRadius: 8, color: '#ff8888', fontSize: 12, marginBottom: 12 }}>{agiCoreData.error}</div>}

              {agiCoreData?.unified_spectrum && (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 16 }}>
                    {[
                      { label: isEn ? 'GWS Winner' : 'GWS 胜者', value: agiCoreData.unified_spectrum.gws_winner || '---', color: '#f59e0b' },
                      { label: isEn ? 'Signal Norm' : '信号范数', value: agiCoreData.unified_spectrum.signal_norm?.toFixed(3) || '---', color: '#06b6d4' },
                      { label: isEn ? 'Energy Save' : '能量节省', value: agiCoreData.unified_spectrum.energy_saving || '---', color: '#10b981' },
                      { label: isEn ? 'Memory Slots' : '记忆槽位', value: agiCoreData.unified_spectrum.memory_slots ?? '---', color: '#a855f7' },
                    ].map((m, i) => <div key={i} style={S.metric()}><div style={S.metricLabel(m.color)}>{m.label}</div><div style={{ ...S.metricValue, fontSize: 15 }}>{m.value}</div></div>)}
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 16 }}>
                    <div style={S.card}>
                      <div style={{ ...S.cardTitle, display: 'flex', alignItems: 'center', gap: 4 }}><Heart size={10} /> {isEn ? 'Emotion Homeostasis' : '情感稳态'}</div>
                      {agiCoreData.unified_spectrum.emotion && Object.entries(agiCoreData.unified_spectrum.emotion).map(([k, v]) => (
                        <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: '1px solid #222' }}>
                          <span style={{ fontSize: 10, color: '#888', fontFamily: 'monospace', textTransform: 'uppercase' }}>{k}</span>
                          <span style={{ fontSize: 12, fontWeight: 800, color: '#eee', fontFamily: 'monospace' }}>{typeof v === 'number' ? v.toFixed(2) : String(v)}</span>
                        </div>
                      ))}
                    </div>
                    <div style={S.card}>
                      <div style={{ ...S.cardTitle, display: 'flex', alignItems: 'center', gap: 4 }}><Layers size={10} /> {isEn ? 'Active Modules' : '活跃模块'}</div>
                      {(agiCoreData.active_modules || []).map((mod, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0' }}>
                          <div style={{ width: 8, height: 8, borderRadius: '50%', background: mod === agiCoreData.unified_spectrum.gws_winner ? '#f59e0b' : '#444' }} />
                          <span style={{ fontSize: 11, fontFamily: 'monospace', color: '#ccc' }}>{mod}</span>
                          {mod === agiCoreData.unified_spectrum.gws_winner && <span style={{ fontSize: 8, background: '#f59e0b22', color: '#f59e0b', padding: '1px 6px', borderRadius: 4, fontWeight: 800, border: '1px solid #f59e0b44' }}>WINNER</span>}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', padding: 10, background: '#111', borderRadius: 8, border: '1px solid #333' }}>
                    <span style={{ fontSize: 9, fontWeight: 800, color: '#888', letterSpacing: '0.1em' }}>{isEn ? 'RLMF FEEDBACK' : 'RLMF 反馈'}</span>
                    <button onClick={async () => { try { await axios.post(`${API_BASE}/nfb_ra/inject_feedback?rating=1`); } catch { } }} style={{ ...S.btn('#10b981'), padding: '4px 12px', fontSize: 11 }}><ThumbsUp size={12} /> {isEn ? 'Align' : '对齐'}</button>
                    <button onClick={async () => { try { await axios.post(`${API_BASE}/nfb_ra/inject_feedback?rating=-1`); } catch { } }} style={{ ...S.btn('#ef4444'), padding: '4px 12px', fontSize: 11 }}><ThumbsDown size={12} /> {isEn ? 'Correct' : '纠正'}</button>
                    <div style={{ marginLeft: 'auto', width: 10, height: 10, borderRadius: '50%', background: agiCoreData.unified_spectrum.glow_color === 'amber' ? '#f59e0b' : '#818cf8', boxShadow: `0 0 8px ${agiCoreData.unified_spectrum.glow_color === 'amber' ? '#f59e0b' : '#818cf8'}` }} />
                  </div>
                </>
              )}

              {!agiCoreData && !isRunningCycle && <div style={{ height: 120, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#444', fontSize: 10, fontWeight: 700 }}>{isEn ? 'Click "Run Cycle" to trigger a conscious step' : '点击「运行周期」触发一次意识步进'}</div>}
            </div>
          </div>
        )}

        {/* ==================== TAB: DNN Probe ==================== */}
        {activeTab === 'dnn_probe' && (
          <div>
            <div style={{ ...S.card, borderLeft: '3px solid #0ea5e9' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}><Microscope size={18} color="#0ea5e9" /> <span style={{ fontSize: 15, fontWeight: 900 }}>{isEn ? 'DNN Probe' : 'DNN 探针'}</span></div>
                  <p style={{ fontSize: 11, color: '#888', maxWidth: 420, margin: 0 }}>{isEn ? 'Probes the internal activation topology of the loaded Transformer across all residual stream layers.' : '探测已加载 Transformer 所有残差流层级的内部激活拓扑结构。'}</p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#111', padding: '4px 10px', borderRadius: 6, border: '1px solid #333' }}>
                    <span style={{ fontSize: 9, fontWeight: 700, color: '#888' }}>{isEn ? 'Layer' : '层级'}</span>
                    <input type="range" min="0" max="11" value={probeLayer} onChange={e => setProbeLayer(parseInt(e.target.value))} style={{ ...S.slider('#0ea5e9'), width: 80 }} />
                    <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#0ea5e9', width: 16 }}>{probeLayer}</span>
                  </div>
                  <button style={S.btn('#0284c7', isProbing)} disabled={isProbing} onClick={async () => {
                    setIsProbing(true);
                    try { const r = await axios.get(`${API_BASE}/layer_detail/${probeLayer}`); setProbeData(r.data); }
                    catch (e) { setProbeData({ error: e.message }); } finally { setIsProbing(false); }
                  }}>
                    {isProbing ? <><Loader2 size={14} className="animate-spin" /> {isEn ? 'Probing...' : '探测中...'}</> : <><Microscope size={14} /> {isEn ? 'Probe' : '探测'}</>}
                  </button>
                </div>
              </div>

              {probeData?.error && <div style={{ padding: 12, background: 'rgba(255,50,50,0.1)', border: '1px solid #882222', borderRadius: 8, color: '#ff8888', fontSize: 12, marginBottom: 12 }}>{probeData.error}</div>}

              {probeData && !probeData.error && (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 16 }}>
                    {[
                      { label: isEn ? 'Layer' : '层级', value: `L${probeData.layer_idx ?? probeLayer}`, color: '#0ea5e9' },
                      { label: isEn ? 'Attn Heads' : '注意力头', value: probeData.num_heads ?? probeData.attention_heads ?? '---', color: '#a855f7' },
                      { label: isEn ? 'Hidden Dim' : '隐藏维度', value: probeData.hidden_dim ?? probeData.d_model ?? '---', color: '#10b981' },
                      { label: isEn ? 'Residual Norm' : '残差范数', value: typeof probeData.residual_norm === 'number' ? probeData.residual_norm.toFixed(3) : (probeData.mean_activation?.toFixed(3) ?? '---'), color: '#f97316' },
                    ].map((m, i) => <div key={i} style={S.metric()}><div style={S.metricLabel(m.color)}>{m.label}</div><div style={S.metricValue}>{m.value}</div></div>)}
                  </div>

                  {probeData.top_tokens && (
                    <div style={{ ...S.card, marginBottom: 12 }}>
                      <div style={S.cardTitle}>{isEn ? 'Top Activated Tokens' : '最活跃词元'}</div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {probeData.top_tokens.slice(0, 20).map((tok, i) => (
                          <span key={i} style={{ padding: '2px 8px', background: '#0ea5e922', border: '1px solid #0ea5e944', borderRadius: 4, fontSize: 10, fontFamily: 'monospace', color: '#7dd3fc' }}>{typeof tok === 'object' ? tok.token : tok}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  {probeData.attention_pattern && (
                    <div style={S.card}>
                      <div style={S.cardTitle}>{isEn ? 'Attention Heatmap (Head 0)' : '注意力热图 (Head 0)'}</div>
                      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 1, height: 80 }}>
                        {(Array.isArray(probeData.attention_pattern[0]) ? probeData.attention_pattern[0] : probeData.attention_pattern).slice(0, 50).map((v, i) => (
                          <div key={i} style={{ flex: 1, background: '#0ea5e966', borderRadius: '2px 2px 0 0', height: `${Math.max(v * 100, 2)}%`, minHeight: 1 }} title={v.toFixed(4)} />
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}

              {!probeData && !isProbing && <div style={{ height: 120, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#444', fontSize: 10, fontWeight: 700 }}>{isEn ? 'Select a layer and click "Probe" to inspect activations' : '选择层级并点击「探测」查看激活状态'}</div>}
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default FiberNetPanel;
