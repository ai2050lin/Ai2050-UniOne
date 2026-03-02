import React from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Orbit, Atom, Sigma } from 'lucide-react';

const clusterNodes = [
  { id: 'logic', x: 22, y: 38, r: 10, label: 'Logic', color: '#34d399' },
  { id: 'math', x: 40, y: 24, r: 9, label: 'Math', color: '#60a5fa' },
  { id: 'vision', x: 62, y: 34, r: 11, label: 'Vision', color: '#f472b6' },
  { id: 'syntax', x: 34, y: 58, r: 8, label: 'Syntax', color: '#f59e0b' },
  { id: 'memory', x: 60, y: 62, r: 10, label: 'Memory', color: '#a78bfa' },
];

const sparseLaw = [
  { name: '全局激活率', value: '0.8% - 2.4%', hint: '仅少量神经元参与一次推理' },
  { name: '跨簇重叠率', value: '< 0.21', hint: '概念簇之间强隔离，减少干扰' },
  { name: '有效维度占比', value: '6% - 11%', hint: '高维中只用稀疏子空间承载知识' },
];

export default function UniversalManifoldGraph() {
  return (
    <div
      style={{
        marginTop: 16,
        padding: 18,
        borderRadius: 12,
        border: '1px solid rgba(56,189,248,0.32)',
        background: 'linear-gradient(145deg, rgba(12,20,35,0.9), rgba(8,14,26,0.92))',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
        <div style={{ width: 36, height: 36, borderRadius: 10, background: 'rgba(56,189,248,0.16)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Orbit size={18} color="#7dd3fc" />
        </div>
        <div>
          <div style={{ color: '#dbeafe', fontWeight: 700, fontSize: 15 }}>星团宇宙拓扑与全局极稀疏定律</div>
          <div style={{ color: '#7dd3fc', fontSize: 12 }}>Universal Manifold Graph + Global Sparse Law</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 14 }}>
        <div style={{ borderRadius: 10, border: '1px solid rgba(148,163,184,0.2)', background: 'rgba(2,6,23,0.5)', padding: 10 }}>
          <svg viewBox="0 0 100 84" style={{ width: '100%', height: 220 }}>
            <defs>
              <radialGradient id="gA" cx="50%" cy="50%" r="70%">
                <stop offset="0%" stopColor="#1f2937" stopOpacity="0.15" />
                <stop offset="100%" stopColor="#0b1222" stopOpacity="1" />
              </radialGradient>
            </defs>
            <rect x="0" y="0" width="100" height="84" fill="url(#gA)" />
            {clusterNodes.map((n, i) => (
              <motion.g key={n.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.35, delay: i * 0.08 }}>
                <circle cx={n.x} cy={n.y} r={n.r + 5} fill={n.color} opacity={0.08} />
                <circle cx={n.x} cy={n.y} r={n.r} fill={n.color} opacity={0.22} stroke={n.color} strokeWidth="0.6" />
                <text x={n.x} y={n.y + 0.5} textAnchor="middle" fill="#e5e7eb" fontSize="3.2">{n.label}</text>
              </motion.g>
            ))}
            <line x1="22" y1="38" x2="40" y2="24" stroke="#334155" strokeWidth="0.6" />
            <line x1="40" y1="24" x2="62" y2="34" stroke="#334155" strokeWidth="0.6" />
            <line x1="34" y1="58" x2="60" y2="62" stroke="#334155" strokeWidth="0.6" />
            <line x1="22" y1="38" x2="34" y2="58" stroke="#334155" strokeWidth="0.6" />
          </svg>
        </div>

        <div style={{ display: 'grid', gap: 8 }}>
          {sparseLaw.map((m, idx) => (
            <motion.div
              key={m.name}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.28, delay: idx * 0.07 }}
              style={{
                borderRadius: 10,
                border: '1px solid rgba(125,211,252,0.22)',
                background: 'rgba(15,23,42,0.62)',
                padding: 10,
              }}
            >
              <div style={{ color: '#93c5fd', fontSize: 11 }}>{m.name}</div>
              <div style={{ color: '#f8fafc', fontWeight: 700, fontSize: 16, margin: '3px 0' }}>{m.value}</div>
              <div style={{ color: '#94a3b8', fontSize: 11 }}>{m.hint}</div>
            </motion.div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 10, display: 'flex', gap: 12, color: '#cbd5e1', fontSize: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}><Sparkles size={14} color="#22d3ee" />星团拓扑: 概念分簇</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}><Atom size={14} color="#fbbf24" />稀疏编码: 低激活高区分</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}><Sigma size={14} color="#a78bfa" />全局定律: 稀疏+正交</div>
      </div>
    </div>
  );
}

