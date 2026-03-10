import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/real_multistep_agi_closure_memory_boost_scan_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && typeof payload.systems === 'object');
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`;
}

export default function RealMultistepMemoryBoostDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const plain = payload?.systems?.plain_local?.global_summary || {};
  const trace = payload?.systems?.trace_gated_local?.global_summary || {};
  const anchor = payload?.systems?.trace_anchor_local?.global_summary || {};
  const gains = payload?.anchor_vs_trace_by_length || {};
  const compare = payload?.global_comparison || {};
  const hypotheses = payload?.hypotheses || {};

  const curveRows = useMemo(() => {
    const lengths = Array.isArray(anchor?.lengths) ? anchor.lengths : [];
    return lengths.map((length, idx) => ({
      length: `L=${length}`,
      plain: Number(plain?.closure_curve?.[idx] || 0),
      trace: Number(trace?.closure_curve?.[idx] || 0),
      anchor: Number(anchor?.closure_curve?.[idx] || 0),
      anchor_retention: Number(anchor?.retention_curve?.[idx] || 0),
    }));
  }, [plain, trace, anchor]);

  const gainRows = useMemo(
    () =>
      Object.entries(gains).map(([length, row]) => ({
        length: `L=${length}`,
        closure_gain_vs_trace: Number(row.closure_gain_vs_trace || 0),
        retention_gain_vs_trace: Number(row.retention_gain_vs_trace || 0),
        closure_gain_vs_plain: Number(row.closure_gain_vs_plain || 0),
      })),
    [gains]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 systems 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`长程增强机制 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
    setSource('内置样例');
    setError('');
  }

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(244,63,94,0.12), transparent 28%), radial-gradient(circle at top right, rgba(14,165,233,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(244,63,94,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>长程增强机制扫描</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            对比 `plain_local / trace_gated_local / trace_anchor_local`，观察加入慢记忆锚点后，是否能进一步压平 `L=3..12` 的长程衰减。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <label style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px', cursor: 'pointer' }}>
            导入 JSON
            <input type="file" accept="application/json" onChange={onUpload} style={{ display: 'none' }} />
          </label>
          <button
            type="button"
            onClick={resetAll}
            style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px', background: 'transparent', cursor: 'pointer' }}
          >
            重置
          </button>
        </div>
      </div>

      <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px' }}>{`数据源: ${source}`}</div>
      {error && <div style={{ marginTop: '8px', color: '#fca5a5', fontSize: '11px' }}>{error}</div>}

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>anchor 对 trace 优势面积</div>
          <div style={{ color: '#fb7185', fontSize: '20px', fontWeight: 'bold' }}>{fmt(compare.anchor_advantage_area_over_trace)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最长长度闭环增益</div>
          <div style={{ color: '#7dd3fc', fontSize: '20px', fontWeight: 'bold' }}>{fmt(compare.anchor_final_length_gain_vs_trace)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>anchor 闭环掉幅</div>
          <div style={{ color: '#86efac', fontSize: '20px', fontWeight: 'bold' }}>{fmt(anchor.closure_relative_drop)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>anchor 保留掉幅</div>
          <div style={{ color: '#f59e0b', fontSize: '20px', fontWeight: 'bold' }}>{fmt(anchor.retention_relative_drop)}</div>
        </div>
      </div>

      <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {Object.entries(hypotheses).map(([key, value]) => (
          <div
            key={key}
            style={{
              borderRadius: '999px',
              padding: '6px 10px',
              fontSize: '11px',
              color: value ? '#dcfce7' : '#fee2e2',
              background: value ? 'rgba(34,197,94,0.18)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(34,197,94,0.3)' : 'rgba(248,113,113,0.3)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '不成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1.1fr) minmax(320px, 0.9fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三种机制的真实闭环曲线</div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={curveRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="plain" name="plain" stroke="#94a3b8" strokeWidth={2.0} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="trace" name="trace" stroke="#38bdf8" strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="anchor" name="trace+anchor" stroke="#f43f5e" strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="anchor_retention" name="anchor retention" stroke="#22c55e" strokeWidth={2.0} dot={{ r: 2.5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. anchor 的净增益</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={gainRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="closure_gain_vs_trace" name="对 trace 闭环增益" fill="#f43f5e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="retention_gain_vs_trace" name="对 trace 保留增益" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="closure_gain_vs_plain" name="对 plain 闭环增益" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
