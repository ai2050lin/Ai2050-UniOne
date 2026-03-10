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
import samplePayload from './data/real_multistep_agi_closure_length_scan_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && typeof payload.systems === 'object');
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`;
}

export default function RealMultistepLengthScanDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const plain = payload?.systems?.plain_local?.global_summary || {};
  const trace = payload?.systems?.trace_gated_local?.global_summary || {};
  const improvements = payload?.improvements_by_length || {};
  const compare = payload?.global_comparison || {};
  const hypotheses = payload?.hypotheses || {};

  const curveRows = useMemo(() => {
    const lengths = Array.isArray(trace?.lengths) ? trace.lengths : [];
    return lengths.map((length, idx) => ({
      length: `L=${length}`,
      plain_closure: Number(plain?.closure_curve?.[idx] || 0),
      trace_closure: Number(trace?.closure_curve?.[idx] || 0),
      plain_retention: Number(plain?.retention_curve?.[idx] || 0),
      trace_retention: Number(trace?.retention_curve?.[idx] || 0),
    }));
  }, [plain, trace]);

  const gainRows = useMemo(
    () =>
      Object.entries(improvements).map(([length, row]) => ({
        length: `L=${length}`,
        closure_gain: Number(row.real_closure_gain || 0),
        episode_gain: Number(row.episode_success_gain || 0),
        retention_gain: Number(row.retention_gain || 0),
      })),
    [improvements]
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
      setError(`真实多步长度扫描 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(249,115,22,0.12), transparent 28%), radial-gradient(circle at top right, rgba(14,165,233,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(249,115,22,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>真实多步长度扫描</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            扫描 `L=3..6` 的任务长度，直接观察真实闭环分数、保留率，以及 `trace` 在任务变长后还能保留多少优势。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>优势面积</div>
          <div style={{ color: '#fb923c', fontSize: '20px', fontWeight: 'bold' }}>{fmt(compare.trace_advantage_area)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最长长度增益</div>
          <div style={{ color: '#7dd3fc', fontSize: '20px', fontWeight: 'bold' }}>{fmt(compare.final_length_gain)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>trace 闭环掉幅</div>
          <div style={{ color: '#86efac', fontSize: '20px', fontWeight: 'bold' }}>{fmt(compare.trace_closure_relative_drop)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>trace 保留掉幅</div>
          <div style={{ color: '#f87171', fontSize: '20px', fontWeight: 'bold' }}>{fmt(compare.trace_retention_relative_drop)}</div>
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
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 闭环分数与保留率随长度变化</div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={curveRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="plain_closure" name="plain 闭环" stroke="#60a5fa" strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="trace_closure" name="trace 闭环" stroke="#f97316" strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="plain_retention" name="plain 保留" stroke="#94a3b8" strokeWidth={2.0} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="trace_retention" name="trace 保留" stroke="#22c55e" strokeWidth={2.0} dot={{ r: 2.5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 各长度上的净增益</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={gainRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="closure_gain" name="闭环增益" fill="#fb923c" radius={[4, 4, 0, 0]} />
              <Bar dataKey="episode_gain" name="回合成功增益" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="retention_gain" name="保留增益" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
