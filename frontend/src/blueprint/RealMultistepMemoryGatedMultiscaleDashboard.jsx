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
import samplePayload from './data/real_multistep_memory_gated_multiscale_scan_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && typeof payload.systems === 'object');
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`;
}

function labelOf(name) {
  if (name === 'trace_gated_local') return 'trace';
  if (name === 'single_anchor_beta_086') return '单锚点 β=0.86';
  if (name === 'dual_anchor_beta_050_086') return '双锚点';
  if (name === 'gated_dual_anchor_beta_050_086') return '门控双锚点';
  if (name === 'gated_triple_anchor_beta_050_080_092') return '门控三锚点';
  return name;
}

export default function RealMultistepMemoryGatedMultiscaleDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const systems = payload?.systems || {};
  const ranking = Array.isArray(payload?.ranking) ? payload.ranking : [];
  const best = payload?.best_systems || {};
  const hypotheses = payload?.hypotheses || {};
  const trace = systems?.trace_gated_local?.global_summary || {};
  const single = systems?.single_anchor_beta_086?.global_summary || {};
  const gatedDual = systems?.gated_dual_anchor_beta_050_086?.global_summary || {};
  const gatedTriple = systems?.gated_triple_anchor_beta_050_080_092?.global_summary || {};
  const gains = payload?.gains || {};

  const closureRows = useMemo(() => {
    const lengths = Array.isArray(trace?.lengths) ? trace.lengths : [];
    return lengths.map((length, idx) => ({
      length: `L=${length}`,
      trace: Number(trace?.closure_curve?.[idx] || 0),
      single: Number(single?.closure_curve?.[idx] || 0),
      gated_dual: Number(gatedDual?.closure_curve?.[idx] || 0),
      gated_triple: Number(gatedTriple?.closure_curve?.[idx] || 0),
    }));
  }, [trace, single, gatedDual, gatedTriple]);

  const gateRows = useMemo(() => {
    const lengths = Array.isArray(trace?.lengths) ? trace.lengths : [];
    return lengths.map((length, idx) => ({
      length: `L=${length}`,
      gated_dual_entropy: Number(gatedDual?.gate_entropy_curve?.[idx] || 0),
      gated_dual_peak: Number(gatedDual?.gate_peak_curve?.[idx] || 0),
      gated_triple_entropy: Number(gatedTriple?.gate_entropy_curve?.[idx] || 0),
      gated_triple_peak: Number(gatedTriple?.gate_peak_curve?.[idx] || 0),
    }));
  }, [trace, gatedDual, gatedTriple]);

  const rankingRows = useMemo(
    () =>
      ranking.map((row) => ({
        system: labelOf(row.system),
        mean_closure: Number(row.mean_closure_score || 0),
        max_length: Number(row.max_length_score || 0),
      })),
    [ranking]
  );

  const gainRows = useMemo(
    () =>
      Object.entries(gains?.gated_triple_vs_single?.per_length || {}).map(([length, row]) => ({
        length: `L=${length}`,
        closure_gain: Number(row.closure_gain || 0),
        retention_gain: Number(row.retention_gain || 0),
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
      setError(`门控多时间常数 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(56,189,248,0.12), transparent 28%), radial-gradient(circle at top right, rgba(244,114,182,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(56,189,248,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>门控多时间常数读出</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            在多时间常数记忆簇上加入上下文门控，观察它能否把保留优势转成更高的长程闭环，并检查门控是否真的在选择不同时间尺度。
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均闭环最优</div>
          <div style={{ color: '#22c55e', fontSize: '17px', fontWeight: 'bold' }}>{labelOf(best?.best_mean_closure?.system)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最长任务最优</div>
          <div style={{ color: '#60a5fa', fontSize: '17px', fontWeight: 'bold' }}>{labelOf(best?.best_max_length?.system)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最佳门控系统</div>
          <div style={{ color: '#f59e0b', fontSize: '17px', fontWeight: 'bold' }}>{labelOf(best?.best_gated_max_length?.system)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>门控选择性最高</div>
          <div style={{ color: '#f472b6', fontSize: '17px', fontWeight: 'bold' }}>{labelOf(best?.best_gate_selectivity?.system)}</div>
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 闭环曲线对比</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={closureRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="trace" name="trace" stroke="#94a3b8" strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="single" name="单锚点 β=0.86" stroke="#22c55e" strokeWidth={2.1} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="gated_dual" name="门控双锚点" stroke="#38bdf8" strokeWidth={2.1} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="gated_triple" name="门控三锚点" stroke="#f472b6" strokeWidth={2.1} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 门控统计</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={gateRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="gated_dual_entropy" name="双锚点 entropy" stroke="#38bdf8" strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="gated_dual_peak" name="双锚点 peak" stroke="#0ea5e9" strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="gated_triple_entropy" name="三锚点 entropy" stroke="#f472b6" strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="gated_triple_peak" name="三锚点 peak" stroke="#ec4899" strokeWidth={2} dot={{ r: 2.5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 平均闭环与最长任务分数</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={rankingRows} margin={{ top: 10, right: 12, left: 0, bottom: 30 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="system" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-15} textAnchor="end" height={54} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="mean_closure" name="平均闭环" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="max_length" name="最长任务分数" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 门控三锚点相对单锚点增益</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={gainRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="closure_gain" name="闭环增益" fill="#f472b6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="retention_gain" name="保留增益" fill="#60a5fa" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
