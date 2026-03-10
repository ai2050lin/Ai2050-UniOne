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
import samplePayload from './data/real_multistep_memory_beta_scan_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.betas && typeof payload.betas === 'object');
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`;
}

function betaOptions(payload) {
  return Object.keys(payload?.betas || {});
}

export default function RealMultistepBetaScanDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const betaList = useMemo(() => betaOptions(payload), [payload]);
  const [selectedBeta, setSelectedBeta] = useState(betaList[0] || '0.86');

  const trace = payload?.trace_reference?.global_summary || {};
  const betaRow = payload?.betas?.[selectedBeta] || {};
  const betaSummary = betaRow?.global_summary || {};
  const ranking = Array.isArray(payload?.ranking) ? payload.ranking : [];
  const best = payload?.best_betas || {};
  const hypotheses = payload?.hypotheses || {};

  const curveRows = useMemo(() => {
    const lengths = Array.isArray(trace?.lengths) ? trace.lengths : [];
    return lengths.map((length, idx) => ({
      length: `L=${length}`,
      trace: Number(trace?.closure_curve?.[idx] || 0),
      beta: Number(betaSummary?.closure_curve?.[idx] || 0),
      trace_retention: Number(trace?.retention_curve?.[idx] || 0),
      beta_retention: Number(betaSummary?.retention_curve?.[idx] || 0),
    }));
  }, [trace, betaSummary]);

  const rankingRows = useMemo(
    () =>
      ranking.map((row) => ({
        beta: `β=${Number(row.beta).toFixed(2)}`,
        mean_closure: Number(row.mean_closure_score || 0),
        max_gain: Number(row.max_length_gain_vs_trace || 0),
      })),
    [ranking]
  );

  const gainRows = useMemo(
    () =>
      Object.entries(betaRow?.gains_vs_trace || {}).map(([length, row]) => ({
        length: `L=${length}`,
        closure_gain: Number(row.closure_gain_vs_trace || 0),
        retention_gain: Number(row.retention_gain_vs_trace || 0),
      })),
    [betaRow]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 betas 字段');
      }
      const nextBetas = betaOptions(parsed);
      setPayload(parsed);
      setSelectedBeta(nextBetas[0] || '');
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`慢记忆 beta 扫描 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    const nextBetas = betaOptions(samplePayload);
    setPayload(samplePayload);
    setSelectedBeta(nextBetas[0] || '');
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
          'radial-gradient(circle at top left, rgba(16,185,129,0.12), transparent 28%), radial-gradient(circle at top right, rgba(59,130,246,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(16,185,129,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>慢记忆 beta 扫描</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            在 `trace_anchor_local` 内扫描慢记忆时间常数 `beta`，区分“平均最优 beta”和“最长任务最优 beta”是否一致。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <select
            value={selectedBeta}
            onChange={(event) => setSelectedBeta(event.target.value)}
            style={{ background: 'rgba(15,23,42,0.9)', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px' }}
          >
            {betaList.map((beta) => (
              <option key={beta} value={beta}>
                {`β=${beta}`}
              </option>
            ))}
          </select>
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>当前 beta 平均闭环</div>
          <div style={{ color: '#34d399', fontSize: '20px', fontWeight: 'bold' }}>{fmt(betaSummary.mean_closure_score)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>当前 beta 最长长度增益</div>
          <div style={{ color: '#60a5fa', fontSize: '20px', fontWeight: 'bold' }}>{fmt(betaRow?.global_comparison?.final_length_gain_vs_trace)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均最优 beta</div>
          <div style={{ color: '#fbbf24', fontSize: '20px', fontWeight: 'bold' }}>{`β=${fmt(best?.best_mean_closure?.beta, 2)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最长任务最优 beta</div>
          <div style={{ color: '#f472b6', fontSize: '20px', fontWeight: 'bold' }}>{`β=${fmt(best?.best_max_length?.beta, 2)}`}</div>
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
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 选中 beta 与 trace 的长度曲线对比</div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={curveRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="trace" name="trace 闭环" stroke="#38bdf8" strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="beta" name={`β=${selectedBeta} 闭环`} stroke="#10b981" strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="trace_retention" name="trace 保留" stroke="#94a3b8" strokeWidth={2.0} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="beta_retention" name={`β=${selectedBeta} 保留`} stroke="#f59e0b" strokeWidth={2.0} dot={{ r: 2.5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ display: 'grid', gap: '12px' }}>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. beta 排名</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={rankingRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
                <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
                <XAxis dataKey="beta" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
                <Legend />
                <Bar dataKey="mean_closure" name="平均闭环" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="max_gain" name="最长长度增益" fill="#60a5fa" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 选中 beta 的长度增益</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={gainRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
                <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
                <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
                <Legend />
                <Bar dataKey="closure_gain" name="闭环增益" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="retention_gain" name="保留增益" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
