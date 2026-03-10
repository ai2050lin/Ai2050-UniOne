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
import samplePayload from './data/real_multistep_memory_multiscale_scan_sample.json';

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
  if (name === 'dual_anchor_beta_050_086') return '双锚点 β=(0.50,0.86)';
  if (name === 'triple_anchor_beta_050_080_092') return '三锚点 β=(0.50,0.80,0.92)';
  return name;
}

export default function RealMultistepMemoryMultiscaleDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const systems = payload?.systems || {};
  const ranking = Array.isArray(payload?.ranking) ? payload.ranking : [];
  const best = payload?.best_systems || {};
  const hypotheses = payload?.hypotheses || {};
  const trace = systems?.trace_gated_local?.global_summary || {};
  const single = systems?.single_anchor_beta_086?.global_summary || {};
  const dual = systems?.dual_anchor_beta_050_086?.global_summary || {};
  const triple = systems?.triple_anchor_beta_050_080_092?.global_summary || {};

  const curveRows = useMemo(() => {
    const lengths = Array.isArray(trace?.lengths) ? trace.lengths : [];
    return lengths.map((length, idx) => ({
      length: `L=${length}`,
      trace: Number(trace?.closure_curve?.[idx] || 0),
      single: Number(single?.closure_curve?.[idx] || 0),
      dual: Number(dual?.closure_curve?.[idx] || 0),
      triple: Number(triple?.closure_curve?.[idx] || 0),
    }));
  }, [trace, single, dual, triple]);

  const retentionRows = useMemo(() => {
    const lengths = Array.isArray(trace?.lengths) ? trace.lengths : [];
    return lengths.map((length, idx) => ({
      length: `L=${length}`,
      trace: Number(trace?.retention_curve?.[idx] || 0),
      single: Number(single?.retention_curve?.[idx] || 0),
      dual: Number(dual?.retention_curve?.[idx] || 0),
      triple: Number(triple?.retention_curve?.[idx] || 0),
    }));
  }, [trace, single, dual, triple]);

  const rankingRows = useMemo(
    () =>
      ranking.map((row) => ({
        system: labelOf(row.system),
        mean_closure: Number(row.mean_closure_score || 0),
        mean_retention: Number(row.mean_retention_score || 0),
      })),
    [ranking]
  );

  const decayRows = useMemo(
    () =>
      ranking.map((row) => ({
        system: labelOf(row.system),
        closure_drop: Number(row.closure_relative_drop || 0),
        max_length: Number(row.max_length_score || 0),
      })),
    [ranking]
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
      setError(`多时间常数记忆簇 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(34,197,94,0.12), transparent 28%), radial-gradient(circle at top right, rgba(251,191,36,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(34,197,94,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>多时间常数记忆簇</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            比较 `trace`、单锚点和多锚点系统，直接看多时间常数是提升闭环，还是主要改善保留率与长程衰减平坦度。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均保留最优</div>
          <div style={{ color: '#f59e0b', fontSize: '17px', fontWeight: 'bold' }}>{labelOf(best?.best_mean_retention?.system)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>闭环掉幅最小</div>
          <div style={{ color: '#f472b6', fontSize: '17px', fontWeight: 'bold' }}>{labelOf(best?.slowest_decay?.system)}</div>
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
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 闭环曲线</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={curveRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="trace" name="trace" stroke="#94a3b8" strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="single" name="单锚点 β=0.86" stroke="#22c55e" strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="dual" name="双锚点" stroke="#60a5fa" strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="triple" name="三锚点" stroke="#f59e0b" strokeWidth={2.2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 保留率曲线</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={retentionRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="trace" name="trace" stroke="#94a3b8" strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="single" name="单锚点 β=0.86" stroke="#22c55e" strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="dual" name="双锚点" stroke="#60a5fa" strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="triple" name="三锚点" stroke="#f59e0b" strokeWidth={2.2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 平均闭环与平均保留排序</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={rankingRows} margin={{ top: 10, right: 12, left: 0, bottom: 30 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="system" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-15} textAnchor="end" height={54} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="mean_closure" name="平均闭环" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="mean_retention" name="平均保留" fill="#60a5fa" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 掉幅与最长任务表现</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={decayRows} margin={{ top: 10, right: 12, left: 0, bottom: 30 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="system" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-15} textAnchor="end" height={54} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="closure_drop" name="闭环掉幅" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="max_length" name="最长任务分数" fill="#f472b6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
