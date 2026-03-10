import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/unified_update_law_candidate_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.law_form && payload.best_law && Array.isArray(payload.view_results));
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function MetricCard({ label, value }) {
  return (
    <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
      <div style={{ color: '#94a3b8', fontSize: '10px' }}>{label}</div>
      <div style={{ color: '#f8fafc', fontSize: '18px', fontWeight: 'bold', marginTop: '4px' }}>{value}</div>
    </div>
  );
}

export default function UnifiedUpdateLawDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const lineRows = useMemo(
    () =>
      (payload?.view_results || []).map((row) => ({
        view: row.view_id,
        reference_score: Number(row.reference_score || 0),
        raw_four_factor_score: Number(row.raw_four_factor_score || 0),
        unified_score: Number(row.unified_score || 0),
      })),
    [payload]
  );

  const gapRows = useMemo(
    () =>
      (payload?.view_results || []).map((row) => ({
        view: row.view_id,
        raw_gap: Number(row.raw_gap || 0),
        absolute_gap: Number(row.absolute_gap || 0),
      })),
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) throw new Error('缺少 law_form / best_law / view_results 字段');
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`统一更新律 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(14,165,233,0.12), transparent 28%), radial-gradient(circle at top right, rgba(250,204,21,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(56,189,248,0.22)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>统一更新律候选</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            在 `base / adaptive_offset / routing / stabilization` 之上，只用两参数修正 `adaptive_offset`，
            检查是否已经能逼近当前桥接分数。
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px' }}>
        <MetricCard label="统一律是否通过" value={payload?.best_law?.pass ? '通过' : '未通过'} />
        <MetricCard label="route_gain" value={fmt(payload?.best_law?.route_gain, 1)} />
        <MetricCard label="stabilize_drag" value={fmt(payload?.best_law?.stabilize_drag, 1)} />
        <MetricCard label="统一律平均误差" value={fmt(payload?.best_law?.mean_absolute_gap)} />
        <MetricCard label="四因子平均误差" value={fmt(payload?.best_law?.raw_mean_absolute_gap)} />
        <MetricCard label="误差改进" value={fmt(payload?.best_law?.gap_improvement)} />
        <MetricCard label="统一律相关系数" value={fmt(payload?.best_law?.score_correlation)} />
        <MetricCard label="留一法平均误差" value={fmt(payload?.leave_one_out?.mean_held_out_gap)} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 参考分数 vs 两参数统一律</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={lineRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="view" tick={{ fill: '#cbd5e1', fontSize: 10 }} angle={-18} textAnchor="end" height={58} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Line type="monotone" dataKey="reference_score" name="reference_score" stroke="#38bdf8" strokeWidth={2} />
              <Line type="monotone" dataKey="raw_four_factor_score" name="raw_four_factor_score" stroke="#f59e0b" strokeWidth={2} />
              <Line type="monotone" dataKey="unified_score" name="unified_score" stroke="#22c55e" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 原始误差 vs 统一律误差</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={gapRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="view" tick={{ fill: '#cbd5e1', fontSize: 10 }} angle={-18} textAnchor="end" height={58} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="raw_gap" name="raw_gap" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="absolute_gap" name="absolute_gap" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(280px, 0.95fr) minmax(0, 1.05fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 候选律</div>
          <div style={{ color: '#f8fafc', fontSize: '11px', lineHeight: '1.8', whiteSpace: 'pre-wrap' }}>
            effective_offset = {payload?.law_form?.effective_offset || '-'}
            {'\n'}
            unified_score = {payload?.law_form?.unified_score || '-'}
          </div>
          <div style={{ marginTop: '10px', color: '#dbeafe', fontSize: '12px' }}>直觉解释</div>
          <div style={{ marginTop: '8px', display: 'grid', gap: '8px' }}>
            {(payload?.law_form?.intuition || []).map((item) => (
              <div key={item} style={{ border: '1px solid rgba(148,163,184,0.16)', borderRadius: '10px', padding: '8px 10px', color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7' }}>
                {item}
              </div>
            ))}
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>{payload?.project_readout?.current_verdict || '-'}</div>
          <div style={{ marginTop: '10px', color: '#fbbf24', fontSize: '11px', lineHeight: '1.7' }}>
            风险提示：{payload?.project_readout?.risk_note || '-'}
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            下一步问题：{payload?.project_readout?.next_question || '-'}
          </div>
        </div>
      </div>
    </div>
  );
}
