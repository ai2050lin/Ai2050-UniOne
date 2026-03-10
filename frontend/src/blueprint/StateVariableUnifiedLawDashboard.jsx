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
import samplePayload from './data/state_variable_calibrated_unified_law_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.phase_gated_baseline && payload.state_variable_law && Array.isArray(payload.rows));
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

export default function StateVariableUnifiedLawDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const rows = useMemo(
    () =>
      (payload?.rows || []).map((row) => ({
        row_id: row.row_id,
        reference_score: Number(row.reference_score || 0),
        calibrated_score: Number(row.calibrated_score || 0),
        z_state: Number(row.z_state || 0),
        phase_gate: Number(row.phase_gate || 0),
      })),
    [payload]
  );

  const compareRows = useMemo(
    () => [
      {
        law: '相位门控律',
        mean_absolute_gap: Number(payload?.phase_gated_baseline?.mean_absolute_gap || 0),
        score_correlation: Number(payload?.phase_gated_baseline?.score_correlation || 0),
      },
      {
        law: '状态变量律',
        mean_absolute_gap: Number(payload?.state_variable_law?.mean_absolute_gap || 0),
        score_correlation: Number(payload?.state_variable_law?.score_correlation || 0),
      },
    ],
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) throw new Error('缺少 phase_gated_baseline / state_variable_law / rows 字段');
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`状态变量统一律 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(250,204,21,0.12), transparent 28%), radial-gradient(circle at top right, rgba(16,185,129,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(250,204,21,0.22)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>状态变量统一律</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            在相位门控律之上加入 `z_state` 和小标定项，直接看它能不能同时改善误差和相关性。
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
        <MetricCard label="状态变量律误差" value={fmt(payload?.state_variable_law?.mean_absolute_gap)} />
        <MetricCard label="状态变量律相关系数" value={fmt(payload?.state_variable_law?.score_correlation)} />
        <MetricCard label="误差改进" value={fmt(payload?.state_variable_law?.gap_improvement_vs_phase_gated)} />
        <MetricCard label="相关改进" value={fmt(payload?.state_variable_law?.correlation_improvement_vs_phase_gated)} />
        <MetricCard label="bridge_mean_gap" value={fmt(payload?.state_variable_law?.bridge_mean_gap)} />
        <MetricCard label="d_mean_gap" value={fmt(payload?.state_variable_law?.d_mean_gap)} />
        <MetricCard label="alpha / beta / bias" value={`${fmt(payload?.state_variable_law?.alpha, 1)} / ${fmt(payload?.state_variable_law?.beta, 1)} / ${fmt(payload?.state_variable_law?.bias, 1)}`} />
        <MetricCard label="cal_shift / cal_scale" value={`${fmt(payload?.state_variable_law?.cal_shift, 1)} / ${fmt(payload?.state_variable_law?.cal_scale, 1)}`} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 相位门控律 vs 状态变量律</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={compareRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="law" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="mean_absolute_gap" name="mean_absolute_gap" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="score_correlation" name="score_correlation" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 参考分数 vs 状态变量律预测</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={rows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="row_id" tick={{ fill: '#cbd5e1', fontSize: 10 }} angle={-18} textAnchor="end" height={58} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Line type="monotone" dataKey="reference_score" name="reference_score" stroke="#38bdf8" strokeWidth={2} />
              <Line type="monotone" dataKey="calibrated_score" name="calibrated_score" stroke="#f59e0b" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. z_state 与 phase_gate</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={rows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="row_id" tick={{ fill: '#cbd5e1', fontSize: 10 }} angle={-18} textAnchor="end" height={58} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Line type="monotone" dataKey="z_state" name="z_state" stroke="#a855f7" strokeWidth={2} />
              <Line type="monotone" dataKey="phase_gate" name="phase_gate" stroke="#22c55e" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            状态变量律已经把相位门控律的联合误差明显压下来了，但还没有保住上一轮的超高相关性。也就是说，它开始同时管“排序”和“标定”，但还没有把两件事一起做到最好。
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            下一步问题：{payload?.project_readout?.next_question || '-'}
          </div>
        </div>
      </div>
    </div>
  );
}
