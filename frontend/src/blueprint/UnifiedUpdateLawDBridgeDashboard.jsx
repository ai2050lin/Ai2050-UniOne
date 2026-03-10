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
import samplePayload from './data/unified_update_law_d_bridge_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.transfer_law && payload.fitted_d_law && Array.isArray(payload.methods));
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

export default function UnifiedUpdateLawDBridgeDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const methodRows = useMemo(
    () =>
      (payload?.methods || []).map((row) => ({
        method: row.method,
        reference_score: Number(row.reference_score || 0),
        predicted_score: Number(row.predicted_score || 0),
        absolute_gap: Number(row.absolute_gap || 0),
        novel_gain: Number(row.novel_gain || 0),
        retention_gain: Number(row.retention_gain || 0),
      })),
    [payload]
  );

  const lawRows = useMemo(
    () => [
      {
        law: '桥接统一律',
        route_gain: Number(payload?.transfer_law?.route_gain || 0),
        stabilize_drag: Number(payload?.transfer_law?.stabilize_drag || 0),
        mean_absolute_gap: Number(payload?.transfer_law?.mean_absolute_gap || 0),
      },
      {
        law: 'D 专用统一律',
        route_gain: Number(payload?.fitted_d_law?.route_gain || 0),
        stabilize_drag: Number(payload?.fitted_d_law?.stabilize_drag || 0),
        mean_absolute_gap: Number(payload?.fitted_d_law?.mean_absolute_gap || 0),
      },
    ],
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) throw new Error('缺少 transfer_law / fitted_d_law / methods 字段');
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`D 桥接统一律 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(244,114,182,0.12), transparent 28%), radial-gradient(circle at top right, rgba(16,185,129,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(244,114,182,0.22)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>统一更新律到 D 桥接</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            对比“桥接里学到的小律”和“D 上重新拟合的小律”，直接观察统一结构一进入接地闭环后，主导项是否从 routing 切到 stabilization。
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
        <MetricCard label="桥接律误差" value={fmt(payload?.transfer_law?.mean_absolute_gap)} />
        <MetricCard label="桥接律相关系数" value={fmt(payload?.transfer_law?.score_correlation)} />
        <MetricCard label="D 律误差" value={fmt(payload?.fitted_d_law?.mean_absolute_gap)} />
        <MetricCard label="D 律相关系数" value={fmt(payload?.fitted_d_law?.score_correlation)} />
        <MetricCard label="route_gain 转移" value={`${fmt(payload?.transfer_law?.route_gain, 1)} -> ${fmt(payload?.fitted_d_law?.route_gain, 1)}`} />
        <MetricCard label="stabilize_drag 转移" value={`${fmt(payload?.transfer_law?.stabilize_drag, 1)} -> ${fmt(payload?.fitted_d_law?.stabilize_drag, 1)}`} />
        <MetricCard label="误差改进" value={fmt(payload?.fitted_d_law?.gap_improvement_vs_transfer)} />
        <MetricCard label="相关改进" value={fmt(payload?.fitted_d_law?.correlation_improvement_vs_transfer)} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 两条统一律参数对比</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={lawRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="law" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="route_gain" name="route_gain" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="stabilize_drag" name="stabilize_drag" fill="#f472b6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. D 方法参考分数 vs D 律预测</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={methodRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="method" tick={{ fill: '#cbd5e1', fontSize: 10 }} angle={-18} textAnchor="end" height={58} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Line type="monotone" dataKey="reference_score" name="reference_score" stroke="#22c55e" strokeWidth={2} />
              <Line type="monotone" dataKey="predicted_score" name="predicted_score" stroke="#f59e0b" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. novel_gain vs retention_gain</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={methodRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="method" tick={{ fill: '#cbd5e1', fontSize: 10 }} angle={-18} textAnchor="end" height={58} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="novel_gain" name="novel_gain" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="retention_gain" name="retention_gain" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            桥接里学到的小律在 D 上直接迁移失败，说明“从内部结构到接地闭环”的关键变化不在 `base`，而在 `adaptive_offset` 的稳定化。
          </div>
          <div style={{ marginTop: '10px', color: '#fbbf24', fontSize: '11px', lineHeight: '1.7' }}>
            当前信号：`route_gain` 从 {fmt(payload?.transfer_law?.route_gain, 1)} 降到 {fmt(payload?.fitted_d_law?.route_gain, 1)}，
            `stabilize_drag` 从 {fmt(payload?.transfer_law?.stabilize_drag, 1)} 升到 {fmt(payload?.fitted_d_law?.stabilize_drag, 1)}。
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            下一步问题：{payload?.project_readout?.next_question || '-'}
          </div>
        </div>
      </div>
    </div>
  );
}
