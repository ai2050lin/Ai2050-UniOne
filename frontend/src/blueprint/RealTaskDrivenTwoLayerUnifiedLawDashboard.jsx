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
import samplePayload from './data/real_task_driven_two_layer_unified_law_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.baseline && payload.ranking_layer && payload.real_task_two_layer_law && payload.leave_one_out);
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

export default function RealTaskDrivenTwoLayerUnifiedLawDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const compareRows = useMemo(
    () => [
      {
        law: '手工结构基线',
        score_correlation: Number(payload?.baseline?.score_correlation || 0),
        mean_absolute_gap: 0,
      },
      {
        law: '真实任务双层统一律',
        score_correlation: Number(payload?.real_task_two_layer_law?.score_correlation || 0),
        mean_absolute_gap: Number(payload?.real_task_two_layer_law?.mean_absolute_gap || 0),
      },
    ],
    [payload]
  );

  const taskRows = useMemo(
    () =>
      (payload?.rows || []).map((row) => ({
        row_id: row.row_id,
        reference_score: Number(row.reference_score || 0),
        baseline_score: Number(row.baseline_score || 0),
        ranking_score: Number(row.ranking_score || 0),
        calibrated_score: Number(row.calibrated_score || 0),
      })),
    [payload]
  );

  const holdoutRows = useMemo(
    () =>
      (payload?.leave_one_out?.rows || []).map((row) => ({
        row_id: row.row_id,
        held_out_gap: Number(row.held_out_gap || 0),
      })),
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) throw new Error('缺少 baseline、ranking_layer、real_task_two_layer_law 或 leave_one_out 字段');
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`真实任务双层统一律 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(249,115,22,0.12), transparent 28%), radial-gradient(circle at top right, rgba(16,185,129,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(249,115,22,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>真实任务驱动双层统一律</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            不再只在桥接行上拟合，而是直接用概念条件真实任务的 `behavior_gain` 来训练和评估双层统一律。
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
        <MetricCard label="基线相关系数" value={fmt(payload?.baseline?.score_correlation)} />
        <MetricCard label="统一律误差" value={fmt(payload?.real_task_two_layer_law?.mean_absolute_gap)} />
        <MetricCard label="统一律相关系数" value={fmt(payload?.real_task_two_layer_law?.score_correlation)} />
        <MetricCard label="留一法误差" value={fmt(payload?.real_task_two_layer_law?.held_out_mean_gap)} />
        <MetricCard label="留一法相关系数" value={fmt(payload?.real_task_two_layer_law?.held_out_score_correlation)} />
        <MetricCard label="相关改善" value={fmt(payload?.real_task_two_layer_law?.correlation_improvement_vs_baseline)} />
        <MetricCard label="排序层正则" value={fmt(payload?.ranking_layer?.ridge_lambda, 2)} />
        <MetricCard label="标定层正则" value={fmt(payload?.calibration_layer?.ridge_lambda, 2)} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 手工结构基线 vs 真实任务双层统一律</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={compareRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="law" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="score_correlation" name="score_correlation" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="mean_absolute_gap" name="mean_absolute_gap" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 真实任务行上的预测对比</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={taskRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="row_id" tick={{ fill: '#cbd5e1', fontSize: 9 }} angle={-20} textAnchor="end" height={72} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Line type="monotone" dataKey="reference_score" name="reference_score" stroke="#38bdf8" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="baseline_score" name="baseline_score" stroke="#f97316" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="calibrated_score" name="calibrated_score" stroke="#22c55e" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 留一法误差</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={holdoutRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="row_id" tick={{ fill: '#cbd5e1', fontSize: 9 }} angle={-20} textAnchor="end" height={72} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="held_out_gap" name="held_out_gap" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            这一步专门回答统一律能不能开始解释真实任务里的收益变化。如果它在真实任务行上也稳定，
            那这条路线就不再只是“内部结构好看”，而是开始具备外部行为闭环。
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            下一步问题：{payload?.project_readout?.next_question || '-'}
          </div>
        </div>
      </div>
    </div>
  );
}
