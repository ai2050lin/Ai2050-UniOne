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
import samplePayload from './data/learnable_two_layer_unified_law_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.ranking_layer && payload.learnable_calibration_layer && payload.learnable_two_layer_law && payload.leave_one_out);
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

export default function LearnableTwoLayerUnifiedLawDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const compareRows = useMemo(
    () => [
      {
        law: '排序层',
        mean_absolute_gap: Number(payload?.ranking_layer?.mean_absolute_gap || 0),
        score_correlation: Number(payload?.ranking_layer?.score_correlation || 0),
      },
      {
        law: '可学习双层',
        mean_absolute_gap: Number(payload?.learnable_two_layer_law?.mean_absolute_gap || 0),
        score_correlation: Number(payload?.learnable_two_layer_law?.score_correlation || 0),
      },
    ],
    [payload]
  );

  const fittedRows = useMemo(
    () =>
      (payload?.rows || []).map((row) => ({
        row_id: row.row_id,
        reference_score: Number(row.reference_score || 0),
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
      if (!isValidPayload(parsed)) throw new Error('缺少 ranking_layer、learnable_calibration_layer、learnable_two_layer_law 或 leave_one_out 字段');
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`可学习双层统一律 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(16,185,129,0.14), transparent 28%), radial-gradient(circle at top right, rgba(59,130,246,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(16,185,129,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>可学习双层统一律</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            用带正则的可学习标定层替代解析拟合，直接检查双层统一律是不是已经从“原型”进入“可训练方向”。
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
        <MetricCard label="排序层误差" value={fmt(payload?.ranking_layer?.mean_absolute_gap)} />
        <MetricCard label="排序层相关系数" value={fmt(payload?.ranking_layer?.score_correlation)} />
        <MetricCard label="双层统一律误差" value={fmt(payload?.learnable_two_layer_law?.mean_absolute_gap)} />
        <MetricCard label="双层统一律相关系数" value={fmt(payload?.learnable_two_layer_law?.score_correlation)} />
        <MetricCard label="误差改善" value={fmt(payload?.learnable_two_layer_law?.gap_improvement_vs_ranking)} />
        <MetricCard label="相关改善" value={fmt(payload?.learnable_two_layer_law?.correlation_improvement_vs_ranking)} />
        <MetricCard label="留一法平均误差" value={fmt(payload?.leave_one_out?.mean_held_out_gap)} />
        <MetricCard label="正则系数" value={fmt(payload?.learnable_calibration_layer?.ridge_lambda, 1)} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 排序层 vs 可学习双层统一律</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={compareRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="law" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="mean_absolute_gap" name="mean_absolute_gap" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="score_correlation" name="score_correlation" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 参考分数 / 排序分数 / 学习后分数</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={fittedRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="row_id" tick={{ fill: '#cbd5e1', fontSize: 10 }} angle={-18} textAnchor="end" height={58} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Line type="monotone" dataKey="reference_score" name="reference_score" stroke="#38bdf8" strokeWidth={2} />
              <Line type="monotone" dataKey="ranking_score" name="ranking_score" stroke="#a855f7" strokeWidth={2} />
              <Line type="monotone" dataKey="calibrated_score" name="calibrated_score" stroke="#22c55e" strokeWidth={2} />
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
              <XAxis dataKey="row_id" tick={{ fill: '#cbd5e1', fontSize: 10 }} angle={-18} textAnchor="end" height={58} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="held_out_gap" name="held_out_gap" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            可学习标定层已经把双层统一律从“样本内强、外推弱”的解析原型推进到“低误差、高相关、留一法通过”的可训练方向。
            下一步最值得做的不是继续打磨解析式，而是把排序层也推进到可学习版本。
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            下一步问题：{payload?.project_readout?.next_question || '-'}
          </div>
        </div>
      </div>
    </div>
  );
}
