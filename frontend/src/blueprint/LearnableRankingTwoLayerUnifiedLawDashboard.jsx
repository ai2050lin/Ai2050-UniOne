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
import samplePayload from './data/learnable_ranking_two_layer_unified_law_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.baseline_learnable_two_layer && payload.ranking_layer && payload.learnable_ranking_two_layer_law && payload.leave_one_out);
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

export default function LearnableRankingTwoLayerUnifiedLawDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const compareRows = useMemo(
    () => [
      {
        law: '只学习标定层',
        mean_absolute_gap: Number(payload?.baseline_learnable_two_layer?.mean_absolute_gap || 0),
        score_correlation: Number(payload?.baseline_learnable_two_layer?.score_correlation || 0),
      },
      {
        law: '排序层和标定层都学习',
        mean_absolute_gap: Number(payload?.learnable_ranking_two_layer_law?.mean_absolute_gap || 0),
        score_correlation: Number(payload?.learnable_ranking_two_layer_law?.score_correlation || 0),
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
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 baseline_learnable_two_layer、ranking_layer、learnable_ranking_two_layer_law 或 leave_one_out 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`可学习排序层双层统一律 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(59,130,246,0.14), transparent 28%), radial-gradient(circle at top right, rgba(34,197,94,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(59,130,246,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>可学习排序层双层统一律</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把排序层也推进到可学习版本，直接比较“只学习标定层”和“排序层 + 标定层都学习”哪条路更稳。
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
        <MetricCard label="基线平均误差" value={fmt(payload?.baseline_learnable_two_layer?.mean_absolute_gap)} />
        <MetricCard label="基线相关系数" value={fmt(payload?.baseline_learnable_two_layer?.score_correlation)} />
        <MetricCard label="新双层平均误差" value={fmt(payload?.learnable_ranking_two_layer_law?.mean_absolute_gap)} />
        <MetricCard label="新双层相关系数" value={fmt(payload?.learnable_ranking_two_layer_law?.score_correlation)} />
        <MetricCard label="留一法平均误差" value={fmt(payload?.learnable_ranking_two_layer_law?.held_out_mean_gap)} />
        <MetricCard label="留一法相关系数" value={fmt(payload?.learnable_ranking_two_layer_law?.held_out_score_correlation)} />
        <MetricCard label="排序层正则" value={fmt(payload?.ranking_layer?.ridge_lambda, 2)} />
        <MetricCard label="标定层正则" value={fmt(payload?.calibration_layer?.ridge_lambda, 2)} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 基线与完整学习版对比</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={compareRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="law" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="mean_absolute_gap" name="平均误差" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="score_correlation" name="相关系数" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 参考分数、排序分数、标定分数</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={fittedRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="row_id" tick={{ fill: '#cbd5e1', fontSize: 10 }} angle={-18} textAnchor="end" height={58} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Line type="monotone" dataKey="reference_score" name="参考分数" stroke="#38bdf8" strokeWidth={2} />
              <Line type="monotone" dataKey="ranking_score" name="排序层分数" stroke="#a855f7" strokeWidth={2} />
              <Line type="monotone" dataKey="calibrated_score" name="标定后分数" stroke="#22c55e" strokeWidth={2} />
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
              <Bar dataKey="held_out_gap" name="留一法误差" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            这一步回答的是：统一律是不是必须把排序层也做成可学习结构。当前结果说明答案是肯定的，排序层学习后，
            样本内误差、留一法误差和相关性都明显更稳，说明双层统一律已经从“静态公式”进入“可训练机制”。
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            下一步问题：{payload?.project_readout?.next_question || '-'}
          </div>
        </div>
      </div>
    </div>
  );
}
