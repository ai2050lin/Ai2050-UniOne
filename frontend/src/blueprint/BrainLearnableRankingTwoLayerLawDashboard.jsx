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
import samplePayload from './data/brain_learnable_ranking_two_layer_unified_law_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.baseline_brain_d_real && payload.ranking_layer && payload.brain_learnable_ranking_two_layer_law && payload.leave_one_out);
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

export default function BrainLearnableRankingTwoLayerLawDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const compareRows = useMemo(
    () => [
      {
        item: '脑侧平均误差',
        baseline: Number(payload?.baseline_brain_d_real?.brain_mean_gap || 0),
        learnable: Number(payload?.brain_learnable_ranking_two_layer_law?.brain_mean_gap || 0),
      },
      {
        item: '脑侧留一法误差',
        baseline: Number(payload?.baseline_brain_d_real?.brain_held_out_gap || 0),
        learnable: Number(payload?.brain_learnable_ranking_two_layer_law?.brain_held_out_gap || 0),
      },
      {
        item: '联合平均误差',
        baseline: Number(payload?.baseline_brain_d_real?.mean_absolute_gap || 0),
        learnable: Number(payload?.brain_learnable_ranking_two_layer_law?.mean_absolute_gap || 0),
      },
      {
        item: '留一法平均误差',
        baseline: Number(payload?.baseline_brain_d_real?.held_out_mean_gap || 0),
        learnable: Number(payload?.brain_learnable_ranking_two_layer_law?.held_out_mean_gap || 0),
      },
    ],
    [payload]
  );

  const coefficientRows = useMemo(
    () =>
      Object.entries(payload?.ranking_layer?.brain_component_coefficients || {}).map(([key, value]) => ({
        component: key,
        coefficient: Number(value || 0),
      })),
    [payload]
  );

  const brainRows = useMemo(
    () =>
      (payload?.brain_breakdown || []).map((row) => ({
        model_name: row.model_name,
        reference_score: Number(row.overall_bridge_score || 0),
        ranking_score: Number(row.ranking_score || 0),
        calibrated_score: Number(row.calibrated_score || 0),
      })),
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 baseline_brain_d_real、ranking_layer、brain_learnable_ranking_two_layer_law 或 leave_one_out 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`脑侧可学习排序层双层统一律 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(6,182,212,0.14), transparent 28%), radial-gradient(circle at top right, rgba(132,204,22,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(6,182,212,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>脑侧可学习排序层双层统一律</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            不再用手工脑侧聚合，而是直接把脑侧组件分数送入排序层，检查脑侧约束能否作为可学习域并入统一律。
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
        <MetricCard label="脑侧平均误差" value={fmt(payload?.brain_learnable_ranking_two_layer_law?.brain_mean_gap)} />
        <MetricCard label="脑侧留一法误差" value={fmt(payload?.brain_learnable_ranking_two_layer_law?.brain_held_out_gap)} />
        <MetricCard label="脑侧误差改善" value={fmt(payload?.brain_learnable_ranking_two_layer_law?.brain_gap_improvement)} />
        <MetricCard label="脑侧留一法改善" value={fmt(payload?.brain_learnable_ranking_two_layer_law?.brain_held_out_improvement)} />
        <MetricCard label="联合平均误差" value={fmt(payload?.brain_learnable_ranking_two_layer_law?.mean_absolute_gap)} />
        <MetricCard label="联合留一法误差" value={fmt(payload?.brain_learnable_ranking_two_layer_law?.held_out_mean_gap)} />
        <MetricCard label="联合相关系数" value={fmt(payload?.brain_learnable_ranking_two_layer_law?.score_correlation)} />
        <MetricCard label="联合留一法相关" value={fmt(payload?.brain_learnable_ranking_two_layer_law?.held_out_score_correlation)} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 手工脑侧分解 vs 可学习脑侧排序层</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={compareRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="baseline" name="手工脑侧分解" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="learnable" name="可学习脑侧排序层" fill="#06b6d4" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 脑侧组件系数</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={coefficientRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="component" tick={{ fill: '#cbd5e1', fontSize: 10 }} angle={-18} textAnchor="end" height={58} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="coefficient" name="排序层系数" fill="#84cc16" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 脑侧参考分数与统一律分数</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={brainRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model_name" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Line type="monotone" dataKey="reference_score" name="脑桥接参考分数" stroke="#38bdf8" strokeWidth={2} />
              <Line type="monotone" dataKey="calibrated_score" name="统一律分数" stroke="#f59e0b" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            这一步说明脑侧约束已经可以从“手工外部规则”推进到“可学习排序域”。当前脑侧域误差明显下降，
            但联合留一法略有变差，说明下一步要解决的是跨域平衡，而不是继续堆脑侧手工特征。
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            下一步问题：{payload?.project_readout?.next_question || '-'}
          </div>
        </div>
      </div>
    </div>
  );
}
