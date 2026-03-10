import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/semantic_4d_confidence_vector_domain_correction_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.semantic_4d_vector_domain_correction);
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function MetricCard({ label, value }) {
  return (
    <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
      <div style={{ color: '#94a3b8', fontSize: '10px' }}>{label}</div>
      <div style={{ color: '#f8fafc', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{value}</div>
    </div>
  );
}

export default function Semantic4DVectorDomainCorrectionDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const bars = useMemo(
    () => [
      { item: 'brain', gap: Number(payload?.semantic_4d_vector_domain_correction?.brain_held_out_gap || 0) },
      { item: 'D', gap: Number(payload?.semantic_4d_vector_domain_correction?.d_held_out_gap || 0) },
      { item: 'real-task', gap: Number(payload?.semantic_4d_vector_domain_correction?.real_task_held_out_gap || 0) },
    ],
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 semantic_4d_vector_domain_correction 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`语义 4D 向量域修正 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(29,78,216,0.14), transparent 28%), radial-gradient(circle at top right, rgba(37,99,235,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(29,78,216,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>语义 4D 骨架 + 极小向量域修正</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把单标量域修正升级成 `2D/3D` 极小向量修正，测试修正层的最小复杂度是否至少需要向量化。
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
        <MetricCard label="最佳修正维度" value={`${payload?.best_correction_dim || '-'} 维`} />
        <MetricCard label="留一法平均误差" value={fmt(payload?.semantic_4d_vector_domain_correction?.held_out_mean_gap)} />
        <MetricCard label="相对标量修正误差改善" value={fmt(payload?.semantic_4d_vector_domain_correction?.gap_improvement_vs_scalar)} />
        <MetricCard label="相对大基线误差差值" value={fmt(payload?.semantic_4d_vector_domain_correction?.gap_delta_vs_larger_baseline)} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三域留一法误差</div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={bars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip
              formatter={(value) => [fmt(value), '留一法误差']}
              contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }}
            />
            <Bar dataKey="gap" name="留一法误差" fill="#1d4ed8" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 当前判断</div>
        <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
          这一层测试修正层的最小复杂度是否至少要向量化。
          如果 `2D/3D` 已经明显优于单标量，就说明当前路线里“最小骨架 + 极小向量修正”比“最小骨架 + 标量修正”更接近真实结构。
        </div>
        <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
          下一步问题：{payload?.project_readout?.next_question || '-'}
        </div>
      </div>
    </div>
  );
}
