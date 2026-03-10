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
import samplePayload from './data/semantic_4d_brain_constraint_expansion_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.baseline_semantic_4d_vector && payload.brain_constraint_expansion_leave_one_out);
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

export default function Semantic4DBrainConstraintExpansionDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const bars = useMemo(() => ([
    {
      item: 'brain',
      baseline: Number(payload?.baseline_semantic_4d_vector?.brain_held_out_gap || 0),
      expanded: Number(payload?.brain_constraint_expansion_leave_one_out?.brain_held_out_gap || 0),
    },
    {
      item: 'D',
      baseline: Number(payload?.baseline_semantic_4d_vector?.d_held_out_gap || 0),
      expanded: Number(payload?.brain_constraint_expansion_leave_one_out?.d_held_out_gap || 0),
    },
    {
      item: 'real-task',
      baseline: Number(payload?.baseline_semantic_4d_vector?.real_task_held_out_gap || 0),
      expanded: Number(payload?.brain_constraint_expansion_leave_one_out?.real_task_held_out_gap || 0),
    },
  ]), [payload]);

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 brain_constraint_expansion_leave_one_out 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`脑侧候选约束扩展 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(15,118,110,0.16), transparent 30%), radial-gradient(circle at top right, rgba(8,145,178,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(13,148,136,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>语义 4D + 3D 的脑侧候选约束系统扩展</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            不再做噪声扩增，而是把脑侧桥接结果拆成“组件聚焦约束 + 跨模型聚合约束”，直接看更厚的脑侧候选面是否还能稳定支撑当前统一骨架。
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
        <MetricCard label="模板约束行数" value={payload?.meta?.constraint_row_count_template || 0} />
        <MetricCard label="平均约束行数/留一折" value={fmt(payload?.brain_constraint_expansion_leave_one_out?.mean_constraint_count, 2)} />
        <MetricCard label="脑侧留一法误差改善" value={fmt(payload?.improvement?.brain_gap_improvement)} />
        <MetricCard label="整体相关提升" value={fmt(payload?.improvement?.corr_improvement)} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三域留一法误差对比</div>
        <ResponsiveContainer width="100%" height={270}>
          <BarChart data={bars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip
              formatter={(value) => [fmt(value), '留一法误差']}
              contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }}
            />
            <Bar dataKey="baseline" name="基线 4D + 3D" fill="#0f766e" radius={[4, 4, 0, 0]} />
            <Bar dataKey="expanded" name="结构化脑侧约束扩展" fill="#0891b2" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 约束混合构成</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.9' }}>
            <div>{`组件聚焦约束: ${payload?.constraint_mix?.component_focus_rows || 0}`}</div>
            <div>{`跨模型聚合约束: ${payload?.constraint_mix?.component_aggregate_rows || 0}`}</div>
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            这一步的重点不是增加噪声样本，而是让脑侧候选约束从“少量模型级桥接分数”扩成“组件级、聚合级”的系统面。
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            如果结构化脑侧约束扩展后，`brain_held_out_gap` 继续下降，就说明主骨架并不依赖“加噪扩增”才稳定，而是能在更系统的脑侧候选面上继续成立。
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            下一步问题：{payload?.project_readout?.next_question || '-'}
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 样例约束行</div>
        <div style={{ display: 'grid', gap: '8px' }}>
          {(payload?.sample_constraints || []).map((row) => (
            <div key={row.row_id} style={{ border: '1px solid rgba(148,163,184,0.16)', borderRadius: '10px', padding: '10px', background: 'rgba(15,23,42,0.38)' }}>
              <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 600 }}>{row.subdomain}</div>
              <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '4px' }}>
                {`类型: ${row.constraint_kind} | 参考分数: ${fmt(row.reference_score)}`}
              </div>
              <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>
                {`来源脑侧行: ${(row.source_brain_rows || []).join(', ')}`}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
