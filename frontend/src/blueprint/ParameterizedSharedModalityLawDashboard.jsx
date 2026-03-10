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
import samplePayload from './data/parameterized_shared_modality_law_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.meta && payload.fully_shared_law && payload.parameterized_shared_law && payload.modality_separate_oracle && payload.hypotheses);
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

function hypothesisText(flag) {
  return flag ? '成立' : '未成立';
}

export default function ParameterizedSharedModalityLawDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const compareRows = useMemo(
    () => [
      {
        law: '完全共享',
        in_sample_gap: Number(payload?.fully_shared_law?.mean_absolute_gap || 0),
        held_out_gap: Number(payload?.fully_shared_law?.mean_held_out_gap || 0),
      },
      {
        law: '共享机制 + 模态参数',
        in_sample_gap: Number(payload?.parameterized_shared_law?.mean_absolute_gap || 0),
        held_out_gap: Number(payload?.parameterized_shared_law?.mean_held_out_gap || 0),
      },
      {
        law: '模态独立拟合上限',
        in_sample_gap: Number(payload?.modality_separate_oracle?.mean_absolute_gap || 0),
        held_out_gap: Number(payload?.modality_separate_oracle?.mean_held_out_gap || 0),
      },
    ],
    [payload]
  );

  const modalityRows = useMemo(
    () =>
      (payload?.meta?.modalities || []).map((modality) => ({
        modality,
        fully_shared: Number(payload?.fully_shared_law?.modality_held_out_gap?.[modality] || 0),
        parameterized: Number(payload?.parameterized_shared_law?.modality_held_out_gap?.[modality] || 0),
        oracle: Number(payload?.modality_separate_oracle?.modality_held_out_gap?.[modality] || 0),
      })),
    [payload]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 fully_shared_law、parameterized_shared_law、modality_separate_oracle 或 hypotheses 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`跨模态共享机制参数化实验 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(16,185,129,0.14), transparent 28%), radial-gradient(circle at top right, rgba(245,158,11,0.08), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(16,185,129,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>跨模态共享机制参数化实验</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            直接比较三种写法：完全共享、共享机制加模态参数、模态独立拟合。核心问题是视觉、触觉、语言是否更像同一机制的不同参数区。
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
        <MetricCard label="完全共享留一法误差" value={fmt(payload?.fully_shared_law?.mean_held_out_gap)} />
        <MetricCard label="参数化共享留一法误差" value={fmt(payload?.parameterized_shared_law?.mean_held_out_gap)} />
        <MetricCard label="独立拟合留一法误差" value={fmt(payload?.modality_separate_oracle?.mean_held_out_gap)} />
        <MetricCard label="H1 参数化优于完全共享" value={hypothesisText(payload?.hypotheses?.H1_parameterized_beats_fully_shared)} />
        <MetricCard label="H2 参数化接近独立上限" value={hypothesisText(payload?.hypotheses?.H2_parameterized_close_to_oracle)} />
        <MetricCard label="H3 同机制不同参数" value={hypothesisText(payload?.hypotheses?.H3_same_mechanism_different_params_is_supported)} />
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三种写法对比</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={compareRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="law" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="in_sample_gap" name="样本内误差" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="held_out_gap" name="留一法误差" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 各模态留一法误差</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={modalityRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="modality" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="fully_shared" name="完全共享" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="parameterized" name="共享机制 + 模态参数" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="oracle" name="模态独立拟合" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 当前判断</div>
        <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
          现在最准确的结论不是“视觉、听觉、语言已经证实是同一机制”，也不是“这个假设被否定”。
          当前证据更接近“兼容但未封口”：
          参数化共享律在样本内明显更接近模态独立上限，但留一法没有稳定打赢完全共享律。
          所以这更像一个值得继续追的强候选，而不是已完成证明。
        </div>
        <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
          下一步问题：{payload?.project_readout?.next_question || '-'}
        </div>
      </div>
    </div>
  );
}
