import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import baseSample from './data/open_world_continuous_grounding_stream_sample.json';
import scanSample from './data/open_world_continuous_grounding_stream_scan_sample.json';

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

function isValidBase(payload) {
  return Boolean(payload && payload.systems && payload.gains_vs_direct);
}

function isValidScan(payload) {
  return Boolean(payload && payload.best_config && Array.isArray(payload.rows));
}

export default function OpenWorldContinuousGroundingDashboard() {
  const [basePayload, setBasePayload] = useState(baseSample);
  const [scanPayload, setScanPayload] = useState(scanSample);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const bars = useMemo(() => ([
    {
      item: 'stable-old',
      direct: Number(basePayload?.systems?.direct_stream?.stable_old_concept_accuracy || 0),
      baseShared: Number(basePayload?.systems?.shared_offset_stream?.stable_old_concept_accuracy || 0),
      tuned: Number(scanPayload?.best_config?.stable_old_concept_accuracy || 0),
    },
    {
      item: 'novel',
      direct: Number(basePayload?.systems?.direct_stream?.novel_concept_accuracy || 0),
      baseShared: Number(basePayload?.systems?.shared_offset_stream?.novel_concept_accuracy || 0),
      tuned: Number(scanPayload?.best_config?.novel_concept_accuracy || 0),
    },
    {
      item: 'drifted',
      direct: Number(basePayload?.systems?.direct_stream?.drifted_concept_accuracy || 0),
      baseShared: Number(basePayload?.systems?.shared_offset_stream?.drifted_concept_accuracy || 0),
      tuned: Number(scanPayload?.best_config?.drifted_concept_accuracy || 0),
    },
    {
      item: 'closure',
      direct: Number(basePayload?.systems?.direct_stream?.closure_score || 0),
      baseShared: Number(basePayload?.systems?.shared_offset_stream?.closure_score || 0),
      tuned: Number(scanPayload?.best_config?.closure_score || 0),
    },
  ]), [basePayload, scanPayload]);

  async function onUploadBase(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidBase(parsed)) throw new Error('缺少 systems / gains_vs_direct 字段');
      setBasePayload(parsed);
      setSource(`基准文件: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`开放世界流式接地基准 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  async function onUploadScan(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidScan(parsed)) throw new Error('缺少 best_config / rows 字段');
      setScanPayload(parsed);
      setSource(`扫描文件: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`开放世界流式接地扫描 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setBasePayload(baseSample);
    setScanPayload(scanSample);
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
          'radial-gradient(circle at top left, rgba(14,165,233,0.14), transparent 28%), radial-gradient(circle at top right, rgba(16,185,129,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(14,165,233,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>开放世界连续流接地闭环</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把接地测试从阶段化原型推进到连续流环境，加入背景漂移、模态缺失、噪声片段和旧概念重访，再配合更新律扫描，直接看闭环能否翻正。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <label style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px', cursor: 'pointer' }}>
            导入基准 JSON
            <input type="file" accept="application/json" onChange={onUploadBase} style={{ display: 'none' }} />
          </label>
          <label style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px', cursor: 'pointer' }}>
            导入扫描 JSON
            <input type="file" accept="application/json" onChange={onUploadScan} style={{ display: 'none' }} />
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
        <MetricCard label="基线 closure 增益" value={fmt(basePayload?.gains_vs_direct?.closure_score_gain)} />
        <MetricCard label="扫描最优 closure 增益" value={fmt(scanPayload?.best_config?.closure_score_gain_vs_direct)} />
        <MetricCard label="扫描最优 novel 增益" value={fmt(scanPayload?.best_config?.novel_concept_gain_vs_direct)} />
        <MetricCard label="最优更新率 (family / offset)" value={`${scanPayload?.best_config?.family_alpha ?? '-'} / ${scanPayload?.best_config?.offset_alpha ?? '-'}`} />
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 连续流环境四个核心指标</div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={bars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip
              formatter={(value) => [fmt(value), 'score']}
              contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }}
            />
            <Legend />
            <Bar dataKey="direct" name="direct_stream" fill="#64748b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="baseShared" name="shared_offset_stream" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
            <Bar dataKey="tuned" name="tuned_shared_offset" fill="#16a34a" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            第一版连续流环境是负闭环：旧概念稳定和漂移鲁棒性有增益，但新概念接入掉了。更新律扫描后，`closure_score_gain` 已经翻正，说明当前问题更像更新区间问题，不是主结构立刻失效。
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 下一步</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            下一步不该再停在感知流本身，而是把动作回路、长期状态和自主纠错接进来，测试接地闭环能不能过渡到长期代理闭环。
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            {scanPayload?.project_readout?.next_question || '-'}
          </div>
        </div>
      </div>
    </div>
  );
}
