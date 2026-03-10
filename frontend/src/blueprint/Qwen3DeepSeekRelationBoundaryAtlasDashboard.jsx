import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/qwen3_deepseek7b_relation_boundary_atlas_sample.json';

const MODEL_LABELS = {
  qwen3_4b: 'Qwen3-4B',
  deepseek_7b: 'DeepSeek-7B',
};

function isValidPayload(payload) {
  return Boolean(payload && payload.models && typeof payload.models === 'object');
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function modelOptions(payload) {
  return Object.keys(payload?.models || {});
}

function clsColor(name) {
  if (name === 'compact_boundary') return '#22c55e';
  if (name === 'mixed_boundary') return '#38bdf8';
  if (name === 'layer_cluster_only') return '#f59e0b';
  if (name === 'distributed_none') return '#f87171';
  return '#94a3b8';
}

function clsLabel(name) {
  if (name === 'compact_boundary') return '紧致边界';
  if (name === 'mixed_boundary') return '混合边界';
  if (name === 'layer_cluster_only') return '仅层簇边界';
  if (name === 'distributed_none') return '分布式无边界';
  return name;
}

export default function Qwen3DeepSeekRelationBoundaryAtlasDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const models = useMemo(() => modelOptions(payload), [payload]);
  const [selectedModel, setSelectedModel] = useState(models[0] || 'qwen3_4b');

  const modelRow = payload?.models?.[selectedModel];
  const summary = modelRow?.global_summary || {};
  const relations = modelRow?.relations || {};

  const histData = useMemo(
    () =>
      Object.entries(summary?.classification_histogram || {}).map(([key, value]) => ({
        cls: key,
        label: clsLabel(key),
        count: Number(value || 0),
        fill: clsColor(key),
      })),
    [summary]
  );

  const relationRows = useMemo(
    () =>
      Object.entries(relations).map(([name, row]) => ({
        relation: name,
        classification: row.classification,
        classificationLabel: clsLabel(row.classification),
        color: clsColor(row.classification),
        minK: row.minimal_stronger_than_control_k ?? row.minimal_positive_margin_k ?? null,
        layerMargin: Number(row.layer_cluster_margin || 0),
        bestK: row.best_k_by_causal_margin ?? '-',
      })),
    [relations]
  );

  const kTrend = useMemo(
    () =>
      Object.entries(summary?.mean_causal_margin_by_k || {}).map(([k, value]) => ({
        k: `k=${k}`,
        margin: Number(value || 0),
      })),
    [summary]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 models 字段');
      }
      const nextModels = modelOptions(parsed);
      setPayload(parsed);
      setSelectedModel(nextModels[0] || '');
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`关系边界图谱 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    const nextModels = modelOptions(samplePayload);
    setPayload(samplePayload);
    setSelectedModel(nextModels[0] || '');
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
          'radial-gradient(circle at top left, rgba(250,204,21,0.12), transparent 28%), radial-gradient(circle at top right, rgba(34,197,94,0.1), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(250,204,21,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>Qwen3 / DeepSeek7B 关系族边界图谱</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把六类关系协议分成紧致边界、混合边界、仅层簇边界和分布式无边界四类，直接比较两模型在关系协议实现形态上的差异。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <select
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value)}
            style={{ background: 'rgba(15,23,42,0.9)', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px' }}
          >
            {models.map((item) => (
              <option key={item} value={item}>
                {MODEL_LABELS[item] || item}
              </option>
            ))}
          </select>
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 0.9fr) minmax(0, 1.1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 边界类型直方图</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={histData} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="count" name="关系数" radius={[4, 4, 0, 0]}>
                {histData.map((entry) => (
                  <Cell key={entry.cls} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ marginTop: '8px', color: '#cbd5e1', fontSize: '11px', lineHeight: '1.8' }}>
            {histData.map((row) => `${row.label}: ${row.count}`).join(' / ') || '-'}
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 平均 causal_margin 随 k 变化</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={kTrend} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="k" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="margin" name="平均 causal_margin" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gap: '8px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '2px' }}>3. 逐关系分类表</div>
        {relationRows.map((row) => (
          <div
            key={row.relation}
            style={{
              display: 'grid',
              gridTemplateColumns: '120px 120px 110px 120px 100px',
              gap: '8px',
              alignItems: 'center',
              border: '1px solid rgba(148,163,184,0.18)',
              borderRadius: '10px',
              padding: '8px 10px',
              background: 'rgba(15,23,42,0.35)',
              color: '#cbd5e1',
              fontSize: '11px',
            }}
          >
            <div style={{ color: '#f8fafc', fontWeight: 'bold' }}>{row.relation}</div>
            <div style={{ color: row.color, fontWeight: 'bold' }}>{row.classificationLabel}</div>
            <div>{`最小 k=${row.minK ?? 'none'}`}</div>
            <div>{`层簇边际=${fmt(row.layerMargin)}`}</div>
            <div>{`best_k=${row.bestK}`}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
