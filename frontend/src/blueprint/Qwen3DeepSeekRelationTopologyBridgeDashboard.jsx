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
import samplePayload from './data/qwen3_deepseek7b_relation_topology_boundary_bridge_sample.json';

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

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`;
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

export default function Qwen3DeepSeekRelationTopologyBridgeDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const models = useMemo(() => modelOptions(payload), [payload]);
  const [selectedModel, setSelectedModel] = useState(models[0] || 'qwen3_4b');

  const modelRow = payload?.models?.[selectedModel];
  const summary = modelRow?.global_summary || {};
  const relations = modelRow?.relations || {};

  const classMeanRows = useMemo(
    () =>
      Object.entries(summary?.classification_bridge_mean || {}).map(([cls, value]) => ({
        cls,
        label: clsLabel(cls),
        bridge_mean: Number(value || 0),
        fill: clsColor(cls),
      })),
    [summary]
  );

  const relationRows = useMemo(
    () =>
      Object.entries(relations)
        .map(([relation, row]) => ({
          relation,
          classification: row.classification,
          classificationLabel: clsLabel(row.classification),
          bridge_score: Number(row.bridge_score || 0),
          topology_compactness: Number(row.topology_compactness || 0),
          endpoint_margin_mean: Number(row.endpoint_margin_mean || 0),
          endpoint_support_rate: Number(row.endpoint_support_rate || 0),
          top4_bridge_share_in_top20: Number(row.top4_bridge_share_in_top20 || 0),
          top8_bridge_share_in_top20: Number(row.top8_bridge_share_in_top20 || 0),
          layer_cluster_margin: Number(row.layer_cluster_margin || 0),
          minimal_stronger_than_control_k: row.minimal_stronger_than_control_k,
          endpoint_families: row.endpoint_families || [],
          endpoint_words: row.endpoint_words || [],
        }))
        .sort((a, b) => b.bridge_score - a.bridge_score),
    [relations]
  );

  const barRows = useMemo(
    () =>
      relationRows.map((row) => ({
        relation: row.relation,
        bridge_score: row.bridge_score,
        topology_compactness: row.topology_compactness,
        endpoint_margin_mean: row.endpoint_margin_mean,
        fill: clsColor(row.classification),
      })),
    [relationRows]
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
      setError(`关系拓扑-边界桥接 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(16,185,129,0.14), transparent 28%), radial-gradient(circle at top right, rgba(56,189,248,0.12), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(16,185,129,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>Qwen3 / DeepSeek7B 关系拓扑-边界桥接</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把关系端点 family 的拓扑支持、头群集中度和边界分型放到同一张图里，直接回答为什么有些关系能收缩成紧致边界，而有些关系长期停在层簇或分布式形态。
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 0.95fr) minmax(0, 1.05fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 边界类型的平均桥接分数</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={classMeanRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="bridge_mean" name="平均桥接分数" radius={[4, 4, 0, 0]}>
                {classMeanRows.map((entry) => (
                  <Cell key={entry.cls} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ marginTop: '8px', color: '#cbd5e1', fontSize: '11px', lineHeight: '1.8' }}>
            {classMeanRows.map((row) => `${row.label}: ${fmt(row.bridge_mean)}`).join(' / ') || '-'}
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 关系桥接分数排序</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={barRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="relation" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="bridge_score" name="bridge_score" radius={[4, 4, 0, 0]}>
                {barRows.map((entry) => (
                  <Cell key={entry.relation} fill={entry.fill} />
                ))}
              </Bar>
              <Bar dataKey="topology_compactness" name="topology_compactness" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gap: '8px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '2px' }}>3. 逐关系桥接明细</div>
        {relationRows.map((row) => (
          <div
            key={row.relation}
            style={{
              border: '1px solid rgba(148,163,184,0.18)',
              borderRadius: '12px',
              padding: '10px',
              background: 'rgba(15,23,42,0.38)',
              color: '#cbd5e1',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
              <div style={{ color: '#f8fafc', fontSize: '13px', fontWeight: 'bold' }}>{row.relation}</div>
              <div style={{ color: clsColor(row.classification), fontSize: '11px', fontWeight: 'bold' }}>{row.classificationLabel}</div>
            </div>
            <div style={{ marginTop: '8px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '8px', fontSize: '11px' }}>
              <div>{`bridge_score: ${fmt(row.bridge_score)}`}</div>
              <div>{`topology_compactness: ${fmt(row.topology_compactness)}`}</div>
              <div>{`endpoint_margin_mean: ${fmt(row.endpoint_margin_mean)}`}</div>
              <div>{`endpoint_support_rate: ${pct(row.endpoint_support_rate)}`}</div>
              <div>{`top4/top20: ${pct(row.top4_bridge_share_in_top20)}`}</div>
              <div>{`top8/top20: ${pct(row.top8_bridge_share_in_top20)}`}</div>
              <div>{`layer_cluster_margin: ${fmt(row.layer_cluster_margin)}`}</div>
              <div>{`最小边界 k*: ${row.minimal_stronger_than_control_k ?? 'none'}`}</div>
            </div>
            <div style={{ marginTop: '8px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
              {`端点 family: ${row.endpoint_families.join(' / ') || '-'} ｜ 端点词: ${row.endpoint_words.join(' / ') || '-'}`}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
