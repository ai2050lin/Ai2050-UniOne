import React, { useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import samplePayload from './data/relation_protocol_head_group_causal_sample.json';

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

export default function RelationProtocolHeadGroupCausalDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const models = useMemo(() => modelOptions(payload), [payload]);
  const [selectedModel, setSelectedModel] = useState(models[0] || 'gpt2');
  const modelRow = payload?.models?.[selectedModel];
  const globalSummary = modelRow?.global_summary || {};
  const groupSize = modelRow?.meta?.group_size;

  const relationBars = useMemo(() => {
    const rows = modelRow?.relations || {};
    return Object.entries(rows).map(([relation, value]) => ({
      relation,
      top_group_collapse_ratio: Number(value?.summary?.top_group_collapse_ratio || 0),
      control_group_collapse_ratio: Number(value?.summary?.control_group_collapse_ratio || 0),
      causal_margin: Number(value?.summary?.causal_margin || 0),
    }));
  }, [modelRow]);

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
      setError(`关系协议头群因果 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(245,158,11,0.12), transparent 28%), radial-gradient(circle at top right, rgba(56,189,248,0.1), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(251,191,36,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>关系协议头群因果验证</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            评估 `top-{groupSize || '?'}` 头群联合消融是否比同层对照群更能打塌 `TT` 协议。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} style={{ background: 'rgba(15,23,42,0.9)', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px' }}>
            {models.map((item) => (
              <option key={item} value={item}>
                {item}
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均头群塌缩率</div>
          <div style={{ color: '#fcd34d', fontSize: '20px', fontWeight: 'bold' }}>{pct(globalSummary.mean_top_group_collapse_ratio)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均对照群塌缩率</div>
          <div style={{ color: '#7dd3fc', fontSize: '20px', fontWeight: 'bold' }}>{pct(globalSummary.mean_control_group_collapse_ratio)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均因果边际</div>
          <div style={{ color: '#fde68a', fontSize: '20px', fontWeight: 'bold' }}>{fmt(globalSummary.mean_causal_margin)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>优于对照的关系数</div>
          <div style={{ color: '#86efac', fontSize: '20px', fontWeight: 'bold' }}>{globalSummary.stronger_than_control_count ?? '-'}</div>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 各关系族头群塌缩率</div>
        <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
          若黄柱系统性高于蓝柱，说明小头群已经具备更强因果性；若仍不明显，说明协议实现比 `top-{groupSize || '?'}` 头群更分散。
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={relationBars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="relation" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
            <Bar dataKey="top_group_collapse_ratio" name="头群塌缩率" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="control_group_collapse_ratio" name="对照群塌缩率" fill="#38bdf8" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
