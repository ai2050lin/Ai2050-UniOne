import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/relation_coupling_trace_sample.json';

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

function relationOptions(payload, modelName) {
  return Object.keys(payload?.models?.[modelName]?.relations || {});
}

function stageRows(stageMap) {
  return ['early', 'mid', 'late'].map((name) => ({
    stage: name,
    layer: stageMap?.[name]?.layer,
    value: Number(stageMap?.[name]?.value || 0),
  }));
}

function protocolLabel(protocol) {
  if (protocol === 'tt') return 'T-T 主导';
  if (protocol === 'ht') return 'H-T 主导';
  if (protocol === 'hh') return 'H-H 主导';
  return protocol || '-';
}

function protocolColor(protocol) {
  if (protocol === 'tt') return '#22c55e';
  if (protocol === 'ht') return '#38bdf8';
  if (protocol === 'hh') return '#f472b6';
  return '#94a3b8';
}

export default function RelationCouplingTraceDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const models = useMemo(() => modelOptions(payload), [payload]);
  const [selectedModel, setSelectedModel] = useState(models[0] || 'gpt2');
  const relations = useMemo(() => relationOptions(payload, selectedModel), [payload, selectedModel]);
  const [selectedRelation, setSelectedRelation] = useState(relations[0] || 'gender');

  const modelRow = payload?.models?.[selectedModel];
  const relationRow = modelRow?.relations?.[selectedRelation];
  const summary = relationRow?.summary || {};
  const globalSummary = modelRow?.global_summary || {};
  const atlasRows = Array.isArray(modelRow?.atlas_rows) ? modelRow.atlas_rows : [];

  const lineData = useMemo(() => {
    if (!relationRow) return [];
    const n = relationRow.bridge_ht_by_layer?.length || 0;
    return Array.from({ length: n }, (_, layer) => ({
      layer,
      endpoint_repr_basis: Number(relationRow.endpoint_repr_basis_by_layer?.[layer] || 0),
      endpoint_topo_basis: Number(relationRow.endpoint_topo_basis_by_layer?.[layer] || 0),
      relation_align_repr: Number(relationRow.relation_align_repr_by_layer?.[layer] || 0),
      relation_align_topo: Number(relationRow.relation_align_topo_by_layer?.[layer] || 0),
      bridge_ht: Number(relationRow.bridge_ht_by_layer?.[layer] || 0),
      bridge_hh: Number(relationRow.bridge_hh_by_layer?.[layer] || 0),
      bridge_tt: Number(relationRow.bridge_tt_by_layer?.[layer] || 0),
    }));
  }, [relationRow]);

  const stageData = useMemo(() => stageRows(summary.bridge_stage_ht), [summary]);

  const atlasBarData = useMemo(
    () =>
      atlasRows.map((row) => ({
        relation: row.relation,
        max_bridge_tt: Number(row.max_bridge_tt || 0),
        max_bridge_ht: Number(row.max_bridge_ht || 0),
        max_bridge_hh: Number(row.max_bridge_hh || 0),
        topo_dominant_ratio: Number(row.topo_dominant_ratio || 0),
        protocol: row.protocol,
        fill: protocolColor(row.protocol),
      })),
    [atlasRows]
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
      const nextModel = nextModels[0] || '';
      const nextRelations = relationOptions(parsed, nextModel);
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setSelectedModel(nextModel);
      setSelectedRelation(nextRelations[0] || '');
      setError('');
    } catch (err) {
      setError(`关系耦合 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    const nextModels = modelOptions(samplePayload);
    const nextModel = nextModels[0] || '';
    const nextRelations = relationOptions(samplePayload, nextModel);
    setPayload(samplePayload);
    setSource('内置样例');
    setSelectedModel(nextModel);
    setSelectedRelation(nextRelations[0] || '');
    setError('');
  }

  function onChangeModel(event) {
    const nextModel = event.target.value;
    const nextRelations = relationOptions(payload, nextModel);
    setSelectedModel(nextModel);
    setSelectedRelation(nextRelations[0] || '');
  }

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(56,189,248,0.14), transparent 30%), radial-gradient(circle at top right, rgba(16,185,129,0.1), transparent 26%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(125,211,252,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>关系耦合路径看板</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            展示 6 类关系族如何在逐层处理中，把概念骨架与拓扑关系场耦合起来。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <select value={selectedModel} onChange={onChangeModel} style={{ background: 'rgba(15,23,42,0.9)', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px' }}>
            {models.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
          <select value={selectedRelation} onChange={(event) => setSelectedRelation(event.target.value)} style={{ background: 'rgba(15,23,42,0.9)', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px' }}>
            {relations.map((item) => (
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>关系族数量</div>
          <div style={{ color: '#e0f2fe', fontSize: '20px', fontWeight: 'bold' }}>{globalSummary.relation_count ?? '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均 T-T 峰值</div>
          <div style={{ color: '#86efac', fontSize: '20px', fontWeight: 'bold' }}>{fmt(globalSummary.mean_max_bridge_tt)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均 H-T 峰值</div>
          <div style={{ color: '#67e8f9', fontSize: '20px', fontWeight: 'bold' }}>{fmt(globalSummary.mean_max_bridge_ht)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均 H-H 峰值</div>
          <div style={{ color: '#f9a8d4', fontSize: '20px', fontWeight: 'bold' }}>{fmt(globalSummary.mean_max_bridge_hh)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均拓扑主导占比</div>
          <div style={{ color: '#fde68a', fontSize: '20px', fontWeight: 'bold' }}>{pct(globalSummary.mean_topo_dominant_ratio)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>当前协议</div>
          <div style={{ color: protocolColor(summary.coupling_protocol), fontSize: '20px', fontWeight: 'bold' }}>{protocolLabel(summary.coupling_protocol)}</div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1.1fr) minmax(320px, 0.9fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 六类关系族 atlas</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
            每个关系族同时显示 `T-T / H-T / H-H` 三种桥接峰值。若 `T-T` 普遍最高，说明关系场整体拓扑化。
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={atlasBarData} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="relation" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Bar dataKey="max_bridge_tt" name="T-T 峰值" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="max_bridge_ht" name="H-T 峰值" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="max_bridge_hh" name="H-H 峰值" fill="#f472b6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ display: 'grid', gap: '12px' }}>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 当前关系族摘要</div>
            <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
              <div>{`关系族: ${selectedRelation}`}</div>
              <div>{`样例对: ${relationRow?.pairs?.map((pair) => pair.join('→')).join(' / ') || '-'}`}</div>
              <div>{`最强 H-T 层: L${summary.best_bridge_ht_layers?.[0] ?? '-'}`}</div>
              <div>{`最强 T-T 层: L${summary.best_bridge_tt_layers?.[0] ?? '-'}`}</div>
              <div>{`最强表征对齐层: L${summary.best_relation_repr_layers?.[0] ?? '-'}`}</div>
              <div>{`拓扑主导层占比: ${pct(summary.topo_dominant_ratio)}`}</div>
              <div>{`协议判定: ${protocolLabel(summary.coupling_protocol)}`}</div>
            </div>
          </div>

          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 阶段峰值</div>
            <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
              观察桥接峰值落在早期、中期还是后期，用来判断关系何时开始接管概念骨架。
            </div>
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={stageData} margin={{ top: 10, right: 8, left: 0, bottom: 10 }}>
                <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
                <XAxis dataKey="stage" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
                <Bar dataKey="value" name="H-T 阶段峰值" fill="#22d3ee" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 当前关系族逐层曲线</div>
        <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
          蓝线是端点概念的表征基底稳定度，绿线是拓扑关系对齐，青线是 H-T 桥接，浅绿线是 T-T 桥接，粉线是表征关系对齐。
        </div>
        <ResponsiveContainer width="100%" height={330}>
          <LineChart data={lineData} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="layer" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
            <Legend wrapperStyle={{ fontSize: '11px' }} />
            <Line type="monotone" dataKey="endpoint_repr_basis" name="端点表征基底" stroke="#60a5fa" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="relation_align_topo" name="拓扑关系对齐" stroke="#22c55e" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="relation_align_repr" name="表征关系对齐" stroke="#f472b6" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="bridge_ht" name="H-T 桥接" stroke="#22d3ee" strokeWidth={2.4} dot={false} />
            <Line type="monotone" dataKey="bridge_tt" name="T-T 桥接" stroke="#a3e635" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
