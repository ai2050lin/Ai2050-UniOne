import React, { useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import samplePayload from './data/relation_protocol_head_atlas_sample.json';

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

export default function RelationProtocolHeadAtlasDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const models = useMemo(() => modelOptions(payload), [payload]);
  const [selectedModel, setSelectedModel] = useState(models[0] || 'gpt2');
  const relations = useMemo(() => relationOptions(payload, selectedModel), [payload, selectedModel]);
  const [selectedRelation, setSelectedRelation] = useState(relations[0] || 'gender');

  const modelRow = payload?.models?.[selectedModel];
  const globalSummary = modelRow?.global_summary || {};
  const relationRow = modelRow?.relations?.[selectedRelation];
  const sharedHeads = Array.isArray(modelRow?.shared_heads) ? modelRow.shared_heads.slice(0, 8) : [];

  const topHeadBars = useMemo(() => {
    const rows = Array.isArray(relationRow?.top_heads) ? relationRow.top_heads.slice(0, 8) : [];
    return rows.map((row) => ({
      name: `L${row.layer}H${row.head}`,
      bridge_tt: Number(row.bridge_tt || 0),
      endpoint_topo_basis: Number(row.endpoint_topo_basis || 0),
      relation_align_topo: Number(row.relation_align_topo || 0),
    }));
  }, [relationRow]);

  const sharedHeadBars = useMemo(
    () =>
      sharedHeads.map((row) => ({
        name: `L${row.layer}H${row.head}`,
        frequency: Number(row.frequency || 0),
      })),
    [sharedHeads]
  );

  const overlapRows = useMemo(() => {
    const matrix = modelRow?.top_head_overlap_jaccard || {};
    const keys = Object.keys(matrix);
    return keys.map((name) => ({
      relation: name,
      values: keys.map((other) => ({
        other,
        value: Number(matrix?.[name]?.[other] || 0),
      })),
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
      const nextModel = nextModels[0] || '';
      const nextRelations = relationOptions(parsed, nextModel);
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setSelectedModel(nextModel);
      setSelectedRelation(nextRelations[0] || '');
      setError('');
    } catch (err) {
      setError(`关系协议头 atlas JSON 导入失败: ${err?.message || '未知错误'}`);
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

  function overlapColor(value) {
    const alpha = 0.1 + Math.max(0, Math.min(1, value)) * 0.55;
    return `rgba(34, 197, 94, ${alpha.toFixed(3)})`;
  }

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(34,197,94,0.14), transparent 28%), radial-gradient(circle at top right, rgba(56,189,248,0.1), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(134,239,172,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>关系协议头级 atlas</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            展示 6 类关系族在头级别的承载分布，用来判断“统一协议”是由共享头完成，还是由专职头群完成。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <select value={selectedModel} onChange={(event) => {
            const nextModel = event.target.value;
            const nextRelations = relationOptions(payload, nextModel);
            setSelectedModel(nextModel);
            setSelectedRelation(nextRelations[0] || '');
          }} style={{ background: 'rgba(15,23,42,0.9)', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px' }}>
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>top-k</div>
          <div style={{ color: '#e0f2fe', fontSize: '20px', fontWeight: 'bold' }}>{globalSummary.top_k ?? '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>唯一 top 头数</div>
          <div style={{ color: '#86efac', fontSize: '20px', fontWeight: 'bold' }}>{globalSummary.unique_top_head_count ?? '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>专职头数</div>
          <div style={{ color: '#fcd34d', fontSize: '20px', fontWeight: 'bold' }}>{globalSummary.specialized_relation_count ?? '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>复用头数</div>
          <div style={{ color: '#67e8f9', fontSize: '20px', fontWeight: 'bold' }}>{globalSummary.reused_relation_count ?? '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最共享头</div>
          <div style={{ color: '#f9a8d4', fontSize: '20px', fontWeight: 'bold' }}>
            {globalSummary.most_shared_head ? `L${globalSummary.most_shared_head.layer}H${globalSummary.most_shared_head.head}` : '-'}
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(320px, 0.9fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 当前关系族 top 头</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
            柱高是 `bridge_tt`，代表这个头在当前关系族中的拓扑协议承载强度。
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={topHeadBars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="bridge_tt" name="bridge_tt" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ display: 'grid', gap: '12px' }}>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 当前关系族摘要</div>
            <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
              <div>{`关系族: ${selectedRelation}`}</div>
              <div>{`样例对: ${relationRow?.pairs?.map((pair) => pair.join('→')).join(' / ') || '-'}`}</div>
              <div>{`最佳头: ${relationRow?.summary?.best_head ? `L${relationRow.summary.best_head.layer}H${relationRow.summary.best_head.head}` : '-'}`}</div>
              <div>{`最大 bridge_tt: ${fmt(relationRow?.summary?.max_bridge_tt)}`}</div>
              <div>{`平均 bridge_tt: ${fmt(relationRow?.summary?.mean_bridge_tt)}`}</div>
            </div>
          </div>

          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 最共享头</div>
            <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
              如果这里频次很低，说明协议虽然统一，但头级载体仍然高度分工。
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={sharedHeadBars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
                <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" allowDecimals={false} />
                <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
                <Bar dataKey="frequency" name="复用频次" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 关系族之间的 top 头重叠矩阵</div>
        <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
          使用 top-k 头集合的 Jaccard 重叠。若大多数非对角项接近 0，说明是“统一协议 + 专职头群”。
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', color: '#e2e8f0' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.2)' }}>关系族</th>
                {overlapRows.map((row) => (
                  <th key={row.relation} style={{ textAlign: 'center', padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.2)' }}>
                    {row.relation}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overlapRows.map((row) => (
                <tr key={row.relation}>
                  <td style={{ padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.12)', color: '#cbd5e1' }}>{row.relation}</td>
                  {row.values.map((cell) => (
                    <td key={`${row.relation}-${cell.other}`} style={{ padding: '8px', textAlign: 'center', borderBottom: '1px solid rgba(148,163,184,0.12)', background: overlapColor(cell.value) }}>
                      {fmt(cell.value, 3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
