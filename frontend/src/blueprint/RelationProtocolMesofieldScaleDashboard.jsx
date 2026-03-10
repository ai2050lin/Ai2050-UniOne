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
import samplePayload from './data/relation_protocol_mesofield_scale_sample.json';

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

function heatColor(value) {
  const v = Math.max(-0.5, Math.min(0.5, Number(value || 0)));
  if (v >= 0) {
    const alpha = 0.08 + (v / 0.5) * 0.5;
    return `rgba(34, 197, 94, ${alpha.toFixed(3)})`;
  }
  const alpha = 0.08 + (Math.abs(v) / 0.5) * 0.5;
  return `rgba(248, 113, 113, ${alpha.toFixed(3)})`;
}

export default function RelationProtocolMesofieldScaleDashboard() {
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
  const kValues = Array.isArray(globalSummary?.k_values) ? globalSummary.k_values : [];

  const trendData = useMemo(
    () =>
      kValues.map((k) => {
        const key = String(k);
        return {
          k: `k=${key}`,
          top_group_collapse_ratio: Number(globalSummary?.mean_top_group_collapse_ratio_by_k?.[key] || 0),
          control_group_collapse_ratio: Number(globalSummary?.mean_control_group_collapse_ratio_by_k?.[key] || 0),
          causal_margin: Number(globalSummary?.mean_causal_margin_by_k?.[key] || 0),
          stronger_count: Number(globalSummary?.stronger_than_control_count_by_k?.[key] || 0),
        };
      }),
    [globalSummary, kValues]
  );

  const heatRows = useMemo(() => {
    const rows = modelRow?.relations || {};
    return Object.entries(rows).map(([relation, value]) => ({
      relation,
      values: kValues.map((k) => {
        const key = String(k);
        const summary = value?.k_scan?.[key]?.summary || {};
        return {
          k: key,
          causal_margin: Number(summary?.causal_margin || 0),
          top_group_collapse_ratio: Number(summary?.top_group_collapse_ratio || 0),
          control_group_collapse_ratio: Number(summary?.control_group_collapse_ratio || 0),
        };
      }),
    }));
  }, [kValues, modelRow]);

  const layerClusterBars = useMemo(() => {
    const summary = relationRow?.layer_cluster_scan?.summary || {};
    return [
      {
        name: '中观层簇',
        collapse_ratio: Number(summary?.top_group_collapse_ratio || 0),
      },
      {
        name: '对照层簇',
        collapse_ratio: Number(summary?.control_group_collapse_ratio || 0),
      },
    ];
  }, [relationRow]);

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
      setSelectedModel(nextModel);
      setSelectedRelation(nextRelations[0] || '');
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`中观场规模扫描 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    const nextModels = modelOptions(samplePayload);
    const nextModel = nextModels[0] || '';
    const nextRelations = relationOptions(samplePayload, nextModel);
    setPayload(samplePayload);
    setSelectedModel(nextModel);
    setSelectedRelation(nextRelations[0] || '');
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
          'radial-gradient(circle at top left, rgba(251,191,36,0.12), transparent 28%), radial-gradient(circle at top right, rgba(16,185,129,0.1), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(250,204,21,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>关系协议中观场规模扫描</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            主视图回答三个问题：`top-k` 是否出现统一最小因果规模、不同关系族的最小规模是否一致、层簇级消融是否比小头群更像真正的中观场。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <select
            value={selectedModel}
            onChange={(event) => {
              const nextModel = event.target.value;
              const nextRelations = relationOptions(payload, nextModel);
              setSelectedModel(nextModel);
              setSelectedRelation(nextRelations[0] || '');
            }}
            style={{ background: 'rgba(15,23,42,0.9)', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px' }}
          >
            {models.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
          <select
            value={selectedRelation}
            onChange={(event) => setSelectedRelation(event.target.value)}
            style={{ background: 'rgba(15,23,42,0.9)', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px' }}
          >
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>层簇大小</div>
          <div style={{ color: '#fde68a', fontSize: '20px', fontWeight: 'bold' }}>{globalSummary.layer_cluster_size ?? '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>层簇平均塌缩率</div>
          <div style={{ color: '#fbbf24', fontSize: '20px', fontWeight: 'bold' }}>{pct(globalSummary.mean_layer_cluster_collapse_ratio)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>层簇平均因果边际</div>
          <div style={{ color: '#86efac', fontSize: '20px', fontWeight: 'bold' }}>{fmt(globalSummary.mean_layer_cluster_margin)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>层簇优于对照的关系数</div>
          <div style={{ color: '#7dd3fc', fontSize: '20px', fontWeight: 'bold' }}>{globalSummary.layer_cluster_stronger_than_control_count ?? '-'}</div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1.1fr) minmax(320px, 0.9fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. k 规模曲线</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
            黄色看 `top-k` 头群塌缩率，蓝色看同规模对照群塌缩率，绿色看平均 `causal_margin`。如果绿色没有随 k 单调抬升，就说明不存在统一的固定最小规模。
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={trendData} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="k" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="top_group_collapse_ratio" name="top-k 塌缩率" stroke="#fbbf24" strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="control_group_collapse_ratio" name="对照塌缩率" stroke="#38bdf8" strokeWidth={2.2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="causal_margin" name="平均因果边际" stroke="#22c55e" strokeWidth={2.2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ display: 'grid', gap: '12px' }}>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 选中关系的层簇对比</div>
            <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
              <div>{`关系族: ${selectedRelation}`}</div>
              <div>{`最小正边际 k: ${relationRow?.mesofield_summary?.minimal_positive_margin_k ?? '-'}`}</div>
              <div>{`最小优于对照的 k: ${relationRow?.mesofield_summary?.minimal_stronger_than_control_k ?? '-'}`}</div>
              <div>{`层簇边际: ${fmt(relationRow?.mesofield_summary?.layer_cluster_margin)}`}</div>
              <div>{`中观层簇: ${(relationRow?.layer_cluster_scan?.top_layers || []).map((x) => `L${x}`).join(', ') || '-'}`}</div>
              <div>{`对照层簇: ${(relationRow?.layer_cluster_scan?.control_layers || []).map((x) => `L${x}`).join(', ') || '-'}`}</div>
            </div>
            <div style={{ marginTop: '10px' }}>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={layerClusterBars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
                  <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                  <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                  <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
                  <Bar dataKey="collapse_ratio" name="塌缩率" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. k 扫描摘要</div>
            <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
              {trendData.map((row) => (
                <div key={row.k}>{`${row.k}: top=${fmt(row.top_group_collapse_ratio)}, ctrl=${fmt(row.control_group_collapse_ratio)}, margin=${fmt(row.causal_margin)}, stronger=${row.stronger_count}`}</div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 关系 x k 热图</div>
        <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
          单元格颜色表示 `causal_margin`。绿色越深，说明该关系在对应 `k` 上越像真正的因果规模；红色越深，说明该规模仍不稳定或被对照压过。
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', color: '#e2e8f0' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.2)' }}>关系</th>
                {kValues.map((k) => (
                  <th key={k} style={{ textAlign: 'center', padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.2)' }}>
                    {`k=${k}`}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {heatRows.map((row) => (
                <tr key={row.relation}>
                  <td style={{ padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.12)', color: '#cbd5e1' }}>{row.relation}</td>
                  {row.values.map((cell) => (
                    <td
                      key={`${row.relation}-${cell.k}`}
                      style={{
                        padding: '8px',
                        textAlign: 'center',
                        borderBottom: '1px solid rgba(148,163,184,0.12)',
                        background: heatColor(cell.causal_margin),
                      }}
                    >
                      <div>{fmt(cell.causal_margin, 3)}</div>
                      <div style={{ color: '#cbd5e1', fontSize: '10px', marginTop: '2px' }}>{`${pct(cell.top_group_collapse_ratio)} / ${pct(cell.control_group_collapse_ratio)}`}</div>
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
