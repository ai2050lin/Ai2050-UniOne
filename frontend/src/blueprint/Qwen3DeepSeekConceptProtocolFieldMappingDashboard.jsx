import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/qwen3_deepseek7b_concept_protocol_field_mapping_sample.json';

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

function conceptOptions(payload, modelName) {
  return Object.keys(payload?.models?.[modelName]?.concepts || {});
}

function fieldColor(field) {
  if (field === 'fruit') return '#22c55e';
  if (field === 'animal') return '#38bdf8';
  if (field === 'abstract') return '#f59e0b';
  return '#94a3b8';
}

function usageColor(v) {
  const value = Math.max(0, Math.min(1, Number(v || 0)));
  const alpha = 0.08 + value * 0.6;
  return `rgba(34, 197, 94, ${alpha.toFixed(3)})`;
}

export default function Qwen3DeepSeekConceptProtocolFieldMappingDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const models = useMemo(() => modelOptions(payload), [payload]);
  const [selectedModel, setSelectedModel] = useState(models[0] || 'qwen3_4b');
  const concepts = useMemo(() => conceptOptions(payload, selectedModel), [payload, selectedModel]);
  const [selectedConcept, setSelectedConcept] = useState(concepts[0] || 'apple');

  const modelRow = payload?.models?.[selectedModel];
  const conceptRow = modelRow?.concepts?.[selectedConcept];
  const summary = conceptRow?.summary || {};
  const preferredField = summary?.preferred_field;
  const fieldRow = conceptRow?.field_scores?.[preferredField] || {};
  const massSummary = fieldRow?.mass_summary || {};
  const nHeads = Number(modelRow?.meta?.n_heads || 0);

  const fieldBars = useMemo(
    () =>
      Array.isArray(summary?.ranked_fields)
        ? summary.ranked_fields.map((row) => ({
            field: row.field,
            total_usage: Number(row.total_usage || 0),
            fill: fieldColor(row.field),
          }))
        : [],
    [summary]
  );

  const layerBars = useMemo(() => {
    const rows = Array.isArray(fieldRow?.layer_usage_by_layer) ? fieldRow.layer_usage_by_layer : [];
    return rows.map((value, layer) => ({
      layer: `L${layer}`,
      usage_score: Number(value || 0),
    }));
  }, [fieldRow]);

  const topHeadRows = useMemo(() => {
    const rows = Array.isArray(fieldRow?.top_heads) ? fieldRow.top_heads.slice(0, 12) : [];
    const maxUsage = Math.max(1e-12, ...rows.map((row) => Number(row.usage_score || 0)));
    return rows.map((row) => ({
      ...row,
      usage_ratio: Number(row.usage_score || 0) / maxUsage,
    }));
  }, [fieldRow]);

  const layerRegionRows = useMemo(() => {
    const rows = Array.isArray(fieldRow?.top_layers) ? fieldRow.top_layers.slice(0, 6) : [];
    const topSet = new Map((Array.isArray(fieldRow?.top_heads) ? fieldRow.top_heads : []).map((row) => [`${row.layer}-${row.head}`, row]));
    return rows.map((row) => ({
      layer: row.layer,
      heads: Array.from({ length: nHeads }, (_, head) => {
        const hit = topSet.get(`${row.layer}-${head}`);
        return {
          head,
          usage_score: Number(hit?.usage_score || 0),
          isTop: Boolean(hit),
        };
      }),
    }));
  }, [fieldRow, nHeads]);

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
      const nextConcepts = conceptOptions(parsed, nextModel);
      setPayload(parsed);
      setSelectedModel(nextModel);
      setSelectedConcept(nextConcepts[0] || '');
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`概念到协议场调用映射 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    const nextModels = modelOptions(samplePayload);
    const nextModel = nextModels[0] || '';
    const nextConcepts = conceptOptions(samplePayload, nextModel);
    setPayload(samplePayload);
    setSelectedModel(nextModel);
    setSelectedConcept(nextConcepts[0] || '');
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
          'radial-gradient(circle at top left, rgba(59,130,246,0.12), transparent 28%), radial-gradient(circle at top right, rgba(245,158,11,0.1), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(96,165,250,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>Qwen3 / DeepSeek7B 概念到协议场调用映射</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            展示 `U(c, tau, l, h)` 的可视化摘要，直接看概念最偏好进入哪个协议场、主要调用哪些层群和头群，以及这种调用是集中还是分布式。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <select
            value={selectedModel}
            onChange={(event) => {
              const nextModel = event.target.value;
              const nextConcepts = conceptOptions(payload, nextModel);
              setSelectedModel(nextModel);
              setSelectedConcept(nextConcepts[0] || '');
            }}
            style={{ background: 'rgba(15,23,42,0.9)', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px' }}
          >
            {models.map((item) => (
              <option key={item} value={item}>
                {MODEL_LABELS[item] || item}
              </option>
            ))}
          </select>
          <select
            value={selectedConcept}
            onChange={(event) => setSelectedConcept(event.target.value)}
            style={{ background: 'rgba(15,23,42,0.9)', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px' }}
          >
            {concepts.map((item) => (
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>首选协议场</div>
          <div style={{ color: fieldColor(preferredField), fontSize: '20px', fontWeight: 'bold' }}>{preferredField || '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>是否匹配真实协议场</div>
          <div style={{ color: summary?.preferred_field_matches_truth ? '#86efac' : '#fca5a5', fontSize: '20px', fontWeight: 'bold' }}>
            {summary?.preferred_field_matches_truth ? '是' : '否'}
          </div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>50% 质量所需头数</div>
          <div style={{ color: '#fde68a', fontSize: '20px', fontWeight: 'bold' }}>{massSummary.heads_for_50pct_mass ?? '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>80% 质量所需头数</div>
          <div style={{ color: '#fbbf24', fontSize: '20px', fontWeight: 'bold' }}>{massSummary.heads_for_80pct_mass ?? '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Top8 质量占比</div>
          <div style={{ color: '#67e8f9', fontSize: '20px', fontWeight: 'bold' }}>{pct(massSummary.top8_head_mass_ratio)}</div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(340px, 0.95fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 协议场偏好排序</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
            不是只看概念更像谁，而是看它在协议提示下，真正把哪一片协议场调起来。
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={fieldBars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="field" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="total_usage" name="total_usage" radius={[4, 4, 0, 0]}>
                {fieldBars.map((entry) => (
                  <Cell key={entry.field} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ display: 'grid', gap: '12px' }}>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 调用集中度摘要</div>
            <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
              <div>{`模型: ${MODEL_LABELS[selectedModel] || selectedModel}`}</div>
              <div>{`概念: ${selectedConcept}`}</div>
              <div>{`真实协议场: ${conceptRow?.true_field || '-'}`}</div>
              <div>{`首选协议场: ${preferredField || '-'}`}</div>
              <div>{`best_total_usage: ${fmt(summary?.best_total_usage)}`}</div>
              <div>{`margin_vs_second: ${fmt(summary?.margin_vs_second)}`}</div>
              <div>{`Top16 质量占比: ${pct(massSummary.top16_head_mass_ratio)}`}</div>
            </div>
          </div>

          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 主要层群</div>
            <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
              {(Array.isArray(fieldRow?.top_layers) ? fieldRow.top_layers.slice(0, 6) : []).map((row) => (
                <div key={row.layer}>{`L${row.layer}: usage=${fmt(row.usage_score)}`}</div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 首选协议场的逐层调用强度</div>
        <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
          这条图回答“调用是集中在少数层，还是沿很多层分布”。
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={layerBars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="layer" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
            <Bar dataKey="usage_score" name="usage_score" fill={fieldColor(preferredField)} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>5. 头群-层群区域图</div>
        <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
          这里只展开首选协议场的前 6 个主要层。绿色越深，表示该头在 `U(c, tau, l, h)` 中的调用强度越高。
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', color: '#e2e8f0' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.2)' }}>层</th>
                {Array.from({ length: nHeads }, (_, head) => (
                  <th key={head} style={{ textAlign: 'center', padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.2)' }}>
                    {`H${head}`}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {layerRegionRows.map((row) => (
                <tr key={row.layer}>
                  <td style={{ padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.12)', color: '#cbd5e1' }}>{`L${row.layer}`}</td>
                  {row.heads.map((cell) => (
                    <td
                      key={`${row.layer}-${cell.head}`}
                      style={{
                        padding: '8px',
                        textAlign: 'center',
                        borderBottom: '1px solid rgba(148,163,184,0.12)',
                        background: cell.isTop ? usageColor(Math.min(1, cell.usage_score / (topHeadRows[0]?.usage_score || 1e-12))) : 'transparent',
                        color: cell.isTop ? '#f8fafc' : '#64748b',
                        fontWeight: cell.isTop ? 700 : 400,
                      }}
                    >
                      {cell.isTop ? fmt(cell.usage_score, 3) : '·'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>6. 关键头明细</div>
        <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginBottom: '8px' }}>
          这里把前 12 个关键头拆成三个因子：总调用强度、基底选择性、协议激活差异。
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', color: '#e2e8f0' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.2)' }}>头</th>
                <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.2)' }}>usage</th>
                <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.2)' }}>fit</th>
                <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.2)' }}>fit_selectivity</th>
                <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.2)' }}>protocol_delta</th>
              </tr>
            </thead>
            <tbody>
              {topHeadRows.map((row) => (
                <tr key={`${row.layer}-${row.head}`}>
                  <td style={{ padding: '8px', borderBottom: '1px solid rgba(148,163,184,0.12)', color: '#cbd5e1' }}>{`L${row.layer}H${row.head}`}</td>
                  <td style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid rgba(148,163,184,0.12)' }}>{fmt(row.usage_score)}</td>
                  <td style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid rgba(148,163,184,0.12)' }}>{fmt(row.fit_score)}</td>
                  <td style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid rgba(148,163,184,0.12)' }}>{fmt(row.fit_selectivity)}</td>
                  <td style={{ padding: '8px', textAlign: 'right', borderBottom: '1px solid rgba(148,163,184,0.12)' }}>{fmt(row.protocol_delta)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
