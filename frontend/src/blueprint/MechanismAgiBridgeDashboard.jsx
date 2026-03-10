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
import samplePayload from './data/mechanism_agi_bridge_sample.json';

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

function fieldShapeLabel(name) {
  if (name === 'compact_mesofield') return '紧致中观场';
  if (name === 'layer_cluster_mesofield') return '层簇中观场';
  if (name === 'distributed_mesofield') return '分布式中观场';
  return name || '-';
}

export default function MechanismAgiBridgeDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const models = useMemo(() => modelOptions(payload), [payload]);
  const [selectedModel, setSelectedModel] = useState(models[0] || 'gpt2');

  const modelRow = payload?.models?.[selectedModel] || {};
  const toyRow = payload?.toy_bridge || {};
  const realRow = payload?.real_bridge || {};
  const ranking = Array.isArray(payload?.ranking) ? payload.ranking : [];
  const conclusion = payload?.global_conclusion || {};

  const scoreRows = useMemo(
    () => [
      { metric: 'G 可预测性', value: Number(modelRow?.gate?.gate_predictability || 0) },
      { metric: '线性增益', value: Number(modelRow?.gate?.linear_gain_norm || 0) },
      { metric: '非线性增益', value: Number(modelRow?.gate?.nonlinear_gain_norm || 0) },
      { metric: '边界可定界性', value: Number(modelRow?.field?.boundaryability_score || 0) },
      { metric: 'toy 闭环', value: Number(modelRow?.bridge?.toy_closure_score || 0) },
      { metric: '真实多步闭环', value: Number(modelRow?.bridge?.real_closure_score || 0) },
      { metric: 'AGI 桥接', value: Number(modelRow?.bridge?.agi_bridge_score || 0) },
    ],
    [modelRow]
  );

  const fieldRows = useMemo(
    () => [
      {
        group: selectedModel,
        紧致边界: Number(modelRow?.field?.compact_ratio || 0),
        仅层簇边界: Number(modelRow?.field?.layer_cluster_ratio || 0),
        分布式无边界: Number(modelRow?.field?.distributed_ratio || 0),
      },
    ],
    [modelRow, selectedModel]
  );

  const rankingRows = useMemo(
    () =>
      ranking.map((row) => ({
        model: row.model_name,
        agi_bridge: Number(row.agi_bridge_score || 0),
        mechanism: Number(row.mechanism_score || 0),
      })),
    [ranking]
  );

  const closureRows = useMemo(
    () => [
      { metric: 'toy 闭环分数', value: Number(toyRow.toy_closure_score || 0) },
      { metric: '真实闭环分数', value: Number(realRow.real_closure_score || 0) },
      { metric: '真实闭环增益', value: Number(realRow.score_gain || 0) },
      { metric: '真实保留收益', value: Number(realRow.retention_gain || 0) },
      { metric: '真实回合收益', value: Number(realRow.overall_episode_gain || 0) },
    ],
    [toyRow, realRow]
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
      setError(`机制到 AGI 桥接 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(14,165,233,0.12), transparent 28%), radial-gradient(circle at top right, rgba(16,185,129,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(14,165,233,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>机制到 AGI 桥接总览</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把 `G` 递推、协议场边界、toy 闭环和真实多步闭环压到同一张图里，直接看解释力是否开始转化成真实任务收益。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>AGI 桥接分数</div>
          <div style={{ color: '#7dd3fc', fontSize: '20px', fontWeight: 'bold' }}>{fmt(modelRow?.bridge?.agi_bridge_score)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>机制分数</div>
          <div style={{ color: '#86efac', fontSize: '20px', fontWeight: 'bold' }}>{fmt(modelRow?.bridge?.mechanism_score)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>真实多步闭环</div>
          <div style={{ color: '#fcd34d', fontSize: '20px', fontWeight: 'bold' }}>{fmt(modelRow?.bridge?.real_closure_score)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>toy 闭环</div>
          <div style={{ color: '#c4b5fd', fontSize: '20px', fontWeight: 'bold' }}>{fmt(modelRow?.bridge?.toy_closure_score)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>协议场形态</div>
          <div style={{ color: '#f8fafc', fontSize: '16px', fontWeight: 'bold' }}>{fieldShapeLabel(modelRow?.field?.field_shape)}</div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1.1fr) minmax(320px, 0.9fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 机制到能力的桥接分量</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={scoreRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-20} textAnchor="end" height={58} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="value" name="归一化分数" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ display: 'grid', gap: '12px' }}>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 协议场边界组成</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={fieldRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
                <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
                <XAxis dataKey="group" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
                <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
                <Legend />
                <Bar dataKey="紧致边界" stackId="field" fill="#22c55e" />
                <Bar dataKey="仅层簇边界" stackId="field" fill="#f59e0b" />
                <Bar dataKey="分布式无边界" stackId="field" fill="#f87171" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 代理闭环 vs 真实闭环</div>
            <div style={{ display: 'grid', gap: '8px' }}>
              {closureRows.map((row) => (
                <div key={row.metric} style={{ display: 'flex', justifyContent: 'space-between', color: '#cbd5e1', fontSize: '11px', border: '1px solid rgba(148,163,184,0.18)', borderRadius: '10px', padding: '8px 10px' }}>
                  <span>{row.metric}</span>
                  <span style={{ color: '#f8fafc', fontWeight: 'bold' }}>{fmt(row.value)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(320px, 0.9fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 模型桥接排序</div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={rankingRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="agi_bridge" name="AGI 桥接" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
              <Bar dataKey="mechanism" name="机制分数" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>5. 结论与下一步</div>
          <div style={{ color: '#e2e8f0', fontSize: '12px', lineHeight: '1.8', marginBottom: '10px' }}>
            {conclusion?.statement || '-'}
          </div>
          <div style={{ display: 'grid', gap: '8px' }}>
            {(conclusion?.why || []).map((item) => (
              <div key={item} style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7', border: '1px solid rgba(148,163,184,0.18)', borderRadius: '10px', padding: '8px 10px' }}>
                {item}
              </div>
            ))}
          </div>
          <div style={{ marginTop: '10px', display: 'grid', gap: '8px' }}>
            {(modelRow?.recommended_next_steps || []).map((item) => (
              <div key={item} style={{ color: '#dcfce7', fontSize: '11px', lineHeight: '1.7', background: 'rgba(34,197,94,0.12)', border: '1px solid rgba(34,197,94,0.22)', borderRadius: '10px', padding: '8px 10px' }}>
                {item}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
