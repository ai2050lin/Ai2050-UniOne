import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  ComposedChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const DEFAULT_PAYLOAD = {
  timestamp: '2026-03-07 13:22:49',
  model_id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
  dtype: 'float16',
  dims: ['color', 'size', 'text', 'sound'],
  dimension_stats: {
    color: { n_pairs: 6.0, mean_abs_alignment_to_axis: 0.4842621756180791, std_abs_alignment_to_axis: 0.10262651103129596 },
    size: { n_pairs: 6.0, mean_abs_alignment_to_axis: 0.5044693007456608, std_abs_alignment_to_axis: 0.063840576719053 },
    text: { n_pairs: 6.0, mean_abs_alignment_to_axis: 0.6356639343762372, std_abs_alignment_to_axis: 0.15483844445931486 },
    sound: { n_pairs: 6.0, mean_abs_alignment_to_axis: 0.4995869806309184, std_abs_alignment_to_axis: 0.06881507367437582 },
  },
  pairwise: [
    { pair: 'color__size', cosine: -0.022414995786708806, abs_cosine: 0.022414995786708806, signature_jaccard: 0.0, principal_similarity: 0.11202926188707352, principal_orthogonality: 0.8879707381129265 },
    { pair: 'color__text', cosine: -0.04004848647142088, abs_cosine: 0.04004848647142088, signature_jaccard: 0.0, principal_similarity: 0.11875274777412415, principal_orthogonality: 0.8812472522258759 },
    { pair: 'color__sound', cosine: 0.02513496626000455, abs_cosine: 0.02513496626000455, signature_jaccard: 0.003134796238244514, principal_similarity: 0.1535027027130127, principal_orthogonality: 0.8464972972869873 },
    { pair: 'size__text', cosine: 0.07854606074821492, abs_cosine: 0.07854606074821492, signature_jaccard: 0.003134796238244514, principal_similarity: 0.10129495710134506, principal_orthogonality: 0.8987050428986549 },
    { pair: 'size__sound', cosine: -0.03356022958822603, abs_cosine: 0.03356022958822603, signature_jaccard: 0.003134796238244514, principal_similarity: 0.21873661875724792, principal_orthogonality: 0.7812633812427521 },
    { pair: 'text__sound', cosine: -0.026793445079753755, abs_cosine: 0.026793445079753755, signature_jaccard: 0.0, principal_similarity: 0.13844598829746246, principal_orthogonality: 0.8615540117025375 },
  ],
  metrics: {
    mean_abs_pairwise_cosine: 0.03774969732238816,
    mean_signature_jaccard: 0.0015673981191222572,
    mean_principal_similarity: 0.14046037942171097,
    mean_principal_orthogonality: 0.859539620578289,
    axis_identifiability_accuracy: 1.0,
    compositional_r2: 0.1426162966041996,
    compositional_recon_cosine: 0.3776445469967642,
    controlled_additive_r2: -0.020736867900062128,
    controlled_additive_cosine: 0.7171257920215101,
  },
  hypotheses: {
    H1_axis_decoupling: true,
    H2_subspace_near_orthogonal: true,
    H3_linear_composition_exists: false,
    H4_controlled_additivity: false,
  },
};

function isValidPayload(payload) {
  return Boolean(
    payload &&
      payload.metrics &&
      payload.hypotheses &&
      payload.dimension_stats &&
      Array.isArray(payload.pairwise)
  );
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(2)}%`;
}

function toLabel(name) {
  if (name === 'color') return '颜色';
  if (name === 'size') return '大小';
  if (name === 'text') return '文字';
  if (name === 'sound') return '声音';
  return name;
}

function pairLabel(pair) {
  const parts = String(pair || '').split('__');
  if (parts.length !== 2) return pair;
  return `${toLabel(parts[0])}-${toLabel(parts[1])}`;
}

function HypoTag({ value }) {
  return (
    <span
      style={{
        color: value ? '#86efac' : '#fca5a5',
        border: `1px solid ${value ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)'}`,
        borderRadius: '999px',
        padding: '2px 8px',
        fontSize: '11px',
      }}
    >
      {value ? 'PASS' : 'FAIL'}
    </span>
  );
}

export default function AppleOrthogonalityDashboard() {
  const [payload, setPayload] = useState(DEFAULT_PAYLOAD);
  const [source, setSource] = useState('内置样本 (apple_multifeature_orthogonality_20260307)');
  const [error, setError] = useState('');

  const pairwise = useMemo(
    () =>
      (payload.pairwise || []).map((x) => ({
        ...x,
        label: pairLabel(x.pair),
      })),
    [payload]
  );

  const dimRows = useMemo(
    () =>
      Object.entries(payload.dimension_stats || {}).map(([k, v]) => ({
        dim: toLabel(k),
        mean_alignment: Number(v?.mean_abs_alignment_to_axis || 0),
        std_alignment: Number(v?.std_abs_alignment_to_axis || 0),
      })),
    [payload]
  );

  const m = payload.metrics || {};
  const h = payload.hypotheses || {};

  const onUploadJson = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      if (!isValidPayload(parsed)) {
        throw new Error('JSON 缺少 metrics / hypotheses / dimension_stats / pairwise');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (errObj) {
      setError(`导入失败：${errObj?.message || '未知错误'}`);
    }
  };

  return (
    <div style={{ marginTop: '14px', padding: '16px', borderRadius: '12px', background: 'rgba(2,6,23,0.46)', border: '1px solid rgba(56,189,248,0.26)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
        <div>
          <div style={{ color: '#7dd3fc', fontSize: '14px', fontWeight: 'bold' }}>苹果四轴正交探针看板（颜色/大小/文字/声音）</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '4px' }}>数据源：{source}</div>
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <label style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '8px', padding: '6px 10px', cursor: 'pointer' }}>
            导入 JSON
            <input type="file" accept="application/json" onChange={onUploadJson} style={{ display: 'none' }} />
          </label>
          <button
            type="button"
            onClick={() => {
              setPayload(DEFAULT_PAYLOAD);
              setSource('内置样本 (apple_multifeature_orthogonality_20260307)');
              setError('');
            }}
            style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '8px', padding: '6px 10px', background: 'transparent', cursor: 'pointer' }}
          >
            重置
          </button>
        </div>
      </div>

      {error && <div style={{ color: '#fca5a5', fontSize: '11px', marginTop: '8px' }}>{error}</div>}

      <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '8px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px', padding: '8px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>轴间平均绝对余弦</div>
          <div style={{ color: '#e2e8f0', fontSize: '18px', fontWeight: 'bold' }}>{Number(m.mean_abs_pairwise_cosine || 0).toFixed(4)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px', padding: '8px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>签名重叠 Jaccard</div>
          <div style={{ color: '#e2e8f0', fontSize: '18px', fontWeight: 'bold' }}>{Number(m.mean_signature_jaccard || 0).toFixed(4)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px', padding: '8px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>子空间正交性</div>
          <div style={{ color: '#e2e8f0', fontSize: '18px', fontWeight: 'bold' }}>{pct(m.mean_principal_orthogonality || 0)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px', padding: '8px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>轴可识别率</div>
          <div style={{ color: '#e2e8f0', fontSize: '18px', fontWeight: 'bold' }}>{pct(m.axis_identifiability_accuracy || 0)}</div>
        </div>
      </div>

      <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '12px' }}>
        <div style={{ minHeight: '250px', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px', padding: '8px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1) 四轴两两解耦指标</div>
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={pairwise} margin={{ top: 8, right: 12, left: 6, bottom: 8 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.2)" strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.45)', borderRadius: '8px' }}
                formatter={(value, name) => [Number(value).toFixed(4), name]}
              />
              <Legend wrapperStyle={{ fontSize: '11px', color: '#cbd5e1' }} />
              <Bar dataKey="abs_cosine" name="|cos|" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="signature_jaccard" name="Jaccard" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Line type="monotone" dataKey="principal_orthogonality" name="1-principal_sim" stroke="#22c55e" strokeWidth={2} dot={{ r: 2 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px', padding: '10px', background: 'rgba(15,23,42,0.4)' }}>
          <div style={{ color: '#e2e8f0', fontSize: '12px', fontWeight: 'bold', marginBottom: '8px' }}>2) 假设判定</div>
          <div style={{ display: 'grid', gap: '8px', fontSize: '12px', color: '#cbd5e1' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
              <span>H1 轴解耦存在</span>
              <HypoTag value={Boolean(h.H1_axis_decoupling)} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
              <span>H2 子空间近正交</span>
              <HypoTag value={Boolean(h.H2_subspace_near_orthogonal)} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
              <span>H3 严格线性组合</span>
              <HypoTag value={Boolean(h.H3_linear_composition_exists)} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
              <span>H4 受控可加性</span>
              <HypoTag value={Boolean(h.H4_controlled_additivity)} />
            </div>
          </div>
          <div style={{ marginTop: '10px', fontSize: '11px', color: '#94a3b8', lineHeight: '1.6' }}>
            判读：正交解耦成立，但严格线性叠加不成立，说明组合阶段仍受非线性门控影响。
          </div>
        </div>
      </div>

      <div style={{ marginTop: '12px', minHeight: '240px', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px', padding: '8px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3) 单轴稳定性（样本差分对本轴均值方向的对齐）</div>
        <ResponsiveContainer width="100%" height={210}>
          <ComposedChart data={dimRows} margin={{ top: 8, right: 12, left: 6, bottom: 8 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.2)" strokeDasharray="3 3" />
            <XAxis dataKey="dim" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.45)', borderRadius: '8px' }}
              formatter={(value, name) => [Number(value).toFixed(4), name]}
            />
            <Legend wrapperStyle={{ fontSize: '11px', color: '#cbd5e1' }} />
            <Bar dataKey="mean_alignment" name="mean_align" fill="#a78bfa" radius={[4, 4, 0, 0]} />
            <Line type="monotone" dataKey="std_alignment" name="std_align" stroke="#f43f5e" strokeWidth={2} dot={{ r: 3 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

