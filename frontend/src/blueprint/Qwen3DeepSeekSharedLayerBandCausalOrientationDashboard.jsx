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
import samplePayload from './data/qwen3_deepseek7b_shared_layer_band_causal_orientation_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.models && payload.headline_metrics && payload.hypotheses);
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

export default function Qwen3DeepSeekSharedLayerBandCausalOrientationDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const orientationRows = useMemo(
    () => [
      {
        model: 'Qwen3-4B',
        概念命中: Number(payload?.headline_metrics?.qwen_concept_hit_mean || 0),
        关系命中: Number(payload?.headline_metrics?.qwen_relation_hit_mean || 0),
      },
      {
        model: 'DeepSeek-7B',
        概念命中: Number(payload?.headline_metrics?.deepseek_concept_hit_mean || 0),
        关系命中: Number(payload?.headline_metrics?.deepseek_relation_hit_mean || 0),
      },
    ],
    [payload]
  );

  const signatureRows = useMemo(
    () => [
      {
        metric: '概念边界相关',
        qwen3: Number(payload?.models?.qwen3_4b?.global_summary?.concept_hit_margin_corr || 0),
        deepseek7b: Number(payload?.models?.deepseek_7b?.global_summary?.concept_hit_margin_corr || 0),
      },
      {
        metric: '关系边界相关',
        qwen3: Number(payload?.models?.qwen3_4b?.global_summary?.relation_hit_margin_corr || 0),
        deepseek7b: Number(payload?.models?.deepseek_7b?.global_summary?.relation_hit_margin_corr || 0),
      },
      {
        metric: '关系任务增益相关',
        qwen3: Number(payload?.models?.qwen3_4b?.global_summary?.relation_hit_behavior_gain_corr || 0),
        deepseek7b: Number(payload?.models?.deepseek_7b?.global_summary?.relation_hit_behavior_gain_corr || 0),
      },
      {
        metric: '机制桥接分',
        qwen3: Number(payload?.headline_metrics?.qwen_mechanism_bridge || 0),
        deepseek7b: Number(payload?.headline_metrics?.deepseek_mechanism_bridge || 0),
      },
    ],
    [payload]
  );

  const conceptRows = useMemo(() => {
    const qwen = payload?.models?.qwen3_4b?.concepts || [];
    const deepseek = payload?.models?.deepseek_7b?.concepts || [];
    const names = Array.from(new Set([...qwen.map((row) => row.concept), ...deepseek.map((row) => row.concept)]));
    return names.map((name) => ({
      concept: name,
      qwen3: Number(qwen.find((row) => row.concept === name)?.shared_layer_hit_ratio || 0),
      deepseek7b: Number(deepseek.find((row) => row.concept === name)?.shared_layer_hit_ratio || 0),
    }));
  }, [payload]);

  const relationRows = useMemo(() => {
    const qwen = payload?.models?.qwen3_4b?.relations || [];
    const deepseek = payload?.models?.deepseek_7b?.relations || [];
    const names = Array.from(new Set([...qwen.map((row) => row.relation), ...deepseek.map((row) => row.relation)]));
    return names.map((name) => ({
      relation: name,
      qwen3: Number(qwen.find((row) => row.relation === name)?.shared_layer_hit_ratio || 0),
      deepseek7b: Number(deepseek.find((row) => row.relation === name)?.shared_layer_hit_ratio || 0),
    }));
  }, [payload]);

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 models / headline_metrics / hypotheses 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`共享层带因果取向 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
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
          'radial-gradient(circle at top left, rgba(16,185,129,0.13), transparent 26%), radial-gradient(circle at bottom right, rgba(59,130,246,0.10), transparent 24%), rgba(2,6,23,0.66)',
        border: '1px solid rgba(16,185,129,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>五点三十九续、Qwen3 / DeepSeek7B 共享层带因果取向</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把共享层带与概念边界、关系中观因果尺度和结构化任务增益拼到一起，直接看真实模型里的共享支撑到底偏向概念边界还是偏向关系协议。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen3 取向</div>
          <div style={{ color: '#38bdf8', fontSize: '16px', fontWeight: 'bold' }}>{payload?.models?.qwen3_4b?.global_summary?.orientation_label || '-'}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`orientation ${fmt(payload?.headline_metrics?.qwen_orientation)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 取向</div>
          <div style={{ color: '#34d399', fontSize: '16px', fontWeight: 'bold' }}>{payload?.models?.deepseek_7b?.global_summary?.orientation_label || '-'}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`orientation ${fmt(payload?.headline_metrics?.deepseek_orientation)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>取向差</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.gains?.deepseek_minus_qwen_orientation)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>机制桥接差</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.gains?.deepseek_minus_qwen_mechanism_bridge)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 关系增益相关</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.headline_metrics?.deepseek_relation_gain_corr)}</div>
        </div>
      </div>

      <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {Object.entries(payload?.hypotheses || {}).map(([key, value]) => (
          <div
            key={key}
            style={{
              borderRadius: '999px',
              padding: '6px 10px',
              fontSize: '11px',
              color: value ? '#dcfce7' : '#fee2e2',
              background: value ? 'rgba(16,185,129,0.18)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(16,185,129,0.35)' : 'rgba(248,113,113,0.3)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '不成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 共享层带取向</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={orientationRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 0.35]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="概念命中" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="关系命中" fill="#34d399" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 因果与任务读数</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={signatureRows} margin={{ top: 10, right: 12, left: 0, bottom: 38 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-18} textAnchor="end" height={56} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[-1, 1]} />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="qwen3" name="Qwen3-4B" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="deepseek7b" name="DeepSeek-7B" fill="#34d399" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 概念层带命中</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={conceptRows} margin={{ top: 10, right: 12, left: 0, bottom: 46 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="concept" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-18} textAnchor="end" height={60} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 0.6]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="qwen3" name="Qwen3-4B" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="deepseek7b" name="DeepSeek-7B" fill="#34d399" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 关系层带命中</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={relationRows} margin={{ top: 10, right: 12, left: 0, bottom: 46 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="relation" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-18} textAnchor="end" height={60} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 0.45]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="qwen3" name="Qwen3-4B" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="deepseek7b" name="DeepSeek-7B" fill="#34d399" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>5. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>{payload?.project_readout?.summary || '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>6. 下一步</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>{payload?.project_readout?.next_question || '-'}</div>
        </div>
      </div>
    </div>
  );
}
