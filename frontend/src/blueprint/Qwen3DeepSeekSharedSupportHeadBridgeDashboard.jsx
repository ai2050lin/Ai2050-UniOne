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
import samplePayload from './data/qwen3_deepseek7b_shared_support_head_bridge_sample.json';

const MODEL_LABELS = {
  qwen3_4b: 'Qwen3-4B',
  deepseek_7b: 'DeepSeek-7B',
};

function isValidPayload(payload) {
  return Boolean(payload && payload.models && payload.headline_metrics && payload.hypotheses);
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

export default function Qwen3DeepSeekSharedSupportHeadBridgeDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const comparisonRows = useMemo(
    () => [
      {
        metric: '精确头重合',
        qwen3: Number(payload?.headline_metrics?.qwen_exact_head_overlap || 0),
        deepseek7b: Number(payload?.headline_metrics?.deepseek_exact_head_overlap || 0),
      },
      {
        metric: '软头重合',
        qwen3: Number(payload?.headline_metrics?.qwen_soft_head_overlap || 0),
        deepseek7b: Number(payload?.headline_metrics?.deepseek_soft_head_overlap || 0),
      },
      {
        metric: '层级重合',
        qwen3: Number(payload?.headline_metrics?.qwen_layer_overlap || 0),
        deepseek7b: Number(payload?.headline_metrics?.deepseek_layer_overlap || 0),
      },
      {
        metric: '层级软重合',
        qwen3: Number(payload?.headline_metrics?.qwen_soft_layer_overlap || 0),
        deepseek7b: Number(payload?.headline_metrics?.deepseek_soft_layer_overlap || 0),
      },
      {
        metric: '共享质量',
        qwen3: Number(payload?.headline_metrics?.qwen_shared_mass || 0),
        deepseek7b: Number(payload?.headline_metrics?.deepseek_shared_mass || 0),
      },
      {
        metric: '机制桥接分',
        qwen3: Number(payload?.models?.qwen3_4b?.global_summary?.mechanism_bridge_score || 0),
        deepseek7b: Number(payload?.models?.deepseek_7b?.global_summary?.mechanism_bridge_score || 0),
      },
    ],
    [payload]
  );

  const relationRows = useMemo(() => {
    const qwenRelations = payload?.models?.qwen3_4b?.relations || [];
    const deepseekRelations = payload?.models?.deepseek_7b?.relations || [];
    const relationNames = Array.from(new Set([...qwenRelations.map((row) => row.relation), ...deepseekRelations.map((row) => row.relation)]));
    return relationNames.map((name) => {
      const qwen = qwenRelations.find((row) => row.relation === name) || {};
      const deepseek = deepseekRelations.find((row) => row.relation === name) || {};
      return {
        relation: name,
        qwen3: Number(qwen.shared_mass_ratio || 0),
        deepseek7b: Number(deepseek.shared_mass_ratio || 0),
      };
    });
  }, [payload]);

  const layerRows = useMemo(() => {
    const qwenLayers = payload?.models?.qwen3_4b?.layer_sets?.top_shared_layers || [];
    const deepseekLayers = payload?.models?.deepseek_7b?.layer_sets?.top_shared_layers || [];
    const layerNames = Array.from(new Set([...qwenLayers.map((row) => row[0]), ...deepseekLayers.map((row) => row[0])])).sort(
      (a, b) => Number(a.slice(1)) - Number(b.slice(1))
    );
    return layerNames.map((name) => {
      const qwen = qwenLayers.find((row) => row[0] === name);
      const deepseek = deepseekLayers.find((row) => row[0] === name);
      return {
        layer: name,
        qwen3: Number(qwen?.[1] || 0),
        deepseek7b: Number(deepseek?.[1] || 0),
      };
    });
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
      setError(`共享支撑桥 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(14,165,233,0.14), transparent 26%), radial-gradient(circle at bottom right, rgba(16,185,129,0.10), transparent 24%), rgba(2,6,23,0.66)',
        border: '1px solid rgba(14,165,233,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>五点三十九补、Qwen3 / DeepSeek7B 共享支撑桥</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把概念调用头和关系中观场头接到同一张共享支撑图上，同时检查精确头复用、层级软重合和关系共享质量，判断真实模型里的同源支撑是头级的还是中观层级的。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen3 层级软重合</div>
          <div style={{ color: '#38bdf8', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.headline_metrics?.qwen_soft_layer_overlap)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 层级软重合</div>
          <div style={{ color: '#34d399', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.headline_metrics?.deepseek_soft_layer_overlap)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DeepSeek 相对层级增益</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.gains?.deepseek_minus_qwen_soft_layer_overlap)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Qwen / DeepSeek 共享质量</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>
            {`${fmt(payload?.headline_metrics?.qwen_shared_mass)} / ${fmt(payload?.headline_metrics?.deepseek_shared_mass)}`}
          </div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>精确头重合是否成立</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>
            {`${pct(payload?.headline_metrics?.qwen_exact_head_overlap)} / ${pct(payload?.headline_metrics?.deepseek_exact_head_overlap)}`}
          </div>
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
              color: value ? '#dbeafe' : '#fee2e2',
              background: value ? 'rgba(14,165,233,0.18)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(14,165,233,0.35)' : 'rgba(248,113,113,0.3)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '不成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 头级与层级共享对比</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={comparisonRows} margin={{ top: 10, right: 12, left: 0, bottom: 18 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-16} textAnchor="end" height={56} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="qwen3" name="Qwen3-4B" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="deepseek7b" name="DeepSeek-7B" fill="#34d399" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 关系共享质量</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={relationRows} margin={{ top: 10, right: 12, left: 0, bottom: 18 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="relation" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-16} textAnchor="end" height={56} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 0.12]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="qwen3" name="Qwen3-4B" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="deepseek7b" name="DeepSeek-7B" fill="#34d399" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 高共享层带</div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={layerRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="layer" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
            <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
            <Legend />
            <Bar dataKey="qwen3" name="Qwen3-4B" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            <Bar dataKey="deepseek7b" name="DeepSeek-7B" fill="#34d399" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>{payload?.project_readout?.summary || '-'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>5. 下一步</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>{payload?.project_readout?.next_question || '-'}</div>
        </div>
      </div>

      <div style={{ marginTop: '12px', color: '#94a3b8', fontSize: '11px' }}>
        {`模型标签: ${Object.keys(payload?.models || {})
          .map((key) => MODEL_LABELS[key] || key)
          .join(' / ')}`}
      </div>
    </div>
  );
}
