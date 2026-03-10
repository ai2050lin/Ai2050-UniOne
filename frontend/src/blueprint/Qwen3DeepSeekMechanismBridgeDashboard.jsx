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
import samplePayload from './data/qwen3_deepseek7b_mechanism_bridge_sample.json';

const COMPONENT_LABELS = {
  shared_basis: '共享基底',
  offset: '个体偏移',
  H_representation: '表征层级 H',
  G_gating: '门控 G',
  R_relation: '关系项 R',
  T_topology: '拓扑项 T',
  protocol_calling: '协议场调用',
  evidence_directness: '证据直接性',
};

const MODEL_LABELS = {
  qwen3_4b: 'Qwen3-4B',
  deepseek_7b: 'DeepSeek-7B',
};

function isValidPayload(payload) {
  return Boolean(payload && payload.models && payload.ranking);
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

export default function Qwen3DeepSeekMechanismBridgeDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const ranking = Array.isArray(payload?.ranking) ? payload.ranking : [];
  const verdict = payload?.cross_model_verdict || {};
  const models = payload?.models || {};
  const conclusion = payload?.global_conclusion || {};

  const modelRows = useMemo(
    () =>
      Object.entries(models).map(([name, row]) => ({
        model: MODEL_LABELS[name] || name,
        mechanism_bridge: Number(row.mechanism_bridge_score || 0),
        directness: Number(row.components?.evidence_directness || 0),
      })),
    [models]
  );

  const componentRows = useMemo(() => {
    const qwen = models?.qwen3_4b?.components || {};
    const deepseek = models?.deepseek_7b?.components || {};
    return Object.keys(COMPONENT_LABELS).map((key) => ({
      component: COMPONENT_LABELS[key],
      qwen3: Number(qwen[key] || 0),
      deepseek7b: Number(deepseek[key] || 0),
    }));
  }, [models]);

  const weakestRows = useMemo(
    () =>
      Object.entries(models).flatMap(([name, row]) =>
        (row.weakest_links || []).map((item) => ({
          model: MODEL_LABELS[name] || name,
          component: COMPONENT_LABELS[item.component] || item.component,
          score: Number(item.score || 0),
        }))
      ),
    [models]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 models 或 ranking 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`Qwen3 / DeepSeek7B 机制桥接 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(34,197,94,0.10), transparent 28%), radial-gradient(circle at top right, rgba(56,189,248,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(34,197,94,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>Qwen3 / DeepSeek7B 机制桥接</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            统一比较共享基底、偏移、表征层级、门控、关系、拓扑和协议场调用，直接看两模型的主干机制链条闭合到什么程度。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <label
            style={{
              color: '#e2e8f0',
              fontSize: '11px',
              border: '1px solid rgba(148,163,184,0.35)',
              borderRadius: '999px',
              padding: '6px 10px',
              cursor: 'pointer',
            }}
          >
            导入 JSON
            <input type="file" accept="application/json" onChange={onUpload} style={{ display: 'none' }} />
          </label>
          <button
            type="button"
            onClick={resetAll}
            style={{
              color: '#e2e8f0',
              fontSize: '11px',
              border: '1px solid rgba(148,163,184,0.35)',
              borderRadius: '999px',
              padding: '6px 10px',
              background: 'transparent',
              cursor: 'pointer',
            }}
          >
            重置
          </button>
        </div>
      </div>

      <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px' }}>{`数据源: ${source}`}</div>
      {error && <div style={{ marginTop: '8px', color: '#fca5a5', fontSize: '11px' }}>{error}</div>}

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px' }}>
        {ranking.map((row) => (
          <div key={row.model_name} style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
            <div style={{ color: '#94a3b8', fontSize: '11px' }}>{MODEL_LABELS[row.model_name] || row.model_name}</div>
            <div style={{ color: '#22c55e', fontSize: '18px', fontWeight: 'bold' }}>{fmt(row.mechanism_bridge_score)}</div>
            <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>
              {`当前薄弱环节: ${(row.weakest_links || []).map((item) => COMPONENT_LABELS[item.component] || item.component).join(' / ')}`}
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {Object.entries(verdict?.component_status || {}).map(([key, value]) => (
          <div
            key={key}
            style={{
              borderRadius: '999px',
              padding: '6px 10px',
              fontSize: '11px',
              color: value === 'consistent' ? '#dcfce7' : '#fde68a',
              background: value === 'consistent' ? 'rgba(34,197,94,0.18)' : 'rgba(245,158,11,0.18)',
              border: `1px solid ${value === 'consistent' ? 'rgba(34,197,94,0.3)' : 'rgba(245,158,11,0.3)'}`,
            }}
          >
            {`${COMPONENT_LABELS[key] || key}: ${value}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 两模型总桥接分数</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={modelRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="mechanism_bridge" name="机制桥接分数" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="directness" name="证据直接性" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 组件级对比</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={componentRows} margin={{ top: 10, right: 12, left: 0, bottom: 54 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="component" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-20} textAnchor="end" height={70} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="qwen3" name="Qwen3-4B" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="deepseek7b" name="DeepSeek-7B" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(320px, 0.9fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 当前薄弱环节</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={weakestRows} margin={{ top: 10, right: 12, left: 0, bottom: 40 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="component" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-18} textAnchor="end" height={60} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value, _name, props) => [pct(value), props?.payload?.model]} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="score" name="薄弱分量分数" fill="#f87171" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ display: 'grid', gap: '12px' }}>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 结论</div>
            <div style={{ color: '#e2e8f0', fontSize: '13px', lineHeight: '1.7' }}>{conclusion?.statement || '-'}</div>
          </div>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>5. 当前证据边界</div>
            <div style={{ display: 'grid', gap: '6px' }}>
              {Object.entries(models).map(([name, row]) => (
                <div key={name} style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.7' }}>
                  {`${MODEL_LABELS[name] || name}: ${row?.boundaries?.length ? row.boundaries.join('；') : '当前主干已无额外边界说明'}`}
                </div>
              ))}
            </div>
          </div>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>6. 下一步</div>
            <div style={{ display: 'grid', gap: '6px' }}>
              {(conclusion?.next_steps || []).map((item) => (
                <div key={item} style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.7' }}>
                  {item}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
