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
import samplePayload from './data/local_pulse_trainable_region_family_generator_sample.json';

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && payload.headline_metrics && payload.hypotheses);
}

const SYSTEM_LABELS = {
  shared_local_replay: '共享局部回放',
  learned_region_family: '学习参数族',
  fixed_generated_family: '固定生成器',
  trainable_generated_family: '可训练生成器',
};

export default function LocalPulseTrainableRegionFamilyGeneratorDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const systemRows = useMemo(() => {
    const systems = payload?.systems || {};
    return Object.entries(systems).map(([key, system]) => ({
      system: SYSTEM_LABELS[key] || key,
      总分目标: Number(system?.aggregate_objective || 0),
      三阶段闭环分: Number(system?.three_stage_score || 0),
      闭环平衡: Number(system?.closure_balance_score || 0),
    }));
  }, [payload]);

  const historyRows = useMemo(() => {
    return (payload?.optimization_history || []).map((row) => ({
      step: Number(row?.step || 0),
      三阶段闭环分: Number(row?.three_stage_score || 0),
      闭环平衡: Number(row?.closure_balance_score || 0),
      总分目标: Number(row?.aggregate_objective || 0),
    }));
  }, [payload]);

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 systems / headline_metrics / hypotheses 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`可训练生成器 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(34,197,94,0.16), transparent 28%), radial-gradient(circle at bottom right, rgba(14,165,233,0.12), transparent 26%), rgba(2,6,23,0.68)',
        border: '1px solid rgba(34,197,94,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: '15px', fontWeight: 'bold' }}>五点四十八、可训练脑区参数族生成器</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px', maxWidth: '860px' }}>
            这组基准把 5 维脑区参数族生成器从固定映射推进成可训练更新。重点不再是再做一次离线搜索，而是看同一组低维生成坐标在迭代更新后，能否同时把三阶段闭环分和闭环平衡继续推高。
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>相对固定生成器闭环提升</div>
          <div style={{ color: '#22c55e', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>
            {fmt(payload?.headline_metrics?.trainable_three_stage_gain_vs_fixed)}
          </div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>相对固定生成器平衡提升</div>
          <div style={{ color: '#3b82f6', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>
            {fmt(payload?.headline_metrics?.trainable_balance_gain_vs_fixed)}
          </div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>相对学习参数族闭环提升</div>
          <div style={{ color: '#10b981', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>
            {fmt(payload?.headline_metrics?.trainable_three_stage_gain_vs_learned)}
          </div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最终步长</div>
          <div style={{ color: '#f59e0b', fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>
            {fmt(payload?.headline_metrics?.final_step_size, 3)}
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
              color: value ? '#dcfce7' : '#fee2e2',
              background: value ? 'rgba(34,197,94,0.16)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(34,197,94,0.30)' : 'rgba(248,113,113,0.30)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '未成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 系统对照</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={systemRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="system" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="总分目标" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="三阶段闭环分" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="闭环平衡" fill="#a855f7" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 训练轨迹</div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={historyRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="step" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="三阶段闭环分" stroke="#22c55e" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="闭环平衡" stroke="#a855f7" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="总分目标" stroke="#38bdf8" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
