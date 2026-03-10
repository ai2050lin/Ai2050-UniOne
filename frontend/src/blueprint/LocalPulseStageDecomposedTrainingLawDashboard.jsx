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
import samplePayload from './data/local_pulse_stage_decomposed_training_law_sample.json';

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && payload.headline_metrics && payload.hypotheses);
}

const SYSTEM_LABELS = {
  shared_local_replay: '共享局部律',
  regional_unified_multiobjective: '统一多目标局部律',
  regional_stage_decomposed: '阶段分解局部律',
};

const PHASE_LABELS = {
  concept_phase: '概念阶段',
  comparison_phase: '比较阶段',
  recovery_phase: '恢复阶段',
};

export default function LocalPulseStageDecomposedTrainingLawDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const [selectedSystem, setSelectedSystem] = useState('regional_stage_decomposed');

  const systemRows = useMemo(() => {
    const systems = payload?.systems || {};
    return Object.entries(systems).map(([key, system]) => ({
      system: SYSTEM_LABELS[key] || key,
      总分目标: Number(system?.aggregate_objective || 0),
      阶段分解目标: Number(system?.stage_decomposed_score || 0),
      概念上游优势: Number(system?.concept_phase_upstream_advantage || 0),
      比较局部优势: Number(system?.comparison_phase_memory_comparator_advantage || 0),
    }));
  }, [payload]);

  const selected = payload?.systems?.[selectedSystem] || {};

  const dropRows = useMemo(() => {
    const matrix = selected?.phase_drop_matrix || {};
    return Object.entries(matrix).map(([phase, row]) => ({
      phase: PHASE_LABELS[phase] || phase,
      感知区: Number(row?.sensory || 0),
      记忆区: Number(row?.memory || 0),
      比较区: Number(row?.comparator || 0),
      动作区: Number(row?.motor || 0),
    }));
  }, [selected]);

  const summaryCards = [
    {
      label: '总分最优',
      value: SYSTEM_LABELS[payload?.headline_metrics?.aggregate_best_system] || '-',
      tone: '#38bdf8',
    },
    {
      label: '阶段最优',
      value: SYSTEM_LABELS[payload?.headline_metrics?.stage_best_system] || '-',
      tone: '#22c55e',
    },
    {
      label: '阶段平衡提升',
      value: fmt(payload?.headline_metrics?.stage_vs_aggregate_structure_balance_gain),
      tone: '#f59e0b',
    },
    {
      label: '总分代价',
      value: fmt(payload?.headline_metrics?.stage_vs_aggregate_score_gap),
      tone: '#ef4444',
    },
  ];

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
      setError(`阶段分解训练律 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
    setSource('内置样例');
    setError('');
    setSelectedSystem('regional_stage_decomposed');
  }

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(234,179,8,0.16), transparent 28%), radial-gradient(circle at bottom right, rgba(34,197,94,0.12), transparent 26%), rgba(2,6,23,0.68)',
        border: '1px solid rgba(234,179,8,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: '15px', fontWeight: 'bold' }}>五点四十三、阶段分解训练律基准</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px', maxWidth: '860px' }}>
            这组基准把统一多目标再拆成阶段分解训练律，单独约束概念阶段和比较阶段。核心问题是：同一套局部编码机制，能不能在不回到共享平均解的前提下，同时把两个关键阶段都拉到正确方向。
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
        {summaryCards.map((card) => (
          <div key={card.label} style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
            <div style={{ color: '#94a3b8', fontSize: '11px' }}>{card.label}</div>
            <div style={{ color: card.tone, fontSize: '16px', fontWeight: 'bold', marginTop: '4px' }}>{card.value}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {Object.entries(payload?.hypotheses || {}).map(([key, value]) => (
          <div
            key={key}
            style={{
              borderRadius: '999px',
              padding: '6px 10px',
              fontSize: '11px',
              color: value ? '#fef9c3' : '#fee2e2',
              background: value ? 'rgba(234,179,8,0.16)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(234,179,8,0.30)' : 'rgba(248,113,113,0.30)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '未成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 总分目标与阶段目标分化</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={systemRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="system" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="总分目标" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="阶段分解目标" fill="#eab308" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 关键读法</div>
          <div style={{ display: 'grid', gap: '8px' }}>
            <div style={{ borderRadius: '12px', padding: '10px', background: 'rgba(15,23,42,0.45)', border: '1px solid rgba(148,163,184,0.16)' }}>
              <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>阶段分解律拿回了阶段最优</div>
              <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.75', marginTop: '4px' }}>
                总分最优仍是共享局部律，但阶段目标最优已经切到阶段分解局部律。
              </div>
            </div>
            <div style={{ borderRadius: '12px', padding: '10px', background: 'rgba(15,23,42,0.45)', border: '1px solid rgba(148,163,184,0.16)' }}>
              <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>比较阶段被明显拉回</div>
              <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.75', marginTop: '4px' }}>
                相比统一多目标版，比较局部优势提升 {fmt(payload?.headline_metrics?.stage_vs_unified_comparison_gain)}，而概念上游优势基本持平。
              </div>
            </div>
            <div style={{ borderRadius: '12px', padding: '10px', background: 'rgba(15,23,42,0.45)', border: '1px solid rgba(148,163,184,0.16)' }}>
              <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>阶段平衡大幅提高</div>
              <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.75', marginTop: '4px' }}>
                相比总分最优解，阶段平衡提升 {fmt(payload?.headline_metrics?.stage_vs_aggregate_structure_balance_gain)}，总分代价为 {fmt(payload?.headline_metrics?.stage_vs_aggregate_score_gap)}。
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap', alignItems: 'center', marginBottom: '8px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px' }}>3. 阶段结构指标</div>
            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
              {Object.keys(payload?.systems || {}).map((key) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setSelectedSystem(key)}
                  style={{
                    borderRadius: '999px',
                    border: '1px solid rgba(148,163,184,0.28)',
                    padding: '5px 9px',
                    fontSize: '11px',
                    cursor: 'pointer',
                    color: selectedSystem === key ? '#422006' : '#e2e8f0',
                    background: selectedSystem === key ? '#fde68a' : 'transparent',
                  }}
                >
                  {SYSTEM_LABELS[key] || key}
                </button>
              ))}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={systemRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="system" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="概念上游优势" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="比较局部优势" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 当前系统阶段掉分</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={dropRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="phase" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="感知区" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="记忆区" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="比较区" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="动作区" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
