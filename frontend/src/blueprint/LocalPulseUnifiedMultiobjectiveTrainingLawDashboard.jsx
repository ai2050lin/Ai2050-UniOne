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
import samplePayload from './data/local_pulse_unified_multiobjective_training_law_sample.json';

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && payload.headline_metrics && payload.hypotheses);
}

const SYSTEM_LABELS = {
  shared_local_replay: '共享局部律',
  regional_decoupled_replay: '分区解耦局部律',
  score_push_regional: '分区冲分局部律',
  balanced_multiobjective: '分区平衡局部律',
  structure_push_regional: '分区结构优先局部律',
};

const PHASE_LABELS = {
  concept_phase: '概念阶段',
  comparison_phase: '比较阶段',
  recovery_phase: '恢复阶段',
};

export default function LocalPulseUnifiedMultiobjectiveTrainingLawDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const [selectedSystem, setSelectedSystem] = useState('regional_decoupled_replay');

  const systemRows = useMemo(() => {
    const systems = payload?.systems || {};
    return Object.entries(systems).map(([key, system]) => ({
      system: SYSTEM_LABELS[key] || key,
      总分目标: Number(system?.aggregate_objective || 0),
      结构目标: Number(system?.structure_objective || 0),
      多目标分: Number(system?.multiobjective_score || 0),
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
      label: '多目标最优',
      value: SYSTEM_LABELS[payload?.headline_metrics?.multiobjective_best_system] || '-',
      tone: '#22c55e',
    },
    {
      label: '结构提升',
      value: fmt(payload?.headline_metrics?.multiobjective_structure_gain),
      tone: '#f59e0b',
    },
    {
      label: '总分代价',
      value: fmt(payload?.headline_metrics?.multiobjective_aggregate_gap),
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
      setError(`统一多目标训练律 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
    setSource('内置样例');
    setError('');
    setSelectedSystem('regional_decoupled_replay');
  }

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(14,165,233,0.14), transparent 28%), radial-gradient(circle at bottom right, rgba(34,197,94,0.12), transparent 26%), rgba(2,6,23,0.68)',
        border: '1px solid rgba(14,165,233,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: '15px', fontWeight: 'bold' }}>五点四十二、统一多目标训练律基准</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px', maxWidth: '860px' }}>
            这组基准把“训练律”本身显式成型。它直接比较单目标冲分和多目标守结构两种训练口径，看统一编码机制会被推向共享平均解，还是推向更接近脑区差异和阶段局部核职责的区域化解。
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
              color: value ? '#dbeafe' : '#fee2e2',
              background: value ? 'rgba(59,130,246,0.18)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(59,130,246,0.30)' : 'rgba(248,113,113,0.30)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '未成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三种目标的选择分化</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={systemRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="system" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="总分目标" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="结构目标" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="多目标分" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 关键权衡</div>
          <div style={{ display: 'grid', gap: '8px' }}>
            <div style={{ borderRadius: '12px', padding: '10px', background: 'rgba(15,23,42,0.45)', border: '1px solid rgba(148,163,184,0.16)' }}>
              <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>多目标训练会换掉共享平均解</div>
              <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.75', marginTop: '4px' }}>
                总分最优仍是共享局部律，但多目标最优已经切到分区解耦局部律。
              </div>
            </div>
            <div style={{ borderRadius: '12px', padding: '10px', background: 'rgba(15,23,42,0.45)', border: '1px solid rgba(148,163,184,0.16)' }}>
              <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>结构分明显提高</div>
              <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.75', marginTop: '4px' }}>
                结构提升为 {fmt(payload?.headline_metrics?.multiobjective_structure_gain)}，说明多目标训练开始压缩“高分但错组织”的假解。
              </div>
            </div>
            <div style={{ borderRadius: '12px', padding: '10px', background: 'rgba(15,23,42,0.45)', border: '1px solid rgba(148,163,184,0.16)' }}>
              <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>但中期比较核还没一起修好</div>
              <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.75', marginTop: '4px' }}>
                当前概念上游优势提升 {fmt(payload?.headline_metrics?.multiobjective_concept_gain)}，但比较局部优势变化为 {fmt(payload?.headline_metrics?.multiobjective_comparison_gain)}。这意味着下一步要把多目标训练继续推到中期比较核稳相。
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap', alignItems: 'center', marginBottom: '8px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px' }}>3. 结构指标展开</div>
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
                    color: selectedSystem === key ? '#0f172a' : '#e2e8f0',
                    background: selectedSystem === key ? '#7dd3fc' : 'transparent',
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
