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
import samplePayload from './data/local_pulse_region_differentiated_multiobjective_selector_sample.json';

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && payload.headline_metrics && payload.hypotheses);
}

const SYSTEM_LABELS = {
  shared_local_replay: '共享局部律回放',
  regional_local_replay: '分区局部律回放',
  regional_phase_tuned_replay: '分区阶段调谐回放',
};

const PHASE_LABELS = {
  concept_phase: '概念阶段',
  comparison_phase: '比较阶段',
  recovery_phase: '恢复阶段',
};

export default function LocalPulseRegionDifferentiatedSelectorDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const [selectedSystem, setSelectedSystem] = useState('regional_phase_tuned_replay');

  const systemRows = useMemo(() => {
    const systems = payload?.systems || {};
    return Object.entries(systems).map(([key, system]) => ({
      system: SYSTEM_LABELS[key] || key,
      总分: Number(system?.aggregate_score || 0),
      结构分: Number(system?.structure_score || 0),
      概念上游优势: Number(system?.concept_phase_upstream_advantage || 0),
      比较局部优势: Number(system?.comparison_phase_memory_comparator_advantage || 0),
    }));
  }, [payload]);

  const summaryCards = [
    {
      label: '总分最优',
      value: SYSTEM_LABELS[payload?.headline_metrics?.aggregate_best_system] || '-',
      tone: '#38bdf8',
    },
    {
      label: '结构最优',
      value: SYSTEM_LABELS[payload?.headline_metrics?.structure_best_system] || '-',
      tone: '#22c55e',
    },
    {
      label: '共享减调谐总分差',
      value: fmt(payload?.headline_metrics?.shared_vs_tuned_score_gap),
      tone: '#f59e0b',
    },
    {
      label: '调谐减共享结构差',
      value: fmt(payload?.headline_metrics?.tuned_vs_shared_structure_gap),
      tone: '#ef4444',
    },
  ];

  const selected = payload?.systems?.[selectedSystem] || {};

  const phaseRows = useMemo(() => {
    const phaseSummary = selected?.phase_summary || {};
    return Object.entries(phaseSummary)
      .filter(([key]) => key !== 'distinct_top_region_count' && key !== 'top_region_sequence')
      .map(([phase, row]) => ({
        phase: PHASE_LABELS[phase] || phase,
        上游质量: Number(row?.upstream_mass || 0),
        下游质量: Number(row?.downstream_mass || 0),
        记忆比较质量: Number(row?.memory_comparator_mass || 0),
        感知动作质量: Number(row?.sensory_motor_mass || 0),
      }));
  }, [selected]);

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
      setError(`多目标选择基准 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
    setSource('内置样例');
    setError('');
    setSelectedSystem('regional_phase_tuned_replay');
  }

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(16,185,129,0.16), transparent 28%), radial-gradient(circle at bottom right, rgba(251,191,36,0.12), transparent 24%), rgba(2,6,23,0.68)',
        border: '1px solid rgba(16,185,129,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: '15px', fontWeight: 'bold' }}>五点四十一、脑区差异多目标选择基准</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px', maxWidth: '860px' }}>
            这组基准专门回答一个核心问题：如果训练对象真的是统一编码机制，但不同脑区只是参数不同，那么选模型时不能只看总分。这个面板把
            “总分最优”和“结构最优”分开外部化，直接看单目标选择会不会把系统推回过度统一的共享局部律。
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
              color: value ? '#dcfce7' : '#fee2e2',
              background: value ? 'rgba(34,197,94,0.16)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(34,197,94,0.32)' : 'rgba(248,113,113,0.30)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '未成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 总分与结构分对照</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={systemRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="system" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="总分" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="结构分" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="概念上游优势" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="比较局部优势" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>2. 结论读法</div>
          <div style={{ display: 'grid', gap: '8px' }}>
            <div style={{ borderRadius: '12px', padding: '10px', background: 'rgba(15,23,42,0.45)', border: '1px solid rgba(148,163,184,0.16)' }}>
              <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>只看总分会选共享局部律</div>
              <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.75', marginTop: '4px' }}>
                当前总分最优是 `shared_local_replay`，比阶段调谐版高出 {fmt(payload?.headline_metrics?.shared_vs_tuned_score_gap)}。
              </div>
            </div>
            <div style={{ borderRadius: '12px', padding: '10px', background: 'rgba(15,23,42,0.45)', border: '1px solid rgba(148,163,184,0.16)' }}>
              <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>看结构会改选分区阶段调谐律</div>
              <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.75', marginTop: '4px' }}>
                当前结构最优是 `regional_phase_tuned_replay`，结构分比共享版高 {fmt(payload?.headline_metrics?.tuned_vs_shared_structure_gap)}。
              </div>
            </div>
            <div style={{ borderRadius: '12px', padding: '10px', background: 'rgba(15,23,42,0.45)', border: '1px solid rgba(148,163,184,0.16)' }}>
              <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>Pareto 前沿保留了两类候选</div>
              <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.75', marginTop: '4px' }}>
                {`当前 Pareto 前沿: ${(payload?.headline_metrics?.pareto_front || []).map((item) => SYSTEM_LABELS[item] || item).join(' / ')}`}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap', alignItems: 'center', marginBottom: '8px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px' }}>3. 阶段质量分布</div>
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
                    color: selectedSystem === key ? '#14532d' : '#e2e8f0',
                    background: selectedSystem === key ? '#86efac' : 'transparent',
                  }}
                >
                  {SYSTEM_LABELS[key] || key}
                </button>
              ))}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={phaseRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="phase" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="上游质量" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="下游质量" fill="#ef4444" radius={[4, 4, 0, 0]} />
              <Bar dataKey="记忆比较质量" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="感知动作质量" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 各阶段局部掉分</div>
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
