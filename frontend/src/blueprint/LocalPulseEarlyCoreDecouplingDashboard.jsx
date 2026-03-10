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
import samplePayload from './data/local_pulse_early_core_decoupling_benchmark_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && payload.headline_metrics && payload.hypotheses);
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

const SYSTEM_LABELS = {
  baseline_coupled_replay: '原始耦合回放',
  decoupled_upstream_replay: '上游解耦回放',
};

const PHASE_LABELS = {
  concept_phase: '概念阶段',
  comparison_phase: '比较阶段',
  recovery_phase: '恢复阶段',
};

export default function LocalPulseEarlyCoreDecouplingDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const [selectedSystem, setSelectedSystem] = useState('decoupled_upstream_replay');

  const systemRows = useMemo(() => {
    const systems = payload?.systems || {};
    return Object.entries(systems).map(([key, system]) => ({
      system: SYSTEM_LABELS[key] || key,
      局部整合分数: Number(system?.local_integration_score || 0),
      概念阶段上游优势: Number(system?.concept_phase_upstream_advantage || 0),
      比较阶段局部优势: Number(system?.comparison_phase_memory_comparator_advantage || 0),
      基线准确率: Number(system?.baseline_accuracy || 0),
    }));
  }, [payload]);

  const gainRows = useMemo(
    () => [
      { item: '上游优势增益', value: Number(payload?.headline_metrics?.upstream_advantage_gain || 0) },
      { item: '整合分数增益', value: Number(payload?.headline_metrics?.integration_gain || 0) },
      { item: '恢复增益', value: Number(payload?.headline_metrics?.recovery_gain || 0) },
    ],
    [payload],
  );

  const selected = payload?.systems?.[selectedSystem] || payload?.systems?.decoupled_upstream_replay || {};

  const phaseRows = useMemo(() => {
    const matrix = selected?.phase_drop_matrix || {};
    return Object.entries(matrix).map(([phase, rows]) => ({
      phase: PHASE_LABELS[phase] || phase,
      感知区: Number(rows?.sensory || 0),
      记忆区: Number(rows?.memory || 0),
      比较区: Number(rows?.comparator || 0),
      动作区: Number(rows?.motor || 0),
    }));
  }, [selected]);

  const summaryRows = useMemo(() => {
    const summary = selected?.phase_summary || {};
    return ['concept_phase', 'comparison_phase', 'recovery_phase'].map((phase) => ({
      phase: PHASE_LABELS[phase] || phase,
      topRegion: summary?.[phase]?.top_region || '-',
      topDrop: Number(summary?.[phase]?.top_drop || 0),
      upstreamMass: Number(summary?.[phase]?.upstream_mass || 0),
      downstreamMass: Number(summary?.[phase]?.downstream_mass || 0),
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
      setError(`早期核心解耦基准 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
    setSource('内置样例');
    setError('');
    setSelectedSystem('decoupled_upstream_replay');
  }

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(245,158,11,0.16), transparent 28%), radial-gradient(circle at bottom right, rgba(16,185,129,0.12), transparent 24%), rgba(2,6,23,0.66)',
        border: '1px solid rgba(245,158,11,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: '15px', fontWeight: 'bold' }}>五点三十九终续、早期感知核解耦基准</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            直接比较“原始耦合回放”和“上游解耦回放”，测试把早期流量留在 `sensory / memory` 后，能否把 `concept_phase` 的局部核心拉回上游，同时不牺牲整体整合。
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>原始上游优势</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.headline_metrics?.baseline_upstream_advantage)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>解耦后上游优势</div>
          <div style={{ color: '#f59e0b', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.headline_metrics?.decoupled_upstream_advantage)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>上游优势增益</div>
          <div style={{ color: '#10b981', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.headline_metrics?.upstream_advantage_gain)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>解耦后整合分数</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.headline_metrics?.decoupled_local_integration_score)}</div>
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
              color: value ? '#fef3c7' : '#fee2e2',
              background: value ? 'rgba(245,158,11,0.18)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(245,158,11,0.35)' : 'rgba(248,113,113,0.30)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '未成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1.1fr) minmax(0, 0.9fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 两组系统总览</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={systemRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="system" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="局部整合分数" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="概念阶段上游优势" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="比较阶段局部优势" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="基线准确率" fill="#a78bfa" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 关键增益</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={gainRows} layout="vertical" margin={{ top: 10, right: 12, left: 20, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis type="category" dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" width={82} />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Bar dataKey="value" fill="#f59e0b" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap', alignItems: 'center', marginBottom: '8px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px' }}>3. 相位掉分矩阵</div>
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
                    color: selectedSystem === key ? '#451a03' : '#e2e8f0',
                    background: selectedSystem === key ? '#fcd34d' : 'transparent',
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
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" domain={[0, 0.25]} />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="感知区" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="记忆区" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="比较区" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="动作区" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 阶段顶部核心</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(145px, 1fr))', gap: '8px' }}>
            {summaryRows.map((row) => (
              <div key={row.phase} style={{ borderRadius: '12px', padding: '10px', background: 'rgba(15,23,42,0.45)', border: '1px solid rgba(148,163,184,0.16)' }}>
                <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold', marginBottom: '6px' }}>{row.phase}</div>
                <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.75' }}>
                  <div>{`顶部区域: ${row.topRegion}`}</div>
                  <div>{`顶部掉分: ${fmt(row.topDrop)}`}</div>
                  <div>{`上游质量: ${fmt(row.upstreamMass)}`}</div>
                  <div>{`下游质量: ${fmt(row.downstreamMass)}`}</div>
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: '10px', color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            {payload?.project_readout?.summary}
          </div>
          <div style={{ marginTop: '10px', color: '#fde68a', fontSize: '12px', lineHeight: '1.8' }}>
            {payload?.project_readout?.next_question}
          </div>
        </div>
      </div>
    </div>
  );
}
