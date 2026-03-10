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
import samplePayload from './data/local_pulse_region_heterogeneity_benchmark_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && payload.headline_metrics && payload.gains);
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

const SYSTEM_LABELS = {
  homogeneous_local_stdp: '同质局部 STDP',
  heterogeneous_local_stdp: '异质局部 STDP',
  heterogeneous_local_replay: '异质局部回放',
};

export default function LocalPulseRegionHeterogeneityDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const [selectedSystem, setSelectedSystem] = useState('heterogeneous_local_replay');

  const systemRows = useMemo(() => {
    const systems = payload?.systems || {};
    return Object.entries(systems).map(([key, system]) => ({
      system: SYSTEM_LABELS[key] || key,
      整合分数: Number(system?.local_integration_score || 0),
      同族成功率: Number(system?.same_family_success_rate || 0),
      损伤恢复率: Number(system?.lesion_recovery_rate || 0),
      编码分离度: Number(system?.encoding_separation || 0),
    }));
  }, [payload]);

  const gainRows = useMemo(
    () => [
      {
        item: '异质性整合增益',
        value: Number(payload?.gains?.heterogeneity_gain_vs_homogeneous || 0),
      },
      {
        item: '回放整合增益',
        value: Number(payload?.gains?.replay_gain_vs_heterogeneous || 0),
      },
      {
        item: '回放恢复增益',
        value: Number(payload?.gains?.replay_recovery_gain_vs_homogeneous || 0),
      },
      {
        item: '异质性专职增益',
        value: Number(payload?.gains?.specialization_gain_vs_homogeneous || 0),
      },
    ],
    [payload],
  );

  const selected = payload?.systems?.[selectedSystem] || payload?.systems?.heterogeneous_local_replay || {};
  const regionRows = selected?.region_params || [];

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 systems / headline_metrics / gains 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`局部脉冲脑区异质性基准 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setPayload(samplePayload);
    setSource('内置样例');
    setError('');
    setSelectedSystem('heterogeneous_local_replay');
  }

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(16,185,129,0.16), transparent 28%), radial-gradient(circle at bottom right, rgba(14,165,233,0.12), transparent 24%), rgba(2,6,23,0.66)',
        border: '1px solid rgba(16,185,129,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: '15px', fontWeight: 'bold' }}>五点三十九后、局部脉冲脑区异质性基准</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            只允许“上一步局部脉冲 + 局部可塑性 + 脑区异质性”，不引入任何显式全局控制器，直接看系统级整合、损伤恢复和编码分离是否还能涌现。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>局部整合分数</div>
          <div style={{ color: '#10b981', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.headline_metrics?.local_integration_score)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>同族成功率</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{pct(payload?.headline_metrics?.same_family_success_rate)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>损伤恢复率</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{pct(payload?.headline_metrics?.lesion_recovery_rate)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>编码分离度</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{pct(payload?.headline_metrics?.encoding_separation)}</div>
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
              background: value ? 'rgba(34,197,94,0.18)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(34,197,94,0.32)' : 'rgba(248,113,113,0.30)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '未成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1.15fr) minmax(0, 0.85fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 三组系统对照</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={systemRows} margin={{ top: 10, right: 12, left: 0, bottom: 16 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="system" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="整合分数" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="同族成功率" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="损伤恢复率" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="编码分离度" fill="#f97316" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 关键增益</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={gainRows} layout="vertical" margin={{ top: 10, right: 12, left: 20, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" />
              <YAxis type="category" dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.35)" width={88} />
              <Tooltip formatter={(value) => fmt(value)} contentStyle={{ background: 'rgba(15,23,42,0.96)', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px' }} />
              <Bar dataKey="value" fill="#22c55e" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 0.9fr) minmax(0, 1.1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 当前判读</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            {payload?.project_readout?.summary}
          </div>
          <div style={{ marginTop: '10px', color: '#86efac', fontSize: '12px', lineHeight: '1.8' }}>
            {payload?.project_readout?.next_question}
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px' }}>4. 脑区参数与读出</div>
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
                    color: selectedSystem === key ? '#052e16' : '#e2e8f0',
                    background: selectedSystem === key ? '#86efac' : 'transparent',
                  }}
                >
                  {SYSTEM_LABELS[key] || key}
                </button>
              ))}
            </div>
          </div>
          <div style={{ marginTop: '10px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '8px' }}>
            {regionRows.map((row) => (
              <div key={row.region} style={{ borderRadius: '12px', padding: '10px', background: 'rgba(15,23,42,0.45)', border: '1px solid rgba(148,163,184,0.16)' }}>
                <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold', marginBottom: '6px' }}>{row.region}</div>
                <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.75' }}>
                  <div>{`leak: ${fmt(row.leak, 2)}`}</div>
                  <div>{`threshold: ${fmt(row.threshold, 2)}`}</div>
                  <div>{`inhibition: ${fmt(row.inhibition, 2)}`}</div>
                  <div>{`plasticity: ${fmt(row.plasticity, 3)}`}</div>
                  <div>{`feedback_gain: ${fmt(row.feedback_gain, 2)}`}</div>
                  <div>{`tonic_bias: ${fmt(row.tonic_bias, 2)}`}</div>
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.7' }}>
            {`当前读出阈值: ${fmt(selected?.decision_threshold, 4)} | 判定规则: ${
              selected?.same_if_margin_below ? 'margin 更低判为同族' : 'margin 更高判为同族'
            }`}
          </div>
        </div>
      </div>
    </div>
  );
}
