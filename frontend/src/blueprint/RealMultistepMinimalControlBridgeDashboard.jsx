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
import samplePayload from './data/real_multistep_minimal_control_bridge_benchmark_sample.json';

const SYSTEM_META = {
  single_anchor_beta_086: { label: '单锚点', color: '#94a3b8' },
  unified_control_manifold_h12: { label: '4 维统一流形', color: '#14b8a6' },
  minimal_control_bridge_h12: { label: '2 维最小控制桥', color: '#f97316' },
};

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && Array.isArray(payload.ranking) && payload.gains);
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

export default function RealMultistepMinimalControlBridgeDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const lengthRows = useMemo(() => {
    const systems = payload?.systems || {};
    const lengths = systems?.minimal_control_bridge_h12?.global_summary?.lengths || [];
    return lengths.map((length, idx) => ({
      length: `L=${length}`,
      单锚点: Number(systems?.single_anchor_beta_086?.global_summary?.bridge_curve?.[idx] || 0),
      统一流形: Number(systems?.unified_control_manifold_h12?.global_summary?.bridge_curve?.[idx] || 0),
      最小控制桥: Number(systems?.minimal_control_bridge_h12?.global_summary?.bridge_curve?.[idx] || 0),
      脑侧对齐: Number(systems?.minimal_control_bridge_h12?.global_summary?.brain_curve?.[idx] || 0),
    }));
  }, [payload]);

  const maxLengthBars = useMemo(() => {
    const systems = payload?.systems || {};
    return [
      {
        item: '真实任务综合分',
        单锚点: Number(systems?.single_anchor_beta_086?.per_length?.['32']?.summary?.real_task_score?.mean || 0),
        统一流形: Number(systems?.unified_control_manifold_h12?.per_length?.['32']?.summary?.real_task_score?.mean || 0),
        最小控制桥: Number(systems?.minimal_control_bridge_h12?.per_length?.['32']?.summary?.real_task_score?.mean || 0),
      },
      {
        item: '回退恢复率',
        单锚点: Number(systems?.single_anchor_beta_086?.per_length?.['32']?.summary?.rollback_recovery_rate?.mean || 0),
        统一流形: Number(systems?.unified_control_manifold_h12?.per_length?.['32']?.summary?.rollback_recovery_rate?.mean || 0),
        最小控制桥: Number(systems?.minimal_control_bridge_h12?.per_length?.['32']?.summary?.rollback_recovery_rate?.mean || 0),
      },
      {
        item: '旧概念保留率',
        单锚点: Number(systems?.single_anchor_beta_086?.per_length?.['32']?.summary?.retention_after_phase2?.mean || 0),
        统一流形: Number(systems?.unified_control_manifold_h12?.per_length?.['32']?.summary?.retention_after_phase2?.mean || 0),
        最小控制桥: Number(systems?.minimal_control_bridge_h12?.per_length?.['32']?.summary?.retention_after_phase2?.mean || 0),
      },
      {
        item: '脑侧对齐',
        单锚点: Number(systems?.single_anchor_beta_086?.per_length?.['32']?.summary?.brain_alignment_score?.mean || 0),
        统一流形: Number(systems?.unified_control_manifold_h12?.per_length?.['32']?.summary?.brain_alignment_score?.mean || 0),
        最小控制桥: Number(systems?.minimal_control_bridge_h12?.per_length?.['32']?.summary?.brain_alignment_score?.mean || 0),
      },
      {
        item: '控制桥综合分',
        单锚点: Number(systems?.single_anchor_beta_086?.per_length?.['32']?.control_bridge_score || 0),
        统一流形: Number(systems?.unified_control_manifold_h12?.per_length?.['32']?.control_bridge_score || 0),
        最小控制桥: Number(systems?.minimal_control_bridge_h12?.per_length?.['32']?.control_bridge_score || 0),
      },
    ];
  }, [payload]);

  const timelineRows = useMemo(() => {
    const timeline = payload?.systems?.minimal_control_bridge_h12?.per_length?.['32']?.timeline || {};
    const positions = timeline?.positions || [];
    return positions.map((position, idx) => ({
      step: `t${position}`,
      触发率: Number(timeline?.trigger_rate_curve?.[idx] || 0),
      恢复率: Number(timeline?.recovery_rate_curve?.[idx] || 0),
      回退前置信度: Number(timeline?.confidence_before_curve?.[idx] || 0),
      回退后置信度: Number(timeline?.confidence_after_curve?.[idx] || 0),
    }));
  }, [payload]);

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 systems / ranking / gains 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`最小控制桥 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(249,115,22,0.16), transparent 26%), radial-gradient(circle at bottom right, rgba(251,191,36,0.10), transparent 24%), rgba(2,6,23,0.66)',
        border: '1px solid rgba(249,115,22,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>五点三十三补、最小统一控制桥</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把 4 维统一控制流形继续压成 2 维最小控制桥，直接比较压缩前后在真实多步 episode 上的综合分、回退恢复、旧概念保留、
            脑侧约束对齐和回退时间线。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>当前最优系统</div>
          <div style={{ color: '#fb923c', fontSize: '16px', fontWeight: 'bold' }}>{payload?.best_system?.system || '-'}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`平均控制桥分 ${fmt(payload?.best_system?.mean_bridge_score)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最大长度相对统一流形增益</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.gains?.minimal_vs_unified_at_max_length)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>脑侧对齐增益</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.gains?.minimal_brain_gain_vs_unified)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>控制压缩增益</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.gains?.minimal_compaction_gain_vs_unified)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最大长度恢复 / 保留 / 脑侧对齐</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>
            {`${pct(payload?.headline_metrics?.max_length_recovery_rate)} / ${pct(payload?.headline_metrics?.max_length_retention)} / ${pct(payload?.headline_metrics?.max_length_brain_alignment)}`}
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
              color: value ? '#fed7aa' : '#fee2e2',
              background: value ? 'rgba(249,115,22,0.18)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(249,115,22,0.35)' : 'rgba(248,113,113,0.3)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '不成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#ffedd5', fontSize: '12px', marginBottom: '6px' }}>1. 长度曲线</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lengthRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="单锚点" stroke={SYSTEM_META.single_anchor_beta_086.color} strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="统一流形" stroke={SYSTEM_META.unified_control_manifold_h12.color} strokeWidth={2.2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="最小控制桥" stroke={SYSTEM_META.minimal_control_bridge_h12.color} strokeWidth={2.6} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="脑侧对齐" stroke="#facc15" strokeDasharray="5 4" strokeWidth={2} dot={{ r: 2.5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#ffedd5', fontSize: '12px', marginBottom: '6px' }}>2. 最大长度指标</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={maxLengthBars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="单锚点" fill={SYSTEM_META.single_anchor_beta_086.color} radius={[4, 4, 0, 0]} />
              <Bar dataKey="统一流形" fill={SYSTEM_META.unified_control_manifold_h12.color} radius={[4, 4, 0, 0]} />
              <Bar dataKey="最小控制桥" fill={SYSTEM_META.minimal_control_bridge_h12.color} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#ffedd5', fontSize: '12px', marginBottom: '6px' }}>3. 最大长度回退时间线</div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={timelineRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="step" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={3} />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
            <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
            <Legend />
            <Line type="monotone" dataKey="触发率" stroke="#38bdf8" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="恢复率" stroke="#22c55e" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="回退前置信度" stroke="#f97316" strokeDasharray="5 4" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="回退后置信度" stroke="#facc15" strokeDasharray="3 3" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#ffedd5', fontSize: '12px', marginBottom: '8px' }}>4. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            最小控制桥不是单纯减参数。它把共享基底、协议与门控压进更小坐标后，最大长度综合分继续上升，脑侧对齐读数明显改善，
            同时回退恢复率基本保住，说明统一结构可以继续缩而不是必须越堆越厚。
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#ffedd5', fontSize: '12px', marginBottom: '8px' }}>5. 下一步</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            下一步要把这套 2 维桥接坐标接回真实工具调用和环境反馈，把时间线里的高触发区变成在线可干预节点，再继续区分哪些维度属于共享底座，
            哪些只是关系协议或门控壳。
          </div>
        </div>
      </div>
    </div>
  );
}
