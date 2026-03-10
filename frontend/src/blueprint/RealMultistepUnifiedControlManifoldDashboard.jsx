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
import samplePayload from './data/real_multistep_unified_control_manifold_benchmark_sample.json';

const SYSTEM_META = {
  single_anchor_beta_086: { label: '单锚点 beta=0.86', color: '#94a3b8' },
  learnable_state_machine_h12: { label: '可学习状态机', color: '#f59e0b' },
  unified_control_manifold_h12: { label: '统一控制流形', color: '#14b8a6' },
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

function systemLabel(name) {
  return SYSTEM_META[name]?.label || name || '-';
}

export default function RealMultistepUnifiedControlManifoldDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const lineRows = useMemo(() => {
    const systems = payload?.systems || {};
    const lengths = systems?.unified_control_manifold_h12?.global_summary?.lengths || [];
    return lengths.map((length, idx) => ({
      length: `L=${length}`,
      单锚点: Number(systems?.single_anchor_beta_086?.global_summary?.unified_curve?.[idx] || 0),
      可学习状态机: Number(systems?.learnable_state_machine_h12?.global_summary?.unified_curve?.[idx] || 0),
      统一控制流形: Number(systems?.unified_control_manifold_h12?.global_summary?.unified_curve?.[idx] || 0),
      回退恢复: Number(systems?.unified_control_manifold_h12?.global_summary?.recovery_curve?.[idx] || 0),
    }));
  }, [payload]);

  const maxLengthBars = useMemo(() => {
    const systems = payload?.systems || {};
    return [
      {
        item: '最大长度综合分',
        单锚点: Number(systems?.single_anchor_beta_086?.per_length?.['32']?.unified_real_task_score || 0),
        可学习状态机: Number(systems?.learnable_state_machine_h12?.per_length?.['32']?.unified_real_task_score || 0),
        统一控制流形: Number(systems?.unified_control_manifold_h12?.per_length?.['32']?.unified_real_task_score || 0),
      },
      {
        item: 'episode 成功率',
        单锚点: Number(systems?.single_anchor_beta_086?.per_length?.['32']?.summary?.overall_episode_success?.mean || 0),
        可学习状态机: Number(systems?.learnable_state_machine_h12?.per_length?.['32']?.summary?.overall_episode_success?.mean || 0),
        统一控制流形: Number(systems?.unified_control_manifold_h12?.per_length?.['32']?.summary?.overall_episode_success?.mean || 0),
      },
      {
        item: '回退恢复率',
        单锚点: Number(systems?.single_anchor_beta_086?.per_length?.['32']?.summary?.rollback_recovery_rate?.mean || 0),
        可学习状态机: Number(systems?.learnable_state_machine_h12?.per_length?.['32']?.summary?.rollback_recovery_rate?.mean || 0),
        统一控制流形: Number(systems?.unified_control_manifold_h12?.per_length?.['32']?.summary?.rollback_recovery_rate?.mean || 0),
      },
      {
        item: '旧概念保留率',
        单锚点: Number(systems?.single_anchor_beta_086?.per_length?.['32']?.summary?.retention_after_phase2?.mean || 0),
        可学习状态机: Number(systems?.learnable_state_machine_h12?.per_length?.['32']?.summary?.retention_after_phase2?.mean || 0),
        统一控制流形: Number(systems?.unified_control_manifold_h12?.per_length?.['32']?.summary?.retention_after_phase2?.mean || 0),
      },
    ];
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
      setError(`真实多步统一控制流形 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(20,184,166,0.15), transparent 28%), radial-gradient(circle at top right, rgba(59,130,246,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(20,184,166,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>二点四十四、真实多步统一控制流形</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把低维统一控制流形接回真实多步 episode，统一调制阶段状态、记忆门控和失败回退，评价继续外部化到真实任务成功率、
            回退恢复率、旧概念保留率和长程稳定性。
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
          <div style={{ color: '#14b8a6', fontSize: '16px', fontWeight: 'bold' }}>{systemLabel(payload?.best_system?.system)}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`平均综合分 ${fmt(payload?.best_system?.mean_unified_score)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最大长度相对单锚点增益</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.gains?.unified_vs_single_at_max_length)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最大长度相对状态机增益</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.gains?.unified_vs_learnable_at_max_length)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>相对状态机的恢复增益</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.gains?.unified_recovery_gain_vs_learnable)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最大长度恢复 / 保留</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>
            {`${pct(payload?.headline_metrics?.max_length_recovery_rate)} / ${pct(payload?.headline_metrics?.max_length_retention)}`}
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
              background: value ? 'rgba(34,197,94,0.18)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(34,197,94,0.3)' : 'rgba(248,113,113,0.3)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '不成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 长度曲线</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lineRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="单锚点" stroke={SYSTEM_META.single_anchor_beta_086.color} strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="可学习状态机" stroke={SYSTEM_META.learnable_state_machine_h12.color} strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="统一控制流形" stroke={SYSTEM_META.unified_control_manifold_h12.color} strokeWidth={2.4} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="回退恢复" stroke="#38bdf8" strokeDasharray="5 4" strokeWidth={2} dot={{ r: 2.5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 最大长度对比</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={maxLengthBars} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="单锚点" fill={SYSTEM_META.single_anchor_beta_086.color} radius={[4, 4, 0, 0]} />
              <Bar dataKey="可学习状态机" fill={SYSTEM_META.learnable_state_machine_h12.color} radius={[4, 4, 0, 0]} />
              <Bar dataKey="统一控制流形" fill={SYSTEM_META.unified_control_manifold_h12.color} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>3. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            这一步说明统一控制流形不是纯 toy 里的协议壳。接回真实多步 episode 后，它已经能在最大长度任务上同时压过单锚点基线与旧状态机，
            主增益来自更稳定的回退恢复和更平坦的旧概念保留。
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 下一步</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            下一步应该继续压缩这套统一结构，把共享基底、关系协议和门控调制合并到更小的低维桥接坐标里，同时把回退恢复接回真实工具接口和脑侧候选约束。
          </div>
        </div>
      </div>
    </div>
  );
}
