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
import samplePayload from './data/shared_atom_causal_unification_benchmark_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && payload.headline_metrics && payload.gains);
}

function fmt(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

export default function SharedAtomCausalUnificationDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const systemRows = useMemo(() => {
    const systems = payload?.systems || {};
    return [
      {
        item: '概念准确率',
        共享原子: Number(systems?.unified_shared_atoms?.metrics?.concept_accuracy || 0),
        独立字典: Number(systems?.independent_atoms?.metrics?.concept_accuracy || 0),
        稠密基线: Number(systems?.dense_baseline?.metrics?.concept_accuracy || 0),
      },
      {
        item: '关系准确率',
        共享原子: Number(systems?.unified_shared_atoms?.metrics?.relation_accuracy || 0),
        独立字典: Number(systems?.independent_atoms?.metrics?.relation_accuracy || 0),
        稠密基线: Number(systems?.dense_baseline?.metrics?.relation_accuracy || 0),
      },
      {
        item: '噪声恢复率',
        共享原子: Number(systems?.unified_shared_atoms?.metrics?.recovery_accuracy || 0),
        独立字典: Number(systems?.independent_atoms?.metrics?.recovery_accuracy || 0),
        稠密基线: Number(systems?.dense_baseline?.metrics?.recovery_accuracy || 0),
      },
    ];
  }, [payload]);

  const unifiedDropRows = useMemo(() => {
    const ablations = payload?.systems?.unified_shared_atoms?.ablations || {};
    return ['shared_top', 'concept_only', 'relation_only', 'random_control'].map((key) => ({
      mode:
        key === 'shared_top'
          ? '共享原子'
          : key === 'concept_only'
            ? '概念侧原子'
            : key === 'relation_only'
              ? '关系侧原子'
              : '随机对照',
      联动掉分: Number(ablations?.[key]?.drops?.joint_drop || 0),
      恢复掉分: Number(ablations?.[key]?.drops?.recovery_drop || 0),
      概念掉分: Number(ablations?.[key]?.drops?.concept_drop || 0),
      关系掉分: Number(ablations?.[key]?.drops?.relation_drop || 0),
    }));
  }, [payload]);

  const usageRows = useMemo(() => {
    const usage = payload?.systems?.unified_shared_atoms?.usage || {};
    const concept = usage?.concept_usage || [];
    const relation = usage?.relation_usage || [];
    return concept.map((value, idx) => ({
      atom: `a${idx}`,
      概念使用: Number(value || 0),
      关系使用: Number(relation?.[idx] || 0),
    }));
  }, [payload]);

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
      setError(`共享原子因果桥 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(56,189,248,0.14), transparent 26%), radial-gradient(circle at bottom right, rgba(34,197,94,0.10), transparent 24%), rgba(2,6,23,0.66)',
        border: '1px solid rgba(56,189,248,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>五点三十三续、共享原子因果桥</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            不再只看共享字典相关性，直接做共享原子因果消融，检查同一批低层原子是否会同时打掉概念解码、关系解码和噪声恢复。
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
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>跨维度相关性</div>
          <div style={{ color: '#38bdf8', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.headline_metrics?.cross_dim_corr)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>共享原子联动掉分</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.headline_metrics?.shared_joint_drop)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>共享原子恢复掉分</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.headline_metrics?.shared_recovery_drop)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>共享原子相对随机增益</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{fmt(payload?.gains?.shared_vs_random_joint_drop)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>统一系统恢复率</div>
          <div style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 'bold' }}>{pct(payload?.headline_metrics?.unified_recovery_accuracy)}</div>
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
              background: value ? 'rgba(56,189,248,0.18)' : 'rgba(248,113,113,0.18)',
              border: `1px solid ${value ? 'rgba(56,189,248,0.35)' : 'rgba(248,113,113,0.3)'}`,
            }}
          >
            {`${key}: ${value ? '成立' : '不成立'}`}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 系统表现</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={systemRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="item" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="共享原子" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="独立字典" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="稠密基线" fill="#94a3b8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 共享原子因果消融</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={unifiedDropRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="mode" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 0.4]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="联动掉分" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="恢复掉分" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="概念掉分" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="关系掉分" fill="#f97316" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 共享原子使用图谱</div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={usageRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
            <XAxis dataKey="atom" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={1} />
            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
            <Legend />
            <Line type="monotone" dataKey="概念使用" stroke="#38bdf8" strokeWidth={2.2} dot={false} />
            <Line type="monotone" dataKey="关系使用" stroke="#22c55e" strokeWidth={2.2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 当前判断</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            这一步把“共享字典相关性”推进成了“共享原子因果桥”。同一批共享原子被打掉后，概念、关系和恢复三路一起明显下跌，
            而随机原子和单侧原子做不到这个幅度，说明同源性开始进入因果证据链。
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>5. 下一步</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            下一步应该把这种共享原子因果消融搬到真实模型里，并把恢复链和中观冗余场一起纳入同一套因果读数，确认真实 DNN 里是否也存在同样的联动破坏模式。
          </div>
        </div>
      </div>
    </div>
  );
}
