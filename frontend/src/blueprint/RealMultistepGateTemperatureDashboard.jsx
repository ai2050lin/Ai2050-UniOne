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
import samplePayload from './data/real_multistep_memory_gate_temperature_scan_sample.json';

function isValidPayload(payload) {
  return Boolean(payload && payload.systems && Array.isArray(payload.tau_rows));
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`;
}

function labelOf(name) {
  if (name === 'trace_gated_local') return 'trace';
  if (name === 'single_anchor_beta_086') return '单锚点 beta=0.86';
  if (name.startsWith('gated_triple_tau_')) {
    return `门控三锚点 tau=${(Number(name.split('_').at(-1)) / 100).toFixed(2)}`;
  }
  return name;
}

export default function RealMultistepGateTemperatureDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const tauRows = Array.isArray(payload?.tau_rows) ? payload.tau_rows : [];
  const best = payload?.best || {};
  const gains = payload?.gains || {};
  const hypotheses = payload?.hypotheses || {};

  const curveRows = useMemo(
    () =>
      tauRows.map((row) => ({
        tau: Number(row.tau_g || 0),
        mean_closure: Number(row.mean_closure_score || 0),
        max_length: Number(row.max_length_score || 0),
        mean_retention: Number(row.mean_retention_score || 0),
      })),
    [tauRows]
  );

  const gateRows = useMemo(
    () =>
      tauRows.map((row) => ({
        tau: Number(row.tau_g || 0),
        entropy: Number(row.mean_gate_entropy || 0),
        peak: Number(row.mean_gate_peak || 0),
        drop: Number(row.closure_relative_drop || 0),
      })),
    [tauRows]
  );

  const lengthBestRows = useMemo(
    () =>
      Object.entries(gains?.per_length_best_tau || {}).map(([length, row]) => ({
        length: `L=${length}`,
        best_tau: Number(row.best_tau_g || 0),
        gain_vs_tau_100: Number(row.gain_vs_tau_100 || 0),
        gain_vs_single_anchor: Number(row.gain_vs_single_anchor || 0),
      })),
    [gains]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 systems 或 tau_rows 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`门控温度扫描 JSON 导入失败: ${err?.message || '未知错误'}`);
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
          'radial-gradient(circle at top left, rgba(245,158,11,0.14), transparent 28%), radial-gradient(circle at top right, rgba(56,189,248,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(245,158,11,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>门控温度 tau_g 扫描</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            直接看门控变硬还是变软，对平均闭环、最长任务和时间尺度选择性分别有什么影响。
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>平均闭环最优</div>
          <div style={{ color: '#22c55e', fontSize: '16px', fontWeight: 'bold' }}>{labelOf(best?.best_mean_closure?.system || '-')}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`分数 ${fmt(best?.best_mean_closure?.mean_closure_score)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最长任务最优</div>
          <div style={{ color: '#60a5fa', fontSize: '16px', fontWeight: 'bold' }}>{labelOf(best?.best_max_length?.system || '-')}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`L_max 分数 ${fmt(best?.best_max_length?.max_length_score)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>选择性最强</div>
          <div style={{ color: '#f59e0b', fontSize: '16px', fontWeight: 'bold' }}>{labelOf(best?.best_gate_selectivity?.system || '-')}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`gate peak ${fmt(best?.best_gate_selectivity?.mean_gate_peak)}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>核心增益</div>
          <div style={{ color: '#f472b6', fontSize: '16px', fontWeight: 'bold' }}>{`vs 单锚点 ${fmt(gains?.best_max_vs_single_anchor)}`}</div>
          <div style={{ color: '#cbd5e1', fontSize: '11px', marginTop: '4px' }}>{`vs tau=1.0 ${fmt(gains?.best_max_vs_tau_100)}`}</div>
        </div>
      </div>

      <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {Object.entries(hypotheses).map(([key, value]) => (
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
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 温度对闭环与保留的影响</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={curveRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="tau" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="mean_closure" name="平均闭环" stroke="#22c55e" strokeWidth={2.1} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="max_length" name="最长任务" stroke="#38bdf8" strokeWidth={2.1} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="mean_retention" name="平均保留" stroke="#f472b6" strokeWidth={2.1} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 温度对门控选择性的影响</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={gateRows} margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="tau" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Line type="monotone" dataKey="entropy" name="gate entropy" stroke="#f59e0b" strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="peak" name="gate peak" stroke="#ef4444" strokeWidth={2} dot={{ r: 2.5 }} />
              <Line type="monotone" dataKey="drop" name="相对衰减" stroke="#94a3b8" strokeWidth={2} dot={{ r: 2.5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(320px, 0.9fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 各长度的最优温度</div>
          <ResponsiveContainer width="100%" height={270}>
            <BarChart data={lengthBestRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="length" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="best_tau" name="最优 tau_g" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="gain_vs_tau_100" name="相对 tau=1.0 增益" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="gain_vs_single_anchor" name="相对单锚点增益" fill="#22c55e" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px', display: 'grid', gap: '10px' }}>
          <div>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 当前结论</div>
            <div style={{ color: '#e2e8f0', fontSize: '12px', lineHeight: '1.8' }}>
              这轮扫描说明，门控温度确实会系统性改变时间尺度选择性，但当前最优点仍落在
              <span style={{ color: '#f59e0b', fontWeight: 'bold' }}> tau_g = 1.0 </span>
              附近。更硬的门控会提高 `gate peak`、降低 `gate entropy`，但没有进一步抬高当前任务上的最长闭环表现。
            </div>
          </div>
          <div>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>5. 含义</div>
            <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.8' }}>
              当前系统的短板不是“门控不够硬”，而更像“门控输入特征还不够好”。也就是温度已经能控制选择性，但还没有带来更优的决策边界。
            </div>
          </div>
          <div>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>6. 下一步</div>
            <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.8' }}>
              下一步应该做长度依赖温度 `tau_g(L)` 或上下文依赖温度，而不是继续死扫固定常数；同时要把门控输入从当前局部状态扩到更强的任务阶段特征。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
