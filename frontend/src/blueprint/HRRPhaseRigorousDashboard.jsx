import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const DEFAULT_PAYLOAD = {
  config: {
    d_values: [128, 256, 512],
    m_values: [8, 16, 32, 64],
    n_dict: 256,
    trials: 40,
    seed: 20260306,
  },
  fit: {
    c_constant_median: 0.7570216592192924,
  },
  hrr_grid: [
    { d: 128, m: 8, n_dict: 256, trials: 40, error_rate: 0.0, mean_margin: 0.46483393876918566, predicted_bound: 0.0014002800840280118 },
    { d: 128, m: 16, n_dict: 256, trials: 40, error_rate: 0.125, mean_margin: 0.2818806077596073, predicted_bound: 0.5975545342704237 },
    { d: 128, m: 32, n_dict: 256, trials: 40, error_rate: 0.45, mean_margin: 0.012694170722269887, predicted_bound: 1.0 },
    { d: 128, m: 64, n_dict: 256, trials: 40, error_rate: 0.75, mean_margin: -0.4326916093049489, predicted_bound: 1.0 },
    { d: 256, m: 8, n_dict: 256, trials: 40, error_rate: 0.0, mean_margin: 0.6286759018341216, predicted_bound: 7.689350249903907e-9 },
    { d: 256, m: 16, n_dict: 256, trials: 40, error_rate: 0.0, mean_margin: 0.4468488315989595, predicted_bound: 0.0014002800840280118 },
    { d: 256, m: 32, n_dict: 256, trials: 40, error_rate: 0.175, mean_margin: 0.2563550451271779, predicted_bound: 0.5975545342704237 },
    { d: 256, m: 64, n_dict: 256, trials: 40, error_rate: 0.45, mean_margin: 0.0425181421869955, predicted_bound: 1.0 },
    { d: 512, m: 8, n_dict: 256, trials: 40, error_rate: 0.0, mean_margin: 0.7447407179024192, predicted_bound: 2.3186708731645984e-19 },
    { d: 512, m: 16, n_dict: 256, trials: 40, error_rate: 0.0, mean_margin: 0.6195771822440138, predicted_bound: 7.689350249903907e-9 },
    { d: 512, m: 32, n_dict: 256, trials: 40, error_rate: 0.025, mean_margin: 0.4972799235152732, predicted_bound: 0.0014002800840280118 },
    { d: 512, m: 64, n_dict: 256, trials: 40, error_rate: 0.2, mean_margin: 0.23160067722739686, predicted_bound: 0.5975545342704237 },
  ],
  phase_cases: [
    { case: 'sync_same_freq', numeric_gate: 0.5, analytic_gate: 0.5, abs_diff: 0.0 },
    { case: 'phase_shift_pi_over_2', numeric_gate: -3.794298208958935e-16, analytic_gate: 3.061616997868383e-17, abs_diff: 4.1004599087457736e-16 },
    { case: 'freq_mismatch', numeric_gate: -0.0490313291130884, analytic_gate: -0.049131575207350266, abs_diff: 0.00010024609426186742 },
  ],
};

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function toPercent(x) {
  return `${(x * 100).toFixed(2)}%`;
}

function errorToColor(errorRate) {
  const t = clamp01(errorRate);
  const r = Math.round(16 + 239 * t);
  const g = Math.round(185 - 120 * t);
  const b = Math.round(129 - 95 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function formatCase(caseName) {
  if (caseName === 'sync_same_freq') return '同频同相';
  if (caseName === 'phase_shift_pi_over_2') return '同频相差 π/2';
  if (caseName === 'freq_mismatch') return '频差失配';
  return caseName;
}

function isValidPayload(payload) {
  return Boolean(payload && Array.isArray(payload.hrr_grid) && Array.isArray(payload.phase_cases));
}

const BubblePoint = (props) => {
  const { cx, cy, payload } = props;
  const radius = 7 + (payload?.error_rate || 0) * 14;
  const fill = errorToColor(payload?.error_rate || 0);
  return (
    <g>
      <circle cx={cx} cy={cy} r={radius} fill={fill} fillOpacity={0.82} stroke="rgba(255,255,255,0.85)" strokeWidth={1} />
    </g>
  );
};

export default function HRRPhaseRigorousDashboard() {
  const [payload, setPayload] = useState(DEFAULT_PAYLOAD);
  const [selected, setSelected] = useState(DEFAULT_PAYLOAD.hrr_grid[0]);
  const [dataSource, setDataSource] = useState('内置样本 (2026-03-06)');
  const [error, setError] = useState('');

  const scatterData = useMemo(() => payload.hrr_grid || [], [payload]);

  const trendData = useMemo(() => {
    const rows = (payload.hrr_grid || []).map((row) => ({
      ...row,
      d_over_m: row.m > 0 ? row.d / row.m : 0,
      tag: `d=${row.d},m=${row.m}`,
    }));
    rows.sort((a, b) => {
      if (a.d_over_m !== b.d_over_m) return a.d_over_m - b.d_over_m;
      return a.d - b.d;
    });
    return rows;
  }, [payload]);

  const phaseData = useMemo(
    () => (payload.phase_cases || []).map((x) => ({
      label: formatCase(x.case),
      numeric: x.numeric_gate,
      analytic: x.analytic_gate,
      abs_diff: x.abs_diff,
    })),
    [payload]
  );

  const onUploadJson = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      if (!isValidPayload(parsed)) {
        throw new Error('JSON 缺少 hrr_grid 或 phase_cases');
      }
      setPayload(parsed);
      setSelected(parsed.hrr_grid?.[0] || null);
      setDataSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`导入失败：${err?.message || '未知错误'}`);
    }
  };

  return (
    <div style={{ marginTop: '18px', padding: '16px', borderRadius: '12px', background: 'rgba(15,23,42,0.45)', border: '1px solid rgba(99,102,241,0.25)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
        <div>
          <div style={{ color: '#c4b5fd', fontSize: '14px', fontWeight: 'bold' }}>严格数学实测看板：HRR 容量与相位门控</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '4px' }}>数据源：{dataSource}</div>
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <label style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '8px', padding: '6px 10px', cursor: 'pointer' }}>
            导入 JSON
            <input type="file" accept="application/json" onChange={onUploadJson} style={{ display: 'none' }} />
          </label>
          <button
            type="button"
            onClick={() => {
              setPayload(DEFAULT_PAYLOAD);
              setSelected(DEFAULT_PAYLOAD.hrr_grid[0]);
              setDataSource('内置样本 (2026-03-06)');
              setError('');
            }}
            style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '8px', padding: '6px 10px', background: 'transparent', cursor: 'pointer' }}
          >
            重置
          </button>
        </div>
      </div>

      {error && <div style={{ color: '#fca5a5', fontSize: '11px', marginTop: '8px' }}>{error}</div>}

      <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: '1fr', gap: '12px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '12px' }}>
          <div style={{ minHeight: '250px', border: '1px solid rgba(148,163,184,0.2)', borderRadius: '10px', padding: '8px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1) 容量相图（x=M, y=d, 气泡大小/颜色=错误率）</div>
            <ResponsiveContainer width="100%" height={220}>
              <ScatterChart margin={{ top: 8, right: 12, left: 6, bottom: 8 }}>
                <CartesianGrid stroke="rgba(148,163,184,0.2)" strokeDasharray="3 3" />
                <XAxis type="number" dataKey="m" name="M" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <YAxis type="number" dataKey="d" name="d" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
                <Tooltip
                  cursor={{ strokeDasharray: '4 4' }}
                  contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.45)', borderRadius: '8px' }}
                  formatter={(value, name) => {
                    if (name === 'error_rate') return [toPercent(value), '错误率'];
                    return [value, name];
                  }}
                />
                <Scatter
                  data={scatterData}
                  shape={<BubblePoint />}
                  onClick={(entry) => {
                    const next = entry?.payload || entry;
                    if (next) setSelected(next);
                  }}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          <div style={{ border: '1px solid rgba(148,163,184,0.2)', borderRadius: '10px', padding: '10px', background: 'rgba(2,6,23,0.45)' }}>
            <div style={{ color: '#e2e8f0', fontSize: '12px', fontWeight: 'bold' }}>当前选中点</div>
            {selected ? (
              <div style={{ marginTop: '8px', display: 'grid', gap: '6px', fontSize: '12px', color: '#cbd5e1' }}>
                <div><span style={{ color: '#a5b4fc' }}>d:</span> {selected.d}</div>
                <div><span style={{ color: '#a5b4fc' }}>M:</span> {selected.m}</div>
                <div><span style={{ color: '#a5b4fc' }}>d/M:</span> {(selected.d / selected.m).toFixed(2)}</div>
                <div><span style={{ color: '#a5b4fc' }}>实测错误率:</span> {toPercent(selected.error_rate)}</div>
                <div><span style={{ color: '#a5b4fc' }}>理论上界:</span> {toPercent(selected.predicted_bound)}</div>
                <div><span style={{ color: '#a5b4fc' }}>平均 margin:</span> {selected.mean_margin.toFixed(4)}</div>
              </div>
            ) : (
              <div style={{ marginTop: '8px', color: '#94a3b8', fontSize: '12px' }}>点击左侧气泡查看详情。</div>
            )}
            <div style={{ marginTop: '10px', fontSize: '11px', color: '#94a3b8', lineHeight: '1.6' }}>
              推理链：固定词典规模 N 下，随着 d/M 增大，实测错误率下降，与指数上界趋势一致。
            </div>
          </div>
        </div>

        <div style={{ minHeight: '250px', border: '1px solid rgba(148,163,184,0.2)', borderRadius: '10px', padding: '8px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2) 理论-实测对比（按 d/M 排序）</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trendData} margin={{ top: 8, right: 14, left: 6, bottom: 8 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.2)" strokeDasharray="3 3" />
              <XAxis type="number" dataKey="d_over_m" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={['auto', 'auto']} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.45)', borderRadius: '8px' }}
                formatter={(value, name) => [toPercent(value), name === 'error_rate' ? '实测错误率' : '理论上界']}
                labelFormatter={(label) => `d/M = ${Number(label).toFixed(2)}`}
              />
              <Line type="monotone" dataKey="error_rate" name="error_rate" stroke="#fb7185" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="predicted_bound" name="predicted_bound" stroke="#38bdf8" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ minHeight: '250px', border: '1px solid rgba(148,163,184,0.2)', borderRadius: '10px', padding: '8px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3) 相位门控示波（数值积分 vs 解析积分）</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={phaseData} margin={{ top: 8, right: 14, left: 6, bottom: 8 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.2)" strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.45)', borderRadius: '8px' }}
                formatter={(value, name) => [Number(value).toExponential(4), name]}
              />
              <Bar dataKey="numeric" name="numeric_gate" fill="#22d3ee" radius={[4, 4, 0, 0]} />
              <Bar dataKey="analytic" name="analytic_gate" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Line type="monotone" dataKey="abs_diff" name="abs_diff" stroke="#f43f5e" strokeWidth={2} dot={{ r: 2 }} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
