import React, { useMemo, useState } from 'react';
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const DEFAULT_PAYLOAD = {
  meta: {
    ts: '2026-03-07 15:03:11',
    model_id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
  },
  base: {
    base_gap_red_minus_green: 2.32790740331014,
  },
  best: {
    layer: 27,
    k: 64,
    scale: -4.0,
    new_gap: -0.06173102060953772,
    pair_flip_rate_from_base: 0.6666666666666666,
    anchor_retention: 0.8333333333333334,
    target_reversal_strong: true,
  },
  min_k_reversal_anchor80_soft: 32,
  min_k_reversal_anchor80_strong: 64,
  trials: [
    { layer: 27, k: 32, scale: -4.0, new_gap: 0.211, pair_flip_rate_from_base: 0.3333, anchor_retention: 0.8333, target_reversal_soft: true, target_reversal_strong: false },
    { layer: 27, k: 64, scale: -4.0, new_gap: -0.0617, pair_flip_rate_from_base: 0.6667, anchor_retention: 0.8333, target_reversal_soft: true, target_reversal_strong: true },
    { layer: 27, k: 128, scale: -4.0, new_gap: -0.1020, pair_flip_rate_from_base: 0.6667, anchor_retention: 0.6667, target_reversal_soft: true, target_reversal_strong: false },
  ],
};

function isValidPayload(payload) {
  return Boolean(payload && payload.base && Array.isArray(payload.trials));
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`;
}

export default function RealModelChannelEditDashboard() {
  const [payload, setPayload] = useState(DEFAULT_PAYLOAD);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');

  const curveByK = useMemo(() => {
    const rows = Array.isArray(payload?.trials) ? payload.trials : [];
    const map = new Map();
    rows.forEach((r) => {
      const k = Number(r.k || 0);
      if (!map.has(k)) {
        map.set(k, {
          k,
          best_pair_flip: Number(r.pair_flip_rate_from_base || 0),
          best_anchor_retention: Number(r.anchor_retention || 0),
          min_new_gap: Number(r.new_gap || 0),
          strong_exists: Boolean(r.target_reversal_strong),
        });
        return;
      }
      const cur = map.get(k);
      cur.best_pair_flip = Math.max(cur.best_pair_flip, Number(r.pair_flip_rate_from_base || 0));
      cur.best_anchor_retention = Math.max(cur.best_anchor_retention, Number(r.anchor_retention || 0));
      cur.min_new_gap = Math.min(cur.min_new_gap, Number(r.new_gap || 0));
      cur.strong_exists = cur.strong_exists || Boolean(r.target_reversal_strong);
      map.set(k, cur);
    });
    return Array.from(map.values()).sort((a, b) => a.k - b.k);
  }, [payload]);

  const onUploadJson = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      if (!isValidPayload(parsed)) {
        throw new Error('JSON 缺少 base/trials 字段');
      }
      setPayload(parsed);
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (errObj) {
      setError(`导入失败: ${errObj?.message || '未知错误'}`);
    }
  };

  const best = payload?.best || {};
  const baseGap = Number(payload?.base?.base_gap_red_minus_green || 0);

  return (
    <div style={{ marginTop: '14px', padding: '16px', borderRadius: '12px', background: 'rgba(2,6,23,0.46)', border: '1px solid rgba(34,197,94,0.26)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
        <div>
          <div style={{ color: '#86efac', fontSize: '14px', fontWeight: 'bold' }}>真实模型知识改写边界看板（通道干预）</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', marginTop: '4px' }}>数据源：{source}</div>
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
              setSource('内置样例');
              setError('');
            }}
            style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '8px', padding: '6px 10px', background: 'transparent', cursor: 'pointer' }}
          >
            重置
          </button>
        </div>
      </div>

      {error && <div style={{ color: '#fca5a5', fontSize: '11px', marginTop: '8px' }}>{error}</div>}

      <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '8px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px', padding: '8px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>基线 gap(red-green)</div>
          <div style={{ color: '#e2e8f0', fontSize: '18px', fontWeight: 'bold' }}>{baseGap.toFixed(4)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px', padding: '8px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>最优 (layer,k,scale)</div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>{`${best.layer ?? '-'}, ${best.k ?? '-'}, ${best.scale ?? '-'}`}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px', padding: '8px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>min_k (soft@0.8)</div>
          <div style={{ color: '#e2e8f0', fontSize: '18px', fontWeight: 'bold' }}>{payload?.min_k_reversal_anchor80_soft ?? 'N/A'}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px', padding: '8px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>min_k (strong@0.8)</div>
          <div style={{ color: '#e2e8f0', fontSize: '18px', fontWeight: 'bold' }}>{payload?.min_k_reversal_anchor80_strong ?? 'N/A'}</div>
        </div>
      </div>

      <div style={{ marginTop: '12px', minHeight: '260px', border: '1px solid rgba(148,163,184,0.24)', borderRadius: '10px', padding: '8px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1) k-规模与翻转/保真权衡</div>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={curveByK} margin={{ top: 8, right: 12, left: 6, bottom: 8 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.2)" strokeDasharray="3 3" />
            <XAxis dataKey="k" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <YAxis yAxisId="left" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
            <YAxis yAxisId="right" orientation="right" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.45)', borderRadius: '8px' }}
              formatter={(value, name) => {
                if (name === 'pair_flip' || name === 'anchor_retention') return [pct(value), name];
                return [Number(value).toFixed(4), name];
              }}
            />
            <Legend wrapperStyle={{ fontSize: '11px', color: '#cbd5e1' }} />
            <Line yAxisId="left" type="monotone" dataKey="best_pair_flip" name="pair_flip" stroke="#22d3ee" strokeWidth={2} dot={{ r: 2 }} />
            <Line yAxisId="left" type="monotone" dataKey="best_anchor_retention" name="anchor_retention" stroke="#34d399" strokeWidth={2} dot={{ r: 2 }} />
            <Line yAxisId="right" type="monotone" dataKey="min_new_gap" name="min_new_gap" stroke="#f59e0b" strokeWidth={2} dot={{ r: 2 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '10px', color: '#94a3b8', fontSize: '11px', lineHeight: '1.6' }}>
        判读：`pair_flip` 越高说明目标关系翻转越充分；`anchor_retention` 越高说明副作用越小；`min_new_gap` 过零代表均值方向已反转。
      </div>
    </div>
  );
}

