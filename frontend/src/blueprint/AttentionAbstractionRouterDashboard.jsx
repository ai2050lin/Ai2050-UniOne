import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import routerSample from './data/attention_abstraction_router_sample.json';
import stabilitySample from './data/attention_abstraction_router_stability_sample.json';

function isValidRouterPayload(payload) {
  return Boolean(payload && payload.baseline && Array.isArray(payload.all_rows));
}

function isValidStabilityPayload(payload) {
  return Boolean(payload && payload.group_summary && Array.isArray(payload.head_summary));
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`;
}

function roleColor(role) {
  if (role === 'lift1') return '#38bdf8';
  if (role === 'lift2') return '#f59e0b';
  return '#94a3b8';
}

function preferenceColor(v) {
  const value = Math.max(-1, Math.min(1, Number(v || 0)));
  if (value >= 0) {
    const alpha = 0.18 + value * 0.5;
    return `rgba(56, 189, 248, ${alpha.toFixed(3)})`;
  }
  const alpha = 0.18 + Math.abs(value) * 0.5;
  return `rgba(245, 158, 11, ${alpha.toFixed(3)})`;
}

export default function AttentionAbstractionRouterDashboard() {
  const [routerPayload, setRouterPayload] = useState(routerSample);
  const [stabilityPayload, setStabilityPayload] = useState(stabilitySample);
  const [routerSource, setRouterSource] = useState('内置样例');
  const [stabilitySource, setStabilitySource] = useState('内置样例');
  const [error, setError] = useState('');

  const scatterData = useMemo(() => {
    const rows = Array.isArray(routerPayload?.all_rows) ? routerPayload.all_rows : [];
    return {
      lift1: rows
        .filter((row) => Number(row.preference || 0) >= 0.35)
        .map((row) => ({
          x: Number(row.collapse_lift1 || 0),
          y: Number(row.collapse_lift2 || 0),
          layer: Number(row.layer || 0),
          head: Number(row.head || 0),
        })),
      lift2: rows
        .filter((row) => Number(row.preference || 0) <= -0.35)
        .map((row) => ({
          x: Number(row.collapse_lift1 || 0),
          y: Number(row.collapse_lift2 || 0),
          layer: Number(row.layer || 0),
          head: Number(row.head || 0),
        })),
      mixed: rows
        .filter((row) => Math.abs(Number(row.preference || 0)) < 0.35)
        .map((row) => ({
          x: Number(row.collapse_lift1 || 0),
          y: Number(row.collapse_lift2 || 0),
          layer: Number(row.layer || 0),
          head: Number(row.head || 0),
        })),
    };
  }, [routerPayload]);

  const preferenceGrid = useMemo(() => {
    const rows = Array.isArray(routerPayload?.all_rows) ? routerPayload.all_rows : [];
    const nLayers = Number(routerPayload?.meta?.n_layers || 0);
    const nHeads = Number(routerPayload?.meta?.n_heads || 0);
    const rowMap = new Map(rows.map((row) => [`${row.layer}-${row.head}`, row]));
    return Array.from({ length: nLayers }, (_, layer) => ({
      layer,
      heads: Array.from({ length: nHeads }, (_, head) => {
        const row = rowMap.get(`${layer}-${head}`);
        return {
          head,
          preference: Number(row?.preference || 0),
          collapse1: Number(row?.collapse_lift1 || 0),
          collapse2: Number(row?.collapse_lift2 || 0),
        };
      }),
    }));
  }, [routerPayload]);

  const stableBars = useMemo(() => {
    const rows = Array.isArray(stabilityPayload?.head_summary) ? stabilityPayload.head_summary : [];
    return rows.slice(0, 10).map((row) => ({
      name: `L${row.layer}H${row.head}`,
      mean_margin: Number(row.mean_margin || 0),
      role_group: row.role_group,
      fill: roleColor(row.role_group),
    }));
  }, [stabilityPayload]);

  const specializedRatio = useMemo(() => {
    const count = Number(routerPayload?.global_stats?.specialized_head_count_abs_pref_ge_0_35 || 0);
    const total = Number(routerPayload?.global_stats?.scanned_head_count || 0);
    return total > 0 ? count / total : 0;
  }, [routerPayload]);

  async function onUploadRouter(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidRouterPayload(parsed)) {
        throw new Error('缺少 baseline 或 all_rows 字段');
      }
      setRouterPayload(parsed);
      setRouterSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`抽象路由 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  async function onUploadStability(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidStabilityPayload(parsed)) {
        throw new Error('缺少 group_summary 或 head_summary 字段');
      }
      setStabilityPayload(parsed);
      setStabilitySource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`稳定性 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    setRouterPayload(routerSample);
    setStabilityPayload(stabilitySample);
    setRouterSource('内置样例');
    setStabilitySource('内置样例');
    setError('');
  }

  return (
    <div
      style={{
        marginTop: '14px',
        padding: '18px',
        borderRadius: '16px',
        background:
          'radial-gradient(circle at top left, rgba(14,165,233,0.12), transparent 34%), radial-gradient(circle at top right, rgba(245,158,11,0.1), transparent 28%), rgba(2,6,23,0.62)',
        border: '1px solid rgba(125,211,252,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>抽象路由看板</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.6', marginTop: '4px' }}>
            展示 `实例 -&gt; 类别` 与 `类别 -&gt; 抽象系统` 两类提升的头级职责分化，以及跨模板稳定性。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <label style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px', cursor: 'pointer' }}>
            导入路由 JSON
            <input type="file" accept="application/json" onChange={onUploadRouter} style={{ display: 'none' }} />
          </label>
          <label style={{ color: '#e2e8f0', fontSize: '11px', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px', cursor: 'pointer' }}>
            导入稳定性 JSON
            <input type="file" accept="application/json" onChange={onUploadStability} style={{ display: 'none' }} />
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

      <div style={{ marginTop: '10px', display: 'flex', gap: '16px', flexWrap: 'wrap', color: '#94a3b8', fontSize: '11px' }}>
        <div>{`路由数据源: ${routerSource}`}</div>
        <div>{`稳定性数据源: ${stabilitySource}`}</div>
      </div>
      {error && <div style={{ marginTop: '8px', color: '#fca5a5', fontSize: '11px' }}>{error}</div>}

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>实例到类别基线差值</div>
          <div style={{ color: '#e0f2fe', fontSize: '20px', fontWeight: 'bold' }}>{fmt(routerPayload?.baseline?.base_gap_instance_to_category)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>类别到抽象基线差值</div>
          <div style={{ color: '#fde68a', fontSize: '20px', fontWeight: 'bold' }}>{fmt(routerPayload?.baseline?.base_gap_category_to_abstract)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>Top20 重合 Jaccard</div>
          <div style={{ color: '#fca5a5', fontSize: '20px', fontWeight: 'bold' }}>{fmt(routerPayload?.top20_overlap_jaccard)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>专职头占比</div>
          <div style={{ color: '#86efac', fontSize: '20px', fontWeight: 'bold' }}>{pct(specializedRatio)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>实例到类别稳定率</div>
          <div style={{ color: '#38bdf8', fontSize: '20px', fontWeight: 'bold' }}>{pct(stabilityPayload?.group_summary?.lift1?.role_consistency_rate)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>类别到抽象稳定率</div>
          <div style={{ color: '#f59e0b', fontSize: '20px', fontWeight: 'bold' }}>{pct(stabilityPayload?.group_summary?.lift2?.role_consistency_rate)}</div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1.2fr) minmax(320px, 0.9fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 头级职责散点图</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', marginBottom: '8px', lineHeight: '1.6' }}>
            横轴是 `collapse_lift1`，纵轴是 `collapse_lift2`。蓝色头更偏向实例到类别，橙色头更偏向类别到抽象系统。
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 10, right: 16, bottom: 10, left: 6 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="collapse_lift1" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis type="number" dataKey="y" name="collapse_lift2" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.45)', borderRadius: '10px' }}
                formatter={(value, name) => [fmt(value), name]}
                labelFormatter={(_, payload) => {
                  const row = payload?.[0]?.payload;
                  return row ? `L${row.layer} H${row.head}` : '';
                }}
              />
              <Scatter name="实例到类别头" data={scatterData.lift1} fill="#38bdf8" />
              <Scatter name="类别到抽象头" data={scatterData.lift2} fill="#f59e0b" />
              <Scatter name="混合头" data={scatterData.mixed} fill="#94a3b8" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 最稳定的专职头</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', marginBottom: '8px', lineHeight: '1.6' }}>
            条形长度表示跨模板的平均职责 margin，颜色区分两类抽象路由。
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={stableBars} layout="vertical" margin={{ top: 8, right: 12, bottom: 8, left: 6 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.16)" strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis type="category" dataKey="name" width={62} tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.45)', borderRadius: '10px' }}
                formatter={(value, name) => [fmt(value), name]}
              />
              <Bar dataKey="mean_margin" radius={[0, 6, 6, 0]}>
                {stableBars.map((entry) => (
                  <Cell key={`${entry.name}-${entry.role_group}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '12px', border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. Layer × Head 偏好热图</div>
        <div style={{ color: '#94a3b8', fontSize: '11px', marginBottom: '10px', lineHeight: '1.6' }}>
          蓝色越深表示越偏向实例到类别，橙色越深表示越偏向类别到抽象系统。
        </div>
        <div style={{ overflowX: 'auto', paddingBottom: '8px' }}>
          <div style={{ minWidth: '720px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: `64px repeat(${preferenceGrid[0]?.heads?.length || 0}, 1fr)`, gap: '4px', marginBottom: '4px' }}>
              <div />
              {(preferenceGrid[0]?.heads || []).map((cell) => (
                <div key={`head-label-${cell.head}`} style={{ color: '#94a3b8', fontSize: '10px', textAlign: 'center' }}>{`H${cell.head}`}</div>
              ))}
            </div>
            {preferenceGrid.map((row) => (
              <div key={`row-${row.layer}`} style={{ display: 'grid', gridTemplateColumns: `64px repeat(${row.heads.length}, 1fr)`, gap: '4px', marginBottom: '4px' }}>
                <div style={{ color: '#94a3b8', fontSize: '10px', display: 'flex', alignItems: 'center' }}>{`Layer ${row.layer}`}</div>
                {row.heads.map((cell) => (
                  <div
                    key={`cell-${row.layer}-${cell.head}`}
                    title={`L${row.layer} H${cell.head} | pref=${fmt(cell.preference, 3)} | c1=${fmt(cell.collapse1, 3)} | c2=${fmt(cell.collapse2, 3)}`}
                    style={{
                      height: '18px',
                      borderRadius: '5px',
                      background: preferenceColor(cell.preference),
                      border: '1px solid rgba(255,255,255,0.05)',
                    }}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>4. 代表性头</div>
          <div style={{ display: 'grid', gap: '8px' }}>
            {(routerPayload?.top_heads_instance_to_category || []).slice(0, 4).map((row) => (
              <div key={`top-l1-${row.layer}-${row.head}`} style={{ borderRadius: '10px', padding: '8px', background: 'rgba(56,189,248,0.08)', border: '1px solid rgba(56,189,248,0.22)' }}>
                <div style={{ color: '#e0f2fe', fontSize: '12px', fontWeight: 'bold' }}>{`实例 -> 类别 | L${row.layer} H${row.head}`}</div>
                <div style={{ color: '#bae6fd', fontSize: '11px', marginTop: '4px' }}>{`collapse_lift1 = ${fmt(row.collapse_lift1)}`}</div>
              </div>
            ))}
            {(routerPayload?.top_heads_category_to_abstract || []).slice(0, 4).map((row) => (
              <div key={`top-l2-${row.layer}-${row.head}`} style={{ borderRadius: '10px', padding: '8px', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.22)' }}>
                <div style={{ color: '#fef3c7', fontSize: '12px', fontWeight: 'bold' }}>{`类别 -> 抽象 | L${row.layer} H${row.head}`}</div>
                <div style={{ color: '#fde68a', fontSize: '11px', marginTop: '4px' }}>{`collapse_lift2 = ${fmt(row.collapse_lift2)}`}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '12px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '8px' }}>5. 稳定性结论</div>
          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.8' }}>
            <div>{`实例 -> 类别：平均职责 margin ${fmt(stabilityPayload?.group_summary?.lift1?.mean_margin)}，稳定率 ${pct(stabilityPayload?.group_summary?.lift1?.role_consistency_rate)}`}</div>
            <div>{`类别 -> 抽象系统：平均职责 margin ${fmt(stabilityPayload?.group_summary?.lift2?.mean_margin)}，稳定率 ${pct(stabilityPayload?.group_summary?.lift2?.role_consistency_rate)}`}</div>
          </div>
          <div style={{ marginTop: '10px', padding: '10px', borderRadius: '10px', background: 'rgba(15,23,42,0.5)', border: '1px solid rgba(148,163,184,0.16)' }}>
            <div style={{ color: '#e2e8f0', fontSize: '12px', fontWeight: 'bold' }}>当前解释</div>
            <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '6px' }}>
              类别到抽象系统头的职责更稳定，说明高层抽象路由更像通用模板。
              <br />
              实例到类别头更依赖具体提示语境，说明前一级抽象仍保留较强的输入相位敏感性。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
