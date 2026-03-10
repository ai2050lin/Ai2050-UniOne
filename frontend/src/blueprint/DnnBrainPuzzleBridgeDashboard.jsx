import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import samplePayload from './data/dnn_brain_puzzle_bridge_sample.json';

const COMPONENT_META = {
  shared_basis: {
    label: '共享基底',
    brainMapping: '候选对应脑中可重入细胞群模式与层级概念基底。',
    whyItMatters: '支持“概念不是一词一槽，而是共享骨架复用”的方向。',
  },
  sparse_offset: {
    label: '稀疏偏移',
    brainMapping: '候选对应少量突触增益和少量神经群偏置形成的个体化修正。',
    whyItMatters: '支持新概念通过最小改动接入旧基底的持续学习图景。',
  },
  topology_basis: {
    label: '拓扑基底',
    brainMapping: '候选对应动态路由与脑区间有效连通模式，而不是固定局部词表。',
    whyItMatters: '支持智能核心是结构路由，而不只是静态向量存储。',
  },
  abstraction_operator: {
    label: '抽象提升算子',
    brainMapping: '候选对应跨语义域可复用的抽象化操作，而不是每类概念单独发明规则。',
    whyItMatters: '支持统一微回路在不同域上复用相似算子。',
  },
  protocol_routing: {
    label: '协议场调用',
    brainMapping: '候选对应脑中任务/关系相关子区域的选择性招募。',
    whyItMatters: '支持概念进入关系协议层时存在结构化场调用。',
  },
  multi_timescale_control: {
    label: '多时间尺度门控',
    brainMapping: '候选对应快慢变量、不同时间常数记忆回路及其上下文调制。',
    whyItMatters: '支持长程信用分配需要时间尺度分工，而不是单一记忆池。',
  },
};

const OPEN_PROBLEM_META = {
  symbol_grounding: {
    label: '符号接地',
    description: '概念仍主要来自文本内部结构，还没有从连续世界信号中自发生长出来。',
    nextStep: '把编码分解实验和连续输入基准接通。',
  },
  brain_microcircuit_law: {
    label: '统一微回路定律',
    description: '已经看到若干同构线索，但还没证明存在统一底层结构。',
    nextStep: '继续做 DNN 结构提纯，并用脑约束做最小化建模。',
  },
  long_horizon_credit: {
    label: '长程信用分配',
    description: '长任务上已经有提升，但长度继续增加时仍会退化。',
    nextStep: '继续扫描门控温度、长度依赖门控和更长任务图。',
  },
  continuous_multimodal_closure: {
    label: '连续多模态闭环',
    description: '目前闭环仍偏结构化任务，尚未跨到开放连续状态流。',
    nextStep: '把机制链带到真实多模态代理环境。',
  },
  energy_efficiency_gap: {
    label: '能效差距',
    description: '模型已经出现有效机制，但与脑系统的能效和鲁棒性仍有明显差距。',
    nextStep: '把稀疏激活、局部更新和慢变量控制联动起来评估。',
  },
};

const CONCLUSION_TEXT = {
  statement:
    '第三路线已经从口头主张推进到可量化桥接：DNN 中已提取出多块可复用数学拼图，并且其中若干块已经能和脑机制候选形式做结构同构对照。',
  supported: [
    '共享基底、稀疏偏移、拓扑基底、抽象提升、协议场调用、多时间尺度门控都已经出现可测证据。',
    '深度神经网络与脑机制的同构点在增加，尤其是结构复用、动态路由和多时间尺度控制。',
  ],
  missing: [
    '符号接地仍未闭环。',
    '统一微回路定律还未被证明。',
    '长程门控温度律和连续多模态闭环还缺关键实验。',
  ],
  nextSteps: [
    '先做门控温度 tau_g 扫描，建立长度依赖门控律。',
    '再做概念编码分解，把 B_f / Delta_c / R_tau 在 apple / king / queen 上实测化。',
    '最后把文本内部机制推进到连续输入的接地闭环。',
  ],
};

function isValidPayload(payload) {
  return Boolean(payload && payload.models && payload.component_rows);
}

function fmt(v, digits = 4) {
  return Number(v || 0).toFixed(digits);
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`;
}

function severityColor(level) {
  if (level === 'critical') return '#ef4444';
  if (level === 'high') return '#f59e0b';
  return '#38bdf8';
}

function statusText(status) {
  if (status === 'partial') return '部分推进';
  if (status === 'open') return '未闭环';
  return status || '-';
}

function modelOptions(payload) {
  return Object.keys(payload?.models || {});
}

export default function DnnBrainPuzzleBridgeDashboard() {
  const [payload, setPayload] = useState(samplePayload);
  const [source, setSource] = useState('内置样例');
  const [error, setError] = useState('');
  const models = useMemo(() => modelOptions(payload), [payload]);
  const [selectedModel, setSelectedModel] = useState(models[0] || 'qwen3_4b');

  const modelRow = payload?.models?.[selectedModel] || {};
  const components = Array.isArray(payload?.component_rows) ? payload.component_rows : [];
  const ranking = Array.isArray(payload?.ranking) ? payload.ranking : [];
  const openProblems = Array.isArray(payload?.open_problems) ? payload.open_problems : [];

  const componentRows = useMemo(
    () =>
      components.map((row) => ({
        id: row.id,
        label: COMPONENT_META[row.id]?.label || row.id,
        score: Number(row?.model_scores?.[selectedModel] || 0),
        meanScore: Number(row?.mean_score || 0),
      })),
    [components, selectedModel]
  );

  const rankingRows = useMemo(
    () =>
      ranking.map((row) => ({
        model: row.model_name,
        reverse: Number(row.dnn_reverse_score || 0),
        brain: Number(row.brain_alignment_score || 0),
        bridge: Number(row.overall_bridge_score || 0),
      })),
    [ranking]
  );

  const problemRows = useMemo(
    () =>
      openProblems.map((row) => ({
        id: row.id,
        label: OPEN_PROBLEM_META[row.id]?.label || row.id,
        progress: Number(row.progress || 0),
        severity: row.severity,
        status: row.status,
      })),
    [openProblems]
  );

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      if (!isValidPayload(parsed)) {
        throw new Error('缺少 models 或 component_rows 字段');
      }
      const nextModels = modelOptions(parsed);
      setPayload(parsed);
      setSelectedModel(nextModels[0] || '');
      setSource(`文件导入: ${file.name}`);
      setError('');
    } catch (err) {
      setError(`DNN-脑拼图桥 JSON 导入失败: ${err?.message || '未知错误'}`);
    }
  }

  function resetAll() {
    const nextModels = modelOptions(samplePayload);
    setPayload(samplePayload);
    setSelectedModel(nextModels[0] || '');
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
          'radial-gradient(circle at top left, rgba(8,145,178,0.12), transparent 28%), radial-gradient(circle at top right, rgba(245,158,11,0.10), transparent 24%), rgba(2,6,23,0.64)',
        border: '1px solid rgba(8,145,178,0.24)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <div style={{ color: '#e2e8f0', fontSize: '15px', fontWeight: 'bold' }}>DNN-脑拼图桥</div>
          <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>
            把共享基底、稀疏偏移、拓扑基底、抽象提升、协议场调用和多时间尺度门控压成同一张图，直接看第三路线已经拼出哪些数学部件。
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
          <select
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value)}
            style={{ background: 'rgba(15,23,42,0.9)', color: '#e2e8f0', border: '1px solid rgba(148,163,184,0.35)', borderRadius: '999px', padding: '6px 10px' }}
          >
            {models.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
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

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>DNN 逆向分数</div>
          <div style={{ color: '#22c55e', fontSize: '20px', fontWeight: 'bold' }}>{fmt(modelRow?.dnn_reverse_score)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>脑对齐分数</div>
          <div style={{ color: '#38bdf8', fontSize: '20px', fontWeight: 'bold' }}>{fmt(modelRow?.brain_alignment_score)}</div>
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px' }}>总桥接分数</div>
          <div style={{ color: '#f59e0b', fontSize: '20px', fontWeight: 'bold' }}>{fmt(modelRow?.overall_bridge_score)}</div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(340px, 0.95fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>1. 当前模型的拼图部件分数</div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={componentRows} margin={{ top: 10, right: 12, left: 0, bottom: 40 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-20} textAnchor="end" height={64} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="score" name="当前模型分数" fill="#0891b2" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ display: 'grid', gap: '12px' }}>
          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>2. 总结</div>
            <div style={{ color: '#e2e8f0', fontSize: '12px', lineHeight: '1.8' }}>{CONCLUSION_TEXT.statement}</div>
          </div>

          <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
            <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>3. 已支持的拼图</div>
            <div style={{ display: 'grid', gap: '8px' }}>
              {CONCLUSION_TEXT.supported.map((item) => (
                <div key={item} style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7', border: '1px solid rgba(148,163,184,0.18)', borderRadius: '10px', padding: '8px 10px' }}>
                  {item}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '12px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>4. 模型桥接排序</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={rankingRows} margin={{ top: 10, right: 12, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Legend />
              <Bar dataKey="reverse" name="DNN 逆向" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="brain" name="脑对齐" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="bridge" name="总桥接" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ border: '1px solid rgba(148,163,184,0.22)', borderRadius: '14px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>5. 未闭环硬伤</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={problemRows} margin={{ top: 10, right: 12, left: 0, bottom: 40 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.18)" strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={{ fill: '#cbd5e1', fontSize: 10 }} stroke="rgba(148,163,184,0.4)" interval={0} angle={-20} textAnchor="end" height={64} />
              <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} stroke="rgba(148,163,184,0.4)" domain={[0, 1]} />
              <Tooltip formatter={(value) => pct(value)} contentStyle={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '10px' }} />
              <Bar dataKey="progress" name="推进度" radius={[4, 4, 0, 0]}>
                {problemRows.map((entry) => (
                  <Cell key={entry.id} fill={severityColor(entry.severity)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gap: '10px' }}>
        <div style={{ color: '#dbeafe', fontSize: '12px' }}>6. 脑侧映射与下一步</div>
        {componentRows.map((row) => (
          <div key={row.id} style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
            <div style={{ color: '#f8fafc', fontSize: '12px', fontWeight: 'bold' }}>{`${row.label} | 当前模型 ${fmt(row.score)} | 全局均值 ${fmt(row.meanScore)}`}</div>
            <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>{COMPONENT_META[row.id]?.brainMapping || '-'}</div>
            <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>{COMPONENT_META[row.id]?.whyItMatters || '-'}</div>
          </div>
        ))}
        {problemRows.map((row) => (
          <div key={row.id} style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
            <div style={{ color: severityColor(row.severity), fontSize: '12px', fontWeight: 'bold' }}>{`${row.label} | ${statusText(row.status)} | 进度 ${fmt(row.progress)}`}</div>
            <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>{OPEN_PROBLEM_META[row.id]?.description || '-'}</div>
            <div style={{ color: '#94a3b8', fontSize: '11px', lineHeight: '1.7', marginTop: '4px' }}>{`下一步: ${OPEN_PROBLEM_META[row.id]?.nextStep || '-'}`}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '10px' }}>
        <div style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>7. 当前缺口</div>
          {CONCLUSION_TEXT.missing.map((item) => (
            <div key={item} style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7' }}>
              {item}
            </div>
          ))}
        </div>
        <div style={{ border: '1px solid rgba(148,163,184,0.18)', borderRadius: '12px', padding: '10px' }}>
          <div style={{ color: '#dbeafe', fontSize: '12px', marginBottom: '6px' }}>8. 建议行动</div>
          {CONCLUSION_TEXT.nextSteps.map((item) => (
            <div key={item} style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.7' }}>
              {item}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
