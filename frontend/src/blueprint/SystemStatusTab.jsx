import { Activity, Brain, CheckCircle, Search, X } from 'lucide-react';
import { useState } from 'react';

const BrainModel = () => (
  <div
    style={{
      width: '320px',
      height: '320px',
      margin: '0 auto',
      position: 'relative',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'radial-gradient(circle, rgba(0, 210, 255, 0.15) 0%, transparent 70%)',
      borderRadius: '50%',
      animation: 'brainPulse 4s infinite alternate',
    }}
  >
    <div
      style={{
        position: 'absolute',
        width: '100%',
        height: '100%',
        border: '1px solid rgba(0, 210, 255, 0.2)',
        borderRadius: '50%',
        animation: 'brainRotate 20s linear infinite',
      }}
    />
    <div
      style={{
        position: 'absolute',
        width: '85%',
        height: '85%',
        border: '1px dashed rgba(168, 85, 247, 0.3)',
        borderRadius: '50%',
        animation: 'brainRotateReverse 15s linear infinite',
      }}
    />
    <Brain
      size={180}
      color="#00d2ff"
      style={{
        filter: 'drop-shadow(0 0 30px rgba(0, 210, 255, 0.4))',
        zIndex: 2,
      }}
    />
    {Array.from({ length: 8 }).map((_, i) => (
      <div
        key={i}
        style={{
          position: 'absolute',
          width: '6px',
          height: '6px',
          background: '#00ff88',
          borderRadius: '50%',
          boxShadow: '0 0 10px #00ff88',
          transform: `rotate(${i * 45}deg) translateY(-130px)`,
          animation: 'synapsePulse 2s infinite',
          animationDelay: `${i * 0.2}s`,
        }}
      />
    ))}
  </div>
);

const badgeStyle = (bg, border, color) => ({
  padding: '4px 8px',
  borderRadius: '999px',
  background: bg,
  border,
  fontSize: '10px',
  color,
});

const runtimePill = (label, value, color = '#7dd3fc') => (
  <div
    key={label}
    style={{
      padding: '10px 12px',
      borderRadius: '14px',
      background: 'rgba(0,0,0,0.22)',
      border: '1px solid rgba(255,255,255,0.06)',
    }}
  >
    <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '4px' }}>{label}</div>
    <div style={{ fontSize: '18px', fontWeight: 900, color, fontFamily: 'monospace' }}>{value}</div>
  </div>
);

const summaryBlock = (title, color, items) => (
  <div
    style={{
      padding: '20px',
      background: 'rgba(255,255,255,0.025)',
      borderRadius: '22px',
      border: `1px solid ${color}26`,
    }}
  >
    <div
      style={{
        fontSize: '12px',
        color,
        textTransform: 'uppercase',
        letterSpacing: '1.5px',
        marginBottom: '14px',
        fontWeight: 'bold',
      }}
    >
      {title}
    </div>
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      {items.map((item) => (
        <div
          key={item}
          style={{
            fontSize: '13px',
            lineHeight: '1.7',
            color: '#cbd5e1',
            padding: '10px 12px',
            borderRadius: '12px',
            background: 'rgba(0,0,0,0.18)',
            border: '1px solid rgba(255,255,255,0.04)',
          }}
        >
          {item}
        </div>
      ))}
    </div>
  </div>
);

const progressCard = (label, value, color, hint) => (
  <div
    key={label}
    style={{
      padding: '14px 16px',
      background: 'rgba(0,0,0,0.24)',
      borderRadius: '16px',
      border: '1px solid rgba(255,255,255,0.05)',
    }}
  >
    <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '4px' }}>{label}</div>
    <div style={{ fontSize: '20px', color, fontWeight: 900, fontFamily: 'monospace' }}>{value}</div>
    <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '4px' }}>{hint}</div>
  </div>
);

export const SystemStatusTab = ({
  consciousField,
  systemRouteOptions,
  routeList,
  setSelectedRouteId,
  selectedRouteId,
  activeSystemProfile,
  statusData,
  selectedRoute,
  getRouteImpl,
}) => {
  const modelSummary = statusData?.model_summary || {};
  const runtimeLanguage = statusData?.runtime_language || {};
  const phaseaRuntime = statusData?.phasea_runtime || {};
  const researchOverview = statusData?.research_overview || {};
  const [expandedParam, setExpandedParam] = useState(null);

  return (
    <div style={{ animation: 'roadmapFade 0.5s ease-out' }}>
      <BrainModel />

      <div style={{ textAlign: 'center', marginBottom: '60px' }}>
        <h2
          style={{
            fontSize: '32px',
            fontWeight: '900',
            color: consciousField?.glow_color === 'amber' ? '#ffaa00' : '#10b981',
            margin: '20px 0 8px 0',
            transition: 'color 1s',
          }}
        >
          {consciousField ? '实时意识场与系统状态' : '系统状态总览'}
        </h2>
        <p style={{ color: '#94a3b8', fontSize: '14px' }}>
          {consciousField
            ? `当前稳定度 ${(Number(consciousField.stability || 0) * 100).toFixed(1)}% | 全局工作空间强度 ${Number(consciousField.gws_intensity || 0).toFixed(2)}`
            : '按 DNN 分析结果、脑编码特性、理论距离与新模型测试效果重组项目视图。'}
        </p>
        <div style={{ marginTop: '14px', display: 'flex', justifyContent: 'center', gap: '8px', flexWrap: 'wrap' }}>
          {(systemRouteOptions.length > 0 ? systemRouteOptions : routeList).map((route) => (
            <button
              key={route.id}
              onClick={() => setSelectedRouteId(route.id)}
              style={{
                border: '1px solid rgba(255,255,255,0.12)',
                background: selectedRouteId === route.id ? 'rgba(0, 210, 255, 0.2)' : 'rgba(255,255,255,0.03)',
                color: selectedRouteId === route.id ? '#67e8f9' : '#94a3b8',
                borderRadius: '999px',
                fontSize: '11px',
                padding: '6px 10px',
                cursor: 'pointer',
              }}
            >
              {route.title}
            </button>
          ))}
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '20px',
          marginBottom: '40px',
          animation: 'fadeIn 1s',
        }}
      >
        {(activeSystemProfile?.metricCards || []).map((m, i) => (
          <div
            key={i}
            style={{
              padding: '18px',
              background: 'rgba(255,255,255,0.02)',
              borderRadius: '20px',
              border: `1px solid ${m.color}30`,
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '6px', fontWeight: 'bold' }}>{m.label}</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: m.color, marginBottom: '6px' }}>{m.value}</div>
            <div style={{ fontSize: '10px', color: '#9ca3af' }}>{m.brain_ability}</div>
          </div>
        ))}
      </div>

      <div
        style={{
          marginBottom: '40px',
          padding: '24px',
          background: 'rgba(255,255,255,0.025)',
          borderRadius: '28px',
          border: '1px solid rgba(0, 210, 255, 0.16)',
          boxShadow: '0 0 0 1px rgba(255,255,255,0.02) inset',
        }}
      >
        <div
          style={{
            fontSize: '12px',
            color: '#00d2ff',
            textTransform: 'uppercase',
            letterSpacing: '2px',
            marginBottom: '16px',
            fontWeight: 'bold',
          }}
        >
          当前训练与理论状态
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr 1fr 1fr', gap: '14px', marginBottom: '14px' }}>
          <div
            style={{
              padding: '14px 16px',
              background: 'rgba(0,0,0,0.24)',
              borderRadius: '16px',
              border: '1px solid rgba(255,255,255,0.05)',
            }}
          >
            <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '4px' }}>当前在线模型 / 路径</div>
            <div style={{ fontSize: '13px', color: '#e2e8f0', fontWeight: 700, marginBottom: '4px', wordBreak: 'break-all' }}>
              {modelSummary.current_model_file}
            </div>
            <div style={{ fontSize: '11px', color: '#7dd3fc' }}>{modelSummary.current_model_name}</div>
          </div>
          {progressCard('理论骨架完成度', modelSummary.theory_skeleton_progress || '96% - 98%', '#00d2ff', '统一候选理论骨架')}
          {progressCard('工程闭合度', modelSummary.engineering_closure_progress || '95% - 97%', '#10b981', '三闭环工程口径')}
          {progressCard('严格本体破解度', modelSummary.strict_brain_encoding_progress || '45% - 53%', '#f59e0b', '更严格的脑编码本体口径')}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '14px' }}>
          {progressCard('总参数量', modelSummary.total_parameters || '-', '#ffffff', `可训练参数：${modelSummary.trainable_parameters || '-'}`)}
          {progressCard('原型训练闭合度', modelSummary.prototype_training_progress || '-', '#22c55e', '当前工程原型口径')}
          {progressCard('人类智能训练进度', modelSummary.human_level_training_progress || '-', '#f59e0b', '严格目标口径')}
          {progressCard('语言训练进度', modelSummary.language_training_progress || '-', '#38bdf8', '当前语言主链口径')}
        </div>
      </div>

      <div
        style={{
          marginBottom: '40px',
          padding: '24px',
          background: 'rgba(255,255,255,0.025)',
          borderRadius: '28px',
          border: '1px solid rgba(34, 197, 94, 0.14)',
        }}
      >
        <div
          style={{
            fontSize: '12px',
            color: '#22c55e',
            textTransform: 'uppercase',
            letterSpacing: '2px',
            marginBottom: '16px',
            fontWeight: 'bold',
          }}
        >
          新语言主干测试
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr 1fr 1fr', gap: '14px', marginBottom: '14px' }}>
          <div
            style={{
              padding: '14px 16px',
              background: 'rgba(0,0,0,0.24)',
              borderRadius: '16px',
              border: '1px solid rgba(255,255,255,0.05)',
            }}
          >
            <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '4px' }}>PhaseA 模型文件 / 类名</div>
            <div style={{ fontSize: '13px', color: '#e2e8f0', fontWeight: 700, marginBottom: '4px', wordBreak: 'break-all' }}>
              {modelSummary.phasea_model_file || '-'}
            </div>
            <div style={{ fontSize: '11px', color: '#86efac' }}>{modelSummary.phasea_model_name || '-'}</div>
          </div>
          {progressCard('PhaseA 参数量', modelSummary.phasea_total_parameters || '-', '#ffffff', '当前扩容语言主干')}
          {progressCard('PhaseA 准备度', modelSummary.phasea_readiness_progress || '-', '#22c55e', '架构与训练就绪度')}
          {progressCard('PhaseA 生成可用度', modelSummary.phasea_generation_progress || '-', '#f97316', '真实语言生成口径')}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
          {runtimePill('语言等级', phaseaRuntime.language_level || '-', '#86efac')}
          {runtimePill('语言等级总评', Number(phaseaRuntime.language_level_score || 0).toFixed(3), '#7dd3fc')}
          {runtimePill('长程预训练总评', Number(phaseaRuntime.long_pretraining_score || 0).toFixed(3), '#c4b5fd')}
          {runtimePill('生成总评', Number(phaseaRuntime.generation_score || 0).toFixed(3), '#fca5a5')}
        </div>
      </div>

      <div
        style={{
          marginBottom: '40px',
          padding: '24px',
          background: 'rgba(255,255,255,0.025)',
          borderRadius: '28px',
          border: '1px solid rgba(255,255,255,0.08)',
        }}
      >
        <div
          style={{
            fontSize: '12px',
            color: '#a5f3fc',
            textTransform: 'uppercase',
            letterSpacing: '2px',
            marginBottom: '16px',
            fontWeight: 'bold',
          }}
        >
          运行时语言摘要
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
          {runtimePill('语义基准分数', Number(runtimeLanguage.semantic_benchmark_score || 0).toFixed(3))}
          {runtimePill('训练轮数', runtimeLanguage.semantic_training_rounds ?? 0, '#c4b5fd')}
          {runtimePill('语言闭合总评', Number(runtimeLanguage.language_training_closure_score || 0).toFixed(3), '#86efac')}
          {runtimePill('开放域总评', Number(runtimeLanguage.open_domain_assessment_score || 0).toFixed(3), '#fcd34d')}
          {runtimePill('Scaleup 总评', Number(runtimeLanguage.scaleup_training_score || 0).toFixed(3), '#fca5a5')}
          {runtimePill('记忆轨迹深度', runtimeLanguage.memory_trace_depth ?? 0, '#7dd3fc')}
          {runtimePill('本轮训练轮数', runtimeLanguage.latest_scaleup_rounds ?? 0, '#d8b4fe')}
          {runtimePill('语言训练进度', modelSummary.language_training_progress || '-', '#22c55e')}
        </div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginTop: '12px' }}>
          <div style={badgeStyle('rgba(34,197,94,0.12)', '1px solid rgba(34,197,94,0.28)', '#86efac')}>
            语义链路：{runtimeLanguage.semantic_pipeline_ready ? '已接通' : '未接通'}
          </div>
          <div style={badgeStyle('rgba(59,130,246,0.12)', '1px solid rgba(59,130,246,0.28)', '#93c5fd')}>
            对话就绪：{runtimeLanguage.dialog_ready ? '是' : '否'}
          </div>
          <div style={badgeStyle('rgba(245,158,11,0.12)', '1px solid rgba(245,158,11,0.28)', '#fcd34d')}>
            开放域就绪：{runtimeLanguage.open_domain_ready ? '是' : '否'}
          </div>
          <div style={badgeStyle('rgba(168,85,247,0.12)', '1px solid rgba(168,85,247,0.28)', '#d8b4fe')}>
            训练冲刺：{runtimeLanguage.scaleup_ready ? '已通过' : '继续训练中'}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '40px' }}>
        {summaryBlock('深度神经网络分析结果', '#a855f7', researchOverview.dnn_analysis_results || [])}
        {summaryBlock('大脑编码机制特性', '#00d2ff', researchOverview.brain_encoding_traits || [])}
        {summaryBlock('离真正破解还有多远', '#f59e0b', researchOverview.theory_gap || [])}
        {summaryBlock('新网络模型测试效果', '#22c55e', researchOverview.new_model_tests || [])}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px' }}>
        <div
          style={{
            background: 'rgba(0, 255, 136, 0.03)',
            border: '1px solid rgba(0, 255, 136, 0.15)',
            borderRadius: '32px',
            padding: '32px',
          }}
        >
          <div
            style={{
              fontSize: '12px',
              color: '#00ff88',
              textTransform: 'uppercase',
              letterSpacing: '2px',
              marginBottom: '24px',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <CheckCircle size={16} /> 已具备模块
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {(statusData?.capabilities || []).map((c, i) => (
              <div
                key={i}
                style={{
                  padding: '16px',
                  background: 'rgba(255,255,255,0.02)',
                  borderRadius: '16px',
                  border: '1px solid rgba(255,255,255,0.05)',
                }}
              >
                <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '8px' }}>{c.name}</div>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '10px' }}>
                  <div style={badgeStyle('rgba(16,185,129,0.12)', '1px solid rgba(16,185,129,0.28)', '#86efac')}>
                    模块名词：{c.module_term || '-'}
                  </div>
                  <div style={badgeStyle('rgba(59,130,246,0.12)', '1px solid rgba(59,130,246,0.28)', '#93c5fd')}>
                    参数量：{c.parameter_footprint || '-'}
                  </div>
                  <div style={badgeStyle('rgba(245,158,11,0.12)', '1px solid rgba(245,158,11,0.28)', '#fcd34d')}>
                    进度：{c.progress_pct || '-'}
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '0.9fr 1.1fr', gap: '10px' }}>
                  <div
                    style={{
                      padding: '8px 10px',
                      borderRadius: '10px',
                      background: 'rgba(34,197,94,0.08)',
                      border: '1px solid rgba(34,197,94,0.2)',
                    }}
                  >
                    <div style={{ fontSize: '10px', color: '#86efac', marginBottom: '4px' }}>对应脑能力</div>
                    <div style={{ fontSize: '11px', color: '#dcfce7', lineHeight: '1.55' }}>{c.brain_ability || '-'}</div>
                  </div>
                  <div
                    style={{
                      padding: '8px 10px',
                      borderRadius: '10px',
                      background: 'rgba(56,189,248,0.08)',
                      border: '1px solid rgba(56,189,248,0.2)',
                    }}
                  >
                    <div style={{ fontSize: '10px', color: '#7dd3fc', marginBottom: '4px' }}>
                      当前实现（{selectedRoute?.title || selectedRouteId}）
                    </div>
                    <div style={{ fontSize: '11px', color: '#e0f2fe', lineHeight: '1.55' }}>{getRouteImpl(c)}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div
          style={{
            background: 'rgba(255, 68, 68, 0.03)',
            border: '1px solid rgba(255, 68, 68, 0.15)',
            borderRadius: '32px',
            padding: '32px',
          }}
        >
          <div
            style={{
              fontSize: '12px',
              color: '#ff4444',
              textTransform: 'uppercase',
              letterSpacing: '2px',
              marginBottom: '24px',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <X size={16} /> 未闭合模块
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {(statusData?.missing_capabilities || []).map((c, i) => (
              <div
                key={i}
                style={{
                  padding: '16px',
                  background: 'rgba(255,255,255,0.02)',
                  borderRadius: '16px',
                  border: '1px solid rgba(255,255,255,0.05)',
                }}
              >
                <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '8px', color: '#ff8888' }}>{c.name}</div>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '10px' }}>
                  <div style={badgeStyle('rgba(239,68,68,0.12)', '1px solid rgba(239,68,68,0.28)', '#fca5a5')}>
                    模块名词：{c.module_term || '-'}
                  </div>
                  <div style={badgeStyle('rgba(59,130,246,0.12)', '1px solid rgba(59,130,246,0.28)', '#93c5fd')}>
                    参数量：{c.parameter_footprint || '-'}
                  </div>
                  <div style={badgeStyle('rgba(245,158,11,0.12)', '1px solid rgba(245,158,11,0.28)', '#fcd34d')}>
                    进度：{c.progress_pct || '-'}
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '0.9fr 1.1fr', gap: '10px' }}>
                  <div
                    style={{
                      padding: '8px 10px',
                      borderRadius: '10px',
                      background: 'rgba(248,113,113,0.08)',
                      border: '1px solid rgba(248,113,113,0.22)',
                    }}
                  >
                    <div style={{ fontSize: '10px', color: '#fca5a5', marginBottom: '4px' }}>目标脑能力</div>
                    <div style={{ fontSize: '11px', color: '#fee2e2', lineHeight: '1.55' }}>{c.brain_ability || '-'}</div>
                  </div>
                  <div
                    style={{
                      padding: '8px 10px',
                      borderRadius: '10px',
                      background: 'rgba(251,191,36,0.08)',
                      border: '1px solid rgba(251,191,36,0.22)',
                    }}
                  >
                    <div style={{ fontSize: '10px', color: '#fcd34d', marginBottom: '4px' }}>
                      当前实现（{selectedRoute?.title || selectedRouteId}）
                    </div>
                    <div style={{ fontSize: '11px', color: '#fef3c7', lineHeight: '1.55' }}>{getRouteImpl(c)}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px', marginTop: '40px' }}>
        <div
          style={{
            background: 'rgba(0, 210, 255, 0.03)',
            border: '1px solid rgba(0, 210, 255, 0.15)',
            borderRadius: '32px',
            padding: '32px',
          }}
        >
          <div
            style={{
              fontSize: '12px',
              color: '#00d2ff',
              textTransform: 'uppercase',
              letterSpacing: '2px',
              marginBottom: '24px',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <Activity size={16} /> 核心参数
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
            {(statusData?.parameters || []).map((p, i) => (
              <div
                key={i}
                style={{
                  padding: '18px',
                  background: expandedParam === i ? 'rgba(0, 210, 255, 0.1)' : 'rgba(255,255,255,0.02)',
                  borderRadius: '20px',
                  border: `1px solid ${expandedParam === i ? '#00d2ff40' : 'rgba(255,255,255,0.05)'}`,
                  cursor: 'pointer',
                  transition: 'all 0.3s',
                }}
                onClick={() => setExpandedParam(expandedParam === i ? null : i)}
              >
                <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '6px', fontWeight: 'bold' }}>{p.name}</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#00d2ff', marginBottom: '6px' }}>{p.value}</div>
                <div style={{ fontSize: '11px', color: '#64748b', lineHeight: '1.5' }}>{p.desc}</div>
              </div>
            ))}
          </div>
        </div>

        <div
          style={{
            background: 'rgba(168, 85, 247, 0.03)',
            border: '1px solid rgba(168, 85, 247, 0.15)',
            borderRadius: '32px',
            padding: '32px',
          }}
        >
          <div
            style={{
              fontSize: '12px',
              color: '#a855f7',
              textTransform: 'uppercase',
              letterSpacing: '2px',
              marginBottom: '24px',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <Search size={16} /> 已通过验证
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {(statusData?.passed_tests || []).map((test, i) => (
              <div
                key={i}
                style={{
                  padding: '16px',
                  background: 'rgba(255,255,255,0.02)',
                  borderRadius: '16px',
                  border: '1px solid rgba(255,255,255,0.05)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#e9d5ff' }}>{test.name}</div>
                  <div style={badgeStyle('rgba(34,197,94,0.12)', '1px solid rgba(34,197,94,0.28)', '#86efac')}>
                    {test.result}
                  </div>
                </div>
                <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '8px' }}>{test.date}</div>
                <div style={{ fontSize: '11px', color: '#cbd5e1', lineHeight: '1.7' }}>{test.analysis || test.target}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
