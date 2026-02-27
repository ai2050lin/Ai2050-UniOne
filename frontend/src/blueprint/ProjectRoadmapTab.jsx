import { useState } from 'react';
import { ClaudeTab } from './ClaudeTab';
import { GPT5Tab } from './GPT5Tab';
import { GLM5Tab } from './GLM5Tab';

export const ProjectRoadmapTab = ({
  roadmapData,
  analysisPhase,
  evidenceDrivenPlan,
  mathRouteSystemPlan,
  improvements,
  expandedImprovementPhase,
  setExpandedImprovementPhase,
  expandedImprovementTest,
  setExpandedImprovementTest,
}) => {
  const [activeModelTab, setActiveModelTab] = useState('Claude');
  const modelTabs = ['Claude', 'GPT5', 'GLM5'];

  return (
    <div style={{ animation: 'roadmapFade 0.6s ease-out', maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ marginBottom: '34px' }}>
        <h2 style={{ fontSize: '30px', fontWeight: '900', color: '#ffaa00', marginBottom: '10px' }}>项目大纲</h2>
        <div style={{ color: '#777', fontSize: '14px' }}>{roadmapData?.definition?.summary}</div>
      </div>

      <div
        style={{
          padding: '30px',
          background: 'linear-gradient(135deg, rgba(255,170,0,0.12) 0%, rgba(255,170,0,0.03) 100%)',
          border: '1px solid rgba(255,170,0,0.24)',
          borderRadius: '24px',
          marginBottom: '28px',
        }}
      >
        <div style={{ color: '#ffaa00', fontWeight: 'bold', fontSize: '18px', marginBottom: '16px' }}>核心思路</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
          {[
            '1，大脑有非常特殊的数学结构，产生了智能。',
            '2，深度神经网络部分还原了这个结构，产生了语言能力。',
            '3，通过分析深度神经网络，研究这个数学结构，完成智能理论。',
          ].map((line, idx) => (
            <div
              key={idx}
              style={{
                padding: '14px 16px',
                borderRadius: '12px',
                background: 'rgba(255,255,255,0.05)',
                color: '#f4e4c1',
                fontSize: '14px',
                lineHeight: '1.6',
              }}
            >
              {line}
            </div>
          ))}
        </div>
      </div>

      <div
        style={{
          padding: '30px',
          borderRadius: '24px',
          border: '1px solid rgba(99,102,241,0.28)',
          background: 'linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(99,102,241,0.03) 100%)',
          marginBottom: '28px',
        }}
      >
        <div style={{ color: '#818cf8', fontWeight: 'bold', fontSize: '18px', marginBottom: '8px' }}>
          {mathRouteSystemPlan.title}
        </div>
        <div style={{ color: '#c7d2fe', fontSize: '13px', lineHeight: '1.7', marginBottom: '14px' }}>
          {mathRouteSystemPlan.subtitle}
        </div>

        <div
          style={{
            marginTop: '12px',
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.08)',
            background: 'rgba(0,0,0,0.22)',
            padding: '12px',
          }}
        >
          <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '8px' }}>数学路线</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', minWidth: '1440px', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: 'rgba(255,255,255,0.05)' }}>
                  <th
                    style={{
                      textAlign: 'left',
                      padding: '8px 10px',
                      color: '#c7d2fe',
                      fontSize: '11px',
                      borderBottom: '1px solid rgba(255,255,255,0.08)',
                    }}
                  >
                    路线
                  </th>
                  <th
                    style={{
                      textAlign: 'left',
                      padding: '8px 10px',
                      color: '#93c5fd',
                      fontSize: '11px',
                      borderBottom: '1px solid rgba(255,255,255,0.08)',
                    }}
                  >
                    路线说明
                  </th>
                  <th
                    style={{
                      textAlign: 'left',
                      padding: '8px 10px',
                      color: '#86efac',
                      fontSize: '11px',
                      borderBottom: '1px solid rgba(255,255,255,0.08)',
                    }}
                  >
                    优点
                  </th>
                  <th
                    style={{
                      textAlign: 'left',
                      padding: '8px 10px',
                      color: '#fca5a5',
                      fontSize: '11px',
                      borderBottom: '1px solid rgba(255,255,255,0.08)',
                    }}
                  >
                    缺点
                  </th>
                  <th
                    style={{
                      textAlign: 'left',
                      padding: '8px 10px',
                      color: '#93c5fd',
                      fontSize: '11px',
                      borderBottom: '1px solid rgba(255,255,255,0.08)',
                    }}
                  >
                    可行性结论
                  </th>
                  <th
                    style={{
                      textAlign: 'left',
                      padding: '8px 10px',
                      color: '#c7d2fe',
                      fontSize: '11px',
                      borderBottom: '1px solid rgba(255,255,255,0.08)',
                    }}
                  >
                    理论深度
                  </th>
                  <th
                    style={{
                      textAlign: 'left',
                      padding: '8px 10px',
                      color: '#c7d2fe',
                      fontSize: '11px',
                      borderBottom: '1px solid rgba(255,255,255,0.08)',
                    }}
                  >
                    计算可行性
                  </th>
                  <th
                    style={{
                      textAlign: 'left',
                      padding: '8px 10px',
                      color: '#c7d2fe',
                      fontSize: '11px',
                      borderBottom: '1px solid rgba(255,255,255,0.08)',
                    }}
                  >
                    可解释性
                  </th>
                  <th
                    style={{
                      textAlign: 'left',
                      padding: '8px 10px',
                      color: '#c7d2fe',
                      fontSize: '11px',
                      borderBottom: '1px solid rgba(255,255,255,0.08)',
                    }}
                  >
                    与 SHMC/NFBT 兼容
                  </th>
                </tr>
              </thead>
              <tbody>
                {(mathRouteSystemPlan.routeAnalysis || []).map((item, idx) => (
                  <tr key={idx} style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                    <td style={{ padding: '9px 10px', color: '#e0e7ff', fontSize: '12px', fontWeight: 'bold', verticalAlign: 'top' }}>
                      {item.route}
                    </td>
                    <td style={{ padding: '9px 10px', color: '#bfdbfe', fontSize: '11px', lineHeight: '1.55', verticalAlign: 'top' }}>
                      {item.routeSummary || item.description || item.routeDesc || ((item.pros || [])[0] || '—')}
                    </td>
                    <td style={{ padding: '9px 10px', color: '#dcfce7', fontSize: '11px', lineHeight: '1.55', verticalAlign: 'top' }}>
                      {(item.pros || []).map((line, pIdx) => (
                        <div key={pIdx}>{pIdx + 1}. {line}</div>
                      ))}
                    </td>
                    <td style={{ padding: '9px 10px', color: '#fee2e2', fontSize: '11px', lineHeight: '1.55', verticalAlign: 'top' }}>
                      {(item.cons || []).map((line, cIdx) => (
                        <div key={cIdx}>{cIdx + 1}. {line}</div>
                      ))}
                    </td>
                    <td style={{ padding: '9px 10px', color: '#bae6fd', fontSize: '11px', lineHeight: '1.55', verticalAlign: 'top' }}>
                      {item.feasibility}
                    </td>
                    <td style={{ padding: '9px 10px', color: '#dbeafe', fontSize: '12px', verticalAlign: 'top' }}>{item.depth}</td>
                    <td style={{ padding: '9px 10px', color: '#dbeafe', fontSize: '12px', verticalAlign: 'top' }}>{item.compute}</td>
                    <td style={{ padding: '9px 10px', color: '#dbeafe', fontSize: '12px', verticalAlign: 'top' }}>{item.interpret}</td>
                    <td style={{ padding: '9px 10px', color: '#dbeafe', fontSize: '12px', verticalAlign: 'top' }}>{item.compatibility}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr', gap: '12px' }}>
          <div style={{ padding: '14px', borderRadius: '12px', background: 'rgba(0,0,0,0.22)', border: '1px solid rgba(255,255,255,0.08)' }}>
            <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>分层架构</div>
            {(mathRouteSystemPlan.architecture || []).map((line, idx) => (
              <div key={idx} style={{ color: '#e0e7ff', fontSize: '12px', lineHeight: '1.6', marginBottom: '4px' }}>
                {idx + 1}. {line}
              </div>
            ))}
          </div>

          <div style={{ padding: '14px', borderRadius: '12px', background: 'rgba(0,0,0,0.22)', border: '1px solid rgba(255,255,255,0.08)' }}>
            <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>资源配比</div>
            {(mathRouteSystemPlan.allocation || []).map((line, idx) => (
              <div key={idx} style={{ color: '#dbeafe', fontSize: '12px', lineHeight: '1.6', marginBottom: '4px' }}>
                {line}
              </div>
            ))}
          </div>

          <div style={{ padding: '14px', borderRadius: '12px', background: 'rgba(0,0,0,0.22)', border: '1px solid rgba(255,255,255,0.08)' }}>
            <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>阶段里程碑</div>
            {(mathRouteSystemPlan.milestones || []).map((line, idx) => (
              <div key={idx} style={{ color: '#dbeafe', fontSize: '12px', lineHeight: '1.6', marginBottom: '4px' }}>
                {idx + 1}. {line}
              </div>
            ))}
          </div>
        </div>
      </div>


      {/* ====== 模型结构对比模块 ====== */}
      <div
        style={{
          padding: '30px',
          borderRadius: '24px',
          border: '1px solid rgba(244,114,182,0.28)',
          background: 'linear-gradient(135deg, rgba(244,114,182,0.10) 0%, rgba(168,85,247,0.06) 100%)',
          marginBottom: '28px',
        }}
      >
        <div style={{ color: '#f472b6', fontWeight: 'bold', fontSize: '18px', marginBottom: '16px' }}>
          模型结构对比
        </div>
        <div style={{ display: 'flex', gap: '0', marginBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.12)' }}>
          {modelTabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveModelTab(tab)}
              style={{
                padding: '10px 28px',
                background: activeModelTab === tab
                  ? 'linear-gradient(135deg, rgba(244,114,182,0.22) 0%, rgba(168,85,247,0.18) 100%)'
                  : 'transparent',
                border: 'none',
                borderBottom: activeModelTab === tab ? '2px solid #f472b6' : '2px solid transparent',
                color: activeModelTab === tab ? '#f9a8d4' : '#9ca3af',
                fontSize: '14px',
                fontWeight: activeModelTab === tab ? 'bold' : 'normal',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
            >
              {tab}
            </button>
          ))}
        </div>
        {/* Tab 内容区 */}

        {activeModelTab === 'Claude' ? (
          <ClaudeTab />
        ) : activeModelTab === 'GPT5' ? (
          <GPT5Tab
            evidenceDrivenPlan={evidenceDrivenPlan}
            improvements={improvements}
            expandedImprovementPhase={expandedImprovementPhase}
            setExpandedImprovementPhase={setExpandedImprovementPhase}
            expandedImprovementTest={expandedImprovementTest}
            setExpandedImprovementTest={setExpandedImprovementTest}
          />
        ) : (
          <GLM5Tab activeModelTab={activeModelTab} />
        )}
      </div>
    </div >
  );
};
