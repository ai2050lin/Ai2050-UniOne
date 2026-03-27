export const ProjectRoadmapTab = ({
  roadmapData,
  mathRouteSystemPlan,
}) => {
  return (
    <div style={{ animation: 'roadmapFade 0.6s ease-out', maxWidth: '1180px', margin: '0 auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: '30px', fontWeight: '900', color: '#ffaa00', marginBottom: '10px' }}>战略层级路线图</h2>
        <div style={{ color: '#9ca3af', fontSize: '14px' }}>{roadmapData?.definition?.summary || '聚焦结构智能路线。'}</div>
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
            '1. 大脑应存在高度结构化的数学组织，这可能是智能产生的基础。',
            '2. 深度神经网络可能部分还原了该结构，因此具备可扩展的语言与推理能力。',
            '3. 通过结构分析与可控干预，提取可验证的编码规律，形成更通用的智能理论。',
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
          padding: '24px',
          background: 'linear-gradient(135deg, rgba(45, 212, 191, 0.10) 0%, rgba(45, 212, 191, 0.03) 100%)',
          border: '1px solid rgba(45, 212, 191, 0.22)',
          borderRadius: '20px',
          marginBottom: '28px',
        }}
      >
        <div style={{ color: '#5eead4', fontWeight: 'bold', fontSize: '18px', marginBottom: '10px' }}>理论分析承载区</div>
        <div style={{ color: '#d5fffb', fontSize: '13px', lineHeight: '1.8', marginBottom: '12px' }}>
          主界面现在只保留数据观察、差异比较、样本验证和回放。理论对象、理论解释、长期战略路线和第一性原理判断，统一收口到战略层级路线图中。
        </div>
        <div style={{ display: 'grid', gap: '8px' }}>
          {[
            '1. 主界面负责数据：看当前样本、当前拼图、当前差异、当前验证缺口。',
            '2. 路线图负责理论：看统一骨架、长期瓶颈、阶段策略和理论突破路线。',
            '3. 后续所有理论分析都应优先写入路线图和研究文档，而不是重新回流到主数据界面。',
          ].map((line, idx) => (
            <div
              key={idx}
              style={{
                padding: '12px 14px',
                borderRadius: '12px',
                background: 'rgba(255,255,255,0.04)',
                color: '#d9fffb',
                fontSize: '13px',
                lineHeight: '1.7',
              }}
            >
              {line}
            </div>
          ))}
        </div>
      </div>

      <div
        style={{
          padding: '24px',
          background: 'linear-gradient(135deg, rgba(96, 165, 250, 0.10) 0%, rgba(96, 165, 250, 0.03) 100%)',
          border: '1px solid rgba(96, 165, 250, 0.22)',
          borderRadius: '20px',
          marginBottom: '28px',
        }}
      >
        <div style={{ color: '#93c5fd', fontWeight: 'bold', fontSize: '18px', marginBottom: '10px' }}>主界面迁移内容</div>
        <div style={{ color: '#dbeafe', fontSize: '13px', lineHeight: '1.8', marginBottom: '12px' }}>
          控制面板现在只保留操作入口。下面这些原本容易堆在左侧的展示型区块，统一迁到战略层级路线图中承载，用来做全局理解、研究判断和阶段调度。
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '12px' }}>
          {[
            {
              title: '语言主线控制台',
              desc: '负责整理主流程顺序、当前焦点与整体研究节奏，不再占用主界面的操作空间。',
            },
            {
              title: '研究总览',
              desc: '负责汇总当前最关键的数据状态、风险、节点池和研究层关系，用于全局判断。',
            },
            {
              title: '关键性质',
              desc: '负责沉淀稳定性、可分性、耦合强度和观测完整度这类性质判断，不与即时操作混放。',
            },
            {
              title: '验证入口',
              desc: '负责管理样本验证节奏、阶段覆盖、缺口资产和优先验证任务，用于战略调度。',
            },
            {
              title: '概念关联',
              desc: '负责承载跨层概念锚点、层间关联和关键概念观察，不再占用常驻控制区。',
            },
            {
              title: '拼图对比台',
              desc: '负责拼图差异、修复前后对照、变量级比较和共享子回路候选，作为路线级分析入口。',
            },
          ].map((item) => (
            <div
              key={item.title}
              style={{
                padding: '14px',
                borderRadius: '12px',
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
              }}
            >
              <div style={{ color: '#eff6ff', fontSize: '14px', fontWeight: 'bold', marginBottom: '6px' }}>{item.title}</div>
              <div style={{ color: '#cfe3ff', fontSize: '12px', lineHeight: '1.7' }}>{item.desc}</div>
            </div>
          ))}
        </div>
      </div>

      <div
        style={{
          padding: '24px',
          background: 'linear-gradient(135deg, rgba(244, 114, 182, 0.10) 0%, rgba(244, 114, 182, 0.03) 100%)',
          border: '1px solid rgba(244, 114, 182, 0.22)',
          borderRadius: '20px',
          marginBottom: '28px',
        }}
      >
        <div style={{ color: '#f9a8d4', fontWeight: 'bold', fontSize: '18px', marginBottom: '10px' }}>界面模块草稿图</div>
        <div style={{ color: '#fce7f3', fontSize: '13px', lineHeight: '1.8', marginBottom: '12px' }}>
          这不是最终界面，而是后续重构的统一草稿。核心目标是把“操作、观察、数据、验证、战略”五类职责彻底拆开，避免控制面板继续变成信息堆积区。
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, minmax(0, 1fr))', gap: '12px', marginBottom: '14px' }}>
          {[
            { title: '左侧操作栏', desc: '只放研究层切换、场景模式、筛选、样本投射、动画控制。' },
            { title: '中央 3D 主场景', desc: '只负责当前概念、拼图、样本的三维观察和局部回放。' },
            { title: '右侧数据面板', desc: '只负责当前焦点、层数据、样本回放、资产与证据。' },
            { title: '底部时间轴', desc: '只负责 before / bridge / after 阶段切换和验证状态。' },
            { title: '战略层级路线图', desc: '统一承载总览、性质、概念关联、拼图对比和理论主线。' },
          ].map((item) => (
            <div
              key={item.title}
              style={{
                padding: '14px',
                borderRadius: '12px',
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
              }}
            >
              <div style={{ color: '#fff1f7', fontSize: '14px', fontWeight: 'bold', marginBottom: '6px' }}>{item.title}</div>
              <div style={{ color: '#fbcfe8', fontSize: '12px', lineHeight: '1.7' }}>{item.desc}</div>
            </div>
          ))}
        </div>

        <div
          style={{
            padding: '14px',
            borderRadius: '12px',
            background: 'rgba(10, 10, 18, 0.42)',
            border: '1px solid rgba(255,255,255,0.08)',
            color: '#fdf2f8',
            fontSize: '12px',
            lineHeight: '1.7',
            whiteSpace: 'pre-wrap',
            fontFamily: '"Cascadia Code", "Consolas", monospace',
          }}
        >
          {INTERFACE_MODULE_DRAFT_WIREFRAME}
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
          {mathRouteSystemPlan?.title || '数学路线'}
        </div>
        <div style={{ color: '#c7d2fe', fontSize: '13px', lineHeight: '1.7', marginBottom: '14px' }}>
          {mathRouteSystemPlan?.subtitle || '对比多路线理论深度、计算可行性与解释性。'}
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
          <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '8px' }}>路线对比</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', minWidth: '1180px', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: 'rgba(255,255,255,0.05)' }}>
                  <th style={cellHeaderStyle('#c7d2fe')}>路线</th>
                  <th style={cellHeaderStyle('#93c5fd')}>路线说明</th>
                  <th style={cellHeaderStyle('#86efac')}>优点</th>
                  <th style={cellHeaderStyle('#fca5a5')}>缺点</th>
                  <th style={cellHeaderStyle('#93c5fd')}>可行性结论</th>
                  <th style={cellHeaderStyle('#c7d2fe')}>理论深度</th>
                  <th style={cellHeaderStyle('#c7d2fe')}>计算可行性</th>
                  <th style={cellHeaderStyle('#c7d2fe')}>可解释性</th>
                  <th style={cellHeaderStyle('#c7d2fe')}>与 SHMC/NFBT 兼容</th>
                </tr>
              </thead>
              <tbody>
                {(mathRouteSystemPlan?.routeAnalysis || []).map((item, idx) => (
                  <tr key={idx} style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                    <td style={cellBodyStyle('#e0e7ff', true)}>{item.route}</td>
                    <td style={cellBodyStyle('#bfdbfe')}>{item.routeSummary || item.description || item.routeDesc || '-'}</td>
                    <td style={cellBodyStyle('#dcfce7')}>
                      {(item.pros || []).map((line, pIdx) => (
                        <div key={pIdx}>{`${pIdx + 1}. ${line}`}</div>
                      ))}
                    </td>
                    <td style={cellBodyStyle('#fee2e2')}>
                      {(item.cons || []).map((line, cIdx) => (
                        <div key={cIdx}>{`${cIdx + 1}. ${line}`}</div>
                      ))}
                    </td>
                    <td style={cellBodyStyle('#bae6fd')}>{item.feasibility}</td>
                    <td style={cellBodyStyle('#dbeafe')}>{item.depth}</td>
                    <td style={cellBodyStyle('#dbeafe')}>{item.compute}</td>
                    <td style={cellBodyStyle('#dbeafe')}>{item.interpret}</td>
                    <td style={cellBodyStyle('#dbeafe')}>{item.compatibility}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr', gap: '12px' }}>
          <div style={infoCardStyle}>
            <div style={infoCardTitleStyle}>分层架构</div>
            {(mathRouteSystemPlan?.architecture || []).map((line, idx) => (
              <div key={idx} style={infoLineStyle}>
                {`${idx + 1}. ${line}`}
              </div>
            ))}
          </div>

          <div style={infoCardStyle}>
            <div style={infoCardTitleStyle}>资源配比</div>
            {(mathRouteSystemPlan?.allocation || []).map((line, idx) => (
              <div key={idx} style={infoLineStyle}>
                {line}
              </div>
            ))}
          </div>

          <div style={infoCardStyle}>
            <div style={infoCardTitleStyle}>阶段里程碑</div>
            {(mathRouteSystemPlan?.milestones || []).map((line, idx) => (
              <div key={idx} style={infoLineStyle}>
                {`${idx + 1}. ${line}`}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const cellHeaderStyle = (color) => ({
  textAlign: 'left',
  padding: '8px 10px',
  color,
  fontSize: '11px',
  borderBottom: '1px solid rgba(255,255,255,0.08)',
});

const cellBodyStyle = (color, bold = false) => ({
  padding: '9px 10px',
  color,
  fontSize: '11px',
  lineHeight: '1.55',
  verticalAlign: 'top',
  fontWeight: bold ? 'bold' : 'normal',
});

const infoCardStyle = {
  padding: '14px',
  borderRadius: '12px',
  background: 'rgba(0,0,0,0.22)',
  border: '1px solid rgba(255,255,255,0.08)',
};

const infoCardTitleStyle = {
  color: '#a5b4fc',
  fontSize: '11px',
  fontWeight: 'bold',
  marginBottom: '6px',
};

const infoLineStyle = {
  color: '#dbeafe',
  fontSize: '12px',
  lineHeight: '1.6',
  marginBottom: '4px',
};

const INTERFACE_MODULE_DRAFT_WIREFRAME = `┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ 顶栏：系统切换 | 当前工作台 | 当前样本 | 路线图入口 | 帮助 | 语言切换                      │
├───────────────┬──────────────────────────────────────────────┬──────────────────────────────┤
│ 左侧操作栏    │ 中央 3D 主场景                               │ 右侧数据面板                 │
│ 1. 研究层切换 │ 1. 当前概念 / 拼图 / 样本的三维主视图        │ Tab 1 当前焦点               │
│ 2. 场景模式   │ 2. 节点、链路、层间结构                      │ Tab 2 层数据                 │
│ 3. 条件筛选   │ 3. 差异高亮                                  │ Tab 3 样本回放               │
│ 4. 样本投射   │ 4. 局部链路回放                              │ Tab 4 资产与证据             │
│ 5. 动画控制   │ 5. 选中对象的空间反馈                        │                              │
├───────────────┴──────────────────────────────────────────────┴──────────────────────────────┤
│ 底部状态与时间轴：before | bridge | after | 当前验证状态 | 当前风险 | 当前缺口资产         │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ 战略层级路线图：总览 | 性质 | 概念关联 | 拼图对比 | 理论主线 | 阶段路线                   │
└──────────────────────────────────────────────────────────────────────────────────────────────┘`;
