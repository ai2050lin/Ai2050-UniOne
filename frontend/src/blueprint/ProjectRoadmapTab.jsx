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
