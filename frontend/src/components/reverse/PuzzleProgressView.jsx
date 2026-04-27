/**
 * 拼图进度视图
 * T1-T10核心测试的进度追踪
 */
import { TEST_PRESETS, STATUS_CONFIG } from '../../config/testPresets';
import { PUZZLE_STATUS_COLORS } from '../../config/reverseColorMaps';

export default function PuzzleProgressView({ activePreset }) {
  const summary = { confirmed: 0, partial: 0, pending: 0, missing: 0 };
  TEST_PRESETS.forEach((p) => {
    const s = p.status || 'pending';
    if (s in summary) summary[s]++;
  });
  const totalConfirmed = summary.confirmed + summary.partial;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
        <span style={{ fontSize: '11px', fontWeight: 700, color: '#dfe8ff' }}>拼图进度</span>
        <span style={{ fontSize: '10px', color: '#ffd93d' }}>{totalConfirmed}/10</span>
      </div>

      {/* Progress bar */}
      <div style={{
        height: '6px', borderRadius: '3px', background: 'rgba(255,255,255,0.06)',
        overflow: 'hidden', display: 'flex', marginBottom: '8px',
      }}>
        <div style={{ width: `${summary.confirmed * 10}%`, background: '#10b981', transition: 'width 0.3s' }} />
        <div style={{ width: `${summary.partial * 10}%`, background: '#f59e0b', transition: 'width 0.3s' }} />
      </div>

      {/* Test list */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
        {TEST_PRESETS.map((preset) => {
          const statusInfo = STATUS_CONFIG[preset.status] || STATUS_CONFIG.pending;
          const isActive = activePreset === preset.id;
          return (
            <div key={preset.id} style={{
              display: 'flex', alignItems: 'center', gap: '4px',
              padding: '3px 5px', borderRadius: '3px',
              background: isActive ? 'rgba(236, 72, 153, 0.1)' : 'transparent',
              fontSize: '10px',
            }}>
              <span style={{ width: '16px', color: '#888', fontWeight: 600 }}>{preset.id}</span>
              <span style={{ flex: 1, color: isActive ? '#eef7ff' : '#999', fontWeight: isActive ? 600 : 400 }}>
                {preset.label}
              </span>
              <span style={{ color: statusInfo.color, fontSize: '9px' }}>{statusInfo.icon}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
