import React, { useMemo, useState } from 'react';
import { Atom, Brain, Cpu, Network, Sparkles } from 'lucide-react';
import { AppleNeuron3DTab } from './AppleNeuron3DTab';

const MODES = [
  { id: 'main', label: 'MAIN', icon: Sparkles, color: '#38bdf8' },
  { id: 'dnn', label: 'DNN', icon: Cpu, color: '#22c55e' },
  { id: 'snn', label: 'SNN', icon: Brain, color: '#f59e0b' },
  { id: 'fibernet', label: 'FIBERNET', icon: Network, color: '#a855f7' },
];

const MODE_SUMMARY = {
  dnn: {
    title: 'DNN View',
    points: [
      'Standard transformer statistical routing analysis.',
      'Focus: layer-wise activation sparsity and local circuits.',
      'Use MAIN mode for apple neuron 3D and concept textbox generation.',
    ],
  },
  snn: {
    title: 'SNN View',
    points: [
      'Spiking-style sparse event model mapping.',
      'Focus: threshold events, phase-locking, energy efficiency.',
      'Use MAIN mode for full layer + neuron visualization workspace.',
    ],
  },
  fibernet: {
    title: 'FiberNet View',
    points: [
      'Manifold + fiber transport geometric route.',
      'Focus: base manifold, connection transport, global arbitration.',
      'Use MAIN mode for apple neuron compare controls and 3D space.',
    ],
  },
};

const UniversalManifoldGraph = () => {
  const [activeMode, setActiveMode] = useState('main');

  const activeModeMeta = useMemo(
    () => MODES.find((m) => m.id === activeMode) || MODES[0],
    [activeMode]
  );

  return (
    <div
      style={{
        marginTop: '24px',
        borderRadius: '16px',
        border: '1px solid rgba(148, 163, 184, 0.28)',
        background: 'linear-gradient(145deg, rgba(6, 10, 22, 0.92) 0%, rgba(12, 19, 34, 0.92) 100%)',
        padding: '18px',
        boxShadow: '0 16px 40px rgba(0,0,0,0.35)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Atom size={20} color="#67e8f9" />
          <div style={{ color: '#dbeafe', fontSize: '16px', fontWeight: 800 }}>
            Universal Manifold Control
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          {MODES.map((mode) => {
            const Icon = mode.icon;
            const active = mode.id === activeMode;
            return (
              <button
                key={mode.id}
                type="button"
                onClick={() => setActiveMode(mode.id)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  borderRadius: '999px',
                  border: `1px solid ${active ? mode.color : 'rgba(148, 163, 184, 0.3)'}`,
                  background: active ? `${mode.color}22` : 'rgba(2, 6, 23, 0.65)',
                  color: active ? '#e2e8f0' : '#94a3b8',
                  padding: '6px 12px',
                  fontSize: '11px',
                  fontWeight: 700,
                  letterSpacing: '0.4px',
                  cursor: 'pointer',
                }}
              >
                <Icon size={13} color={active ? mode.color : '#94a3b8'} />
                {mode.label}
              </button>
            );
          })}
        </div>
      </div>

      {activeMode === 'main' ? (
        <AppleNeuron3DTab panelPosition="left" sceneHeight="72vh" />
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: '20px' }}>
          <div
            style={{
              borderRadius: '14px',
              border: '1px solid rgba(148,163,184,0.22)',
              background: 'linear-gradient(170deg, rgba(15,24,42,0.94), rgba(7,12,25,0.95))',
              padding: '14px',
            }}
          >
            <div style={{ color: '#e2e8f0', fontSize: '14px', fontWeight: 700, marginBottom: '10px' }}>
              {MODE_SUMMARY[activeMode]?.title || 'Mode'}
            </div>
            {(MODE_SUMMARY[activeMode]?.points || []).map((point) => (
              <div key={point} style={{ color: '#93c5fd', fontSize: '12px', lineHeight: 1.7, marginBottom: '8px' }}>
                {point}
              </div>
            ))}
          </div>

          <div
            style={{
              height: '72vh',
              borderRadius: '16px',
              border: `1px solid ${activeModeMeta.color}66`,
              background:
                'radial-gradient(circle at 20% 10%, rgba(56,189,248,0.2), rgba(9,12,22,0.96) 58%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              padding: '24px',
            }}
          >
            <div>
              <div style={{ color: '#e2e8f0', fontSize: '18px', fontWeight: 800, marginBottom: '10px' }}>
                {activeModeMeta.label} workspace
              </div>
              <div style={{ color: '#94a3b8', fontSize: '13px', lineHeight: 1.7 }}>
                Switch to MAIN to render full layer topology and corresponding neurons in 3D.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UniversalManifoldGraph;
