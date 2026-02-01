import { Html } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef, useState } from 'react';
import * as THREE from 'three';

// --- Geometry Components ---

const Neuron = ({ pos, isFired, layer, id }) => {
  // Color code by layer
  const color = useMemo(() => {
    if (layer.includes("Shape")) return "#4ecdc4"; // Cyan
    if (layer.includes("Color")) return "#ff6b6b"; // Red
    return "#ffe66d"; // Yellow (Fiber)
  }, [layer]);

  // Glow effect when fired
  const scale = isFired ? 1.5 : 1.0;
  const emissive = isFired ? color : "#000000";
  const emissiveIntensity = isFired ? 2.0 : 0.0;

  return (
    <group position={pos}>
      <mesh scale={[scale, scale, scale]}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshStandardMaterial 
            color={color} 
            emissive={emissive}
            emissiveIntensity={emissiveIntensity}
            roughness={0.3}
            metalness={0.8}
        />
      </mesh>
      {/* Label for key neurons */}
      {id.includes("12") && (
         <Html distanceFactor={10}>
            <div style={{ color: 'white', fontSize: '8px', background: 'rgba(0,0,0,0.5)', padding: '2px' }}>
              {layer.includes("Shape") ? "Round" : (layer.includes("Color") ? "Red" : "Concept")}
            </div>
         </Html>
      )}
    </group>
  );
};

const Axon = ({ start, end, isActive }) => {
    const ref = useRef()
    
    // Create geometry
    const points = useMemo(() => [
        new THREE.Vector3(...start),
        new THREE.Vector3(...end)
    ], [start, end])
    
    const lineGeometry = useMemo(() => {
        const geo = new THREE.BufferGeometry().setFromPoints(points)
        return geo
    }, [points])

    return (
        <line geometry={lineGeometry}>
            <lineBasicMaterial 
                color={isActive ? "#ffffff" : "#444444"} 
                transparent 
                opacity={isActive ? 0.8 : 0.2} 
                linewidth={1} 
            />
        </line>
    );
};

// --- Main Visualization Component ---

const BrainVis3D = ({ t, onStatusUpdate }) => {
    const [data, setData] = useState(null);
    const [frames, setFrames] = useState([]);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentStep, setCurrentStep] = useState(0);
    const [activeNeurons, setActiveNeurons] = useState(new Set());
    
    // Fetch data
    const runSimulation = async () => {
        try {
            const res = await fetch("http://127.0.0.1:8888/snn/run_simulation_3d", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ duration: 200 })
            });
            const json = await res.json();
            setData(json.structure);
            setFrames(json.frames);
            setCurrentStep(0);
            setIsPlaying(true);
        } catch (e) {
            console.error("Simulation failed", e);
        }
    };

    // Animation Loop
    useFrame((state, delta) => {
        if (!isPlaying || frames.length === 0) return;

        // Slow down animation 
        // Simple counter based frame advance (approx 30fps -> 1sim step per 2 frames?)
        // Let's just do simple speed control
        if (state.clock.getElapsedTime() % 0.1 < 0.05) { // Update every ~0.1s
            const frame = frames.find(f => f.time === currentStep);
            
            // Auto-decay active neurons
            // Actually, we should just set active to what's in the frame
            // But to make it look "flashy", we can keep them on for a bit?
            // For now, simple: frame.fired are ON, others OFF (unless we want trails)
            
            if (frame) {
                setActiveNeurons(new Set(frame.fired));
            } else {
                 setActiveNeurons(new Set());
            }

            if (currentStep < 200) {
                 setCurrentStep(s => s + 1);
            } else {
                 setIsPlaying(false); // Stop at end
            }
            
            // Report status to parent
            if (onStatusUpdate) {
                onStatusUpdate({
                    step: currentStep,
                    activeCount: frame ? frame.fired.length : 0,
                    isPlaying: true,
                    description: "Simulating Spiking Dynamics..."
                });
            }
        }
    });

    const neurons = useMemo(() => data?.neurons || [], [data]);
    const connections = useMemo(() => {
        if (!data) return [];
        // Map ID to pos
        const posMap = {};
        data.neurons.forEach(n => posMap[n.id] = n.pos);
        
        return data.connections.map(c => ({
            ...c,
            start: posMap[c.srcId],
            end: posMap[c.tgtId]
        })).filter(c => c.start && c.end);
    }, [data]);

    return (
        <group>
            {/* Visuals */}
            <group scale={[0.5, 0.5, 0.5]}> {/* Scale down to fit view */}
                {neurons.map(n => (
                    <Neuron 
                        key={n.id} 
                        {...n} 
                        isFired={activeNeurons.has(n.id)} 
                    />
                ))}
                {connections.map((c, i) => (
                    <Axon 
                        key={i} 
                        start={c.start} 
                        end={c.end} 
                        isActive={activeNeurons.has(c.srcId)} 
                    />
                ))}
            </group>

            {/* Controls (3D UI) */}
             <Html position={[0, -5, 0]} center>
                <div style={{ display: 'flex', gap: '10px', pointerEvents: 'auto' }}>
                    <button 
                        onClick={runSimulation}
                        style={{
                            background: '#4ecdc4',
                            border: 'none',
                            padding: '8px 16px',
                            borderRadius: '4px',
                            color: '#1a1a2e',
                            fontWeight: 'bold',
                            cursor: 'pointer'
                        }}
                    >
                        {t ? t('snn.inject', 'Inject "Apple" Stimulus') : 'Inject "Apple" Stimulus'}
                    </button>
                    <div style={{ color: 'white', padding: '8px' }}>
                        Time: {currentStep}ms
                    </div>
                </div>
            </Html>
        </group>
    );
};

export default BrainVis3D;
