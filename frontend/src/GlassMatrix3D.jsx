
import { Line, OrbitControls, Sphere, Text } from '@react-three/drei';
import { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';

/**
 * GlassMatrix3D - A visualization of the "Neural Fiber Bundle" theory.
 * 
 * Concept:
 * - Base Manifold (M): Represents the logical structure (Syntax).
 *   Rendered as a set of connected points (the centroids of syntactic templates).
 * - Fibers (F): Represent the semantic state space attached to each point on the manifold.
 *   Rendered as vertical beams or vector fields extending from the manifold points.
 * - Parallel Transport: The movement of semantic vectors across the manifold.
 */
const GlassMatrix3D = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load Data
  useEffect(() => {
    fetch('/nfb_data.json')
      .then(res => res.json())
      .then(d => {
        setData(d);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load NFB data:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <group>
        <Text fontSize={0.5} color="white" position={[0, 0, 0]}>
          Loading Structure...
        </Text>
      </group>
    );
  }

  if (!data) {
    return (
      <group>
        <Text fontSize={0.5} color="red" position={[0, 0, 0]}>
          Data Not Found
        </Text>
      </group>
    );
  }

  return (
    <group>
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />

      {/* Visualization Roots */}
      <ManifoldLayer centroids={data.manifold_centroids} />
      <FiberLayer centroids={data.manifold_centroids} basis={data.fiber_basis} variance={data.fiber_variance} />
      
      {/* Controls */}
      <OrbitControls />
    </group>
  );
};

// --- Sub-components ---

const ManifoldLayer = ({ centroids }) => {
  // Render the Base Manifold as a point cloud + connections
  const points = useMemo(() => {
    return centroids.map(c => new THREE.Vector3(c[0], c[1], c[2]));
  }, [centroids]);

  return (
    <group>
      {/* Manifold Points */}
      {points.map((p, i) => (
        <Sphere key={i} args={[0.2, 16, 16]} position={p}>
          <meshStandardMaterial color="#00ffff" emissive="#0088aa" emissiveIntensity={0.5} />
        </Sphere>
      ))}

      {/* Connectivity (Simplified: Connect sequential points for now, or MST in future) */}
      <Line points={points} color="#00ffff" lineWidth={1} transparent opacity={0.3} />
    </group>
  );
};

const FiberLayer = ({ centroids, basis, variance }) => {
  // Render Fibers extending from each Manifold point.
  // Visualized as vertical lines or "hairs" representing the fiber space.
  
  return (
    <group>
      {centroids.map((c, i) => (
        <group key={i} position={[c[0], c[1], c[2]]}>
          {/* Render the Fiber Basis vectors at this point */}
          <FiberBundle basis={basis} variance={variance} />
        </group>
      ))}
    </group>
  );
};

const FiberBundle = ({ basis, variance }) => {
  // Visualizes the local vector space (Fiber)
  // For demo: 3 orthogonal lines colored RGB
  
  return (
    <group>
      {/* Basis Vector 1 (Red) */}
      <Line 
        points={[[0, 0, 0], [basis[0][0]*variance[0]*5, basis[0][1]*variance[0]*5, basis[0][2]*variance[0]*5]]} 
        color="red" 
        lineWidth={2} 
      />
      {/* Basis Vector 2 (Green) */}
      <Line 
        points={[[0, 0, 0], [basis[1][0]*variance[1]*5, basis[1][1]*variance[1]*5, basis[1][2]*variance[1]*5]]} 
        color="green" 
        lineWidth={2} 
      />
      {/* Basis Vector 3 (Blue) */}
      <Line 
        points={[[0, 0, 0], [basis[2][0]*variance[2]*5, basis[2][1]*variance[2]*5, basis[2][2]*variance[2]*5]]} 
        color="blue" 
        lineWidth={2} 
      />
    </group>
  );
};

export default GlassMatrix3D;
