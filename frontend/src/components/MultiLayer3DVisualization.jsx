import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const API_BASE_URL = 'http://localhost:8000';

/**
 * 多层3D可视化组件
 * 用于展示概念在神经网络各层的3D表示和层间关联性
 */
const MultiLayer3DVisualization = ({ concept, showAssociations = true }) => {
  const [layerData, setLayerData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [showFlowPaths, setShowFlowPaths] = useState(true);
  const [showCrossLayerEdges, setShowCrossLayerEdges] = useState(true);

  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const raycasterRef = useRef(null);
  const mouseRef = useRef(new THREE.Vector2());
  const layersRef = useRef({});
  const edgesRef = useRef({});

  useEffect(() => {
    if (concept) {
      loadLayerData();
    }

    return () => {
      cleanupThreeJS();
    };
  }, [concept]);

  useEffect(() => {
    if (layerData && canvasRef.current) {
      initThreeJS();
    }

    // 添加窗口大小监听
    const handleResize = () => {
      if (cameraRef.current && rendererRef.current && canvasRef.current) {
        const width = canvasRef.current.clientWidth;
        const height = canvasRef.current.clientHeight;
        cameraRef.current.aspect = width / height;
        cameraRef.current.updateProjectionMatrix();
        rendererRef.current.setSize(width, height);
      }
    };
    window.addEventListener('resize', handleResize);

    // 添加鼠标点击事件
    const handleClick = (event) => {
      if (!canvasRef.current || !cameraRef.current || !sceneRef.current) return;

      const rect = canvasRef.current.getBoundingClientRect();
      mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);

      // 检测层边框的点击
      const layerMeshes = [];
      Object.values(layersRef.current).forEach(layer => {
        layer.children.forEach(child => {
          if (child.type === 'Mesh' && child.geometry.type === 'BoxGeometry') {
            layerMeshes.push(child);
          }
        });
      });

      const intersects = raycasterRef.current.intersectObjects(layerMeshes);
      if (intersects.length > 0) {
        const clickedMesh = intersects[0].object;
        const layerId = clickedMesh.parent.userData.layerId;
        if (layerId) {
          handleLayerSelect(layerId);
        }
      }
    };

    if (canvasRef.current) {
      canvasRef.current.addEventListener('click', handleClick);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      if (canvasRef.current) {
        canvasRef.current.removeEventListener('click', handleClick);
      }
    };
  }, [layerData]);

  useEffect(() => {
    if (showAssociations && layerData) {
      updateAssociations();
    }
  }, [showAssociations, showFlowPaths, showCrossLayerEdges]);

  const loadLayerData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/layer-association/analyze`,
        concept
      );
      setLayerData(response.data);
      setSelectedLayer(null);
      setSelectedNode(null);
    } catch (error) {
      console.error('加载层数据失败:', error);
      setError('加载层数据失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const initThreeJS = () => {
    if (!canvasRef.current || !layerData) return;

    // 等待容器有尺寸
    if (canvasRef.current.clientWidth === 0 || canvasRef.current.clientHeight === 0) {
      setTimeout(initThreeJS, 100);
      return;
    }

    // 清理旧的ThreeJS实例
    cleanupThreeJS();

    // 创建场景
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    sceneRef.current = scene;

    // 创建相机
    const camera = new THREE.PerspectiveCamera(
      75,
      canvasRef.current.clientWidth / canvasRef.current.clientHeight,
      0.1,
      10000
    );
    camera.position.set(0, 300, 600); // 调整相机位置更接近场景中心
    camera.lookAt(0, 300, 0);
    cameraRef.current = camera;

    // 创建渲染器
    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true
    });
    renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    rendererRef.current = renderer;

    // 初始化Raycaster
    raycasterRef.current = new THREE.Raycaster();

    // 添加OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 300, 0);
    controlsRef.current = controls;

    // 添加光源
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(500, 500, 500);
    scene.add(directionalLight);

    // 创建各层的3D表示
    Object.entries(layerData.layers).forEach(([layerId, layer]) => {
      createLayer3D(layerId, layer);
    });

    // 添加层间关联
    if (showAssociations) {
      createLayerAssociations();
    }

    // 添加坐标轴
    const axesHelper = new THREE.AxesHelper(200);
    scene.add(axesHelper);

    // 添加网格
    const gridHelper = new THREE.GridHelper(1000, 20, 0x444444, 0x222222);
    gridHelper.position.y = 0;
    scene.add(gridHelper);

    // 开始动画循环
    animate();
  };

  const createLayer3D = (layerId, layer) => {
    if (!sceneRef.current) return;

    const scene = sceneRef.current;
    const layers = layersRef.current;

    // 创建层容器
    const layerGroup = new THREE.Group();
    layerGroup.position.set(
      layer.position[0],
      layer.position[1],
      layer.position[2]
    );
    layerGroup.userData = { layerId, ...layer };
    layers[layerId] = layerGroup;

    // 创建层边框（半透明立方体）- 可点击
    const boxGeometry = new THREE.BoxGeometry(300, 150, 150);
    const boxMaterial = new THREE.MeshPhongMaterial({
      color: layer.color,
      transparent: true,
      opacity: 0.2,
      wireframe: true,
      side: THREE.DoubleSide
    });
    const boxMesh = new THREE.Mesh(boxGeometry, boxMaterial);
    boxMesh.userData = { layerId, isLayerBox: true };
    layerGroup.add(boxMesh);

    // 创建层标签
    const label = createTextLabel(layer.layer_name, 0, -80, 0);
    layerGroup.add(label);

    // 创建节点
    layer.nodes.forEach((node, index) => {
      const nodeMesh = createNode(node, index);
      nodeMesh.userData = { nodeId: node.id, ...node, layerId };
      layerGroup.add(nodeMesh);
    });

    // 创建层内边
    const edgesGroup = new THREE.Group();
    layer.edges.forEach((edge, index) => {
      const edgeLine = createEdge(edge, layer.nodes);
      edgeLine.userData = { edgeId: index, ...edge, layerId };
      edgesGroup.add(edgeLine);
    });
    edgesRef.current[layerId] = edgesGroup;
    layerGroup.add(edgesGroup);

    scene.add(layerGroup);
  };

  const createNode = (node, index) => {
    const geometry = new THREE.SphereGeometry(node.size, 32, 32);
    const material = new THREE.MeshPhongMaterial({
      color: node.color,
      emissive: node.color,
      emissiveIntensity: node.activation * 0.5,
      transparent: true,
      opacity: 0.8
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(
      node.position[0],
      node.position[1],
      node.position[2]
    );
    return mesh;
  };

  const createEdge = (edge, nodes) => {
    const sourceNode = nodes.find(n => n.id === edge.source);
    const targetNode = nodes.find(n => n.id === edge.target);

    if (!sourceNode || !targetNode) return null;

    const points = [
      new THREE.Vector3(sourceNode.position[0], sourceNode.position[1], sourceNode.position[2]),
      new THREE.Vector3(targetNode.position[0], targetNode.position[1], targetNode.position[2])
    ];

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: edge.color,
      transparent: true,
      opacity: edge.strength * 0.6
    });

    return new THREE.Line(geometry, material);
  };

  const createLayerAssociations = () => {
    if (!layerData || !layerData.associations) return;

    const scene = sceneRef.current;
    if (!scene) return;

    // 创建跨层边
    if (showCrossLayerEdges) {
      layerData.associations.forEach((association, index) => {
        association.cross_layer_edges.forEach(edge => {
          const sourceLayer = layerData.layers[association.source_layer];
          const targetLayer = layerData.layers[association.target_layer];

          const sourceNode = sourceLayer.nodes.find(n => n.id === edge.source);
          const targetNode = targetLayer.nodes.find(n => n.id === edge.target);

          if (sourceNode && targetNode) {
            const sourcePos = {
              x: sourceLayer.position[0] + sourceNode.position[0],
              y: sourceLayer.position[1] + sourceNode.position[1],
              z: sourceLayer.position[2] + sourceNode.position[2]
            };

            const targetPos = {
              x: targetLayer.position[0] + targetNode.position[0],
              y: targetLayer.position[1] + targetNode.position[1],
              z: targetLayer.position[2] + targetNode.position[2]
            };

            const points = [
              new THREE.Vector3(sourcePos.x, sourcePos.y, sourcePos.z),
              new THREE.Vector3(targetPos.x, targetPos.y, targetPos.z)
            ];

            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({
              color: edge.color,
              transparent: true,
              opacity: edge.strength * 0.4,
              linewidth: 2
            });

            const line = new THREE.Line(geometry, material);
            scene.add(line);
          }
        });
      });
    }

    // 创建流向路径
    if (showFlowPaths) {
      layerData.flow_paths.forEach(path => {
        createFlowPath(path);
      });
    }
  };

  const createFlowPath = (path) => {
    const scene = sceneRef.current;
    if (!scene || !layerData) return;

    const points = [];
    path.layers.forEach(layerId => {
      const layer = layerData.layers[layerId];
      if (layer) {
        points.push(new THREE.Vector3(
          layer.position[0],
          layer.position[1],
          layer.position[2]
        ));
      }
    });

    if (points.length < 2) return;

    const curve = new THREE.CatmullRomCurve3(points);
    const tubeGeometry = new THREE.TubeGeometry(curve, 64, 3, 8, false);
    const tubeMaterial = new THREE.MeshPhongMaterial({
      color: 0x00ff00,
      transparent: true,
      opacity: 0.3,
      emissive: 0x00ff00,
      emissiveIntensity: 0.2
    });

    const tube = new THREE.Mesh(tubeGeometry, tubeMaterial);
    tube.userData = { pathId: path.path_id, ...path };
    scene.add(tube);
  };

  const createTextLabel = (text, x, y, z) => {
    // 简化版：使用线框盒子代替文本标签
    const geometry = new THREE.BoxGeometry(text.length * 10, 15, 5);
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.8
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x, y, z);
    return mesh;
  };

  const updateAssociations = () => {
    if (!sceneRef.current) return;

    // 清除旧的关联可视化
    const children = [...sceneRef.current.children];
    children.forEach(child => {
      if (child.userData.pathId) {
        sceneRef.current.remove(child);
      }
    });

    // 重新创建关联
    if (showFlowPaths && layerData) {
      layerData.flow_paths.forEach(path => {
        createFlowPath(path);
      });
    }
  };

  const animate = () => {
    if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

    requestAnimationFrame(animate);

    // 更新控制器
    if (controlsRef.current) {
      controlsRef.current.update();
    }

    // 旋转所有层
    Object.values(layersRef.current).forEach(layer => {
      layer.rotation.y += 0.002;
    });

    rendererRef.current.render(sceneRef.current, cameraRef.current);
  };

  const cleanupThreeJS = () => {
    if (rendererRef.current) {
      rendererRef.current.dispose();
      rendererRef.current = null;
    }
    if (sceneRef.current) {
      sceneRef.current.clear();
      sceneRef.current = null;
    }
    if (controlsRef.current) {
      controlsRef.current.dispose();
      controlsRef.current = null;
    }
    layersRef.current = {};
    edgesRef.current = {};
  };

  const handleLayerSelect = (layerId) => {
    setSelectedLayer(layerId === selectedLayer ? null : layerId);
    setSelectedNode(null);
  };

  const handleNodeSelect = (nodeData) => {
    setSelectedNode(nodeData);
  };

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="spinner"></div>
        <div className="loading-text">加载3D可视化数据...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-message">
        {error}
        <button
          className="btn btn-primary"
          onClick={loadLayerData}
          style={{ marginLeft: '10px' }}
        >
          重试
        </button>
      </div>
    );
  }

  return (
    <div className="multi-layer-3d-visualization">
      <div className="visualization-controls" style={{ marginBottom: '15px' }}>
        <button
          className="btn btn-secondary"
          onClick={() => setShowFlowPaths(!showFlowPaths)}
        >
          {showFlowPaths ? '隐藏流向路径' : '显示流向路径'}
        </button>
        <button
          className="btn btn-secondary"
          onClick={() => setShowCrossLayerEdges(!showCrossLayerEdges)}
        >
          {showCrossLayerEdges ? '隐藏跨层连接' : '显示跨层连接'}
        </button>
      </div>

      <div className="visualization-container">
        <div className="threejs-canvas" style={{ width: '100%', height: '600px', position: 'relative' }}>
          <canvas ref={canvasRef} style={{ cursor: 'pointer' }} />
          <div style={{
            position: 'absolute',
            top: '10px',
            left: '10px',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            padding: '10px',
            borderRadius: '8px',
            fontSize: '12px',
            color: '#333',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
          }}>
            <strong>操作提示:</strong><br/>
            • 左键拖动：旋转视角<br/>
            • 右键拖动：平移<br/>
            • 滚轮：缩放<br/>
            • 点击层边框：查看详情
          </div>
        </div>

        <div className="layer-info-panel" style={{ marginTop: '15px' }}>
          {selectedLayer && layerData && layerData.layers[selectedLayer] && (
            <div className="selected-layer-info">
              <h3>{layerData.layers[selectedLayer].layer_name}</h3>
              <p>{layerData.layers[selectedLayer].layer_description}</p>
              <div className="layer-statistics">
                <p>节点数: {layerData.layers[selectedLayer].statistics.num_nodes}</p>
                <p>边数: {layerData.layers[selectedLayer].statistics.num_edges}</p>
                <p>平均激活度: {layerData.layers[selectedLayer].statistics.avg_activation.toFixed(3)}</p>
                <p>最大激活度: {layerData.layers[selectedLayer].statistics.max_activation.toFixed(3)}</p>
              </div>
            </div>
          )}

          {selectedNode && (
            <div className="selected-node-info">
              <h3>节点详情</h3>
              <p>ID: {selectedNode.id}</p>
              <p>位置: ({selectedNode.position.join(', ')})</p>
              <p>激活度: {selectedNode.activation.toFixed(3)}</p>
              <p>大小: {selectedNode.size.toFixed(1)}</p>
            </div>
          )}
        </div>
      </div>

      <div className="association-summary" style={{ marginTop: '15px' }}>
        {layerData && layerData.associations && (
          <div className="associations-table">
            <h3>层间关联性</h3>
            <table>
              <thead>
                <tr>
                  <th>源层</th>
                  <th>目标层</th>
                  <th>关联强度</th>
                  <th>共享特征</th>
                </tr>
              </thead>
              <tbody>
                {layerData.associations.map((assoc, index) => (
                  <tr key={index}>
                    <td>{layerData.layers[assoc.source_layer].layer_name}</td>
                    <td>{layerData.layers[assoc.target_layer].layer_name}</td>
                    <td>{assoc.strength.toFixed(3)}</td>
                    <td>{assoc.shared_features.length}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default MultiLayer3DVisualization;
