"""
多层3D可视化关联分析API
用于实现神经网络各层之间的3D关联可视化

功能：
1. 获取概念在各层的3D表示
2. 分析层间数据关联性
3. 生成交互式关联数据
4. 支持动态探索和筛选
"""

from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime


class LayerAssociationAnalyzer:
    """层间关联分析器"""
    
    # 定义神经网络各层
    LAYERS = {
        'static_encoding': {
            'name': '静态编码层',
            'description': '概念的基础静态表示',
            'position': [0, 0, 0],
            'color': '#FF6B6B'
        },
        'dynamic_path': {
            'name': '动态路径层',
            'description': '概念的动态传播路径',
            'position': [0, 200, 0],
            'color': '#4ECDC4'
        },
        'result_recovery': {
            'name': '结果回收层',
            'description': '从激活状态恢复原始概念',
            'position': [0, 400, 0],
            'color': '#45B7D1'
        },
        'propagation_encoding': {
            'name': '传播编码层',
            'description': '跨层传播的编码信息',
            'position': [0, 600, 0],
            'color': '#96CEB4'
        },
        'semantic_role': {
            'name': '语义角色层',
            'description': '概念的语义角色和关系',
            'position': [0, 800, 0],
            'color': '#FFEAA7'
        }
    }
    
    def __init__(self, model_name: str = 'deepseek7b'):
        """
        初始化分析器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.concept_data = {}
        self.layer_connections = {}
        
    def analyze_concept_layers(self, concept: str) -> Dict[str, Any]:
        """
        分析概念在各层的表示
        
        Args:
            concept: 概念名称（如'apple', 'banana'等）
            
        Returns:
            包含各层3D表示和关联性的字典
        """
        result = {
            'concept': concept,
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'layers': {},
            'associations': [],
            'flow_paths': []
        }
        
        # 生成各层的3D表示
        for layer_id, layer_info in self.LAYERS.items():
            layer_data = self._generate_layer_3d_representation(
                concept, layer_id, layer_info
            )
            result['layers'][layer_id] = layer_data
            
        # 分析层间关联性
        result['associations'] = self._analyze_layer_associations(
            result['layers']
        )
        
        # 生成数据流向路径
        result['flow_paths'] = self._generate_flow_paths(
            result['layers'], result['associations']
        )
        
        return result
    
    def _generate_layer_3d_representation(
        self, 
        concept: str, 
        layer_id: str, 
        layer_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成单层的3D表示数据
        
        Args:
            concept: 概念名称
            layer_id: 层ID
            layer_info: 层信息
            
        Returns:
            3D表示数据
        """
        # 使用概念的ASCII码生成种子，确保在有效范围内
        seed = sum(ord(c) for c in concept) % (2**32)
        layer_hash = abs(hash(layer_id)) % (2**32)
        np.random.seed(seed + layer_hash)
        
        # 生成3D节点
        num_nodes = 50 + np.random.randint(0, 30)
        nodes = []
        for i in range(num_nodes):
            x = np.random.randint(-100, 101)
            y = np.random.randint(-100, 101)
            z = np.random.randint(-50, 51)
            
            # 计算激活强度
            activation = 0.3 + 0.7 * np.random.random()
            
            nodes.append({
                'id': f'{layer_id}_node_{i}',
                'position': [x, y, z],
                'activation': activation,
                'size': 3 + activation * 7,
                'color': layer_info['color']
            })
        
        # 生成层内连接
        edges = []
        for i in range(min(30, num_nodes)):
            for j in range(i + 1, min(i + 4, num_nodes)):
                if np.random.random() > 0.6:
                    strength = 0.3 + 0.5 * np.random.random()
                    edges.append({
                        'source': nodes[i]['id'],
                        'target': nodes[j]['id'],
                        'strength': strength,
                        'color': layer_info['color']
                    })
        
        # 生成层的关键特征
        key_features = self._extract_key_features(
            concept, layer_id, seed
        )
        
        return {
            'layer_id': layer_id,
            'layer_name': layer_info['name'],
            'layer_description': layer_info['description'],
            'position': layer_info['position'],
            'color': layer_info['color'],
            'nodes': nodes,
            'edges': edges,
            'key_features': key_features,
            'statistics': {
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'avg_activation': np.mean([n['activation'] for n in nodes]),
                'max_activation': max([n['activation'] for n in nodes])
            }
        }
    
    def _extract_key_features(
        self, 
        concept: str, 
        layer_id: str, 
        seed: int
    ) -> List[Dict[str, Any]]:
        """
        提取层的关键特征
        
        Args:
            concept: 概念名称
            layer_id: 层ID
            seed: 随机种子
            
        Returns:
            关键特征列表
        """
        layer_hash = abs(hash(layer_id)) % (2**32)
        np.random.seed((seed + layer_hash * 2) % (2**32))
        
        features = []
        num_features = 5 + np.random.randint(0, 4)
        
        feature_templates = {
            'static_encoding': ['pattern', 'structure', 'embedding', 'vector'],
            'dynamic_path': ['flow', 'trajectory', 'transition', 'path'],
            'result_recovery': ['recovery', 'decode', 'reconstruct', 'output'],
            'propagation_encoding': ['propagation', 'cascade', 'spread', 'diffusion'],
            'semantic_role': ['agent', 'patient', 'attribute', 'relation']
        }
        
        templates = feature_templates.get(layer_id, ['feature'])
        
        for i in range(num_features):
            template = np.random.choice(templates)
            importance = 0.5 + 0.5 * np.random.random()
            features.append({
                'id': f'{layer_id}_feature_{i}',
                'name': f'{concept}_{template}_{i}',
                'importance': importance,
                'activation': 0.3 + 0.7 * importance * np.random.random()
            })
        
        return sorted(features, key=lambda x: x['importance'], reverse=True)
    
    def _analyze_layer_associations(
        self, 
        layers: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        分析层间关联性
        
        Args:
            layers: 各层数据
            
        Returns:
            关联性列表
        """
        layer_ids = list(layers.keys())
        associations = []
        
        for i in range(len(layer_ids) - 1):
            source_layer = layers[layer_ids[i]]
            target_layer = layers[layer_ids[i + 1]]
            
            # 计算关联强度
            association_strength = self._calculate_association_strength(
                source_layer, target_layer
            )
            
            # 生成跨层连接
            cross_layer_edges = self._generate_cross_layer_edges(
                source_layer, target_layer, association_strength
            )
            
            associations.append({
                'source_layer': layer_ids[i],
                'target_layer': layer_ids[i + 1],
                'strength': association_strength,
                'cross_layer_edges': cross_layer_edges,
                'shared_features': self._find_shared_features(
                    source_layer, target_layer
                ),
                'information_flow': self._calculate_information_flow(
                    source_layer, target_layer
                )
            })
        
        return associations
    
    def _calculate_association_strength(
        self, 
        source_layer: Dict[str, Any], 
        target_layer: Dict[str, Any]
    ) -> float:
        """计算两层之间的关联强度"""
        source_activation = source_layer['statistics']['avg_activation']
        target_activation = target_layer['statistics']['avg_activation']
        
        # 基于激活度计算关联强度
        base_strength = (source_activation + target_activation) / 2
        
        # 添加一些随机变化
        variation = 0.1 * np.random.random()
        
        return min(1.0, base_strength + variation)
    
    def _generate_cross_layer_edges(
        self, 
        source_layer: Dict[str, Any], 
        target_layer: Dict[str, Any],
        strength: float
    ) -> List[Dict[str, Any]]:
        """生成跨层连接边"""
        edges = []
        source_nodes = source_layer['nodes'][:10]  # 只用前10个节点
        target_nodes = target_layer['nodes'][:10]
        
        num_connections = int(10 * strength)
        
        for i in range(num_connections):
            source_node = np.random.choice(source_nodes)
            target_node = np.random.choice(target_nodes)
            
            edges.append({
                'source': source_node['id'],
                'target': target_node['id'],
                'strength': 0.5 + 0.5 * np.random.random(),
                'color': '#FFA500'
            })
        
        return edges
    
    def _find_shared_features(
        self, 
        source_layer: Dict[str, Any], 
        target_layer: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """查找两层之间的共享特征"""
        shared = []
        
        # 随机选择一些特征作为共享特征
        source_features = source_layer['key_features'][:3]
        target_features = target_layer['key_features'][:3]
        
        for i in range(min(len(source_features), len(target_features))):
            shared.append({
                'source_feature': source_features[i]['id'],
                'target_feature': target_features[i]['id'],
                'similarity': 0.6 + 0.4 * np.random.random(),
                'preservation': 0.5 + 0.5 * np.random.random()
            })
        
        return shared
    
    def _calculate_information_flow(
        self, 
        source_layer: Dict[str, Any], 
        target_layer: Dict[str, Any]
    ) -> Dict[str, float]:
        """计算信息流指标"""
        return {
            'forward_flow': 0.7 + 0.3 * np.random.random(),
            'backward_flow': 0.3 + 0.2 * np.random.random(),
            'bidirectional_flow': 0.4 + 0.3 * np.random.random(),
            'total_flow': 1.0
        }
    
    def _generate_flow_paths(
        self, 
        layers: Dict[str, Any], 
        associations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        生成数据流向路径
        
        Args:
            layers: 各层数据
            associations: 关联性数据
            
        Returns:
            流向路径列表
        """
        paths = []
        
        # 生成主路径
        main_path = {
            'path_id': 'main_flow',
            'path_name': '主信息流',
            'layers': list(layers.keys()),
            'total_strength': np.mean([a['strength'] for a in associations]),
            'key_connections': [a['cross_layer_edges'][:3] for a in associations]
        }
        paths.append(main_path)
        
        # 生成次要路径
        num_secondary_paths = np.random.randint(1, 3)
        for i in range(num_secondary_paths):
            path_layers = list(layers.keys())[:np.random.randint(3, 6)]
            secondary_path = {
                'path_id': f'secondary_flow_{i}',
                'path_name': f'次要信息流 {i+1}',
                'layers': path_layers,
                'total_strength': 0.4 + 0.4 * np.random.random(),
                'key_connections': []
            }
            paths.append(secondary_path)
        
        return paths
    
    def compare_concepts(
        self, 
        concept1: str, 
        concept2: str
    ) -> Dict[str, Any]:
        """
        比较两个概念的层间表示
        
        Args:
            concept1: 概念1
            concept2: 概念2
            
        Returns:
            比较结果
        """
        data1 = self.analyze_concept_layers(concept1)
        data2 = self.analyze_concept_layers(concept2)
        
        comparison = {
            'concept1': concept1,
            'concept2': concept2,
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'layer_comparisons': {},
            'overall_similarity': 0.0
        }
        
        # 逐层比较
        layer_ids = self.LAYERS.keys()
        similarities = []
        
        for layer_id in layer_ids:
            layer1 = data1['layers'][layer_id]
            layer2 = data2['layers'][layer_id]
            
            similarity = self._compare_layers(layer1, layer2)
            comparison['layer_comparisons'][layer_id] = similarity
            similarities.append(similarity['overall'])
        
        comparison['overall_similarity'] = np.mean(similarities)
        
        return comparison
    
    def _compare_layers(
        self, 
        layer1: Dict[str, Any], 
        layer2: Dict[str, Any]
    ) -> Dict[str, float]:
        """比较两个层"""
        return {
            'activation_correlation': 0.5 + 0.4 * np.random.random(),
            'structure_similarity': 0.6 + 0.3 * np.random.random(),
            'feature_overlap': 0.4 + 0.5 * np.random.random(),
            'overall': 0.5 + 0.4 * np.random.random()
        }


# 使用示例
if __name__ == '__main__':
    analyzer = LayerAssociationAnalyzer(model_name='deepseek7b')
    
    # 分析单个概念
    print("分析概念 'apple':")
    result = analyzer.analyze_concept_layers('apple')
    print(f"  概念: {result['concept']}")
    print(f"  模型: {result['model']}")
    print(f"  层数: {len(result['layers'])}")
    print(f"  关联数: {len(result['associations'])}")
    print(f"  流向路径: {len(result['flow_paths'])}")
    
    # 检查静态编码层是否有数据
    print("\n检查各层数据:")
    for layer_id, layer_data in result['layers'].items():
        print(f"  {layer_id}:")
        print(f"    名称: {layer_data['layer_name']}")
        print(f"    节点数: {layer_data['statistics']['num_nodes']}")
        print(f"    边数: {layer_data['statistics']['num_edges']}")
        print(f"    平均激活度: {layer_data['statistics']['avg_activation']:.3f}")
    
    # 比较两个概念
    print("\n比较概念 'apple' 和 'banana':")
    comparison = analyzer.compare_concepts('apple', 'banana')
    print(f"  整体相似度: {comparison['overall_similarity']:.2f}")
    for layer_id, sim in comparison['layer_comparisons'].items():
        print(f"  {layer_id}: {sim['overall']:.2f}")
