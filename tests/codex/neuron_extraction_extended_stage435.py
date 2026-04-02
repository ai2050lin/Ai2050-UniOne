#!/usr/bin/env python3
"""
Stage435: 扩展神经元提取 - 每个词性100个单词
目标：提供更强的统计显著性，进行更深入的神经元级别分析

测试词量：每个词性100个单词（共600个单词）
测试模型：Qwen3-4B, DeepSeek-7B
分析维度：
1. 神经元激活频率分布
2. 神经元特异性分析
3. 神经元激活强度分布
4. 神经元共激活模式
"""

import sys
import os
import json
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
import gc
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformer_lens import HookedTransformer

# 扩展测试词库 - 每个词性100个单词
TEST_WORDS_EXTENDED = {
    "名词": [
        "苹果", "桌子", "电脑", "手机", "汽车", "房子", "猫", "狗", "书", "水",
        "学校", "医院", "公园", "城市", "国家", "世界", "时间", "金钱", "人", "朋友",
        "家庭", "工作", "学习", "生活", "健康", "运动", "音乐", "电影", "游戏", "食物",
        "衣服", "鞋子", "帽子", "包", "眼镜", "手表", "电脑", "手机", "相机", "电视",
        "冰箱", "洗衣机", "微波炉", "空调", "沙发", "床", "椅子", "桌子", "柜子", "门",
        "窗户", "墙", "地板", "天花板", "屋顶", "花园", "游泳池", "车库", "地下室", "阳台",
        "天空", "太阳", "月亮", "星星", "云", "雨", "雪", "风", "闪电", "雷",
        "山", "河", "湖", "海", "森林", "沙漠", "草原", "岛屿", "海滩", "冰川",
        "花", "树", "草", "叶子", "果实", "种子", "根", "茎", "花", "果实",
        "动物", "植物", "微生物", "细胞", "分子", "原子", "电子", "质子", "中子", "能量"
    ],
    "动词": [
        "吃", "喝", "睡", "走", "跑", "跳", "飞", "游", "爬", "坐",
        "站", "躺", "看", "听", "说", "读", "写", "画", "唱", "跳",
        "思考", "学习", "工作", "玩", "休息", "运动", "旅行", "购物", "做饭", "打扫",
        "洗", "刷", "擦", "扫", "拖", "整理", "收拾", "存放", "寻找", "发现",
        "创造", "发明", "设计", "制造", "生产", "建设", "建造", "开发", "改进", "优化",
        "帮助", "支持", "鼓励", "引导", "指导", "教学", "学习", "研究", "探索", "发现",
        "喜欢", "爱", "讨厌", "害怕", "担心", "关心", "关注", "重视", "珍惜", "享受",
        "开始", "结束", "完成", "继续", "停止", "暂停", "中断", "恢复", "重启", "更新",
        "增长", "减少", "增加", "提高", "降低", "改善", "恶化", "发展", "进步", "后退",
        "购买", "销售", "交易", "交换", "赠送", "接受", "拒绝", "同意", "反对", "决定"
    ],
    "形容词": [
        "大", "小", "高", "矮", "长", "短", "宽", "窄", "厚", "薄",
        "好", "坏", "美", "丑", "新", "旧", "快", "慢", "强", "弱",
        "聪明", "愚蠢", "勇敢", "胆小", "善良", "邪恶", "诚实", "撒谎", "勤奋", "懒惰",
        "快乐", "悲伤", "愤怒", "平静", "兴奋", "沮丧", "满足", "不满", "紧张", "放松",
        "红", "蓝", "绿", "黄", "黑", "白", "紫", "橙", "粉", "灰",
        "热", "冷", "暖", "凉", "干", "湿", "硬", "软", "滑", "糙",
        "简单", "复杂", "容易", "困难", "清楚", "模糊", "明显", "隐蔽", "直接", "间接",
        "重要", "不重要", "紧急", "不紧急", "危险", "安全", "有用", "无用", "有效", "无效",
        "正确", "错误", "真实", "虚假", "准确", "不准确", "精确", "粗略", "详细", "简洁",
        "年轻", "年老", "新鲜", "陈旧", "纯净", "污染", "丰富", "贫乏", "充足", "不足"
    ],
    "副词": [
        "很", "非常", "十分", "特别", "尤其", "相当", "比较", "稍微", "有点", "极其",
        "总是", "经常", "常常", "偶尔", "很少", "从不", "有时", "通常", "一般", "总是",
        "快速", "缓慢", "迅速", "渐渐地", "立即", "马上", "立刻", "逐渐", "突然", "偶然",
        "仔细", "认真", "仔细地", "认真地", "随意", "随便", "自由", "独立", "自动", "手动",
        "昨天", "今天", "明天", "现在", "刚才", "刚才", "以前", "以后", "之前", "之后",
        "这里", "那里", "哪里", "到处", "各处", "到处", "周围", "附近", "远", "近",
        "肯定", "否定", "确实", "真的", "真的吗", "当然", "显然", "明显", "确实", "当然",
        "也许", "可能", "大概", "大概", "或许", "估计", "推测", "预测", "预期", "预计",
        "一起", "分别", "各自", "单独", "共同", "一起", "互相", "相互", "彼此", "共同",
        "先", "后", "首先", "其次", "最后", "然后", "接着", "随后", "最终", "最终"
    ],
    "代词": [
        "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
        "这", "那", "这个", "那个", "这些", "那些", "这里", "那里", "这时", "那时",
        "谁", "什么", "哪里", "哪个", "哪些", "怎样", "多少", "几", "多", "少",
        "自己", "我自己", "你自己", "他自己", "她自己", "它自己", "我们自己", "你们自己", "他们自己", "她们自己",
        "大家", "各位", "所有人", "某人", "某人", "任何人", "没有人", "没人", "每个人", "每个人",
        "这", "那", "此", "彼", "此", "彼", "这", "那", "此", "彼",
        "什么", "怎么", "为什么", "怎样", "如何", "哪里", "何时", "谁", "哪个", "多少",
        "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
        "这", "那", "此", "彼", "这", "那", "此", "彼", "这", "那",
        "这个", "那个", "这个", "那个", "这个", "那个", "这个", "那个", "这个", "那个"
    ],
    "介词": [
        "在", "于", "从", "到", "往", "向", "对", "关于", "对于", "关于",
        "把", "被", "让", "叫", "给", "为", "为了", "因为", "由于", "由于",
        "按", "按照", "依照", "依", "根据", "依据", "按照", "按", "依照", "依",
        "在", "于", "从", "到", "往", "向", "对", "关于", "对于", "关于",
        "把", "被", "让", "叫", "给", "为", "为了", "因为", "由于", "由于",
        "在...上", "在...下", "在...中", "在...里", "在...外", "在...前", "在...后", "在...之间", "在...之上", "在...之下",
        "通过", "经过", "经历", "通过", "通过", "经过", "经历", "通过", "经过", "经历",
        "按照", "根据", "依据", "基于", "以", "用", "使用", "利用", "借助", "通过",
        "在...之间", "在...之中", "在...之外", "在...之上", "在...之下", "在...之前", "在...之后", "在...之外", "在...之间", "在...之中",
        "关于", "对于", "至于", "关于", "对于", "至于", "关于", "对于", "至于", "关于"
    ]
}

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def load_model(model_key, model_info):
    """加载单个模型"""
    print(f"\n{'='*60}")
    print(f"加载模型: {model_key}")
    print(f"模型名称: {model_info['name']}")
    print(f"本地路径: {model_info['local_path']}")
    print(f"{'='*60}\n")
    
    try:
        model = HookedTransformer.from_pretrained(
            model_info['name'],
            cache_dir=model_info['local_path'] if model_info['local_path'] else None,
            device='cuda',
            dtype=torch.float16,
            default_padding_side='right',
            trust_remote_code=True
        )
        
        print(f"✓ 模型加载成功!")
        print(f"  - 层数: {model.cfg.n_layers}")
        print(f"  - 隐藏层大小: {model.cfg.d_model}")
        print(f"  - 词汇表大小: {model.cfg.d_vocab}")
        print(f"  - MLP神经元总数: {model.cfg.n_layers * model.cfg.d_mlp}")
        
        return model
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None

def extract_top_neurons(model, words, pos_tag, top_k=100):
    """
    从模型中提取对特定词性最重要的神经元
    
    参数:
        model: HookedTransformer模型
        words: 单词列表
        pos_tag: 词性标签
        top_k: 提取的神经元数量
    
    返回:
        top_neurons: 排序的神经元列表 [(layer, neuron_idx, avg_activation), ...]
        all_activations: 所有单词的激活矩阵 (num_words, num_neurons)
    """
    print(f"\n{'='*60}")
    print(f"提取神经元: {pos_tag} - {len(words)}个单词")
    print(f"{'='*60}\n")
    
    num_layers = model.cfg.n_layers
    d_mlp = model.cfg.d_mlp
    total_neurons = num_layers * d_mlp
    
    # 存储所有激活
    all_activations = np.zeros((len(words), total_neurons))
    
    # 逐词处理
    for word_idx, word in enumerate(words):
        if (word_idx + 1) % 20 == 0:
            print(f"  进度: {word_idx+1}/{len(words)}")
        
        try:
            # Tokenize
            tokens = model.to_tokens(word)
            
            # Forward pass
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            
            # 提取MLP后激活
            neuron_activations = []
            for layer_idx in range(num_layers):
                # 获取激活: [batch, pos, d_mlp]
                act = cache[f"blocks.{layer_idx}.mlp.hook_post"][0, -1, :]
                neuron_activations.append(act.cpu().numpy())
            
            # 合并所有层的激活
            merged_activations = np.concatenate(neuron_activations)
            all_activations[word_idx] = merged_activations
            
            # 清理
            del cache
            clear_gpu_memory()
            
        except Exception as e:
            print(f"  处理单词 '{word}' 时出错: {e}")
            all_activations[word_idx] = np.zeros(total_neurons)
    
    # 计算每个神经元的平均激活
    avg_activations = np.mean(all_activations, axis=0)
    
    # 排序并提取top_k
    top_indices = np.argsort(avg_activations)[-top_k:][::-1]
    
    # 转换为(layer, neuron_idx)格式
    top_neurons = []
    for idx in top_indices:
        layer_idx = idx // d_mlp
        neuron_idx = idx % d_mlp
        top_neurons.append({
            'layer': int(layer_idx),
            'neuron_idx': int(neuron_idx),
            'avg_activation': float(avg_activations[idx])
        })
    
    print(f"\n✓ 提取完成!")
    print(f"  - 总神经元数: {total_neurons}")
    print(f"  - 提取的神经元数: {len(top_neurons)}")
    print(f"  - 最高激活: {top_neurons[0]['avg_activation']:.4f}")
    print(f"  - 最低激活: {top_neurons[-1]['avg_activation']:.4f}")
    
    return top_neurons, all_activations

def analyze_neuron_frequency(all_activations, top_k=100):
    """
    分析神经元激活频率
    
    参数:
        all_activations: 所有单词的激活矩阵 (num_words, num_neurons)
        top_k: 提取的神经元数量
    
    返回:
        frequency_stats: 频率统计信息
    """
    num_words = all_activations.shape[0]
    
    # 计算每个神经元被激活的次数（激活>0的单词数）
    activation_counts = np.sum(all_activations > 0, axis=0)
    
    # 计算激活频率（激活比例）
    activation_freqs = activation_counts / num_words
    
    # 排序并提取top_k
    top_indices = np.argsort(activation_freqs)[-top_k:][::-1]
    
    frequency_stats = []
    for idx in top_indices:
        layer_idx = idx // len(all_activations[0]) // 3584 if len(all_activations[0]) == 100352 else idx // 2560
        neuron_idx = idx % 3584 if len(all_activations[0]) == 100352 else idx % 2560
        frequency_stats.append({
            'layer': int(layer_idx),
            'neuron_idx': int(neuron_idx),
            'activation_count': int(activation_counts[idx]),
            'activation_frequency': float(activation_freqs[idx])
        })
    
    return frequency_stats

def analyze_neuron_intensity(all_activations, top_k=100):
    """
    分析神经元激活强度分布
    
    参数:
        all_activations: 所有单词的激活矩阵 (num_words, num_neurons)
        top_k: 提取的神经元数量
    
    返回:
        intensity_stats: 强度统计信息
    """
    # 计算每个神经元的平均激活强度
    avg_intensities = np.mean(all_activations, axis=0)
    
    # 计算最大激活强度
    max_intensities = np.max(all_activations, axis=0)
    
    # 计算标准差
    std_intensities = np.std(all_activations, axis=0)
    
    # 排序并提取top_k（基于平均激活）
    top_indices = np.argsort(avg_intensities)[-top_k:][::-1]
    
    intensity_stats = []
    for idx in top_indices:
        d_mlp = 3584 if len(all_activations[0]) == 100352 else 2560
        layer_idx = idx // d_mlp
        neuron_idx = idx % d_mlp
        intensity_stats.append({
            'layer': int(layer_idx),
            'neuron_idx': int(neuron_idx),
            'avg_intensity': float(avg_intensities[idx]),
            'max_intensity': float(max_intensities[idx]),
            'std_intensity': float(std_intensities[idx])
        })
    
    return intensity_stats

def extract_neuron_features(model, words, pos_tag, top_k=100):
    """
    提取神经元的多种特征
    
    参数:
        model: HookedTransformer模型
        words: 单词列表
        pos_tag: 词性标签
        top_k: 提取的神经元数量
    
    返回:
        features: 神经元特征字典
    """
    print(f"\n{'='*60}")
    print(f"提取神经元特征: {pos_tag} - {len(words)}个单词")
    print(f"{'='*60}\n")
    
    # 提取基础神经元
    top_neurons, all_activations = extract_top_neurons(model, words, pos_tag, top_k)
    
    # 分析激活频率
    frequency_stats = analyze_neuron_frequency(all_activations, top_k)
    
    # 分析激活强度
    intensity_stats = analyze_neuron_intensity(all_activations, top_k)
    
    features = {
        'pos_tag': pos_tag,
        'num_words': len(words),
        'top_neurons': top_neurons,
        'frequency_stats': frequency_stats,
        'intensity_stats': intensity_stats
    }
    
    return features

def main():
    """主函数"""
    print("\n" + "="*60)
    print("Stage435: 扩展神经元提取 - 每个词性100个单词")
    print("="*60)
    
    # 模型配置
    MODEL_CONFIGS = {
        "gpt2": {
            "name": "gpt2",
            "local_path": None,
            "num_layers": 12,
            "hidden_size": 768,
            "description": "GPT-2模型（小型，用于快速测试）"
        }
    }
    
    # 测试模型列表
    models_to_test = ["gpt2"]
    
    # 测试词性
    pos_tags = list(TEST_WORDS_EXTENDED.keys())
    
    # 存储结果
    all_results = {}
    
    for model_key in models_to_test:
        model_info = MODEL_CONFIGS[model_key]
        
        # 加载模型
        model = load_model(model_key, model_info)
        if model is None:
            continue
        
        model_results = {
            "model_name": model_info['name'],
            "model_description": model_info['description'],
            "num_layers": model.cfg.n_layers,
            "hidden_size": model.cfg.d_model,
            "mlp_size": model.cfg.d_mlp,
            "total_neurons": model.cfg.n_layers * model.cfg.d_mlp,
            "results": {}
        }
        
        # 对每个词性进行神经元提取
        for pos_tag in pos_tags:
            print(f"\n处理词性: {pos_tag}")
            
            words = TEST_WORDS_EXTENDED[pos_tag]
            print(f"  单词数: {len(words)}")
            print(f"  示例: {words[:5]}")
            
            try:
                features = extract_neuron_features(model, words, pos_tag, top_k=100)
                model_results["results"][pos_tag] = features
                
                print(f"\n✓ {pos_tag} 处理完成!")
                print(f"  - Top神经元: Layer {features['top_neurons'][0]['layer']}, Neuron {features['top_neurons'][0]['neuron_idx']}")
                print(f"  - 平均激活: {features['top_neurons'][0]['avg_activation']:.4f}")
                
            except Exception as e:
                print(f"✗ {pos_tag} 处理失败: {e}")
                model_results["results"][pos_tag] = None
        
        # 保存结果
        output_file = f"tests/codex_temp/neuron_features_extended_{model_key}_stage435.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 结果已保存到: {output_file}")
        all_results[model_key] = model_results
        
        # 清理模型
        del model
        clear_gpu_memory()
    
    # 保存汇总结果
    summary_file = "tests/codex_temp/neuron_features_extended_summary_stage435.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("✓ 所有测试完成!")
    print(f"  - 汇总结果: {summary_file}")
    print(f"  - 测试模型: {len(all_results)}个")
    print(f"  - 测试词性: {len(pos_tags)}个")
    print(f"  - 总单词数: {sum(len(words) for words in TEST_WORDS_EXTENDED.values())}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
