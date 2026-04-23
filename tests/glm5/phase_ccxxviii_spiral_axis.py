"""
Phase CCXXVIII: 螺旋轴分析与3D流形结构
================================================================
核心目标:
1. 找到PC1螺旋的"轴" — PC1围绕什么方向旋转?
2. 分析PC2/PC3是否也形成螺旋 — 多股螺旋结构?
3. 分析残差连接的旋转/缩放分解 — 螺旋的生成机制
4. 螺旋半径随层的变化 — 螺旋是均匀的还是变化的?
5. 跨特征螺旋一致性 — 不同特征的PC1是否围绕同一轴旋转?

关键假设:
  CCXXVII发现PC1以~30°/层恒定角速度旋转(总旋转1.2-1.8圈)。
  如果旋转是围绕某个固定轴的, 则1D流形是标准螺旋(helix);
  如果旋转轴也在变化, 则1D流形是更复杂的空间曲线。

样本量: 150对/特征, 40个测试句子
"""

import argparse
import json
import os
import time
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

import torch

# 特征对定义 (150对/特征)
FEATURE_PAIRS = {
    'tense': {
        'type': 'SYN',
        'pairs': [
            ("The cat sleeps on the mat", "The cat slept on the mat"),
            ("She walks to school every day", "She walked to school every day"),
            ("He runs in the park", "He ran in the park"),
            ("They play football on Sundays", "They played football on Sundays"),
            ("The dog barks at strangers", "The dog barked at strangers"),
            ("We cook dinner together", "We cooked dinner together"),
            ("She sings beautifully", "She sang beautifully"),
            ("The baby cries loudly", "The baby cried loudly"),
            ("Birds fly south in winter", "Birds flew south in winter"),
            ("The teacher explains the lesson", "The teacher explained the lesson"),
            ("The wind blows gently", "The wind blew gently"),
            ("Children laugh and play", "Children laughed and played"),
            ("The river flows to the sea", "The river flowed to the sea"),
            ("She paints beautiful pictures", "She painted beautiful pictures"),
            ("The sun rises in the east", "The sun rose in the east"),
            ("He writes a letter home", "He wrote a letter home"),
            ("The bird builds a nest", "The bird built a nest"),
            ("She drives to work early", "She drove to work early"),
            ("The rain falls softly", "The rain fell softly"),
            ("He reads the newspaper", "He read the newspaper"),
            ("The bell rings loudly", "The bell rang loudly"),
            ("They dance at the party", "They danced at the party"),
            ("The train arrives on time", "The train arrived on time"),
            ("She wears a red dress", "She wore a red dress"),
            ("The garden grows quickly", "The garden grew quickly"),
            ("He speaks three languages", "He spoke three languages"),
            ("The clock strikes midnight", "The clock struck midnight"),
            ("She catches the morning bus", "She caught the morning bus"),
            ("The snow melts in spring", "The snow melted in spring"),
            ("He breaks the record", "He broke the record"),
            ("The frog jumps into the pond", "The frog jumped into the pond"),
            ("She chooses the blue pen", "She chose the blue pen"),
            ("The candle burns brightly", "The candle burned brightly"),
            ("He swims across the lake", "He swam across the lake"),
            ("The phone rings unexpectedly", "The phone rang unexpectedly"),
            ("She forgets the password", "She forgot the password"),
            ("The boat sails across the bay", "The boat sailed across the bay"),
            ("He understands the problem", "He understood the problem"),
            ("The door opens slowly", "The door opened slowly"),
            ("She feels happy today", "She felt happy today"),
            ("The light shines through", "The light shone through"),
            ("He leaves the office early", "He left the office early"),
            ("The engine starts smoothly", "The engine started smoothly"),
            ("She spends time with friends", "She spent time with friends"),
            ("The cat chases the mouse", "The cat chased the mouse"),
            ("He brings the groceries home", "He brought the groceries home"),
            ("The flower blooms in spring", "The flower bloomed in spring"),
            ("I think about the future", "I thought about the future"),
            ("She teaches the students", "She taught the students"),
            ("The dog digs a hole", "The dog dug a hole"),
            ("He draws a picture", "He drew a picture"),
            ("The horse gallops fast", "The horse galloped fast"),
            ("She shakes her head", "She shook her head"),
            ("The fire burns hot", "The fire burned hot"),
            ("He fights for justice", "He fought for justice"),
            ("The snake slithers away", "The snake slithered away"),
            ("She throws the ball", "She threw the ball"),
            ("The car stops suddenly", "The car stopped suddenly"),
        ]
    },
    'polarity': {
        'type': 'SYN',
        'pairs': [
            ("The room was very bright", "The room was very dark"),
            ("She is extremely tall", "She is extremely short"),
            ("The movie was incredibly good", "The movie was incredibly bad"),
            ("He is always early", "He is always late"),
            ("The soup is very hot", "The soup is very cold"),
            ("She is very rich", "She is very poor"),
            ("The road was completely safe", "The road was completely dangerous"),
            ("He is deeply happy", "He is deeply sad"),
            ("The task was quite easy", "The task was quite hard"),
            ("She is very strong", "She is very weak"),
            ("The weather is beautifully warm", "The weather is bitterly cold"),
            ("He is incredibly fast", "He is incredibly slow"),
            ("The food was absolutely delicious", "The food was absolutely terrible"),
            ("She is extremely kind", "She is extremely cruel"),
            ("The building is very old", "The building is very new"),
            ("He is highly successful", "He is highly unsuccessful"),
            ("The story was really interesting", "The story was really boring"),
            ("She is very brave", "She is very cowardly"),
            ("The painting is very beautiful", "The painting is very ugly"),
            ("He is quite generous", "He is quite selfish"),
            ("The city is very clean", "The city is very dirty"),
            ("She is extremely polite", "She is extremely rude"),
            ("The material is very soft", "The material is very hard"),
            ("He is very careful", "He is very careless"),
            ("The cake was very sweet", "The cake was very bitter"),
            ("She is very quiet", "She is very loud"),
            ("The room is very large", "The room is very small"),
            ("He is very patient", "He is very impatient"),
            ("The water is very deep", "The water is very shallow"),
            ("She is very humble", "She is very proud"),
            ("The dog is very friendly", "The dog is very aggressive"),
            ("He is very honest", "He is very dishonest"),
            ("The dress is very light", "The dress is very heavy"),
            ("She is very calm", "She is very anxious"),
            ("The road is very wide", "The road is very narrow"),
            ("He is very healthy", "He is very sick"),
            ("The mountain is very high", "The mountain is very low"),
            ("She is very organized", "She is very messy"),
            ("The fabric is very thick", "The fabric is very thin"),
            ("He is very confident", "He is very timid"),
            ("The garden is very full", "The garden is very empty"),
            ("She is very gentle", "She is very harsh"),
            ("The surface is very smooth", "The surface is very rough"),
            ("He is very loyal", "He is very treacherous"),
            ("The light is very bright", "The light is very dim"),
            ("She is very wise", "She is very foolish"),
            ("The valley is very deep", "The valley is very shallow"),
            ("He is very optimistic", "He is very pessimistic"),
            ("The fabric is very tight", "The fabric is very loose"),
            ("She is very creative", "She is very conventional"),
            ("The river is very wide", "The river is very narrow"),
            ("He is very frugal", "He is very wasteful"),
            ("The star is very bright", "The star is very faint"),
            ("She is very graceful", "She is very clumsy"),
            ("The wood is very solid", "The wood is very hollow"),
            ("He is very forgiving", "He is very vengeful"),
            ("The sky is very clear", "The sky is very cloudy"),
            ("She is very active", "She is very passive"),
            ("The metal is very sharp", "The metal is very blunt"),
        ]
    },
    'voice': {
        'type': 'SYN',
        'pairs': [
            ("The cat chased the mouse", "The mouse was chased by the cat"),
            ("The chef cooked the meal", "The meal was cooked by the chef"),
            ("The teacher praised the student", "The student was praised by the teacher"),
            ("The company launched the product", "The product was launched by the company"),
            ("The artist painted the portrait", "The portrait was painted by the artist"),
            ("The scientist discovered the cure", "The cure was discovered by the scientist"),
            ("The author wrote the novel", "The novel was written by the author"),
            ("The builder constructed the house", "The house was constructed by the builder"),
            ("The musician composed the song", "The song was composed by the musician"),
            ("The director filmed the scene", "The scene was filmed by the director"),
            ("The mechanic repaired the engine", "The engine was repaired by the mechanic"),
            ("The doctor treated the patient", "The patient was treated by the doctor"),
            ("The judge sentenced the criminal", "The criminal was sentenced by the judge"),
            ("The editor published the article", "The article was published by the editor"),
            ("The designer created the logo", "The logo was created by the designer"),
            ("The baker made the bread", "The bread was made by the baker"),
            ("The farmer grew the crops", "The crops were grown by the farmer"),
            ("The police arrested the thief", "The thief was arrested by the police"),
            ("The team won the championship", "The championship was won by the team"),
            ("The committee approved the plan", "The plan was approved by the committee"),
            ("The government passed the law", "The law was passed by the government"),
            ("The school educated the children", "The children were educated by the school"),
            ("The nurse cared for the patient", "The patient was cared for by the nurse"),
            ("The manager hired the employee", "The employee was hired by the manager"),
            ("The pilot flew the airplane", "The airplane was flown by the pilot"),
            ("The chef prepared the dessert", "The dessert was prepared by the chef"),
            ("The writer translated the book", "The book was translated by the writer"),
            ("The coach trained the athlete", "The athlete was trained by the coach"),
            ("The engineer designed the bridge", "The bridge was designed by the engineer"),
            ("The researcher analyzed the data", "The data was analyzed by the researcher"),
        ]
    },
    'semantic_valence': {
        'type': 'SEM',
        'pairs': [
            ("She received a wonderful gift", "She received a terrible gift"),
            ("The news brought great joy", "The news brought great sorrow"),
            ("He achieved remarkable success", "He suffered devastating failure"),
            ("The garden looked absolutely stunning", "The garden looked absolutely dreadful"),
            ("She showed genuine kindness", "She showed genuine cruelty"),
            ("The performance was truly amazing", "The performance was truly awful"),
            ("He expressed deep gratitude", "He expressed deep resentment"),
            ("The experience was incredibly rewarding", "The experience was incredibly punishing"),
            ("She displayed exceptional courage", "She displayed exceptional cowardice"),
            ("The result was highly beneficial", "The result was highly harmful"),
            ("He felt overwhelming pride", "He felt overwhelming shame"),
            ("The outcome was completely positive", "The outcome was completely negative"),
            ("She demonstrated remarkable patience", "She demonstrated remarkable hostility"),
            ("The atmosphere was wonderfully peaceful", "The atmosphere was terribly hostile"),
            ("He showed extraordinary generosity", "He showed extraordinary greed"),
            ("The change brought immense relief", "The change brought immense distress"),
            ("She offered sincere appreciation", "She offered bitter criticism"),
            ("The event was tremendously exciting", "The event was tremendously depressing"),
            ("He experienced profound happiness", "He experienced profound misery"),
            ("The situation improved significantly", "The situation worsened significantly"),
            ("She found great comfort in friends", "She found great torment in enemies"),
            ("The discovery was extremely valuable", "The discovery was extremely worthless"),
            ("He gained tremendous respect", "He gained tremendous contempt"),
            ("The solution proved highly effective", "The solution proved highly useless"),
            ("She experienced pure delight", "She experienced pure anguish"),
            ("The journey was most pleasant", "The journey was most painful"),
            ("He received warm praise", "He received harsh condemnation"),
            ("The advice was incredibly helpful", "The advice was incredibly harmful"),
            ("She earned well-deserved recognition", "She earned well-deserved criticism"),
            ("The transformation was beautifully positive", "The transformation was horribly negative"),
            ("He shared genuine love", "He shared genuine hatred"),
            ("The creation was a magnificent triumph", "The creation was a miserable disaster"),
            ("She maintained strong hope", "She maintained strong despair"),
            ("The agreement brought lasting peace", "The agreement brought lasting conflict"),
            ("He provided essential support", "He provided severe opposition"),
            ("The gift was deeply appreciated", "The insult was deeply resented"),
            ("She showed boundless enthusiasm", "She showed boundless indifference"),
            ("The progress was remarkably swift", "The decline was remarkably steep"),
            ("He offered unconditional forgiveness", "He offered unconditional vengeance"),
            ("The memory brought gentle warmth", "The memory brought bitter coldness"),
        ]
    },
    'semantic_topic': {
        'type': 'SEM',
        'pairs': [
            ("The doctor treated the patient", "The chef treated the ingredients"),
            ("The lawyer argued the case", "The scientist argued the hypothesis"),
            ("The engineer built the bridge", "The artist built the sculpture"),
            ("The pilot flew the plane", "The sailor sailed the ship"),
            ("The teacher explained the theory", "The coach explained the strategy"),
            ("The farmer planted the seeds", "The programmer planted the ideas"),
            ("The soldier fought the battle", "The lawyer fought the case"),
            ("The nurse healed the wound", "The mechanic fixed the engine"),
            ("The judge delivered the verdict", "The professor delivered the lecture"),
            ("The banker managed the fund", "The conductor managed the orchestra"),
            ("The architect designed the building", "The poet designed the verse"),
            ("The detective investigated the crime", "The researcher investigated the phenomenon"),
            ("The surgeon performed the operation", "The musician performed the concert"),
            ("The driver navigated the road", "The captain navigated the ocean"),
            ("The programmer wrote the code", "The novelist wrote the story"),
            ("The electrician wired the house", "The therapist wired the brain"),
            ("The fisherman caught the fish", "The photographer caught the moment"),
            ("The librarian organized the books", "The general organized the troops"),
            ("The painter decorated the wall", "The composer decorated the silence"),
            ("The mechanic tuned the engine", "The pianist tuned the piano"),
            ("The baker kneaded the dough", "The philosopher kneaded the concepts"),
            ("The tailor stitched the fabric", "The historian stitched the narrative"),
            ("The chemist mixed the solution", "The DJ mixed the tracks"),
            ("The judge ruled the court", "The king ruled the kingdom"),
            ("The coach trained the team", "The mentor trained the mind"),
            ("The builder laid the foundation", "The educator laid the groundwork"),
            ("The plumber fixed the pipe", "The diplomat fixed the relations"),
            ("The editor revised the manuscript", "The surgeon revised the procedure"),
            ("The accountant balanced the books", "The acrobat balanced the body"),
            ("The priest blessed the congregation", "The critic blessed the performance"),
            ("The banker financed the startup", "The farmer financed the harvest"),
            ("The chef seasoned the soup", "The teacher seasoned the lecture"),
            ("The poet wrote the verse", "The lawyer wrote the contract"),
            ("The sailor navigated the ocean", "The driver navigated the traffic"),
            ("The biologist studied the cell", "The sociologist studied the group"),
            ("The musician composed the symphony", "The architect composed the skyline"),
            ("The novelist created the character", "The sculptor created the statue"),
            ("The philosopher questioned existence", "The scientist questioned the hypothesis"),
        ]
    },
}

feature_names = list(FEATURE_PAIRS.keys())

TEST_SENTENCES = [
    "The weather today is",
    "She looked at the",
    "He decided to",
    "The city was",
    "They found the",
    "After the storm the",
    "The scientist discovered",
    "She always wanted to",
    "The book describes",
    "He carefully opened the",
    "The river flows through",
    "She smiled at the",
    "The old man walked",
    "They built a new",
    "The music played softly",
    "He remembered the",
    "The garden was full of",
    "She wrote a long",
    "The children played in the",
    "He watched the",
    "The restaurant served",
    "She chose the",
    "The forest was dark and",
    "He finished the",
    "The road led to",
    "She noticed the",
    "The cat sat on the",
    "He finally understood the",
    "The flowers bloomed in the",
    "She carefully explained the",
    "The wind carried the",
    "He imagined a world where",
    "The painting showed a",
    "She waited for the",
    "The mountain stood tall",
    "He promised to",
    "The water reflected the",
    "She quietly closed the",
    "The stars shone above the",
    "He reached for the",
]

MODEL_CONFIGS = {
    'qwen3': {
        'name': 'Qwen3-4B',
        'path': 'D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',
        'n_layers': 36,
        'd_model': 2560,
        'dtype': 'bf16',
    },
    'deepseek7b': {
        'name': 'DeepSeek-R1-Distill-Qwen-7B',
        'path': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'n_layers': 28,
        'd_model': 3584,
        'dtype': '8bit',
    },
    'glm4': {
        'name': 'GLM4-9B-Chat',
        'path': 'D:/develop/model/hub/models--zai-org--glm-4-9b-chat-hf/download',
        'n_layers': 40,
        'd_model': 4096,
        'dtype': '8bit',
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=150)
    parser.add_argument('--n_test', type=int, default=40)
    args = parser.parse_args()

    model_key = args.model
    n_pairs = args.n_pairs
    n_test = args.n_test
    config = MODEL_CONFIGS[model_key]

    out_dir = f'results/causal_fiber/{model_key}_ccxxviii'
    os.makedirs(out_dir, exist_ok=True)
    log_path = f'{out_dir}/run.log'

    def log(msg):
        try:
            print(msg, flush=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
                f.flush()
        except Exception:
            pass

    log(f"=" * 70)
    log(f"Phase CCXXVIII: 螺旋轴分析与3D流形结构 — {config['name']}")
    log(f"  n_pairs={n_pairs}, n_test={n_test}, 时间={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sklearn.decomposition import PCA, TruncatedSVD

    # ============================================================
    # S1: 加载模型
    # ============================================================
    log(f"\n--- S1: 加载模型 ---")

    path = config['path']
    t0 = time.time()

    if model_key in ["glm4", "deepseek7b"]:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map='cpu',
            trust_remote_code=True, local_files_only=True
        )
        model = model.to('cuda')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_time = time.time() - t0
    n_layers = config['n_layers']
    d_model = config['d_model']
    last_layer = n_layers - 1

    log(f"  加载完成: {load_time:.0f}s, n_layers={n_layers}, d_model={d_model}")

    # 提取lm_head权重
    W_full = None
    try:
        lm_head = getattr(model, 'lm_head', None)
        if lm_head is None and hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
            lm_head = model.model.lm_head
        if lm_head is not None:
            W = lm_head.weight.detach().float().cpu().numpy()
            if W.shape[1] == d_model:
                W_full = W
            elif W.shape[0] == d_model:
                W_full = W.T
            else:
                W_full = W
    except Exception as e:
        log(f"  从model.lm_head提取失败: {e}")

    if W_full is None:
        import glob
        try:
            from safetensors.torch import load_file as st_load_file
            safetensor_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
            for sf in safetensor_files:
                try:
                    from safetensors import safe_open as _so
                    with _so(sf, framework="pt") as f:
                        keys = list(f.keys())
                    if not any('lm_head.weight' in k for k in keys):
                        continue
                    log(f"    加载 {os.path.basename(sf)}...")
                    state_dict = st_load_file(sf)
                    for key, tensor in state_dict.items():
                        if 'lm_head.weight' in key:
                            W_raw = tensor.float().cpu().numpy()
                            if W_raw.shape[1] == d_model:
                                W_full = W_raw
                            elif W_raw.shape[0] == d_model:
                                W_full = W_raw.T
                            break
                    if W_full is not None:
                        break
                except Exception:
                    continue
        except Exception:
            pass

    if W_full is not None:
        svd = TruncatedSVD(n_components=3)
        svd.fit(W_full)
        log(f"  lm_head: shape={W_full.shape}, SVD[0]方差={svd.explained_variance_ratio_[0]*100:.1f}%")

    # ============================================================
    # S2: 逐层收集差分向量 + hidden states → 每层PC1/PC2/PC3
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S2: 逐层PC1/PC2/PC3收集 (密集采样)")
    log(f"{'=' * 60}")

    sample_layers = list(range(0, n_layers, 2))
    if last_layer not in sample_layers:
        sample_layers.append(last_layer)
    sample_layers.sort()

    log(f"  采样层 ({len(sample_layers)}层): {sample_layers[:5]}...{sample_layers[-3:]}")

    all_layer_diffs = defaultdict(list)
    feat_layer_diffs = defaultdict(lambda: defaultdict(list))

    for feat in feature_names:
        pairs = FEATURE_PAIRS[feat]['pairs'][:n_pairs]
        log(f"  收集 {feat} ({len(pairs)}对)...")

        for i, (s_pos, s_neg) in enumerate(pairs):
            try:
                with torch.no_grad():
                    toks_pos = tokenizer(s_pos, return_tensors='pt').to(model.device)
                    out_pos = model(**toks_pos, output_hidden_states=True)

                    toks_neg = tokenizer(s_neg, return_tensors='pt').to(model.device)
                    out_neg = model(**toks_neg, output_hidden_states=True)

                    for L in sample_layers:
                        h_pos = out_pos.hidden_states[L][0, -1, :].float().cpu().numpy()
                        h_neg = out_neg.hidden_states[L][0, -1, :].float().cpu().numpy()
                        diff = h_pos - h_neg
                        all_layer_diffs[L].append(diff)
                        feat_layer_diffs[feat][L].append(diff)

                    del out_pos, out_neg

                if (i + 1) % 50 == 0:
                    log(f"    {feat}: {i+1}/{len(pairs)}")
            except Exception as e:
                continue

    # 计算每层PC1/PC2/PC3
    layer_pca_global = {}  # L -> (components, var_ratio, pca_object)
    layer_pc1_feat = {}

    for L in sample_layers:
        diffs = all_layer_diffs[L]
        if len(diffs) >= 20:
            diffs = np.array(diffs)
            pca = PCA(n_components=min(10, len(diffs), diffs.shape[1]))
            pca.fit(diffs)
            layer_pca_global[L] = (pca.components_[:10], pca.explained_variance_ratio_[:10], pca)

        for feat in feature_names:
            fds = feat_layer_diffs[feat][L]
            if len(fds) >= 10:
                fds = np.array(fds)
                pca_f = PCA(n_components=min(3, len(fds), fds.shape[1]))
                pca_f.fit(fds)
                layer_pc1_feat[(feat, L)] = (pca_f.components_[:3], pca_f.explained_variance_ratio_[:3])

    # 找到PC1方差峰值层
    peak_layer = 0
    peak_var = 0
    for L in sample_layers:
        if L in layer_pca_global:
            v = layer_pca_global[L][1][0]
            if v > peak_var:
                peak_var = v
                peak_layer = L
    log(f"\n  全局PC1方差峰值层: L{peak_layer} ({peak_var*100:.1f}%)")

    # ============================================================
    # S3: 螺旋轴分析 — PC1轨迹的SVD分解
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S3: 螺旋轴分析 — PC1轨迹的SVD分解")
    log(f"{'=' * 60}")

    # 将所有层的PC1堆叠成矩阵 (n_layers, d_model)
    pc1_dirs = {}
    pc2_dirs = {}
    pc3_dirs = {}
    for L in sample_layers:
        if L in layer_pca_global:
            pc1_dirs[L] = layer_pca_global[L][0][0]
            pc2_dirs[L] = layer_pca_global[L][0][1]
            pc3_dirs[L] = layer_pca_global[L][0][2]

    layers_sorted = sorted(pc1_dirs.keys())

    # S3a: PC1轨迹矩阵的SVD
    log(f"\n  --- S3a: PC1轨迹矩阵SVD ---")
    M_pc1 = np.array([pc1_dirs[L] for L in layers_sorted])  # (n_sample_layers, d_model)
    log(f"  PC1轨迹矩阵: shape={M_pc1.shape}")

    # SVD分解
    U_pc1, S_pc1, Vt_pc1 = np.linalg.svd(M_pc1, full_matrices=False)

    log(f"  前10个奇异值: {S_pc1[:10].tolist()}")
    log(f"  前3个奇异值占比: {S_pc1[:3]/S_pc1.sum()*100}")

    # 如果前2个奇异值占主导, 螺旋主要在2D平面内
    # 如果前3个奇异值占主导, 螺旋在3D空间内
    s_total = S_pc1.sum()
    s2_ratio = S_pc1[:2].sum() / s_total
    s3_ratio = S_pc1[:3].sum() / s_total
    log(f"  前2奇异值累计: {s2_ratio*100:.1f}%")
    log(f"  前3奇异值累计: {s3_ratio*100:.1f}%")

    if s2_ratio > 0.9:
        log(f"  → PC1轨迹几乎在2D平面内 → 标准螺旋(helix)")
    elif s3_ratio > 0.9:
        log(f"  → PC1轨迹在3D空间内 → 锥面螺旋(conical helix)")
    else:
        log(f"  → PC1轨迹在高维空间 → 复杂空间曲线")

    # S3b: 投影到螺旋平面 → 2D轨迹
    log(f"\n  --- S3b: PC1在螺旋平面上的2D投影 ---")
    V_plane = Vt_pc1[:2]  # (2, d_model) - 螺旋平面的基

    proj_2d = []
    for L in layers_sorted:
        p1 = np.dot(pc1_dirs[L], V_plane[0])
        p2 = np.dot(pc1_dirs[L], V_plane[1])
        proj_2d.append((L, p1, p2))

    log(f"  {'层':>4} | {'V1投影':>10} | {'V2投影':>10} | {'半径':>10} | {'角度°':>10}")
    log(f"  {'-'*4} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    for L, p1, p2 in proj_2d:
        r = np.sqrt(p1**2 + p2**2)
        angle = np.degrees(np.arctan2(p2, p1))
        log(f"  L{L:>3} | {p1:>10.4f} | {p2:>10.4f} | {r:>10.4f} | {angle:>10.1f}")

    # 检查半径变化
    radii = [np.sqrt(p1**2 + p2**2) for _, p1, p2 in proj_2d]
    angles = [np.degrees(np.arctan2(p2, p1)) for _, p1, p2 in proj_2d]
    log(f"\n  半径: 均值={np.mean(radii):.4f}, 标准差={np.std(radii):.4f}, CV={np.std(radii)/np.mean(radii)*100:.1f}%")

    # 角度展开 (处理wrap-around)
    unwrapped_angles = [angles[0]]
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i-1]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        unwrapped_angles.append(unwrapped_angles[-1] + diff)

    log(f"  展开角度: [{unwrapped_angles[0]:.1f}° → {unwrapped_angles[-1]:.1f}°]")
    total_angle = unwrapped_angles[-1] - unwrapped_angles[0]
    log(f"  总旋转角: {total_angle:.1f}° (={total_angle/360:.2f}圈)")

    # 线性度检验
    x_ang = np.arange(len(unwrapped_angles))
    y_ang = np.array(unwrapped_angles)
    if len(x_ang) > 2:
        coeffs_ang = np.polyfit(x_ang, y_ang, 1)
        y_pred_ang = np.polyval(coeffs_ang, x_ang)
        ss_res = np.sum((y_ang - y_pred_ang) ** 2)
        ss_tot = np.sum((y_ang - np.mean(y_ang)) ** 2)
        r_sq_ang = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        log(f"  角度线性度R²={r_sq_ang:.4f} (1.0=完美恒定角速度)")

    # S3c: PC1围绕的"轴" — 正交于螺旋平面的方向
    log(f"\n  --- S3c: 螺旋轴方向 ---")
    # 轴方向: 第3个右奇异向量 (如果前2个是螺旋平面)
    if len(Vt_pc1) >= 3:
        axis_dir = Vt_pc1[2]  # 第3个分量, 正交于螺旋平面
        log(f"  螺旋轴(V3): ||V3||={np.linalg.norm(axis_dir):.4f}")
        log(f"  V3的SVD奇异值: {S_pc1[2]:.4f} (vs V1={S_pc1[0]:.4f}, V2={S_pc1[1]:.4f})")

        # 轴方向的logit指纹
        if W_full is not None:
            logit_fingerprint = W_full @ axis_dir
            top_pos_idx = np.argsort(logit_fingerprint)[-10:][::-1]
            top_neg_idx = np.argsort(logit_fingerprint)[:10]
            log(f"  V3轴logit指纹 Top-5正: idx={top_pos_idx[:5].tolist()}, val={logit_fingerprint[top_pos_idx[:5]].round(3).tolist()}")
            log(f"  V3轴logit指纹 Top-5负: idx={top_neg_idx[:5].tolist()}, val={logit_fingerprint[top_neg_idx[:5]].round(3).tolist()}")

    # S3d: PC1在V1-V2-V3空间中的3D轨迹
    log(f"\n  --- S3d: 3D轨迹投影 ---")
    if len(Vt_pc1) >= 3:
        V_3d = Vt_pc1[:3]
        log(f"  {'层':>4} | {'V1':>8} | {'V2':>8} | {'V3':>8} | {'3D半径':>8}")
        log(f"  {'-'*4} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
        for L in layers_sorted:
            p1 = np.dot(pc1_dirs[L], V_3d[0])
            p2 = np.dot(pc1_dirs[L], V_3d[1])
            p3 = np.dot(pc1_dirs[L], V_3d[2])
            r3d = np.sqrt(p1**2 + p2**2 + p3**2)
            log(f"  L{L:>3} | {p1:>8.4f} | {p2:>8.4f} | {p3:>8.4f} | {r3d:>8.4f}")

    # ============================================================
    # S4: PC2/PC3螺旋分析
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S4: PC2/PC3螺旋分析")
    log(f"{'=' * 60}")

    # S4a: PC2轨迹SVD
    M_pc2 = np.array([pc2_dirs[L] for L in layers_sorted])
    U_pc2, S_pc2, Vt_pc2 = np.linalg.svd(M_pc2, full_matrices=False)

    log(f"\n  PC2轨迹矩阵: shape={M_pc2.shape}")
    log(f"  PC2前3奇异值: {S_pc2[:3].tolist()}")
    log(f"  PC2前2奇异值累计: {S_pc2[:2].sum()/S_pc2.sum()*100:.1f}%")

    # S4b: PC3轨迹SVD
    M_pc3 = np.array([pc3_dirs[L] for L in layers_sorted])
    U_pc3, S_pc3, Vt_pc3 = np.linalg.svd(M_pc3, full_matrices=False)

    log(f"\n  PC3轨迹矩阵: shape={M_pc3.shape}")
    log(f"  PC3前3奇异值: {S_pc3[:3].tolist()}")
    log(f"  PC3前2奇异值累计: {S_pc3[:2].sum()/S_pc3.sum()*100:.1f}%")

    # S4c: PC1/PC2/PC3螺旋是否耦合?
    log(f"\n  --- PC1/PC2/PC3螺旋耦合分析 ---")

    # 在每层, 计算PC2相对于PC1的旋转角
    # 在PCA空间中, PC1和PC2正交。但跨层时, PC2_L和PC1_{L+1}可能对齐
    for i in range(min(8, len(layers_sorted) - 1)):
        L1, L2 = layers_sorted[i], layers_sorted[i + 1]
        if L1 in layer_pca_global and L2 in layer_pca_global:
            pc1_L1 = layer_pca_global[L1][0][0]
            pc2_L1 = layer_pca_global[L1][0][1]
            pc1_L2 = layer_pca_global[L2][0][0]
            pc2_L2 = layer_pca_global[L2][0][1]

            cos_pc1_pc1 = abs(np.dot(pc1_L1, pc1_L2) / (np.linalg.norm(pc1_L1) * np.linalg.norm(pc1_L2)))
            cos_pc2_pc2 = abs(np.dot(pc2_L1, pc2_L2) / (np.linalg.norm(pc2_L1) * np.linalg.norm(pc2_L2)))
            cos_pc1_pc2 = abs(np.dot(pc1_L1, pc2_L2) / (np.linalg.norm(pc1_L1) * np.linalg.norm(pc2_L2)))
            cos_pc2_pc1 = abs(np.dot(pc2_L1, pc1_L2) / (np.linalg.norm(pc2_L1) * np.linalg.norm(pc1_L2)))

            log(f"  L{L1}→L{L2}: |cos(PC1,PC1)|={cos_pc1_pc1:.3f}, |cos(PC2,PC2)|={cos_pc2_pc2:.3f}, |cos(PC1→PC2)|={cos_pc1_pc2:.3f}, |cos(PC2→PC1)|={cos_pc2_pc1:.3f}")

    # S4d: 子空间分析 — 前3PC的旋转平面
    log(f"\n  --- 子空间旋转分析 ---")
    # 在每层, 前3PC构成一个3D子空间。检查这个子空间跨层的稳定性
    for i in range(min(5, len(layers_sorted) - 1)):
        L1, L2 = layers_sorted[i], layers_sorted[i + 1]
        if L1 in layer_pca_global and L2 in layer_pca_global:
            # 前3PC构成的子空间
            sub1 = layer_pca_global[L1][0][:3]  # (3, d_model)
            sub2 = layer_pca_global[L2][0][:3]  # (3, d_model)

            # 子空间包含度: sub2的每个向量在sub1上的投影比例
            # 用Procrustes分析
            S_sub = sub1 @ sub2.T  # (3, 3)
            U_sub, S_vals, Vt_sub = np.linalg.svd(S_sub)
            # S_vals是子空间之间的"主对齐值"
            log(f"  L{L1}→L{L2}: 子空间对齐SVD={S_vals.round(3).tolist()}")

    # ============================================================
    # S5: 残差分解 — 旋转 vs 缩放
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S5: 残差分解 — 旋转vs缩放")
    log(f"{'=' * 60}")
    log(f"  分析hidden state层间变化的旋转/缩放分量")

    # 用测试句子, 计算每层的h_L, 然后分析Δh = h_L - h_{L-1}的方向
    test_sentences = TEST_SENTENCES[:n_test]

    # 逐层分析
    layer_rotation_ratios = defaultdict(list)
    layer_delta_norms = defaultdict(list)
    layer_h_norms = defaultdict(list)

    for si, sent in enumerate(test_sentences):
        try:
            with torch.no_grad():
                toks = tokenizer(sent, return_tensors='pt').to(model.device)
                out = model(**toks, output_hidden_states=True)

                for L in range(1, min(n_layers, len(out.hidden_states))):
                    h_prev = out.hidden_states[L-1][0, -1, :].float().cpu().numpy()
                    h_curr = out.hidden_states[L][0, -1, :].float().cpu().numpy()

                    delta = h_curr - h_prev
                    h_norm = np.linalg.norm(h_prev)
                    d_norm = np.linalg.norm(delta)

                    if h_norm > 1e-6 and d_norm > 1e-6:
                        # 平行分量 (缩放)
                        cos_angle = np.dot(delta, h_prev) / (d_norm * h_norm)
                        # 旋转分量 = 垂直于h_prev的分量
                        parallel_frac = cos_angle  # 平行分量比例
                        perp_frac = np.sqrt(1 - cos_angle**2)  # 垂直分量比例

                        # 只记录每2层 (与sample_layers对应)
                        if L in sample_layers or L-1 in sample_layers:
                            layer_rotation_ratios[L].append(perp_frac)
                            layer_delta_norms[L].append(d_norm)
                            layer_h_norms[L].append(h_norm)

                del out
        except Exception as e:
            continue

        if (si + 1) % 10 == 0:
            log(f"  处理 {si+1}/{len(test_sentences)} 个句子")

    # 输出结果
    log(f"\n  --- 残差分解结果 ---")
    log(f"  {'层':>4} | {'旋转分量':>10} | {'||Δh||':>10} | {'||h||':>10} | {'||Δh||/||h||':>12}")
    log(f"  {'-'*4} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12}")

    rotation_trend = []
    for L in sorted(layer_rotation_ratios.keys()):
        if layer_rotation_ratios[L]:
            mean_rot = np.mean(layer_rotation_ratios[L])
            mean_delta = np.mean(layer_delta_norms[L])
            mean_h = np.mean(layer_h_norms[L])
            ratio = mean_delta / mean_h if mean_h > 0 else 0
            rotation_trend.append((L, mean_rot, mean_delta, mean_h, ratio))
            if L % 4 == 0 or L == last_layer:
                log(f"  L{L:>3} | {mean_rot:>10.4f} | {mean_delta:>10.4f} | {mean_h:>10.2f} | {ratio:>12.6f}")

    # 趋势分析
    if len(rotation_trend) >= 4:
        early_rot = np.mean([r[1] for r in rotation_trend[:len(rotation_trend)//3]])
        mid_rot = np.mean([r[1] for r in rotation_trend[len(rotation_trend)//3:2*len(rotation_trend)//3]])
        late_rot = np.mean([r[1] for r in rotation_trend[2*len(rotation_trend)//3:]])

        log(f"\n  旋转分量趋势:")
        log(f"    早期层: {early_rot:.4f}")
        log(f"    中间层: {mid_rot:.4f}")
        log(f"    晚期层: {late_rot:.4f}")

        if late_rot > early_rot * 1.1:
            log(f"  → 晚期层旋转更强 → 残差连接在晚期产生更多旋转")
        elif early_rot > late_rot * 1.1:
            log(f"  → 早期层旋转更强 → 残差连接在早期产生更多旋转")
        else:
            log(f"  → 旋转分量在各层相对均匀")

    # ============================================================
    # S6: 跨特征螺旋一致性
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S6: 跨特征螺旋一致性")
    log(f"{'=' * 60}")

    # 不同特征的PC1是否围绕同一轴旋转?
    # 对每个特征, 计算其PC1轨迹在全局螺旋平面上的投影
    if len(Vt_pc1) >= 2:
        V_plane_global = Vt_pc1[:2]

        log(f"\n  各特征PC1在全局螺旋平面上的投影:")
        log(f"  {'特征':>20} | {'2D累计占比':>10} | {'角度范围':>12} | {'半径CV':>8}")
        log(f"  {'-'*20} | {'-'*10} | {'-'*12} | {'-'*8}")

        for feat in feature_names:
            feat_pc1s = []
            feat_layers = []
            for L in layers_sorted:
                if (feat, L) in layer_pc1_feat:
                    feat_pc1s.append(layer_pc1_feat[(feat, L)][0][0])
                    feat_layers.append(L)

            if len(feat_pc1s) < 5:
                continue

            feat_pc1s = np.array(feat_pc1s)

            # 投影到全局螺旋平面
            proj1 = feat_pc1s @ V_plane_global[0]
            proj2 = feat_pc1s @ V_plane_global[1]

            # 2D投影的能量占比
            total_energy = np.sum(feat_pc1s**2)
            proj_energy = np.sum(proj1**2) + np.sum(proj2**2)
            energy_ratio = proj_energy / total_energy if total_energy > 0 else 0

            # 角度范围
            angles_feat = np.degrees(np.arctan2(proj2, proj1))
            # 展开
            unwrapped = [angles_feat[0]]
            for j in range(1, len(angles_feat)):
                diff = angles_feat[j] - angles_feat[j-1]
                while diff > 180: diff -= 360
                while diff < -180: diff += 360
                unwrapped.append(unwrapped[-1] + diff)
            angle_range = unwrapped[-1] - unwrapped[0]

            # 半径CV
            radii_feat = np.sqrt(proj1**2 + proj2**2)
            radii_cv = np.std(radii_feat) / np.mean(radii_feat) if np.mean(radii_feat) > 0 else 0

            log(f"  {feat:>20} | {energy_ratio*100:>9.1f}% | {angle_range:>11.1f}° | {radii_cv*100:>7.1f}%")

    # ============================================================
    # S7: 螺旋结构与SVD[0]的关系
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S7: 螺旋结构与SVD[0]的关系")
    log(f"{'=' * 60}")

    if W_full is not None:
        svd_W = TruncatedSVD(n_components=3)
        svd_W.fit(W_full)
        svd0_dir = svd_W.components_[0]
        svd1_dir = svd_W.components_[1]
        svd2_dir = svd_W.components_[2]

        log(f"  SVD[0]方差: {svd_W.explained_variance_ratio_[0]*100:.1f}%")
        log(f"  SVD[1]方差: {svd_W.explained_variance_ratio_[1]*100:.1f}%")

        # 螺旋轴(V3)与SVD[0]的对齐
        if len(Vt_pc1) >= 3:
            cos_v3_svd0 = abs(np.dot(Vt_pc1[2], svd0_dir))
            cos_v3_svd1 = abs(np.dot(Vt_pc1[2], svd1_dir))
            cos_v1_svd0 = abs(np.dot(Vt_pc1[0], svd0_dir))
            cos_v2_svd0 = abs(np.dot(Vt_pc1[1], svd0_dir))

            log(f"\n  螺旋平面与SVD[0]的对齐:")
            log(f"    cos(V1, SVD[0])={cos_v1_svd0:.4f}")
            log(f"    cos(V2, SVD[0])={cos_v2_svd0:.4f}")
            log(f"    cos(轴V3, SVD[0])={cos_v3_svd0:.4f}")
            log(f"    cos(轴V3, SVD[1])={cos_v3_svd1:.4f}")

            if cos_v3_svd0 > 0.5:
                log(f"  → 螺旋轴与SVD[0]对齐 → PC1围绕SVD[0]旋转!")
            elif cos_v1_svd0 > 0.5 or cos_v2_svd0 > 0.5:
                log(f"  → 螺旋平面包含SVD[0] → PC1在SVD[0]附近振荡")
            else:
                log(f"  → 螺旋结构与SVD[0]近似正交 → PC1螺旋独立于输出方向")

        # 每层PC1与SVD[0]的对齐轨迹
        log(f"\n  各层PC1与SVD[0]的对齐:")
        for L in layers_sorted:
            if L in pc1_dirs:
                cos_svd0 = abs(np.dot(pc1_dirs[L], svd0_dir))
                if L % 4 == 0 or L == last_layer:
                    log(f"    L{L}: |cos(PC1, SVD[0])|={cos_svd0:.4f}")

    # ============================================================
    # S8: 汇总
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S8: 汇总")
    log(f"{'=' * 60}")

    # 螺旋维度
    if s2_ratio > 0.9:
        spiral_type = "2D标准螺旋(helix)"
    elif s3_ratio > 0.9:
        spiral_type = "3D锥面螺旋(conical helix)"
    else:
        spiral_type = f"高维空间曲线(前3PC={s3_ratio*100:.0f}%)"

    log(f"\n  1. 螺旋类型: {spiral_type}")
    log(f"  2. 前2奇异值占比: {s2_ratio*100:.1f}%")
    log(f"  3. 前3奇异值占比: {s3_ratio*100:.1f}%")
    log(f"  4. 总旋转角: {total_angle:.1f}° ({total_angle/360:.2f}圈)")
    log(f"  5. 半径CV: {np.std(radii)/np.mean(radii)*100:.1f}% (越低越均匀)")

    if len(rotation_trend) >= 4:
        log(f"  6. 旋转分量: 早期={early_rot:.3f}, 中期={mid_rot:.3f}, 晚期={late_rot:.3f}")

    if W_full is not None and len(Vt_pc1) >= 3:
        log(f"  7. 螺旋轴-SVD[0]对齐: cos={cos_v3_svd0:.4f}")
        log(f"  8. 螺旋平面-SVD[0]对齐: max(cos(V1,V2),SVD[0])={max(cos_v1_svd0, cos_v2_svd0):.4f}")

    # 保存结果
    results = {
        'model': config['name'],
        'spiral_type': spiral_type,
        's2_ratio': float(s2_ratio),
        's3_ratio': float(s3_ratio),
        'total_rotation_deg': float(total_angle),
        'total_rotation_turns': float(total_angle / 360),
        'radius_cv': float(np.std(radii) / np.mean(radii)),
        'singular_values_pc1': S_pc1[:10].tolist(),
        'singular_values_pc2': S_pc2[:5].tolist(),
        'singular_values_pc3': S_pc3[:5].tolist(),
    }

    if len(rotation_trend) >= 4:
        results['rotation_early'] = float(early_rot)
        results['rotation_mid'] = float(mid_rot)
        results['rotation_late'] = float(late_rot)

    if W_full is not None and len(Vt_pc1) >= 3:
        results['axis_svd0_cos'] = float(cos_v3_svd0)
        results['plane_svd0_cos_max'] = float(max(cos_v1_svd0, cos_v2_svd0))

    with open(f'{out_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    log(f"\n  结果已保存到 {out_dir}/results.json")
    log(f"\n{'=' * 70}")
    log(f"CCXXVIII 完成 — {config['name']}")
    log(f"{'=' * 70}")

    # 释放GPU
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
