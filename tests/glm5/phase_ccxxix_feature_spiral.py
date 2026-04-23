"""
Phase CCXXIX: 特征独立螺旋分析与频分复用验证
================================================================
核心目标:
1. 对每个特征单独做PC1轨迹SVD → 各特征是否是2D螺旋?
2. 验证不同特征的PC1是否占据正交子空间 → 频分复用假说
3. 分析各特征子空间的维度 → 5个特征需要多少维?
4. 检查子空间重叠度 → 特征之间是否有信息共享?
5. 对比: 特征子空间 vs 全局PC1子空间

关键假设:
  CCXXVIII发现各特征的PC1在全局螺旋平面上投影极低(0.3-19.5%),
  暗示每个特征有自己的旋转平面。如果这些平面正交, 则是"频分复用":
  不同语义信息在d_model维空间的不同子空间中独立处理。

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

# 特征对定义 (复用CCXXVIII)
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

    out_dir = f'results/causal_fiber/{model_key}_ccxxix'
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
    log(f"Phase CCXXIX: 特征独立螺旋分析与频分复用验证 — {config['name']}")
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

    # ============================================================
    # S2: 逐层收集各特征的差分向量 → 每特征每层PC1/PC2/PC3
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S2: 逐层收集各特征差分向量")
    log(f"{'=' * 60}")

    sample_layers = list(range(0, n_layers, 2))
    if last_layer not in sample_layers:
        sample_layers.append(last_layer)
    sample_layers.sort()

    log(f"  采样层 ({len(sample_layers)}层): {sample_layers[:5]}...{sample_layers[-3:]}")

    # feat -> L -> [diff_vectors]
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
                        feat_layer_diffs[feat][L].append(diff)

                    del out_pos, out_neg

                if (i + 1) % 50 == 0:
                    log(f"    {feat}: {i+1}/{len(pairs)}")
            except Exception as e:
                continue

    # 计算每特征每层的PC1/PC2/PC3
    feat_layer_pca = {}  # (feat, L) -> (components[:3], var_ratio[:3], pca_obj)

    for feat in feature_names:
        for L in sample_layers:
            diffs = feat_layer_diffs[feat][L]
            if len(diffs) >= 10:
                diffs = np.array(diffs)
                pca = PCA(n_components=min(3, len(diffs), diffs.shape[1]))
                pca.fit(diffs)
                feat_layer_pca[(feat, L)] = (pca.components_[:3], pca.explained_variance_ratio_[:3], pca)

    log(f"\n  计算完成: {len(feat_layer_pca)} 个(特征,层)组合有PCA结果")

    # ============================================================
    # S3: 各特征独立PC1轨迹SVD → 是否2D螺旋?
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S3: 各特征独立PC1轨迹SVD分析")
    log(f"{'=' * 60}")

    feat_spiral_results = {}

    for feat in feature_names:
        log(f"\n  --- {feat} PC1轨迹SVD ---")

        # 收集该特征在各层的PC1方向
        pc1_dirs = {}
        for L in sample_layers:
            if (feat, L) in feat_layer_pca:
                pc1_dirs[L] = feat_layer_pca[(feat, L)][0][0]

        if len(pc1_dirs) < 5:
            log(f"    不足5层有数据, 跳过")
            continue

        layers_sorted = sorted(pc1_dirs.keys())
        M = np.array([pc1_dirs[L] for L in layers_sorted])  # (n_layers, d_model)

        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        s_total = S.sum()
        s2 = S[:2].sum() / s_total
        s3 = S[:3].sum() / s_total

        # 2D投影旋转分析
        V_plane = Vt[:2]
        proj_2d = [(L, np.dot(pc1_dirs[L], V_plane[0]), np.dot(pc1_dirs[L], V_plane[1]))
                    for L in layers_sorted]

        angles = [np.degrees(np.arctan2(p2, p1)) for _, p1, p2 in proj_2d]
        radii = [np.sqrt(p1**2 + p2**2) for _, p1, p2 in proj_2d]

        # 展开角度
        unwrapped = [angles[0]]
        for j in range(1, len(angles)):
            diff = angles[j] - angles[j-1]
            while diff > 180: diff -= 360
            while diff < -180: diff += 360
            unwrapped.append(unwrapped[-1] + diff)
        total_angle = unwrapped[-1] - unwrapped[0]

        # 线性度
        x_a = np.arange(len(unwrapped))
        y_a = np.array(unwrapped)
        if len(x_a) > 2:
            coeffs = np.polyfit(x_a, y_a, 1)
            y_pred = np.polyval(coeffs, x_a)
            ss_res = np.sum((y_a - y_pred) ** 2)
            ss_tot = np.sum((y_a - np.mean(y_a)) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            r_sq = 0

        spiral_type = "2D螺旋" if s2 > 0.85 else ("3D锥面螺旋" if s3 > 0.85 else "高维曲线")

        feat_spiral_results[feat] = {
            's2': float(s2), 's3': float(s3),
            'total_angle': float(total_angle), 'turns': float(total_angle / 360),
            'r_sq': float(r_sq), 'spiral_type': spiral_type,
            'radius_cv': float(np.std(radii) / np.mean(radii)) if np.mean(radii) > 0 else 0,
        }

        log(f"    前2奇异值占比: {s2*100:.1f}%")
        log(f"    前3奇异值占比: {s3*100:.1f}%")
        log(f"    2D投影旋转: {total_angle:.1f}° ({total_angle/360:.2f}圈)")
        log(f"    半径CV: {np.std(radii)/np.mean(radii)*100:.1f}%")
        log(f"    线性度R²: {r_sq:.4f}")
        log(f"    类型: {spiral_type}")

    # 汇总
    log(f"\n  --- 各特征螺旋类型汇总 ---")
    log(f"  {'特征':>20} | {'前2PC':>6} | {'前3PC':>6} | {'旋转°':>8} | {'圈数':>5} | {'R²':>6} | {'类型':>12}")
    log(f"  {'-'*20} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*5} | {'-'*6} | {'-'*12}")
    for feat in feature_names:
        if feat in feat_spiral_results:
            r = feat_spiral_results[feat]
            log(f"  {feat:>20} | {r['s2']*100:>5.1f}% | {r['s3']*100:>5.1f}% | {r['total_angle']:>8.1f} | {r['turns']:>5.2f} | {r['r_sq']:>6.3f} | {r['spiral_type']:>12}")

    # ============================================================
    # S4: 特征间PC1正交性分析 — 频分复用验证
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S4: 特征间PC1正交性分析 — 频分复用验证")
    log(f"{'=' * 60}")

    # 在每一层, 计算5个特征PC1之间的两两cos值
    for L in sample_layers:
        feat_pc1s = {}
        for feat in feature_names:
            if (feat, L) in feat_layer_pca:
                feat_pc1s[feat] = feat_layer_pca[(feat, L)][0][0]

        if len(feat_pc1s) < 3:
            continue

        if L % 6 == 0 or L == last_layer:  # 只打印部分层
            log(f"\n  Layer {L} 特征间|cos|矩阵:")
            feats_present = sorted(feat_pc1s.keys())
            # 打印表头
            header = f"  {'':>20}"
            for f2 in feats_present:
                header += f" | {f2[:8]:>8}"
            log(header)
            log(f"  {'-'*20}" + f" | {'-'*8}" * len(feats_present))

            for f1 in feats_present:
                row = f"  {f1:>20}"
                for f2 in feats_present:
                    if f1 == f2:
                        row += f" | {'1.000':>8}"
                    else:
                        cos_val = abs(np.dot(feat_pc1s[f1], feat_pc1s[f2]) /
                                     (np.linalg.norm(feat_pc1s[f1]) * np.linalg.norm(feat_pc1s[f2])))
                        row += f" | {cos_val:>8.3f}"
                log(row)

    # S4b: 平均跨层正交性
    log(f"\n  --- 跨层平均特征间|cos| ---")
    pair_cos_values = defaultdict(list)

    for L in sample_layers:
        feat_pc1s = {}
        for feat in feature_names:
            if (feat, L) in feat_layer_pca:
                feat_pc1s[feat] = feat_layer_pca[(feat, L)][0][0]

        feats_present = sorted(feat_pc1s.keys())
        for i, f1 in enumerate(feats_present):
            for f2 in feats_present[i+1:]:
                cos_val = abs(np.dot(feat_pc1s[f1], feat_pc1s[f2]) /
                             (np.linalg.norm(feat_pc1s[f1]) * np.linalg.norm(feat_pc1s[f2])))
                pair_cos_values[(f1, f2)].append(cos_val)

    log(f"  {'特征对':>40} | {'平均|cos|':>10} | {'最大|cos|':>10} | {'层数':>4}")
    log(f"  {'-'*40} | {'-'*10} | {'-'*10} | {'-'*4}")

    all_pair_means = []
    for (f1, f2), vals in sorted(pair_cos_values.items()):
        mean_cos = np.mean(vals)
        max_cos = np.max(vals)
        all_pair_means.append(mean_cos)
        log(f"  {f1+' × '+f2:>40} | {mean_cos:>10.4f} | {max_cos:>10.4f} | {len(vals):>4}")

    if all_pair_means:
        grand_mean = np.mean(all_pair_means)
        log(f"\n  ** 所有特征对的平均|cos| = {grand_mean:.4f} **")
        if grand_mean < 0.15:
            log(f"  → 特征间PC1高度正交 → ** 频分复用假说成立! **")
        elif grand_mean < 0.3:
            log(f"  → 特征间PC1中度正交 → 部分频分复用")
        else:
            log(f"  → 特征间PC1低正交性 → 不支持频分复用假说")

    # ============================================================
    # S5: 特征子空间维度分析
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S5: 特征子空间维度分析")
    log(f"{'=' * 60}")

    # 在最后一层, 将5个特征的PC1放在一起做SVD
    # 理论上如果完全正交, 5个PC1张成5维子空间
    for L_key in [peak_layer if 'peak_layer' in dir() else sample_layers[len(sample_layers)//2],
                  last_layer]:
        feat_pc1s = {}
        for feat in feature_names:
            if (feat, L_key) in feat_layer_pca:
                feat_pc1s[feat] = feat_layer_pca[(feat, L_key)][0][0]

        if len(feat_pc1s) < 3:
            continue

        log(f"\n  Layer {L_key}: {len(feat_pc1s)}个特征的PC1子空间分析")

        M_feats = np.array([feat_pc1s[f] for f in sorted(feat_pc1s.keys())])  # (n_feat, d_model)
        U_f, S_f, Vt_f = np.linalg.svd(M_feats, full_matrices=False)

        log(f"  SVD奇异值: {S_f.tolist()}")
        log(f"  前1奇异值占比: {S_f[0]**2/(S_f**2).sum()*100:.1f}%")

        # 有效维度 (奇异值>0.5*最大值的个数)
        n_eff = sum(1 for s in S_f if s > 0.5 * S_f[0])
        log(f"  有效维度(σ>0.5σ_max): {n_eff}")

        # 累计方差占比
        cumvar = np.cumsum(S_f**2) / (S_f**2).sum()
        for k in range(len(cumvar)):
            log(f"  前{k+1}奇异值累计: {cumvar[k]*100:.1f}%")

        if n_eff == len(feat_pc1s):
            log(f"  → 所有特征PC1近似线性独立 → 子空间维度={len(feat_pc1s)}")
        else:
            log(f"  → 部分特征PC1共面 → 有效维度={n_eff} < {len(feat_pc1s)}")

    # S5b: 更全面的分析 — 在多层合并所有特征的PC1
    log(f"\n  --- 全层特征PC1合并SVD ---")
    all_feat_pc1s = []
    for L in sample_layers:
        for feat in feature_names:
            if (feat, L) in feat_layer_pca:
                all_feat_pc1s.append(feat_layer_pca[(feat, L)][0][0])

    if len(all_feat_pc1s) >= 5:
        M_all = np.array(all_feat_pc1s)  # (n_feat*n_layers, d_model)
        U_all, S_all, Vt_all = np.linalg.svd(M_all, full_matrices=False)

        log(f"  矩阵shape: {M_all.shape}")
        log(f"  前10奇异值: {S_all[:10].round(3).tolist()}")
        log(f"  前20奇异值: {S_all[:20].round(3).tolist()}")

        cumvar_all = np.cumsum(S_all**2) / (S_all**2).sum()
        for k in [4, 9, 14, 19]:
            if k < len(cumvar_all):
                log(f"  前{k+1}奇异值累计: {cumvar_all[k]*100:.1f}%")

        # 有效维度
        n_eff_all = sum(1 for s in S_all if s > 0.1 * S_all[0])
        log(f"  有效维度(σ>0.1σ_max): {n_eff_all}")

        n_eff_50 = sum(1 for s in S_all if s > 0.5 * S_all[0])
        log(f"  有效维度(σ>0.5σ_max): {n_eff_50}")

    # ============================================================
    # S6: 特征子空间重叠度 — Grassmann距离
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S6: 特征子空间重叠度 (Grassmann距离)")
    log(f"{'=' * 60}")

    # 在最后一层, 每个特征用前3PC构成子空间, 计算子空间间的重叠度
    log(f"  在最后一层, 各特征前3PC子空间的重叠度:")

    feat_subspaces = {}
    for feat in feature_names:
        if (feat, last_layer) in feat_layer_pca:
            comps = feat_layer_pca[(feat, last_layer)][0]  # (3, d_model)
            if comps.shape[0] >= 2:
                feat_subspaces[feat] = comps[:2]  # 取前2PC

    if len(feat_subspaces) >= 3:
        feats_with_sub = sorted(feat_subspaces.keys())
        log(f"  {'特征对':>40} | {'Grassmann距离':>14} | {'最大主对齐':>12}")
        log(f"  {'-'*40} | {'-'*14} | {'-'*12}")

        grassmann_dists = []
        for i, f1 in enumerate(feats_with_sub):
            for f2 in feats_with_sub[i+1:]:
                S1 = feat_subspaces[f1]  # (2, d_model)
                S2 = feat_subspaces[f2]  # (2, d_model)

                # 子空间重叠: S1 @ S2.T → (2, 2), 做SVD得主角度的cos
                M_sub = S1 @ S2.T  # (2, 2)
                U_s, S_s, Vt_s = np.linalg.svd(M_sub)

                # Grassmann距离 = sqrt(sum(1 - σ_i^2))
                g_dist = np.sqrt(sum(1 - s**2 for s in S_s))
                max_align = max(S_s)

                grassmann_dists.append(g_dist)
                log(f"  {f1+' × '+f2:>40} | {g_dist:>14.4f} | {max_align:>12.4f}")

        if grassmann_dists:
            mean_gd = np.mean(grassmann_dists)
            log(f"\n  ** 平均Grassmann距离 = {mean_gd:.4f} **")
            if mean_gd > 1.3:
                log(f"  → 子空间高度分离 → 频分复用强")
            elif mean_gd > 1.0:
                log(f"  → 子空间中度分离 → 频分复用中等")
            else:
                log(f"  → 子空间低分离度 → 频分复用弱")

    # ============================================================
    # S7: 各特征PC1轨迹之间的对齐轨迹
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S7: 各特征PC1跨层对齐演化")
    log(f"{'=' * 60}")

    # 选2对特征, 看它们PC1之间的cos如何随层变化
    test_pairs = [('tense', 'polarity'), ('tense', 'voice'), ('voice', 'semantic_valence'),
                  ('semantic_valence', 'semantic_topic')]

    for f1, f2 in test_pairs:
        log(f"\n  --- {f1} vs {f2} PC1跨层|cos| ---")
        cos_by_layer = []
        for L in sample_layers:
            if (f1, L) in feat_layer_pca and (f2, L) in feat_layer_pca:
                d1 = feat_layer_pca[(f1, L)][0][0]
                d2 = feat_layer_pca[(f2, L)][0][0]
                cos_val = abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
                cos_by_layer.append((L, cos_val))
                if L % 6 == 0 or L == last_layer:
                    log(f"    L{L}: |cos|={cos_val:.4f}")

        if cos_by_layer:
            mean_cos = np.mean([c for _, c in cos_by_layer])
            log(f"    平均|cos|={mean_cos:.4f}")

    # ============================================================
    # S8: 汇总
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S8: 汇总")
    log(f"{'=' * 60}")

    # 各特征螺旋类型
    log(f"\n  1. 各特征PC1轨迹类型:")
    for feat in feature_names:
        if feat in feat_spiral_results:
            r = feat_spiral_results[feat]
            log(f"     {feat}: {r['spiral_type']} (前2PC={r['s2']*100:.0f}%, 旋转={r['total_angle']:.0f}°={r['turns']:.2f}圈)")

    # 频分复用结论
    if all_pair_means:
        log(f"\n  2. 频分复用验证:")
        log(f"     特征间PC1平均|cos| = {grand_mean:.4f}")
        if grand_mean < 0.15:
            log(f"     → 高度正交 → 频分复用假说成立")
        elif grand_mean < 0.3:
            log(f"     → 中度正交 → 部分频分复用")
        else:
            log(f"     → 低正交性 → 频分复用假说不成立")

    # 子空间维度
    if 'n_eff_all' in dir() or True:
        # 重新获取
        all_pc1s_last = []
        for feat in feature_names:
            if (feat, last_layer) in feat_layer_pca:
                all_pc1s_last.append(feat_layer_pca[(feat, last_layer)][0][0])
        if len(all_pc1s_last) >= 3:
            M_last = np.array(all_pc1s_last)
            _, S_last, _ = np.linalg.svd(M_last, full_matrices=False)
            n_eff_last = sum(1 for s in S_last if s > 0.5 * S_last[0])
            log(f"\n  3. 最后一层特征子空间维度:")
            log(f"     5个特征的PC1有效维度: {n_eff_last}")
            log(f"     奇异值: {S_last.round(3).tolist()}")

    # 保存结果
    results = {
        'model': config['name'],
        'feat_spiral': feat_spiral_results,
        'grand_mean_cos': float(grand_mean) if all_pair_means else 0,
    }

    with open(f'{out_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\n  结果已保存到 {out_dir}/results.json")
    log(f"\n{'=' * 70}")
    log(f"CCXXIX 完成 — {config['name']}")
    log(f"{'=' * 70}")

    # 释放GPU
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
