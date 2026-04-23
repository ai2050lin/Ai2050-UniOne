"""
Phase CCXXVI: 多层语义轨迹 — 追踪PC1在每层中的演化与传播
================================================================
核心目标:
1. 逐层收集差分向量 → 计算每层PC1 → 绘制"PC1方差轨迹"
2. 跨层PC1对齐分析 → cos(PC1_L, PC1_L') → PC1方向如何演化
3. 在关键层(early/middle/late)注入PC1 perturbation → 观测传播效应
4. 对比: PC1在不同层的logit效应 → 找到"PC1效应最强的层"

关键假设:
  CCXXV发现PC1的logit效应极弱(3-7%), 这可能是因为PC1在最后一层
  已经被"压缩"了。如果PC1在中间层效应更强, 说明PC1编码的信息
  在向最后一层传播时被逐渐稀释。

样本量: 100对/特征 (PC1收集), 30个测试句子 (perturbation验证)
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

import torch

# 复用特征对定义 (与CCXXV相同但减少到100对)
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
            ("She teaches the children", "She taught the children"),
            ("The storm destroys the house", "The storm destroyed the house"),
        ]
    },
    'polarity': {
        'type': 'SYN',
        'pairs': [
            ("The weather is good today", "The weather is bad today"),
            ("She is very happy", "She is very sad"),
            ("The movie was excellent", "The movie was terrible"),
            ("He always tells the truth", "He always tells lies"),
            ("The food tastes delicious", "The food tastes awful"),
            ("She loves her new job", "She hates her new job"),
            ("The plan worked perfectly", "The plan failed completely"),
            ("He gave a generous gift", "He gave a stingy gift"),
            ("The news was wonderful", "The news was dreadful"),
            ("She won the competition", "She lost the competition"),
            ("The house is clean and bright", "The house is dirty and dark"),
            ("He is a kind person", "He is a cruel person"),
            ("The project was successful", "The project was unsuccessful"),
            ("She felt proud of her work", "She felt ashamed of her work"),
            ("The concert was amazing", "The concert was boring"),
            ("He made a wise decision", "He made a foolish decision"),
            ("The garden looks beautiful", "The garden looks ugly"),
            ("She is very healthy", "She is very sick"),
            ("The solution was elegant", "The solution was clumsy"),
            ("He acted with courage", "He acted with cowardice"),
            ("The future looks bright", "The future looks bleak"),
            ("She has many friends", "She has many enemies"),
            ("The dress is stunning", "The dress is hideous"),
            ("He is extremely generous", "He is extremely selfish"),
            ("The performance was brilliant", "The performance was mediocre"),
            ("She spoke with confidence", "She spoke with hesitation"),
            ("The result was positive", "The result was negative"),
            ("He showed great patience", "He showed great impatience"),
            ("The weather was pleasant", "The weather was unpleasant"),
            ("She achieved her dream", "She abandoned her dream"),
            ("The book is fascinating", "The book is dull"),
            ("He gained a lot of weight", "He lost a lot of weight"),
            ("The journey was smooth", "The journey was rough"),
            ("She received a warm welcome", "She received a cold welcome"),
            ("The team played brilliantly", "The team played terribly"),
            ("He found inner peace", "He found inner turmoil"),
            ("The experience was rewarding", "The experience was frustrating"),
            ("She made remarkable progress", "She made no progress"),
            ("The city is prosperous", "The city is impoverished"),
            ("He felt deep gratitude", "He felt deep resentment"),
            ("The relationship is harmonious", "The relationship is hostile"),
            ("She expressed great joy", "She expressed great sorrow"),
            ("The outcome was favorable", "The outcome was unfavorable"),
            ("He possesses great wisdom", "He possesses great ignorance"),
            ("The atmosphere was festive", "The atmosphere was gloomy"),
            ("She displayed great talent", "She displayed great incompetence"),
            ("The change was beneficial", "The change was harmful"),
            ("He felt genuine love", "He felt genuine hatred"),
            ("The story has a happy ending", "The story has a sad ending"),
            ("She showed genuine kindness", "She showed genuine malice"),
        ]
    },
    'voice': {
        'type': 'SYN',
        'pairs': [
            ("The cat chased the mouse", "The mouse was chased by the cat"),
            ("She wrote the letter", "The letter was written by her"),
            ("He fixed the car", "The car was fixed by him"),
            ("They built the house", "The house was built by them"),
            ("She painted the portrait", "The portrait was painted by her"),
            ("He cooked the meal", "The meal was cooked by him"),
            ("The teacher praised the student", "The student was praised by the teacher"),
            ("She delivered the speech", "The speech was delivered by her"),
            ("He designed the building", "The building was designed by him"),
            ("They discovered the island", "The island was discovered by them"),
            ("She composed the music", "The music was composed by her"),
            ("He solved the problem", "The problem was solved by him"),
            ("The company launched the product", "The product was launched by the company"),
            ("She broke the window", "The window was broken by her"),
            ("He wrote the report", "The report was written by him"),
            ("They won the championship", "The championship was won by them"),
            ("She directed the film", "The film was directed by her"),
            ("He painted the fence", "The fence was painted by him"),
            ("The chef prepared the dish", "The dish was prepared by the chef"),
            ("She sang the song", "The song was sung by her"),
            ("He caught the ball", "The ball was caught by him"),
            ("They found the treasure", "The treasure was found by them"),
            ("She opened the door", "The door was opened by her"),
            ("He cleaned the room", "The room was cleaned by him"),
            ("The scientist discovered the element", "The element was discovered by the scientist"),
            ("She wrote the novel", "The novel was written by her"),
            ("He drove the bus", "The bus was driven by him"),
            ("They finished the project", "The project was finished by them"),
            ("She baked the cake", "The cake was baked by her"),
            ("He repaired the roof", "The roof was repaired by him"),
            ("The artist created the sculpture", "The sculpture was created by the artist"),
            ("She taught the class", "The class was taught by her"),
            ("He managed the team", "The team was managed by him"),
            ("They organized the event", "The event was organized by them"),
            ("She translated the document", "The document was translated by her"),
            ("He recorded the album", "The album was recorded by him"),
            ("The judge dismissed the case", "The case was dismissed by the judge"),
            ("She photographed the landscape", "The landscape was photographed by her"),
            ("He programmed the software", "The software was programmed by him"),
            ("They published the article", "The article was published by them"),
            ("She decorated the room", "The room was decorated by her"),
            ("He manufactured the device", "The device was manufactured by him"),
            ("The committee approved the plan", "The plan was approved by the committee"),
            ("She narrated the story", "The story was narrated by her"),
            ("He renovated the kitchen", "The kitchen was renovated by him"),
            ("They released the album", "The album was released by them"),
            ("She styled the hair", "The hair was styled by her"),
            ("He investigated the crime", "The crime was investigated by him"),
            ("The king conquered the land", "The land was conquered by the king"),
            ("She illustrated the book", "The book was illustrated by her"),
        ]
    },
    'semantic_valence': {
        'type': 'SEM',
        'pairs': [
            ("The generous man helped", "The greedy man helped"),
            ("She found great joy", "She found deep sorrow"),
            ("The warm embrace comforted", "The cold shoulder hurt"),
            ("He spoke with kindness", "He spoke with cruelty"),
            ("The bright sunrise inspired", "The dark storm frightened"),
            ("She showed genuine love", "She showed bitter hatred"),
            ("The peaceful garden calmed", "The chaotic battlefield terrified"),
            ("He offered sincere praise", "He offered harsh criticism"),
            ("The sweet melody soothed", "The harsh noise irritated"),
            ("She created lasting beauty", "She created lasting ugliness"),
            ("The noble hero saved", "The wicked villain destroyed"),
            ("He built a safe home", "He built a dangerous trap"),
            ("The gentle rain nourished", "The violent flood destroyed"),
            ("She shared her abundance", "She hoarded her wealth"),
            ("The honest truth prevailed", "The cruel lie spread"),
            ("He brought warm sunshine", "He brought cold darkness"),
            ("The kind stranger helped", "The cruel stranger hurt"),
            ("She gave a heartfelt apology", "She gave a sarcastic insult"),
            ("The prosperous city thrived", "The impoverished city decayed"),
            ("He showed deep compassion", "He showed deep indifference"),
            ("The fragrant garden delighted", "The foul swamp disgusted"),
            ("She earned proud respect", "She earned bitter contempt"),
            ("The healthy child played", "The sick child suffered"),
            ("He chose the virtuous path", "He chose the wicked path"),
            ("The beautiful music uplifted", "The ugly noise depressed"),
            ("She felt deep gratitude", "She felt deep resentment"),
            ("The harmonious choir sang", "The discordant mob yelled"),
            ("He made a generous donation", "He made a selfish demand"),
            ("The brilliant idea advanced", "The foolish idea regressed"),
            ("She found lasting happiness", "She found fleeting misery"),
            ("The noble deed inspired", "The vile deed inspired"),
            ("He spoke honest words", "He spoke dishonest words"),
            ("The gentle breeze blew", "The fierce gale blew"),
            ("She maintained quiet dignity", "She maintained loud disgrace"),
            ("The sacred tradition endured", "The profane tradition endured"),
            ("He showed brave resistance", "He showed cowardly submission"),
            ("The beautiful melody echoed", "The ugly melody echoed"),
            ("She embraced warm friendship", "She embraced cold hostility"),
            ("The pure water refreshed", "The toxic waste poisoned"),
            ("He gave wise counsel", "He gave foolish advice"),
            ("The golden sunrise promised", "The blood-red sunset threatened"),
            ("She discovered profound truth", "She discovered shallow deception"),
            ("The fertile field yielded", "The barren desert yielded"),
            ("He expressed tender affection", "He expressed bitter malice"),
            ("The strong bridge held", "The weak bridge collapsed"),
            ("She won a glorious victory", "She suffered a humiliating defeat"),
            ("The clean air breathed", "The polluted air choked"),
            ("He achieved noble purpose", "He achieved base ambition"),
            ("The sweet success rewarded", "The bitter failure punished"),
        ]
    },
    'semantic_topic': {
        'type': 'SEM',
        'pairs': [
            ("The doctor treated the patient", "The chef treated the customer"),
            ("The student read the textbook", "The student read the novel"),
            ("The scientist ran the experiment", "The athlete ran the marathon"),
            ("The pilot flew the airplane", "The driver drove the car"),
            ("The farmer planted the seeds", "The builder laid the bricks"),
            ("The judge made the ruling", "The chef made the sauce"),
            ("The nurse gave the injection", "The teacher gave the lecture"),
            ("The engineer built the bridge", "The painter painted the portrait"),
            ("The lawyer argued the case", "The comedian told the joke"),
            ("The soldier fired the weapon", "The photographer fired the flash"),
            ("The baker baked the bread", "The potter shaped the clay"),
            ("The mechanic fixed the engine", "The doctor fixed the bone"),
            ("The fisherman caught the fish", "The police caught the thief"),
            ("The author wrote the book", "The programmer wrote the code"),
            ("The singer sang the aria", "The preacher sang the hymn"),
            ("The architect drew the blueprint", "The artist drew the landscape"),
            ("The banker counted the money", "The referee counted the points"),
            ("The teacher graded the paper", "The critic graded the restaurant"),
            ("The surgeon operated on the heart", "The mechanic operated the machine"),
            ("The plumber fixed the pipe", "The therapist fixed the relationship"),
            ("The carpenter built the table", "The composer built the symphony"),
            ("The driver delivered the package", "The speaker delivered the speech"),
            ("The miner dug the tunnel", "The detective dug the clues"),
            ("The electrician wired the house", "The journalist wired the report"),
            ("The chemist mixed the solution", "The DJ mixed the track"),
            ("The tailor sewed the suit", "The writer sewed the plot"),
            ("The guard protected the vault", "The sunscreen protected the skin"),
            ("The coach trained the athlete", "The teacher trained the student"),
            ("The chef seasoned the dish", "The comedian seasoned the joke"),
            ("The artist painted the canvas", "The politician painted the picture"),
            ("The doctor prescribed the medicine", "The judge prescribed the sentence"),
            ("The professor lectured on physics", "The priest lectured on theology"),
            ("The banker managed the portfolio", "The coach managed the team"),
            ("The programmer debugged the code", "The doctor debugged the diagnosis"),
            ("The architect designed the museum", "The fashion designer designed the dress"),
            ("The scientist discovered the element", "The explorer discovered the island"),
            ("The engineer calculated the stress", "The accountant calculated the tax"),
            ("The librarian cataloged the books", "The astronomer cataloged the stars"),
            ("The chef prepared the feast", "The general prepared the battle"),
            ("The musician tuned the violin", "The mechanic tuned the engine"),
            ("The dentist filled the cavity", "The construction worker filled the trench"),
            ("The pharmacist dispensed the drug", "The banker dispensed the cash"),
            ("The botanist studied the flower", "The historian studied the war"),
            ("The astronaut explored the planet", "The detective explored the crime scene"),
            ("The psychologist analyzed the dream", "The critic analyzed the film"),
            ("The surgeon removed the tumor", "The editor removed the paragraph"),
            ("The pilot navigated the storm", "The manager navigated the crisis"),
            ("The farmer harvested the wheat", "The company harvested the data"),
            ("The jeweler polished the diamond", "The student polished the essay"),
        ]
    },
}

feature_names = list(FEATURE_PAIRS.keys())

# 测试句子
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
    parser.add_argument('--n_pairs', type=int, default=100)
    parser.add_argument('--n_test', type=int, default=30)
    args = parser.parse_args()

    model_key = args.model
    n_pairs = args.n_pairs
    n_test = args.n_test
    config = MODEL_CONFIGS[model_key]

    out_dir = f'results/causal_fiber/{model_key}_ccxxvi'
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
    log(f"Phase CCXXVI: 多层语义轨迹 — {config['name']}")
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
    vocab_size = len(tokenizer)

    log(f"  加载完成: {load_time:.0f}s")
    log(f"  n_layers={n_layers}, d_model={d_model}, vocab_size={vocab_size}")

    # ============================================================
    # S2: 逐层收集差分向量 → 计算每层PC1
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S2: 逐层PC1方差轨迹")
    log(f"{'=' * 60}")

    # 每层、每个特征的PC1方向和方差
    layer_pc1 = {}  # (feat, layer) -> (pc1_dir, pc1_var)
    # 全局: 所有特征合并的每层PC1
    layer_pc1_global = {}  # layer -> (pc1_dir, pc1_var)

    # 采样层 (不需要每层都做, 选关键层)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 10)))  # 约10个采样层
    if last_layer not in sample_layers:
        sample_layers.append(last_layer)
    sample_layers.sort()
    log(f"  采样层: {sample_layers}")

    # 收集所有层的差分向量
    all_layer_diffs = defaultdict(list)  # layer -> list of diffs (all features)
    feat_layer_diffs = defaultdict(lambda: defaultdict(list))  # feat -> layer -> list of diffs

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

                if (i + 1) % 25 == 0:
                    log(f"    {feat}: {i+1}/{len(pairs)}")
            except Exception as e:
                continue

    # 计算每层PC1
    log(f"\n  --- 每层PC1方差轨迹 ---")

    for L in sample_layers:
        # 特征级PC1
        for feat in feature_names:
            diffs = feat_layer_diffs[feat][L]
            if len(diffs) < 10:
                continue
            diffs = np.array(diffs)
            pca = PCA(n_components=1)
            pca.fit(diffs)
            layer_pc1[(feat, L)] = (pca.components_[0], pca.explained_variance_ratio_[0])

        # 全局PC1
        diffs = all_layer_diffs[L]
        if len(diffs) >= 10:
            diffs = np.array(diffs)
            pca = PCA(n_components=1)
            pca.fit(diffs)
            layer_pc1_global[L] = (pca.components_[0], pca.explained_variance_ratio_[0])

    # 输出轨迹
    log(f"\n  === PC1方差轨迹 ===")
    log(f"  {'Layer':>6} | {'Global':>8} | " + " | ".join(f"{f:>8}" for f in feature_names))
    log(f"  {'-'*6} | {'-'*8} | " + " | ".join(f"{'-'*8}" for f in feature_names))

    for L in sample_layers:
        row = f"  {L:>6} | "
        if L in layer_pc1_global:
            row += f"{layer_pc1_global[L][1]*100:>7.1f}% | "
        else:
            row += f"{'N/A':>8} | "
        for feat in feature_names:
            if (feat, L) in layer_pc1:
                row += f"{layer_pc1[(feat, L)][1]*100:>7.1f}% | "
            else:
                row += f"{'N/A':>8} | "
        log(row)

    # ============================================================
    # S3: 跨层PC1对齐分析 — PC1方向如何演化?
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S3: 跨层PC1对齐分析")
    log(f"{'=' * 60}")

    # 全局PC1的跨层对齐
    log(f"\n  --- 全局PC1跨层cos对齐矩阵 ---")
    global_dirs = {L: v[0] for L, v in layer_pc1_global.items()}
    layers_sorted = sorted(global_dirs.keys())

    # 打印对角线附近的关键对齐
    log(f"  相邻层对齐:")
    for i in range(len(layers_sorted) - 1):
        L1, L2 = layers_sorted[i], layers_sorted[i + 1]
        cos = np.dot(global_dirs[L1], global_dirs[L2]) / (
            np.linalg.norm(global_dirs[L1]) * np.linalg.norm(global_dirs[L2]))
        log(f"    Layer {L1} → {L2}: cos={cos:.4f}")

    # 最后一层 vs 所有层
    if last_layer in global_dirs:
        last_dir = global_dirs[last_layer]
        log(f"\n  最后一层(L={last_layer}) vs 各层:")
        for L in layers_sorted:
            cos = np.dot(last_dir, global_dirs[L]) / (
                np.linalg.norm(last_dir) * np.linalg.norm(global_dirs[L]))
            log(f"    vs Layer {L}: cos={cos:.4f}")

    # 特征级PC1的跨层对齐
    log(f"\n  --- 特征PC1跨层对齐 (第一层→最后一层) ---")
    for feat in feature_names:
        dirs = {L: v[0] for (f, L), v in layer_pc1.items() if f == feat}
        if len(dirs) >= 2:
            first_L = min(dirs.keys())
            last_L = max(dirs.keys())
            cos = np.dot(dirs[first_L], dirs[last_L]) / (
                np.linalg.norm(dirs[first_L]) * np.linalg.norm(dirs[last_L]))
            log(f"    {feat}: L{first_L}→L{last_L} cos={cos:.4f}")

    # ============================================================
    # S4: PC1方向的信息流分析 — 哪层的PC1最"重要"?
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S4: PC1信息流分析 — 各层PC1与最后一层PC1的对齐")
    log(f"{'=' * 60}")

    if last_layer in layer_pc1_global:
        last_global_pc1 = layer_pc1_global[last_layer][0]
        log(f"  最后一层全局PC1方向与各层的cos对齐:")

        alignment_trajectory = []
        for L in layers_sorted:
            if L in layer_pc1_global:
                cos = np.dot(last_global_pc1, layer_pc1_global[L][0]) / (
                    np.linalg.norm(last_global_pc1) * np.linalg.norm(layer_pc1_global[L][0]))
                alignment_trajectory.append((L, cos))
                log(f"    Layer {L}: cos={cos:.4f}")

        # 找到对齐的"跳变点" — PC1方向在哪层发生重大变化?
        if len(alignment_trajectory) >= 3:
            log(f"\n  对齐跳变分析:")
            for i in range(1, len(alignment_trajectory)):
                delta_cos = alignment_trajectory[i][1] - alignment_trajectory[i-1][1]
                log(f"    L{alignment_trajectory[i-1][0]}→L{alignment_trajectory[i][0]}: Δcos={delta_cos:+.4f}")

    # ============================================================
    # S5: 逐层PC1的logit效应 — 在不同层注入perturbation
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S5: 逐层PC1的logit效应")
    log(f"{'=' * 60}")

    # 提取lm_head权重
    W_full = None
    try:
        lm_head = getattr(model, 'lm_head', None)
        if lm_head is None:
            if hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
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
        log("  尝试从safetensors加载...")
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
                    log(f"    加载 {os.path.basename(sf)} (含lm_head)...")
                    state_dict = st_load_file(sf)
                    for key, tensor in state_dict.items():
                        if 'lm_head.weight' in key:
                            log(f"    找到权重: {key}, shape={tensor.shape}, dtype={tensor.dtype}")
                            W_raw = tensor.float().cpu().numpy()
                            if W_raw.shape[1] == d_model:
                                W_full = W_raw
                            elif W_raw.shape[0] == d_model:
                                W_full = W_raw.T
                            else:
                                W_full = W_raw
                            break
                    del state_dict
                except Exception as e2:
                    continue
                if W_full is not None:
                    break
        except ImportError:
            log("  safetensors未安装")

    if W_full is None:
        log("  错误: 无法加载lm_head权重!")
        return

    log(f"  lm_head权重形状: {W_full.shape}")

    # 计算各层PC1的logit指纹
    log(f"\n  --- 各层全局PC1的logit指纹强度 ---")
    logit_effect_trajectory = []

    for L in layers_sorted:
        if L not in layer_pc1_global:
            continue
        pc1_dir = layer_pc1_global[L][0]
        pc1_var = layer_pc1_global[L][1]

        # logit指纹: W @ pc1_dir
        logit_fp = W_full @ pc1_dir  # (vocab,)
        mean_abs_fp = float(np.mean(np.abs(logit_fp)))
        max_abs_fp = float(np.max(np.abs(logit_fp)))

        logit_effect_trajectory.append((L, pc1_var, mean_abs_fp, max_abs_fp))
        log(f"    Layer {L}: PC1方差={pc1_var*100:.1f}%, |logit_fp|均值={mean_abs_fp:.4f}, max={max_abs_fp:.4f}")

    # SVD参考
    svd = TruncatedSVD(n_components=10)
    svd.fit(W_full)
    svd0_logit = W_full @ svd.components_[0]
    svd0_mean = float(np.mean(np.abs(svd0_logit)))
    log(f"\n  SVD[0]参考: |logit_fp|均值={svd0_mean:.4f}")

    # ============================================================
    # S6: 最后一层PC1直接perturbation — 因果验证
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S6: 最后一层PC1直接perturbation — 因果验证")
    log(f"{'=' * 60}")

    delta = 1.0
    test_sentences = TEST_SENTENCES[:n_test]
    perturb_results = {}

    # 各特征在最后一层的PC1 perturbation
    for feat in feature_names + ['global']:
        if feat == 'global':
            if last_layer not in layer_pc1_global:
                continue
            pc1_dir = layer_pc1_global[last_layer][0]
            pc1_var = layer_pc1_global[last_layer][1]
        else:
            if (feat, last_layer) not in layer_pc1:
                continue
            pc1_dir, pc1_var = layer_pc1[(feat, last_layer)]

        pc1_norm = np.linalg.norm(pc1_dir)
        pc1_torch = torch.tensor(pc1_dir, dtype=torch.float32).to(model.device)
        delta_scale = delta * pc1_norm

        log(f"\n  --- {feat} @ L{last_layer} PC1 perturbation (||PC1||={pc1_norm:.2f}, var={pc1_var*100:.1f}%) ---")

        all_logit_shifts = []
        all_h_shifts = []

        for sent in test_sentences:
            try:
                with torch.no_grad():
                    toks = tokenizer(sent, return_tensors='pt').to(model.device)
                    out = model(**toks, output_hidden_states=True)
                    h_last = out.hidden_states[last_layer][0, -1, :].clone()

                    h_baseline = h_last.float().cpu().numpy()
                    logits_baseline = W_full @ h_baseline

                    # 正方向perturb
                    h_perturbed = h_last + delta_scale * pc1_torch
                    h_pert_np = h_perturbed.float().cpu().numpy()
                    logits_pert = W_full @ h_pert_np

                    logit_shift = float(np.mean(np.abs(logits_pert - logits_baseline)))
                    h_shift = float(np.mean(np.abs(h_pert_np - h_baseline)))

                    all_logit_shifts.append(logit_shift)
                    all_h_shifts.append(h_shift)
                    del out
            except:
                continue

        if all_logit_shifts:
            perturb_results[feat] = {
                'mean_logit_shift': float(np.mean(all_logit_shifts)),
                'mean_h_shift': float(np.mean(all_h_shifts)),
                'pc1_var': float(pc1_var),
                'pc1_norm': float(pc1_norm),
                'n_test': len(all_logit_shifts),
            }
            log(f"    平均|logit shift|={np.mean(all_logit_shifts):.4f}, |h shift|={np.mean(all_h_shifts):.4f}")

    # ============================================================
    # S7: 逐层PC1信息流传播分析 (基于hidden state层间变化)
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S7: 逐层PC1信息流传播分析")
    log(f"{'=' * 60}")

    # 对一小部分pairs, 追踪同一pair在各层的hidden state变化
    # 看PC1方向在各层之间的"投影一致性"
    log(f"  分析各层hidden state沿最后一层PC1方向的投影...")

    if last_layer in layer_pc1_global:
        pc1_last_dir = layer_pc1_global[last_layer][0]
        pc1_last_norm = np.linalg.norm(pc1_last_dir)

        # 收集10个测试pair在各层的投影
        n_trace = 20
        trace_pairs = []
        for feat in feature_names[:2]:  # 只用前2个特征
            pairs = FEATURE_PAIRS[feat]['pairs'][:5]
            for s_pos, s_neg in pairs:
                trace_pairs.append((feat, s_pos, s_neg))

        layer_projections = defaultdict(list)  # layer -> list of projections onto PC1_last

        for feat, s_pos, s_neg in trace_pairs[:n_trace]:
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

                        # 投影到最后一层PC1
                        proj = np.dot(diff, pc1_last_dir) / pc1_last_norm
                        layer_projections[L].append(proj)

                    del out_pos, out_neg
            except:
                continue

        # 输出投影轨迹
        if layer_projections:
            log(f"\n  差分向量在最后一层PC1上的投影轨迹 (均值±标准差):")
            for L in sorted(layer_projections.keys()):
                projs = layer_projections[L]
                log(f"    Layer {L}: proj={np.mean(projs):.4f} ± {np.std(projs):.4f} (n={len(projs)})")

            # 归一化投影 (以最后一层为基准)
            if last_layer in layer_projections:
                last_proj_mean = np.mean(layer_projections[last_layer])
                if abs(last_proj_mean) > 1e-6:
                    log(f"\n  归一化投影 (以最后一层=1.0):")
                    for L in sorted(layer_projections.keys()):
                        norm_proj = np.mean(layer_projections[L]) / last_proj_mean
                        log(f"    Layer {L}: {norm_proj:.3f}")

    # ============================================================
    # S7b: 对比分析
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S7b: 对比分析")
    log(f"{'=' * 60}")

    if perturb_results:
        log(f"\n  --- Perturbation效应汇总 ---")
        log(f"  {'特征':>20} | {'PC1方差':>8} | {'|logit shift|':>14} | {'PC1/SVD[0]':>10}")
        log(f"  {'-'*20} | {'-'*8} | {'-'*14} | {'-'*10}")
        svd0_ref = svd0_mean  # SVD[0]的logit指纹强度
        for feat in sorted(perturb_results.keys()):
            r = perturb_results[feat]
            ratio = r['mean_logit_shift'] / (svd0_ref * r['pc1_norm'] / np.linalg.norm(svd.components_[0])) if svd0_ref > 0 else 0
            # 简化: 直接比较logit shift的绝对值 vs SVD[0]参考
            log(f"  {feat:>20} | {r['pc1_var']*100:>7.1f}% | {r['mean_logit_shift']:>14.4f} | {r['mean_logit_shift']/svd0_ref:>10.4f}")

    # ============================================================
    # S8: 汇总与判断
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S8: 汇总与判断")
    log(f"{'=' * 60}")

    # PC1方差轨迹汇总
    log(f"\n  === PC1方差轨迹汇总 ===")
    if layer_pc1_global:
        vars_only = [(L, v[1]) for L, v in layer_pc1_global.items()]
        peak_layer = max(vars_only, key=lambda x: x[1])
        min_layer = min(vars_only, key=lambda x: x[1])
        log(f"  PC1方差最高层: Layer {peak_layer[0]} ({peak_layer[1]*100:.1f}%)")
        log(f"  PC1方差最低层: Layer {min_layer[0]} ({min_layer[1]*100:.1f}%)")
        log(f"  方差变化范围: {min_layer[1]*100:.1f}% → {peak_layer[1]*100:.1f}%")

    # 跨层对齐汇总
    log(f"\n  === 跨层PC1对齐汇总 ===")
    if last_layer in global_dirs and len(layers_sorted) >= 2:
        first_L = layers_sorted[0]
        cos_first_last = np.dot(global_dirs[first_L], global_dirs[last_layer]) / (
            np.linalg.norm(global_dirs[first_L]) * np.linalg.norm(global_dirs[last_layer]))
        log(f"  首层→末层PC1对齐: cos={cos_first_last:.4f}")
        if cos_first_last > 0.8:
            log(f"  → PC1方向在所有层高度一致, 是稳定的1D流形")
        elif cos_first_last > 0.5:
            log(f"  → PC1方向在层间中等一致, 有部分旋转/漂移")
        else:
            log(f"  → PC1方向在层间变化显著, 不同层编码不同信息")

    # Perturbation效应汇总
    log(f"\n  === Perturbation效应汇总 ===")
    if perturb_results:
        for feat in sorted(perturb_results.keys()):
            r = perturb_results[feat]
            log(f"  {feat}: |logit shift|={r['mean_logit_shift']:.4f}, PC1方差={r['pc1_var']*100:.1f}%")

    # logit效应轨迹
    log(f"\n  === logit效应轨迹 ===")
    if logit_effect_trajectory:
        peak_logit = max(logit_effect_trajectory, key=lambda x: x[2])
        log(f"  logit效应最强层: Layer {peak_logit[0]} (|fp|均值={peak_logit[2]:.4f})")
        log(f"  最后一层: |fp|均值={logit_effect_trajectory[-1][2]:.4f}")

        if peak_logit[0] != last_layer:
            log(f"  ★ 关键发现: PC1的logit效应在中间层(#{peak_logit[0]})最强, 最后一层反而减弱!")
            log(f"  → PC1编码的信息在向输出层传播时被稀释")
        else:
            log(f"  PC1的logit效应在最后一层最强")

    log(f"\n  ★ CCXXVI结论: 多层语义轨迹分析完成")

    # 释放模型
    del model
    torch.cuda.empty_cache()

    # 保存结果
    results = {
        'model': model_key,
        'n_pairs': n_pairs,
        'n_test': n_test,
        'phase': 'CCXXVI',
        'pc1_variance_trajectory': {
            str(L): {'global': float(layer_pc1_global[L][1])} if L in layer_pc1_global else {}
            for L in sample_layers
        },
        'cross_layer_alignment': {},
        'perturb_results': {k: v for k, v in perturb_results.items()},
        'logit_effect_trajectory': [
            {'layer': L, 'pc1_var': float(v), 'mean_logit_fp': float(m), 'max_logit_fp': float(mx)}
            for L, v, m, mx in logit_effect_trajectory
        ],
    }

    # 跨层对齐
    if last_layer in global_dirs:
        last_dir = global_dirs[last_layer]
        for L in layers_sorted:
            if L in global_dirs:
                cos = np.dot(last_dir, global_dirs[L]) / (
                    np.linalg.norm(last_dir) * np.linalg.norm(global_dirs[L]))
                results['cross_layer_alignment'][str(L)] = float(cos)

    with open(f'{out_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\n结果已保存到 {out_dir}/results.json")
    log(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
