"""
Phase CCXXVII: PC1旋转结构分解与语义汇聚层解码
================================================================
核心目标:
1. 分解相邻层PC1的旋转结构: 旋转角度θ, 缩放因子s
2. 在PC1方差峰值层(语义汇聚层)解码PC1编码的语义
3. 对比: 峰值层PC1 perturbation vs 最后一层PC1 perturbation
4. 验证: "PC1旋转是否有规律性" vs "随机旋转"

关键假设:
  CCXXVI发现PC1跨层剧烈旋转(cos≈0), 且logit效应在中间层最强。
  如果PC1旋转有规律性(如恒定角速度旋转), 则1D流形可能是
  某种螺旋结构在hidden state空间中的投影。
  如果PC1旋转无规律, 则1D流形可能是不同层独立编码不同语义维度。

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
            ("She teaches the children", "She taught the children"),
            ("The storm destroys the house", "The storm destroyed the house"),
            ("He holds the baby carefully", "He held the baby carefully"),
            ("The water freezes in winter", "The water froze in winter"),
            ("She knows the answer", "She knew the answer"),
            ("The plane takes off smoothly", "The plane took off smoothly"),
            ("He keeps the secret safe", "He kept the secret safe"),
            ("The music plays softly", "The music played softly"),
            ("She thinks about the future", "She thought about the future"),
            ("The tree grows tall", "The tree grew tall"),
            ("He meets his friends", "He met his friends"),
            ("The lake shines in the moonlight", "The lake shone in the moonlight"),
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
            ("The coffee is too hot", "The coffee is too cold"),
            ("He was early for the meeting", "He was late for the meeting"),
            ("The room was spacious", "The room was cramped"),
            ("She accepted the offer", "She rejected the offer"),
            ("The crowd was supportive", "The crowd was hostile"),
            ("He fixed the broken toy", "He broke the working toy"),
            ("The dog was friendly", "The dog was aggressive"),
            ("She included everyone", "She excluded everyone"),
            ("The road was safe", "The road was dangerous"),
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
            ("The police arrested the suspect", "The suspect was arrested by the police"),
            ("He delivered the package", "The package was delivered by him"),
            ("She watered the plants", "The plants were watered by her"),
            ("They repaired the bridge", "The bridge was repaired by them"),
            ("He painted the ceiling", "The ceiling was painted by him"),
            ("The doctor examined the patient", "The patient was examined by the doctor"),
            ("She cleaned the window", "The window was cleaned by her"),
            ("He built the cabinet", "The cabinet was built by him"),
            ("The wind blew the leaves", "The leaves were blown by the wind"),
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
            ("She nurtured young seedlings", "She neglected young seedlings"),
            ("The gentle wave lapped", "The savage wave crashed"),
            ("He welcomed the stranger", "He rejected the stranger"),
            ("The warm fire glowed", "The cold ice froze"),
            ("She preserved the treasure", "She squandered the treasure"),
            ("The clear sky inspired hope", "The murky fog inspired dread"),
            ("He protected the weak", "He exploited the weak"),
            ("The lively music energized", "The somber music drained"),
            ("She enriched the community", "She impoverished the community"),
            ("The noble cause united", "The selfish motive divided"),
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
            ("The physicist measured the wavelength", "The tailor measured the inseam"),
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

    out_dir = f'results/causal_fiber/{model_key}_ccxxvii'
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
    log(f"Phase CCXXVII: PC1旋转结构分解与语义汇聚层解码 — {config['name']}")
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
    # S2: 逐层收集差分向量 → 每层PC1 (密集采样)
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S2: 逐层PC1收集 (密集采样)")
    log(f"{'=' * 60}")

    # 密集采样: 每层都做 (为了精确计算旋转角)
    all_layers = list(range(n_layers))
    # 但太多层会很慢, 用每2层采样
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

    # 计算每层PC1
    layer_pc1_global = {}  # L -> (pc1_dir, pc1_var, pca_object)
    layer_pc1_feat = {}  # (feat, L) -> (pc1_dir, pc1_var)

    for L in sample_layers:
        # 全局
        diffs = all_layer_diffs[L]
        if len(diffs) >= 20:
            diffs = np.array(diffs)
            pca = PCA(n_components=3)  # 取前3个主成分
            pca.fit(diffs)
            layer_pc1_global[L] = (pca.components_[0], pca.explained_variance_ratio_[0], pca)

        # 特征级
        for feat in feature_names:
            fds = feat_layer_diffs[feat][L]
            if len(fds) >= 10:
                fds = np.array(fds)
                pca_f = PCA(n_components=1)
                pca_f.fit(fds)
                layer_pc1_feat[(feat, L)] = (pca_f.components_[0], pca_f.explained_variance_ratio_[0])

    # 找到PC1方差峰值层
    if layer_pc1_global:
        peak_layer = max(layer_pc1_global.keys(), key=lambda L: layer_pc1_global[L][1])
        peak_var = layer_pc1_global[peak_layer][1]
        log(f"\n  全局PC1方差峰值层: L{peak_layer} ({peak_var*100:.1f}%)")
        log(f"  最后一层: L{last_layer} ({layer_pc1_global.get(last_layer, (None, 0))[1]*100:.1f}%)")

    # ============================================================
    # S3: PC1旋转结构分析 — 核心实验
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S3: PC1旋转结构分析")
    log(f"{'=' * 60}")

    global_dirs = {L: v[0] for L, v in layer_pc1_global.items()}
    layers_sorted = sorted(global_dirs.keys())

    # 3a: 相邻层旋转角度
    log(f"\n  --- 相邻层PC1旋转分析 ---")
    rotation_data = []

    for i in range(len(layers_sorted) - 1):
        L1, L2 = layers_sorted[i], layers_sorted[i + 1]
        d1, d2 = global_dirs[L1], global_dirs[L2]

        # cos角度
        cos_val = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
        cos_val = np.clip(cos_val, -1, 1)
        theta = np.arccos(abs(cos_val))  # 取绝对值消除符号不确定性
        theta_deg = np.degrees(theta)

        # 符号
        sign = np.sign(cos_val) if abs(cos_val) > 0.01 else 0

        # 缩放因子: 用PC1方差的比值
        var1 = layer_pc1_global[L1][1]
        var2 = layer_pc1_global[L2][1]
        scale = np.sqrt(var2 / var1) if var1 > 0 else 0

        rotation_data.append({
            'L1': L1, 'L2': L2,
            'cos': float(cos_val), 'theta_deg': float(theta_deg),
            'sign': int(sign), 'scale': float(scale),
            'var1': float(var1), 'var2': float(var2),
        })

        log(f"    L{L1}→L{L2}: cos={cos_val:+.4f}, θ={theta_deg:.1f}°, sign={'+' if sign >= 0 else '-'}, scale={scale:.3f}")

    # 3b: 旋转统计
    thetas = [r['theta_deg'] for r in rotation_data]
    signs = [r['sign'] for r in rotation_data]
    scales = [r['scale'] for r in rotation_data]

    log(f"\n  --- 旋转统计 ---")
    log(f"    旋转角θ: 均值={np.mean(thetas):.1f}°, 中位数={np.median(thetas):.1f}°, 标准差={np.std(thetas):.1f}°")
    log(f"    旋转角范围: [{np.min(thetas):.1f}°, {np.max(thetas):.1f}°]")
    log(f"    符号翻转次数: {sum(1 for s in signs if s < 0)} / {len(signs)}")
    log(f"    缩放因子: 均值={np.mean(scales):.3f}, 范围=[{np.min(scales):.3f}, {np.max(scales):.3f}]")

    # 3c: 旋转是否有规律性? — 与恒定角速度旋转对比
    log(f"\n  --- 旋转规律性检验 ---")

    # 如果是恒定角速度旋转, 累积角度应线性增长
    cumulative_theta = [0]
    for r in rotation_data:
        cumulative_theta.append(cumulative_theta[-1] + r['theta_deg'])

    log(f"    累积旋转角: {cumulative_theta[-1]:.1f}° (={cumulative_theta[-1]/360:.2f}圈)")
    log(f"    每层平均旋转: {cumulative_theta[-1]/len(rotation_data):.1f}°")

    # 线性度: 累积角度与理想直线的R²
    x = np.arange(len(cumulative_theta))
    y = np.array(cumulative_theta)
    if len(x) > 2:
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        log(f"    线性度R²={r_squared:.4f} (1.0=完美恒定角速度)")

    # 3d: PC1→PC2旋转 — 在PC1-PC2平面上的旋转
    log(f"\n  --- PC1-PC2平面旋转分析 ---")
    # 检查L层的PC2是否对齐L+1层的PC1
    for i in range(min(5, len(layers_sorted) - 1)):
        L1, L2 = layers_sorted[i], layers_sorted[i + 1]
        pca1 = layer_pc1_global[L1][2]  # PCA object with 3 components
        pc1_L1 = pca1.components_[0]
        pc2_L1 = pca1.components_[1]

        pc1_L2 = global_dirs[L2]

        # cos(pc2_L1, pc1_L2)
        cos_pc2_pc1 = np.dot(pc2_L1, pc1_L2) / (np.linalg.norm(pc2_L1) * np.linalg.norm(pc1_L2))
        # cos(pc1_L1, pc1_L2)
        cos_pc1_pc1 = np.dot(pc1_L1, pc1_L2) / (np.linalg.norm(pc1_L1) * np.linalg.norm(pc1_L2))

        log(f"    L{L1}→L{L2}: cos(PC1_L1, PC1_L2)={cos_pc1_pc1:+.4f}, cos(PC2_L1, PC1_L2)={cos_pc2_pc1:+.4f}")

    # ============================================================
    # S4: 语义汇聚层(峰值层)PC1解码
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S4: 语义汇聚层(L{peak_layer})PC1解码")
    log(f"{'=' * 60}")

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
                            else:
                                W_full = W_raw
                            break
                    del state_dict
                except:
                    continue
                if W_full is not None:
                    break
        except ImportError:
            pass

    if W_full is None:
        log("  错误: 无法加载lm_head权重!")
        return

    log(f"  lm_head权重: {W_full.shape}")

    # SVD参考
    svd = TruncatedSVD(n_components=10)
    svd.fit(W_full)
    svd0_logit = W_full @ svd.components_[0]
    svd0_mean = float(np.mean(np.abs(svd0_logit)))

    # 峰值层 vs 最后一层 PC1的logit指纹对比
    log(f"\n  --- 峰值层 vs 最后一层 PC1 logit指纹 ---")

    for L_label, L in [('峰值层', peak_layer), ('最后一层', last_layer)]:
        if L not in layer_pc1_global:
            continue
        pc1_dir = layer_pc1_global[L][0]
        pc1_var = layer_pc1_global[L][1]

        logit_fp = W_full @ pc1_dir
        mean_abs = float(np.mean(np.abs(logit_fp)))
        max_abs = float(np.max(np.abs(logit_fp)))

        # Top tokens
        top_pos_idx = np.argsort(-logit_fp)[:15]
        top_neg_idx = np.argsort(logit_fp)[:15]

        log(f"\n  {L_label}(L{L}) PC1 (方差={pc1_var*100:.1f}%):")
        log(f"    |logit_fp|均值={mean_abs:.4f}, max={max_abs:.4f}, /SVD[0]={mean_abs/svd0_mean:.4f}")
        log(f"    正方向Top-10:")
        for idx in top_pos_idx[:10]:
            token = tokenizer.decode([int(idx)])
            log(f"      '{token}' → {logit_fp[idx]:.3f}")
        log(f"    负方向Top-10:")
        for idx in top_neg_idx[:10]:
            token = tokenizer.decode([int(idx)])
            log(f"      '{token}' → {logit_fp[idx]:.3f}")

    # 特征级PC1在峰值层的解码
    log(f"\n  --- 特征PC1在峰值层(L{peak_layer})的logit指纹 ---")
    for feat in feature_names:
        if (feat, peak_layer) not in layer_pc1_feat:
            continue
        pc1_dir, pc1_var = layer_pc1_feat[(feat, peak_layer)]
        logit_fp = W_full @ pc1_dir
        mean_abs = float(np.mean(np.abs(logit_fp)))

        top_pos_idx = np.argsort(-logit_fp)[:5]
        top_neg_idx = np.argsort(logit_fp)[:5]
        pos_tokens = [tokenizer.decode([int(idx)]).strip() for idx in top_pos_idx]
        neg_tokens = [tokenizer.decode([int(idx)]).strip() for idx in top_neg_idx]

        log(f"    {feat}: var={pc1_var*100:.1f}%, |fp|={mean_abs:.4f}, /SVD[0]={mean_abs/svd0_mean:.3f}, pos={pos_tokens[:3]}, neg={neg_tokens[:3]}")

    # ============================================================
    # S5: 峰值层 vs 最后一层 Perturbation对比
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S5: 峰值层 vs 最后一层 Perturbation对比")
    log(f"{'=' * 60}")

    delta = 1.0
    test_sentences = TEST_SENTENCES[:n_test]
    perturb_compare = {}

    for L_label, L in [('峰值层', peak_layer), ('最后一层', last_layer)]:
        if L not in layer_pc1_global:
            continue
        pc1_dir = layer_pc1_global[L][0]
        pc1_var = layer_pc1_global[L][1]
        pc1_norm = np.linalg.norm(pc1_dir)
        pc1_torch = torch.tensor(pc1_dir, dtype=torch.float32).to(model.device)

        # 只在最后一层hidden state上perturb (公平对比)
        delta_scale = delta * pc1_norm
        all_logit_shifts = []

        for sent in test_sentences:
            try:
                with torch.no_grad():
                    toks = tokenizer(sent, return_tensors='pt').to(model.device)
                    out = model(**toks, output_hidden_states=True)
                    h_last = out.hidden_states[last_layer][0, -1, :].clone()

                    h_baseline = h_last.float().cpu().numpy()
                    logits_baseline = W_full @ h_baseline

                    # 用L层的PC1方向perturb最后一层
                    h_perturbed = h_last + delta_scale * pc1_torch
                    h_pert_np = h_perturbed.float().cpu().numpy()
                    logits_pert = W_full @ h_pert_np

                    shift = float(np.mean(np.abs(logits_pert - logits_baseline)))
                    all_logit_shifts.append(shift)
                    del out
            except:
                continue

        if all_logit_shifts:
            perturb_compare[L_label] = {
                'layer': L,
                'mean_logit_shift': float(np.mean(all_logit_shifts)),
                'pc1_var': float(pc1_var),
                'pc1_norm': float(pc1_norm),
            }
            log(f"  {L_label}(L{L}): |logit shift|={np.mean(all_logit_shifts):.4f}, PC1方差={pc1_var*100:.1f}%")

    # 对比
    if '峰值层' in perturb_compare and '最后一层' in perturb_compare:
        peak_shift = perturb_compare['峰值层']['mean_logit_shift']
        last_shift = perturb_compare['最后一层']['mean_logit_shift']
        ratio = peak_shift / last_shift if last_shift > 0 else float('inf')
        log(f"\n  ★ 峰值层/最后一层 logit效应比: {ratio:.2f}x")
        if ratio > 1.5:
            log(f"  → 峰值层PC1的logit效应显著强于最后一层!")
        elif ratio < 0.67:
            log(f"  → 最后一层PC1的logit效应显著强于峰值层")
        else:
            log(f"  → 两层PC1的logit效应相当")

    # ============================================================
    # S6: PC1旋转与PCA次成分的关系
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S6: PC1旋转结构 — 是否是子空间内的旋转?")
    log(f"{'=' * 60}")

    # 检查: 相邻层的PC1是否在前3个主成分构成的子空间内旋转
    log(f"\n  --- 子空间包含性检验 ---")
    for i in range(min(8, len(layers_sorted) - 1)):
        L1, L2 = layers_sorted[i], layers_sorted[i + 1]
        pca1 = layer_pc1_global[L1][2]  # 3-component PCA

        # 将L2的PC1投影到L1的前3个主成分
        pc1_L2 = global_dirs[L2]
        components_L1 = pca1.components_  # (3, d_model)

        # 投影
        proj = components_L1 @ pc1_L2  # (3,)
        residual = pc1_L2 - components_L1.T @ proj
        residual_norm = np.linalg.norm(residual)
        pc1_norm = np.linalg.norm(pc1_L2)
        containment = 1 - (residual_norm / pc1_norm) ** 2

        log(f"    L{L1}→L{L2}: L1前3PC对L2-PC1的包含度={containment:.4f} (1.0=完全包含)")

    # ============================================================
    # S7: 汇总
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S7: 汇总与判断")
    log(f"{'=' * 60}")

    # 旋转结构汇总
    log(f"\n  === 旋转结构汇总 ===")
    log(f"  总旋转角: {cumulative_theta[-1]:.1f}° ({cumulative_theta[-1]/360:.2f}圈)")
    log(f"  平均层间旋转: {np.mean(thetas):.1f}°")
    log(f"  旋转角标准差: {np.std(thetas):.1f}°")
    if len(x) > 2:
        log(f"  恒定角速度线性度R²: {r_squared:.4f}")
    log(f"  符号翻转率: {sum(1 for s in signs if s < 0)}/{len(signs)} = {sum(1 for s in signs if s < 0)/len(signs)*100:.0f}%")

    # 旋转规律性判断
    if r_squared > 0.9:
        log(f"  → PC1旋转高度规律(接近恒定角速度), 可能是螺旋结构")
    elif r_squared > 0.7:
        log(f"  → PC1旋转中等规律, 有加速/减速区域")
    else:
        log(f"  → PC1旋转不规律, 不同层段旋转速度差异大")

    # 峰值层vs最后一层
    log(f"\n  === 峰值层解码汇总 ===")
    if '峰值层' in perturb_compare and '最后一层' in perturb_compare:
        log(f"  峰值层(L{peak_layer}): PC1方差={perturb_compare['峰值层']['pc1_var']*100:.1f}%, logit效应={perturb_compare['峰值层']['mean_logit_shift']:.4f}")
        log(f"  最后一层(L{last_layer}): PC1方差={perturb_compare['最后一层']['pc1_var']*100:.1f}%, logit效应={perturb_compare['最后一层']['mean_logit_shift']:.4f}")

    # 关键判断
    log(f"\n  === 关键判断 ===")
    log(f"  1. PC1旋转是否规律: R²={r_squared:.3f}")
    log(f"  2. 峰值层PC1 logit效应是否更强: 峰值/末层={ratio:.2f}x")
    log(f"  3. 1D流形本质: {'螺旋结构' if r_squared > 0.8 else '非均匀旋转' if r_squared > 0.5 else '分段线性/不规则'}")

    # 释放模型
    del model
    torch.cuda.empty_cache()

    # 保存结果
    results = {
        'model': model_key,
        'n_pairs': n_pairs,
        'n_test': n_test,
        'phase': 'CCXXVII',
        'peak_layer': int(peak_layer),
        'peak_var': float(peak_var),
        'rotation_summary': {
            'total_rotation_deg': float(cumulative_theta[-1]),
            'mean_theta_deg': float(np.mean(thetas)),
            'std_theta_deg': float(np.std(thetas)),
            'linearity_r_squared': float(r_squared) if len(x) > 2 else 0,
            'sign_flip_rate': float(sum(1 for s in signs if s < 0) / len(signs)),
        },
        'perturb_compare': perturb_compare,
        'rotation_data': rotation_data,
    }

    with open(f'{out_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\n结果已保存到 {out_dir}/results.json")
    log(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
