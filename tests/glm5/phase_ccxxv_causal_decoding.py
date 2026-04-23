"""
Phase CCXXV: PC1因果解码 — 沿PC1方向perturb, 解码语义内容
================================================================
核心目标:
1. 收集5个特征的PC1方向 (复用CCXXIV逻辑)
2. 沿PC1方向对最后一层hidden state做±δ perturbation
3. 分析perturb后输出概率分布的变化:
   - 哪些token概率增加? 哪些减少?
   - 变化最大的token是否有语义模式?
4. 对比PC1 perturbation vs SVD[k] perturbation的效应
5. 用多组测试句子验证PC1编码的语义一致性

关键假设:
  PC1编码"因果语义方向" — 沿PC1正方向perturb应系统性地
  改变输出token的语义属性(如valence, formality, concreteness)

样本量: 200对/特征 (PC1收集), 50个测试句子 (perturbation验证)
扰动强度: δ = {0.5, 1.0, 2.0} × ||PC1||
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

# 复用特征对定义
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
            ("The moon rises above hills", "The moon rose above hills"),
            ("He draws a circle", "He drew a circle"),
            ("The ice freezes overnight", "The ice froze overnight"),
            ("She holds the baby gently", "She held the baby gently"),
            ("The storm comes suddenly", "The storm came suddenly"),
        ]
    },
    'polarity': {
        'type': 'SYN',
        'pairs': [
            ("The cat is on the mat", "The cat is not on the mat"),
            ("She likes the movie", "She does not like the movie"),
            ("He can swim well", "He cannot swim well"),
            ("They will come tomorrow", "They will not come tomorrow"),
            ("The door was open", "The door was not open"),
            ("She has finished the work", "She has not finished the work"),
            ("He should go now", "He should not go now"),
            ("The answer is correct", "The answer is not correct"),
            ("They are happy", "They are not happy"),
            ("She could find it", "She could not find it"),
            ("He would agree", "He would not agree"),
            ("The plan will work", "The plan will not work"),
            ("She was present", "She was not present"),
            ("They had arrived", "They had not arrived"),
            ("He does know the truth", "He does not know the truth"),
            ("The car is running", "The car is not running"),
            ("She must leave early", "She must not leave early"),
            ("They were invited", "They were not invited"),
            ("He has seen the film", "He has not seen the film"),
            ("The store is open", "The store is not open"),
            ("She can drive a car", "She cannot drive a car"),
            ("He will succeed", "He will not succeed"),
            ("The food was good", "The food was not good"),
            ("They are coming", "They are not coming"),
            ("She was listening", "She was not listening"),
            ("He had finished", "He had not finished"),
            ("The test is easy", "The test is not easy"),
            ("They could win", "They could not win"),
            ("She is working hard", "She is not working hard"),
            ("He was telling the truth", "He was not telling the truth"),
            ("The weather is nice", "The weather is not nice"),
            ("They have enough money", "They do not have enough money"),
            ("She likes coffee", "She does not like coffee"),
            ("He plays guitar", "He does not play guitar"),
            ("The train is on time", "The train is not on time"),
            ("She speaks French", "She does not speak French"),
            ("He remembers the name", "He does not remember the name"),
            ("The room is clean", "The room is not clean"),
            ("They understand the issue", "They do not understand the issue"),
            ("She enjoys reading", "She does not enjoy reading"),
            ("He owns a house", "He does not own a house"),
            ("The book is interesting", "The book is not interesting"),
            ("They support the idea", "They do not support the idea"),
            ("She trusts him", "She does not trust him"),
            ("He believes the story", "He does not believe the story"),
            ("The water is safe", "The water is not safe"),
            ("They need help", "They do not need help"),
            ("She accepts the offer", "She does not accept the offer"),
            ("He deserves praise", "He does not deserve praise"),
        ]
    },
    'voice': {
        'type': 'SYN',
        'pairs': [
            ("The cat chased the mouse", "The mouse was chased by the cat"),
            ("She wrote the letter", "The letter was written by her"),
            ("He fixed the car", "The car was fixed by him"),
            ("They built the house", "The house was built by them"),
            ("The chef cooked the meal", "The meal was cooked by the chef"),
            ("She painted the wall", "The wall was painted by her"),
            ("He opened the door", "The door was opened by him"),
            ("They found the treasure", "The treasure was found by them"),
            ("The police caught the thief", "The thief was caught by the police"),
            ("She sang the song", "The song was sung by her"),
            ("He broke the window", "The window was broken by him"),
            ("They won the prize", "The prize was won by them"),
            ("The wind blew the roof", "The roof was blown by the wind"),
            ("She cut the cake", "The cake was cut by her"),
            ("He drove the bus", "The bus was driven by him"),
            ("They washed the dishes", "The dishes were washed by them"),
            ("The company launched the product", "The product was launched by the company"),
            ("She cleaned the room", "The room was cleaned by her"),
            ("He designed the bridge", "The bridge was designed by him"),
            ("They delivered the package", "The package was delivered by them"),
            ("The teacher graded the paper", "The paper was graded by the teacher"),
            ("She read the book", "The book was read by her"),
            ("He repaired the roof", "The roof was repaired by him"),
            ("They published the article", "The article was published by them"),
            ("The storm destroyed the village", "The village was destroyed by the storm"),
            ("She wrote the report", "The report was written by her"),
            ("He directed the movie", "The movie was directed by him"),
            ("They announced the winner", "The winner was announced by them"),
            ("The referee called the foul", "The foul was called by the referee"),
            ("She typed the letter", "The letter was typed by her"),
            ("He defended the fort", "The fort was defended by him"),
            ("They guided the class", "The class was guided by them"),
            ("The vet treated the animal", "The animal was treated by the vet"),
            ("She steered the plane", "The plane was steered by her"),
            ("He managed the team", "The team was managed by him"),
            ("They approved the budget", "The budget was approved by them"),
            ("The committee reviewed the proposal", "The proposal was reviewed by the committee"),
            ("She investigated the claim", "The claim was investigated by her"),
            ("He published the report", "The report was published by him"),
            ("They issued the guidelines", "The guidelines were issued by them"),
            ("The museum exhibited the painting", "The painting was exhibited by the museum"),
            ("She analyzed the sample", "The sample was analyzed by her"),
            ("He produced the component", "The component was produced by him"),
            ("They broadcast the show", "The show was broadcast by them"),
            ("The firm designed the campaign", "The campaign was designed by the firm"),
            ("She collected the data", "The data was collected by her"),
            ("He transmitted the signal", "The signal was transmitted by him"),
            ("They recalled the device", "The device was recalled by them"),
            ("The authority regulated the industry", "The industry was regulated by the authority"),
        ]
    },
    'semantic_valence': {
        'type': 'SEM',
        'pairs': [
            ("The brave soldier fought", "The cruel soldier fought"),
            ("She gave a generous gift", "She gave a stingy gift"),
            ("The kind teacher helped", "The mean teacher helped"),
            ("He told the honest truth", "He told the deceptive truth"),
            ("The warm fire glowed", "The cold fire glowed"),
            ("She showed genuine love", "She showed fake love"),
            ("The bright sun rose", "The dark sun rose"),
            ("He felt pure joy", "He felt deep sorrow"),
            ("The gentle rain fell", "The violent rain fell"),
            ("She spoke with wisdom", "She spoke with foolishness"),
            ("The sweet fruit ripened", "The bitter fruit ripened"),
            ("He won a glorious victory", "He won a shameful victory"),
            ("The clean water flowed", "The dirty water flowed"),
            ("She gave a warm smile", "She gave a cold smile"),
            ("The peaceful village slept", "The violent village slept"),
            ("He made a noble sacrifice", "He made a cowardly sacrifice"),
            ("The beautiful garden bloomed", "The ugly garden bloomed"),
            ("She felt proud confidence", "She felt shameful doubt"),
            ("The strong hero stood", "The weak hero stood"),
            ("He showed deep compassion", "He showed cruel indifference"),
            ("The pleasant music played", "The harsh music played"),
            ("She earned honest praise", "She earned false praise"),
            ("The safe harbor sheltered", "The dangerous harbor sheltered"),
            ("He achieved great success", "He achieved total failure"),
            ("The rich soil produced", "The barren soil produced"),
            ("She offered warm friendship", "She offered cold hostility"),
            ("The clear sky brightened", "The cloudy sky brightened"),
            ("He found deep peace", "He found deep turmoil"),
            ("The fresh air revived", "The stale air revived"),
            ("She showed tender mercy", "She showed brutal cruelty"),
            ("The lucky winner smiled", "The unfortunate winner smiled"),
            ("He gained valuable experience", "He gained worthless experience"),
            ("The healthy plant thrived", "The sickly plant thrived"),
            ("She kept faithful loyalty", "She kept treacherous betrayal"),
            ("The wise leader decided", "The foolish leader decided"),
            ("He built solid trust", "He built fragile distrust"),
            ("The calm ocean stretched", "The turbulent ocean stretched"),
            ("She expressed genuine gratitude", "She expressed hollow gratitude"),
            ("The pure spirit soared", "The corrupt spirit soared"),
            ("He held firm conviction", "He held weak conviction"),
            ("The bright future awaited", "The bleak future awaited"),
            ("She found lasting happiness", "She found fleeting misery"),
            ("The noble deed inspired", "The vile deed inspired"),
            ("He spoke honest words", "He spoke dishonest words"),
            ("The gentle breeze blew", "The fierce gale blew"),
            ("She maintained quiet dignity", "She maintained loud disgrace"),
            ("The sacred tradition endured", "The profane tradition endured"),
            ("He showed brave resistance", "He showed cowardly submission"),
            ("The beautiful melody echoed", "The ugly melody echoed"),
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

# 测试句子 (用于perturbation验证) — 覆盖多种语义场景
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
    "The story began with",
    "She deeply appreciated the",
    "The fire warmed the",
    "He suddenly realized the",
    "The bird sang a",
    "She gently touched the",
    "The machine produced a",
    "He proudly announced the",
    "The lake was perfectly",
    "She bravely faced the",
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

# 语义分类词表 (用于分析perturbation效应)
VALENCE_POSITIVE = {'good', 'great', 'beautiful', 'happy', 'love', 'wonderful', 'excellent',
    'amazing', 'perfect', 'joy', 'peace', 'kind', 'brave', 'bright', 'warm', 'gentle',
    'sweet', 'pure', 'noble', 'strong', 'safe', 'healthy', 'rich', 'fresh', 'clean',
    'hope', 'smile', 'laugh', 'light', 'sun', 'success', 'win', 'gift', 'dream', 'paradise'}
VALENCE_NEGATIVE = {'bad', 'terrible', 'ugly', 'sad', 'hate', 'awful', 'horrible',
    'worst', 'evil', 'war', 'cruel', 'dark', 'cold', 'harsh', 'bitter', 'corrupt',
    'vile', 'weak', 'dangerous', 'sick', 'poor', 'stale', 'dirty', 'fear', 'cry',
    'sorrow', 'shadow', 'storm', 'fail', 'lose', 'pain', 'nightmare', 'hell'}
FORMAL_WORDS = {'therefore', 'consequently', 'furthermore', 'nevertheless', 'henceforth',
    'accordingly', 'thus', 'moreover', 'notwithstanding', 'hereby', 'hence', 'whereas',
    'aforementioned', 'subsequently', 'therein', 'albeit', 'hitherto', 'whereby'}
INFORMAL_WORDS = {'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'dunno', 'yeah', 'nah',
    'yep', 'nope', 'hey', 'wow', 'oops', 'haha', 'lol', 'omg', 'btw', 'like', 'totally',
    'super', 'really', 'basically', 'literally', 'awesome', 'cool', 'dude', 'stuff'}


def classify_token(token):
    """对token做简单语义分类"""
    t = token.lower().strip()
    categories = []
    if t in VALENCE_POSITIVE:
        categories.append('POSITIVE')
    if t in VALENCE_NEGATIVE:
        categories.append('NEGATIVE')
    if t in FORMAL_WORDS:
        categories.append('FORMAL')
    if t in INFORMAL_WORDS:
        categories.append('INFORMAL')
    # 频率分类 (基于长度和常见模式)
    if len(t) <= 2 and t.isalpha():
        categories.append('SHORT_FUNC')
    if t.startswith('__') or t.startswith('_'):
        categories.append('CODE')
    if any(c > '\u4e00' for c in t):
        categories.append('CJK')
    if any(c > '\u0400' and c < '\u04FF' for c in t):
        categories.append('CYRILLIC')
    if not categories:
        categories.append('OTHER')
    return categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=200)
    parser.add_argument('--n_test', type=int, default=50)
    args = parser.parse_args()

    model_key = args.model
    n_pairs = args.n_pairs
    n_test = args.n_test
    config = MODEL_CONFIGS[model_key]

    out_dir = f'results/causal_fiber/{model_key}_ccxxv'
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
    log(f"Phase CCXXV: PC1因果解码 — {config['name']}")
    log(f"  n_pairs={n_pairs}, n_test={n_test}, 时间={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sklearn.decomposition import PCA, TruncatedSVD

    # ============================================================
    # S1: 加载模型
    # ============================================================
    log(f"\n--- 加载模型 ---")

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
    log(f"  n_layers={n_layers}, d_model={d_model}, vocab_size={vocab_size}, device={next(model.parameters()).device}")

    # ============================================================
    # S2: 收集5个特征的差分向量, 提取PC1方向
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S2: 收集差分向量 → PC1方向")
    log(f"{'=' * 60}")

    pc1_directions = {}  # feature -> pc1_direction (d_model,)
    all_diffs = []  # 收集所有差分用于全局PC1

    for feat in feature_names:
        pairs = FEATURE_PAIRS[feat]['pairs'][:n_pairs]
        diffs = []

        for i, (s_pos, s_neg) in enumerate(pairs):
            try:
                with torch.no_grad():
                    # 正方向
                    toks_pos = tokenizer(s_pos, return_tensors='pt').to(model.device)
                    out_pos = model(**toks_pos, output_hidden_states=True)
                    h_pos = out_pos.hidden_states[last_layer][0, -1, :].float().cpu().numpy()

                    # 负方向
                    toks_neg = tokenizer(s_neg, return_tensors='pt').to(model.device)
                    out_neg = model(**toks_neg, output_hidden_states=True)
                    h_neg = out_neg.hidden_states[last_layer][0, -1, :].float().cpu().numpy()

                    diff = h_pos - h_neg
                    diffs.append(diff)

                    if (i + 1) % 50 == 0:
                        log(f"    {feat}: {i+1}/{len(pairs)}")

                    del out_pos, out_neg
            except Exception as e:
                continue

        if len(diffs) < 10:
            log(f"  {feat}: 仅{len(diffs)}对有效, 跳过")
            continue

        diffs = np.array(diffs)
        pca = PCA(n_components=1)
        pca.fit(diffs)
        pc1 = pca.components_[0]  # (d_model,)
        pc1_var = pca.explained_variance_ratio_[0]

        pc1_directions[feat] = pc1
        all_diffs.extend(diffs.tolist())
        log(f"  {feat}: PC1方差={pc1_var*100:.1f}%")

    # 全局PC1
    if len(all_diffs) >= 10:
        all_diffs = np.array(all_diffs)
        pca_global = PCA(n_components=1)
        pca_global.fit(all_diffs)
        pc1_global = pca_global.components_[0]
        global_var = pca_global.explained_variance_ratio_[0]
        pc1_directions['global'] = pc1_global
        log(f"\n  全局PC1方差={global_var*100:.1f}%")

    # ============================================================
    # S3: 提取lm_head权重, 计算PC1的logit指纹
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S3: lm_head权重 → PC1 logit指纹")
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
            # lm_head.weight 形状: (vocab, d_model) 或 (d_model, vocab)
            # 我们需要 W_full 的形状为 (vocab, d_model), 即 logit = W_full @ h
            if W.shape[1] == d_model:
                W_full = W  # (vocab, d_model) — 标准形状
            elif W.shape[0] == d_model:
                W_full = W.T  # (d_model, vocab) → 转置为 (vocab, d_model)
            else:
                log(f"  警告: W形状{W.shape}不匹配d_model={d_model}, 尝试使用")
                W_full = W
    except Exception as e:
        log(f"  从model.lm_head提取失败: {e}")

    # 如果从模型中提取失败，尝试safetensors
    if W_full is None:
        log("  尝试从safetensors加载...")
        import glob
        try:
            from safetensors.torch import load_file as st_load_file
            safetensor_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
            for sf in safetensor_files:
                try:
                    # 先检查文件是否包含lm_head
                    from safetensors import safe_open as _so
                    with _so(sf, framework="pt") as f:
                        keys = list(f.keys())
                    if not any('lm_head.weight' in k for k in keys):
                        continue
                    log(f"    加载 {os.path.basename(sf)} (含lm_head)...")
                    # 只加载需要的tensor
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
                    log(f"    读取{os.path.basename(sf)}失败: {e2}")
                    continue
                if W_full is not None:
                    break
        except ImportError:
            log("  safetensors未安装, 跳过")

    if W_full is None:
        log("  错误: 无法加载lm_head权重!")
        return

    log(f"  lm_head权重形状: {W_full.shape}")

    # 计算PC1的logit指纹: W @ pc1_dir
    logit_fingerprints = {}
    for feat, pc1_dir in pc1_directions.items():
        fp = W_full @ pc1_dir  # (vocab,)
        logit_fingerprints[feat] = fp

    # 也提取SVD前10方向的logit指纹
    svd = TruncatedSVD(n_components=10)
    svd.fit(W_full)

    svd_fingerprints = {}
    for k in range(10):
        fp = W_full @ svd.components_[k]
        svd_fingerprints[k] = fp

    # ============================================================
    # S4: PC1 logit指纹分析 — PC1编码了什么语义?
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S4: PC1 logit指纹分析")
    log(f"{'=' * 60}")

    # 获取token频率 (用W的行范数近似)
    token_norms = np.linalg.norm(W_full, axis=1)
    token_freq_rank = np.argsort(-token_norms)  # 降序

    for feat in ['tense', 'polarity', 'voice', 'semantic_valence', 'semantic_topic', 'global']:
        if feat not in logit_fingerprints:
            continue

        fp = logit_fingerprints[feat]
        top_pos_idx = np.argsort(-fp)[:30]
        top_neg_idx = np.argsort(fp)[:30]

        log(f"\n  {feat} PC1 logit指纹:")
        log(f"    正方向 (logit增加) Top-20:")
        pos_cats = Counter()
        for idx in top_pos_idx[:20]:
            token = tokenizer.decode([idx])
            cats = classify_token(token)
            pos_cats.update(cats)
            log(f"      '{token}' → {fp[idx]:.3f} [{','.join(cats)}]")

        log(f"    负方向 (logit减少) Top-20:")
        neg_cats = Counter()
        for idx in top_neg_idx[:20]:
            token = tokenizer.decode([idx])
            cats = classify_token(token)
            neg_cats.update(cats)
            log(f"      '{token}' → {fp[idx]:.3f} [{','.join(cats)}]")

        # 语义类别统计
        log(f"    语义类别统计:")
        log(f"      正方向: {dict(pos_cats.most_common(5))}")
        log(f"      负方向: {dict(neg_cats.most_common(5))}")

        # 频率分析
        pos_freq_ranks = [np.where(token_freq_rank == idx)[0][0] for idx in top_pos_idx[:30]]
        neg_freq_ranks = [np.where(token_freq_rank == idx)[0][0] for idx in top_neg_idx[:30]]
        log(f"    频率排名: 正方向P50={np.median(pos_freq_ranks):.0f}/{vocab_size}, 负方向P50={np.median(neg_freq_ranks):.0f}/{vocab_size}")

        # Valence分析: 正方向token中正面vs负面词的比例
        pos_valence = sum(1 for idx in top_pos_idx[:30] for c in classify_token(tokenizer.decode([idx])) if c in ('POSITIVE', 'NEGATIVE'))
        neg_valence = sum(1 for idx in top_neg_idx[:30] for c in classify_token(tokenizer.decode([idx])) if c in ('POSITIVE', 'NEGATIVE'))
        if pos_valence + neg_valence > 0:
            log(f"    Valence信号: 正方向情感词{pos_valence}个, 负方向情感词{neg_valence}个")

    # ============================================================
    # S5: 因果Perturbation验证 — 核心实验
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S5: 因果Perturbation验证")
    log(f"{'=' * 60}")

    deltas = [0.5, 1.0, 2.0]
    perturb_results = defaultdict(lambda: defaultdict(list))  # feat -> delta -> list of results

    test_sentences = TEST_SENTENCES[:n_test]

    for feat in ['tense', 'polarity', 'voice', 'semantic_valence', 'semantic_topic', 'global']:
        if feat not in pc1_directions:
            continue

        pc1_dir = pc1_directions[feat]
        pc1_norm = np.linalg.norm(pc1_dir)
        pc1_torch = torch.tensor(pc1_dir, dtype=torch.float32).to(model.device)

        log(f"\n  --- {feat} PC1 perturbation (||PC1||={pc1_norm:.2f}) ---")

        for delta in deltas:
            delta_scale = delta * pc1_norm
            n_analyzed = 0
            all_top_changes = []  # 收集所有句子的top变化token
            all_logit_shifts = []  # 收集logit shift统计

            for sent in test_sentences:
                try:
                    with torch.no_grad():
                        # 基线
                        toks = tokenizer(sent, return_tensors='pt').to(model.device)
                        out = model(**toks, output_hidden_states=True)
                        h_last = out.hidden_states[last_layer][0, -1, :].clone()

                        # 基线logits
                        if W_full is not None:
                            h_baseline = h_last.float().cpu().numpy()
                            logits_baseline = W_full @ h_baseline
                        else:
                            logits_baseline = out.logits[0, -1, :].float().cpu().numpy()

                        # 正方向perturb
                        h_perturbed = h_last + delta_scale * pc1_torch
                        h_pert_np = h_perturbed.float().cpu().numpy()
                        logits_pos = W_full @ h_pert_np

                        # 负方向perturb
                        h_perturbed_neg = h_last - delta_scale * pc1_torch
                        h_pert_neg_np = h_perturbed_neg.float().cpu().numpy()
                        logits_neg = W_full @ h_pert_neg_np

                        # logit变化
                        delta_logits_pos = logits_pos - logits_baseline
                        delta_logits_neg = logits_neg - logits_baseline

                        # Top变化的token
                        top_increase = np.argsort(-delta_logits_pos)[:10]
                        top_decrease = np.argsort(delta_logits_pos)[:10]

                        for idx in top_increase[:5]:
                            token = tokenizer.decode([int(idx)])
                            all_top_changes.append(('INC', token, float(delta_logits_pos[idx])))
                        for idx in top_decrease[:5]:
                            token = tokenizer.decode([int(idx)])
                            all_top_changes.append(('DEC', token, float(delta_logits_pos[idx])))

                        # 统计logit shift的幅度
                        all_logit_shifts.append({
                            'mean_abs': float(np.mean(np.abs(delta_logits_pos))),
                            'max': float(np.max(delta_logits_pos)),
                            'min': float(np.min(delta_logits_pos)),
                            'std': float(np.std(delta_logits_pos)),
                        })

                        n_analyzed += 1
                        del out

                except Exception as e:
                    continue

            if n_analyzed == 0:
                log(f"    δ={delta}: 无有效句子")
                continue

            # 汇总perturbation结果
            log(f"    δ={delta} ({n_analyzed}个句子):")

            # 平均logit shift统计
            mean_abs = np.mean([s['mean_abs'] for s in all_logit_shifts])
            mean_max = np.mean([s['max'] for s in all_logit_shifts])
            mean_min = np.mean([s['min'] for s in all_logit_shifts])
            log(f"      平均|logit shift|={mean_abs:.4f}, max={mean_max:.4f}, min={mean_min:.4f}")

            # Top变化token统计
            inc_tokens = [(t, s) for d, t, s in all_top_changes if d == 'INC']
            dec_tokens = [(t, s) for d, t, s in all_top_changes if d == 'DEC']

            # 按token聚合
            inc_counter = Counter()
            dec_counter = Counter()
            for t, s in inc_tokens:
                inc_counter[t] += 1
            for t, s in dec_tokens:
                dec_counter[t] += 1

            log(f"      正方向最常增加的token: {inc_counter.most_common(10)}")
            log(f"      正方向最常减少的token: {dec_counter.most_common(10)}")

            # 语义类别分析
            inc_cats = Counter()
            dec_cats = Counter()
            for t, s in inc_tokens:
                cats = classify_token(t)
                inc_cats.update(cats)
            for t, s in dec_tokens:
                cats = classify_token(t)
                dec_cats.update(cats)

            log(f"      正方向增加的语义类别: {dict(inc_cats.most_common(5))}")
            log(f"      正方向减少的语义类别: {dict(dec_cats.most_common(5))}")

            perturb_results[feat][delta] = {
                'n_analyzed': n_analyzed,
                'mean_abs_shift': mean_abs,
                'inc_cats': dict(inc_cats.most_common(5)),
                'dec_cats': dict(dec_cats.most_common(5)),
                'inc_top5': inc_counter.most_common(5),
                'dec_top5': dec_counter.most_common(5),
            }

    # ============================================================
    # S6: PC1 vs SVD[k] perturbation对比
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S6: PC1 vs SVD perturbation对比")
    log(f"{'=' * 60}")

    # 对比PC1_global vs SVD[0] vs SVD[1]的perturbation效应
    if 'global' in pc1_directions:
        compare_dirs = {
            'PC1_global': pc1_directions['global'],
        }
        for k in [0, 1]:
            if k in svd_fingerprints:
                # SVD方向就是svd.components_[k]
                compare_dirs[f'SVD[{k}]'] = svd.components_[k]

        delta = 1.0  # 固定delta
        n_compare = min(30, len(test_sentences))

        for dir_name, dir_vec in compare_dirs.items():
            dir_norm = np.linalg.norm(dir_vec)
            dir_torch = torch.tensor(dir_vec, dtype=torch.float32).to(model.device)
            delta_scale = delta * dir_norm

            all_shifts = []
            inc_cats_all = Counter()
            dec_cats_all = Counter()

            for sent in test_sentences[:n_compare]:
                try:
                    with torch.no_grad():
                        toks = tokenizer(sent, return_tensors='pt').to(model.device)
                        out = model(**toks, output_hidden_states=True)
                        h_last = out.hidden_states[last_layer][0, -1, :].clone()

                        h_baseline = h_last.float().cpu().numpy()
                        logits_baseline = W_full @ h_baseline

                        h_perturbed = h_last + delta_scale * dir_torch
                        h_pert_np = h_perturbed.float().cpu().numpy()
                        logits_pert = W_full @ h_pert_np

                        delta_logits = logits_pert - logits_baseline

                        top_inc = np.argsort(-delta_logits)[:10]
                        top_dec = np.argsort(delta_logits)[:10]

                        for idx in top_inc[:5]:
                            token = tokenizer.decode([int(idx)])
                            cats = classify_token(token)
                            inc_cats_all.update(cats)
                        for idx in top_dec[:5]:
                            token = tokenizer.decode([int(idx)])
                            cats = classify_token(token)
                            dec_cats_all.update(cats)

                        all_shifts.append(float(np.mean(np.abs(delta_logits))))
                        del out
                except:
                    continue

            if all_shifts:
                log(f"\n  {dir_name} (||dir||={dir_norm:.2f}, δ={delta}):")
                log(f"    平均|logit shift|={np.mean(all_shifts):.4f}")
                log(f"    增加的语义类别: {dict(inc_cats_all.most_common(5))}")
                log(f"    减少的语义类别: {dict(dec_cats_all.most_common(5))}")

    # ============================================================
    # S7: 汇总与判断
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S7: 汇总与判断")
    log(f"{'=' * 60}")

    # PC1 perturbation效应总结
    log(f"\n  === PC1 Perturbation效应总结 ===")
    for feat in ['tense', 'polarity', 'voice', 'semantic_valence', 'semantic_topic', 'global']:
        if feat not in perturb_results:
            continue
        log(f"  {feat}:")
        for delta in sorted(perturb_results[feat].keys()):
            r = perturb_results[feat][delta]
            log(f"    δ={delta}: |shift|={r['mean_abs_shift']:.4f}, INC类别={r['inc_cats']}, DEC类别={r['dec_cats']}")

    # 关键判断
    log(f"\n  === 关键判断 ===")

    # PC1 logit指纹的语义一致性
    log(f"  PC1 logit指纹分析:")
    for feat in ['global', 'semantic_valence', 'tense', 'polarity']:
        if feat not in logit_fingerprints:
            continue
        fp = logit_fingerprints[feat]
        top_pos_idx = np.argsort(-fp)[:30]
        top_neg_idx = np.argsort(fp)[:30]

        pos_cats = Counter()
        neg_cats = Counter()
        for idx in top_pos_idx:
            cats = classify_token(tokenizer.decode([int(idx)]))
            pos_cats.update(cats)
        for idx in top_neg_idx:
            cats = classify_token(tokenizer.decode([int(idx)]))
            neg_cats.update(cats)

        log(f"    {feat}: 正→{dict(pos_cats.most_common(3))}, 负→{dict(neg_cats.most_common(3))}")

    # PC1 vs SVD perturbation对比
    if 'global' in pc1_directions:
        pc1_logit_effect = np.mean(np.abs(logit_fingerprints['global']))
        svd0_logit_effect = np.mean(np.abs(svd_fingerprints[0]))
        svd1_logit_effect = np.mean(np.abs(svd_fingerprints[1]))

        log(f"\n  logit指纹效应强度:")
        log(f"    PC1_global: |fp|均值={pc1_logit_effect:.4f}")
        log(f"    SVD[0]:     |fp|均值={svd0_logit_effect:.4f}")
        log(f"    SVD[1]:     |fp|均值={svd1_logit_effect:.4f}")
        log(f"    PC1/SVD[0]比值: {pc1_logit_effect/svd0_logit_effect:.3f}")

    log(f"\n  ★ 结论: PC1因果解码完成, 基于perturbation效应判断PC1编码的语义方向")

    # 释放模型
    del model
    torch.cuda.empty_cache()

    # 保存结果
    results = {
        'model': model_key,
        'n_pairs': n_pairs,
        'n_test': n_test,
        'phase': 'CCXXV',
        'perturb_summary': {},
    }

    for feat in perturb_results:
        results['perturb_summary'][feat] = {}
        for delta in perturb_results[feat]:
            r = perturb_results[feat][delta]
            results['perturb_summary'][feat][str(delta)] = {
                'n_analyzed': r['n_analyzed'],
                'mean_abs_shift': r['mean_abs_shift'],
                'inc_cats': r['inc_cats'],
                'dec_cats': r['dec_cats'],
            }

    with open(f'{out_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\n结果已保存到 {out_dir}/results.json")
    log(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
