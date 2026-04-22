"""
Phase CCXV: 高效因果原子分解 (基于差分向量+W_o投影)
===================================================
S1: 差分向量收集 (每个特征n_pairs对, 5层)
S2: PCA原子分解 + 子空间正交性
S3: W_o Head贡献分解 (不需要额外forward)
S4: 逐特征PCA分离度 + 因果原子发现
S5: 统计检验

优化: 不做逐head patching(太慢), 改用W_o投影分析head贡献
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from scipy import stats

import torch

# ============================================================
# 模型配置
# ============================================================
MODEL_CONFIGS = {
    'deepseek7b': {
        'path': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'name': 'DeepSeek-R1-Distill-Qwen-7B',
    },
    'qwen3': {
        'path': 'D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',
        'name': 'Qwen3-4B',
    },
    'glm4': {
        'path': 'D:/develop/model/hub/models--zai-org--glm-4-9b-chat-hf/download',
        'name': 'GLM4-9B-Chat',
    },
}

# ============================================================
# 12特征的Minimal Pair定义 (每特征30对)
# ============================================================
FEATURE_PAIRS = {
    'tense': {
        'type': 'SYN',
        'pairs': [
            ("The cat sleeps on the mat", "The cat slept on the mat"),
            ("She walks to school every day", "She walked to school every day"),
            ("He runs fast in the morning", "He ran fast in the morning"),
            ("They play soccer after class", "They played soccer after class"),
            ("I read books every weekend", "I read a book last weekend"),
            ("We eat lunch at noon", "We ate lunch at noon"),
            ("The dog barks loudly", "The dog barked loudly"),
            ("She sings beautifully", "She sang beautifully"),
            ("The bird flies south", "The bird flew south"),
            ("He drives to work", "He drove to work"),
            ("The children laugh together", "The children laughed together"),
            ("I write letters sometimes", "I wrote letters yesterday"),
            ("The wind blows hard", "The wind blew hard"),
            ("She teaches math well", "She taught math well"),
            ("They build houses here", "They built houses here"),
            ("We drink coffee daily", "We drank coffee yesterday"),
            ("The river flows quietly", "The river flowed quietly"),
            ("He reads the newspaper", "He read the newspaper"),
            ("The train arrives late", "The train arrived late"),
            ("She wears red dresses", "She wore a red dress"),
            ("I understand the lesson", "I understood the lesson"),
            ("The sun shines bright", "The sun shone bright"),
            ("They win the game", "They won the game"),
            ("We begin the project", "We began the project"),
            ("The bell rings twice", "The bell rang twice"),
            ("She draws beautiful pictures", "She drew beautiful pictures"),
            ("He catches the ball", "He caught the ball"),
            ("The horse gallops fast", "The horse galloped fast"),
            ("I feel happy today", "I felt happy yesterday"),
            ("The plant grows tall", "The plant grew tall"),
        ],
    },
    'polarity': {
        'type': 'SYN',
        'pairs': [
            ("The cat is on the mat", "The cat is not on the mat"),
            ("She likes the movie", "She does not like the movie"),
            ("He can swim well", "He cannot swim well"),
            ("They will come tomorrow", "They will not come tomorrow"),
            ("I have finished the work", "I have not finished the work"),
            ("We should go now", "We should not go now"),
            ("The door is open", "The door is not open"),
            ("She was happy", "She was not happy"),
            ("He could see the mountain", "He could not see the mountain"),
            ("They must leave early", "They must not leave early"),
            ("I am tired today", "I am not tired today"),
            ("We were ready", "We were not ready"),
            ("The car is working", "The car is not working"),
            ("She has the book", "She does not have the book"),
            ("He knows the answer", "He does not know the answer"),
            ("They found the key", "They did not find the key"),
            ("I believe the story", "I do not believe the story"),
            ("We enjoyed the party", "We did not enjoy the party"),
            ("The bird is singing", "The bird is not singing"),
            ("She bought the dress", "She did not buy the dress"),
            ("He won the prize", "He did not win the prize"),
            ("They passed the test", "They did not pass the test"),
            ("I trust the process", "I do not trust the process"),
            ("We need the money", "We do not need the money"),
            ("The dog is barking", "The dog is not barking"),
            ("She loves the song", "She does not love the song"),
            ("He ate the cake", "He did not eat the cake"),
            ("They saw the sign", "They did not see the sign"),
            ("I want the job", "I do not want the job"),
            ("We like the plan", "We do not like the plan"),
        ],
    },
    'number': {
        'type': 'SYN',
        'pairs': [
            ("The cat sleeps on the mat", "The cats sleep on the mat"),
            ("A dog barks loudly", "Some dogs bark loudly"),
            ("This book is interesting", "These books are interesting"),
            ("That tree looks old", "Those trees look old"),
            ("The child plays outside", "The children play outside"),
            ("A bird sings beautifully", "Some birds sing beautifully"),
            ("This flower smells nice", "These flowers smell nice"),
            ("That house looks big", "Those houses look big"),
            ("The man walks slowly", "The men walk slowly"),
            ("A woman reads quietly", "Some women read quietly"),
            ("This student studies hard", "These students study hard"),
            ("That teacher speaks clearly", "Those teachers speak clearly"),
            ("The fish swims fast", "The fish swim fast"),
            ("A sheep grazes quietly", "Some sheep graze quietly"),
            ("This mouse runs quickly", "These mice run quickly"),
            ("That tooth looks healthy", "Those teeth look healthy"),
            ("The foot hurts badly", "The feet hurt badly"),
            ("A person thinks deeply", "Some people think deeply"),
            ("This box contains books", "These boxes contain books"),
            ("That bus arrives late", "Those buses arrive late"),
            ("The leaf falls gently", "The leaves fall gently"),
            ("A knife cuts sharply", "Some knives cut sharply"),
            ("This shelf holds books", "These shelves hold books"),
            ("That wolf howls loudly", "Those wolves howl loudly"),
            ("The calf runs fast", "The calves run fast"),
            ("A goose swims well", "Some geese swim well"),
            ("This toothbrush is new", "These toothbrushes are new"),
            ("That country is large", "Those countries are large"),
            ("The city grows fast", "The cities grow fast"),
            ("A story ends well", "Some stories end well"),
        ],
    },
    'negation': {
        'type': 'SYN',
        'pairs': [
            ("She is always happy", "She is never happy"),
            ("He sometimes forgets", "He never forgets"),
            ("They often visit us", "They rarely visit us"),
            ("I always wake up early", "I never wake up early"),
            ("We frequently travel", "We seldom travel"),
            ("She usually agrees", "She rarely agrees"),
            ("He generally wins", "He rarely wins"),
            ("They commonly practice", "They rarely practice"),
            ("I mostly succeed", "I rarely succeed"),
            ("We normally finish", "We never finish"),
            ("She constantly worries", "She never worries"),
            ("He certainly knows", "He hardly knows"),
            ("They definitely agree", "They hardly agree"),
            ("I absolutely believe", "I hardly believe"),
            ("We completely understand", "We barely understand"),
            ("She entirely agrees", "She barely agrees"),
            ("He totally gets it", "He hardly gets it"),
            ("They fully support it", "They barely support it"),
            ("I entirely trust them", "I barely trust them"),
            ("We wholly accept it", "We hardly accept it"),
            ("She consistently delivers", "She never delivers"),
            ("He invariably succeeds", "He never succeeds"),
            ("They regularly contribute", "They rarely contribute"),
            ("I perpetually study", "I never study"),
            ("We perpetually improve", "We never improve"),
            ("She always remembers", "She never remembers"),
            ("He constantly improves", "He never improves"),
            ("They invariably succeed", "They rarely succeed"),
            ("I consistently perform", "I rarely perform"),
            ("We always achieve", "We never achieve"),
        ],
    },
    'question': {
        'type': 'SYN',
        'pairs': [
            ("The cat is on the mat", "Is the cat on the mat?"),
            ("She likes the movie", "Does she like the movie?"),
            ("He can swim well", "Can he swim well?"),
            ("They will come tomorrow", "Will they come tomorrow?"),
            ("I have finished the work", "Have I finished the work?"),
            ("We should go now", "Should we go now?"),
            ("The door is open", "Is the door open?"),
            ("She was happy", "Was she happy?"),
            ("He could see the mountain", "Could he see the mountain?"),
            ("They must leave early", "Must they leave early?"),
            ("I am tired today", "Am I tired today?"),
            ("We were ready", "Were we ready?"),
            ("The car is working", "Is the car working?"),
            ("She has the book", "Does she have the book?"),
            ("He knows the answer", "Does he know the answer?"),
            ("They found the key", "Did they find the key?"),
            ("I believe the story", "Do I believe the story?"),
            ("We enjoyed the party", "Did we enjoy the party?"),
            ("The bird is singing", "Is the bird singing?"),
            ("She bought the dress", "Did she buy the dress?"),
            ("He won the prize", "Did he win the prize?"),
            ("They passed the test", "Did they pass the test?"),
            ("I trust the process", "Do I trust the process?"),
            ("We need the money", "Do we need the money?"),
            ("The dog is barking", "Is the dog barking?"),
            ("She loves the song", "Does she love the song?"),
            ("He ate the cake", "Did he eat the cake?"),
            ("They saw the sign", "Did they see the sign?"),
            ("I want the job", "Do I want the job?"),
            ("We like the plan", "Do we like the plan?"),
        ],
    },
    'person': {
        'type': 'SYN',
        'pairs': [
            ("I walk to the store", "She walks to the store"),
            ("I have a book", "She has a book"),
            ("I am happy today", "She is happy today"),
            ("I was running late", "She was running late"),
            ("I can do this", "She can do this"),
            ("We walk to the store", "They walk to the store"),
            ("We have books", "They have books"),
            ("We are happy today", "They are happy today"),
            ("We were running late", "They were running late"),
            ("We can do this", "They can do this"),
            ("I like the food", "He likes the food"),
            ("I go to school", "He goes to school"),
            ("I study hard", "He studies hard"),
            ("I play guitar", "He plays guitar"),
            ("I know the answer", "He knows the answer"),
            ("We eat lunch together", "They eat lunch together"),
            ("We read every day", "They read every day"),
            ("We sing in the choir", "They sing in the choir"),
            ("We work from home", "They work from home"),
            ("We live nearby", "They live nearby"),
            ("I enjoy music", "She enjoys music"),
            ("I need help", "She needs help"),
            ("I think so", "She thinks so"),
            ("I try hard", "She tries hard"),
            ("I watch TV", "She watches TV"),
            ("We drive carefully", "They drive carefully"),
            ("We cook dinner", "They cook dinner"),
            ("We speak English", "They speak English"),
            ("We sleep early", "They sleep early"),
            ("We exercise daily", "They exercise daily"),
        ],
    },
    'definiteness': {
        'type': 'SYN',
        'pairs': [
            ("A cat sleeps on the mat", "The cat sleeps on the mat"),
            ("A dog barks in the yard", "The dog barks in the yard"),
            ("A bird sings in the tree", "The bird sings in the tree"),
            ("A child plays in the park", "The child plays in the park"),
            ("A student reads in the library", "The student reads in the library"),
            ("A flower grows in the garden", "The flower grows in the garden"),
            ("A man walks down the street", "The man walks down the street"),
            ("A woman works in the office", "The woman works in the office"),
            ("A book sits on the shelf", "The book sits on the shelf"),
            ("A car drives on the road", "The car drives on the road"),
            ("A house stands on the hill", "The house stands on the hill"),
            ("A tree grows in the forest", "The tree grows in the forest"),
            ("A river flows through the valley", "The river flows through the valley"),
            ("A mountain rises behind the town", "The mountain rises behind the town"),
            ("A star shines in the sky", "The star shines in the sky"),
            ("A fish swims in the pond", "The fish swims in the pond"),
            ("A horse runs across the field", "The horse runs across the field"),
            ("A cat sits by the window", "The cat sits by the window"),
            ("A dog waits at the door", "The dog waits at the door"),
            ("A bird flies over the lake", "The bird flies over the lake"),
            ("An apple falls from the tree", "The apple falls from the tree"),
            ("An idea comes to mind", "The idea comes to mind"),
            ("An egg sits in the nest", "The egg sits in the nest"),
            ("An old man walks slowly", "The old man walks slowly"),
            ("An artist paints the landscape", "The artist paints the landscape"),
            ("A teacher writes on the board", "The teacher writes on the board"),
            ("A doctor treats the patient", "The doctor treats the patient"),
            ("A singer performs the song", "The singer performs the song"),
            ("A farmer tends the crops", "The farmer tends the crops"),
            ("A writer drafts the novel", "The writer drafts the novel"),
        ],
    },
    'info_structure': {
        'type': 'SYN',
        'pairs': [
            ("John broke the window", "It was John who broke the window"),
            ("Mary found the key", "It was Mary who found the key"),
            ("Tom ate the cake", "It was Tom who ate the cake"),
            ("She bought the book", "It was the book that she bought"),
            ("He met the teacher", "It was the teacher that he met"),
            ("They visited Paris", "It was Paris that they visited"),
            ("We finished the project", "It was the project that we finished"),
            ("I lost my keys", "It was my keys that I lost"),
            ("She loves chocolate", "It is chocolate that she loves"),
            ("He needs money", "It is money that he needs"),
            ("John broke the window yesterday", "What John broke yesterday was the window"),
            ("Mary found the key outside", "What Mary found outside was the key"),
            ("Tom ate the cake quickly", "What Tom ate quickly was the cake"),
            ("She painted the door red", "What she painted red was the door"),
            ("He fixed the car yesterday", "What he fixed yesterday was the car"),
            ("The team won the championship", "It was the championship that the team won"),
            ("The dog chased the cat", "It was the cat that the dog chased"),
            ("The rain ruined the picnic", "It was the picnic that the rain ruined"),
            ("The wind broke the fence", "It was the fence that the wind broke"),
            ("The fire destroyed the barn", "It was the barn that the fire destroyed"),
            ("Alice wrote the report", "It was Alice who wrote the report"),
            ("Bob delivered the package", "It was Bob who delivered the package"),
            ("Carol designed the logo", "It was Carol who designed the logo"),
            ("David built the cabinet", "It was David who built the cabinet"),
            ("Eve composed the song", "It was Eve who composed the song"),
            ("Frank directed the movie", "It was Frank who directed the movie"),
            ("Grace organized the event", "It was Grace who organized the event"),
            ("Henry managed the team", "It was Henry who managed the team"),
            ("Irene edited the article", "It was Irene who edited the article"),
            ("Jack painted the portrait", "It was Jack who painted the portrait"),
        ],
    },
    'sentiment': {
        'type': 'SEM',
        'pairs': [
            ("The movie was wonderful and exciting", "The movie was terrible and boring"),
            ("She had a great day at work", "She had a awful day at work"),
            ("The food was delicious and fresh", "The food was disgusting and stale"),
            ("He felt happy and grateful", "He felt sad and resentful"),
            ("The weather was beautiful today", "The weather was dreadful today"),
            ("They enjoyed the lovely concert", "They suffered through the awful concert"),
            ("The gift was thoughtful and generous", "The gift was thoughtless and cheap"),
            ("She gave a brilliant performance", "She gave a terrible performance"),
            ("The garden looked vibrant and healthy", "The garden looked withered and dying"),
            ("He received a kind and warm welcome", "He received a cold and hostile welcome"),
            ("The team celebrated their amazing victory", "The team mourned their devastating defeat"),
            ("She found the book fascinating and insightful", "She found the book dull and pointless"),
            ("The hotel was comfortable and clean", "The hotel was filthy and uncomfortable"),
            ("He spoke with confidence and pride", "He spoke with shame and embarrassment"),
            ("The children played joyfully together", "The children fought bitterly together"),
            ("She smiled warmly at the audience", "She glared coldly at the audience"),
            ("The project was a remarkable success", "The project was a complete failure"),
            ("He told a hilarious joke", "He told an offensive joke"),
            ("The sunrise was breathtaking and peaceful", "The storm was terrifying and destructive"),
            ("She felt loved and appreciated", "She felt ignored and rejected"),
            ("The park was serene and inviting", "The park was chaotic and threatening"),
            ("His recovery was miraculous and swift", "His decline was tragic and rapid"),
            ("The news was encouraging and hopeful", "The news was devastating and hopeless"),
            ("She wore an elegant and stunning dress", "She wore a shabby and hideous dress"),
            ("The music was soothing and harmonious", "The noise was grating and discordant"),
            ("He showed genuine compassion and care", "He showed blatant cruelty and indifference"),
            ("The meal was exquisite and satisfying", "The meal was revolting and unsatisfying"),
            ("She expressed deep gratitude and joy", "She expressed bitter resentment and anger"),
            ("The experience was magical and memorable", "The experience was horrific and forgettable"),
            ("He demonstrated remarkable courage and strength", "He demonstrated shameful cowardice and weakness"),
        ],
    },
    'semantic_topic': {
        'type': 'SEM',
        'pairs': [
            ("The car engine needs repair", "The kitchen faucet needs repair"),
            ("He scored a goal in the match", "He passed the exam with honors"),
            ("The stock market rose today", "The river level rose today"),
            ("She programmed the computer", "She painted the landscape"),
            ("The surgery was successful", "The concert was successful"),
            ("The rocket launched successfully", "The bakery opened successfully"),
            ("The software update is ready", "The dinner order is ready"),
            ("He fixed the broken circuit", "He fixed the broken fence"),
            ("The database crashed overnight", "The car crashed overnight"),
            ("The vaccine was approved today", "The building was approved today"),
            ("The processor runs at high speed", "The athlete runs at high speed"),
            ("The experiment produced results", "The recipe produced results"),
            ("The algorithm found the solution", "The detective found the solution"),
            ("The satellite orbits the earth", "The bird orbits the nest"),
            ("The network connection failed", "The marriage connection failed"),
            ("The code compiled successfully", "The cake baked successfully"),
            ("The battery needs charging", "The student needs encouragement"),
            ("The firewall blocked the attack", "The shield blocked the blow"),
            ("The robot performed the task", "The worker performed the task"),
            ("The microscope revealed the structure", "The telescope revealed the galaxy"),
            ("The printer needs more paper", "The chef needs more flour"),
            ("The bridge carries heavy traffic", "The gene carries important information"),
            ("The engine produces power", "The factory produces goods"),
            ("The camera captured the moment", "The painter captured the scene"),
            ("The telescope observed the star", "The microscope observed the cell"),
            ("The keyboard needs cleaning", "The floor needs cleaning"),
            ("The server is down today", "The patient is down today"),
            ("The memory stores the data", "The brain stores the memory"),
            ("The monitor displays the image", "The screen displays the movie"),
            ("The circuit carries the signal", "The road carries the traffic"),
        ],
    },
    'voice': {
        'type': 'SEM',
        'pairs': [
            ("The chef cooked the meal", "The meal was cooked by the chef"),
            ("The company launched the product", "The product was launched by the company"),
            ("The artist painted the portrait", "The portrait was painted by the artist"),
            ("The writer composed the letter", "The letter was composed by the writer"),
            ("The team won the championship", "The championship was won by the team"),
            ("The committee approved the proposal", "The proposal was approved by the committee"),
            ("The teacher explained the concept", "The concept was explained by the teacher"),
            ("The scientist discovered the formula", "The formula was discovered by the scientist"),
            ("The architect designed the building", "The building was designed by the architect"),
            ("The police arrested the suspect", "The suspect was arrested by the police"),
            ("The mechanic fixed the engine", "The engine was fixed by the mechanic"),
            ("The judge sentenced the criminal", "The criminal was sentenced by the judge"),
            ("The doctor treated the patient", "The patient was treated by the doctor"),
            ("The builder constructed the house", "The house was constructed by the builder"),
            ("The author wrote the novel", "The novel was written by the author"),
            ("The chef prepared the dinner", "The dinner was prepared by the chef"),
            ("The band performed the song", "The song was performed by the band"),
            ("The gardener planted the tree", "The tree was planted by the gardener"),
            ("The driver delivered the package", "The package was delivered by the driver"),
            ("The student solved the problem", "The problem was solved by the student"),
            ("The manager hired the employee", "The employee was hired by the manager"),
            ("The chef baked the cake", "The cake was baked by the chef"),
            ("The programmer debugged the code", "The code was debugged by the programmer"),
            ("The tailor made the suit", "The suit was made by the tailor"),
            ("The musician played the violin", "The violin was played by the musician"),
            ("The editor revised the manuscript", "The manuscript was revised by the editor"),
            ("The referee called the foul", "The foul was called by the referee"),
            ("The captain steered the ship", "The ship was steered by the captain"),
            ("The pilot flew the plane", "The plane was flown by the pilot"),
            ("The conductor led the orchestra", "The orchestra was led by the conductor"),
        ],
    },
    'formality': {
        'type': 'SEM',
        'pairs': [
            ("It is imperative that we proceed", "We really need to get going"),
            ("I would like to request assistance", "Can you help me out"),
            ("The individual demonstrated proficiency", "The guy showed he is good at it"),
            ("We shall commence the proceedings", "Let us get started"),
            ("Please accept my sincere gratitude", "Thanks a lot"),
            ("I am writing to inquire regarding", "I am writing to ask about"),
            ("The aforementioned document indicates", "That paper says"),
            ("It is recommended that you consider", "You should think about"),
            ("Furthermore it should be noted that", "Also you should know"),
            ("In accordance with the regulations", "Following the rules"),
            ("We regret to inform you that", "Sorry to say but"),
            ("Please do not hesitate to contact", "Just let us know"),
            ("The organization endeavors to provide", "The group tries to give"),
            ("Subsequent to the aforementioned event", "After that thing happened"),
            ("It is essential to acknowledge", "You gotta admit"),
            ("We would greatly appreciate your cooperation", "Please work with us on this"),
            ("The implementation of this strategy", "Putting this plan into action"),
            ("Please find attached the documentation", "Here are the papers you wanted"),
            ("I look forward to your response", "Hope to hear from you"),
            ("With reference to your correspondence", "About your letter"),
            ("The administration has determined that", "The bosses decided"),
            ("It has come to our attention that", "We noticed that"),
            ("We are pleased to announce that", "Happy to say that"),
            ("The facility will be closed indefinitely", "The place is shut down"),
            ("Please ensure compliance with the protocol", "Make sure you follow the rules"),
            ("The committee has resolved to proceed", "The group decided to go ahead"),
            ("We acknowledge receipt of your communication", "We got your message"),
            ("The modification will be implemented shortly", "The change will happen soon"),
            ("Kindly remit payment at your earliest convenience", "Please pay when you can"),
            ("We apologize for any inconvenience caused", "Sorry for the trouble"),
        ],
    },
}

def get_residual_stream(model, tokenizer, text, layer_idx, device='cuda'):
    """获取残差流向量"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
    input_ids = inputs['input_ids'].to(device)
    
    resid = {}
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        resid['val'] = out[0, -1, :].detach().cpu().float().clone()
    
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_ids)
    handle.remove()
    
    return resid.get('val', None)

def run_phase_ccxv(model_name, n_pairs=120):
    cfg = MODEL_CONFIGS[model_name]
    
    out_dir = f'results/causal_fiber/{model_name}_ccxv'
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'run.log')
    
    def log(msg):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    log(f"{'='*60}")
    log(f"Phase CCXV: 因果原子分解 (差分向量+PCA+W_o投影)")
    log(f"Model: {cfg['name']}, n_pairs={n_pairs}")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*60}")
    
    # 加载模型
    log("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    t0 = time.time()
    
    path = cfg['path']
    
    if model_name in ["glm4", "deepseek7b"]:
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
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log(f"Model loaded in {time.time()-t0:.0f}s")
    
    device = next(model.parameters()).device
    device_str = str(device)
    
    # 从模型配置获取参数
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    d_model = model.config.hidden_size
    
    # 层选择
    layer_indices = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    layer_indices = sorted(list(set(layer_indices)))
    layer_names = [f"L{l}" for l in layer_indices]
    log(f"Layers: {layer_names}")
    
    feature_names = list(FEATURE_PAIRS.keys())
    syn_features = [f for f in feature_names if FEATURE_PAIRS[f]['type'] == 'SYN']
    sem_features = [f for f in feature_names if FEATURE_PAIRS[f]['type'] == 'SEM']
    
    # ===== S1: 差分向量收集 =====
    log(f"\n{'='*60}")
    log("S1: 差分向量收集")
    log(f"{'='*60}")
    
    all_deltas = {}  # {feat: {layer: [delta_vectors]}}
    all_norms = {}   # {feat: {layer: [norms]}}
    
    for feat in feature_names:
        pairs = FEATURE_PAIRS[feat]['pairs']
        actual_n = min(n_pairs, len(pairs))
        
        all_deltas[feat] = {l: [] for l in layer_indices}
        all_norms[feat] = {l: [] for l in layer_indices}
        
        t_start = time.time()
        for i in range(actual_n):
            sent_a, sent_b = pairs[i]
            for l in layer_indices:
                vec_a = get_residual_stream(model, tokenizer, sent_a, l, device_str)
                vec_b = get_residual_stream(model, tokenizer, sent_b, l, device_str)
                if vec_a is not None and vec_b is not None:
                    delta = (vec_b - vec_a).numpy()
                    all_deltas[feat][l].append(delta)
                    all_norms[feat][l].append(float(np.linalg.norm(delta)))
                else:
                    all_deltas[feat][l].append(None)
                    all_norms[feat][l].append(0.0)
        
        avg_norms = [np.mean(all_norms[feat][l]) if all_norms[feat][l] else 0 for l in layer_indices]
        elapsed = time.time() - t_start
        log(f"  {feat} [{FEATURE_PAIRS[feat]['type']}] [{elapsed:.0f}s]: norms={', '.join([f'{n:.0f}' for n in avg_norms])}")
    
    # ===== S2: PCA原子分解 =====
    log(f"\n{'='*60}")
    log("S2: PCA原子分解 + 子空间正交性")
    log(f"{'='*60}")
    
    pca_results = {}
    
    for l in layer_indices:
        syn_deltas = []
        sem_deltas = []
        feat_indices = {}  # {feat: (start, end)}
        
        idx = 0
        for feat in feature_names:
            valid_deltas = [d for d in all_deltas[feat][l] if d is not None]
            start_idx = idx
            if valid_deltas:
                if FEATURE_PAIRS[feat]['type'] == 'SYN':
                    syn_deltas.extend(valid_deltas)
                else:
                    sem_deltas.extend(valid_deltas)
                idx += len(valid_deltas)
            feat_indices[feat] = (start_idx, idx)
        
        all_vecs = syn_deltas + sem_deltas
        if len(all_vecs) < 10:
            log(f"  L{l}: insufficient vectors ({len(all_vecs)})")
            continue
        
        mat = np.array(all_vecs, dtype=np.float32)  # 使用float32减少内存
        n_samples, d = mat.shape
        log(f"  L{l}: {n_samples} vectors (SYN={len(syn_deltas)}, SEM={len(sem_deltas)}), d={d}")
        
        # 使用TruncatedSVD (随机SVD, 内存友好)
        from sklearn.decomposition import TruncatedSVD
        n_pc = min(20, n_samples - 1, d)
        svd = TruncatedSVD(n_components=n_pc, random_state=42)
        projected = svd.fit_transform(mat)
        
        var_top5 = svd.explained_variance_ratio_[:5].tolist()
        cumvar_top5 = np.cumsum(var_top5).tolist()
        cumvar_10 = np.cumsum(svd.explained_variance_ratio_[:10]).tolist()[-1] if len(var_top5) >= 5 else cumvar_top5[-1]
        
        log(f"  L{l} PCA: PC1={var_top5[0]:.4f}, PC2={var_top5[1]:.4f}, cum5={cumvar_top5[-1]:.4f}, cum10={cumvar_10:.4f}")
        
        syn_proj = projected[:len(syn_deltas)]
        sem_proj = projected[len(syn_deltas):]
        
        # PC1 Cohen's d
        pc1_syn = syn_proj[:, 0]
        pc1_sem = sem_proj[:, 0]
        pooled_std = np.sqrt((np.var(pc1_syn) + np.var(pc1_sem)) / 2)
        cohens_d = (np.mean(pc1_syn) - np.mean(pc1_sem)) / (pooled_std + 1e-8)
        
        # 逐特征centroid
        feat_centroids = {}
        for feat in feature_names:
            start_i, end_i = feat_indices[feat]
            if end_i > start_i:
                feat_proj = projected[start_i:end_i]
                feat_centroids[feat] = {
                    'pc1': float(feat_proj[:, 0].mean()),
                    'pc2': float(feat_proj[:, 1].mean()),
                    'pc1_std': float(feat_proj[:, 0].std()),
                    'pc2_std': float(feat_proj[:, 1].std()),
                }
        
        # 子空间正交性 - 用centroid余弦相似度代替SVD (更省内存)
        k = min(5, len(syn_deltas), len(sem_deltas))
        subspace_overlap = None
        if k >= 3:
            # 用PC空间中的centroid距离作为正交性度量
            syn_centroid = np.mean(syn_proj[:, :5], axis=0)
            sem_centroid = np.mean(sem_proj[:, :5], axis=0)
            n1 = np.linalg.norm(syn_centroid)
            n2 = np.linalg.norm(sem_centroid)
            if n1 > 1e-8 and n2 > 1e-8:
                cos_sim = np.dot(syn_centroid, sem_centroid) / (n1 * n2)
                subspace_overlap = float(abs(cos_sim))
            else:
                subspace_overlap = 0.0
            log(f"  L{l} centroid_cos_sim(PC1-5): {subspace_overlap:.4f}")
        
        # 逐特征分离度
        # 计算每个语法特征与所有语义特征的PC1 centroid距离
        feat_separability = {}
        for feat in feature_names:
            if feat in feat_centroids and feat_centroids[feat]['pc1_std'] > 1e-8:
                # 分离度 = |centroid_syn - centroid_sem_mean| / pooled_std
                if FEATURE_PAIRS[feat]['type'] == 'SYN':
                    sem_centroids_pc1 = [feat_centroids[f]['pc1'] for f in sem_features if f in feat_centroids]
                    sem_stds_pc1 = [feat_centroids[f]['pc1_std'] for f in sem_features if f in feat_centroids]
                    if sem_centroids_pc1:
                        sem_mean = np.mean(sem_centroids_pc1)
                        pooled_s = np.sqrt(feat_centroids[feat]['pc1_std']**2 + np.mean(sem_stds_pc1)**2)
                        sep = abs(feat_centroids[feat]['pc1'] - sem_mean) / (pooled_s + 1e-8)
                        feat_separability[feat] = float(sep)
                else:
                    syn_centroids_pc1 = [feat_centroids[f]['pc1'] for f in syn_features if f in feat_centroids]
                    syn_stds_pc1 = [feat_centroids[f]['pc1_std'] for f in syn_features if f in feat_centroids]
                    if syn_centroids_pc1:
                        syn_mean = np.mean(syn_centroids_pc1)
                        pooled_s = np.sqrt(feat_centroids[feat]['pc1_std']**2 + np.mean(syn_stds_pc1)**2)
                        sep = abs(feat_centroids[feat]['pc1'] - syn_mean) / (pooled_s + 1e-8)
                        feat_separability[feat] = float(sep)
        
        pca_results[l] = {
            'var_top5': var_top5,
            'cumvar_top5': cumvar_top5,
            'cumvar_10': float(cumvar_10),
            'n_syn': len(syn_deltas),
            'n_sem': len(sem_deltas),
            'pc1_cohens_d': float(cohens_d),
            'subspace_overlap': subspace_overlap,
            'feat_centroids': feat_centroids,
            'feat_separability': feat_separability,
        }
    
    # ===== S3: W_o Head贡献分解 =====
    log(f"\n{'='*60}")
    log("S3: W_o Head贡献分解")
    log(f"{'='*60}")
    
    head_contrib_results = {}
    
    for l in layer_indices:
        try:
            w = model.model.layers[l].self_attn.o_proj.weight
            # 8bit模型需要dequantize
            if hasattr(w, 'CB') or hasattr(w, 'SCB'):  # 8bit quantized
                log(f"  L{l}: W_o is 8bit quantized, dequantizing...")
                W_o = w.dequantize().detach().float().cpu().numpy()
            else:
                W_o = w.detach().float().cpu().numpy()
        except Exception as e:
            log(f"  L{l}: W_o unavailable ({e})")
            continue
        
        if W_o is None:
            log(f"  L{l}: W_o is None")
            continue
        
        # W_o: (d_model, n_heads*head_dim)
        d_model_local = W_o.shape[0]
        
        if W_o.shape[1] != n_heads * head_dim:
            log(f"  L{l}: W_o shape mismatch ({W_o.shape} vs ({d_model_local}, {n_heads*head_dim}))")
            continue
        
        # 分割W_o为每个head
        W_o_heads = W_o.reshape(d_model_local, n_heads, head_dim)
        
        head_specs = {}
        for feat in feature_names:
            valid_deltas = [d for d in all_deltas[feat][l] if d is not None]
            if not valid_deltas:
                continue
            
            delta_mean = np.mean(valid_deltas, axis=0)
            
            # 计算每个head的贡献范数
            head_norms = []
            for h in range(n_heads):
                # delta在head h输出方向的投影
                W_h = W_o_heads[:, h, :]  # (d_model, head_dim)
                # 投影: delta @ W_h / ||W_h||_F * ||W_h||_F = ||delta @ W_h|| / ||W_h||
                proj = delta_mean @ W_h  # (head_dim,)
                contrib_norm = np.linalg.norm(proj)
                head_norms.append(contrib_norm)
            
            total = sum(head_norms) + 1e-10
            head_contrib_pct = [n / total for n in head_norms]
            
            head_specs[feat] = {
                'contrib_norms': head_norms,
                'contrib_pct': head_contrib_pct,
            }
        
        # Head专化分析
        syn_avg = np.zeros(n_heads)
        sem_avg = np.zeros(n_heads)
        n_syn_f = 0
        n_sem_f = 0
        
        for feat in feature_names:
            if feat in head_specs:
                pcts = np.array(head_specs[feat]['contrib_pct'])
                if FEATURE_PAIRS[feat]['type'] == 'SYN':
                    syn_avg += pcts
                    n_syn_f += 1
                else:
                    sem_avg += pcts
                    n_sem_f += 1
        
        if n_syn_f > 0 and n_sem_f > 0:
            syn_avg /= n_syn_f
            sem_avg /= n_sem_f
            
            spec_indices = []
            for h in range(n_heads):
                total_pct = syn_avg[h] + sem_avg[h]
                if total_pct > 1e-8:
                    spec = (syn_avg[h] - sem_avg[h]) / total_pct
                else:
                    spec = 0.0
                spec_indices.append(spec)
            
            n_syn_heads = sum(1 for s in spec_indices if s > 0.3)
            n_sem_heads = sum(1 for s in spec_indices if s < -0.3)
            n_mixed = n_heads - n_syn_heads - n_sem_heads
            
            top_syn = sorted(range(n_heads), key=lambda h: spec_indices[h], reverse=True)[:3]
            top_sem = sorted(range(n_heads), key=lambda h: spec_indices[h])[:3]
            
            log(f"  L{l}: SYN={n_syn_heads}, SEM={n_sem_heads}, MIXED={n_mixed}")
            log(f"    Top SYN: {top_syn} spec={[round(spec_indices[h],3) for h in top_syn]}")
            log(f"    Top SEM: {top_sem} spec={[round(spec_indices[h],3) for h in top_sem]}")
            
            head_contrib_results[l] = {
                'spec_indices': spec_indices,
                'n_syn_heads': n_syn_heads,
                'n_sem_heads': n_sem_heads,
                'n_mixed': n_mixed,
                'top_syn_heads': top_syn,
                'top_sem_heads': top_sem,
            }
        else:
            log(f"  L{l}: insufficient data for head spec")
    
    # ===== S4: 因果原子发现 =====
    log(f"\n{'='*60}")
    log("S4: 因果原子发现")
    log(f"{'='*60}")
    
    # 通过PCA发现因果原子
    # 原子 = PCA成分中被语法/语义主导的维度
    for l in layer_indices:
        if l not in pca_results:
            continue
        
        pca_data = pca_results[l]
        centroids = pca_data['feat_centroids']
        
        # 计算PC1上语法vs语义的分离
        syn_pc1 = [centroids[f]['pc1'] for f in syn_features if f in centroids]
        sem_pc1 = [centroids[f]['pc1'] for f in sem_features if f in centroids]
        
        if syn_pc1 and sem_pc1:
            # PC1是语法原子还是语义原子？
            pc1_syn_spread = np.std(syn_pc1)
            pc1_sem_spread = np.std(sem_pc1)
            
            log(f"  L{l}: PC1 syn_spread={pc1_syn_spread:.3f}, sem_spread={pc1_sem_spread:.3f}")
            
            # 逐特征分离度排名
            sep_rank = sorted(pca_data['feat_separability'].items(), key=lambda x: x[1], reverse=True)
            log(f"  L{l}: Feature separability ranking:")
            for feat, sep in sep_rank[:6]:
                log(f"    {feat} [{FEATURE_PAIRS[feat]['type']}]: {sep:.3f}")
    
    # ===== S5: 统计检验 =====
    log(f"\n{'='*60}")
    log("S5: 统计检验")
    log(f"{'='*60}")
    
    # 纤维growth
    syn_growth = []
    sem_growth = []
    for feat in feature_names:
        norms = all_norms[feat]
        norm_values = [np.mean(norms[l]) if norms[l] else 0 for l in layer_indices]
        if norm_values[0] > 0 and norm_values[-1] > 0:
            growth = norm_values[-1] / norm_values[0]
            if FEATURE_PAIRS[feat]['type'] == 'SYN':
                syn_growth.append(growth)
            else:
                sem_growth.append(growth)
    
    if len(syn_growth) >= 3 and len(sem_growth) >= 3:
        _, p_growth = stats.mannwhitneyu(syn_growth, sem_growth, alternative='greater')
        log(f"  Growth: syn_mean={np.mean(syn_growth):.2f} vs sem_mean={np.mean(sem_growth):.2f}, p={p_growth:.4f}")
    else:
        p_growth = 1.0
        log(f"  Growth: insufficient data")
    
    # PCA Cohen's d
    cohens_ds = [pca_results[l]['pc1_cohens_d'] for l in layer_indices if l in pca_results]
    if cohens_ds:
        log(f"  PC1 Cohen's d: {[f'{d:.3f}' for d in cohens_ds]}")
        mean_d = np.mean([abs(d) for d in cohens_ds])
        log(f"  Mean |Cohen's d|: {mean_d:.3f}")
    
    # 子空间正交性
    overlaps = [pca_results[l]['subspace_overlap'] for l in layer_indices if l in pca_results and pca_results[l]['subspace_overlap'] is not None]
    if overlaps:
        log(f"  Subspace overlaps: {[f'{o:.4f}' for o in overlaps]}")
        log(f"  Mean overlap: {np.mean(overlaps):.4f}")
    
    # ===== FINAL =====
    log(f"\n{'='*60}")
    log(f"FINAL: {model_name}")
    log(f"{'='*60}")
    
    log(f"\n--- Delta Norms ---")
    for feat in feature_names:
        norms = [np.mean(all_norms[feat][l]) if all_norms[feat][l] else 0 for l in layer_indices]
        log(f"  {feat} [{FEATURE_PAIRS[feat]['type']}]: {', '.join([f'{n:.0f}' for n in norms])}")
    
    log(f"\n--- PCA ---")
    for l in layer_indices:
        if l in pca_results:
            p = pca_results[l]
            log(f"  L{l}: PC1={p['var_top5'][0]:.4f}, cum5={p['cumvar_top5'][-1]:.4f}, Cohen_d={p['pc1_cohens_d']:.3f}, overlap={p.get('subspace_overlap', 'N/A')}")
    
    log(f"\n--- Head Specialization ---")
    for l in layer_indices:
        if l in head_contrib_results:
            h = head_contrib_results[l]
            log(f"  L{l}: SYN={h['n_syn_heads']}, SEM={h['n_sem_heads']}, MIXED={h['n_mixed']}")
    
    log(f"\n--- Fiber Growth ---")
    for feat in feature_names:
        norms = [np.mean(all_norms[feat][l]) if all_norms[feat][l] else 0 for l in layer_indices]
        if norms[0] > 0:
            growth = norms[-1] / norms[0]
            log(f"  {feat} [{FEATURE_PAIRS[feat]['type']}]: growth={growth:.2f}x")
    
    # 保存
    save_data = {
        'model': model_name,
        'n_pairs': n_pairs,
        'layer_indices': layer_indices,
        'pca': {str(k): v for k, v in pca_results.items()},
        'head_contrib': {str(k): {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) for kk, vv in v.items()} for k, v in head_contrib_results.items()},
        'stats': {'p_growth': float(p_growth)},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    with open(os.path.join(out_dir, 'full_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    log(f"\nDONE! Saved to {out_dir}")
    
    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=120)
    args = parser.parse_args()
    run_phase_ccxv(args.model, args.n_pairs)
