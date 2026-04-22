"""
Phase CCXIV v3: Head-wise因果纤维分解
策略: 
  1. 收集residual差分向量 (用CCXIII验证过的方法)
  2. W_o矩阵分解Head贡献 (后处理)
  3. Attn/MLP差分分离 (单独运行)
"""
import os, sys, gc, time, json, argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'w', buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

PATHS = {
    "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
    "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

TENSE = [("The cat sat quietly on the mat", "The cat sits quietly on the mat"), ("She walked to the store yesterday", "She walks to the store today"), ("He played guitar every evening", "He plays guitar every evening"), ("They worked on the project", "They work on the project"), ("The dog ran across the field", "The dog runs across the field"), ("She wrote a long letter", "She writes a long letter"), ("He drove the car fast", "He drives the car fast"), ("They built a new house", "They build a new house"), ("The bird flew over the lake", "The bird flies over the lake"), ("She cooked dinner for us", "She cooks dinner for us"), ("He read many books last year", "He reads many books this year"), ("They sang beautiful songs", "They sing beautiful songs"), ("The train arrived late", "The train arrives late"), ("She taught the children math", "She teaches the children math"), ("He caught the ball easily", "He catches the ball easily"), ("They grew vegetables in spring", "They grow vegetables in spring"), ("The water froze overnight", "The water freezes overnight"), ("She drew a colorful picture", "She draws a colorful picture"), ("He held the baby gently", "He holds the baby gently"), ("They knew the answer quickly", "They know the answer quickly")]
POLARITY = [("The cat is happy today", "The cat is not happy today"), ("She likes the new movie", "She does not like the new movie"), ("He can solve the problem", "He cannot solve the problem"), ("They will attend the meeting", "They will not attend the meeting"), ("The dog is friendly to strangers", "The dog is not friendly to strangers"), ("She has finished the work", "She has not finished the work"), ("He was available yesterday", "He was not available yesterday"), ("They are coming to the party", "They are not coming to the party"), ("The food was delicious", "The food was not delicious"), ("She does know the answer", "She does not know the answer"), ("He should go to school", "He should not go to school"), ("They would help with chores", "They would not help with chores"), ("The test was easy for me", "The test was not easy for me"), ("She could swim very well", "She could not swim very well"), ("He must leave right now", "He must not leave right now"), ("The door was open this morning", "The door was not open this morning"), ("She did enjoy the concert", "She did not enjoy the concert"), ("He has been working hard", "He has not been working hard"), ("They were ready on time", "They were not ready on time"), ("The plan was successful", "The plan was not successful")]
NUMBER = [("The cat is sleeping now", "The cats are sleeping now"), ("A book was on the table", "The books were on the table"), ("The child plays outside", "The children play outside"), ("A dog runs in the park", "The dogs run in the park"), ("The woman walks to work", "The women walk to work"), ("A man drives the bus", "The men drive the bus"), ("The bird flies high above", "The birds fly high above"), ("A student reads the text", "The students read the text"), ("The teacher explains clearly", "The teachers explain clearly"), ("A flower grows in spring", "The flowers grow in spring"), ("The tree stands very tall", "The trees stand very tall"), ("A car drives down the road", "The cars drive down the road"), ("The house looks beautiful", "The houses look beautiful"), ("A river flows through town", "The rivers flow through town"), ("The mountain rises above clouds", "The mountains rise above clouds"), ("A star shines in the sky", "The stars shine in the sky"), ("The country borders two seas", "The countries border two seas"), ("A city grows very fast", "The cities grow very fast"), ("The church stands on the hill", "The churches stand on the hill"), ("A bridge crosses the river", "The bridges cross the river")]
NEGATION = [("She is going to the store", "She is not going to the store"), ("He was working late today", "He was not working late today"), ("They have seen the movie", "They have not seen the movie"), ("The cat will eat the food", "The cat will not eat the food"), ("I could hear the music", "I could not hear the music"), ("She did finish the report", "She did not finish the report"), ("He would agree to help", "He would not agree to help"), ("We should try the method", "We should not try the method"), ("The door was locked tight", "The door was not locked tight"), ("She had written the essay", "She had not written the essay"), ("He can speak the language", "He cannot speak the language"), ("They were watching the show", "They were not watching the show"), ("I am enjoying the book", "I am not enjoying the book"), ("The kids are playing outside", "The kids are not playing outside"), ("She has visited Paris", "She has not visited Paris"), ("He does understand the rules", "He does not understand the rules"), ("They might come tomorrow", "They might not come tomorrow"), ("We need to leave early", "We do not need to leave early"), ("The system is working well", "The system is not working well"), ("She knows the correct answer", "She does not know the correct answer")]
QUESTION = [("The cat is sleeping now", "Is the cat sleeping now"), ("She walked to the store", "Did she walk to the store"), ("He can solve the problem", "Can he solve the problem"), ("They will attend the meeting", "Will they attend the meeting"), ("The dog is friendly to strangers", "Is the dog friendly to strangers"), ("She has finished the work", "Has she finished the work"), ("He was available yesterday", "Was he available yesterday"), ("They are coming to the party", "Are they coming to the party"), ("The food was delicious", "Was the food delicious"), ("She should go to school", "Should she go to school"), ("He could swim very well", "Could he swim very well"), ("They would help with chores", "Would they help with chores"), ("The test was easy for me", "Was the test easy for me"), ("He must leave right now", "Must he leave right now"), ("The door was open this morning", "Was the door open this morning"), ("He has been working hard", "Has he been working hard"), ("They were ready on time", "Were they ready on time"), ("The plan was successful", "Was the plan successful"), ("She will accept the offer", "Will she accept the offer"), ("He can drive the truck", "Can he drive the truck")]
PERSON = [("I am going to the store", "She is going to the store"), ("I have finished the work", "She has finished the work"), ("I was running late today", "She was running late today"), ("I can solve the problem", "She can solve the problem"), ("I will attend the meeting", "She will attend the meeting"), ("I am happy with the result", "She is happy with the result"), ("I have seen the movie", "She has seen the movie"), ("I was working on the task", "She was working on the task"), ("I am reading the book", "She is reading the book"), ("I have written the essay", "She has written the essay"), ("I am enjoying the food", "She is enjoying the food"), ("I have visited the city", "She has visited the city"), ("I was studying the material", "She was studying the material"), ("I am learning the language", "She is learning the language"), ("I have completed the project", "She has completed the project"), ("I am cooking dinner now", "She is cooking dinner now"), ("I have bought the tickets", "She has bought the tickets"), ("I was watching the show", "She was watching the show"), ("I am feeling much better", "She is feeling much better"), ("I have heard the news", "She has heard the news")]
DEFINITENESS = [("A cat sat on the mat", "The cat sat on the mat"), ("A student read the book", "The student read the book"), ("A doctor treated patients", "The doctor treated patients"), ("A teacher explained the rule", "The teacher explained the rule"), ("A bird flew over the lake", "The bird flew over the lake"), ("A car drove down the road", "The car drove down the road"), ("A dog chased the ball", "The dog chased the ball"), ("A child played in the park", "The child played in the park"), ("A woman walked to work", "The woman walked to work"), ("A man opened the door", "The man opened the door"), ("A flower grew in spring", "The flower grew in spring"), ("A river flowed through town", "The river flowed through town"), ("A star shone in the sky", "The star shone in the sky"), ("A house stood on the hill", "The house stood on the hill"), ("A tree fell in the storm", "The tree fell in the storm"), ("A book was on the shelf", "The book was on the shelf"), ("A song played on the radio", "The song played on the radio"), ("A train arrived at the station", "The train arrived at the station"), ("A boat sailed across the sea", "The boat sailed across the sea"), ("A plane flew through clouds", "The plane flew through clouds")]
INFO_STRUCTURE = [("Mary broke the window", "It was Mary who broke the window"), ("John found the solution", "It was John who found the solution"), ("She wrote the report", "It was she who wrote the report"), ("They built the house", "It was they who built the house"), ("He solved the puzzle", "It was he who solved the puzzle"), ("We finished the project", "It was we who finished the project"), ("The dog chased the cat", "It was the dog that chased the cat"), ("The wind broke the branch", "It was the wind that broke the branch"), ("The rain ruined the crops", "It was the rain that ruined the crops"), ("The fire destroyed the building", "It was the fire that destroyed the building"), ("Tom designed the bridge", "It was Tom who designed the bridge"), ("Anna composed the music", "It was Anna who composed the music"), ("The team won the game", "It was the team that won the game"), ("The company launched the product", "It was the company that launched the product"), ("The teacher gave the assignment", "It was the teacher who gave the assignment"), ("Mike fixed the engine", "It was Mike who fixed the engine"), ("The student answered the question", "It was the student who answered the question"), ("Sarah painted the portrait", "It was Sarah who painted the portrait"), ("The manager approved the budget", "It was the manager who approved the budget"), ("The chef prepared the meal", "It was the chef who prepared the meal")]
SENTIMENT = [("The happy child played outside", "The sad child played outside"), ("She gave a wonderful performance", "She gave a terrible performance"), ("He found a beautiful solution", "He found an ugly solution"), ("They enjoyed the delicious meal", "They suffered through the awful meal"), ("The warm sun brightened the day", "The cold rain darkened the day"), ("She received a generous gift", "She received a stingy gift"), ("He spoke with gentle words", "He spoke with harsh words"), ("The kind woman helped everyone", "The cruel woman hurt everyone"), ("They celebrated the joyful occasion", "They mourned the tragic occasion"), ("The peaceful garden felt calm", "The violent storm felt chaotic"), ("She wore a lovely dress", "She wore a hideous dress"), ("He told a fascinating story", "He told a boring story"), ("The clean room looked perfect", "The dirty room looked awful"), ("They built a magnificent castle", "They built a shabby shack"), ("The sweet music filled the air", "The harsh noise filled the air"), ("She made a brilliant decision", "She made a foolish decision"), ("He showed great courage today", "He showed great cowardice today"), ("The bright stars lit the sky", "The dark clouds covered the sky"), ("They shared a pleasant evening", "They endured a miserable evening"), ("The fresh flowers smelled nice", "The rotten flowers smelled bad")]
SEMANTIC_TOPIC = [("The doctor examined the patient carefully", "The teacher examined the student carefully"), ("She cooked a delicious meal", "She programmed a complex algorithm"), ("The river flowed through the valley", "The current flowed through the circuit"), ("He planted seeds in the garden", "He invested money in the market"), ("The bird built a nest in spring", "The programmer built an app last year"), ("She painted a landscape with oils", "She designed a website with code"), ("The ship sailed across the ocean", "The data traveled across the network"), ("He climbed the mountain trail", "He solved the math equation"), ("The chef seasoned the soup perfectly", "The engineer calibrated the device perfectly"), ("She played violin in the orchestra", "She wrote code in the company"), ("The farmer harvested the wheat field", "The scientist analyzed the data field"), ("He rode his horse through the forest", "He drove his car through the city"), ("The artist sketched the portrait", "The architect drafted the blueprint"), ("She sang a beautiful melody", "She proved a beautiful theorem"), ("The fish swam in the clear lake", "The electron moved in the clear field"), ("He fixed the broken fence", "He debugged the broken software"), ("The baker kneaded the dough", "The writer drafted the chapter"), ("She grew roses in her garden", "She grew profits in her business"), ("The mechanic repaired the engine", "The doctor treated the patient"), ("He threw the ball to the catcher", "He sent the email to the manager")]
VOICE = [("The cat chased the mouse", "The mouse was chased by the cat"), ("She wrote the report yesterday", "The report was written by her yesterday"), ("He fixed the broken window", "The broken window was fixed by him"), ("They built the new bridge", "The new bridge was built by them"), ("The teacher explained the lesson", "The lesson was explained by the teacher"), ("She cooked the delicious meal", "The delicious meal was cooked by her"), ("He painted the entire house", "The entire house was painted by him"), ("They discovered the ancient ruin", "The ancient ruin was discovered by them"), ("The company launched the product", "The product was launched by the company"), ("She translated the document", "The document was translated by her"), ("He designed the new logo", "The new logo was designed by him"), ("They organized the charity event", "The charity event was organized by them"), ("The chef prepared the special dish", "The special dish was prepared by the chef"), ("She composed the beautiful song", "The beautiful song was composed by her"), ("He directed the award-winning film", "The award-winning film was directed by him"), ("They published the research paper", "The research paper was published by them"), ("The artist created the sculpture", "The sculpture was created by the artist"), ("She managed the large project", "The large project was managed by her"), ("He developed the software tool", "The software tool was developed by him"), ("They renovated the old building", "The old building was renovated by them")]
FORMALITY = [("It is imperative that we proceed", "We really need to get going"), ("I would like to request assistance", "I need some help"), ("The individual in question departed", "The guy left"), ("We shall commence the operation", "We'll start the job"), ("Please refrain from smoking herein", "Don't smoke here"), ("I am unable to attend the function", "I can't make it to the party"), ("The aforementioned document requires", "That paper needs"), ("It is recommended that you comply", "You should do it"), ("We regret to inform you that", "Sorry but we have to say"), ("The consumption of beverages is prohibited", "No drinks allowed"), ("I respectfully decline the invitation", "I'm passing on the invite"), ("Kindly submit your documentation", "Please send your stuff"), ("The procedure necessitates careful attention", "You gotta be careful with this"), ("We hereby acknowledge your correspondence", "We got your message"), ("It is advisable to consult a professional", "You should ask a pro"), ("The facility will be closed temporarily", "The place is closed for now"), ("I wish to express my gratitude", "Thanks a lot"), ("The meeting has been rescheduled", "The meeting got moved"), ("Please ensure punctuality for the event", "Don't be late"), ("We appreciate your cooperation in this matter", "Thanks for working with us on this")]

ALL_FEATURES = {
    'tense': TENSE, 'polarity': POLARITY, 'number': NUMBER,
    'negation': NEGATION, 'question': QUESTION, 'person': PERSON,
    'definiteness': DEFINITENESS, 'info_structure': INFO_STRUCTURE,
    'sentiment': SENTIMENT, 'semantic_topic': SEMANTIC_TOPIC,
    'voice': VOICE, 'formality': FORMALITY,
}

SYNTACTIC = ['tense', 'polarity', 'number', 'negation', 'question', 'person', 'definiteness', 'info_structure']
SEMANTIC = ['sentiment', 'semantic_topic', 'voice', 'formality']

SUFFIXES = ["", " at this point", " in the end", " for sure", " without doubt", " as expected"]

def gen_pairs(templates, n):
    np.random.seed(42)
    pairs = []
    for i in range(n):
        idx = i % len(templates)
        a, b = templates[idx]
        if np.random.random() < 0.3:
            s = SUFFIXES[np.random.randint(len(SUFFIXES))]
            pairs.append((a + s, b + s))
        else:
            pairs.append((a, b))
    return pairs[:n]


def get_residual_vector(model, tokenizer, device, layer_idx, text):
    """Get residual stream at a layer - proven method from CCXIII."""
    try:
        input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
        activations = {}
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                activations['h'] = out[0][0, -1, :].detach().clone()
            else:
                activations['h'] = out[0, -1, :].detach().clone()

        layer = model.model.layers[layer_idx]
        h = layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(input_ids)
        h.remove()

        if 'h' in activations:
            return activations['h'].float().cpu().numpy()
        return None
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=80)
    args = parser.parse_args()

    out_dir = Path(f"results/causal_fiber/{args.model}_ccxiv")
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(out_dir / 'run.log')

    path = PATHS[args.model]
    print(f"\n{'='*60}")
    print(f"Phase CCXIV v3: {args.model} (n_pairs={args.n_pairs})")
    print(f"Path: {path}")
    print(f"{'='*60}")

    print(f"[{time.strftime('%H:%M:%S')}] Loading model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model in ["glm4", "deepseek7b"]:
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
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"[{time.strftime('%H:%M:%S')}] Loaded in {time.time()-t0:.1f}s: n_layers={n_layers}, n_heads={n_heads}, device={device}", flush=True)

    N = args.n_pairs
    sample_layers = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
    print(f"Sample layers (4): {sample_layers}", flush=True)

    all_pairs = {}
    for feat_name, templates in ALL_FEATURES.items():
        all_pairs[feat_name] = gen_pairs(templates, N)

    # ===== S1: 收集差分向量 + W_o Head分解 =====
    print(f"\n{'='*60}")
    print(f"S1: 差分向量 + W_o Head分解")
    print(f"{'='*60}", flush=True)

    delta_vectors = {}  # delta_vectors[feat][layer] = (n, d)
    head_contribs = {}  # head_contribs[feat][layer] = {H0: frac, ...}

    # Pre-extract W_o for all layers
    W_o_by_layer = {}
    for layer_idx in sample_layers:
        layer = model.model.layers[layer_idx]
        try:
            W_o = layer.self_attn.o_proj.weight.detach().float().cpu().numpy()
            if W_o is not None and W_o.size > 0:
                W_o_by_layer[layer_idx] = W_o
        except:
            pass

    for layer_idx in sample_layers:
        layer_name = f'L{layer_idx}'
        t_layer = time.time()
        has_W_o = layer_idx in W_o_by_layer

        for feat_name in list(ALL_FEATURES.keys()):
            if feat_name not in delta_vectors:
                delta_vectors[feat_name] = {}
                head_contribs[feat_name] = {}

            pairs = all_pairs[feat_name]
            n_test = min(len(pairs), N)
            deltas = []

            for i in range(n_test):
                a_text, b_text = pairs[i]
                h_a = get_residual_vector(model, tokenizer, device, layer_idx, a_text)
                h_b = get_residual_vector(model, tokenizer, device, layer_idx, b_text)
                if h_a is not None and h_b is not None:
                    deltas.append(h_a - h_b)
                if i > 0 and i % 20 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if deltas:
                dv = np.array(deltas)
                delta_vectors[feat_name][layer_name] = dv

                # W_o Head contribution
                if has_W_o:
                    W_o = W_o_by_layer[layer_idx]
                    d_model = W_o.shape[0]
                    head_dim = d_model // n_heads
                    mean_delta = dv.mean(axis=0)
                    delta_norm = np.linalg.norm(mean_delta)
                    if delta_norm > 1e-8:
                        hcontribs = {}
                        for h in range(n_heads):
                            W_h = W_o[:, h * head_dim:(h + 1) * head_dim]
                            proj = W_h @ (W_h.T @ mean_delta)
                            hcontribs[f'H{h}'] = np.linalg.norm(proj) / delta_norm
                        head_contribs[feat_name][layer_name] = hcontribs

        elapsed = time.time() - t_layer
        parts = []
        for feat_name in ['tense', 'sentiment', 'voice']:
            if layer_name in delta_vectors.get(feat_name, {}):
                med_l2 = float(np.median(np.linalg.norm(delta_vectors[feat_name][layer_name], axis=1)))
                n = len(delta_vectors[feat_name][layer_name])
                parts.append(f"{feat_name}={med_l2:.0f}(n={n})")
        print(f"  {layer_name} [{elapsed:.0f}s] W_o={'Y' if has_W_o else 'N'}: {', '.join(parts)}", flush=True)

    # ===== S2: Head Specialization =====
    print(f"\n{'='*60}\nS2: Head Specialization\n{'='*60}", flush=True)

    for layer_idx in sample_layers:
        layer_name = f'L{layer_idx}'
        if layer_name not in W_o_by_layer:
            print(f"  {layer_name}: W_o unavailable, skipping")
            continue

        syn_head_scores = {f'H{h}': [] for h in range(n_heads)}
        sem_head_scores = {f'H{h}': [] for h in range(n_heads)}

        for feat_name in SYNTACTIC:
            if layer_name in head_contribs.get(feat_name, {}):
                for h_name, frac in head_contribs[feat_name][layer_name].items():
                    syn_head_scores[h_name].append(frac)

        for feat_name in SEMANTIC:
            if layer_name in head_contribs.get(feat_name, {}):
                for h_name, frac in head_contribs[feat_name][layer_name].items():
                    sem_head_scores[h_name].append(frac)

        specs = {}
        for h in range(n_heads):
            h_name = f'H{h}'
            syn_m = np.mean(syn_head_scores[h_name]) if syn_head_scores[h_name] else 0
            sem_m = np.mean(sem_head_scores[h_name]) if sem_head_scores[h_name] else 0
            total = syn_m + sem_m
            spec = (syn_m - sem_m) / total if total > 0 else 0
            specs[h_name] = {'syn': float(syn_m), 'sem': float(sem_m), 'spec': float(spec),
                             'cat': 'SYN' if spec > 0.1 else ('SEM' if spec < -0.1 else 'MIX')}

        n_syn = sum(1 for v in specs.values() if v['cat'] == 'SYN')
        n_sem = sum(1 for v in specs.values() if v['cat'] == 'SEM')
        n_mix = sum(1 for v in specs.values() if v['cat'] == 'MIX')

        top_syn = sorted(specs.items(), key=lambda x: x[1]['spec'], reverse=True)[:3]
        top_sem = sorted(specs.items(), key=lambda x: x[1]['spec'])[:3]
        ts = ', '.join([h + '=' + str(round(v['spec'], 3)) for h, v in top_syn])
        te = ', '.join([h + '=' + str(round(v['spec'], 3)) for h, v in top_sem])

        print(f"  {layer_name}: SYN={n_syn}, SEM={n_sem}, MIX={n_mix}")
        print(f"    Top SYN: [{ts}]")
        print(f"    Top SEM: [{te}]")

    # ===== S3: 跨层Head功能一致性 =====
    print(f"\n{'='*60}\nS3: 跨层Head功能一致性\n{'='*60}", flush=True)

    for h in range(min(n_heads, 8)):
        h_name = f'H{h}'
        specs_across = []
        for layer_idx in sample_layers:
            ln = f'L{layer_idx}'
            if ln in head_contribs.get('tense', {}) and ln in head_contribs.get('sentiment', {}):
                syn_f = head_contribs['tense'][ln].get(h_name, 0)
                sem_f = head_contribs['sentiment'][ln].get(h_name, 0)
                t = syn_f + sem_f
                spec = (syn_f - sem_f) / t if t > 0 else 0
                specs_across.append(spec)
            else:
                specs_across.append(0)
        print(f"  {h_name}: spec=[{', '.join(str(round(s, 3)) for s in specs_across)}]")

    # ===== S4: 统计检验 =====
    print(f"\n{'='*60}\nS4: 统计检验\n{'='*60}", flush=True)

    # At final layer, compare SYN vs SEM head spec indices
    final_ln = f'L{sample_layers[-1]}'
    if final_ln in head_contribs.get('tense', {}):
        all_specs = []
        all_cats = []
        for h in range(n_heads):
            h_name = f'H{h}'
            syn_fs = [head_contribs[f][final_ln].get(h_name, 0) for f in SYNTACTIC if final_ln in head_contribs.get(f, {})]
            sem_fs = [head_contribs[f][final_ln].get(h_name, 0) for f in SEMANTIC if final_ln in head_contribs.get(f, {})]
            syn_m = np.mean(syn_fs) if syn_fs else 0
            sem_m = np.mean(sem_fs) if sem_fs else 0
            total = syn_m + sem_m
            if total > 0:
                spec = (syn_m - sem_m) / total
                all_specs.append(spec)
                all_cats.append('SYN' if spec > 0.1 else ('SEM' if spec < -0.1 else 'MIX'))

        syn_specs = [s for s, c in zip(all_specs, all_cats) if c == 'SYN']
        sem_specs = [s for s, c in zip(all_specs, all_cats) if c == 'SEM']
        if syn_specs and sem_specs:
            from scipy import stats as ss
            u, p = ss.mannwhitneyu(syn_specs, sem_specs, alternative='greater')
            print(f"  SYN vs SEM head spec: U={u:.1f}, p={p:.6f}")
            print(f"  SYN heads: n={len(syn_specs)}, mean={np.mean(syn_specs):.3f}")
            print(f"  SEM heads: n={len(sem_specs)}, mean={np.mean(sem_specs):.3f}")

    # ===== 保存 =====
    all_results = {
        'delta_norms': {
            feat: {ln: float(np.median(np.linalg.norm(delta_vectors[feat][ln], axis=1)))
                   for ln in delta_vectors[feat]}
            for feat in delta_vectors
        },
        'head_contribs': head_contribs,
    }

    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        return obj

    with open(out_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    # ===== FINAL =====
    print(f"\n{'='*60}\nFINAL: {args.model}\n{'='*60}", flush=True)

    print(f"\n--- Delta Norms ---")
    for feat_name in SYNTACTIC + SEMANTIC:
        norms = []
        for layer_idx in sample_layers:
            ln = f'L{layer_idx}'
            if ln in delta_vectors.get(feat_name, {}):
                norms.append(str(round(float(np.median(np.linalg.norm(delta_vectors[feat_name][ln], axis=1))), 0)))
            else:
                norms.append('?')
        cat = 'SYN' if feat_name in SYNTACTIC else 'SEM'
        print(f"  {feat_name} [{cat}]: {', '.join(norms)}")

    print(f"\nDONE! Saved to {out_dir}")


if __name__ == '__main__':
    main()
