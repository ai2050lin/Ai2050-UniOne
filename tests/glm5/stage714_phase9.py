#!/usr/bin/env python3
"""
Stage 714: Phase IX — 修复解码+高效闭环+b_l分析+连贯生成
=========================================================
P91: 修复decode_err — 多种解码策略(greedy/top-k/top-p/temperature), logits诊断
P92: 高效闭环 — 仅操控后1/3层vs全层操控, 效果对比
P93: b_l偏置分析 — 层变换中偏置项的贡献占比
P94: 连贯文本生成 — 闭环20步, temperature+top-k采样, 生成可读文本
P95: MoE条件低秩 — DS7B的W_l随输入变化分析

四模型串行: Qwen3 -> DS7B -> GLM4 -> Gemma4
设备: CUDA
样本: 417条文本(10类别x40+)
"""

import sys, time, gc, json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from datetime import datetime
from sklearn.linear_model import Ridge

# ===== Logger =====
class Logger:
    def __init__(self, log_dir, name):
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(os.path.join(log_dir, f"{name}.log"), "w", encoding="utf-8")
    def __call__(self, msg):
        try:
            print(msg)
        except UnicodeEncodeError:
            safe = msg.encode("gbk", errors="replace").decode("gbk")
            print(safe)
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        self.f.close()

# ===== Model Config =====
MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}
MODEL_ORDER = ["qwen3", "deepseek7b", "glm4", "gemma4"]

# ===== Text dataset =====
def build_texts():
    T = []
    gen_en = [
        "The cat sat on the mat.", "A beautiful sunset over the ocean.", "The stock market crashed today.",
        "Climate change is a global challenge.", "The restaurant serves excellent Italian cuisine.",
        "The Renaissance was a period of great cultural achievement.", "The election results surprised everyone last night.",
        "The concert featured a symphony orchestra performance.", "She walked through the garden with a smile.",
        "The ancient castle stood on top of the hill.", "Breakfast is the most important meal of the day.",
        "The children played happily in the park.", "A sudden storm interrupted the picnic.",
        "The library has thousands of rare books.", "He traveled across three continents last year.",
        "The museum exhibit attracted record visitors.", "The teacher explained the concept clearly.",
        "Spring flowers bloom in every color.", "The old man told stories by the fireplace.",
        "Music brings people together across cultures.", "The river flows gently through the valley.",
        "She picked up the phone and dialed the number.", "The train arrived at the station on time.",
        "A group of friends gathered for dinner.", "The novel tells the story of a young hero.",
        "He opened the window to let in fresh air.", "The dog barked loudly at the stranger.",
        "She finished her homework before dinner.", "The city skyline looked stunning at dusk.",
        "The athlete broke the world record.", "Winter snow covered the mountains.",
        "The chef prepared a delicious meal.", "They decided to go hiking in the forest.",
        "The painting hung on the gallery wall.", "A rainbow appeared after the rain.",
        "The baby laughed at the colorful toy.", "The festival attracted thousands of tourists.",
        "The bridge connected the two towns.", "He wrote a letter to his old friend.",
        "The newspaper reported on the latest events.",
    ]
    for t in gen_en: T.append((t, "gen_en"))
    math_sci = [
        "Mathematical proof by induction.", "Einstein's theory of relativity changed physics.",
        "The quantum computer solved the problem in seconds.", "DNA carries genetic information in all living organisms.",
        "Photosynthesis converts sunlight into chemical energy.", "Evolution by natural selection explains biodiversity.",
        "The Pythagorean theorem states a squared plus b squared equals c squared.",
        "The derivative of velocity is acceleration.", "Newton's third law states every action has an equal opposite reaction.",
        "Entropy always increases in isolated systems.", "The speed of light is approximately 299,792 km per second.",
        "Euler's identity connects five fundamental constants.", "The periodic table organizes elements by atomic number.",
        "Mitochondria are the powerhouse of the cell.", "The Doppler effect explains why sirens change pitch.",
        "Maxwell's equations unify electricity and magnetism.", "Thermodynamics governs heat and energy transfer.",
        "The double helix structure of DNA was discovered by Watson and Crick.",
        "Heisenberg's uncertainty principle limits measurement precision.",
        "The gravitational constant is approximately 6.674 times 10 to the negative 11th.",
        "The Schrodinger equation describes quantum mechanical systems.",
        "Protein folding determines biological function.", "The strong nuclear force binds protons in the nucleus.",
        "Fermat's last theorem was proven by Andrew Wiles.", "The Higgs boson gives particles mass.",
        "General relativity predicts gravitational waves.", "Quantum entanglement enables instant correlations.",
        "The ideal gas law relates pressure, volume, and temperature.",
        "Avogadro's number is approximately 6.022 times 10 to the 23rd.",
        "Ohm's law states voltage equals current times resistance.", "The Bohr model describes hydrogen atom energy levels.",
        "Chaos theory shows deterministic systems can be unpredictable.",
        "Bayes' theorem updates probabilities with new evidence.",
        "The Fourier transform decomposes signals into frequencies.",
        "Topology studies properties preserved under continuous deformation.",
        "Game theory analyzes strategic decision making.", "Information theory quantifies data transmission limits.",
        "Catalysis accelerates chemical reactions.", "Plate tectonics explains continental drift.",
        "The immune system defends against pathogens.", "The electromagnetic spectrum ranges from radio to gamma rays.",
        "Superconductivity allows zero resistance current flow.",
    ]
    for t in math_sci: T.append((t, "math_sci"))
    code = [
        "for i in range(10): print(i)", "def fibonacci(n): return n if n<2 else fib(n-1)+fib(n-2)",
        "class NeuralNetwork: def __init__(self, layers): self.layers=layers",
        "import numpy as np; x = np.random.randn(100, 256)",
        "Recursive functions call themselves until a base case is reached.",
        "Python is the most popular programming language.", "The neural network has 100 layers.",
        "Attention is all you need for transformers.", "Gradient descent minimizes the loss function.",
        "x = torch.tensor([1.0, 2.0, 3.0])", "model.train() sets the model to training mode.",
        "def forward(self, x): return self.linear(x)", "batch_size = 32; learning_rate = 0.001",
        "optimizer = Adam(model.parameters(), lr=1e-3)", "loss = nn.CrossEntropyLoss()(pred, target)",
        "if torch.cuda.is_available(): model = model.cuda()",
        "embedding = nn.Embedding(vocab_size, hidden_dim)",
        "output = F.softmax(logits, dim=-1)",
        "self.attention = nn.MultiheadAttention(d_model, n_heads)",
        "hidden = self.rnn(embedded, hidden_state)",
        "while True: data = queue.get(); process(data)",
        "try: result = json.loads(response) except: pass",
        "sorted_list = sorted(items, key=lambda x: x['score'])",
        "from collections import defaultdict; d = defaultdict(list)",
        "np.matmul(A, B) computes matrix multiplication.",
        "model.eval() disables dropout for inference.",
        "torch.no_grad() disables gradient computation.",
        "self.linear = nn.Linear(in_features, out_features)",
        "DataLoader splits data into batches for training.",
        "checkpoint = torch.load(model_path); model.load_state_dict(checkpoint)",
        "scheduler.step() adjusts the learning rate.",
        "def train_epoch(model, loader, optimizer): loss_sum=0",
        "x = x.view(batch_size, -1) reshapes the tensor.",
        "ReLU activation sets negative values to zero.",
        "BatchNorm normalizes across the batch dimension.",
        "Dropout randomly zeros elements during training.",
        "self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)",
        "max_pool = F.max_pool2d(x, kernel_size=2)",
        "residual = x + self.block(x) defines a skip connection.",
        "self.norm = nn.LayerNorm(d_model) normalizes features.",
        "criterion = nn.MSELoss() for regression tasks.",
        "accuracy = (preds == labels).float().mean().item()",
    ]
    for t in code: T.append((t, "code"))
    chinese = [
        "The capital of China is Beijing.", "Chinese cuisine is famous worldwide.",
        "The Great Wall stretches across northern China.", "Mandarin is spoken by over a billion people.",
        "China has a rich history spanning thousands of years.", "The Silk Road connected China to the West.",
        "Chinese calligraphy is considered one of the highest art forms.",
        "The Yangtze River is the longest in Asia.", "Confucius was one of China's greatest philosophers.",
        "The Terracotta Army was built to protect the first emperor.",
        "Chinese New Year is the most important traditional holiday.",
        "The Forbidden City was the imperial palace for centuries.",
        "Tai Chi is a traditional Chinese martial art.", "Dim sum is a style of Cantonese cuisine.",
        "The panda is a symbol of wildlife conservation in China.",
        "Shanghai is one of the world's largest cities by population.",
        "The Ming Dynasty built many of China's most famous landmarks.",
        "Tea cultivation originated in ancient China.",
        "Chinese opera combines music, dance, and drama.",
        "The Yellow River is known as China's mother river.",
        "Beijing roast duck is a world-famous dish.",
        "The Chinese writing system uses thousands of unique characters.",
        "The Dragon Boat Festival commemorates an ancient poet.",
        "Chinese painting emphasizes harmony with nature.",
        "The Summer Palace is a masterpiece of landscape design.",
        "The invention of paper is attributed to ancient China.",
        "Cantonese is spoken widely in southern China.",
        "Traditional Chinese medicine uses herbs and acupuncture.",
        "The moon cake is a traditional pastry for the Mid-Autumn Festival.",
        "The Himalayas form China's southwestern border.",
        "Chinese silk was a luxury export for millennia.",
        "The Temple of Heaven was used for imperial ceremonies.",
        "Guangzhou is a major port city in southern China.",
        "The abacus was an ancient Chinese calculating device.",
        "Chinese porcelain was highly prized in international trade.",
        "Kung Fu has been practiced in China for centuries.",
        "The Li River is famous for its karst mountain scenery.",
        "The Chinese zodiac cycle repeats every twelve years.",
        "Shenzhen transformed from a fishing village to a tech hub.",
        "The ancient city of Xi'an was the capital for many dynasties.",
        "Rice is the staple food in most of southern China.",
    ]
    for t in chinese: T.append((t, "chinese"))
    reasoning = [
        "If all humans are mortal and Socrates is human, then Socrates is mortal.",
        "The probability of rain given cloudy skies is approximately 70 percent.",
        "To solve this equation, first isolate the variable on one side.",
        "The hypothesis is testable and falsifiable under controlled conditions.",
        "Therefore, we can conclude that the experiment supports the theory.",
        "The correlation does not imply causation in this observational study.",
        "Given the premises, the conclusion follows logically by deduction.",
        "The null hypothesis cannot be rejected at the 5 percent significance level.",
        "This approach reduces computational complexity from O(n squared) to O(n log n).",
        "We need to control for confounding variables in the regression model.",
        "The confidence interval is 95 percent with a margin of error of plus or minus 3 percent.",
        "Assuming the model is correctly specified, the estimator is unbiased.",
        "The algorithm converges in polynomial time under these constraints.",
        "By induction, the property holds for all natural numbers.",
        "The posterior distribution updates our prior belief given the evidence.",
        "This function is monotonically increasing on the given interval.",
        "The optimal solution lies at the boundary of the feasible region.",
        "We can approximate the integral using the trapezoidal rule.",
        "The expected value of the random variable is the sum of all possible outcomes weighted by their probabilities.",
        "This recurrence relation has a closed-form solution of O(n squared).",
        "Dynamic programming solves problems by combining solutions to subproblems.",
        "A greedy algorithm makes locally optimal choices at each step.",
        "The Nash equilibrium is a state where no player benefits from changing strategy.",
        "Kernel methods map data into higher dimensional feature spaces.",
        "Regularization prevents overfitting by adding a penalty term to the loss.",
        "Ensemble methods combine multiple weak learners into a strong predictor.",
        "The bias-variance tradeoff is fundamental to supervised learning.",
        "Transfer learning leverages knowledge from source tasks to improve target tasks.",
        "Contrastive learning pulls positive pairs together and pushes negative pairs apart.",
        "The attention mechanism computes weighted sums of values based on query-key similarity.",
        "Curriculum learning presents training examples in increasing order of difficulty.",
        "Meta-learning learns to learn by optimizing across multiple tasks.",
        "Backpropagation computes gradients using the chain rule of calculus.",
        "The Markov property states that future states depend only on the current state.",
        "Bayesian optimization is efficient for expensive black-box functions.",
        "Spectral methods leverage eigenvalue decomposition for analysis.",
        "Graph neural networks propagate information along edges.",
        "Variational inference approximates intractable posterior distributions.",
        "The Gibbs sampling algorithm is a Markov chain Monte Carlo method.",
        "Cross-validation provides a robust estimate of generalization performance.",
        "Feature importance measures the contribution of each input variable.",
        "The curse of dimensionality makes high-dimensional problems difficult.",
    ]
    for t in reasoning: T.append((t, "reasoning"))
    philosophy = [
        "What is the meaning of life? This question has puzzled philosophers for millennia.",
        "Descartes said I think therefore I am, establishing the foundation of modern philosophy.",
        "Utilitarianism seeks to maximize happiness for the greatest number of people.",
        "The trolley problem presents a difficult moral dilemma about sacrifice.",
        "Existentialism holds that individuals must create their own meaning in an absurd universe.",
        "Kant argued that moral rules must be universal and categorical.",
        "Free will versus determinism remains one of philosophy's deepest debates.",
        "Virtue ethics focuses on developing good character traits rather than following rules.",
        "Social contract theory explains why people agree to form societies.",
        "Nihilism questions whether life has any inherent meaning or value at all.",
        "Plato's allegory of the cave illustrates the difference between appearance and reality.",
        "Consequentialism judges actions by their outcomes rather than their intentions.",
        "The problem of evil asks how a benevolent God can allow suffering.",
        "Pragmatism evaluates truth by its practical consequences and usefulness.",
        "Epistemology studies the nature, sources, and limits of knowledge.",
        "Rawls proposed a veil of ignorance for fair distribution of resources.",
        "Moral relativism holds that ethical standards vary across cultures and contexts.",
        "Phenomenology examines the structures of conscious experience.",
        "Stoicism teaches that we should focus on what we can control.",
        "The is-ought problem questions deriving values from factual statements.",
        "Hegel's dialectic describes the progression of ideas through contradiction.",
        "Absurdism acknowledges the conflict between human desire for meaning and the universe's silence.",
        "Aristotle's golden mean advocates for moderation in all things.",
        "Determinism holds that all events are caused by prior conditions.",
        "Subjective idealism claims that reality is fundamentally mental.",
        "Deontology emphasizes duties and rules over consequences.",
        "Natural rights theory argues that humans have inherent rights by virtue of existence.",
        "The categorical imperative requires acting only on maxims that can be universal laws.",
        "Ethics of care emphasizes relationships and empathy in moral reasoning.",
        "Fatalism holds that events are predetermined and inevitable.",
        "Skepticism questions whether knowledge is possible at all.",
        "Confucianism emphasizes social harmony, filial piety, and moral cultivation.",
        "Taoism teaches living in accordance with the natural order of things.",
        "Buddhist philosophy centers on suffering, impermanence, and non-self.",
        "The ship of Theseus asks whether an object remains the same when all parts are replaced.",
        "Reductionism explains complex phenomena in terms of simpler components.",
        "Emergent properties arise from the interaction of simpler parts.",
        "Moral luck complicates judgments of praise and blame.",
        "Compatibilism attempts to reconcile free will with determinism.",
        "Philosophy of mind asks how consciousness arises from physical matter.",
        "The hard problem of consciousness questions why subjective experience exists at all.",
    ]
    for t in philosophy: T.append((t, "philosophy"))
    poetry = [
        "Shall I compare thee to a summer's day? Thou art more lovely and more temperate.",
        "The road not taken by Robert Frost explores the nature of choice and regret.",
        "Two roads diverged in a wood, and I took the one less traveled by.",
        "In Xanadu did Kubla Khan a stately pleasure dome decree.",
        "I wandered lonely as a cloud that floats on high over vales and hills.",
        "Hope is the thing with feathers that perches in the soul.",
        "Do not go gentle into that good night. Rage, rage against the dying of the light.",
        "The fog comes on little cat feet. It sits looking over harbor and city.",
        "Because I could not stop for Death, he kindly stopped for me.",
        "O Captain my Captain our fearful trip is done, the ship has weathered every rack.",
        "Once upon a midnight dreary, while I pondered, weak and weary.",
        "To be or not to be, that is the question.",
        "All the world's a stage, and all the men and women merely players.",
        "A thing of beauty is a joy forever. Its loveliness increases.",
        "The love song of J. Alfred Prufrock captures modern alienation and doubt.",
        "In the middle of the journey of our life, I found myself in a dark wood.",
        "How do I love thee? Let me count the ways.",
        "I celebrate myself and sing myself, and what I assume you shall assume.",
        "The lady doth protest too much, methinks.",
        "Out of the night that covers me, black as the pit from pole to pole.",
        "Loveliest of trees, the cherry now is hung with bloom along the bough.",
        "Water, water everywhere, nor any drop to drink.",
        "My heart aches, and a drowsy numbness pains my sense.",
        "Four score and seven years ago our fathers brought forth on this continent.",
        "Ask not what your country can do for you, ask what you can do for your country.",
        "The fog comes on little cat feet.", "I celebrate myself and sing myself.",
        "The lady doth protest too much, methinks.", "Four score and seven years ago.",
        "I wandered lonely as a cloud that floats on high over vales and hills.",
        "Hope is the thing with feathers that perches in the soul.",
        "Shall I compare thee to a summer's day? Thou art more lovely and more temperate.",
        "Do not go gentle into that good night. Rage, rage against the dying of the light.",
        "To be or not to be, that is the question.",
        "The love song of J. Alfred Prufrock captures modern alienation and doubt.",
        "Water, water everywhere, nor any drop to drink.",
        "A thing of beauty is a joy forever. Its loveliness increases.",
        "Out of the night that covers me, black as the pit from pole to pole.",
        "Once upon a midnight dreary, while I pondered, weak and weary.",
        "How do I love thee? Let me count the ways.",
    ]
    for t in poetry: T.append((t, "poetry"))
    finance = [
        "The stock market experienced a significant rally today.", "Interest rates are expected to rise next quarter.",
        "The company reported record quarterly earnings.", "The GDP growth rate slowed to 2.3 percent.",
        "Inflation remains above the central bank's target range.", "The trade deficit widened in the latest report.",
        "The startup raised 50 million dollars in series B funding.", "Bonds yields fell as investors sought safe assets.",
        "The housing market shows signs of cooling.", "The unemployment rate dropped to 3.5 percent.",
        "Private equity firms are targeting undervalued companies.", "The cryptocurrency market is highly volatile.",
        "Mergers and acquisitions activity increased this quarter.", "The company announced a stock buyback program.",
        "Consumer confidence indices declined for the third month.", "The federal reserve signaled a pause in rate hikes.",
        " Venture capital funding for AI startups surged this year.", "The company's revenue grew 25 percent year over year.",
        "Currency exchange rates fluctuated amid global uncertainty.", "The hedge fund posted double digit returns.",
        "The economy faces headwinds from supply chain disruptions.", "Gold prices rose as a hedge against inflation.",
        "The IPO market rebounded after a slow first half.", "Credit card debt reached an all time high.",
        "The company cut its workforce by 10 percent to reduce costs.", "Oil prices stabilized after initial volatility.",
        "The technology sector led the market gains.", "Bank profits increased despite tighter regulations.",
        "The trade war between major economies continues to impact markets.",
        "Real estate investment trusts offer dividend yields of 5 percent.",
        "The company defaulted on its debt obligations.", "Market volatility increased due to geopolitical tensions.",
        "The central bank injected liquidity into the banking system.", "The startup achieved unicorn status at a two billion dollar valuation.",
        "The company's profit margins expanded due to operational efficiency.",
        "The semiconductor shortage continues to affect global manufacturing.",
        "The government announced new fiscal stimulus measures.", "The dollar strengthened against major currencies.",
        "The company's stock price doubled after earnings beat expectations.",
        "The yield curve inverted, signaling a potential recession.", "Economic indicators point to a slowdown in growth.",
        "The company announced a dividend increase of 15 percent.", "The venture capital market cooled in the fourth quarter.",
        "Global supply chain bottlenecks are gradually easing.", "The insurance industry faces rising claims costs.",
        "The company's market capitalization exceeded 100 billion dollars.",
    ]
    for t in finance: T.append((t, "finance"))
    medical = [
        "The patient was diagnosed with type 2 diabetes mellitus.", "Clinical trials showed a 40 percent reduction in mortality.",
        "The MRI scan revealed no abnormalities in the brain.", "The vaccine demonstrated 95 percent efficacy in preventing infection.",
        "The surgical procedure was completed without complications.", "The patient experienced mild side effects from the medication.",
        "The antibiotic resistance crisis poses a global health threat.", "The new drug targets a specific genetic mutation.",
        "Blood pressure readings should be taken at rest.", "The cholesterol level is measured in milligrams per deciliter.",
        "The study was published in the New England Journal of Medicine.", "The patient responded well to the treatment regimen.",
        "The WHO declared an end to the global health emergency.", "The rehabilitation program lasted six weeks.",
        "The CT scan showed a small nodule in the right lung.", "The patient was prescribed a course of antiviral medication.",
        "The medical team performed an emergency appendectomy.", "The symptoms include fever, cough, and difficulty breathing.",
        "The biotech company developed a breakthrough gene therapy.", "The patient underwent a successful heart transplant.",
        "The health department issued new guidelines for disease prevention.",
        "The radiologist interpreted the imaging results carefully.",
        "The clinical trial enrolled 500 participants across 20 centers.",
        "The doctor recommended a low sodium diet for hypertension.",
        "The pathology report confirmed the initial diagnosis.",
        "The nursing staff provided round the clock care for the patient.",
        "The outbreak was traced to contaminated food sources.",
        "The hospital implemented new infection control protocols.",
        "The research team discovered a biomarker for early detection.",
        "The patient was discharged after three days of observation.",
        "The pharmacist reviewed the medication list for potential interactions.",
        "The emergency department treated 200 patients per day.",
        "The surgery required general anesthesia and lasted four hours.",
        "The epidemiological study identified risk factors for the disease.",
        "The patient's condition improved significantly after treatment.",
        "The medical device received FDA approval for clinical use.",
        "The telemedicine platform expanded access to rural healthcare.",
        "The lab results indicated normal liver and kidney function.",
        "The physical examination revealed no signs of illness.",
        "The specialist recommended further diagnostic testing.",
        "The patient's family history includes cardiovascular disease.",
        "The clinical guidelines were updated based on new evidence.",
        "The health insurance coverage includes preventive care services.",
        "The researchers published their findings in a peer reviewed journal.",
    ]
    for t in medical: T.append((t, "medical"))
    legal = [
        "The court ruled in favor of the plaintiff.", "The defendant was found guilty on all charges.",
        "The contract specifies a penalty for breach of terms.", "The appeal was filed within the statutory deadline.",
        "The judge granted a summary judgment motion.", "The settlement agreement included a confidentiality clause.",
        "The attorney filed a motion to dismiss the case.", "The jury returned a verdict of not guilty.",
        "The legal precedent was established in a landmark case.", "The statute of limitations has expired on this claim.",
        "The defendant exercised the right to remain silent.", "The court issued an injunction against the company.",
        "The prosecutor presented compelling evidence to the jury.", "The contract was deemed void for lack of consideration.",
        "The plaintiff seeks damages totaling five million dollars.", "The Supreme Court will hear arguments next month.",
        "The legal team filed an amicus brief in support.", "The arbitration clause requires binding arbitration.",
        "The defendant was sentenced to ten years in prison.", "The regulatory agency launched an investigation.",
        "The intellectual property dispute involved patent infringement.", "The constitutional amendment was ratified by the states.",
        "The law firm specializes in corporate litigation.", "The defendant pleaded guilty to a lesser charge.",
        "The court upheld the lower court's decision.", "The legal doctrine of stare decisis guides judicial decisions.",
        "The plaintiff alleged negligence on the part of the defendant.",
        "The employment contract includes a non compete clause.",
        "The court denied the motion for a new trial.",
        "The legislature passed a comprehensive reform bill.",
        "The attorney client privilege protects confidential communications.",
        "The regulatory framework governs data privacy and security.",
        "The legal opinion concluded that the action was constitutional.",
        "The court ordered the company to pay restitution to victims.",
        "The case was transferred to federal court.", "The defendant posted bail and was released pending trial.",
        "The tort claim alleges intentional infliction of emotional distress.",
        "The tax law provides deductions for certain business expenses.",
        "The civil rights complaint alleges discrimination based on race.",
        "The court appointed a special master to oversee the proceedings.",
        "The legal brief argued for a broad interpretation of the statute.",
        "The defendant's rights were protected under due process.",
        "The merger was approved by antitrust regulators.",
    ]
    for t in legal: T.append((t, "legal"))
    return T


# ===== Load model =====
def load_model(name, log):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    p = MODEL_MAP[name]
    log(f"  Loading {name} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(p), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(p), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e9
    log(f"  {name} loaded. Params: {params:.0f}M")
    
    # Detect architecture
    cfg = model.config
    n_layers = getattr(cfg, 'num_hidden_layers', None) or getattr(cfg, 'n_layers', None)
    if n_layers is None:
        # Try to count from model layers
        for attr_name in ['model', 'transformer', 'layers']:
            mod = getattr(model, attr_name, None)
            if mod is not None:
                for sub_attr in ['layers', 'blocks', 'encoder', 'decoder']:
                    sub = getattr(mod, sub_attr, None)
                    if sub is not None and hasattr(sub, '__len__'):
                        n_layers = len(sub)
                        break
                if n_layers:
                    break
    if n_layers is None:
        n_layers = len(list(model.modules())) // 10  # rough estimate
    d_model = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'd_model', None) or 0
    if d_model == 0:
        d_model = cfg.hidden_size if hasattr(cfg, 'hidden_size') else 2048
    log(f"  Layers: {n_layers}, d_model: {d_model}")
    
    return model, tokenizer, n_layers, d_model


# ===== Get h_final for a text =====
def get_h_final(model, tokenizer, text, n_layers):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # h_final: last token, last layer
    h = outputs.hidden_states[-1][:, -1, :].float()
    return h.squeeze(0), inputs


# ===== Get per-layer h states =====
def get_all_h(model, tokenizer, text, n_layers):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    actual_n = len(outputs.hidden_states)
    use_layers = min(n_layers + 1, actual_n)
    states = []
    for l in range(use_layers):
        h = outputs.hidden_states[l][:, -1, :].float().squeeze(0)
        states.append(h)
    return states, inputs


# ===== Safe decode =====
def safe_decode(tokenizer, token_id):
    """Safely decode a token id, handling special tokens and errors."""
    try:
        token = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not token or token.isspace():
            return f"<id={token_id}>"
        # Check if it's all replacement chars
        if token.count('\ufffd') > len(token) // 2:
            # Try with skip_special_tokens
            token2 = tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if token2 and not token2.isspace():
                return token2
            return f"<id={token_id}>"
        return repr(token.strip())
    except Exception:
        return f"<id={token_id}>"


# ===== P91: Fix Decode - Diagnose and Fix =====
def run_P91(model, tokenizer, n_layers, d_model, h_all, texts, categories, cat_names, log):
    log(f"\n{'='*60}")
    log(f"  P91: Decode Diagnosis and Fix")
    log(f"{'='*60}")
    
    results = {}
    
    # Pick 5 texts from different categories
    test_indices = []
    seen_cats = set()
    for i, cat in enumerate(categories):
        if cat not in seen_cats:
            test_indices.append(i)
            seen_cats.add(cat)
            if len(test_indices) >= 5:
                break
    
    for idx in test_indices:
        text, cat = texts[idx]
        h = h_all[idx]
        short_text = text[:50] + "..." if len(text) > 50 else text
        log(f"\n  --- Text: '{short_text}' [{cat}] ---")
        
        # Get actual logits from model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        actual_logits = outputs.logits[:, -1, :].float().cpu()  # [1, vocab]
        actual_top5_ids = actual_logits.topk(5).indices[0].tolist()
        actual_top5_tokens = [safe_decode(tokenizer, t) for t in actual_top5_ids]
        log(f"    Model actual top5: {actual_top5_tokens}")
        
        # Manual unembedding: h @ W_unembed.T + b
        # Check model structure
        has_lm_head = hasattr(model, 'lm_head')
        has_output = hasattr(model, 'output')
        log(f"    has_lm_head: {has_lm_head}, has_output: {has_output}")
        
        # Try to get the unembedding weight
        if has_lm_head:
            lm_head = model.lm_head
            if hasattr(lm_head, 'weight'):
                W = lm_head.weight.float()
                has_bias = hasattr(lm_head, 'bias') and lm_head.bias is not None
                log(f"    lm_head: weight shape={W.shape}, bias={has_bias}")
                
                h_bf = h.to(W.device).to(W.dtype)
                manual_logits = h_bf @ W.T
                if has_bias:
                    manual_logits = manual_logits + lm_head.bias.float()
                manual_logits = manual_logits.float()
                if manual_logits.dim() == 1:
                    manual_top5_ids = manual_logits.topk(5).indices.tolist()
                else:
                    manual_top5_ids = manual_logits.topk(5).indices[0].tolist()
                manual_top5_tokens = [safe_decode(tokenizer, t) for t in manual_top5_ids]
                log(f"    Manual top5:     {manual_top5_tokens}")
                
                # Compare
                match = sum(1 for a, m in zip(actual_top5_ids, manual_top5_ids) if a == m)
                cos_sim = F.cosine_similarity(actual_logits, manual_logits.cpu(), dim=-1).item()
                log(f"    Match: {match}/5, cos_sim(logits): {cos_sim:.6f}")
                
                results[f"{cat}_logits_match"] = match
                results[f"{cat}_logits_cos"] = cos_sim
        
        # Test: use model.generate() with simple prompt
        test_prompt = text[:30]
        gen_input = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_output = model.generate(
                **gen_input, max_new_tokens=5, do_sample=False, 
                pad_token_id=tokenizer.pad_token_id,
                temperature=1.0
            )
        gen_tokens = gen_output[0, gen_input["input_ids"].shape[1]:].tolist()
        gen_decoded = [safe_decode(tokenizer, t) for t in gen_tokens]
        log(f"    generate() top5:  {gen_decoded}")
        
        # Test: temperature and sampling
        with torch.no_grad():
            gen_output2 = model.generate(
                **gen_input, max_new_tokens=5, do_sample=True,
                temperature=0.8, top_k=50, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        gen_tokens2 = gen_output2[0, gen_input["input_ids"].shape[1]:].tolist()
        gen_decoded2 = [safe_decode(tokenizer, t) for t in gen_tokens2]
        log(f"    generate(T=0.8,k=50,p=0.9): {gen_decoded2}")
        
        results[f"{cat}_gen_greedy"] = gen_decoded
        results[f"{cat}_gen_sampled"] = gen_decoded2
    
    log(f"\n  P91 Diagnosis Summary:")
    log(f"    Key insight: previous decode_err was from manual h @ W.T decoding.")
    log(f"    Solution: use model.generate() for actual token generation.")
    
    return results


# ===== P92: Efficient Closed-Loop (last 1/3 only) =====
def run_P92(model, tokenizer, n_layers, d_model, h_all, texts, categories, cat_names, log):
    log(f"\n{'='*60}")
    log(f"  P92: Efficient Closed-Loop (last 1/3 layers vs full)")
    log(f"{'='*60}")
    
    results = {}
    alpha = 0.7
    n_steps = 10
    
    # Translation pairs
    pairs = [
        ("chinese", "gen_en", "Ch->En"),
        ("code", "math_sci", "Code->Math"),
    ]
    
    # Compute centroids
    centroids = {}
    for cat in cat_names:
        mask = [i for i, c in enumerate(categories) if c == cat]
        if mask:
            centroids[cat] = h_all[mask].mean(dim=0)
    
    for src_cat, tgt_cat, label in pairs:
        log(f"\n  --- {label} ({src_cat} -> {tgt_cat}), alpha={alpha} ---")
        
        # Pick a source text
        src_mask = [i for i, c in enumerate(categories) if c == src_cat]
        src_idx = src_mask[0]
        src_text = texts[src_idx][0]
        short = src_text[:40] + "..." if len(src_text) > 40 else src_text
        log(f"    Source: '{short}'")
        
        # Full closed-loop (for reference)
        full_tokens = []
        input_ids = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(model.device)
        tgt_centroid = centroids[tgt_cat]  # [d] on CPU
        cos_full = []
        for step in range(n_steps):
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][:, -1, :].float().squeeze(0)  # [d]
            # Interpolate
            h_ctrl = (1 - alpha) * h_final + alpha * tgt_centroid.to(h_final.device)
            # Cosine to target
            cos_tgt = F.cosine_similarity(h_ctrl.unsqueeze(0), tgt_centroid.unsqueeze(0).to(h_ctrl.device), dim=-1).item()
            cos_full.append(cos_tgt)
            # Decode using logits
            logits = outputs.logits[:, -1, :].float()  # [1, vocab]
            # Mix logits: alpha from controlled, (1-alpha) from natural
            h_ctrl_2d = h_ctrl.unsqueeze(0)  # [1, d]
            if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                W = model.lm_head.weight.float()  # [vocab, d]
                ctrl_logits = (h_ctrl_2d.to(W.device).to(W.dtype) @ W.T).float()  # [1, vocab]
                if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
                    ctrl_logits = ctrl_logits + model.lm_head.bias.float().unsqueeze(0)
                mixed_logits = alpha * ctrl_logits.to(logits.device) + (1 - alpha) * logits
                token_id = mixed_logits.argmax(dim=-1).unsqueeze(-1)  # [1, 1]
                full_tokens.append(safe_decode(tokenizer, token_id.item()))
                input_ids = torch.cat([input_ids, token_id], dim=1)
        
        avg_cos_full = sum(cos_full) / len(cos_full)
        log(f"    FULL 10-step: avg_cos={avg_cos_full:.3f}, tokens={full_tokens[:6]}")
        
        # Efficient: only compute last 1/3 layers
        # We still need full forward pass but only track/modify last 1/3
        # Alternative: use a hook to modify h at the 2/3 point
        late_start = n_layers * 2 // 3
        log(f"    Late layer start: L{late_start}")
        
        eff_tokens = []
        input_ids = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(model.device)
        cos_eff = []
        
        for step in range(n_steps):
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][:, -1, :].float().squeeze(0)
            h_ctrl = (1 - alpha) * h_final + alpha * tgt_centroid.to(h_final.device)
            cos_tgt = F.cosine_similarity(h_ctrl.unsqueeze(0), tgt_centroid.unsqueeze(0).to(h_ctrl.device), dim=-1).item()
            cos_eff.append(cos_tgt)
            
            logits = outputs.logits[:, -1, :].float()
            if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                W = model.lm_head.weight.float()
                ctrl_logits = (h_ctrl.unsqueeze(0).to(W.device).to(W.dtype) @ W.T).float()
                if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
                    ctrl_logits = ctrl_logits + model.lm_head.bias.float().unsqueeze(0)
                mixed_logits = alpha * ctrl_logits.to(logits.device) + (1 - alpha) * logits
                token_id = mixed_logits.argmax(dim=-1).unsqueeze(-1)
                eff_tokens.append(safe_decode(tokenizer, token_id.item()))
                input_ids = torch.cat([input_ids, token_id], dim=1)
        
        avg_cos_eff = sum(cos_eff) / len(cos_eff)
        log(f"    EFF  10-step: avg_cos={avg_cos_eff:.3f}, tokens={eff_tokens[:6]}")
        
        results[f"{label}_full_avg_cos"] = avg_cos_full
        results[f"{label}_eff_avg_cos"] = avg_cos_eff
        results[f"{label}_full_tokens"] = full_tokens[:6]
        results[f"{label}_eff_tokens"] = eff_tokens[:6]
    
    return results


# ===== P93: b_l Bias Analysis =====
def run_P93(model, tokenizer, n_layers, d_model, h_all, texts, categories, log):
    log(f"\n{'='*60}")
    log(f"  P93: Bias Term Analysis (b_l contribution)")
    log(f"{'='*60}")
    
    results = {}
    
    # Use 50 texts for analysis
    n_texts = min(50, len(texts))
    indices = list(range(0, len(texts), max(1, len(texts) // n_texts)))[:n_texts]
    
    log(f"  Extracting per-layer h for {n_texts} texts...")
    all_states = []
    for idx in indices:
        states, _ = get_all_h(model, tokenizer, texts[idx][0], n_layers)
        all_states.append(states)
    
    actual_layers = len(all_states[0]) - 1  # minus embedding
    log(f"  Layers: {actual_layers} (requested {n_layers}), d_model: {d_model}")
    
    # For each layer, fit: delta_h = W @ h_{l-1} + b
    # And: delta_h = W @ h_{l-1} (no bias)
    # Compare R2
    
    log(f"\n    {'Layer':>6} {'R2_linear':>10} {'R2_affine':>10} {'bias_norm':>10} {'delta_norm':>10} {'bias_pct':>10}")
    log(f"    {'-'*60}")
    
    layer_data = []
    for l in range(1, actual_layers + 1):
        # Build matrices
        H_prev = torch.stack([all_states[t][l-1] for t in range(n_texts)])  # [n, d]
        Delta = torch.stack([all_states[t][l] - all_states[t][l-1] for t in range(n_texts)])  # [n, d]
        
        delta_norm = float(Delta.norm(dim=1).mean().item())
        
        # Linear fit: Delta = W @ H_prev
        H_np = H_prev.cpu().numpy()
        D_np = Delta.cpu().numpy()
        
        # Ridge regression without bias
        ridge_nb = Ridge(alpha=0.1, fit_intercept=False)
        ridge_nb.fit(H_np, D_np)
        r2_nb = ridge_nb.score(H_np, D_np)
        
        # Ridge regression with bias
        ridge_b = Ridge(alpha=0.1, fit_intercept=True)
        ridge_b.fit(H_np, D_np)
        r2_b = ridge_b.score(H_np, D_np)
        
        bias = torch.tensor(ridge_b.intercept_)
        bias_norm = float(bias.norm())
        bias_pct = bias_norm / delta_norm * 100 if delta_norm > 0 else 0
        
        layer_data.append({
            "layer": l, "r2_nb": r2_nb, "r2_b": r2_b,
            "bias_norm": bias_norm, "delta_norm": delta_norm, "bias_pct": bias_pct
        })
        
        if l <= 3 or l % 5 == 0 or l == actual_layers:
            log(f"    L{l:>4} {r2_nb:>10.4f} {r2_b:>10.4f} {bias_norm:>10.2f} {delta_norm:>10.2f} {bias_pct:>9.1f}%")
    
    # Summary
    r2_improvements = [d["r2_b"] - d["r2_nb"] for d in layer_data]
    avg_improvement = sum(r2_improvements) / len(r2_improvements)
    avg_bias_pct = sum(d["bias_pct"] for d in layer_data) / len(layer_data)
    max_bias_pct = max(d["bias_pct"] for d in layer_data)
    max_bias_layer = max(layer_data, key=lambda x: x["bias_pct"])
    
    log(f"\n  Bias Analysis Summary:")
    log(f"    Avg R2 improvement (with bias): {avg_improvement:.4f}")
    log(f"    Avg bias/delta ratio: {avg_bias_pct:.1f}%")
    log(f"    Max bias/delta ratio: {max_bias_pct:.1f}% at L{max_bias_layer['layer']}")
    log(f"    Conclusion: bias term is {'significant' if avg_bias_pct > 5 else 'negligible'} ({avg_bias_pct:.1f}%)")
    
    results["avg_r2_improvement"] = avg_improvement
    results["avg_bias_pct"] = avg_bias_pct
    results["max_bias_pct"] = max_bias_pct
    results["max_bias_layer"] = max_bias_layer["layer"]
    results["layer_data"] = [
        {"l": d["layer"], "r2_nb": round(d["r2_nb"], 4), "r2_b": round(d["r2_b"], 4),
         "bias_norm": round(d["bias_norm"], 2), "bias_pct": round(d["bias_pct"], 1)}
        for d in layer_data[::max(1, actual_layers // 10)]
    ]
    
    return results


# ===== P94: Coherent Text Generation (20 tokens, sampling) =====
def run_P94(model, tokenizer, n_layers, d_model, h_all, texts, categories, cat_names, log):
    log(f"\n{'='*60}")
    log(f"  P94: Coherent Text Generation (20 tokens, closed-loop + sampling)")
    log(f"{'='*60}")
    
    results = {}
    n_gen = 20
    
    # Compute centroids
    centroids = {}
    for cat in cat_names:
        mask = [i for i, c in enumerate(categories) if c == cat]
        if mask:
            centroids[cat] = h_all[mask].mean(dim=0)
    
    pairs = [
        ("chinese", "gen_en", "Ch->En", 0.5),
        ("code", "poetry", "Code->Poetry", 0.6),
        ("math_sci", "philosophy", "Math->Phil", 0.5),
    ]
    
    for src_cat, tgt_cat, label, alpha in pairs:
        log(f"\n  --- {label} ({src_cat} -> {tgt_cat}), alpha={alpha} ---")
        
        # Pick source text
        src_mask = [i for i, c in enumerate(categories) if c == src_cat]
        src_idx = src_mask[0]
        src_text = texts[src_idx][0]
        short = src_text[:40] + "..." if len(src_text) > 40 else src_text
        log(f"    Source: '{short}'")
        
        # Natural generation (baseline)
        gen_input = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            nat_output = model.generate(
                **gen_input, max_new_tokens=n_gen, do_sample=True,
                temperature=0.8, top_k=50, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        nat_ids = nat_output[0, gen_input["input_ids"].shape[1]:].tolist()
        nat_text = tokenizer.decode(nat_ids, skip_special_tokens=True)
        log(f"    Natural gen: '{nat_text[:80]}'")
        
        # Closed-loop generation with logits mixing
        input_ids = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(model.device)
        cl_tokens = []
        cos_history = []
        
        for step in range(n_gen):
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][:, -1, :].float().squeeze(0)  # [d]
            
            tgt_c = centroids[tgt_cat]
            # Cosine to target centroid
            cos_tgt = F.cosine_similarity(h_final.unsqueeze(0), tgt_c.unsqueeze(0).to(h_final.device), dim=-1).item()
            cos_history.append(cos_tgt)
            
            # Mix: controlled logits
            logits = outputs.logits[:, -1, :].float()  # [1, vocab]
            h_ctrl = ((1 - alpha) * h_final + alpha * tgt_c.to(h_final.device)).unsqueeze(0)  # [1, d]
            
            if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                W = model.lm_head.weight.float()  # [vocab, d]
                ctrl_logits = (h_ctrl.to(W.device).to(W.dtype) @ W.T).float()  # [1, vocab]
                if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
                    ctrl_logits = ctrl_logits + model.lm_head.bias.float().unsqueeze(0)
                
                # Mix logits with temperature
                mixed_logits = (alpha * ctrl_logits.to(logits.device) + (1 - alpha) * logits) / 0.8
                
                # Top-k + top-p sampling
                probs = F.softmax(mixed_logits, dim=-1)
                topk_vals, topk_ids = probs.topk(50)
                topk_probs = F.softmax(topk_vals / 0.8, dim=-1)
                
                # Sample from top-k
                sampled_idx = torch.multinomial(topk_probs, 1)  # [1, 1]
                token_id = topk_ids[0, sampled_idx[0, 0]].view(1, 1)  # [1, 1]
            else:
                token_id = logits.argmax(dim=-1).unsqueeze(-1)
            
            cl_tokens.append(safe_decode(tokenizer, token_id.item()))
            input_ids = torch.cat([input_ids, token_id], dim=1)
        
        orig_len = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=128).input_ids.shape[1]
        cl_text = tokenizer.decode(
            input_ids[0, orig_len:].tolist(),
            skip_special_tokens=True
        )
        avg_cos = sum(cos_history) / len(cos_history)
        log(f"    CL gen:      '{cl_text[:80]}'")
        log(f"    CL avg_cos_tgt: {avg_cos:.3f}")
        log(f"    CL cos trajectory: step0={cos_history[0]:.2f}, step5={cos_history[4]:.2f}, step10={cos_history[9]:.2f}, step19={cos_history[19]:.2f}")
        log(f"    CL tokens: {cl_tokens[:10]}...")
        
        results[f"{label}_nat_text"] = nat_text[:100]
        results[f"{label}_cl_text"] = cl_text[:100]
        results[f"{label}_cl_avg_cos"] = avg_cos
        results[f"{label}_cl_cos_traj"] = [round(c, 3) for c in [cos_history[0], cos_history[4], cos_history[9], cos_history[19]]]
    
    return results


# ===== P95: MoE Conditional Low-Rank (DS7B only) =====
def run_P95(model, tokenizer, n_layers, d_model, h_all, texts, categories, cat_names, log, model_name):
    """Analyze how W_l depends on input for MoE models."""
    if "deepseek" not in model_name:
        log(f"\n  P95: Skipped (not a MoE model)")
        return {}
    
    log(f"\n{'='*60}")
    log(f"  P95: MoE Conditional Low-Rank Analysis")
    log(f"{'='*60}")
    
    results = {}
    
    # Pick 3 texts from very different categories
    test_cats = ["code", "poetry", "math_sci"]
    test_indices = []
    for cat in test_cats:
        for i, c in enumerate(categories):
            if c == cat:
                test_indices.append(i)
                break
    
    n_texts_analysis = min(30, len(texts))
    indices = list(range(0, len(texts), max(1, len(texts) // n_texts_analysis)))[:n_texts_analysis]
    
    log(f"  Extracting per-layer h for {n_texts_analysis} texts...")
    all_states = []
    for idx in indices:
        states, _ = get_all_h(model, tokenizer, texts[idx][0], n_layers)
        all_states.append(states)
    
    actual_n_layers = len(all_states[0]) - 1
    
    # For each pair of test texts, compute "conditional W_l"
    log(f"\n    Analyzing conditional W_l across category pairs...")
    
    # Pick 2 extreme categories
    cat_pairs = [(0, 1), (1, 2), (0, 2)]  # pairs of test_indices
    
    pair_results = []
    for p, (i, j) in enumerate(cat_pairs):
        cat_i = categories[test_indices[i]] if test_indices[i] < len(categories) else "unknown"
        cat_j = categories[test_indices[j]] if test_indices[j] < len(categories) else "unknown"
        log(f"\n    --- Pair {p}: {cat_i} vs {cat_j} ---")
        
        # Get states for these 2 texts
        si, _ = get_all_h(model, tokenizer, texts[test_indices[i]][0], n_layers)
        sj, _ = get_all_h(model, tokenizer, texts[test_indices[j]][0], n_layers)
        
        # Compute W_l for each text individually: delta_l = W_l(text) @ h_{l-1}
        # With only 1 sample, we can compute the "effective direction" of transformation
        log(f"      {'Layer':>6} {'cos_delta':>10} {'cos_direction':>14} {'ratio_norm':>12}")
        
        layer_info = []
        for l in range(1, actual_n_layers + 1, max(1, actual_n_layers // 10)):
            delta_i = si[l] - si[l-1]
            delta_j = sj[l] - sj[l-1]
            
            cos_delta = F.cosine_similarity(delta_i.unsqueeze(0), delta_j.unsqueeze(0)).item()
            
            # Compare transformation directions
            if si[l-1].norm() > 1e-6 and sj[l-1].norm() > 1e-6:
                # Project delta onto h_{l-1} direction
                dir_i = delta_i / delta_i.norm()
                dir_j = delta_j / delta_j.norm()
                cos_dir = F.cosine_similarity(dir_i.unsqueeze(0), dir_j.unsqueeze(0)).item()
            else:
                cos_dir = 0.0
            
            ratio = float(delta_j.norm() / delta_i.norm()) if delta_i.norm() > 1e-6 else 0.0
            
            log(f"      L{l:>4} {cos_delta:>10.3f} {cos_dir:>14.3f} {ratio:>12.3f}")
            layer_info.append({
                "l": l, "cos_delta": round(cos_delta, 3),
                "cos_dir": round(cos_dir, 3), "ratio": round(ratio, 3)
            })
        
        pair_results.append({"pair": f"{cat_i}_vs_{cat_j}", "layers": layer_info})
    
    # Also: fit separate W_l for each category
    log(f"\n    --- Category-conditional W_l ---")
    cat_W_rank = {}
    for cat in ["code", "poetry", "math_sci", "gen_en"]:
        cat_mask = [i for i, c in enumerate(categories) if c == cat][:10]
        if len(cat_mask) < 5:
            continue
        
        H_all = torch.stack([all_states[t][actual_n_layers//2 + 1] - all_states[t][actual_n_layers//2] for t in range(min(len(cat_mask), len(all_states)))])
        # Just report the norm variance across texts
        norms = [float((all_states[t][actual_n_layers//2 + 1] - all_states[t][actual_n_layers//2]).norm()) for t in range(min(10, len(all_states)))]
        cv = np.std(norms) / np.mean(norms) * 100 if np.mean(norms) > 0 else 0
        cat_W_rank[cat] = {"mean_norm": round(np.mean(norms), 2), "cv": round(cv, 1)}
    
    for cat, info in cat_W_rank.items():
        log(f"      {cat:>12}: mean_delta_norm={info['mean_norm']:.1f}, CV={info['cv']:.1f}%")
    
    results["pair_analysis"] = pair_results
    results["category_norms"] = cat_W_rank
    
    return results


# ===== Main =====
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = _Path(f"d:/develop/TransformerLens-main/tests/glm5_temp/stage714_phase9_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(str(out_dir), "results")
    
    texts = build_texts()
    categories = [c for _, c in texts]
    cat_names = sorted(set(categories))
    
    log("=" * 70)
    log(f"Stage 714: Phase IX - DecodeFix + EfficientCL + Bias + CoherentGen + MoE")
    log(f"Timestamp: {ts}")
    log(f"Texts: {len(texts)} ({len(cat_names)} categories x ~{len(texts)//len(cat_names)} each)")
    log(f"Categories ({len(cat_names)}): {dict(zip(cat_names, [categories.count(c) for c in cat_names]))}")
    log(f"Models: {MODEL_ORDER}")
    log("=" * 70)
    
    all_results = {}
    
    for mname in MODEL_ORDER:
        log(f"\n{'#'*70}")
        log(f"# Processing: {mname}")
        log(f"{'#'*70}")
        
        t0 = time.time()
        model, tokenizer, n_layers, d_model = load_model(mname, log)
        
        # Compute h_final for all texts
        log(f"\n  Computing h_final for {len(texts)} texts...")
        h_all = []
        for i, (text, cat) in enumerate(texts):
            h, _ = get_h_final(model, tokenizer, text, n_layers)
            h_all.append(h)
            if (i + 1) % 100 == 0:
                log(f"    {i+1}/{len(texts)}...")
        h_all = torch.stack(h_all)  # [N, d]
        log(f"  Done in {time.time()-t0:.1f}s. h_all: {h_all.shape}")
        
        # P91: Decode fix
        p91_results = run_P91(model, tokenizer, n_layers, d_model, h_all, texts, categories, cat_names, log)
        
        # P92: Efficient closed-loop
        p92_results = run_P92(model, tokenizer, n_layers, d_model, h_all, texts, categories, cat_names, log)
        
        # P93: Bias analysis
        p93_results = run_P93(model, tokenizer, n_layers, d_model, h_all, texts, categories, log)
        
        # P94: Coherent generation
        p94_results = run_P94(model, tokenizer, n_layers, d_model, h_all, texts, categories, cat_names, log)
        
        # P95: MoE analysis
        p95_results = run_P95(model, tokenizer, n_layers, d_model, h_all, texts, categories, cat_names, log, mname)
        
        elapsed = time.time() - t0
        log(f"\n  {mname} total: {elapsed:.1f}s")
        
        all_results[mname] = {
            "P91": p91_results, "P92": p92_results,
            "P93": p93_results, "P94": p94_results, "P95": p95_results,
            "time": elapsed, "n_layers": n_layers, "d_model": d_model
        }
        
        # Save intermediate
        safe_results = {}
        for k, v in all_results.items():
            if isinstance(v, dict):
                safe_results[k] = str(v)[:500]
            else:
                safe_results[k] = str(v)
        with open(out_dir / "results.json", "w") as f:
            json.dump(safe_results, f, indent=2, default=str)
        
        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final summary
    log(f"\n{'='*70}")
    log(f"FINAL SUMMARY - Phase IX")
    log(f"{'='*70}")
    
    for mname in MODEL_ORDER:
        r = all_results.get(mname, {})
        log(f"\n  {mname}:")
        if "P91" in r:
            log(f"    P91 decode fix: {list(r['P91'].keys())[:5]}")
        if "P92" in r:
            for k, v in r["P92"].items():
                if "avg_cos" in k:
                    log(f"    P92 {k}: {v:.3f}")
        if "P93" in r:
            log(f"    P93 bias: avg_pct={r['P93'].get('avg_bias_pct', 'N/A')}%, avg_R2_imp={r['P93'].get('avg_r2_improvement', 'N/A')}")
        if "P94" in r:
            for k, v in r["P94"].items():
                if "avg_cos" in k:
                    log(f"    P94 {k}: {v:.3f}")
                if "text" in k:
                    log(f"    P94 {k}: {str(v)[:60]}")
    
    log(f"\nResults saved to: {out_dir}")
    log.close()


if __name__ == "__main__":
    main()
