#!/usr/bin/env python3
"""
Stage 713: Phase VIII — 闭环持久操控 + 崩塌机制 + 层变换参数化
=================================================================
P86: 闭环持久操控(closed-loop) — 每步重新插值,能否持续输出目标类别?
P87: 操控崩塌分析 — step1的h为何与step0的操控h完全不同?
P88: 层变换低秩参数化 — delta_h_l = W_l @ h_{l-1}, W_l的有效秩?
P89: 注意力模式分析 — 操控前后attention pattern的变化
P90: 因果强度逐层分析 — 每层对h_final的因果贡献(用dropout近似)

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
from sklearn.decomposition import PCA

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

# ===== Text dataset (reuse from stage712) =====
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
        "The only thing we have to fear is fear itself.",
        "I have a dream that one day this nation will rise up and live out its creed.",
        "The mass of men lead lives of quiet desperation.",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael. Some years ago, never mind how long precisely.",
        "All happy families are alike, but each unhappy family is unhappy in its own way.",
        "The sun also rises, and the sun goes down, and hastens to the place where it rises.",
        "It is a truth universally acknowledged that a single man in possession of a fortune must be in want of a wife.",
        "In my younger and more vulnerable years my father gave me some advice that I have been turning over in my mind ever since.",
        "It was a bright cold day in April, and the clocks were striking thirteen.",
        "Someone must have slandered Josef K., for one morning, without having done anything truly wrong, he was arrested.",
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendia was to remember that distant afternoon.",
        "Happy families are all alike; every unhappy family is unhappy in its own way.",
        "Through the fence, between the curling flower spaces, I could see them hitting.",
        "The story so far: in the beginning, the universe was created. This has made a lot of people very angry.",
        "He was an old man who fished alone in a skiff in the Gulf Stream.",
    ]
    for t in poetry: T.append((t, "poetry"))
    finance = [
        "The Federal Reserve raised interest rates by 25 basis points today.",
        "Gross domestic product grew by 3.2 percent in the last quarter.",
        "The stock market rally was driven by strong corporate earnings.",
        "Inflation remains above the central bank's target of 2 percent.",
        "Bonds prices fell as yields rose to their highest level in a decade.",
        "The trade deficit widened due to increased import demand.",
        "Consumer confidence index dropped to its lowest reading this year.",
        "The unemployment rate held steady at 3.7 percent.",
        "Venture capital funding in artificial intelligence reached record levels.",
        "The housing market showed signs of cooling after months of rapid appreciation.",
        "Cryptocurrency markets experienced extreme volatility this week.",
        "The European Central Bank maintained its accommodative monetary policy.",
        "Supply chain disruptions continued to affect manufacturing output.",
        "The dollar index strengthened against major currencies.",
        "Corporate profit margins came under pressure from rising labor costs.",
        "Small business optimism declined amid regulatory uncertainty.",
        "The yield curve inverted, historically a recession signal.",
        "Merger and acquisition activity slowed in the technology sector.",
        "Oil prices surged due to geopolitical tensions in the Middle East.",
        "The national debt exceeded 31 trillion dollars for the first time.",
        "Retail sales exceeded expectations, boosted by holiday spending.",
        "The current account deficit narrowed as exports increased.",
        "Gold prices rose as investors sought safe haven assets.",
        "The labor market added 200,000 jobs in the latest report.",
        "Manufacturing PMI contracted for the third consecutive month.",
        "Central bank digital currencies are being explored by many nations.",
        "Private equity firms raised record amounts of capital last year.",
        "The housing affordability crisis worsened as mortgage rates climbed.",
        "Technology stocks led the market recovery from the correction.",
        "Commodity prices stabilized after months of sharp increases.",
        "Bank lending standards tightened amid credit risk concerns.",
        "The gig economy now accounts for a significant portion of employment.",
        "Fiscal stimulus measures are being debated in Congress.",
        "The savings rate declined as consumers spent pandemic accumulated savings.",
        "Emerging market currencies faced selling pressure from strong dollar.",
        "The pharmaceutical sector outperformed on strong drug approval pipeline.",
        "Auto industry production was disrupted by semiconductor shortages.",
        "The commercial real estate market faces challenges from remote work trends.",
        "International trade agreements are being renegotiated between major economies.",
        "Carbon credit markets expanded as climate regulations tightened.",
        "The financial services sector is being transformed by fintech innovation.",
        "Sovereign wealth funds increased their allocation to alternative assets.",
        "Margin debt levels raised concerns about market leverage.",
    ]
    for t in finance: T.append((t, "finance"))
    medical = [
        "Clinical trials showed the vaccine was 95 percent effective at preventing infection.",
        "The patient presented with fever, cough, and difficulty breathing.",
        "MRI scans revealed no abnormalities in the brain tissue.",
        "The new drug targets a specific protein involved in tumor growth.",
        "Blood pressure should be maintained below 120 over 80 millimeters of mercury.",
        "Regular exercise reduces the risk of cardiovascular disease by up to 30 percent.",
        "The surgical procedure was completed successfully with no complications.",
        "Antibiotic resistance is one of the greatest threats to global public health.",
        "The study found a significant correlation between sleep duration and cognitive performance.",
        "Telemedicine has expanded access to healthcare in rural communities.",
        "The virus mutates rapidly, requiring constant updates to vaccine formulations.",
        "Chronic stress can weaken the immune system and increase disease susceptibility.",
        "Screening programs have significantly reduced mortality from certain cancers.",
        "The dosage must be carefully calculated based on body weight and kidney function.",
        "Gene therapy offers promising treatments for previously incurable genetic disorders.",
        "A balanced diet rich in fruits and vegetables is essential for optimal health.",
        "The placebo effect demonstrates the powerful connection between mind and body.",
        "Mental health awareness has improved significantly in recent years.",
        "The outbreak was traced to a contaminated food source at the processing facility.",
        "Physical therapy is recommended for post-operative rehabilitation and recovery.",
        "The incidence of type 2 diabetes has risen dramatically worldwide.",
        "Wearable health monitors can detect irregular heart rhythms early.",
        "The pathogen was identified using polymerase chain reaction testing.",
        "Stem cell research holds potential for regenerative medicine applications.",
        "Vaccination programs have eradicated several deadly diseases globally.",
        "The patient was prescribed a course of broad-spectrum antibiotics.",
        "Obesity is a major risk factor for multiple chronic health conditions.",
        "The clinical guidelines recommend screening for colon cancer starting at age 45.",
        "Palliative care focuses on improving quality of life for seriously ill patients.",
        "The mutation confers resistance to first-line antiviral therapy.",
        "Nutritional deficiencies can cause a wide range of health problems.",
        "The diagnostic accuracy of the AI system matched that of experienced radiologists.",
        "Public health measures including mask-wearing reduced transmission rates significantly.",
        "The tumor was classified as stage 3 with lymph node involvement.",
        "Rehabilitation exercises should be performed daily for optimal recovery.",
        "The epidemic spread rapidly through the densely populated urban area.",
        "Hormone replacement therapy carries both benefits and risks for postmenopausal women.",
        "The biomarker levels indicated early-stage disease progression.",
        "Preventive medicine emphasizes lifestyle modifications to reduce disease risk.",
        "The surgical team used minimally invasive laparoscopic techniques.",
        "Autoimmune disorders occur when the immune system attacks healthy tissue.",
        "The clinical trial enrolled 10,000 participants across 50 sites worldwide.",
        "Advances in neuroimaging have revolutionized our understanding of brain disorders.",
    ]
    for t in medical: T.append((t, "medical"))
    legal = [
        "The defendant pleaded not guilty to all charges at the arraignment.",
        "The Supreme Court ruled that the law violated the First Amendment.",
        "The contract was found to be null and void due to lack of consideration.",
        "The prosecution presented compelling evidence linking the defendant to the crime scene.",
        "The attorney filed a motion to suppress the evidence obtained without a warrant.",
        "The jury reached a unanimous verdict of guilty on all counts.",
        "The appeals court overturned the lower court's decision on procedural grounds.",
        "Intellectual property rights protect the creative works of authors and inventors.",
        "The settlement agreement included a confidentiality clause and monetary compensation.",
        "The statute of limitations has expired on this particular cause of action.",
        "Due process requires that the government respect all legal rights owed to a person.",
        "The burden of proof in a criminal case is beyond a reasonable doubt.",
        "The plaintiff filed a civil lawsuit seeking damages for personal injury.",
        "The constitution establishes the framework for the federal government and its powers.",
        "Precedent plays a crucial role in common law judicial decision making.",
        "The regulatory agency issued new guidelines for environmental compliance.",
        "The defendant's rights were read at the time of arrest as required by Miranda.",
        "The court granted a temporary restraining order pending a full hearing.",
        "Habeas corpus petitions challenge the legality of a person's detention.",
        "The arbitration clause in the contract requires disputes to be resolved privately.",
        "Corporate governance standards aim to protect shareholders and ensure transparency.",
        "The anti-trust laws prohibit monopolistic practices that harm consumer welfare.",
        "The legislative body passed the bill with a majority vote after heated debate.",
        "Criminal law distinguishes between misdemeanors and felonies based on severity.",
        "The attorney-client privilege protects confidential communications between lawyer and client.",
        "The regulatory framework for data privacy has become increasingly complex.",
        "The tribunal found the respondent liable for breach of international trade obligations.",
        "Equal protection under the law is guaranteed by the Fourteenth Amendment.",
        "The merger was blocked by regulators on competition grounds.",
        "The plaintiff's attorney delivered a powerful closing argument to the jury.",
        "The court of appeals remanded the case for a new trial.",
        "Statutory interpretation requires analyzing the plain meaning of legislative text.",
        "The defendant was sentenced to five years in federal prison.",
        "Tort law provides remedies for civil wrongs that cause harm or loss.",
        "The separation of powers doctrine divides government into three branches.",
        "The grand jury declined to issue an indictment based on insufficient evidence.",
        "Environmental regulations require companies to obtain permits before certain activities.",
        "The legal principle of res judicata prevents relitigation of settled matters.",
        "International humanitarian law governs the conduct of armed conflict.",
        "The bail amount was set at one million dollars pending trial.",
        "The judicial review power allows courts to invalidate unconstitutional laws.",
        "The legal age of majority is 18 years in most jurisdictions.",
    ]
    for t in legal: T.append((t, "legal"))
    return T

TEXTS = build_texts()


# ===== Model Loading =====
def load_model(model_name, log):
    model_path = MODEL_MAP[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"  Loading {model_name} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"  {model_name} loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    return model, tokenizer


def get_h_all(model, tokenizer, texts):
    all_h = []
    for text, _ in texts:
        tokens = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(tokens.input_ids, output_hidden_states=True)
        h = outputs.hidden_states[-1][0, -1, :].float().cpu()
        all_h.append(h)
    return torch.stack(all_h)


def get_unembed(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.data.float().cpu()
    elif hasattr(model, "get_output_embeddings"):
        return model.get_output_embeddings().weight.data.float().cpu()
    else:
        raise ValueError("Cannot find unembed")


def safe_decode(token_ids, tokenizer, n=5):
    words = []
    for tid in token_ids.tolist()[:n]:
        try:
            w = tokenizer.decode([tid]).strip().replace("\n", "\\n")[:15]
            if w:
                w_safe = w.encode("ascii", errors="replace").decode("ascii")
                if w_safe:
                    words.append(w_safe)
        except Exception:
            pass
    return " ".join(words) if words else "decode_err"


# ===== P86: Closed-Loop Persistent Manipulation =====
def p86_closed_loop(model, tokenizer, h_all, texts, categories, log):
    """
    P86: At EVERY step, re-interpolate h_final towards target centroid.
    
    Method:
    1. Run model normally to get h_final at step 0
    2. Interpolate: h_ctrl = (1-α)*h_final + α*centroid_target
    3. Decode top-1 token from h_ctrl
    4. Append token, run model again to get new h_final
    5. Re-interpolate at step 1, 2, ... N
    6. Compare: how long does the controlled output match target?
    
    Compare with:
    - Open-loop (no re-interpolation after step 0) - from Stage 712
    - Closed-loop (re-interpolate every step) - new
    """
    log(f"\n{'='*60}")
    log(f"  P86: Closed-Loop Persistent Manipulation")
    log(f"{'='*60}")

    W = get_unembed(model)
    cat_names = sorted(set(categories))
    cat_means = {}
    for cat in cat_names:
        idx = [i for i, c in enumerate(categories) if c == cat]
        cat_means[cat] = h_all[idx].mean(0)

    translations = [
        ("chinese", "gen_en", "Chinese->English"),
        ("code", "math_sci", "Code->Math"),
        ("gen_en", "poetry", "English->Poetry"),
    ]

    n_steps = 10
    alpha = 0.7
    
    results = {}
    for src, tgt, label in translations:
        log(f"\n    --- {label} ({src} -> {tgt}), alpha={alpha} ---")
        
        src_idx = [i for i, c in enumerate(categories) if c == src][:5]
        tgt_idx = [i for i, c in enumerate(categories) if c == tgt]
        
        tgt_logits_list = [(h_all[i] @ W.T).squeeze() for i in tgt_idx[:20]]
        tgt_top_counts = {}
        for tl in tgt_logits_list:
            for tid in torch.topk(tl, 10).indices.tolist():
                tgt_top_counts[tid] = tgt_top_counts.get(tid, 0) + 1
        tgt_common = set(sorted(tgt_top_counts, key=tgt_top_counts.get, reverse=True)[:10])
        tgt_avg_logits = torch.stack(tgt_logits_list).mean(0)
        tgt_avg_dist = F.softmax(tgt_avg_logits, -1)
        
        # Also compute source distribution for comparison
        src_logits_list = [(h_all[i] @ W.T).squeeze() for i in src_idx[:20]]
        src_avg_logits = torch.stack(src_logits_list).mean(0)
        src_avg_dist = F.softmax(src_avg_logits, -1)
        
        log(f"      Target tokens: {safe_decode(torch.tensor(list(tgt_common)[:5]), tokenizer)}")
        
        # ---- Closed-loop ----
        cl_metrics = []
        for ti in src_idx:
            text_src = texts[ti][0]
            tokens = tokenizer(text_src, return_tensors="pt").to(model.device)
            current_input = tokens.input_ids.clone()
            
            cos_tgts = []
            overlaps = []
            generated_tokens = []
            
            for step in range(n_steps + 1):
                # Run model to get h_final
                with torch.no_grad():
                    outputs = model(current_input, output_hidden_states=True)
                h_final = outputs.hidden_states[-1][0, -1, :].float().cpu()
                logits = outputs.logits[0, -1, :].float().cpu()
                
                # CLOSED-LOOP: re-interpolate at every step
                h_ctrl = (1 - alpha) * h_final + alpha * cat_means[tgt]
                ctrl_logits = h_ctrl @ W.T
                probs = F.softmax(ctrl_logits, -1)
                
                # Metrics
                cos_tgt = float(F.cosine_similarity(probs.unsqueeze(0), tgt_avg_dist.unsqueeze(0)))
                cos_src = float(F.cosine_similarity(probs.unsqueeze(0), src_avg_dist.unsqueeze(0)))
                top5 = set(torch.topk(probs, 5).indices.tolist())
                overlap = len(top5 & tgt_common)
                
                cos_tgts.append(cos_tgt)
                overlaps.append(overlap)
                
                # Decode and append
                top1 = ctrl_logits.argmax().unsqueeze(0).unsqueeze(0)
                generated_tokens.append(safe_decode(top1, tokenizer, 1))
                current_input = torch.cat([current_input, top1.to(model.device)], dim=1)
            
            cl_metrics.append({"cos_tgts": cos_tgts, "overlaps": overlaps, "tokens": generated_tokens})
        
        # Average
        avg_cos = np.mean([m["cos_tgts"] for m in cl_metrics], axis=0)
        avg_olap = np.mean([m["overlaps"] for m in cl_metrics], axis=0)
        
        log(f"      CLOSED-LOOP:")
        log(f"        Cos-tgt: step0={avg_cos[0]:.3f}, step5={avg_cos[5]:.3f}, step10={avg_cos[10]:.3f}")
        log(f"        Overlaps: {np.round(avg_olap, 1).tolist()}")
        log(f"        Sample tokens: {' -> '.join(cl_metrics[0]['tokens'][:6])}")
        
        # ---- Open-loop (for comparison) ----
        ol_metrics = []
        for ti in src_idx:
            text_src = texts[ti][0]
            tokens = tokenizer(text_src, return_tensors="pt").to(model.device)
            current_input = tokens.input_ids.clone()
            
            # Only interpolate at step 0
            with torch.no_grad():
                outputs = model(current_input, output_hidden_states=True)
            h_final = outputs.hidden_states[-1][0, -1, :].float().cpu()
            h_ctrl = (1 - alpha) * h_final + alpha * cat_means[tgt]
            ctrl_logits = h_ctrl @ W.T
            top1 = ctrl_logits.argmax().unsqueeze(0).unsqueeze(0)
            current_input = torch.cat([current_input, top1.to(model.device)], dim=1)
            
            ol_cos = []
            ol_olap = []
            for step in range(1, n_steps + 1):
                with torch.no_grad():
                    outputs = model(current_input, output_hidden_states=True)
                h_new = outputs.hidden_states[-1][0, -1, :].float().cpu()
                logits_new = outputs.logits[0, -1, :].float().cpu()
                probs = F.softmax(logits_new, -1)
                
                cos_tgt = float(F.cosine_similarity(probs.unsqueeze(0), tgt_avg_dist.unsqueeze(0)))
                top5 = set(torch.topk(probs, 5).indices.tolist())
                overlap = len(top5 & tgt_common)
                
                ol_cos.append(cos_tgt)
                ol_olap.append(overlap)
                
                top1 = logits_new.argmax().unsqueeze(0).unsqueeze(0)
                current_input = torch.cat([current_input, top1.to(model.device)], dim=1)
            
            ol_metrics.append({"cos": ol_cos, "olap": ol_olap})
        
        avg_ol_cos = np.mean([m["cos"] for m in ol_metrics], axis=0)
        
        log(f"      OPEN-LOOP (step 1-10 only):")
        log(f"        Cos-tgt: step1={avg_ol_cos[0]:.3f}, step5={avg_ol_cos[4]:.3f}, step10={avg_ol_cos[-1]:.3f}")
        
        results[f"{src}->{tgt}"] = {
            "cl_cos": avg_cos.tolist(),
            "cl_olap": avg_olap.tolist(),
            "ol_cos": avg_ol_cos.tolist(),
            "sample_tokens": cl_metrics[0]["tokens"],
        }
    
    return results


# ===== P87: Manipulation Collapse Analysis =====
def p87_collapse_analysis(model, tokenizer, h_all, texts, categories, log):
    """
    P87: WHY does the manipulation collapse at step 1?
    
    Decompose the collapse into components:
    1. h_ctrl = (1-α)*h_orig + α*h_target  (manipulated h at step 0)
    2. Decode token t from h_ctrl, append to input
    3. h_step1 = model(input + t)
    4. Compare: how different is h_step1 from h_ctrl?
    
    Key questions:
    - Is h_step1 closer to h_orig or h_ctrl?
    - Does the new token "reset" the hidden state?
    - What fraction of h_ctrl's direction is preserved in h_step1?
    - Layer-by-layer: at which layer does the collapse happen?
    """
    log(f"\n{'='*60}")
    log(f"  P87: Manipulation Collapse Analysis")
    log(f"{'='*60}")

    W = get_unembed(model)
    cat_means = {}
    for cat in sorted(set(categories)):
        idx = [i for i, c in enumerate(categories) if c == cat]
        cat_means[cat] = h_all[idx].mean(0)

    n_probe = 5
    alpha = 0.7
    translations = [("chinese", "gen_en"), ("code", "math_sci"), ("gen_en", "poetry")]
    
    results = {}
    for src, tgt in translations:
        log(f"\n    --- {src} -> {tgt} ---")
        src_idx = [i for i, c in enumerate(categories) if c == src][:n_probe]
        
        collapse_data = []
        for ti in src_idx:
            text = texts[ti][0]
            h_orig = h_all[ti].float()
            
            # Step 0: manipulate
            h_ctrl = (1 - alpha) * h_orig + alpha * cat_means[tgt]
            
            # Decode token from h_ctrl
            ctrl_logits = h_ctrl @ W.T
            ctrl_token_id = ctrl_logits.argmax().item()
            ctrl_token = safe_decode(torch.tensor([ctrl_token_id]), tokenizer, 1)
            
            # Also get natural next token
            orig_logits = h_orig @ W.T
            nat_token_id = orig_logits.argmax().item()
            nat_token = safe_decode(torch.tensor([nat_token_id]), tokenizer, 1)
            
            # Step 1: run model with ctrl token appended
            tokens = tokenizer(text, return_tensors="pt").to(model.device)
            input_with_ctrl = torch.cat([tokens.input_ids, 
                torch.tensor([[ctrl_token_id]]).to(model.device)], dim=1)
            
            with torch.no_grad():
                outputs_ctrl = model(input_with_ctrl, output_hidden_states=True)
            
            # Step 1 with natural token
            input_with_nat = torch.cat([tokens.input_ids, 
                torch.tensor([[nat_token_id]]).to(model.device)], dim=1)
            
            with torch.no_grad():
                outputs_nat = model(input_with_nat, output_hidden_states=True)
            
            h_step1_ctrl = outputs_ctrl.hidden_states[-1][0, -1, :].float().cpu()
            h_step1_nat = outputs_nat.hidden_states[-1][0, -1, :].float().cpu()
            
            # Layer-by-layer analysis
            n_layers = len(outputs_ctrl.hidden_states) - 1
            layer_cos_ctrl = []
            layer_cos_nat = []
            for l in range(n_layers + 1):
                hl_ctrl = outputs_ctrl.hidden_states[l][0, -1, :].float().cpu()
                hl_nat = outputs_nat.hidden_states[l][0, -1, :].float().cpu()
                
                # Cosine between layer h and h_ctrl (manipulated target)
                cos_ctrl = float(F.cosine_similarity(hl_ctrl.unsqueeze(0), h_ctrl.unsqueeze(0)))
                cos_nat = float(F.cosine_similarity(hl_nat.unsqueeze(0), h_ctrl.unsqueeze(0)))
                
                layer_cos_ctrl.append(cos_ctrl)
                layer_cos_nat.append(cos_nat)
            
            # Key metrics
            cos_ctrl_vs_orig = float(F.cosine_similarity(h_ctrl.unsqueeze(0), h_orig.unsqueeze(0)))
            cos_step1_vs_ctrl = float(F.cosine_similarity(h_step1_ctrl.unsqueeze(0), h_ctrl.unsqueeze(0)))
            cos_step1_vs_orig = float(F.cosine_similarity(h_step1_ctrl.unsqueeze(0), h_orig.unsqueeze(0)))
            cos_nat_vs_ctrl = float(F.cosine_similarity(h_step1_nat.unsqueeze(0), h_ctrl.unsqueeze(0)))
            
            collapse_data.append({
                "ctrl_token": ctrl_token,
                "nat_token": nat_token,
                "cos_ctrl_vs_orig": cos_ctrl_vs_orig,
                "cos_step1_vs_ctrl": cos_step1_vs_ctrl,
                "cos_step1_vs_orig": cos_step1_vs_orig,
                "cos_nat_vs_ctrl": cos_nat_vs_ctrl,
                "layer_cos_ctrl": layer_cos_ctrl,
                "layer_cos_nat": layer_cos_nat,
            })
        
        # Summarize
        avg_cvc = np.mean([d["cos_ctrl_vs_orig"] for d in collapse_data])
        avg_s1c = np.mean([d["cos_step1_vs_ctrl"] for d in collapse_data])
        avg_s1o = np.mean([d["cos_step1_vs_orig"] for d in collapse_data])
        avg_nc = np.mean([d["cos_nat_vs_ctrl"] for d in collapse_data])
        
        log(f"      cos(h_ctrl, h_orig): {avg_cvc:.4f}")
        log(f"      cos(h_step1_ctrl, h_ctrl): {avg_s1c:.4f} (preservation)")
        log(f"      cos(h_step1_ctrl, h_orig): {avg_s1o:.4f} (recovery)")
        log(f"      cos(h_step1_nat, h_ctrl): {avg_nc:.4f} (nat vs ctrl)")
        log(f"      Ctrl tokens: {[d['ctrl_token'] for d in collapse_data]}")
        log(f"      Nat tokens:  {[d['nat_token'] for d in collapse_data]}")
        
        # Layer-by-layer: find the "collapse layer"
        avg_layer_cos_ctrl = np.mean([d["layer_cos_ctrl"] for d in collapse_data], axis=0)
        avg_layer_cos_nat = np.mean([d["layer_cos_nat"] for d in collapse_data], axis=0)
        
        # Find layer where ctrl and nat diverge most
        diff = np.array(avg_layer_cos_ctrl) - np.array(avg_layer_cos_nat)
        max_diff_layer = np.argmax(np.abs(diff[1:])) + 1  # skip layer 0
        
        log(f"      Collapse layer (max ctrl-nat diff): L{max_diff_layer}")
        log(f"      Layer cos_ctrl: L0={avg_layer_cos_ctrl[0]:.3f}, L{max_diff_layer}={avg_layer_cos_ctrl[max_diff_layer]:.3f}, "
            f"L{len(avg_layer_cos_ctrl)-1}={avg_layer_cos_ctrl[-1]:.3f}")
        log(f"      Layer cos_nat:  L0={avg_layer_cos_nat[0]:.3f}, L{max_diff_layer}={avg_layer_cos_nat[max_diff_layer]:.3f}, "
            f"L{len(avg_layer_cos_nat)-1}={avg_layer_cos_nat[-1]:.3f}")
        
        # Recovery metric: does the model "snap back" to original?
        recovery = 1.0 if avg_s1o > avg_s1c else 0.0  # 1 = fully recovered, 0 = still controlled
        log(f"      Recovery: {'FULL (step1 closer to orig)' if avg_s1o > avg_s1c else 'PARTIAL (step1 between orig and ctrl)'}")
        
        results[f"{src}->{tgt}"] = {
            "avg_cos_ctrl_vs_orig": float(avg_cvc),
            "avg_cos_step1_vs_ctrl": float(avg_s1c),
            "avg_cos_step1_vs_orig": float(avg_s1o),
            "avg_cos_nat_vs_ctrl": float(avg_nc),
            "max_diff_layer": int(max_diff_layer),
            "recovery": float(avg_s1o > avg_s1c),
            "ctrl_tokens": [d["ctrl_token"] for d in collapse_data],
            "nat_tokens": [d["nat_token"] for d in collapse_data],
            "layer_cos_ctrl": avg_layer_cos_ctrl.tolist(),
            "layer_cos_nat": avg_layer_cos_nat.tolist(),
        }
    
    return results


# ===== P88: Layer Low-Rank Approximation =====
def p88_layer_lowrank(model, tokenizer, h_all, texts, categories, log):
    """
    P88: Parameterize layer transformation as low-rank linear map.
    
    delta_h_l = h_l - h_{l-1}
    
    Test: can delta_h_l be approximated as W_l @ h_{l-1}?
    - Fit W_l via least squares: W_l = argmin ||delta_h_l - W_l @ h_{l-1}||
    - Analyze rank of W_l via SVD
    - Test: does low-rank W_l preserve PC coordinates?
    
    Also test non-linear: delta_h_l = W_l @ sigma(h_{l-1})
    """
    log(f"\n{'='*60}")
    log(f"  P88: Layer Low-Rank Parameterization")
    log(f"{'='*60}")

    n_texts = 50  # Use 50 texts for fitting
    probe_texts = texts[:n_texts]
    
    # Extract per-layer h for all texts
    log(f"    Extracting per-layer h for {n_texts} texts...")
    all_states = []  # list of (n_layers+1,) lists of (d_model,) tensors
    for text, cat in probe_texts:
        tokens = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(tokens.input_ids, output_hidden_states=True)
        states = [hs[0, -1, :].float().cpu() for hs in outputs.hidden_states]
        all_states.append(states)
    
    n_layers = len(all_states[0]) - 1
    d_model = all_states[0][0].shape[0]
    log(f"    Layers: {n_layers}, d_model: {d_model}")
    
    # Compute delta_h and h_prev for each layer
    h_prev_all = torch.stack([all_states[t][l] for t in range(n_texts) for l in range(n_layers)])  # (n_texts*n_layers, d)
    delta_all = torch.stack([all_states[t][l+1] - all_states[t][l] for t in range(n_texts) for l in range(n_layers)])
    
    # Per-layer analysis
    log(f"\n    {'Layer':>6} {'Rank10':>8} {'Rank50':>8} {'Rank100':>9} {'FitR2':>8} {'SV0/SV1':>8}")
    log(f"    {'-'*55}")
    
    layer_results = {}
    for l in range(1, n_layers + 1):  # skip layer 0 (embedding)
        h_prev = torch.stack([all_states[t][l] for t in range(n_texts)])  # (n, d)
        delta = torch.stack([all_states[t][l+1] - all_states[t][l] for t in range(n_texts)])  # (n, d)
        
        if n_texts < d_model:
            # Solve W_l via least squares: delta = h_prev @ W^T
            # W = (h_prev^T @ h_prev)^{-1} @ h_prev^T @ delta
            try:
                W_fit = torch.linalg.lstsq(h_prev, delta).solution  # (d, d)
                delta_pred = h_prev @ W_fit
                
                # R2 score per dimension
                ss_res = ((delta - delta_pred) ** 2).sum(dim=0)
                ss_tot = ((delta - delta.mean(0)) ** 2).sum(dim=0)
                r2_per_dim = 1 - ss_res / (ss_tot + 1e-10)
                r2_mean = float(r2_per_dim.mean())
                
                # SVD of W_fit to find effective rank
                _, S_W, _ = torch.linalg.svd(W_fit, full_matrices=False)
                total_sv = S_W.sum()
                rank10 = (S_W.cumsum(0) / total_sv >= 0.10).nonzero()
                rank50 = (S_W.cumsum(0) / total_sv >= 0.50).nonzero()
                rank100 = (S_W.cumsum(0) / total_sv >= 0.90).nonzero()
                
                r10 = int(rank10[0].item() + 1) if len(rank10) > 0 else d_model
                r50 = int(rank50[0].item() + 1) if len(rank50) > 0 else d_model
                r100 = int(rank100[0].item() + 1) if len(rank100) > 0 else d_model
                
                sv_ratio = float(S_W[0] / (S_W[1] + 1e-10)) if len(S_W) > 1 else float('inf')
                
                layer_results[f"L{l}"] = {
                    "rank10": r10, "rank50": r50, "rank100": r100,
                    "r2": r2_mean, "sv_ratio": sv_ratio,
                    "sv0": float(S_W[0]),
                }
                
                if l <= 5 or l % 5 == 0 or l == n_layers:
                    log(f"    L{l:>4} {r10:>8} {r50:>8} {r100:>9} {r2_mean:>8.4f} {sv_ratio:>8.1f}")
            except Exception as e:
                log(f"    L{l:>4} FAILED: {e}")
        else:
            log(f"    L{l:>4} SKIPPED (n_texts >= d_model)")
    
    # Summary: average rank across layers
    if layer_results:
        avg_r50 = np.mean([v["rank50"] for v in layer_results.values()])
        avg_r100 = np.mean([v["rank100"] for v in layer_results.values()])
        avg_r2 = np.mean([v["r2"] for v in layer_results.values()])
        log(f"\n    Summary: avg_rank50={avg_r50:.0f}, avg_rank90={avg_r100:.0f}, avg_R2={avg_r2:.4f}")
        log(f"    Linear fit quality: {'EXCELLENT (R2>0.9)' if avg_r2 > 0.9 else 'GOOD (R2>0.5)' if avg_r2 > 0.5 else 'POOR (R2<0.5)'}")
    
    return layer_results


# ===== P89: Attention Pattern During Manipulation =====
def p89_attention_analysis(model, tokenizer, h_all, texts, categories, log):
    """
    P89: Does manipulation change attention patterns?
    
    Compare attention weights between:
    1. Original text (natural)
    2. Text + manipulated token (controlled)
    
    Key metrics:
    - Attention entropy: does manipulation increase/decrease attention concentration?
    - Attention to last token: does manipulation focus attention on the new token?
    - Head-level analysis: which attention heads change most?
    """
    log(f"\n{'='*60}")
    log(f"  P89: Attention Pattern Analysis During Manipulation")
    log(f"{'='*60}")

    W = get_unembed(model)
    cat_means = {}
    for cat in sorted(set(categories)):
        idx = [i for i, c in enumerate(categories) if c == cat]
        cat_means[cat] = h_all[idx].mean(0)
    
    n_probe = 5
    alpha = 0.7
    
    # Check if model outputs attention
    # Try to get attention weights
    test_text = texts[0][0]
    tokens = tokenizer(test_text, return_tensors="pt").to(model.device)
    
    try:
        with torch.no_grad():
            outputs = model(tokens.input_ids, output_attentions=True)
        
        if outputs.attentions is not None and len(outputs.attentions) > 0:
            n_layers = len(outputs.attentions)
            n_heads = outputs.attentions[0].shape[1]
            log(f"    Model has {n_layers} layers, {n_heads} attention heads")
            
            results = {}
            for src, tgt in [("chinese", "gen_en"), ("code", "math_sci")]:
                log(f"\n    --- {src} -> {tgt} ---")
                src_idx = [i for i, c in enumerate(categories) if c == src][:n_probe]
                
                head_entropy_diffs = []
                last_token_attn_diffs = []
                
                for ti in src_idx:
                    text = texts[ti][0]
                    h_orig = h_all[ti].float()
                    h_ctrl = (1 - alpha) * h_orig + alpha * cat_means[tgt]
                    
                    ctrl_logits = h_ctrl @ W.T
                    ctrl_token = ctrl_logits.argmax().item()
                    nat_token = (h_orig @ W.T).argmax().item()
                    
                    toks = tokenizer(text, return_tensors="pt").to(model.device)
                    
                    # Natural
                    with torch.no_grad():
                        out_nat = model(toks.input_ids, output_attentions=True)
                    # Controlled
                    toks_ctrl = torch.cat([toks.input_ids, torch.tensor([[ctrl_token]]).to(model.device)], dim=1)
                    with torch.no_grad():
                        out_ctrl = model(toks_ctrl, output_attentions=True)
                    
                    # Compare attention patterns at the last layer
                    # Focus on attention from the LAST position to all other positions
                    for layer_i in range(min(n_layers, 10)):  # Check first 10 layers
                        attn_nat = out_nat.attentions[layer_i][0, :, -1, :].float().cpu()  # (heads, seq)
                        attn_ctrl = out_ctrl.attentions[layer_i][0, :, -1, :].float().cpu()  # (heads, seq+1)
                        
                        # Attention to last token (the new token)
                        last_attn_ctrl = attn_ctrl[:, -1].mean().item()  # attention to the ctrl token itself
                        last_attn_nat = attn_nat[:, -1].mean().item()  # attention to original last token
                        
                        # Entropy
                        def attn_entropy(attn):
                            p = attn / (attn.sum(-1, keepdim=True) + 1e-10)
                            return -(p * torch.log(p + 1e-10)).sum(-1).mean().item()
                        
                        ent_nat = attn_entropy(attn_nat)
                        ent_ctrl = attn_entropy(attn_ctrl[:, :-1])  # exclude new token for fair comparison
                        
                        head_entropy_diffs.append(ent_ctrl - ent_nat)
                
                avg_ent_diff = np.mean(head_entropy_diffs)
                log(f"      Avg entropy change: {avg_ent_diff:+.4f} "
                    f"({'more uniform' if avg_ent_diff > 0 else 'more focused'})")
                
                results[f"{src}->{tgt}"] = {"avg_entropy_diff": float(avg_ent_diff)}
            
            return results
        else:
            log("    Model does not output attention weights, skipping P89")
            return {}
    except Exception as e:
        log(f"    Attention analysis failed: {e}")
        return {}


# ===== P90: Causal Layer Contribution =====
def p90_causal_layers(model, tokenizer, h_all, texts, categories, log):
    """
    P90: Causal contribution of each layer to PC coordinates.
    
    Method: ablation study
    - For each layer l, compute h with that layer zeroed out
    - Measure how much PC coordinates change
    - This gives the "causal importance" of each layer
    
    Approximation: instead of actual ablation (expensive), use the
    additive decomposition:
    - PC contribution of layer l = delta_h_l projected onto PC directions
    - Layer importance = |contribution_l| / |total contribution|
    """
    log(f"\n{'='*60}")
    log(f"  P90: Causal Layer Contribution to PC Coordinates")
    log(f"{'='*60}")

    n_texts = 30
    probe_texts = texts[:n_texts]
    
    # Compute PCA on full h_all
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S_all, Vt_all = torch.linalg.svd(h_c, full_matrices=False)
    K = 5
    
    # Extract per-layer states
    all_states = []
    for text, cat in probe_texts:
        tokens = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(tokens.input_ids, output_hidden_states=True)
        states = [hs[0, -1, :].float().cpu() for hs in outputs.hidden_states]
        all_states.append(states)
    
    n_layers = len(all_states[0]) - 1
    
    # Compute per-layer contribution to each PC
    log(f"\n    --- Per-Layer PC Contribution (absolute) ---")
    header = f"    {'Layer':>6}"
    for k in range(K):
        header += f" {'PC'+str(k)+'_abs':>10}"
    header += f" {'Total_abs':>10} {'Norm':>8}"
    log(header)
    log(f"    {'-'*65}")
    
    layer_importance = []
    for l in range(n_layers):
        deltas = torch.stack([all_states[t][l+1] - all_states[t][l] for t in range(n_texts)])
        contributions = []
        for k in range(K):
            pc_k = Vt_all[k]
            contrib = (deltas @ pc_k).abs().mean().item()
            contributions.append(contrib)
        total = sum(contributions)
        norm = float(deltas.norm(dim=1).mean())
        layer_importance.append({"contribs": contributions, "total": total, "norm": norm})
        
        if l < 5 or l % 5 == 0 or l == n_layers - 1:
            row = f"    L{l:>4}"
            for c in contributions:
                row += f" {c:>10.2f}"
            row += f" {total:>10.2f} {norm:>8.1f}"
            log(row)
    
    # Identify critical layers
    totals = [li["total"] for li in layer_importance]
    total_sum = sum(totals)
    
    log(f"\n    --- Critical Layers ---")
    # Top 5 contributing layers
    top5 = sorted(range(n_layers), key=lambda i: totals[i], reverse=True)[:5]
    log(f"    Top 5 contributing layers: {top5}")
    log(f"    Their combined contribution: {sum(totals[i] for i in top5)/total_sum*100:.1f}%")
    
    # First half vs second half
    mid = n_layers // 2
    first_half = sum(totals[:mid])
    second_half = sum(totals[mid:])
    log(f"    First half (L0-L{mid-1}): {first_half/total_sum*100:.1f}%")
    log(f"    Second half (L{mid}-L{n_layers-1}): {second_half/total_sum*100:.1f}%")
    
    # Early vs middle vs late
    third = n_layers // 3
    early = sum(totals[:third]) / total_sum * 100
    middle = sum(totals[third:2*third]) / total_sum * 100
    late = sum(totals[2*third:]) / total_sum * 100
    log(f"    Early (0-{third-1}): {early:.1f}%, Middle ({third}-{2*third-1}): {middle:.1f}%, Late ({2*third}-{n_layers-1}): {late:.1f}%")
    
    results = {
        "top5_layers": top5,
        "top5_pct": float(sum(totals[i] for i in top5) / total_sum * 100),
        "first_half_pct": float(first_half / total_sum * 100),
        "second_half_pct": float(second_half / total_sum * 100),
        "early_pct": float(early),
        "middle_pct": float(middle),
        "late_pct": float(late),
    }
    
    return results


# ===== Main =====
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage713_phase8_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    log = Logger(run_dir, "results")
    log("=" * 70)
    log("Stage 713: Phase VIII - ClosedLoop + Collapse + LowRank + Attention + Causal")
    log(f"Timestamp: {timestamp}")
    log(f"Texts: {len(TEXTS)} (10 categories x ~40 each)")
    cat_counts = {}
    for _, c in TEXTS:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    log(f"Categories ({len(cat_counts)}): {cat_counts}")
    log(f"Models: {MODEL_ORDER}")
    log("=" * 70)

    all_results = {}
    categories = [c for _, c in TEXTS]

    for mn in MODEL_ORDER:
        t0 = time.time()
        log(f"\n{'#'*70}")
        log(f"# Processing: {mn}")
        log(f"{'#'*70}")

        try:
            model, tokenizer = load_model(mn, log)

            log(f"\n  Computing h_final for {len(TEXTS)} texts...")
            t1 = time.time()
            h_all = get_h_all(model, tokenizer, TEXTS)
            log(f"  Done in {time.time()-t1:.1f}s. h_all: {h_all.shape}")

            model_results = {}

            # P86: Closed-loop (all models)
            try:
                t_p = time.time()
                r86 = p86_closed_loop(model, tokenizer, h_all, TEXTS, categories, log)
                model_results["p86"] = r86
                log(f"  P86 done in {time.time()-t_p:.1f}s")
            except Exception as e:
                log(f"  P86 FAILED: {e}")
                import traceback
                traceback.print_exc()

            # P87: Collapse analysis (all models)
            try:
                t_p = time.time()
                r87 = p87_collapse_analysis(model, tokenizer, h_all, TEXTS, categories, log)
                model_results["p87"] = r87
                log(f"  P87 done in {time.time()-t_p:.1f}s")
            except Exception as e:
                log(f"  P87 FAILED: {e}")
                import traceback
                traceback.print_exc()

            # P89: Attention analysis (first 2 models)
            if MODEL_ORDER.index(mn) < 2:
                try:
                    t_p = time.time()
                    r89 = p89_attention_analysis(model, tokenizer, h_all, TEXTS, categories, log)
                    model_results["p89"] = r89
                    log(f"  P89 done in {time.time()-t_p:.1f}s")
                except Exception as e:
                    log(f"  P89 FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            # P90: Causal layer contribution (all models)
            try:
                t_p = time.time()
                r90 = p90_causal_layers(model, tokenizer, h_all, TEXTS, categories, log)
                model_results["p90"] = r90
                log(f"  P90 done in {time.time()-t_p:.1f}s")
            except Exception as e:
                log(f"  P90 FAILED: {e}")
                import traceback
                traceback.print_exc()

            # P88: Low-rank (first model only - very expensive)
            if mn == "qwen3":
                try:
                    t_p = time.time()
                    r88 = p88_layer_lowrank(model, tokenizer, h_all, TEXTS, categories, log)
                    model_results["p88"] = r88
                    log(f"  P88 done in {time.time()-t_p:.1f}s")
                except Exception as e:
                    log(f"  P88 FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            all_results[mn] = model_results

            del model
            gc.collect()
            torch.cuda.empty_cache()
            log(f"\n  {mn} total: {time.time()-t0:.1f}s")

        except Exception as e:
            log(f"  FATAL {mn}: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    # Save JSON
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    save_data = make_serializable(all_results)
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # ===== Final Summary =====
    log(f"\n{'='*70}")
    log("FINAL SUMMARY - Phase VIII")
    log(f"{'='*70}")

    for mn in MODEL_ORDER:
        if mn not in all_results:
            continue
        res = all_results[mn]
        log(f"\n  {mn}:")
        if "p86" in res:
            for k, v in res["p86"].items():
                cl_s10 = v["cl_cos"][min(10, len(v["cl_cos"])-1)]
                ol_s10 = v["ol_cos"][min(9, len(v["ol_cos"])-1)]
                log(f"    P86 {k}: CL_step10={cl_s10:.3f}, OL_step10={ol_s10:.3f}")
        if "p87" in res:
            for k, v in res["p87"].items():
                log(f"    P87 {k}: step1_vs_ctrl={v['avg_cos_step1_vs_ctrl']:.3f}, "
                    f"collapse_layer=L{v['max_diff_layer']}, recovery={v['recovery']}")
        if "p90" in res:
            r = res["p90"]
            log(f"    P90: top5_pct={r['top5_pct']:.1f}%, late_pct={r['late_pct']:.1f}%")

    log(f"\nResults saved to: {run_dir}")
    log.close()
    print(f"\nDone! Results at: {run_dir}")


if __name__ == "__main__":
    main()
