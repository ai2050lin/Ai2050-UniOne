#!/usr/bin/env python3
"""
Stage 712: Phase VII — 多步生成 + 流形几何 + 代码刚性 + 因果方程
=====================================================================
P82: 多步生成因果 — 质心插值+自回归,能否让模型持续输出目标类别文本?
     核心问题: 单步操控已验证,多步累积是否可维持?因果效应衰减还是放大?
P83: 语义流形几何 — h_final的流形形状(球面/椭球/环面)和本征维度
     核心问题: 有效秩=2-4的流形是什么几何形状?
P84: 代码刚性突破 — code vs non-code的h-space结构差异
     核心问题: 代码为何抗拒操控?其编码结构有何特殊?
P85: 因果方程形式 — h_final的层贡献分解(加性vs非线性)
     核心问题: h_final = f(text) 的f是加性的还是非线性的?

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
from sklearn.neighbors import NearestNeighbors

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

# ===== 400 texts (10 categories x 40) =====
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
    all_logits = []
    for text, _ in texts:
        tokens = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(tokens.input_ids, output_hidden_states=True)
        h = outputs.hidden_states[-1][0, -1, :].float().cpu()
        logits = outputs.logits[0, -1, :].float().cpu()
        all_h.append(h)
        all_logits.append(logits)
    return torch.stack(all_h), all_logits


def get_h_per_layer(model, tokenizer, text):
    """Get h_final per layer for a single text."""
    tokens = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(tokens.input_ids, output_hidden_states=True)
    # outputs.hidden_states: (n_layers+1, batch, seq, d_model)
    states = [hs[0, -1, :].float().cpu() for hs in outputs.hidden_states]
    return states  # list of (d_model,) tensors


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


# ===== P82: Multi-step Generation with Centroid Interpolation =====
def p82_multistep_generation(model, tokenizer, h_all, texts, categories, log):
    """
    P82: Multi-step autoregressive generation with centroid interpolation.
    
    Key question: Can we make the model CONTINUOUSLY output target-category tokens?
    
    Method:
    1. Take source text, compute h_final
    2. Interpolate: h_new = (1-alpha)*h_src + alpha*h_target_centroid
    3. Decode top-1 token, append to input
    4. Re-run model to get new h_final
    5. Compare new h_final with original path vs target centroid
    6. Repeat for 10 steps
    
    Critical metrics:
    - target_overlap: does the generated token match target category distribution?
    - h_cos_decay: how fast does the manipulation effect decay?
    - divergence_rate: at which step does the model "escape" the target?
    """
    log(f"\n{'='*60}")
    log(f"  P82: Multi-step Generation with Centroid Interpolation")
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
    alphas = [0.3, 0.5, 0.7]
    
    results = {}
    for src, tgt, label in translations:
        log(f"\n    --- {label} ({src} -> {tgt}) ---")
        
        src_idx = [i for i, c in enumerate(categories) if c == src][:5]
        tgt_idx = [i for i, c in enumerate(categories) if c == tgt]
        
        # Target token distribution
        tgt_logits_list = [(h_all[i] @ W.T).squeeze() for i in tgt_idx[:20]]
        tgt_top_counts = {}
        for tl in tgt_logits_list:
            for tid in torch.topk(tl, 10).indices.tolist():
                tgt_top_counts[tid] = tgt_top_counts.get(tid, 0) + 1
        tgt_common_tokens = set(sorted(tgt_top_counts, key=tgt_top_counts.get, reverse=True)[:10])
        tgt_avg_logits = torch.stack(tgt_logits_list).mean(0)
        tgt_avg_dist = F.softmax(tgt_avg_logits, -1)
        
        log(f"      Target common tokens: {safe_decode(torch.tensor(list(tgt_common_tokens)[:5]), tokenizer)}")
        
        for alpha in alphas:
            log(f"\n      alpha={alpha}:")
            step_metrics = []
            
            for ti in src_idx:
                text_src = texts[ti][0]
                
                # Step 0: initial manipulation
                h_src = h_all[ti].float()
                h_interp = (1 - alpha) * h_src + alpha * cat_means[tgt]
                
                # Decode from interpolated h
                logits_interp = h_interp @ W.T
                probs = F.softmax(logits_interp, -1)
                
                # Check target overlap at step 0
                top5_ids = set(torch.topk(probs, 5).indices.tolist())
                overlap_0 = len(top5_ids & tgt_common_tokens)
                
                # Cosine with target avg dist
                cos_tgt_0 = float(F.cosine_similarity(probs.unsqueeze(0), tgt_avg_dist.unsqueeze(0)))
                
                # Now do multi-step autoregressive generation
                # We take the top-1 token from h_interp, append it, re-run model
                tokens = tokenizer(text_src, return_tensors="pt").to(model.device)
                current_input = tokens.input_ids.clone()
                
                overlaps = [overlap_0]
                cos_tgts = [cos_tgt_0]
                h_norms = [float(torch.norm(h_src))]
                
                for step in range(1, n_steps + 1):
                    # Get top-1 token from interpolated h
                    top1_token = logits_interp.argmax().unsqueeze(0).unsqueeze(0)
                    current_input = torch.cat([current_input, top1_token.to(model.device)], dim=1)
                    
                    # Re-run model
                    with torch.no_grad():
                        outputs = model(current_input, output_hidden_states=True)
                    h_new = outputs.hidden_states[-1][0, -1, :].float().cpu()
                    logits_new = outputs.logits[0, -1, :].float().cpu()
                    probs_new = F.softmax(logits_new, -1)
                    
                    # Without re-interpolation: just let model generate
                    top5_new = set(torch.topk(probs_new, 5).indices.tolist())
                    overlap = len(top5_new & tgt_common_tokens)
                    cos_tgt = float(F.cosine_similarity(probs_new.unsqueeze(0), tgt_avg_dist.unsqueeze(0)))
                    
                    overlaps.append(overlap)
                    cos_tgts.append(cos_tgt)
                    h_norms.append(float(torch.norm(h_new)))
                    
                    # Update for next step
                    logits_interp = logits_new
                
                step_metrics.append({
                    "overlaps": overlaps,
                    "cos_tgts": cos_tgts,
                    "h_norms": h_norms,
                })
            
            # Average across texts
            avg_overlaps = np.mean([m["overlaps"] for m in step_metrics], axis=0)
            avg_cos = np.mean([m["cos_tgts"] for m in step_metrics], axis=0)
            avg_hnorms = np.mean([m["h_norms"] for m in step_metrics], axis=0)
            
            log(f"        Step-overlaps (0-{n_steps}): {np.round(avg_overlaps, 1).tolist()}")
            log(f"        Cos-tgt: step0={avg_cos[0]:.3f}, step5={avg_cos[min(5,n_steps)]:.3f}, "
                f"step{n_steps}={avg_cos[-1]:.3f}")
            decay = avg_cos[-1] - avg_cos[0]
            log(f"        Decay (step{n_steps} - step0): {decay:+.3f}")
            
            results[f"{src}->{tgt}_a{alpha}"] = {
                "avg_overlaps": avg_overlaps.tolist(),
                "avg_cos_tgts": avg_cos.tolist(),
                "decay": float(decay),
            }
    
    return results


# ===== P83: Semantic Manifold Geometry =====
def p83_manifold_geometry(h_all, categories, log):
    """
    P83: Analyze the geometry of h_final semantic manifold.
    
    Tests:
    1. Sphere test: is ||h|| constant across texts? (ratio of max/min/mean norm)
    2. Ellipsoid test: PCA of h_all, are eigenvalues exponentially decaying?
    3. Intrinsic dimensionality: MLE estimator + nearest-neighbor method
    4. Local curvature: for each point, is the local neighborhood flat or curved?
    5. Category clustering: inter-category distance vs intra-category distance
    
    Uses ALL 417 texts for robust estimation.
    """
    log(f"\n{'='*60}")
    log(f"  P83: Semantic Manifold Geometry Analysis")
    log(f"{'='*60}")

    n = h_all.shape[0]
    d = h_all.shape[1]
    log(f"  Points: {n}, Ambient dim: {d}")
    
    # 1. Norm statistics
    norms = h_all.norm(dim=1)
    log(f"\n    --- Norm Statistics ---")
    log(f"    Mean norm: {norms.mean():.2f}")
    log(f"    Std norm:  {norms.std():.2f}")
    log(f"    Min/Max:   {norms.min():.2f} / {norms.max():.2f}")
    log(f"    CV (coeff var): {norms.std()/norms.mean()*100:.1f}%")
    
    is_sphere = (norms.std() / norms.mean()) < 0.05
    log(f"    Sphere test (CV<5%): {'YES - approximately spherical' if is_sphere else 'NO - not spherical'}")
    
    # 2. PCA eigenvalue spectrum
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)
    
    log(f"\n    --- PCA Spectrum ---")
    cumvar = torch.cumsum(S**2, 0) / (S**2).sum()
    dims = [1, 2, 3, 5, 10, 20, 50, 100]
    log(f"    {'Dim':>5} {'Var%':>8} {'SV':>10} {'SV_ratio':>10}")
    log(f"    {'-'*38}")
    for dim in dims:
        if dim <= len(S):
            log(f"    {dim:>5} {cumvar[dim-1].item()*100:>7.1f}% {S[dim-1].item():>10.1f} "
                f"{S[dim-1].item()/(S[0].item()+1e-10):>10.4f}")
    
    # 3. Intrinsic dimensionality via MLE (Levina & Bickel 2004)
    log(f"\n    --- Intrinsic Dimensionality (MLE) ---")
    k_list = [5, 10, 20]
    id_estimates = {}
    
    for k in k_list:
        if k >= n:
            continue
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(h_c.numpy())
        distances, _ = nbrs.kneighbors(h_c.numpy())
        # distances[:, 0] = 0 (self), use distances[:, 1:]
        T_k = distances[:, 1:]  # (n, k)
        
        # MLE estimator
        log_Tk = np.log(T_k + 1e-10)
        log_Tk_mean = log_Tk.mean(axis=1)  # (n,)
        mle_per_point = (k - 1) / (log_Tk_mean - log_Tk[:, -1] + 1e-10)
        
        # Filter extreme values
        valid = mle_per_point[np.isfinite(mle_per_point) & (mle_per_point > 0) & (mle_per_point < d)]
        id_est = np.mean(valid) if len(valid) > 0 else float('nan')
        id_estimates[k] = id_est
        log(f"    k={k:>3}: intrinsic_dim = {id_est:.1f} "
            f"(median={np.median(valid):.1f}, std={np.std(valid):.1f}, valid={len(valid)}/{n})")
    
    # 4. Category clustering analysis
    log(f"\n    --- Category Clustering ---")
    cat_names = sorted(set(categories))
    
    intra_dists = []
    inter_dists = []
    
    for cat in cat_names:
        idx = [i for i, c in enumerate(categories) if c == cat]
        h_cat = h_c[idx]
        
        # Intra-category distances
        for i in range(len(idx)):
            for j in range(i+1, min(i+10, len(idx))):  # Limit pairs for speed
                intra_dists.append(float(torch.norm(h_cat[i] - h_cat[j])))
    
    # Inter-category (sample)
    np.random.seed(42)
    for _ in range(min(500, n * (n-1) // 2)):
        i, j = np.random.choice(n, 2, replace=False)
        if categories[i] != categories[j]:
            inter_dists.append(float(torch.norm(h_c[i] - h_c[j])))
    
    intra_mean = np.mean(intra_dists)
    inter_mean = np.mean(inter_dists)
    log(f"    Intra-cat mean dist: {intra_mean:.2f}")
    log(f"    Inter-cat mean dist: {inter_mean:.2f}")
    log(f"    Ratio (inter/intra): {inter_mean/(intra_mean+1e-10):.2f}")
    log(f"    Clustering: {'STRONG' if inter_mean/intra_mean > 2 else 'MODERATE' if inter_mean/intra_mean > 1.3 else 'WEAK'}")
    
    # 5. Per-category norm and spread
    log(f"\n    --- Per-Category Statistics ---")
    log(f"    {'Category':>15} {'N':>4} {'MeanNorm':>10} {'NormCV%':>10} {'Spread':>10} {'SV0':>10}")
    log(f"    {'-'*65}")
    
    cat_stats = {}
    for cat in cat_names:
        idx = [i for i, c in enumerate(categories) if c == cat]
        h_cat = h_all[idx]
        cat_norms = h_cat.norm(dim=1)
        h_cat_c = h_cat - h_cat.mean(0)
        _, cat_S, _ = torch.linalg.svd(h_cat_c, full_matrices=False)
        
        cat_stats[cat] = {
            "n": len(idx),
            "mean_norm": float(cat_norms.mean()),
            "norm_cv": float(cat_norms.std() / cat_norms.mean() * 100),
            "spread": float(torch.norm(h_cat_c, dim=1).mean()),
            "sv0": float(cat_S[0]),
            "sv1": float(cat_S[1]) if len(cat_S) > 1 else 0,
        }
        
        log(f"    {cat:>15} {len(idx):>4} {cat_norms.mean():>10.1f} "
            f"{cat_norms.std()/cat_norms.mean()*100:>9.1f}% "
            f"{torch.norm(h_cat_c, dim=1).mean():>10.1f} {cat_S[0]:>10.1f}")
    
    # 6. Local curvature test
    log(f"\n    --- Local Curvature ---")
    # For each point, compute the "flatness" of its local neighborhood
    # A flat manifold has PCA spectrum that drops sharply to noise floor
    # A curved manifold has more gradual decay
    
    k_local = 20
    nbrs = NearestNeighbors(n_neighbors=k_local+1).fit(h_c.numpy())
    distances, indices = nbrs.kneighbors(h_c.numpy())
    
    local_dims = []
    sample_indices = np.random.choice(n, min(50, n), replace=False)
    
    for si in sample_indices:
        neighbors = h_c[indices[si, 1:]]  # (k_local, d)
        nbr_mean = neighbors.mean(0)
        nbr_c = neighbors - nbr_mean
        _, nbr_S, _ = torch.linalg.svd(nbr_c, full_matrices=False)
        
        # Count dimensions that explain 90% of variance
        total_var = (nbr_S**2).sum()
        cumvar = torch.cumsum(nbr_S**2, 0) / total_var
        n_dims_90 = (cumvar >= 0.90).nonzero()
        if len(n_dims_90) > 0:
            local_dims.append(n_dims_90[0].item() + 1)
        else:
            local_dims.append(k_local)
    
    log(f"    Local dims (90% var, k={k_local}): mean={np.mean(local_dims):.1f}, "
        f"median={np.median(local_dims):.0f}, std={np.std(local_dims):.1f}")
    
    results = {
        "norm_mean": float(norms.mean()),
        "norm_cv": float(norms.std() / norms.mean()),
        "is_sphere": is_sphere,
        "intrinsic_dim": {str(k): float(v) for k, v in id_estimates.items()},
        "intra_dist": float(intra_mean),
        "inter_dist": float(inter_mean),
        "inter_intra_ratio": float(inter_mean / (intra_mean + 1e-10)),
        "local_dim_90_mean": float(np.mean(local_dims)),
        "cat_stats": cat_stats,
        "pca_cumvar": {str(i+1): float(cumvar[i]) for i in range(min(20, len(cumvar)))},
    }
    
    return results


# ===== P84: Code Rigidity Analysis =====
def p84_code_rigidity(model, tokenizer, h_all, texts, categories, log):
    """
    P84: Why does code resist manipulation?
    
    Compare code vs non-code texts:
    1. Norm and spectral properties of h_final
    2. PCA spread and clustering
    3. How "isolated" is code in h-space?
    4. Layer-by-layer analysis: where does code diverge from non-code?
    5. Sensitivity analysis: how much does h_final change with small input perturbations?
    """
    log(f"\n{'='*60}")
    log(f"  P84: Code Rigidity Analysis")
    log(f"{'='*60}")

    code_idx = [i for i, c in enumerate(categories) if c == "code"]
    noncode_idx = [i for i, c in enumerate(categories) if c != "code"]
    
    h_code = h_all[code_idx]
    h_noncode = h_all[noncode_idx]
    
    # 1. Basic statistics
    log(f"\n    --- Basic Statistics ---")
    for name, h, idx_list in [("code", h_code, code_idx), ("non-code", h_noncode, noncode_idx)]:
        norms = h.norm(dim=1)
        h_c = h - h.mean(0)
        _, S, _ = torch.linalg.svd(h_c, full_matrices=False)
        
        log(f"    {name:>10}: N={len(idx_list)}, norm_mean={norms.mean():.1f}, norm_std={norms.std():.1f}, "
            f"SV0={S[0]:.1f}, SV1={S[1] if len(S)>1 else 0:.1f}, "
            f"SV0/SV1={S[0]/(S[1]+1e-10):.1f}")
    
    # 2. Distance from code centroid to other category centroids
    log(f"\n    --- Inter-Category Distances (from code centroid) ---")
    cat_names = sorted(set(categories))
    cat_means = {}
    for cat in cat_names:
        idx = [i for i, c in enumerate(categories) if c == cat]
        cat_means[cat] = h_all[idx].mean(0)
    
    code_centroid = cat_means["code"]
    dists_from_code = {}
    for cat in cat_names:
        dist = float(torch.norm(code_centroid - cat_means[cat]))
        dists_from_code[cat] = dist
        log(f"    code <-> {cat:>12}: {dist:.1f}")
    
    # Which category is closest to code?
    closest = min(cat_names, key=lambda c: dists_from_code[c] if c != "code" else float('inf'))
    farthest = max(cat_names, key=lambda c: dists_from_code[c])
    log(f"    Closest to code: {closest} ({dists_from_code[closest]:.1f})")
    log(f"    Farthest from code: {farthest} ({dists_from_code[farthest]:.1f})")
    
    # 3. Code intra-category spread vs others
    log(f"\n    --- Category Spread Comparison ---")
    for cat in cat_names:
        idx = [i for i, c in enumerate(categories) if c == cat]
        h_cat = h_all[idx]
        h_cat_c = h_cat - h_cat.mean(0)
        spread = float(torch.norm(h_cat_c, dim=1).mean())
        _, cat_S, _ = torch.linalg.svd(h_cat_c, full_matrices=False)
        
        # Effective dimensionality
        total = (cat_S**2).sum()
        cumvar = torch.cumsum(cat_S**2, 0) / total
        n90 = (cumvar >= 0.90).nonzero()
        dim90 = int(n90[0].item() + 1) if len(n90) > 0 else len(cat_S)
        
        log(f"    {cat:>12}: spread={spread:>7.1f}, SV0/SV1={cat_S[0]/(cat_S[1]+1e-10):>5.1f}, "
            f"dim90={dim90:>3}")
    
    # 4. Layer-by-layer: code vs non-code divergence
    log(f"\n    --- Layer-by-Layer Code vs Non-Code Divergence ---")
    n_probe = 5  # 5 code + 5 non-code texts
    code_texts = [texts[i] for i in code_idx[:n_probe]]
    noncode_texts = [texts[i] for i in noncode_idx[:n_probe]]
    
    layer_divergences = []
    try:
        for text, is_code in [(t, True) for t in code_texts] + [(t, False) for t in noncode_texts]:
            states = get_h_per_layer(model, tokenizer, text[0])
            if len(layer_divergences) == 0:
                layer_divergences = [{"code": [], "noncode": []} for _ in range(len(states))]
            
            label = "code" if is_code else "noncode"
            for l, s in enumerate(states):
                layer_divergences[l][label].append(s)
        
        log(f"    {'Layer':>6} {'Code_norm':>10} {'NonCode_norm':>12} {'Ratio':>8} {'Code_dim90':>12}")
        log(f"    {'-'*55}")
        
        layer_results = {}
        for l in range(1, len(layer_divergences)):  # Skip layer 0 (embedding)
            h_code_l = torch.stack(layer_divergences[l]["code"])
            h_nc_l = torch.stack(layer_divergences[l]["noncode"])
            
            code_norm = float(h_code_l.norm(dim=1).mean())
            nc_norm = float(h_nc_l.norm(dim=1).mean())
            ratio = code_norm / (nc_norm + 1e-10)
            
            # Effective dim of code h at this layer
            h_code_c = h_code_l - h_code_l.mean(0)
            _, code_S, _ = torch.linalg.svd(h_code_c, full_matrices=False)
            total = (code_S**2).sum()
            if total > 0:
                cumvar = torch.cumsum(code_S**2, 0) / total
                n90 = (cumvar >= 0.90).nonzero()
                dim90 = int(n90[0].item() + 1) if len(n90) > 0 else len(code_S)
            else:
                dim90 = 0
            
            layer_results[f"L{l}"] = {
                "code_norm": code_norm,
                "nc_norm": nc_norm,
                "ratio": ratio,
                "code_dim90": dim90,
            }
            
            if l <= 5 or l % 5 == 0 or l == len(layer_divergences) - 1:
                log(f"    L{l:>4} {code_norm:>10.1f} {nc_norm:>12.1f} {ratio:>8.2f} {dim90:>12}")
        
        results = {
            "dists_from_code": dists_from_code,
            "closest_cat": closest,
            "farthest_cat": farthest,
            "layer_results": layer_results,
        }
    except Exception as e:
        log(f"    Layer analysis failed: {e}")
        results = {"dists_from_code": dists_from_code, "closest_cat": closest}
    
    return results


# ===== P85: Causal Equation - Layer Contribution Decomposition =====
def p85_causal_equation(model, tokenizer, h_all, texts, categories, log):
    """
    P85: Decompose h_final into layer contributions to find the causal equation.
    
    Tests:
    1. Additive decomposition: h_final = h_0 + sum(delta_h_l)
       Does this hold exactly? (It should, since h_l = h_{l-1} + residual_delta)
    2. Layer contribution spectrum: which layers contribute most to PC coordinates?
    3. Cross-layer interaction: is delta_h_l independent of delta_h_{l-1}?
    4. Additivity test: h_final(A+B) ≈ h_final(A) + h_final(B) - h_final(empty)?
       (Does the model combine information additively?)
    5. Residual stream norm evolution: how does ||h_l|| change?
    """
    log(f"\n{'='*60}")
    log(f"  P85: Causal Equation - Layer Contribution Decomposition")
    log(f"{'='*60}")

    n_texts_probe = 20
    probe_texts = texts[:n_texts_probe]
    
    # 1. Additive decomposition verification
    log(f"\n    --- Additive Decomposition Verification ---")
    log(f"    Testing: h_final = h_0 + sum(delta_h_l)")
    
    additive_errors = []
    for text, cat in probe_texts:
        states = get_h_per_layer(model, tokenizer, text)
        
        h_0 = states[0]
        h_final = states[-1]
        
        # Reconstruct from deltas
        h_reconstructed = h_0.clone()
        for l in range(1, len(states)):
            h_reconstructed = h_reconstructed + (states[l] - states[l-1])
        
        error = float(torch.norm(h_final - h_reconstructed) / (torch.norm(h_final) + 1e-10))
        additive_errors.append(error)
    
    log(f"    Reconstruction error (should be ~0): mean={np.mean(additive_errors):.6f}, "
        f"max={np.max(additive_errors):.6f}")
    additive_exact = np.mean(additive_errors) < 1e-4
    log(f"    Additive: {'EXACT' if additive_exact else 'APPROXIMATE'}")
    
    # 2. Layer contribution to PC coordinates
    log(f"\n    --- Layer Contribution to PC Coordinates ---")
    # PCA on h_final
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S_all, Vt_all = torch.linalg.svd(h_c, full_matrices=False)
    K = 5
    
    layer_contributions = []
    for text, cat in probe_texts:
        states = get_h_per_layer(model, tokenizer, text)
        n_layers = len(states) - 1
        
        deltas = [states[l+1] - states[l] for l in range(n_layers)]
        contributions = []
        for k in range(K):
            pc_k = Vt_all[k]
            pc_contribs = [float(d @ pc_k) for d in deltas]
            contributions.append(pc_contribs)
        layer_contributions.append(contributions)
    
    # Average contribution per layer
    avg_contribs = np.mean(layer_contributions, axis=0)  # (K, n_layers)
    
    # Find which layers contribute most to each PC
    log(f"    {'PC':>4} {'Top3_layers':>15} {'Bot3_layers':>15} {'Max_contrib':>12} {'Total':>10}")
    log(f"    {'-'*60}")
    
    pc_layer_info = {}
    for k in range(K):
        contribs = avg_contribs[k]
        top3 = np.argsort(np.abs(contribs))[-3:][::-1]
        bot3 = np.argsort(np.abs(contribs))[:3]
        
        pc_layer_info[f"PC{k}"] = {
            "top3_layers": [int(x) for x in top3],
            "bot3_layers": [int(x) for x in bot3],
            "max_contrib": float(np.max(np.abs(contribs))),
            "total_abs": float(np.sum(np.abs(contribs))),
        }
        
        log(f"    PC{k:>2} {str([int(x) for x in top3]):>15} {str([int(x) for x in bot3]):>15} "
            f"{np.max(np.abs(contribs)):>12.2f} {np.sum(np.abs(contribs)):>10.1f}")
    
    # 3. Cross-layer interaction
    log(f"\n    --- Cross-Layer Interaction ---")
    # Correlation between |delta_h_l| and |delta_h_{l+1}|
    
    layer_deltas_norms = []
    for text, cat in probe_texts[:10]:
        states = get_h_per_layer(model, tokenizer, text)
        deltas = [float(torch.norm(states[l+1] - states[l])) for l in range(len(states)-1)]
        layer_deltas_norms.append(deltas)
    
    # Correlation of adjacent layer delta norms
    layer_deltas_norms = np.array(layer_deltas_norms)  # (n_texts, n_layers)
    n_layers = layer_deltas_norms.shape[1]
    
    adj_corrs = []
    for l in range(n_layers - 1):
        corr = np.corrcoef(layer_deltas_norms[:, l], layer_deltas_norms[:, l+1])[0, 1]
        adj_corrs.append(corr)
    
    log(f"    Adjacent layer delta-norm correlation:")
    log(f"    Mean: {np.mean(adj_corrs):.3f}, Min: {np.min(adj_corrs):.3f}, Max: {np.max(adj_corrs):.3f}")
    log(f"    Interpretation: {'INDEPENDENT (layers act autonomously)' if abs(np.mean(adj_corrs)) < 0.3 else 'CORRELATED (layers interact)'}")
    
    # 4. Residual stream norm evolution
    log(f"\n    --- Residual Stream Norm Evolution ---")
    avg_norms = []
    for text, cat in probe_texts[:10]:
        states = get_h_per_layer(model, tokenizer, text)
        norms = [float(torch.norm(s)) for s in states]
        avg_norms.append(norms)
    
    avg_norms = np.mean(avg_norms, axis=0)
    log(f"    Layer 0 norm: {avg_norms[0]:.1f}")
    log(f"    Layer {len(avg_norms)//4} norm: {avg_norms[len(avg_norms)//4]:.1f}")
    log(f"    Layer {len(avg_norms)//2} norm: {avg_norms[len(avg_norms)//2]:.1f}")
    log(f"    Layer {3*len(avg_norms)//4} norm: {avg_norms[3*len(avg_norms)//4]:.1f}")
    log(f"    Layer {len(avg_norms)-1} norm: {avg_norms[-1]:.1f}")
    
    # Growth pattern
    growth_ratio = avg_norms[-1] / (avg_norms[0] + 1e-10)
    log(f"    Growth ratio (last/first): {growth_ratio:.1f}x")
    
    # Is growth monotonic?
    monotonic = all(avg_norms[i] <= avg_norms[i+1] + 0.1 for i in range(len(avg_norms)-1))
    log(f"    Monotonically increasing: {'YES' if monotonic else 'NO (some layers decrease norm)'}")
    
    # 5. Text composition test (additivity of h_final)
    log(f"\n    --- Text Composition Additivity ---")
    log(f"    Testing: h(A+B) vs h(A)+h(B)-h(baseline)")
    
    # Use short texts for concatenation
    short_texts = [(t, c) for t, c in texts if len(t.split()) <= 5][:10]
    if len(short_texts) >= 4:
        pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]
        add_errors = []
        
        for i, j in pairs:
            t1, c1 = short_texts[i]
            t2, c2 = short_texts[j]
            t_cat = t1 + " " + t2
            
            h1 = get_h_per_layer(model, tokenizer, t1)[-1]
            h2 = get_h_per_layer(model, tokenizer, t2)[-1]
            h_cat = get_h_per_layer(model, tokenizer, t_cat)[-1]
            
            h_mean_ab = (h1 + h2) / 2
            error = float(torch.norm(h_cat - h_mean_ab) / (torch.norm(h_cat) + 1e-10))
            add_errors.append(error)
            
            cos_sim = float(F.cosine_similarity(h_cat.unsqueeze(0), h_mean_ab.unsqueeze(0)))
            log(f"    [{t1[:20]:>20}] + [{t2[:20]:>20}] -> cos={cos_sim:.3f}, rel_err={error:.3f}")
        
        log(f"    Mean relative error: {np.mean(add_errors):.3f}")
        log(f"    Interpretation: {'ADDITIVE (h_final linear in text)' if np.mean(add_errors) < 0.5 else 'NON-LINEAR (h_final depends on text interactions)'}")
    
    results = {
        "additive_error": float(np.mean(additive_errors)),
        "additive_exact": additive_exact,
        "pc_layer_info": pc_layer_info,
        "adj_layer_corr_mean": float(np.mean(adj_corrs)),
        "norm_growth_ratio": float(growth_ratio),
        "norm_monotonic": monotonic,
    }
    
    return results


# ===== Main =====
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage712_phase7_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    log = Logger(run_dir, "results")
    log("=" * 70)
    log("Stage 712: Phase VII - MultiStep + Manifold + CodeRigidity + CausalEq")
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
            h_all, all_logits = get_h_all(model, tokenizer, TEXTS)
            log(f"  Done in {time.time()-t1:.1f}s. h_all: {h_all.shape}")

            model_results = {}

            # P82: Multi-step generation (all models)
            try:
                t_p = time.time()
                r82 = p82_multistep_generation(model, tokenizer, h_all, TEXTS, categories, log)
                model_results["p82"] = r82
                log(f"  P82 done in {time.time()-t_p:.1f}s")
            except Exception as e:
                log(f"  P82 FAILED: {e}")
                import traceback
                traceback.print_exc()

            # P83: Manifold geometry (all models)
            try:
                t_p = time.time()
                r83 = p83_manifold_geometry(h_all, categories, log)
                model_results["p83"] = r83
                log(f"  P83 done in {time.time()-t_p:.1f}s")
            except Exception as e:
                log(f"  P83 FAILED: {e}")
                import traceback
                traceback.print_exc()

            # P84: Code rigidity (all models)
            try:
                t_p = time.time()
                r84 = p84_code_rigidity(model, tokenizer, h_all, TEXTS, categories, log)
                model_results["p84"] = r84
                log(f"  P84 done in {time.time()-t_p:.1f}s")
            except Exception as e:
                log(f"  P84 FAILED: {e}")
                import traceback
                traceback.print_exc()

            # P85: Causal equation (first 2 models - slow due to layer extraction)
            if MODEL_ORDER.index(mn) < 2:
                try:
                    t_p = time.time()
                    r85 = p85_causal_equation(model, tokenizer, h_all, TEXTS, categories, log)
                    model_results["p85"] = r85
                    log(f"  P85 done in {time.time()-t_p:.1f}s")
                except Exception as e:
                    log(f"  P85 FAILED: {e}")
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
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    save_data = make_serializable(all_results)
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # ===== Final Summary =====
    log(f"\n{'='*70}")
    log("FINAL SUMMARY - Phase VII")
    log(f"{'='*70}")

    for mn in MODEL_ORDER:
        if mn not in all_results:
            continue
        res = all_results[mn]
        log(f"\n  {mn}:")
        if "p83" in res:
            r = res["p83"]
            log(f"    P83 Manifold: norm_CV={r['norm_cv']*100:.1f}%, "
                f"sphere={r['is_sphere']}, "
                f"intrinsic_dim={r.get('intrinsic_dim', 'N/A')}, "
                f"inter/intra={r['inter_intra_ratio']:.2f}")
        if "p84" in res:
            r = res["p84"]
            log(f"    P84 Code: closest={r.get('closest_cat','N/A')}, farthest={r.get('farthest_cat','N/A')}")
        if "p85" in res:
            r = res["p85"]
            log(f"    P85 Causal: additive_error={r['additive_error']:.6f}, "
                f"exact={r['additive_exact']}, "
                f"adj_corr={r['adj_layer_corr_mean']:.3f}, "
                f"norm_growth={r['norm_growth_ratio']:.1f}x, "
                f"monotonic={r['norm_monotonic']}")

    log(f"\nResults saved to: {run_dir}")
    log.close()
    print(f"\nDone! Results at: {run_dir}")


if __name__ == "__main__":
    main()
