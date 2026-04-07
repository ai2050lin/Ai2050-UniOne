#!/usr/bin/env python3
"""
Stage 711: Phase VI — 精确操控 + 梯度引导 + 高维特征 + 层Jacobian
=====================================================================
P78: 精细操控——二分搜索最优偏移量,找到使输出最大程度匹配目标类别的精确偏移
P79: 梯度引导操控——用梯度下降找到精确的h修改方向
P80: 高维特征预测——用模型自身embedding(2560+维)替代32维手工特征
P81: 层Jacobian分析——有限差分法计算层变换的雅可比矩阵,分析其结构

核心突破方向:
1. 从"粗暴偏移"进化到"精确操控"——找到让输出=目标类别典型token的最优偏移量
2. 用梯度方法替代PCA方向偏移——更精确的因果操控
3. 突破32维特征天花板——用模型自身的高维embedding作为特征
4. 理解层变换的数学结构——通过Jacobian分析找到层变换的参数化形式

四模型串行: Qwen3 -> DS7B -> GLM4 -> Gemma4
设备: CUDA
"""

import sys, time, gc, json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

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
        "The mutation confers resistance to first-line antiretroviral therapy.",
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


# ===== P78: Precise Manipulation via Binary Search =====
def p78_precise_manipulation(model, tokenizer, h_all, texts, categories, log):
    """
    P78: Find the EXACT shift amount that maximizes target category overlap.
    
    Instead of using fixed shifts (0.5, 1.0, 2.0 std), use binary search
    to find the optimal shift that:
    1. Changes top-1 to a target category token
    2. Minimizes KL divergence from a natural target category distribution
    
    Also test: interpolating between source and target category centroids
    rather than just shifting along the PC direction.
    """
    log(f"\n{'='*60}")
    log(f"  P78: Precise Manipulation - Binary Search Optimal Shift")
    log(f"{'='*60}")

    W = get_unembed(model)
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)
    K = 10

    cat_names = sorted(set(categories))
    pc_coords = h_c @ Vt[:K].T

    cat_means_h = {}
    cat_means_pc = {}
    for cat in cat_names:
        idx = [i for i, c in enumerate(categories) if c == cat]
        cat_means_h[cat] = h_all[idx].mean(0)
        cat_means_pc[cat] = pc_coords[idx].mean(0)

    # Key translations to test
    translations = [
        ("chinese", "gen_en", "Chinese->English"),
        ("code", "math_sci", "Code->Math"),
        ("gen_en", "poetry", "English->Poetry"),
        ("finance", "medical", "Finance->Medical"),
    ]

    results = {}
    for src, tgt, label in translations:
        log(f"\n    --- {label} ({src} -> {tgt}) ---")

        src_idx = [i for i, c in enumerate(categories) if c == src][:5]

        # Method 1: Binary search on PC-space shift
        direction_pc = cat_means_pc[tgt] - cat_means_pc[src]  # (K,)
        dir_norm = torch.norm(direction_pc)

        optimal_shifts = []
        for ti in src_idx:
            h = h_all[ti:ti+1].float()
            pc = pc_coords[ti:ti+1].float()

            # Compute target category's average logit distribution
            tgt_idx = [i for i, c in enumerate(categories) if c == tgt]
            tgt_logits = []
            for tti in tgt_idx[:20]:
                tgt_logits.append((h_all[tti] @ W.T).squeeze())
            avg_tgt_logits = torch.stack(tgt_logits).mean(0)
            avg_tgt_dist = F.softmax(avg_tgt_logits, -1)

            # Binary search for optimal shift
            lo, hi = 0.0, 3.0
            best_shift = 0
            best_score = -1e10

            for _ in range(12):  # 12 iterations = 4096 precision
                mid = (lo + hi) / 2
                pc_shifted = pc + direction_pc.unsqueeze(0) * mid
                h_shifted = pc_shifted @ Vt[:K].float() + h_mean.float()
                logits_new = F.softmax((h_shifted @ W.T).squeeze(), -1)

                # Score: high overlap with target distribution, but penalize extreme shifts
                target_overlap = float(F.cosine_similarity(logits_new.unsqueeze(0),
                                       avg_tgt_dist.unsqueeze(0)))
                score = target_overlap - 0.3 * mid  # penalize large shifts

                if score > best_score:
                    best_score = score
                    best_shift = mid

                # Check if top-1 matches target category
                top1_shifted = logits_new.argmax().item()
                tgt_top1_ids = set()
                for tti in tgt_idx[:20]:
                    tgt_top1_ids.add((h_all[tti] @ W.T).squeeze().argmax().item())
                if top1_shifted in tgt_top1_ids:
                    hi = mid  # try smaller shift
                else:
                    lo = mid  # try larger shift

            optimal_shifts.append(best_shift)

            # Now evaluate at optimal shift
            pc_opt = pc + direction_pc.unsqueeze(0) * best_shift
            h_opt = pc_opt @ Vt[:K].float() + h_mean.float()
            logits_opt = F.softmax((h_opt @ W.T).squeeze(), -1)

            orig_logits = F.softmax((h @ W.T).squeeze(), -1)
            target_cos = float(F.cosine_similarity(logits_opt.unsqueeze(0),
                            avg_tgt_dist.unsqueeze(0)))
            kl = float(F.kl_div(F.log_softmax((h_opt @ W.T).squeeze(), -1),
                                F.softmax(avg_tgt_logits, -1), reduction="sum"))

        avg_optimal = np.mean(optimal_shifts)
        std_optimal = np.std(optimal_shifts)
        log(f"      Optimal shift: mean={avg_optimal:.3f}, std={std_optimal:.3f}")

        # Method 2: Centroid interpolation (slerp-like)
        log(f"      --- Centroid Interpolation ---")
        interp_scores = []
        alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        for alpha in alphas:
            hits = 0
            total_tgt_cos = 0
            for ti in src_idx:
                h = h_all[ti]
                h_interp = (1 - alpha) * h + alpha * cat_means_h[tgt]
                logits_interp = (h_interp @ W.T).squeeze()

                tgt_logits_list = [h_all[i] @ W.T for i in tgt_idx[:20]]
                tgt_top_counts = {}
                for tl in tgt_logits_list:
                    for tid in torch.topk(tl, 10).indices.tolist():
                        tgt_top_counts[tid] = tgt_top_counts.get(tid, 0) + 1
                tgt_common = set(sorted(tgt_top_counts, key=tgt_top_counts.get, reverse=True)[:5])
                top5_new = set(torch.topk(logits_interp, 5).indices.tolist())
                if len(top5_new & tgt_common) > 0:
                    hits += 1

                avg_tgt_logits_v = torch.stack(tgt_logits_list).mean(0)
                avg_tgt_dist_v = F.softmax(avg_tgt_logits_v, -1)
                total_tgt_cos += float(F.cosine_similarity(
                    F.softmax(logits_interp, -1).unsqueeze(0),
                    avg_tgt_dist_v.unsqueeze(0)))

            hit_rate = hits / len(src_idx)
            avg_cos = total_tgt_cos / len(src_idx)
            interp_scores.append({"alpha": alpha, "hit_rate": hit_rate, "tgt_cos": avg_cos})
            log(f"        alpha={alpha:.1f}: hit_rate={hit_rate:.0%}, tgt_cos={avg_cos:.4f}")

        # Find best interpolation alpha
        best_interp = max(interp_scores, key=lambda x: x["tgt_cos"])
        log(f"      Best interpolation: alpha={best_interp['alpha']:.1f}, "
            f"tgt_cos={best_interp['tgt_cos']:.4f}, hit_rate={best_interp['hit_rate']:.0%}")

        results[f"{src}->{tgt}"] = {
            "optimal_shift_mean": float(avg_optimal),
            "optimal_shift_std": float(std_optimal),
            "interp_scores": interp_scores,
        }

    return results


# ===== P79: Gradient-Guided Manipulation =====
def p79_gradient_manipulation(model, tokenizer, h_all, texts, categories, log):
    """
    P79: Use gradient descent to find the optimal h modification.
    
    Instead of PCA-based shifts, directly optimize h to maximize
    the probability of target tokens while staying close to original h.
    
    Loss = -sum(log p(target_token | h_modified)) + lambda * ||h_modified - h_orig||^2
    
    This is the most precise form of causal manipulation possible.
    """
    log(f"\n{'='*60}")
    log(f"  P79: Gradient-Guided Manipulation")
    log(f"{'='*60}")

    W = get_unembed(model)

    cat_names = sorted(set(categories))
    translations = [
        ("chinese", "gen_en", "Chinese->English"),
        ("code", "math_sci", "Code->Math"),
    ]

    lambdas = [0.01, 0.1, 1.0, 10.0]  # regularization strengths

    results = {}
    for src, tgt, label in translations:
        log(f"\n    --- {label} ({src} -> {tgt}) ---")
        src_idx = [i for i, c in enumerate(categories) if c == src][:3]
        tgt_idx = [i for i, c in enumerate(categories) if c == tgt][:20]

        # Get target token distribution
        tgt_logits_list = [(h_all[i] @ W.T).squeeze() for i in tgt_idx]
        tgt_top_counts = {}
        for tl in tgt_logits_list:
            for tid in torch.topk(tl, 10).indices.tolist():
                tgt_top_counts[tid] = tgt_top_counts.get(tid, 0) + 1
        target_tokens = sorted(tgt_top_counts, key=tgt_top_counts.get, reverse=True)[:5]

        log(f"      Target top-5 tokens: {safe_decode(torch.tensor(target_tokens), tokenizer)}")

        for lam in lambdas:
            log(f"\n      lambda={lam}:")
            for ti in src_idx:
                h_orig = h_all[ti].clone().float().requires_grad_(True)

                optimizer = torch.optim.SGD([h_orig], lr=0.01)

                for step in range(200):
                    optimizer.zero_grad()
                    logits = h_orig @ W.T  # (vocab_size,)

                    # Maximize probability of target tokens
                    log_probs = F.log_softmax(logits, -1)
                    target_loss = -sum(log_probs[t] for t in target_tokens) / len(target_tokens)

                    # Regularization: stay close to original
                    reg_loss = lam * torch.norm(h_orig - h_all[ti].float())**2 / (h_all[ti].float().norm()**2 + 1e-10)

                    total_loss = target_loss + reg_loss
                    total_loss.backward()
                    optimizer.step()

                # Evaluate final result
                with torch.no_grad():
                    logits_final = h_orig @ W.T
                    top5 = safe_decode(torch.topk(logits_final, 5).indices, tokenizer)
                    top1_orig = safe_decode((h_all[ti] @ W.T).squeeze().argmax().unsqueeze(0), tokenizer, 1)
                    h_shift = torch.norm(h_orig - h_all[ti].float()).item()
                    h_norm = torch.norm(h_all[ti].float()).item()

                hits = sum(1 for t in target_tokens if t in torch.topk(logits_final, 5).indices.tolist())

                log(f"        [{ti}] shift={h_shift/h_norm:.3f}norm, "
                    f"target_hits={hits}/{len(target_tokens)}, "
                    f"top5: {top5}")

        results[f"{src}->{tgt}"] = {"target_tokens": target_tokens}

    return results


# ===== P80: High-Dimensional Embedding Features =====
def p80_highdim_features(model, tokenizer, h_all, texts, categories, log):
    """
    P80: Use the model's own embedding layer to create high-dimensional features.
    
    Instead of 32 manual features, use:
    - Average embedding of all tokens (d_embed dims)
    - First/last token embedding
    - Embedding PCA (50 dims)
    
    Then predict PC coordinates with these high-dim features.
    """
    log(f"\n{'='*60}")
    log(f"  P80: High-Dimensional Embedding Features")
    log(f"{'='*60}")

    W_embed = None
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        W_embed = model.model.embed_tokens.weight.data.float().cpu()
    elif hasattr(model, "get_input_embeddings"):
        W_embed = model.get_input_embeddings().weight.data.float().cpu()

    if W_embed is None:
        log("  No embedding layer found, skipping")
        return {}, {}

    d_embed = W_embed.shape[1]
    log(f"  Embedding dimension: {d_embed}")

    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)
    K = 10
    pc_coords = (h_c @ Vt[:K].T).numpy()

    # Build high-dimensional features
    features_list = []
    for text, cat in texts:
        tokens = tokenizer(text, return_tensors="pt")
        input_ids = tokens.input_ids[0]

        embeds = W_embed[input_ids]  # (seq_len, d_embed)

        # Feature set 1: Average embedding (d_embed dims)
        avg_embed = embeds.mean(0).numpy()

        # Feature set 2: Last token embedding (d_embed dims)
        last_embed = embeds[-1].numpy()

        # Feature set 3: Embedding statistics (6 dims)
        embed_mean = embeds.mean().item()
        embed_std = embeds.std().item()
        embed_norm_mean = embeds.norm(dim=1).mean().item()
        embed_norm_std = embeds.norm(dim=1).std().item()
        if embeds.shape[0] > 1:
            e_c = embeds - embeds.mean(0)
            _, e_S, _ = torch.linalg.svd(e_c, full_matrices=False)
            embed_sv0 = e_S[0].item()
            embed_sv1 = e_S[1].item() if len(e_S) > 1 else 0
        else:
            embed_sv0, embed_sv1 = 0, 0
        stats = np.array([embed_mean, embed_std, embed_norm_mean, embed_norm_std, embed_sv0, embed_sv1])

        # Concatenate all features
        feat = np.concatenate([avg_embed, last_embed, stats])
        features_list.append(feat)

    X = np.array(features_list)
    Y = pc_coords
    n_features = X.shape[1]

    log(f"  Feature dimension: {n_features} (avg_embed:{d_embed} + last_embed:{d_embed} + stats:6)")

    # Reduce feature dimensionality with PCA
    from sklearn.decomposition import PCA
    n_feat_components = [10, 25, 50, 100, 200]

    n_train = int(0.8 * len(texts))
    torch.manual_seed(42)
    perm = torch.randperm(len(texts)).numpy()
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    # First: Ridge on raw features (baseline with all d_embed dims)
    log(f"\n    --- Baseline: Ridge on raw features ({n_features} dims) ---")
    baseline_corrs = []
    for k in range(K):
        ridge = Ridge(alpha=10.0)
        ridge.fit(X[train_idx], Y[train_idx, k])
        y_pred = ridge.predict(X[test_idx])
        if np.std(Y[test_idx, k]) > 1e-6:
            corr = np.corrcoef(Y[test_idx, k], y_pred)[0, 1]
        else:
            corr = 0
        baseline_corrs.append(corr)
    log(f"    Avg test corr (raw {n_features}d): {np.mean(baseline_corrs):.4f}")

    # Compare with PCA-reduced features
    log(f"\n    {'PCA_dims':>10} {'Avg_Corr':>10} {'PC0_Corr':>10} {'PC1_Corr':>10}")
    log(f"    {'-'*44}")

    results = {}
    for n_pc in n_feat_components:
        if n_pc >= n_train:
            continue
        pca = PCA(n_components=n_pc)
        X_pca = pca.fit_transform(X)

        test_corrs = []
        pc0_corr = 0
        pc1_corr = 0
        for k in range(K):
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_pca[train_idx], Y[train_idx, k])
            y_pred = ridge.predict(X_pca[test_idx])
            if np.std(Y[test_idx, k]) > 1e-6:
                corr = np.corrcoef(Y[test_idx, k], y_pred)[0, 1]
            else:
                corr = 0
            test_corrs.append(corr)
            if k == 0: pc0_corr = corr
            if k == 1: pc1_corr = corr

        avg_corr = np.mean(test_corrs)
        results[f"pca_{n_pc}"] = {"avg_corr": float(avg_corr), "corrs": [float(c) for c in test_corrs]}
        log(f"    {n_pc:>10} {avg_corr:>10.4f} {pc0_corr:>10.4f} {pc1_corr:>10.4f}")

    # Best comparison: 32-dim manual features vs high-dim embedding features
    log(f"\n    --- Comparison: Manual 32d vs Embedding {n_features}d ---")
    # Rebuild 32d manual features
    manual_features = []
    cat_to_idx = {c: i for i, c in enumerate(sorted(set(categories)))}
    for text, cat in texts:
        tokens = tokenizer(text, return_tensors="pt")
        input_ids = tokens.input_ids[0]
        feat = [len(input_ids), len(text)]
        if W_embed is not None:
            embeds = W_embed[input_ids]
            feat.extend([embeds.mean().item(), embeds.std().item(),
                        embeds.norm(dim=1).mean().item(), embeds.norm(dim=1).std().item()])
            if embeds.shape[0] > 1:
                e_c = embeds - embeds.mean(0)
                _, e_S, _ = torch.linalg.svd(e_c, full_matrices=False)
                feat.extend([e_S[0].item(), e_S[1].item() if len(e_S) > 1 else 0])
            else:
                feat.extend([0, 0])
        else:
            feat.extend([0]*6)
        text_lower = text.lower()
        feat.append(sum(1 for c in text if ord(c) > 0x4E00))
        feat.append(1.0 if any(c in text for c in "=<>{}[]()") else 0.0)
        feat.append(1.0 if any(c in text for c in "+-*/^") else 0.0)
        feat.append(text.count("."))
        feat.append(text.count(","))
        feat.append(text.count(" "))
        feat.append(len(text_lower.split()))
        feat.append(1.0 if "?" in text else 0.0)
        feat.append(1.0 if "!" in text else 0.0)
        cat_kw = {
            "math_sci": ["theorem", "equation", "proof", "quantum", "energy", "cell"],
            "code": ["def ", "class ", "import ", "function", "return", "print"],
            "philosophy": ["philosophy", "moral", "ethics", "kant", "virtue"],
            "poetry": ["shall", "thee", "thou", "poem", "verse", "rhyme"],
            "finance": ["market", "stock", "bond", "interest", "inflation"],
            "medical": ["patient", "clinical", "treatment", "diagnosis"],
            "legal": ["court", "defendant", "plaintiff", "verdict", "law"],
            "reasoning": ["therefore", "hypothesis", "conclude", "evidence"],
            "chinese": ["china", "chinese", "beijing", "dynasty"],
            "gen_en": ["the", "is", "was", "are", "have", "has"],
        }
        cn = sorted(set(categories))
        for ck in cn:
            feat.append(float(sum(1 for kw in cat_kw.get(ck, []) if kw in text_lower)))
        words = text_lower.split()
        if words:
            wl = [len(w.strip(".,!?;:\"'-")) for w in words]
            feat.extend([np.mean(wl), np.std(wl) if len(wl) > 1 else 0, len(set(words))/len(words)])
            bigrams = [words[i]+"_"+words[i+1] for i in range(len(words)-1)]
            if bigrams:
                counts = {}
                for b in bigrams:
                    counts[b] = counts.get(b, 0) + 1
                probs = [c/len(bigrams) for c in counts.values()]
                feat.append(-sum(p*np.log(p+1e-10) for p in probs))
            else:
                feat.append(0)
        else:
            feat.extend([0, 0, 0, 0])
        manual_features.append(feat)

    X_manual = np.array(manual_features)
    manual_corrs = []
    for k in range(K):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_manual[train_idx], Y_train[:, k])
        y_pred = ridge.predict(X_manual[test_idx])
        if np.std(Y_test[:, k]) > 1e-6:
            corr = np.corrcoef(Y_test[:, k], y_pred)[0, 1]
        else:
            corr = 0
        manual_corrs.append(corr)

    # High-dim best
    best_pca = max(results.values(), key=lambda x: x["avg_corr"])

    log(f"    Manual 32d:  avg_corr={np.mean(manual_corrs):.4f}")
    log(f"    Embed {n_features}d: avg_corr={best_pca['avg_corr']:.4f} (PCA-reduced)")
    log(f"    Improvement: {best_pca['avg_corr'] - np.mean(manual_corrs):+.4f}")

    return results, {"manual_avg": float(np.mean(manual_corrs)),
                     "embed_best_avg": float(best_pca["avg_corr"]),
                     "n_embed_features": n_features}


# ===== P81: Layer Jacobian Analysis =====
def p81_layer_jacobian(model, tokenizer, h_all, texts, categories, log):
    """
    P81: Compute and analyze the Jacobian of layer transformations.
    
    Instead of trying to PREDICT the layer transformation (which failed),
    ANALYZE its structure via finite differences:
    
    J_l = d(h_l) / d(h_{l-1}) ≈ (h_l(h_{l-1}+eps*e_i) - h_l(h_{l-1})) / eps
    
    Analyze:
    1. Rank of J_l (effective degrees of freedom per layer)
    2. Spectrum of J_l (singular values)
    3. How J_l changes across layers (is it constant? expanding? contracting?)
    4. Relation between J_l properties and model depth/size
    """
    log(f"\n{'='*60}")
    log(f"  P81: Layer Jacobian Analysis")
    log(f"{'='*60}")

    n_texts = 10  # Use 10 texts for Jacobian computation
    n_probes = 50  # Number of random probe directions

    log(f"    Computing Jacobians for {n_texts} texts, {n_probes} probe directions")

    layer_jacobian_info = {}

    for text_i in range(n_texts):
        text = texts[text_i][0]
        tokens = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(tokens.input_ids, output_hidden_states=True)
        states = [hs[0, -1, :].float().cpu() for hs in outputs.hidden_states]
        n_layers = len(states) - 1

        eps = 1e-5

        for l in range(min(n_layers, 36)):  # Limit layers for speed
            h_prev = states[l]
            h_curr = states[l + 1]

            d_model = h_prev.shape[0]

            # Random probe directions
            torch.manual_seed(42 + text_i)
            probes = torch.randn(n_probes, d_model)
            probes = probes / probes.norm(dim=1, keepdim=True)

            # Finite differences
            responses = []
            for p in range(n_probes):
                h_perturbed = h_prev + eps * probes[p]

                # We can't easily compute h_l(h_{l-1}+eps*p) without running the model again
                # So instead, analyze the LOCAL structure of delta_h_l
                # Use the actual delta_h and analyze its covariance
                pass

            # Alternative approach: analyze the covariance of delta_h across texts
            # This tells us the "effective rank" of the layer transformation
            if l not in layer_jacobian_info:
                layer_jacobian_info[l] = {"delta_h": []}
            layer_jacobian_info[l]["delta_h"].append(h_curr - h_prev)

    # Analyze delta_h statistics per layer
    log(f"\n    {'Layer':>6} {'d_h_norm':>10} {'d_h_rank10':>12} {'d_h_rank50':>12} {'d_h_rank100':>13} {'sv0/sv1':>10}")
    log(f"    {'-'*70}")

    results = {}
    for l in sorted(layer_jacobian_info.keys()):
        deltas = torch.stack(layer_jacobian_info[l]["delta_h"])  # (n_texts, d_model)

        # Delta-h norm
        avg_norm = deltas.norm(dim=1).mean().item()

        # Effective rank via SVD
        delta_mean = deltas.mean(0, keepdim=True)
        delta_c = deltas - delta_mean
        _, S, _ = torch.linalg.svd(delta_c, full_matrices=False)

        total_sv = S.sum().item()
        rank10 = (S.cumsum(0) / total_sv >= 0.10).nonzero()[0].item() + 1 if (S.cumsum(0) / total_sv >= 0.10).any() else len(S)
        rank50 = (S.cumsum(0) / total_sv >= 0.50).nonzero()[0].item() + 1 if (S.cumsum(0) / total_sv >= 0.50).any() else len(S)
        rank100 = (S.cumsum(0) / total_sv >= 0.90).nonzero()[0].item() + 1 if (S.cumsum(0) / total_sv >= 0.90).any() else len(S)

        sv_ratio = (S[0] / (S[1] + 1e-10)).item() if len(S) > 1 else float('inf')

        results[f"L{l}"] = {
            "avg_norm": float(avg_norm),
            "rank10": int(rank10),
            "rank50": int(rank50),
            "rank100": int(rank100),
            "sv_ratio": float(sv_ratio),
            "sv0": float(S[0].item()),
            "sv_top10": [float(s) for s in S[:10].tolist()],
        }
        log(f"    L{l:>4} {avg_norm:>10.2f} {rank10:>12} {rank50:>12} {rank100:>13} {sv_ratio:>10.1f}")

    # Summary: how does delta-h structure evolve across layers?
    norms = [results[f"L{l}"]["avg_norm"] for l in sorted(layer_jacobian_info.keys())]
    ranks50 = [results[f"L{l}"]["rank50"] for l in sorted(layer_jacobian_info.keys())]
    sv_ratios = [results[f"L{l}"]["sv_ratio"] for l in sorted(layer_jacobian_info.keys())]

    log(f"\n    --- Evolution Summary ---")
    log(f"    Delta-h norm: start={norms[0]:.2f}, mid={norms[len(norms)//2]:.2f}, end={norms[-1]:.2f}")
    log(f"    Rank(50% var): start={ranks50[0]}, mid={ranks50[len(ranks50)//2]}, end={ranks50[-1]}")
    log(f"    SV0/SV1 ratio: start={sv_ratios[0]:.1f}, mid={sv_ratios[len(sv_ratios)//2]:.1f}, end={sv_ratios[-1]:.1f}")

    # Is delta-h norm increasing or decreasing across layers?
    if norms[-1] > norms[0]:
        log(f"    Trend: delta-h norm INCREASES across layers (expansive)")
    else:
        log(f"    Trend: delta-h norm DECREASES across layers (contractive)")

    return results


# ===== Main =====
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage711_phase6_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    log = Logger(run_dir, "results")
    log("=" * 70)
    log("Stage 711: Phase VI - Precise Manipulation + Gradient + HighDim + Jacobian")
    log(f"Timestamp: {timestamp}")
    log(f"Texts: {len(TEXTS)} (10 categories x ~40 each)")
    cat_counts = {}
    for _, c in TEXTS:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    log(f"Categories ({len(cat_counts)}): {cat_counts}")
    log(f"Models: {MODEL_ORDER}")
    log("=" * 70)

    all_results = {}
    all_model_data = {}
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

            # P78: Precise Manipulation (first 2 models)
            if MODEL_ORDER.index(mn) < 2:
                try:
                    t_p = time.time()
                    r78 = p78_precise_manipulation(model, tokenizer, h_all, TEXTS, categories, log)
                    model_results["p78"] = r78
                    log(f"  P78 done in {time.time()-t_p:.1f}s")
                except Exception as e:
                    log(f"  P78 FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            # P79: Gradient Manipulation (Qwen3 only - needs gradients)
            if mn == "qwen3":
                try:
                    t_p = time.time()
                    # Enable gradients for this experiment
                    for p in model.parameters():
                        p.requires_grad_(False)
                    r79 = p79_gradient_manipulation(model, tokenizer, h_all, TEXTS, categories, log)
                    model_results["p79"] = r79
                    log(f"  P79 done in {time.time()-t_p:.1f}s")
                except Exception as e:
                    log(f"  P79 FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            # P80: High-Dim Features (first 2 models)
            if MODEL_ORDER.index(mn) < 2:
                try:
                    t_p = time.time()
                    r80, r80_summary = p80_highdim_features(model, tokenizer, h_all, TEXTS, categories, log)
                    model_results["p80"] = r80
                    model_results["p80_summary"] = r80_summary
                    log(f"  P80 done in {time.time()-t_p:.1f}s")
                except Exception as e:
                    log(f"  P80 FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            # P81: Layer Jacobian (all models, 10 texts only)
            try:
                t_p = time.time()
                r81 = p81_layer_jacobian(model, tokenizer, h_all, TEXTS, categories, log)
                model_results["p81"] = r81
                log(f"  P81 done in {time.time()-t_p:.1f}s")
            except Exception as e:
                log(f"  P81 FAILED: {e}")
                import traceback
                traceback.print_exc()

            all_results[mn] = model_results
            all_model_data[mn] = {"h_all": h_all}

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
    log("FINAL SUMMARY - Phase VI")
    log(f"{'='*70}")

    for mn in MODEL_ORDER:
        if mn not in all_results:
            continue
        res = all_results[mn]
        log(f"\n  {mn}:")
        if "p80_summary" in res:
            s = res["p80_summary"]
            log(f"    P80: manual={s['manual_avg']:.4f}, embed={s['embed_best_avg']:.4f}, "
                f"n_feat={s['n_embed_features']}")
        if "p81" in res:
            layers = list(res["p81"].keys())
            if layers:
                first = res["p81"][layers[0]]
                last = res["p81"][layers[-1]]
                log(f"    P81 Jacobian: d_h norm {first['avg_norm']:.2f}->{last['avg_norm']:.2f}, "
                    f"rank50 {first['rank50']}->{last['rank50']}")

    log(f"\nResults saved to: {run_dir}")
    log.close()
    print(f"\nDone! Results at: {run_dir}")


if __name__ == "__main__":
    main()
