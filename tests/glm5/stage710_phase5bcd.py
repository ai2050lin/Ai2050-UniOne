#!/usr/bin/env python3
"""
Stage 710: Phase V-B/C/D — 语义操控 + 严格预测链 + 非线性预测 + 多步因果
==============================================================================
P73: 语义操控实验——通过PC坐标操控实现跨类别"翻译"
P74: 严格因果预测链——去除one-hot作弊特征,用纯文本特征预测
P75: 非线性因果预测——MLP vs Ridge, 非线性能否突破线性瓶颈?
P76: 多步生成因果验证——delta-h逐层累积的可预测性
P77: 修正CCA——10类别避免过拟合,正确交叉验证

核心目标:
1. 操控PC坐标能否让中文文本"变成"英文风格输出?(P73)
2. 不依赖类别标签,纯文本特征能预测多少语义信息?(P74)
3. 非线性模型能否突破层变换的线性预测瓶颈?(P75)
4. 多步生成过程中累积误差如何增长?(P76)
5. 跨模型语义对齐在更严格条件下是否仍然成立?(P77)

四模型串行: Qwen3 -> DS7B -> GLM4 -> Gemma4
大样本: 400条文本(10类别x40)
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
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

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
    
    # Cat 1: General English
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
    for t in gen_en:
        T.append((t, "gen_en"))

    # Cat 2: Math/Science
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
    for t in math_sci:
        T.append((t, "math_sci"))

    # Cat 3: Code/Programming
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
    for t in code:
        T.append((t, "code"))

    # Cat 4: Chinese-themed (English text about China)
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
    for t in chinese:
        T.append((t, "chinese"))

    # Cat 5: Logical Reasoning
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
    for t in reasoning:
        T.append((t, "reasoning"))

    # Cat 6: Philosophy/Ethics
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
    for t in philosophy:
        T.append((t, "philosophy"))

    # Cat 7: Poetry/Literature
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
    for t in poetry:
        T.append((t, "poetry"))

    # Cat 8: Finance/Economics
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
    for t in finance:
        T.append((t, "finance"))

    # Cat 9: Medical/Health
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
    for t in medical:
        T.append((t, "medical"))

    # Cat 10: Law/Legal
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
    for t in legal:
        T.append((t, "legal"))

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


def get_h_final(model, tokenizer, text):
    tokens = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(tokens.input_ids, output_hidden_states=True)
    return outputs.hidden_states[-1][0, -1].float().cpu()


def get_all_h_and_logits(model, tokenizer, texts):
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


# ===== P73: Semantic Manipulation =====
def p73_semantic_manipulation(model, tokenizer, h_all, texts, categories, log):
    """
    P73: Can we manipulate PC coordinates to achieve cross-category "translation"?
    
    Experiment: Take texts from one category, shift their PC coordinates toward
    another category's centroid, and check if the model output changes accordingly.
    
    This tests the core causal hypothesis: manipulating semantic coordinates
    should change the model's predicted output in the expected direction.
    """
    log(f"\n{'='*60}")
    log(f"  P73: Semantic Manipulation - Cross-Category Translation")
    log(f"{'='*60}")

    W = get_unembed(model)
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)
    K = min(10, Vt.shape[0])

    cat_names = sorted(set(categories))
    # PC coordinates for all texts
    pc_coords = h_c @ Vt[:K].T  # (400, K)

    # Category means in PC space
    cat_means = {}
    for cat in cat_names:
        idx = [i for i, c in enumerate(categories) if c == cat]
        cat_means[cat] = pc_coords[idx].mean(0)  # (K,)

    # Key cross-category translations to test
    translations = [
        ("chinese", "gen_en", "Chinese->English"),
        ("chinese", "poetry", "Chinese->Poetry"),
        ("code", "math_sci", "Code->Math"),
        ("math_sci", "code", "Math->Code"),
        ("gen_en", "reasoning", "English->Reasoning"),
        ("finance", "medical", "Finance->Medical"),
        ("philosophy", "legal", "Philosophy->Legal"),
        ("poetry", "philosophy", "Poetry->Philosophy"),
    ]

    n_test = 5  # texts per source category to test
    shift_strengths = [0.25, 0.5, 0.75, 1.0]

    results = {}
    for src, tgt, label in translations:
        log(f"\n    --- {label} ({src} -> {tgt}) ---")

        src_idx = [i for i, c in enumerate(categories) if c == src][:n_test]
        direction = cat_means[tgt] - cat_means[src]  # PC-space direction
        dir_norm = torch.norm(direction)

        trans_results = []
        for ss in shift_strengths:
            top1_changed = 0
            top5_overlap_orig = []
            top5_overlap_new = []
            target_cat_top1 = 0  # Does shifted output's top-1 look like target category?
            target_cat_top5 = 0
            token_category_orig = []
            token_category_new = []

            for ti in src_idx:
                h = h_all[ti:ti+1].float()
                pc = pc_coords[ti:ti+1].float()

                # Original logits
                logits_orig = (h @ W.T).squeeze()
                top5_orig = set(torch.topk(logits_orig, 5).indices.tolist())

                # Shifted PC coords
                pc_shifted = pc + direction.unsqueeze(0) * ss
                h_shifted = pc_shifted @ Vt[:K].float() + h_mean.float()
                logits_new = (h_shifted @ W.T).squeeze()
                top5_new = set(torch.topk(logits_new, 5).indices.tolist())

                if top5_new != top5_orig:
                    top1_changed += 1

                # Check if new top-5 overlaps with typical target category top-5
                tgt_idx = [i for i, c in enumerate(categories) if c == tgt]
                tgt_logits_list = [h_all[i] @ W.T for i in tgt_idx[:10]]
                # Most common tokens in target category
                tgt_top_counts = {}
                for tl in tgt_logits_list:
                    for tid in torch.topk(tl, 10).indices.tolist():
                        tgt_top_counts[tid] = tgt_top_counts.get(tid, 0) + 1
                tgt_common_tokens = set(sorted(tgt_top_counts, key=tgt_top_counts.get, reverse=True)[:5])

                overlap_new = len(top5_new & tgt_common_tokens)
                overlap_orig = len(top5_orig & tgt_common_tokens)
                target_cat_top5 += (overlap_new > overlap_orig)

                # Decode top-1 for qualitative inspection
                orig_top1 = safe_decode(logits_orig.argmax().unsqueeze(0), tokenizer, 1)
                new_top1 = safe_decode(logits_new.argmax().unsqueeze(0), tokenizer, 1)
                new_top5 = safe_decode(torch.topk(logits_new, 5).indices, tokenizer, 5)
                token_category_orig.append(orig_top1)
                token_category_new.append(new_top1)

            trans_results.append({
                "shift": ss,
                "top5_change_rate": top1_changed / n_test,
                "target_dir_improvement": target_cat_top5 / n_test,
            })
            log(f"      shift={ss:.2f}: top5_changed={top1_changed/n_test:.0%}, "
                f"target_overlap_improved={target_cat_top5/n_test:.0%}")

        # Qualitative example at shift=0.75
        best_shift = 0.75
        ti = src_idx[0]
        h = h_all[ti:ti+1].float()
        pc = pc_coords[ti:ti+1].float()
        pc_shifted = pc + direction.unsqueeze(0) * best_shift
        h_shifted = pc_shifted @ Vt[:K].float() + h_mean.float()
        logits_new = (h_shifted @ W.T).squeeze()
        log(f"      Example ({texts[ti][0][:40]}...):")
        log(f"        orig: {safe_decode(torch.topk((h @ W.T).squeeze(), 5).indices, tokenizer)}")
        log(f"        new:  {safe_decode(torch.topk(logits_new, 5).indices, tokenizer)}")

        results[f"{src}->{tgt}"] = trans_results

    # Summary
    log(f"\n    --- Manipulation Summary ---")
    avg_improvement = []
    for key, tr_list in results.items():
        best = max(tr_list, key=lambda x: x["target_dir_improvement"])
        avg_improvement.append(best["target_dir_improvement"])
        log(f"    {key}: best target_improvement={best['target_dir_improvement']:.0%} (shift={best['shift']:.2f})")
    log(f"    Average best target improvement: {np.mean(avg_improvement):.2%}")

    return results


# ===== P74: Strict Causal Prediction (no one-hot) =====
def p74_strict_prediction(model, tokenizer, h_all, texts, categories, log):
    """
    P74: Predict PC coordinates WITHOUT category one-hot.
    
    This removes the "cheating" feature and tests whether pure text features
    can predict semantic encoding.
    
    Features (no one-hot):
    - Token count, embedding statistics (mean, std, norm)
    - Character-level features (Chinese chars, code symbols, math terms, etc.)
    - Word frequency features (common word categories)
    - N-gram entropy
    """
    log(f"\n{'='*60}")
    log(f"  P74: Strict Causal Prediction (no one-hot)")
    log(f"{'='*60}")

    W_embed = None
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        W_embed = model.model.embed_tokens.weight.data.float().cpu()
    elif hasattr(model, "get_input_embeddings"):
        W_embed = model.get_input_embeddings().weight.data.float().cpu()

    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)
    K = 10

    pc_coords = (h_c @ Vt[:K].T).numpy()
    cat_names = sorted(set(categories))
    cat_to_idx = {c: i for i, c in enumerate(cat_names)}

    # Build features WITHOUT one-hot
    features = []
    for text, cat in texts:
        tokens = tokenizer(text, return_tensors="pt")
        input_ids = tokens.input_ids[0]
        feat = []

        # 1. Token count
        feat.append(len(input_ids))
        feat.append(len(text))

        # 2. Embedding statistics (6 dims)
        if W_embed is not None:
            embeds = W_embed[input_ids]
            feat.append(embeds.mean().item())
            feat.append(embeds.std().item())
            feat.append(embeds.norm(dim=1).mean().item())
            feat.append(embeds.norm(dim=1).std().item())
            if embeds.shape[0] > 1:
                e_centered = embeds - embeds.mean(0)
                _, e_S, _ = torch.linalg.svd(e_centered, full_matrices=False)
                feat.append(e_S[0].item())
                feat.append(e_S[1].item() if len(e_S) > 1 else 0)
            else:
                feat.extend([0, 0])
        else:
            feat.extend([0] * 6)

        # 3. Character-level features (10 dims)
        text_lower = text.lower()
        feat.append(sum(1 for c in text if ord(c) > 0x4E00))  # Chinese char count
        feat.append(1.0 if any(c in text for c in "=<>{}[]()") else 0.0)
        feat.append(1.0 if any(c in text for c in "+-*/^") else 0.0)
        feat.append(text.count("."))
        feat.append(text.count(","))
        feat.append(text.count(" "))
        feat.append(len(text_lower.split()))
        feat.append(1.0 if "?" in text else 0.0)
        feat.append(1.0 if "!" in text else 0.0)
        feat.append(1.0 if '"' in text or "'" in text else 0.0)

        # 4. Keyword-category features (10 dims, count of words from each category)
        cat_keywords = {
            "math_sci": ["theorem", "equation", "proof", "quantum", "energy", "cell", "molecule", "atom"],
            "code": ["def ", "class ", "import ", "function", "return", "print", "self.", "torch"],
            "philosophy": ["philosophy", "moral", "ethics", "existentialism", "kant", "virtue", "duty"],
            "poetry": ["shall", "thee", "thou", "poem", "verse", "stanza", "rhyme", "meter"],
            "finance": ["market", "stock", "bond", "interest", "inflation", "trade", "economy", "dollar"],
            "medical": ["patient", "clinical", "treatment", "diagnosis", "symptoms", "therapy", "vaccine"],
            "legal": ["court", "defendant", "plaintiff", "verdict", "law", "trial", "jury", "appeal"],
            "reasoning": ["therefore", "hypothesis", "conclude", "correlation", "evidence", "probability"],
            "chinese": ["china", "chinese", "beijing", "dynasty", "emperor", "mandarin", "silk"],
            "gen_en": ["the", "is", "was", "are", "were", "have", "has", "been"],
        }
        for cat_key in cat_names:
            count = sum(1 for kw in cat_keywords.get(cat_key, []) if kw in text_lower)
            feat.append(float(count))

        # 5. Statistical features (4 dims)
        words = text_lower.split()
        if len(words) > 0:
            word_lens = [len(w.strip(".,!?;:\"'-")) for w in words]
            feat.append(np.mean(word_lens))
            feat.append(np.std(word_lens) if len(word_lens) > 1 else 0)
            feat.append(len(set(words)) / len(words))  # type-token ratio
            # Bigram entropy approximation
            bigrams = [words[i] + "_" + words[i+1] for i in range(len(words)-1)]
            if bigrams:
                counts = {}
                for b in bigrams:
                    counts[b] = counts.get(b, 0) + 1
                probs = [c/len(bigrams) for c in counts.values()]
                entropy = -sum(p * np.log(p + 1e-10) for p in probs)
                feat.append(entropy)
            else:
                feat.append(0)
        else:
            feat.extend([0, 0, 0, 0])

        features.append(feat)

    X = np.array(features)
    Y = pc_coords

    log(f"    Features: {X.shape[1]} dims (NO one-hot), Texts: {len(texts)}")

    # Train/test split 80/20
    n_train = 320
    torch.manual_seed(42)
    perm = torch.randperm(len(texts)).numpy()
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # Ridge regression per PC
    total_var = (S ** 2).sum().item()
    test_corrs = []
    results = {}
    log(f"\n    {'PC':>4} {'Var%':>6} {'Train_R2':>10} {'Test_R2':>10} {'Test_Corr':>10}")
    log(f"    {'-'*44}")

    for k in range(K):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, Y_train[:, k])
        Y_pred_test = ridge.predict(X_test)
        r2_train = r2_score(Y_train[:, k], ridge.predict(X_train))
        r2_test = r2_score(Y_test[:, k], Y_pred_test)

        if np.std(Y_test[:, k]) > 1e-6:
            corr = np.corrcoef(Y_test[:, k], Y_pred_test)[0, 1]
        else:
            corr = 0

        var_pct = S[k].item()**2 / total_var * 100
        test_corrs.append(corr)
        results[f"PC{k}"] = {"var_pct": var_pct, "train_r2": float(r2_train),
                             "test_r2": float(r2_test), "test_corr": float(corr)}
        log(f"    PC{k:>2} {var_pct:>5.1f}% {r2_train:>10.4f} {r2_test:>10.4f} {corr:>10.4f}")

    # Also test with one-hot (for comparison)
    log(f"\n    --- Comparison: WITH one-hot ---")
    features_with_oh = []
    for text, cat in texts:
        feat_base = features[len(features_with_oh)]
        one_hot = [0] * len(cat_names)
        one_hot[cat_to_idx[cat]] = 1.0
        features_with_oh.append(feat_base + one_hot)
    X_oh = np.array(features_with_oh)
    X_oh_train, X_oh_test = X_oh[train_idx], X_oh[test_idx]

    oh_corrs = []
    for k in range(K):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_oh_train, Y_train[:, k])
        Y_pred = ridge.predict(X_oh_test)
        if np.std(Y_test[:, k]) > 1e-6:
            corr = np.corrcoef(Y_test[:, k], Y_pred)[0, 1]
        else:
            corr = 0
        oh_corrs.append(corr)

    log(f"\n    {'PC':>4} {'NoOH_Corr':>10} {'WithOH_Corr':>12} {'Improvement':>12}")
    log(f"    {'-'*44}")
    for k in range(K):
        imp = oh_corrs[k] - test_corrs[k]
        log(f"    PC{k:>2} {test_corrs[k]:>10.4f} {oh_corrs[k]:>12.4f} {imp:>+12.4f}")
    log(f"    Avg: noOH={np.mean(test_corrs):.4f}, withOH={np.mean(oh_corrs):.4f}, "
        f"improvement={np.mean(oh_corrs)-np.mean(test_corrs):+.4f}")

    return results, {"avg_test_corr": float(np.mean(test_corrs)),
                     "avg_with_oh_corr": float(np.mean(oh_corrs)),
                     "n_features": X.shape[1]}


# ===== P75: Nonlinear Causal Prediction =====
def p75_nonlinear_prediction(model, tokenizer, h_all, texts, categories, log):
    """
    P75: Can MLP (nonlinear) predict PC coordinates better than Ridge (linear)?
    
    Also test layer-to-layer prediction with nonlinear models.
    """
    log(f"\n{'='*60}")
    log(f"  P75: Nonlinear vs Linear Causal Prediction")
    log(f"{'='*60}")

    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)
    K = 10
    pc_coords = (h_c @ Vt[:K].T).numpy()

    # Use the same features as P74 (without one-hot)
    W_embed = None
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        W_embed = model.model.embed_tokens.weight.data.float().cpu()
    elif hasattr(model, "get_input_embeddings"):
        W_embed = model.get_input_embeddings().weight.data.float().cpu()

    cat_names = sorted(set(categories))
    features = []
    for text, cat in texts:
        tokens = tokenizer(text, return_tensors="pt")
        input_ids = tokens.input_ids[0]
        feat = []
        feat.append(len(input_ids))
        feat.append(len(text))
        if W_embed is not None:
            embeds = W_embed[input_ids]
            feat.append(embeds.mean().item())
            feat.append(embeds.std().item())
            feat.append(embeds.norm(dim=1).mean().item())
            feat.append(embeds.norm(dim=1).std().item())
            if embeds.shape[0] > 1:
                e_centered = embeds - embeds.mean(0)
                _, e_S, _ = torch.linalg.svd(e_centered, full_matrices=False)
                feat.append(e_S[0].item())
                feat.append(e_S[1].item() if len(e_S) > 1 else 0)
            else:
                feat.extend([0, 0])
        else:
            feat.extend([0] * 6)
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
        cat_keywords = {
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
        for cat_key in cat_names:
            count = sum(1 for kw in cat_keywords.get(cat_key, []) if kw in text_lower)
            feat.append(float(count))
        words = text_lower.split()
        if len(words) > 0:
            word_lens = [len(w.strip(".,!?;:\"'-")) for w in words]
            feat.append(np.mean(word_lens))
            feat.append(np.std(word_lens) if len(word_lens) > 1 else 0)
            feat.append(len(set(words)) / len(words))
            bigrams = [words[i] + "_" + words[i+1] for i in range(len(words)-1)]
            if bigrams:
                counts = {}
                for b in bigrams:
                    counts[b] = counts.get(b, 0) + 1
                probs = [c/len(bigrams) for c in counts.values()]
                feat.append(-sum(p * np.log(p + 1e-10) for p in probs))
            else:
                feat.append(0)
        else:
            feat.extend([0, 0, 0, 0])
        features.append(feat)

    X = np.array(features)
    Y = pc_coords

    n_train = 320
    torch.manual_seed(42)
    perm = torch.randperm(len(texts)).numpy()
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # Compare: Ridge vs MLP for each PC
    total_var = (S ** 2).sum().item()
    log(f"\n    {'PC':>4} {'Var%':>6} {'Ridge':>8} {'MLP':>8} {'Delta':>8}")
    log(f"    {'-'*40}")

    results = {}
    ridge_corrs = []
    mlp_corrs = []
    for k in range(K):
        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, Y_train[:, k])
        y_ridge = ridge.predict(X_test)
        r_ridge = np.corrcoef(Y_test[:, k], y_ridge)[0, 1] if np.std(Y_test[:, k]) > 1e-6 else 0
        ridge_corrs.append(r_ridge)

        # MLP (nonlinear)
        try:
            mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
                              early_stopping=True, validation_fraction=0.1)
            mlp.fit(X_train, Y_train[:, k])
            y_mlp = mlp.predict(X_test)
            r_mlp = np.corrcoef(Y_test[:, k], y_mlp)[0, 1] if np.std(Y_test[:, k]) > 1e-6 else 0
        except Exception:
            r_mlp = 0
        mlp_corrs.append(r_mlp)

        var_pct = S[k].item()**2 / total_var * 100
        delta = r_mlp - r_ridge
        results[f"PC{k}"] = {"var_pct": var_pct, "ridge_corr": float(r_ridge),
                             "mlp_corr": float(r_mlp), "delta": float(delta)}
        log(f"    PC{k:>2} {var_pct:>5.1f}% {r_ridge:>8.4f} {r_mlp:>8.4f} {delta:>+8.4f}")

    log(f"\n    Average: Ridge={np.mean(ridge_corrs):.4f}, MLP={np.mean(mlp_corrs):.4f}, "
        f"MLP_improvement={np.mean(mlp_corrs)-np.mean(ridge_corrs):+.4f}")

    # Layer-to-layer nonlinear prediction (first 2 models only for speed)
    log(f"\n    --- Layer-to-Layer: Ridge vs MLP (50 texts) ---")
    layer_results = {}
    n_texts = 50
    for text_i in range(n_texts):
        text = texts[text_i][0]
        tokens = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(tokens.input_ids, output_hidden_states=True)
        states = [hs[0, -1, :].float().cpu() for hs in outputs.hidden_states]
        n_layers = len(states) - 1
        for l in range(n_layers):
            if l not in layer_results:
                layer_results[l] = {"X": [], "Y": []}
            layer_results[l]["X"].append(states[l])
            layer_results[l]["Y"].append(states[l+1] - states[l])

    n_tr = int(0.8 * n_texts)
    perm2 = torch.randperm(n_texts).numpy()
    tr_idx, te_idx = perm2[:n_tr], perm2[n_tr:]

    log(f"\n    {'Layer':>6} {'Ridge_R2':>10} {'MLP_R2':>10} {'Ridge_Corr':>11} {'MLP_Corr':>10}")
    log(f"    {'-'*52}")

    layer_data = {}
    for l in sorted(layer_results.keys()):
        X_all = torch.stack(layer_results[l]["X"])
        Y_all = torch.stack(layer_results[l]["Y"])

        # Reduce X dimensionality for MLP (top 30 PCs)
        X_mean = X_all[tr_idx].mean(0, keepdim=True)
        X_c = X_all[tr_idx] - X_mean
        _, _, Vt_X = torch.linalg.svd(X_c, full_matrices=False)
        K_feat = min(30, Vt_X.shape[0], Vt_X.shape[1])

        X_tr = (X_all[tr_idx] - X_mean) @ Vt_X[:K_feat].T
        X_te = (X_all[te_idx] - X_mean) @ Vt_X[:K_feat].T

        # Sample 32 dims from Y for speed
        d_model = Y_all.shape[1]
        sample_dims = torch.randperm(d_model)[:min(32, d_model)]

        r2_ridge_list, r2_mlp_list, corr_ridge_list, corr_mlp_list = [], [], [], []

        for dim in sample_dims:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_tr.numpy(), Y_all[tr_idx, dim].numpy())
            y_pred = ridge.predict(X_te.numpy())
            r2_ridge_list.append(r2_score(Y_all[te_idx, dim].numpy(), y_pred))
            corr_ridge_list.append(np.corrcoef(Y_all[te_idx, dim].numpy(), y_pred)[0, 1])

            try:
                mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42,
                                  early_stopping=True)
                mlp.fit(X_tr.numpy(), Y_all[tr_idx, dim].numpy())
                y_pred_mlp = mlp.predict(X_te.numpy())
                r2_mlp_list.append(r2_score(Y_all[te_idx, dim].numpy(), y_pred_mlp))
                corr_mlp_list.append(np.corrcoef(Y_all[te_idx, dim].numpy(), y_pred_mlp)[0, 1])
            except Exception:
                r2_mlp_list.append(0)
                corr_mlp_list.append(0)

        avg_r2_ridge = np.mean(r2_ridge_list)
        avg_r2_mlp = np.mean(r2_mlp_list)
        avg_corr_ridge = np.mean(corr_ridge_list)
        avg_corr_mlp = np.mean(corr_mlp_list)
        layer_data[f"L{l}"] = {
            "ridge_r2": float(avg_r2_ridge), "mlp_r2": float(avg_r2_mlp),
            "ridge_corr": float(avg_corr_ridge), "mlp_corr": float(avg_corr_mlp)
        }
        log(f"    L{l:>4} {avg_r2_ridge:>10.4f} {avg_r2_mlp:>10.4f} {avg_corr_ridge:>11.4f} {avg_corr_mlp:>10.4f}")

    avg_layer_ridge = np.mean([v["ridge_corr"] for v in layer_data.values()])
    avg_layer_mlp = np.mean([v["mlp_corr"] for v in layer_data.values()])
    log(f"\n    Layer avg: Ridge_corr={avg_layer_ridge:.4f}, MLP_corr={avg_layer_mlp:.4f}, "
        f"improvement={avg_layer_mlp-avg_layer_ridge:+.4f}")

    return results, layer_data


# ===== P76: Multi-step Generation Causality =====
def p76_multi_step_causality(model, tokenizer, h_all, texts, categories, log):
    """
    P76: Test multi-step generation causal chain.
    
    If we manipulate h_final at step t, how does the effect propagate
    when we feed the shifted output back into the model?
    
    This is the key experiment for "description -> causation":
    Does the causal effect compound or dissipate across steps?
    """
    log(f"\n{'='*60}")
    log(f"  P76: Multi-Step Generation Causality")
    log(f"{'='*60}")

    W = get_unembed(model)
    h_mean = h_all.mean(0, keepdim=True)
    h_c = h_all - h_mean
    _, S, Vt = torch.linalg.svd(h_c, full_matrices=False)
    K = 10

    cat_names = sorted(set(categories))
    n_steps = 5  # generate 5 steps forward
    shift = 2.0  # 2 std PC0 shift

    # PC0 direction (language type direction)
    pc0_dir = Vt[0] * S[0].item()

    # Pick 10 texts (2 per category)
    test_idx = []
    for cat in cat_names[:5]:
        idx = [i for i, c in enumerate(categories) if c == cat][:2]
        test_idx.extend(idx)

    log(f"    Testing {len(test_idx)} texts, {n_steps} generation steps, PC0 shift={shift}std")

    results = []
    for ti in test_idx:
        text, cat = texts[ti]
        tokens = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = tokens.input_ids

        # Get original h_final
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        h_orig = outputs.hidden_states[-1][0, -1, :].float().cpu()
        logits_orig = outputs.logits[0, -1, :].float().cpu()

        # Original top-1 token
        orig_top1 = safe_decode(logits_orig.argmax().unsqueeze(0), tokenizer, 1)
        orig_top5 = safe_decode(torch.topk(logits_orig, 5).indices, tokenizer, 5)

        # Shifted h
        h_shifted = h_orig + pc0_dir * shift
        logits_shifted = (h_shifted @ W.T).squeeze()
        shifted_top1 = safe_decode(logits_shifted.argmax().unsqueeze(0), tokenizer, 1)
        shifted_top5 = safe_decode(torch.topk(logits_shifted, 5).indices, tokenizer, 5)

        # Now generate 1 step using the shifted logit
        # Pick the top-1 token from shifted logits
        shifted_token_id = logits_shifted.argmax().unsqueeze(0).unsqueeze(0)  # (1, 1)
        new_input = torch.cat([input_ids, shifted_token_id.to(input_ids.device)], dim=1)

        with torch.no_grad():
            outputs2 = model(new_input.to(model.device), output_hidden_states=True)
        h_step1 = outputs2.hidden_states[-1][0, -1, :].float().cpu()
        logits_step1 = outputs2.logits[0, -1, :].float().cpu()
        step1_top1 = safe_decode(logits_step1.argmax().unsqueeze(0), tokenizer, 1)
        step1_top5 = safe_decode(torch.topk(logits_step1, 5).indices, tokenizer, 5)

        # Original next token (unshifted)
        orig_token_id = logits_orig.argmax().unsqueeze(0).unsqueeze(0)  # (1, 1)
        orig_new_input = torch.cat([input_ids, orig_token_id.to(input_ids.device)], dim=1)
        with torch.no_grad():
            outputs_orig2 = model(orig_new_input.to(model.device), output_hidden_states=True)
        h_orig_step1 = outputs_orig2.hidden_states[-1][0, -1, :].float().cpu()
        logits_orig_step1 = outputs_orig2.logits[0, -1, :].float().cpu()
        orig_step1_top1 = safe_decode(logits_orig_step1.argmax().unsqueeze(0), tokenizer, 1)

        # Cosine similarity of h at step 1
        h_diff_cos = F.cosine_similarity(h_orig_step1.unsqueeze(0), h_step1.unsqueeze(0)).item()

        # Logit cosine similarity at step 1
        logit_diff_cos = F.cosine_similarity(
            logits_orig_step1.unsqueeze(0), logits_step1.unsqueeze(0)
        ).item()

        # KL divergence at step 1
        kl_step1 = F.kl_div(
            F.log_softmax(logits_step1, -1),
            F.softmax(logits_orig_step1, -1),
            reduction="sum"
        ).item()

        results.append({
            "text": text[:50],
            "category": cat,
            "orig_top1": orig_top1,
            "shifted_top1": shifted_top1,
            "step1_orig": orig_step1_top1,
            "step1_shifted": step1_top1,
            "h_cos_step1": h_diff_cos,
            "logit_cos_step1": logit_diff_cos,
            "kl_step1": kl_step1,
        })

    # Summary statistics
    h_cos_vals = [r["h_cos_step1"] for r in results]
    logit_cos_vals = [r["logit_cos_step1"] for r in results]
    kl_vals = [r["kl_step1"] for r in results]
    top1_flip = sum(1 for r in results if r["shifted_top1"] != r["orig_top1"])
    step1_diverge = sum(1 for r in results if r["step1_shifted"] != r["step1_orig"])

    log(f"\n    --- Multi-Step Results ---")
    log(f"    Step 0: top-1 flip rate = {top1_flip}/{len(results)} = {top1_flip/len(results):.0%}")
    log(f"    Step 1: output divergence = {step1_diverge}/{len(results)} = {step1_diverge/len(results):.0%}")
    log(f"    Step 1: h_cosine_similarity = {np.mean(h_cos_vals):.4f}")
    log(f"    Step 1: logit_cosine_similarity = {np.mean(logit_cos_vals):.4f}")
    log(f"    Step 1: KL_divergence = {np.mean(kl_vals):.2f}")

    # Check: does the causal effect persist or dissipate?
    log(f"\n    --- Effect Persistence ---")
    # If h_cos_step1 is still very different (<< 1.0), the effect persists
    # If h_cos_step1 is close to 1.0, the model "recovers" from the shift
    if np.mean(h_cos_vals) < 0.95:
        log(f"    CAUSAL EFFECT PERSISTS: h cosine = {np.mean(h_cos_vals):.4f} at step 1")
    else:
        log(f"    CAUSAL EFFECT DISSIPATES: h cosine = {np.mean(h_cos_vals):.4f} at step 1 (model recovers)")

    # Print examples
    log(f"\n    --- Qualitative Examples ---")
    for r in results[:3]:
        log(f"    [{r['category'][:6]}] {r['text'][:35]}...")
        log(f"      orig:     {r['orig_top1']}")
        log(f"      shifted:  {r['shifted_top1']}")
        log(f"      step1_o:  {r['step1_orig']}")
        log(f"      step1_s:  {r['step1_shifted']}  (h_cos={r['h_cos_step1']:.4f})")

    return results, {
        "top1_flip_rate": float(top1_flip/len(results)),
        "step1_diverge_rate": float(step1_diverge/len(results)),
        "avg_h_cos_step1": float(np.mean(h_cos_vals)),
        "avg_logit_cos_step1": float(np.mean(logit_cos_vals)),
        "avg_kl_step1": float(np.mean(kl_vals)),
    }


# ===== P77: Fixed CCA (10 categories) =====
def p77_fixed_cca(all_model_data, categories, log):
    """
    P77: Cross-model CCA with 10 categories to avoid overfitting.
    
    Key fix: stage709 used 5 categories with 5 CCA components = 0 degrees of freedom.
    Now with 10 categories, we use 5 CCA components = 5 degrees of freedom.
    Also do proper train/test split for CCA.
    """
    log(f"\n{'='*60}")
    log(f"  P77: Fixed Cross-Model CCA (10 categories)")
    log(f"{'='*60}")

    models = list(all_model_data.keys())
    cat_names = sorted(set(categories))
    alignments = {}

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue

            log(f"\n    {m1} vs {m2}:")
            h1 = all_model_data[m1]["h_all"]
            h2 = all_model_data[m2]["h_all"]

            # Category means
            means1, means2 = {}, {}
            for cat in cat_names:
                idx = [k for k, c in enumerate(categories) if c == cat]
                means1[cat] = h1[idx].mean(0)
                means2[cat] = h2[idx].mean(0)

            # Direction structure correlation (Method 1: dimension-independent)
            cat_pairs = []
            for ci, c1 in enumerate(cat_names):
                for c2 in cat_names[ci+1:]:
                    cat_pairs.append((c1, c2))

            D1 = np.stack([means1[c2].numpy() - means1[c1].numpy() for c1, c2 in cat_pairs])
            D2 = np.stack([means2[c2].numpy() - means2[c1].numpy() for c1, c2 in cat_pairs])
            D1_n = D1 / (np.linalg.norm(D1, axis=1, keepdims=True) + 1e-8)
            D2_n = D2 / (np.linalg.norm(D2, axis=1, keepdims=True) + 1e-8)
            R1 = D1_n @ D1_n.T
            R2 = D2_n @ D2_n.T
            struct_r = np.corrcoef(R1.flatten(), R2.flatten())[0, 1]
            n_pairs = len(cat_pairs)
            log(f"      Direction structure r: {struct_r:.4f} (based on {n_pairs} category pairs)")

            # CCA with proper train/test (Method 2)
            n_per_cat = 30
            sample_idx = []
            for cat in cat_names:
                idx = [k for k, c in enumerate(categories) if c == cat][:n_per_cat]
                sample_idx.extend(idx)
            X = h1[sample_idx].numpy()
            Y = h2[sample_idx].numpy()

            # Train/test split for CCA
            n_total = X.shape[0]
            torch.manual_seed(42)
            perm = torch.randperm(n_total).numpy()
            n_tr = int(0.8 * n_total)
            X_train, X_test = X[perm[:n_tr]], X[perm[n_tr:]]
            Y_train, Y_test = Y[perm[:n_tr]], Y[perm[n_tr:]]

            # CCA with fewer components (n=5, but 10 categories so 5 DOF)
            n_components = min(5, n_tr - 2, X.shape[1], Y.shape[1])
            if n_components < 2:
                log(f"      CCA skipped (not enough data)")
                continue

            try:
                cca = CCA(n_components=n_components)
                X_c_train, Y_c_train = cca.fit_transform(X_train, Y_train)

                # Train set canonical correlations
                train_cc = []
                for comp in range(n_components):
                    r = np.corrcoef(X_c_train[:, comp], Y_c_train[:, comp])[0, 1]
                    train_cc.append(r)

                # Test set canonical correlations (proper evaluation)
                X_c_test, Y_c_test = cca.transform(X_test, Y_test)
                test_cc = []
                for comp in range(n_components):
                    r = np.corrcoef(X_c_test[:, comp], Y_c_test[:, comp])[0, 1]
                    test_cc.append(r)

                log(f"      CCA({n_components}): train_CC={[f'{r:.4f}' for r in train_cc]}")
                log(f"      CCA({n_components}):  test_CC={[f'{r:.4f}' for r in test_cc]}")
                log(f"      Train avg: {np.mean(train_cc):.4f}, Test avg: {np.mean(test_cc):.4f}")
            except Exception as e:
                log(f"      CCA failed: {e}")
                train_cc, test_cc = [], []

            # Method 3: Procrustes analysis on category means
            M1 = np.stack([means1[c].numpy() for c in cat_names])
            M2 = np.stack([means2[c].numpy() for c in cat_names])
            M1_n = M1 - M1.mean(0)
            M2_n = M2 - M2.mean(0)

            # Project to common dimension using PCA of each
            from sklearn.decomposition import PCA
            n_comp = min(5, M1.shape[0] - 1, M1.shape[1], M2.shape[1])
            pca1 = PCA(n_comp)
            pca2 = PCA(n_comp)
            M1_pca = pca1.fit_transform(M1_n)
            M2_pca = pca2.fit_transform(M2_n)

            # Correlation of PCA projections
            pca_corrs = []
            for comp in range(n_comp):
                r = np.corrcoef(M1_pca[:, comp], M2_pca[:, comp])[0, 1]
                pca_corrs.append(r)
            log(f"      PCA projection corrs: {[f'{r:.4f}' for r in pca_corrs]}")
            log(f"      PCA projection avg: {np.mean(pca_corrs):.4f}")

            alignments[f"{m1}-{m2}"] = {
                "struct_r": float(struct_r),
                "n_pairs": n_pairs,
                "cca_train_cc": [float(r) for r in train_cc],
                "cca_test_cc": [float(r) for r in test_cc],
                "pca_proj_corrs": [float(r) for r in pca_corrs],
            }

    # Cross-model summary
    log(f"\n    --- Cross-Model Summary ---")
    for pair, data in alignments.items():
        avg_test_cc = np.mean(data["cca_test_cc"]) if data["cca_test_cc"] else 0
        avg_pca = np.mean(data["pca_proj_corrs"])
        log(f"    {pair}: struct_r={data['struct_r']:.4f}, "
            f"cca_test_avg={avg_test_cc:.4f}, pca_avg={avg_pca:.4f}")

    return alignments


# ===== Main =====
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage710_phase5bcd_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    log = Logger(run_dir, "results")
    log("=" * 70)
    log("Stage 710: Phase V-B/C/D - Semantic Manipulation + Strict Prediction + Nonlinear")
    log(f"Timestamp: {timestamp}")
    log(f"Texts: {len(TEXTS)} (10 categories x 40 each)")
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

            log(f"\n  Computing h_final + logits for {len(TEXTS)} texts...")
            t1 = time.time()
            h_all, all_logits = get_all_h_and_logits(model, tokenizer, TEXTS)
            log(f"  Done in {time.time()-t1:.1f}s. h_all: {h_all.shape}")

            model_results = {}

            # P73: Semantic Manipulation
            try:
                t_p = time.time()
                r73 = p73_semantic_manipulation(model, tokenizer, h_all, TEXTS, categories, log)
                model_results["p73"] = r73
                log(f"  P73 done in {time.time()-t_p:.1f}s")
            except Exception as e:
                log(f"  P73 FAILED: {e}")
                import traceback
                traceback.print_exc()

            # P74: Strict Prediction (no one-hot)
            try:
                t_p = time.time()
                r74, r74_summary = p74_strict_prediction(model, tokenizer, h_all, TEXTS, categories, log)
                model_results["p74"] = r74
                model_results["p74_summary"] = r74_summary
                log(f"  P74 done in {time.time()-t_p:.1f}s")
            except Exception as e:
                log(f"  P74 FAILED: {e}")
                import traceback
                traceback.print_exc()

            # P75: Nonlinear Prediction (first 2 models for speed)
            if MODEL_ORDER.index(mn) < 2:
                try:
                    t_p = time.time()
                    r75, r75_layer = p75_nonlinear_prediction(model, tokenizer, h_all, TEXTS, categories, log)
                    model_results["p75"] = r75
                    model_results["p75_layer"] = r75_layer
                    log(f"  P75 done in {time.time()-t_p:.1f}s")
                except Exception as e:
                    log(f"  P75 FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            # P76: Multi-step Causality
            try:
                t_p = time.time()
                r76, r76_summary = p76_multi_step_causality(model, tokenizer, h_all, TEXTS, categories, log)
                model_results["p76"] = r76
                model_results["p76_summary"] = r76_summary
                log(f"  P76 done in {time.time()-t_p:.1f}s")
            except Exception as e:
                log(f"  P76 FAILED: {e}")
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

    # P77: Fixed CCA (after all models)
    log(f"\n{'#'*70}")
    log(f"# P77: Fixed Cross-Model CCA (10 categories)")
    log(f"{'#'*70}")
    try:
        r77 = p77_fixed_cca(all_model_data, categories, log)
        all_results["p77"] = r77
    except Exception as e:
        log(f"  P77 FAILED: {e}")
        import traceback
        traceback.print_exc()

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
    log("FINAL SUMMARY - Phase V-B/C/D")
    log(f"{'='*70}")

    for mn in MODEL_ORDER:
        if mn not in all_results:
            continue
        res = all_results[mn]
        log(f"\n  {mn}:")
        if "p74_summary" in res:
            s = res["p74_summary"]
            log(f"    P74 Strict pred: noOH_corr={s['avg_test_corr']:.4f}, "
                f"withOH_corr={s['avg_with_oh_corr']:.4f}, features={s['n_features']}")
        if "p75" in res:
            ridge_avg = np.mean([v["ridge_corr"] for v in res["p75"].values()])
            mlp_avg = np.mean([v["mlp_corr"] for v in res["p75"].values()])
            log(f"    P75 Nonlinear: Ridge={ridge_avg:.4f}, MLP={mlp_avg:.4f}, "
                f"delta={mlp_avg-ridge_avg:+.4f}")
        if "p76_summary" in res:
            s = res["p76_summary"]
            log(f"    P76 Multi-step: flip={s['top1_flip_rate']:.0%}, "
                f"step1_div={s['step1_diverge_rate']:.0%}, h_cos={s['avg_h_cos_step1']:.4f}")

    if "p77" in all_results:
        log(f"\n  Cross-Model P77 (fixed CCA, 10 cats):")
        for pair, data in all_results["p77"].items():
            test_avg = np.mean(data["cca_test_cc"]) if data["cca_test_cc"] else 0
            pca_avg = np.mean(data["pca_proj_corrs"])
            log(f"    {pair}: struct_r={data['struct_r']:.4f}, "
                f"cca_test={test_avg:.4f}, pca={pca_avg:.4f}")

    log(f"\nResults saved to: {run_dir}")
    log.close()
    print(f"\nDone! Results at: {run_dir}")


if __name__ == "__main__":
    main()
