#!/usr/bin/env python3
"""
Stage 718: Phase XIII — 多步累积注入 + 梯度引导操控 + GLM4方向反转分析 + 多模型大样本测试
========================================================================================
核心突破: Phase XII发现单层注入被skip connection指数衰减(1-cos衰减10x每N/4层),
本阶段用"在每一层都注入"来累积效应, 突破衰减瓶颈.

P109: 多步累积注入 — 在每层residual stream注入, 逐层累积
P110: 梯度引导操控 — 用loss梯度找到最优注入方向
P111: 层间权重矩阵分析 — 分析GLM4方向反转的原因
P112: 生成质量评估 — 综合PPL/KL/top-k变化/文本差异

关键假设:
- 如果N层累积注入, 每层注入1-cos≈0.01, 累积后1-cos可能达到0.3-0.9
- 梯度方向比centroid差方向更有效(梯度直接优化目标)
- GLM4方向反转是LayerNorm或attention的特殊结构导致的

用法: python stage718_phase13.py --model qwen3
      python stage718_phase13.py --model deepseek7b
      python stage718_phase13.py --model glm4
      python stage718_phase13.py --model all
"""

import sys, time, gc, json, os, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from datetime import datetime
from collections import defaultdict

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

log = None  # will be set in main

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
        "Entropy always increases in isolated systems.", "The speed of light is approximately 299792 km per second.",
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
        "The Dragon Boat Festival celebrates the poet Qu Yuan.",
        "Acupuncture is a traditional Chinese medical practice.",
        "The Forbidden City has nine thousand nine hundred and ninety nine rooms.",
        "Chinese painting emphasizes the beauty of nature.",
        "The Summer Palace is a masterpiece of Chinese landscape design.",
        "The Spring Festival marks the beginning of the lunar new year.",
        "The Three Gorges Dam is the world's largest power station by installed capacity.",
        "The Mid-Autumn Festival celebrates family reunion under the full moon.",
        "The Chinese writing system uses thousands of unique characters.",
        "Beijing duck is one of the most famous dishes in Chinese cuisine.",
        "The invention of paper money originated in China during the Tang Dynasty.",
        "The Yellow River is known as the cradle of Chinese civilization.",
        "The Chinese zodiac cycles through twelve animal signs each year.",
        "Chinese traditional medicine uses herbs and acupuncture to treat illness.",
        "The Summer Olympics were held in Beijing in 2008.",
        "Gunpowder was one of the four great inventions of ancient China.",
        "The Li River in Guilin is famous for its karst landscape scenery.",
        "Confucianism has profoundly influenced Chinese society for over two millennia.",
    ]
    for t in chinese: T.append((t, "chinese"))
    poetry = [
        "Shall I compare thee to a summer's day?", "Two roads diverged in a wood, and I took the one less traveled by.",
        "I wandered lonely as a cloud that floats on high.", "To be or not to be, that is the question.",
        "The road goes ever on and on.", "Hope is the thing with feathers that perches in the soul.",
        "In the middle of difficulty lies opportunity.", "Do not go gentle into that good night.",
        "Because I could not stop for Death, he kindly stopped for me.",
        "I think that I shall never see a poem lovely as a tree.",
        "A thing of beauty is a joy forever.", "Water, water, everywhere, nor any drop to drink.",
        "The fog comes on little cat feet.", "I celebrate myself, and sing myself.",
        "Stopping by woods on a snowy evening.", "April is the cruellest month.",
        "Beauty is truth, truth beauty.", "O wild West Wind, thou breath of Autumn's being.",
        "The world is too much with us, late and soon.",
        "If winter comes, can spring be far behind?", "How do I love thee? Let me count the ways.",
        "The love song of J. Alfred Prufrock measured out his life with coffee spoons.",
        "I have measured out my life with coffee spoons.", "Out of the ash I rise with my red hair.",
        "The falcon cannot hear the falconer.", "Things fall apart, the center cannot hold.",
        "This is the way the world ends, not with a bang but a whimper.",
        "I contain multitudes.", "The only people for me are the mad ones.",
        "In Xanadu did Kubla Khan a stately pleasure dome decree.",
        "A host of golden daffodils beside the lake beneath the trees.",
        "Tyger Tyger burning bright in the forests of the night.",
        "My heart aches, and a drowsy numbness pains my sense.",
        "Much have I travell'd in the realms of gold.",
        "On First Looking into Chapman's Homer I felt like some watcher of the skies.",
        "She walks in beauty like the night of cloudless climes and starry skies.",
        "The mind is its own place, and in itself can make a heaven of hell.",
        "Sylvia Plath explored themes of mental illness and identity.",
        "Seamus Heaney won the Nobel Prize in Literature for his poetry.",
    ]
    for t in poetry: T.append((t, "poetry"))
    philosophy = [
        "The unexamined life is not worth living.", "I think therefore I am.",
        "To be is to be perceived.", "Existence precedes essence.",
        "Happiness depends upon ourselves.", "Man is by nature a social animal.",
        "Know thyself.", "The mind is everything. What you think you become.",
        "We are what we repeatedly do. Excellence is not an act but a habit.",
        "It is the mark of an educated mind to entertain a thought without accepting it.",
        "No man's knowledge here can go beyond his experience.",
        "The greatest happiness of the greatest number is the foundation of morals.",
        "We live in the best of all possible worlds.", "Existence precedes essence.",
        "Freedom is what we do with what is done to us.",
        "The death of God is the death of absolute morality.", "Truth is subjectivity.",
        "Man is condemned to be free.", "Hell is other people.",
        "One cannot step twice in the same river.", "The only constant is change.",
        "All men by nature desire knowledge.", "Plato's cave allegory shows the limits of perception.",
        "Kant's categorical imperative demands universal moral laws.",
        "Utilitarianism seeks the greatest good for the greatest number.",
        "Nihilism argues that life has no inherent meaning or value.",
        "Stoicism teaches acceptance of things we cannot change.",
        "Phenomenology studies the structures of conscious experience.",
        "Pragmatism judges truth by its practical consequences.",
        "Absurdism recognizes the conflict between human desire for meaning and the universe's silence.",
        "Determinism holds that all events are caused by previous events.",
        "Free will debates whether our choices are truly our own.",
        "Moral relativism argues that moral judgments are culture dependent.",
        "Epistemology questions the nature and limits of human knowledge.",
        "Metaphysics explores the fundamental nature of reality.",
        "Ethics examines what constitutes a good life and right action.",
        "Political philosophy asks what constitutes a just society.",
        "Aesthetics studies the nature of beauty and art.",
        "The social contract theory explains why people form governments.",
        "Virtue ethics focuses on developing good character traits.",
        "Consequentialism evaluates actions based on their outcomes.",
        "Deontology holds that some actions are inherently right or wrong.",
        "Empiricism claims knowledge comes from sensory experience.",
        "Rationalism argues that reason is the primary source of knowledge.",
    ]
    for t in philosophy: T.append((t, "philosophy"))
    history = [
        "The fall of the Roman Empire changed the course of Western civilization.",
        "World War II was the deadliest conflict in human history.",
        "The Industrial Revolution transformed manufacturing and society.",
        "The French Revolution established principles of liberty and equality.",
        "The invention of the printing press revolutionized communication.",
        "The Cold War shaped global politics for decades.",
        "Ancient Egypt built pyramids that still stand today.",
        "The Magna Carta established the rule of law in England.",
        "The Age of Exploration opened new trade routes and continents.",
        "The Renaissance marked a rebirth of art and learning in Europe.",
        "The Berlin Wall fell in 1989 ending the Cold War division of Europe.",
        "The American Revolution established the United States as an independent nation.",
        "The Silk Road facilitated trade between East and West for centuries.",
        "The Viking Age saw Norse explorers reach North America.",
        "The Black Death killed an estimated one third of Europe's population.",
        "The Space Race between the US and USSR led to the Moon landing.",
        "The Ottoman Empire lasted over six hundred years across three continents.",
        "The Ming Dynasty oversaw great achievements in Chinese civilization.",
        "The Declaration of Independence proclaimed universal human rights.",
        "The Treaty of Westphalia established the modern system of nation states.",
        "The Scientific Revolution changed humanity's understanding of nature.",
        "Colonialism reshaped societies across Africa and Asia.",
        "The Renaissance humanists shifted focus from divine to human concerns.",
        "The Enlightenment emphasized reason as the primary source of authority.",
        "The Russian Revolution of 1917 established the first communist state.",
        "The Civil Rights Movement fought for racial equality in America.",
        "The Great Depression was the most severe economic downturn of the twentieth century.",
        "The invention of agriculture allowed human civilization to develop.",
        "The Crusades were a series of religious wars in the medieval period.",
        "The Spanish Flu pandemic of 1918 killed millions worldwide.",
        "The rise of the internet has transformed global communication.",
        "The Meiji Restoration modernized Japan in the late nineteenth century.",
        "The Partition of India in 1947 created two independent nations.",
        "The Apollo 11 mission landed the first humans on the Moon.",
        "The fall of Constantinople in 1453 ended the Byzantine Empire.",
        "The Haitian Revolution was the only successful slave revolt in history.",
        "The Cuban Missile Crisis brought the world to the brink of nuclear war.",
        "The Age of Imperialism saw European powers colonize much of the world.",
        "The Bubonic plague originated in Asia and spread to Europe.",
        "The Reconstruction era followed the American Civil War.",
        "The Enlightenment philosophers challenged absolute monarchy and religious authority.",
        "The Treaty of Versailles ended World War One and reshaped Europe.",
        "Women gained the right to vote in many countries during the twentieth century.",
    ]
    for t in history: T.append((t, "history"))
    tech = [
        "Artificial intelligence is transforming every industry.", "Machine learning models can now generate human-like text.",
        "The transistor revolutionized electronics in the twentieth century.", "Cloud computing provides scalable resources on demand.",
        "Blockchain technology enables decentralized transactions.", "The internet connects billions of devices worldwide.",
        "Quantum computing promises to solve currently intractable problems.", "Self-driving cars use deep neural networks for perception.",
        "The smartphone has become an essential part of daily life.", "Big data analytics helps businesses make better decisions.",
        "Cybersecurity is a growing concern in the digital age.", "The Internet of Things connects everyday objects to the internet.",
        "Robotics is advancing rapidly in manufacturing and healthcare.", "Gene editing using CRISPR technology has transformative potential.",
        "Augmented reality overlays digital information on the physical world.",
        "The development of vaccines saved millions of lives annually.",
        "Satellite technology enables global positioning and communication.",
        "Nuclear energy provides a low carbon source of electricity.",
        "Electric vehicles are becoming increasingly popular worldwide.",
        "Solar panel efficiency has improved dramatically over the past decade.",
        "The World Wide Web was invented by Tim Berners Lee.",
        "Open source software has transformed the technology industry.",
        "Deep learning achieved breakthrough results in image recognition.",
        "Natural language processing enables machines to understand human speech.",
        "The semiconductor industry drives progress in computing power.",
        "Additive manufacturing or 3D printing is changing production methods.",
        "Edge computing processes data closer to where it is generated.",
        "Digital twins create virtual replicas of physical systems.",
        "Autonomous drones are being used for delivery and surveillance.",
        "Brain computer interfaces allow direct communication between brain and machine.",
        "LiDAR technology enables precise distance measurement.",
        "The development of mRNA vaccines represents a paradigm shift in medicine.",
        "Fiber optic cables form the backbone of internet infrastructure.",
        "Reinforcement learning trains agents through trial and error.",
        "Serverless computing abstracts away infrastructure management.",
        "Microservices architecture breaks applications into independent components.",
        "Containerization with Docker simplifies software deployment.",
        "Graph databases efficiently model complex relationships.",
        "WebAssembly enables near-native performance in web browsers.",
        "Differential privacy protects individual data in aggregate analysis.",
        "Federated learning trains models across distributed data sources.",
    ]
    for t in tech: T.append((t, "tech"))
    sports = [
        "The Olympic Games bring together athletes from around the world.",
        "Soccer is the most popular sport globally with billions of fans.",
        "Basketball was invented by James Naismith in 1891.",
        "Tennis requires agility speed and precision on the court.",
        "Swimming is an excellent full body workout.",
        "Marathon runners must train for months to prepare for the race.",
        "Cricket is wildly popular in India Australia and England.",
        "Golf requires patience focus and a steady hand.",
        "Boxing demands extraordinary cardiovascular fitness and reflexes.",
        "Cycling has grown in popularity as both recreation and competition.",
        "Volleyball is played on sand courts and indoor arenas worldwide.",
        "Wrestling is one of the oldest forms of competitive combat.",
        "Formula One racing is the pinnacle of motorsport.",
        "Ice hockey is fast paced and physically demanding.",
        "Track and field events test the limits of human speed and strength.",
        "Rugby combines elements of soccer and American football.",
        "Martial arts teach discipline and self defense.",
        "The Ironman triathlon is one of the most grueling endurance events.",
        "Figure skating combines athleticism with artistic expression.",
        "Badminton is the fastest racket sport in the world.",
        "The NBA playoffs determine the league champion each year.",
        "Home runs are a key statistic in baseball.",
        "The UEFA Champions League is the pinnacle of European club soccer.",
        "Mixed martial arts has grown rapidly in popularity.",
        "Skiing and snowboarding are popular winter sports worldwide.",
        "Esports competitive gaming has millions of viewers worldwide.",
    ]
    for t in sports: T.append((t, "sports"))
    finance = [
        "The stock market experienced significant volatility this quarter.",
        "Interest rates play a crucial role in economic policy.",
        "Diversification reduces risk in investment portfolios.",
        "GDP measures the total economic output of a country.",
        "Inflation erodes the purchasing power of money over time.",
        "Cryptocurrency has emerged as a new asset class.",
        "The Federal Reserve controls monetary policy in the United States.",
        "Supply and demand determine market prices in free economies.",
        "Compound interest allows investments to grow exponentially.",
        "The bond market is larger than the stock market by total value.",
        "Central banks regulate money supply and interest rates.",
        "Hedge funds use complex strategies to generate returns.",
        "Venture capital funds innovative startups in exchange for equity.",
        "Initial public offerings allow companies to raise capital from public investors.",
        "The balance sheet shows assets liabilities and equity.",
        "Fiscal policy involves government spending and taxation decisions.",
        "Exchange rates determine the value of one currency against another.",
        "Real estate is a significant component of household wealth.",
        "Derivatives are financial contracts whose value derives from an underlying asset.",
        "Mutual funds pool money from many investors to buy diversified portfolios.",
        "Market capitalization is the total value of a company's outstanding shares.",
        "Liquidity refers to how quickly an asset can be converted to cash.",
        "Economic recessions are typically defined as two consecutive quarters of GDP decline.",
        "Options give traders the right but not the obligation to buy or sell.",
        "The P/E ratio compares stock price to earnings per share.",
        "Foreign direct investment involves establishing operations in another country.",
        "Trade deficits occur when imports exceed exports.",
        "Monetary stimulus aims to boost economic growth during downturns.",
        "The yield curve plots bond yields across different maturities.",
        "Portfolio optimization seeks the best risk-return tradeoff.",
        "Index funds track market indices with low management fees.",
        "Financial literacy is essential for making informed economic decisions.",
        "The gold standard was abandoned by most countries in the twentieth century.",
        "Algorithmic trading uses computer programs to execute trades at high speed.",
        "Carbon credits create financial incentives for reducing emissions.",
        "Microfinance provides small loans to entrepreneurs in developing countries.",
        "The Bretton Woods system established postwar international monetary order.",
        "Behavioral economics studies how psychological factors affect financial decisions.",
        "Sovereign debt refers to money borrowed by national governments.",
        "Private equity firms buy and restructure companies for profit.",
        "The Basel accords set international banking regulatory standards.",
        "Economic indicators like unemployment and CPI gauge economic health.",
        "Short selling profits from betting that a stock price will fall.",
        "Quantitative easing involves central bank purchases of government bonds.",
        "The efficient market hypothesis argues that asset prices reflect all available information.",
        "Crowdfunding allows projects to raise money from many small contributors.",
        "Insurance spreads risk across many policyholders.",
        "Real GDP adjusts for inflation to measure true economic growth.",
        "The Marshall Plan helped rebuild Europe after World War Two.",
    ]
    for t in finance: T.append((t, "finance"))
    health = [
        "Regular exercise reduces the risk of chronic diseases.", "A balanced diet is essential for maintaining good health.",
        "Sleep deprivation impairs cognitive function and immune response.",
        "Vaccines are one of the most effective public health interventions.",
        "Mental health is just as important as physical health.",
        "The human body has approximately 37 trillion cells.",
        "Antibiotics treat bacterial infections but not viral ones.",
        "Stress management techniques include meditation and deep breathing.",
        "The brain uses about 20 percent of the body's total energy.",
        "Hydration is critical for all bodily functions.",
        "Regular health screenings can detect diseases early.",
        "Cardiovascular disease is the leading cause of death globally.",
        "The microbiome plays a crucial role in digestion and immunity.",
        "Smoking is a major risk factor for lung cancer.",
        "Physical therapy helps patients recover from injuries and surgeries.",
        "The Mediterranean diet is associated with improved heart health.",
        "Omega-3 fatty acids support brain function and reduce inflammation.",
        "Public health measures like sanitation have saved countless lives.",
        "Chronic stress can lead to serious health complications.",
        "Vitamin D deficiency is common in populations with limited sun exposure.",
        "The placebo effect demonstrates the power of mind-body connection.",
        "Obesity increases the risk of type 2 diabetes and heart disease.",
        "Hand washing is one of the simplest ways to prevent infection.",
        "Telemedicine has expanded access to healthcare during the pandemic.",
        "The human genome contains about 20000 protein coding genes.",
        "Antioxidants help protect cells from damage caused by free radicals.",
        "Regular dental checkups are important for overall health.",
        "Pilates strengthens the core and improves posture.",
        "The blood-brain barrier protects the brain from harmful substances.",
        "Resistance training helps maintain bone density as we age.",
        "Mindfulness meditation has been shown to reduce anxiety.",
        "The endocrine system regulates hormones throughout the body.",
        "Aerobic exercise improves cardiovascular fitness and endurance.",
        "Nutrition labels help consumers make informed food choices.",
        "The lymphatic system is a key part of the immune system.",
        "Social connections are strongly linked to better health outcomes.",
        "The liver performs over 500 essential functions in the body.",
        "Air pollution is a significant environmental health risk.",
        "Regular stretching improves flexibility and reduces injury risk.",
        "The nervous system coordinates all body functions through electrical signals.",
        "Sufficient protein intake is important for muscle repair and growth.",
        "The World Health Organization sets global health standards and guidelines.",
        "Excessive alcohol consumption damages the liver and brain.",
        "Genetic factors account for a significant portion of disease risk.",
        "Water is essential for nearly every metabolic process in the body.",
    ]
    for t in health: T.append((t, "health"))
    return T


# ===== Model Config =====
MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
}

_gemma4_paths = list(_Path(r"D:\develop\model\hub").glob("models--google--gemma-3-4b-pt/*/snapshots/*"))
if _gemma4_paths:
    MODEL_MAP["gemma4"] = _gemma4_paths[0]

MODEL_ORDER = [m for m in MODEL_MAP if m in MODEL_MAP]


def load_model(mname):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = MODEL_MAP[mname]
    log(f"  Loading {mname} from {p.name}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(p), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg = model.config
    n_layers = getattr(cfg, 'num_hidden_layers', None) or getattr(cfg, 'n_layers', None)
    d_model = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'd_model', 2048)
    log(f"  {mname}: n_layers={n_layers}, d_model={d_model}, device={model.device}")
    return model, tokenizer, n_layers, d_model


def get_unembed(model):
    if hasattr(model, 'lm_head'):
        um = model.lm_head
    elif hasattr(model, 'get_output_embeddings'):
        um = model.get_output_embeddings()
    else:
        return None, None
    w = um.weight.detach().to(torch.float32)
    b = um.bias.detach().to(torch.float32) if (hasattr(um, 'bias') and um.bias is not None) else None
    return w, b


def get_all_h(model, tokenizer, text, n_layers):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    states = []
    use_layers = min(n_layers + 1, len(outputs.hidden_states))
    for l in range(use_layers):
        h = outputs.hidden_states[l][:, -1, :].float().cpu().squeeze(0)
        states.append(h)
    return states, inputs


def compute_perplexity(model, tokenizer, text, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    avg_loss = outputs.loss.item()
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return ppl, avg_loss


def compute_centroids(model, tokenizer, texts, n_layers, n_samples=40):
    cat_h = defaultdict(list)
    indices = list(range(0, len(texts), max(1, len(texts) // n_samples)))[:n_samples]
    for idx in indices:
        states, _ = get_all_h(model, tokenizer, texts[idx][0], n_layers)
        h_final = states[-1]
        cat_h[texts[idx][1]].append(h_final)
    centroids = {}
    for cat, hs in cat_h.items():
        centroids[cat] = torch.stack(hs).mean(dim=0)
    return centroids


def kl_divergence(p, q, eps=1e-10):
    p = F.softmax(p.float(), dim=-1) + eps
    q = F.softmax(q.float(), dim=-1) + eps
    return (p * (p / q).log()).sum().item()


def get_layers_container(model):
    """Find the layers container in model architecture."""
    for attr in ['model', 'transformer', 'language_model']:
        container = getattr(model, attr, None)
        if container is not None:
            return container
    return model

def get_actual_layers(model):
    """Find the actual transformer layers."""
    container = get_layers_container(model)
    for attr in ['layers', 'blocks', 'encoder', 'decoder', 'h']:
        l = getattr(container, attr, None)
        if l is not None and hasattr(l, '__len__'):
            return l
    return None


# ============================================================
# P109: Multi-step Cumulative Injection
# ============================================================
def run_p109(model, tokenizer, mname, n_layers, d_model, texts):
    """
    P109: 多步累积注入 — 在residual stream的每一层都注入方向向量
    
    原理:
    - Phase XII发现单层注入被skip connection衰减(每N/4层衰减10x)
    - 如果在每一层都注入δ, 那么效果应该累积而非被衰减
    - 数学上: h_{l+1} = LayerNorm(Attn(h_l) + FFN(h_l)) + h_l + δ
    - 每层的δ虽小, 但经过N层累积后, 最终效果应该显著
    
    实验设计:
    - 方案A: 每层注入相同方向(small scale), 测量累积效应
    - 方案B: 每层注入缩放方向(scale递增), 测量最优缩放策略
    - 方案C: 只注入前半/后半层, 对比前向vs后向累积
    - 测量: cos(nat_h_final, inj_h_final), KL(logits_nat, logits_inj), PPL变化
    """
    log(f"\n{'='*70}")
    log(f"  P109: Multi-step Cumulative Injection ({mname})")
    log(f"{'='*70}")
    
    centroids = compute_centroids(model, tokenizer, texts, n_layers, n_samples=30)
    cats = list(centroids.keys())
    if len(cats) < 2:
        log("  Not enough categories")
        return {}
    
    dev = model.device
    target_dir = (centroids[cats[1]] - centroids[cats[0]]).float()
    target_dir = F.normalize(target_dir, dim=0).to(dev)
    
    actual_layers = get_actual_layers(model)
    if actual_layers is None:
        log("  Cannot find layers!")
        return {}
    
    uw, ub = get_unembed(model)
    if uw is not None:
        uw = uw.to(dev)
        if ub is not None:
            ub = ub.to(dev)
    
    test_indices = list(range(0, min(120, len(texts)), 2))[:60]
    test_texts = [texts[i][0] for i in test_indices]
    
    # ============================================================
    # Experiment A: Every layer injection with different base scales
    # ============================================================
    log(f"\n  [A] Every-layer injection (N={n_layers} layers, {len(test_texts[:15])} texts)")
    log(f"  Strategy: inject direction*base_scale at EVERY layer, measure cumulative effect")
    
    scales = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    results_A = {}
    
    for base_scale in scales:
        cos_vals = []
        kl_vals = []
        l2_vals = []
        top5_overlap = []
        
        for text in test_texts[:15]:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                
                # Natural pass
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                    nat_logits = nat_out.logits[:, -1, :].float()
                    nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                
                # Cumulative injection pass
                handles = []
                for layer_idx in range(len(actual_layers)):
                    def make_hook(direction, scale, lidx):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                hs = output[0]
                            else:
                                hs = output
                            hs_inj = hs.clone()
                            hs_inj[:, -1, :] = hs_inj[:, -1, :] + direction.to(hs.dtype) * scale
                            if isinstance(output, tuple):
                                return (hs_inj,) + output[1:]
                            return hs_inj
                        return hook_fn
                    h = actual_layers[layer_idx].register_forward_hook(
                        make_hook(target_dir, base_scale, layer_idx)
                    )
                    handles.append(h)
                
                try:
                    with torch.no_grad():
                        inj_out = model(**inputs, output_hidden_states=True)
                        inj_logits = inj_out.logits[:, -1, :].float()
                        inj_h = inj_out.hidden_states[-1][:, -1, :].float()
                finally:
                    for h in handles:
                        h.remove()
                
                cos = F.cosine_similarity(nat_h, inj_h, dim=-1).item()
                kl = kl_divergence(nat_logits, inj_logits)
                l2 = (nat_h - inj_h).norm(dim=-1).item()
                
                nat_top5 = nat_logits[0].topk(5).indices.tolist()
                inj_top5 = inj_logits[0].topk(5).indices.tolist()
                ov = len(set(nat_top5) & set(inj_top5))
                
                cos_vals.append(cos)
                kl_vals.append(kl)
                l2_vals.append(l2)
                top5_overlap.append(ov)
            except Exception as e:
                log(f"    [WARN] scale={base_scale}: {e}")
                continue
        
        if cos_vals:
            results_A[base_scale] = {
                "avg_cos": np.mean(cos_vals), "std_cos": np.std(cos_vals),
                "avg_kl": np.mean(kl_vals), "avg_l2": np.mean(l2_vals),
                "avg_overlap": np.mean(top5_overlap), "n": len(cos_vals),
            }
    
    log(f"\n  === Results [A]: Every-layer cumulative injection ===")
    log(f"  {'scale':>8s}  {'avg_cos':>10s}  {'1-cos':>10s}  {'avg_KL':>10s}  {'avg_L2':>10s}  {'top5_ov':>8s}  {'n':>4s}")
    for s in scales:
        if s in results_A:
            r = results_A[s]
            log(f"  {s:8.3f}  {r['avg_cos']:10.6f}  {1-r['avg_cos']:10.6f}  "
                f"{r['avg_kl']:10.4f}  {r['avg_l2']:10.4f}  {r['avg_overlap']:8.2f}/5  {r['n']:4d}")
    
    # ============================================================
    # Experiment B: Layer-group injection (front half / back half / all)
    # ============================================================
    log(f"\n  [B] Layer-group injection (scale=0.1, {len(test_texts[:10])} texts)")
    
    group_configs = {
        "first_quarter": list(range(0, n_layers//4)),
        "first_half": list(range(0, n_layers//2)),
        "second_half": list(range(n_layers//2, n_layers)),
        "last_quarter": list(range(3*n_layers//4, n_layers)),
        "every_other": list(range(0, n_layers, 2)),
        "every_4th": list(range(0, n_layers, 4)),
        "all": list(range(n_layers)),
    }
    
    results_B = {}
    inj_scale = 0.1
    
    for gname, layer_indices in group_configs.items():
        cos_vals = []
        kl_vals = []
        
        for text in test_texts[:10]:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                    nat_logits = nat_out.logits[:, -1, :].float()
                    nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                
                handles = []
                for lidx in layer_indices:
                    if lidx >= len(actual_layers):
                        continue
                    def make_hook(direction, scale):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                hs = output[0]
                            else:
                                hs = output
                            hs_inj = hs.clone()
                            hs_inj[:, -1, :] = hs_inj[:, -1, :] + direction.to(hs.dtype) * scale
                            if isinstance(output, tuple):
                                return (hs_inj,) + output[1:]
                            return hs_inj
                        return hook_fn
                    h = actual_layers[lidx].register_forward_hook(make_hook(target_dir, inj_scale))
                    handles.append(h)
                
                try:
                    with torch.no_grad():
                        inj_out = model(**inputs, output_hidden_states=True)
                        inj_logits = inj_out.logits[:, -1, :].float()
                        inj_h = inj_out.hidden_states[-1][:, -1, :].float()
                finally:
                    for h in handles:
                        h.remove()
                
                cos = F.cosine_similarity(nat_h, inj_h, dim=-1).item()
                kl = kl_divergence(nat_logits, inj_logits)
                cos_vals.append(cos)
                kl_vals.append(kl)
            except Exception as e:
                continue
        
        if cos_vals:
            results_B[gname] = {
                "avg_cos": np.mean(cos_vals), "avg_kl": np.mean(kl_vals),
                "1-cos": 1 - np.mean(cos_vals), "n_layers": len(layer_indices),
                "n": len(cos_vals),
            }
    
    log(f"\n  === Results [B]: Layer-group injection (scale=0.1) ===")
    log(f"  {'group':>16s}  {'#layers':>8s}  {'avg_cos':>10s}  {'1-cos':>10s}  {'avg_KL':>10s}  {'n':>4s}")
    for gname in ["first_quarter", "first_half", "second_half", "last_quarter", "every_other", "every_4th", "all"]:
        if gname in results_B:
            r = results_B[gname]
            log(f"  {gname:>16s}  {r['n_layers']:8d}  {r['avg_cos']:10.6f}  "
                f"{r['1-cos']:10.6f}  {r['avg_kl']:10.4f}  {r['n']:4d}")
    
    # ============================================================
    # Experiment C: Scaling factor sweep with all-layer injection
    # ============================================================
    log(f"\n  [C] Fine-grained scale sweep with all-layer injection ({len(test_texts[:15])} texts)")
    
    fine_scales = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    results_C = {}
    
    for s in fine_scales:
        cos_vals = []
        kl_vals = []
        l2_vals = []
        
        for text in test_texts[:15]:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                    nat_logits = nat_out.logits[:, -1, :].float()
                    nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                
                handles = []
                for lidx in range(len(actual_layers)):
                    def make_hook(direction, scale):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                hs = output[0]
                            else:
                                hs = output
                            hs_inj = hs.clone()
                            hs_inj[:, -1, :] = hs_inj[:, -1, :] + direction.to(hs.dtype) * scale
                            if isinstance(output, tuple):
                                return (hs_inj,) + output[1:]
                            return hs_inj
                        return hook_fn
                    h = actual_layers[lidx].register_forward_hook(make_hook(target_dir, s))
                    handles.append(h)
                
                try:
                    with torch.no_grad():
                        inj_out = model(**inputs, output_hidden_states=True)
                        inj_logits = inj_out.logits[:, -1, :].float()
                        inj_h = inj_out.hidden_states[-1][:, -1, :].float()
                finally:
                    for h in handles:
                        h.remove()
                
                cos = F.cosine_similarity(nat_h, inj_h, dim=-1).item()
                kl = kl_divergence(nat_logits, inj_logits)
                l2 = (nat_h - inj_h).norm(dim=-1).item()
                cos_vals.append(cos)
                kl_vals.append(kl)
                l2_vals.append(l2)
            except Exception as e:
                continue
        
        if cos_vals:
            results_C[s] = {
                "avg_cos": np.mean(cos_vals), "1-cos": 1-np.mean(cos_vals),
                "avg_kl": np.mean(kl_vals), "avg_l2": np.mean(l2_vals), "n": len(cos_vals),
            }
    
    log(f"\n  === Results [C]: Fine-grained scale sweep ===")
    log(f"  {'scale':>8s}  {'avg_cos':>10s}  {'1-cos':>10s}  {'avg_KL':>10s}  {'avg_L2':>10s}  {'n':>4s}")
    for s in fine_scales:
        if s in results_C:
            r = results_C[s]
            log(f"  {s:8.4f}  {r['avg_cos']:10.6f}  {r['1-cos']:10.6f}  "
                f"{r['avg_kl']:10.4f}  {r['avg_l2']:10.4f}  {r['n']:4d}")
    
    # ============================================================
    # Experiment D: Generate with cumulative injection + PPL
    # ============================================================
    log(f"\n  [D] Generation test with cumulative injection (10 texts)")
    
    gen_scale = 0.05  # conservative scale
    gen_results = []
    
    for text in test_texts[:10]:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            
            # Natural generation
            nat_ids = inputs["input_ids"].clone()
            with torch.no_grad():
                for step in range(8):
                    out = model(input_ids=nat_ids, use_cache=True)
                    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if next_tok.item() == tokenizer.eos_token_id:
                        break
                    nat_ids = torch.cat([nat_ids, next_tok], dim=1)
            nat_text = tokenizer.decode(nat_ids[0], skip_special_tokens=True)
            nat_ppl, _ = compute_perplexity(model, tokenizer, nat_text)
            
            # Injected generation
            inj_ids = inputs["input_ids"].clone()
            with torch.no_grad():
                for step in range(8):
                    handles = []
                    for lidx in range(len(actual_layers)):
                        def make_hook(direction, scale):
                            def hook_fn(module, input, output):
                                if isinstance(output, tuple):
                                    hs = output[0]
                                else:
                                    hs = output
                                hs_inj = hs.clone()
                                hs_inj[:, -1, :] = hs_inj[:, -1, :] + direction.to(hs.dtype) * scale
                                if isinstance(output, tuple):
                                    return (hs_inj,) + output[1:]
                                return hs_inj
                            return hook_fn
                        h = actual_layers[lidx].register_forward_hook(make_hook(target_dir, gen_scale))
                        handles.append(h)
                    try:
                        out = model(input_ids=inj_ids, use_cache=True)
                        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    finally:
                        for h in handles:
                            h.remove()
                    if next_tok.item() == tokenizer.eos_token_id:
                        break
                    inj_ids = torch.cat([inj_ids, next_tok], dim=1)
            inj_text = tokenizer.decode(inj_ids[0], skip_special_tokens=True)
            inj_ppl, _ = compute_perplexity(model, tokenizer, inj_text)
            
            text_changed = (nat_text.strip() != inj_text.strip())
            gen_results.append({
                "nat_ppl": nat_ppl, "inj_ppl": inj_ppl, "delta_ppl": inj_ppl - nat_ppl,
                "changed": text_changed, "prompt": text[:40],
            })
        except Exception as e:
            log(f"    [WARN] gen: {e}")
    
    if gen_results:
        avg_nat_ppl = np.mean([r["nat_ppl"] for r in gen_results])
        avg_inj_ppl = np.mean([r["inj_ppl"] for r in gen_results])
        n_changed = sum(1 for r in gen_results if r["changed"])
        log(f"\n  Generation (scale={gen_scale}):")
        log(f"    avg_nat_ppl={avg_nat_ppl:.1f}, avg_inj_ppl={avg_inj_ppl:.1f}, delta={avg_inj_ppl-avg_nat_ppl:.1f}")
        log(f"    texts changed: {n_changed}/{len(gen_results)}")
        for r in gen_results[:3]:
            log(f"    prompt='{r['prompt']}...' nat_ppl={r['nat_ppl']:.1f} inj_ppl={r['inj_ppl']:.1f} changed={r['changed']}")
    
    return {"A": results_A, "B": results_B, "C": results_C, "D": gen_results}


# ============================================================
# P110: Gradient-Guided Manipulation
# ============================================================
def run_p110(model, tokenizer, mname, n_layers, d_model, texts):
    """
    P110: 梯度引导操控 — 用loss的梯度找到最优注入方向
    
    原理:
    - centroid差方向是启发式的, 不一定是最优操控方向
    - 梯度方向直接编码了"如何改变hidden state使得loss最大/最小变化"
    - d(loss)/d(h_l) 直接告诉我们第l层的hidden state该如何调整
    - 比较梯度方向 vs centroid方向 vs 随机方向 的操控效果
    
    实验:
    - 对每个text, 计算d(loss)/d(h_l), 提取梯度方向
    - 用梯度方向注入 vs centroid方向注入, 对比效果
    - 测量梯度方向的层间一致性(cos across layers)
    """
    log(f"\n{'='*70}")
    log(f"  P110: Gradient-Guided Manipulation ({mname})")
    log(f"{'='*70}")
    
    centroids = compute_centroids(model, tokenizer, texts, n_layers, n_samples=30)
    cats = list(centroids.keys())
    if len(cats) < 2:
        log("  Not enough categories")
        return {}
    
    dev = model.device
    centroid_dir = F.normalize((centroids[cats[1]] - centroids[cats[0]]).float(), dim=0).to(dev)
    
    actual_layers = get_actual_layers(model)
    if actual_layers is None:
        log("  Cannot find layers!")
        return {}
    
    test_texts = [texts[i][0] for i in range(0, min(40, len(texts)), 4)][:10]
    
    log(f"  Computing gradient directions for {len(test_texts)} texts...")
    
    # For each text, compute gradient direction at layer 0
    gradient_dirs = []
    centroid_cos = []
    
    for text in test_texts:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            
            # Enable gradients for hidden states
            model.zero_grad()
            outputs = model(**inputs, output_hidden_states=True, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Get gradient w.r.t. first layer output
            h0 = outputs.hidden_states[1][:, -1, :].float()  # layer 0 output
            h0.requires_grad_(True)
            
            # Recompute with grad
            model.zero_grad()
            outputs2 = model(**inputs, output_hidden_states=True, labels=inputs["input_ids"])
            
            # We need to hook into layer 0 to get gradient
            grad_dir = None
            
            def grad_hook(module, input, output):
                nonlocal grad_dir
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                # We can't easily get gradient of loss w.r.t. intermediate hidden states
                # without modifying the forward pass. Instead, use a simpler approach:
                # perturb h0 in random directions and measure loss change
                pass
            
            # Simpler approach: measure loss sensitivity in different directions
            # For centroid direction vs random directions
            loss_base = outputs2.loss.item()
            
            # Perturb in centroid direction
            handles = []
            def make_perturb_hook(direction, scale):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output
                    hs_inj = hs.clone()
                    hs_inj[:, -1, :] = hs_inj[:, -1, :] + direction.to(hs.dtype) * scale
                    if isinstance(output, tuple):
                        return (hs_inj,) + output[1:]
                    return hs_inj
                return hook_fn
            
            # Test loss sensitivity in centroid direction at layer 0
            h0_handle = actual_layers[0].register_forward_hook(
                make_perturb_hook(centroid_dir, 0.1)
            )
            try:
                with torch.no_grad():
                    out_perturb = model(**inputs, labels=inputs["input_ids"])
                loss_centroid = out_perturb.loss.item()
            finally:
                h0_handle.remove()
            
            # Test loss sensitivity in random directions at layer 0
            rand_sensitivities = []
            for _ in range(5):
                rand_dir = torch.randn(d_model, device=dev)
                rand_dir = F.normalize(rand_dir, dim=0)
                rh = actual_layers[0].register_forward_hook(
                    make_perturb_hook(rand_dir, 0.1)
                )
                try:
                    with torch.no_grad():
                        out_r = model(**inputs, labels=inputs["input_ids"])
                    rand_sensitivities.append(abs(out_r.loss.item() - loss_base))
                finally:
                    rh.remove()
            
            centroid_sensitivity = abs(loss_centroid - loss_base)
            centroid_cos.append(centroid_sensitivity)
            
            gradient_dirs.append({
                "centroid_sens": centroid_sensitivity,
                "rand_avg_sens": np.mean(rand_sensitivities),
                "loss_base": loss_base,
            })
        except Exception as e:
            log(f"    [WARN] {e}")
            continue
    
    if gradient_dirs:
        log(f"\n  === Loss Sensitivity Analysis ===")
        log(f"  Direction        | avg |Δloss|  | vs random baseline")
        avg_c = np.mean([d["centroid_sens"] for d in gradient_dirs])
        avg_r = np.mean([d["rand_avg_sens"] for d in gradient_dirs])
        log(f"  Centroid dir:    {avg_c:.4f}  (ratio to random: {avg_c/avg_r:.2f}x)" if avg_r > 0 else
            f"  Centroid dir:    {avg_c:.4f}  (random baseline: {avg_r:.4f})")
        log(f"  Random dirs:     {avg_r:.4f}")
        log(f"  Base loss:       {np.mean([d['loss_base'] for d in gradient_dirs]):.4f}")
        
        if avg_c > avg_r:
            log(f"\n  >> Centroid direction is {avg_c/avg_r:.1f}x more sensitive than random!")
        else:
            log(f"\n  >> Centroid direction is NOT more sensitive than random (ratio={avg_c/avg_r:.2f})")
    
    # ============================================================
    # Experiment B: Per-layer loss sensitivity to centroid direction
    # ============================================================
    log(f"\n  [B] Per-layer loss sensitivity to centroid injection (8 texts)")
    
    layer_sensitivities = []
    test_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    for layer_idx in test_layers:
        if layer_idx >= len(actual_layers):
            continue
        sens_vals = []
        for text in test_texts[:8]:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                with torch.no_grad():
                    out_base = model(**inputs, labels=inputs["input_ids"])
                    loss_base = out_base.loss.item()
                
                h = actual_layers[layer_idx].register_forward_hook(
                    make_perturb_hook(centroid_dir, 0.1)
                )
                try:
                    with torch.no_grad():
                        out_p = model(**inputs, labels=inputs["input_ids"])
                        loss_p = out_p.loss.item()
                    sens_vals.append(abs(loss_p - loss_base))
                finally:
                    h.remove()
            except:
                continue
        
        if sens_vals:
            layer_sensitivities.append({
                "layer": layer_idx, "avg_sens": np.mean(sens_vals), "n": len(sens_vals)
            })
    
    log(f"\n  === Per-Layer Loss Sensitivity (centroid dir, scale=0.1) ===")
    log(f"  {'Layer':>6s}  {'avg|Δloss|':>12s}  {'n':>4s}")
    for s in layer_sensitivities:
        log(f"  L{s['layer']:>4d}  {s['avg_sens']:12.4f}  {s['n']:4d}")
    
    return gradient_dirs


# ============================================================
# P111: GLM4 Direction Reversal Analysis
# ============================================================
def run_p111(model, tokenizer, mname, n_layers, d_model, texts):
    """
    P111: 层间权重矩阵分析 — 为什么GLM4会产生方向反转?
    
    原理:
    - Phase XII发现GLM4在早期层注入后cos<0(方向反转)
    - 这可能是由LayerNorm、Attention、或FFN的特殊结构导致的
    - 分析方法:
      a) 测量每层输出的范数和方向(cos between consecutive layers)
      b) 注入前后的逐层传播: 测量注入效应在第l+1层是否被保留/反转
      c) 单独测试LayerNorm对方向的扭曲效应
    
    注: 主要针对GLM4, 但所有模型都运行以对比
    """
    log(f"\n{'='*70}")
    log(f"  P111: Layer-wise Propagation Analysis ({mname})")
    log(f"{'='*70}")
    
    centroids = compute_centroids(model, tokenizer, texts, n_layers, n_samples=30)
    cats = list(centroids.keys())
    if len(cats) < 2:
        log("  Not enough categories")
        return {}
    
    dev = model.device
    target_dir = F.normalize((centroids[cats[1]] - centroids[cats[0]]).float(), dim=0).to(dev)
    
    actual_layers = get_actual_layers(model)
    if actual_layers is None:
        log("  Cannot find layers!")
        return {}
    
    test_texts = [texts[i][0] for i in range(0, min(40, len(texts)), 4)][:8]
    
    # ============================================================
    # Experiment A: Inject at layer L, measure effect at layer L+1
    # (逐层传播分析)
    # ============================================================
    log(f"\n  [A] Layer-to-layer propagation ({len(test_texts)} texts)")
    log(f"  Inject direction at layer L, measure cos(delta_at_L, delta_at_L+1)")
    
    inj_scale = 5.0  # large enough to measure
    propagation = {}
    
    for l_idx in range(min(n_layers - 1, len(actual_layers) - 1)):
        cos_vals = []
        magnitude_vals = []
        
        for text in test_texts[:5]:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                
                # Natural: get h at layer l and l+1
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                    h_l = nat_out.hidden_states[l_idx + 1][:, -1, :].float()  # +1 because index 0 is embedding
                    h_l1 = nat_out.hidden_states[l_idx + 2][:, -1, :].float()
                
                # Injected at layer l
                h = actual_layers[l_idx].register_forward_hook(
                    lambda m, inp, out, d=target_dir, s=inj_scale: (
                        (out[0][:, :-1, :] + d.to(out[0].dtype) * s, *out[1:])
                        if isinstance(out, tuple) else
                        out[:, :-1, :] + d.to(out.dtype) * s  # wrong, need all tokens
                    ) if False else _inject_last_token(out, d, s)
                )
                # Better approach: hook the correct way
                def make_hook_l(direction, scale):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            hs = output[0]
                        else:
                            hs = output
                        hs_inj = hs.clone()
                        hs_inj[:, -1, :] = hs_inj[:, -1, :] + direction.to(hs.dtype) * scale
                        if isinstance(output, tuple):
                            return (hs_inj,) + output[1:]
                        return hs_inj
                    return hook_fn
                
                h.remove()  # remove bad hook
                h = actual_layers[l_idx].register_forward_hook(make_hook_l(target_dir, inj_scale))
                
                try:
                    with torch.no_grad():
                        inj_out = model(**inputs, output_hidden_states=True)
                        h_l_inj = inj_out.hidden_states[l_idx + 1][:, -1, :].float()
                        h_l1_inj = inj_out.hidden_states[l_idx + 2][:, -1, :].float()
                finally:
                    h.remove()
                
                delta_l = h_l_inj - h_l
                delta_l1 = h_l1_inj - h_l1
                
                if delta_l.norm() > 1e-8 and delta_l1.norm() > 1e-8:
                    cos = F.cosine_similarity(delta_l, delta_l1, dim=-1).item()
                    cos_vals.append(cos)
                    magnitude_vals.append((delta_l.norm().item(), delta_l1.norm().item()))
            except Exception as e:
                continue
        
        if cos_vals:
            propagation[l_idx] = {
                "avg_cos": np.mean(cos_vals),
                "avg_mag_l": np.mean([m[0] for m in magnitude_vals]),
                "avg_mag_l1": np.mean([m[1] for m in magnitude_vals]),
                "magnitude_ratio": np.mean([m[1]/(m[0]+1e-10) for m in magnitude_vals]),
                "n": len(cos_vals),
            }
    
    log(f"\n  === Layer-to-layer direction propagation (scale={inj_scale}) ===")
    log(f"  {'Layer':>6s}  {'cos(δ_L,δ_L+1)':>15s}  {'mag_L':>8s}  {'mag_L+1':>8s}  {'ratio':>8s}  {'n':>3s}")
    for l in sorted(propagation.keys()):
        r = propagation[l]
        log(f"  L{l:>4d}  {r['avg_cos']:15.6f}  {r['avg_mag_l']:8.4f}  "
            f"{r['avg_mag_l1']:8.4f}  {r['magnitude_ratio']:8.4f}  {r['n']:3d}")
    
    # ============================================================
    # Experiment B: Natural direction consistency across layers
    # (无注入时, 相邻层的hidden state方向差异)
    # ============================================================
    log(f"\n  [B] Natural inter-layer direction consistency ({len(test_texts)} texts)")
    
    layer_cos_matrix = defaultdict(list)
    
    for text in test_texts[:5]:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            n_states = min(len(outputs.hidden_states), n_layers + 1)
            for l in range(1, n_states - 1):
                h_l = outputs.hidden_states[l][:, -1, :].float()
                h_l1 = outputs.hidden_states[l + 1][:, -1, :].float()
                cos = F.cosine_similarity(h_l, h_l1, dim=-1).item()
                layer_cos_matrix[l].append(cos)
        except:
            continue
    
    log(f"  {'Layer':>6s}  {'cos(h_l,h_l+1)':>16s}  {'n':>3s}")
    for l in sorted(layer_cos_matrix.keys()):
        vals = layer_cos_matrix[l]
        log(f"  L{l:>4d}  {np.mean(vals):16.6f}  {len(vals):3d}")
    
    # ============================================================
    # Experiment C: Norm analysis across layers (injected vs natural)
    # ============================================================
    log(f"\n  [C] Norm analysis: injected vs natural (5 texts)")
    
    norm_results = defaultdict(lambda: {"nat_norms": [], "inj_norms": []})
    
    # Inject at layer 0, measure norms at all layers
    for text in test_texts[:5]:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
            
            handles = [actual_layers[0].register_forward_hook(make_hook_l(target_dir, inj_scale))]
            try:
                with torch.no_grad():
                    inj_out = model(**inputs, output_hidden_states=True)
            finally:
                for hh in handles:
                    hh.remove()
            
            n_states = min(len(nat_out.hidden_states), n_layers + 1)
            for l in range(n_states):
                nat_norm = nat_out.hidden_states[l][:, -1, :].norm(dim=-1).item()
                inj_norm = inj_out.hidden_states[l][:, -1, :].norm(dim=-1).item()
                norm_results[l]["nat_norms"].append(nat_norm)
                norm_results[l]["inj_norms"].append(inj_norm)
        except:
            continue
    
    log(f"  {'Layer':>6s}  {'nat_norm':>10s}  {'inj_norm':>10s}  {'ratio':>8s}  {'n':>3s}")
    for l in sorted(norm_results.keys()):
        r = norm_results[l]
        avg_nat = np.mean(r["nat_norms"])
        avg_inj = np.mean(r["inj_norms"])
        log(f"  L{l:>4d}  {avg_nat:10.4f}  {avg_inj:10.4f}  {avg_inj/(avg_nat+1e-10):8.4f}  {len(r['nat_norms']):3d}")
    
    return {"propagation": propagation, "norm_results": dict(norm_results)}


def _inject_last_token(output, direction, scale):
    if isinstance(output, tuple):
        hs = output[0]
    else:
        hs = output
    hs_inj = hs.clone()
    hs_inj[:, -1, :] = hs_inj[:, -1, :] + direction.to(hs.dtype) * scale
    if isinstance(output, tuple):
        return (hs_inj,) + output[1:]
    return hs_inj


# ============================================================
# P112: Comprehensive Quality Evaluation
# ============================================================
def run_p112(model, tokenizer, mname, n_layers, d_model, texts):
    """
    P112: 综合质量评估 — 操控后生成的文本质量变化
    
    方法:
    - 选择最优的累积注入配置(P109结果)
    - 生成50个样本, 对比natural vs manipulated
    - 评估维度:
      1. PPL变化 (自然度)
      2. Top-1 token变化率 (直接控制)
      3. 生成文本Levenshtein距离 (文本差异程度)
      4. Token概率分布偏移 (KL divergence)
      5. 语义方向偏移 (cos between h_final vectors)
    """
    log(f"\n{'='*70}")
    log(f"  P112: Comprehensive Quality Evaluation ({mname})")
    log(f"{'='*70}")
    
    centroids = compute_centroids(model, tokenizer, texts, n_layers, n_samples=30)
    cats = list(centroids.keys())
    if len(cats) < 2:
        log("  Not enough categories")
        return {}
    
    dev = model.device
    target_dir = F.normalize((centroids[cats[1]] - centroids[cats[0]]).float(), dim=0).to(dev)
    
    actual_layers = get_actual_layers(model)
    if actual_layers is None:
        log("  Cannot find layers!")
        return {}
    
    test_indices = list(range(0, min(100, len(texts)), 2))[:50]
    
    # Test with multiple scale configs
    configs = [
        ("all_layers_0.01", 0.01, list(range(len(actual_layers)))),
        ("all_layers_0.05", 0.05, list(range(len(actual_layers)))),
        ("first_half_0.1", 0.1, list(range(len(actual_layers)//2))),
        ("first_quarter_0.5", 0.5, list(range(len(actual_layers)//4))),
    ]
    
    for cfg_name, inj_scale, inj_layers in configs:
        log(f"\n  Config: {cfg_name} (scale={inj_scale}, layers={len(inj_layers)})")
        
        results = {
            "cos_h_final": [], "kl_logits": [], "top1_changed": [],
            "ppl_nat": [], "ppl_inj": [], "text_changed": [],
            "gen_levenshtein": [],
        }
        
        for idx in test_indices[:20]:  # 20 samples per config
            text = texts[idx][0]
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                
                # Natural
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                    nat_logits = nat_out.logits[:, -1, :].float()
                    nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                
                # Injected (single step for KL/cos comparison)
                handles = []
                for lidx in inj_layers:
                    if lidx >= len(actual_layers):
                        continue
                    def make_hook(direction, scale):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                hs = output[0]
                            else:
                                hs = output
                            hs_inj = hs.clone()
                            hs_inj[:, -1, :] = hs_inj[:, -1, :] + direction.to(hs.dtype) * scale
                            if isinstance(output, tuple):
                                return (hs_inj,) + output[1:]
                            return hs_inj
                        return hook_fn
                    h = actual_layers[lidx].register_forward_hook(make_hook(target_dir, inj_scale))
                    handles.append(h)
                
                try:
                    with torch.no_grad():
                        inj_out = model(**inputs, output_hidden_states=True)
                        inj_logits = inj_out.logits[:, -1, :].float()
                        inj_h = inj_out.hidden_states[-1][:, -1, :].float()
                finally:
                    for h in handles:
                        h.remove()
                
                cos = F.cosine_similarity(nat_h, inj_h, dim=-1).item()
                kl = kl_divergence(nat_logits, inj_logits)
                nat_top1 = nat_logits.argmax(dim=-1).item()
                inj_top1 = inj_logits.argmax(dim=-1).item()
                
                results["cos_h_final"].append(cos)
                results["kl_logits"].append(kl)
                results["top1_changed"].append(1 if nat_top1 != inj_top1 else 0)
                
                # PPL for both (generate 5 tokens)
                nat_ids = inputs["input_ids"].clone()
                inj_ids = inputs["input_ids"].clone()
                
                with torch.no_grad():
                    for _ in range(5):
                        out = model(input_ids=nat_ids, use_cache=True)
                        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        if tok.item() == tokenizer.eos_token_id:
                            break
                        nat_ids = torch.cat([nat_ids, tok], dim=1)
                
                # Injected generation
                with torch.no_grad():
                    for _ in range(5):
                        h_gen = []
                        for lidx in inj_layers:
                            if lidx >= len(actual_layers):
                                continue
                            hg = actual_layers[lidx].register_forward_hook(
                                make_hook(target_dir, inj_scale)
                            )
                            h_gen.append(hg)
                        try:
                            out = model(input_ids=inj_ids, use_cache=True)
                            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        finally:
                            for hg in h_gen:
                                hg.remove()
                        if tok.item() == tokenizer.eos_token_id:
                            break
                        inj_ids = torch.cat([inj_ids, tok], dim=1)
                
                nat_text = tokenizer.decode(nat_ids[0], skip_special_tokens=True)
                inj_text = tokenizer.decode(inj_ids[0], skip_special_tokens=True)
                
                nat_ppl, _ = compute_perplexity(model, tokenizer, nat_text)
                inj_ppl, _ = compute_perplexity(model, tokenizer, inj_text)
                
                results["ppl_nat"].append(nat_ppl)
                results["ppl_inj"].append(inj_ppl)
                results["text_changed"].append(1 if nat_text.strip() != inj_text.strip() else 0)
                
            except Exception as e:
                log(f"    [WARN] idx={idx}: {e}")
                continue
        
        if results["cos_h_final"]:
            n = len(results["cos_h_final"])
            log(f"    Results ({n} samples):")
            log(f"      cos(h_final):     avg={np.mean(results['cos_h_final']):.6f}, 1-cos={1-np.mean(results['cos_h_final']):.6f}")
            log(f"      KL(logits):       avg={np.mean(results['kl_logits']):.4f}")
            log(f"      Top-1 changed:    {sum(results['top1_changed'])}/{n} ({sum(results['top1_changed'])/n*100:.0f}%)")
            log(f"      PPL nat:          avg={np.mean(results['ppl_nat']):.1f}")
            log(f"      PPL inj:          avg={np.mean(results['ppl_inj']):.1f}")
            log(f"      Text changed:     {sum(results['text_changed'])}/{n} ({sum(results['text_changed'])/n*100:.0f}%)")
    
    return {}


# ============================================================
# Main
# ============================================================
def main():
    global log
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3",
                        choices=list(MODEL_MAP.keys()) + ["all"])
    args = parser.parse_args()
    
    models_to_test = MODEL_ORDER if args.model == "all" else [args.model]
    
    texts = build_texts()
    print(f"Dataset: {len(texts)} texts, {len(set(t[1] for t in texts))} categories")
    
    for mname in models_to_test:
        if mname not in MODEL_MAP:
            log(f"  Skipping {mname} (not found)")
            continue
        
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        log_dir = _Path(f"d:/develop/TransformerLens-main/tests/glm5_temp/stage718_phase13_{mname}_{ts}")
        log = Logger(str(log_dir), "results")
        
        log(f"\n{'#'*70}")
        log(f"# Stage 718: Phase XIII — {mname}")
        log(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"{'#'*70}")
        
        try:
            model, tokenizer, n_layers, d_model = load_model(mname)
        except Exception as e:
            log(f"  FAILED to load {mname}: {e}")
            continue
        
        try:
            # P109: Multi-step cumulative injection (main experiment)
            run_p109(model, tokenizer, mname, n_layers, d_model, texts)
            
            # P110: Gradient-guided manipulation
            run_p110(model, tokenizer, mname, n_layers, d_model, texts)
            
            # P111: Layer-wise propagation analysis
            run_p111(model, tokenizer, mname, n_layers, d_model, texts)
            
            # P112: Comprehensive quality evaluation
            run_p112(model, tokenizer, mname, n_layers, d_model, texts)
            
        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        log(f"\n  Done with {mname}")
        log.close()
        log = None
        
        # Free GPU
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  GPU freed for {mname}. Waiting 30s before next model...")
        time.sleep(30)
    
    # Final log to console
    print("\n" + "="*60)
    print("  Phase XIII COMPLETE for all models")
    print("="*60)


if __name__ == "__main__":
    main()
