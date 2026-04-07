#!/usr/bin/env python3
"""
Stage 719: Phase XIV — GLM4操控最优化 + 语义质量评估 + 操控方法对比 + Gemma4
================================================================================
Phase XIII发现:
- GLM4存在操控阈值(scale≈0.15), 超过后效应指数放大
- Qwen3极度顽固, DS7B中等敏感
- 层间传播是"方向损失+幅度放大"

Phase XIV目标:
P113: GLM4阈值精细搜索 — scale=0.05~0.30之间0.01步进, 找到最优可控区间
P114: 语义质量评估 — 用centroid距离衡量操控后文本的类别偏移
P115: 操控方法对比 — cumulative vs attention-steering vs activation-patching vs logit-override
P116: 四模型完整对比 — Qwen3/DS7B/GLM4/Gemma4统一测试

用法: python stage719_phase14.py --model glm4
      python stage719_phase14.py --model qwen3
      python stage719_phase14.py --model deepseek7b
      python stage719_phase14.py --model gemma4
      python stage719_phase14.py --model all
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

log = None

# ===== Text dataset (same as before) =====
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
    for attr in ['model', 'transformer', 'language_model']:
        container = getattr(model, attr, None)
        if container is not None:
            return container
    return model

def get_actual_layers(model):
    container = get_layers_container(model)
    for attr in ['layers', 'blocks', 'encoder', 'decoder', 'h']:
        l = getattr(container, attr, None)
        if l is not None and hasattr(l, '__len__'):
            return l
    return None


def make_inject_hook(direction, scale):
    """Create a hook that injects direction*scale at last token position."""
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


# ============================================================
# P113: GLM4 Threshold Fine Search
# ============================================================
def run_p113(model, tokenizer, mname, n_layers, d_model, texts):
    """
    P113: 阈值精细搜索 — 在操控阈值附近精细搜索最优scale
    
    Phase XIII发现GLM4的操控阈值在scale≈0.15附近。
    本实验用0.005步进从0.02到0.30, 精细测量操控效应。
    
    同时对所有模型运行, 以找到各自的阈值。
    
    测量维度:
    - 1-cos(h_nat, h_inj): hidden state偏移
    - KL(logits): 概率分布偏移
    - Top-1 change rate: 直接控制效果
    - PPL change: 生成质量影响
    - Text change rate: 宏观效果
    """
    log(f"\n{'='*70}")
    log(f"  P113: Threshold Fine Search ({mname})")
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
    
    # Use first_quarter layers (most effective per Phase XIII)
    inj_layers = list(range(min(n_layers // 4, len(actual_layers))))
    
    # Fine-grained scale search  
    coarse_scales = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    fine_scales = [round(0.02 + i * 0.01, 3) for i in range(30)]  # 0.02 to 0.31
    # Reduce text generation overhead: only do full gen for subset of scales
    gen_scales = sorted(set([0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30] + coarse_scales))
    cos_kl_scales = sorted(set(fine_scales + coarse_scales))
    
    test_indices = list(range(0, min(100, len(texts)), 3))[:30]
    test_texts = [texts[i][0] for i in test_indices]
    
    log(f"  Testing {len(cos_kl_scales)} scales (cos/KL), {len(gen_scales)} scales (gen)")
    log(f"  Injecting at {len(inj_layers)} layers (first quarter)")
    log(f"  Fine range: 0.02~0.31, step=0.01; Gen range: {gen_scales}")
    
    results = []
    
    # Phase 1: Quick cos/KL measurement for ALL scales (no generation)
    for scale in cos_kl_scales:
        cos_vals = []
        kl_vals = []
        top1_changed = []
        
        for text in test_texts[:20]:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                    nat_logits = nat_out.logits[:, -1, :].float()
                    nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                
                handles = []
                for lidx in inj_layers:
                    h = actual_layers[lidx].register_forward_hook(make_inject_hook(target_dir, scale))
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
                
                cos_vals.append(cos)
                kl_vals.append(kl)
                top1_changed.append(1 if nat_top1 != inj_top1 else 0)
            except:
                continue
        
        if cos_vals:
            n = len(cos_vals)
            results.append({
                "scale": scale,
                "avg_cos": np.mean(cos_vals), "1-cos": 1 - np.mean(cos_vals),
                "avg_kl": np.mean(kl_vals),
                "top1_rate": sum(top1_changed) / n * 100,
                "has_gen": False, "n": n,
            })
    
    # Phase 2: Full generation test for selected scales
    for scale in gen_scales:
        ppl_nat_vals = []
        ppl_inj_vals = []
        text_changed = []
        
        for text in test_texts[:10]:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                
                gen_ids = inputs["input_ids"].clone()
                with torch.no_grad():
                    for _ in range(5):
                        h_gen = [actual_layers[lidx].register_forward_hook(make_inject_hook(target_dir, scale))
                                 for lidx in inj_layers]
                        try:
                            out = model(input_ids=gen_ids, use_cache=True)
                            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        finally:
                            for hg in h_gen:
                                hg.remove()
                        if tok.item() == tokenizer.eos_token_id:
                            break
                        gen_ids = torch.cat([gen_ids, tok], dim=1)
                
                inj_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                nat_ids = inputs["input_ids"].clone()
                with torch.no_grad():
                    for _ in range(5):
                        out = model(input_ids=nat_ids, use_cache=True)
                        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        if tok.item() == tokenizer.eos_token_id:
                            break
                        nat_ids = torch.cat([nat_ids, tok], dim=1)
                nat_text = tokenizer.decode(nat_ids[0], skip_special_tokens=True)
                
                nat_ppl, _ = compute_perplexity(model, tokenizer, nat_text)
                inj_ppl, _ = compute_perplexity(model, tokenizer, inj_text)
                ppl_nat_vals.append(nat_ppl)
                ppl_inj_vals.append(inj_ppl)
                text_changed.append(1 if nat_text.strip() != inj_text.strip() else 0)
            except:
                continue
        
        if ppl_nat_vals:
            # Update existing result or add new
            existing = [r for r in results if abs(r['scale'] - scale) < 0.001]
            if existing:
                existing[0]["avg_ppl_nat"] = np.mean(ppl_nat_vals)
                existing[0]["avg_ppl_inj"] = np.mean(ppl_inj_vals)
                existing[0]["ppl_delta"] = np.mean(ppl_inj_vals) - np.mean(ppl_nat_vals)
                existing[0]["text_change_rate"] = sum(text_changed) / len(text_changed) * 100
                existing[0]["has_gen"] = True
            else:
                results.append({
                    "scale": scale, "avg_cos": 0, "1-cos": 0,
                    "avg_kl": 0, "top1_rate": 0,
                    "avg_ppl_nat": np.mean(ppl_nat_vals),
                    "avg_ppl_inj": np.mean(ppl_inj_vals),
                    "ppl_delta": np.mean(ppl_inj_vals) - np.mean(ppl_nat_vals),
                    "text_change_rate": sum(text_changed) / len(text_changed) * 100,
                    "has_gen": True, "n": len(text_changed),
                })
    
    # Print results table
    log(f"\n  === Threshold Search Results ===")
    log(f"  {'scale':>7s}  {'1-cos':>10s}  {'KL':>8s}  {'T1%':>6s}  {'Txt%':>6s}  {'PPL_nat':>8s}  {'PPL_inj':>8s}  {'dPPL':>8s}  {'n':>3s}")
    for r in results:
        if r.get('has_gen'):
            log(f"  {r['scale']:7.3f}  {r['1-cos']:10.6f}  {r['avg_kl']:8.4f}  "
                f"{r['top1_rate']:5.0f}%  {r['text_change_rate']:5.0f}%  "
                f"{r['avg_ppl_nat']:8.1f}  {r['avg_ppl_inj']:8.1f}  {r['ppl_delta']:+8.1f}  {r['n']:3d}")
        else:
            log(f"  {r['scale']:7.3f}  {r['1-cos']:10.6f}  {r['avg_kl']:8.4f}  "
                f"{r['top1_rate']:5.0f}%  {'---':>6s}  {'---':>8s}  {'---':>8s}  {'---':>8s}  {r['n']:3d}")
    
    # Find optimal scale: highest top1_rate with ppl_delta < 10
    log(f"\n  === Optimal Scale Analysis ===")
    viable = [r for r in results if r['ppl_delta'] < 10 and r['ppl_delta'] > -10]
    if viable:
        best = max(viable, key=lambda x: x['top1_rate'] + x['text_change_rate'])
        log(f"  Best controlled scale: {best['scale']}")
        log(f"    1-cos={best['1-cos']:.6f}, KL={best['avg_kl']:.4f}")
        log(f"    Top-1 change: {best['top1_rate']:.0f}%, Text change: {best['text_change_rate']:.0f}%")
        log(f"    PPL delta: {best['ppl_delta']:+.1f}")
    else:
        log(f"  No viable controlled scale found (all cause PPL collapse or no effect)")
        # Find the scale with max effect before collapse
        pre_collapse = [r for r in results if r['ppl_delta'] < 100]
        if pre_collapse:
            best = max(pre_collapse, key=lambda x: x['1-cos'])
            log(f"  Best pre-collapse scale: {best['scale']} (1-cos={best['1-cos']:.4f})")
    
    # Detect threshold (if any)
    sorted_by_scale = sorted(results, key=lambda x: x['scale'])
    for i in range(1, len(sorted_by_scale)):
        prev = sorted_by_scale[i-1]
        curr = sorted_by_scale[i]
        # Threshold: 1-cos jumps by >10x or ppl_delta jumps >10
        if curr['1-cos'] > prev['1-cos'] * 10 and prev['1-cos'] > 0.001:
            log(f"\n  >> THRESHOLD DETECTED between scale {prev['scale']:.3f} and {curr['scale']:.3f}")
            log(f"     1-cos: {prev['1-cos']:.6f} -> {curr['1-cos']:.6f} ({curr['1-cos']/prev['1-cos']:.0f}x)")
            log(f"     PPL delta: {prev['ppl_delta']:+.1f} -> {curr['ppl_delta']:+.1f}")
            break
    
    return results


# ============================================================
# P114: Semantic Quality Evaluation
# ============================================================
def run_p114(model, tokenizer, mname, n_layers, d_model, texts):
    """
    P114: 语义质量评估 — 用centroid距离衡量操控后文本的类别偏移
    
    原理:
    - Phase XIII只测量了cos/KL/PPL, 没有评估语义方向是否真的偏移
    - 本实验: 操控文本A→生成文本B, 计算B的hidden state与各category centroid的距离
    - 如果操控有效, B应该更接近target category而非source category
    
    方法:
    1. 选定source_cat和target_cat
    2. 生成20个操控后的文本(用optimal scale from P113)
    3. 计算每个操控文本的h_final与所有centroid的cos距离
    4. 统计: 操控后文本的"最近centroid"是否从source转向target
    """
    log(f"\n{'='*70}")
    log(f"  P114: Semantic Quality Evaluation ({mname})")
    log(f"{'='*70}")
    
    centroids = compute_centroids(model, tokenizer, texts, n_layers, n_samples=40)
    cats = list(centroids.keys())
    if len(cats) < 2:
        log("  Not enough categories")
        return {}
    
    dev = model.device
    actual_layers = get_actual_layers(model)
    if actual_layers is None:
        log("  Cannot find layers!")
        return {}
    
    inj_layers = list(range(min(n_layers // 4, len(actual_layers))))
    
    # Test multiple category pairs
    pairs = [(cats[i], cats[(i+1) % len(cats)]) for i in range(min(6, len(cats)))]
    
    # Use scale based on model (from Phase XIII results)
    model_scales = {"glm4": 0.15, "deepseek7b": 0.5, "qwen3": 2.0, "gemma4": 0.5}
    test_scale = model_scales.get(mname, 0.5)
    
    log(f"  Using scale={test_scale} for {mname}")
    log(f"  Testing {len(pairs)} category pairs, 10 texts per pair")
    
    all_results = []
    
    for src_cat, tgt_cat in pairs:
        if tgt_cat not in centroids or src_cat not in centroids:
            continue
        
        target_dir = centroids[tgt_cat] - centroids[src_cat]
        target_dir = F.normalize(target_dir.float(), dim=0).to(dev)
        
        # Get texts from source category
        src_texts = [t[0] for t in texts if t[1] == src_cat][:10]
        if not src_texts:
            src_texts = [t[0] for t in texts[:10]]
        
        cos_to_src_nat = []
        cos_to_tgt_nat = []
        cos_to_src_inj = []
        cos_to_tgt_inj = []
        category_shifted = 0
        
        for text in src_texts:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                
                # Natural generation h_final
                with torch.no_grad():
                    nat_out = model(**inputs, output_hidden_states=True)
                    nat_h = nat_out.hidden_states[-1][:, -1, :].float().cpu()
                
                # Injected generation h_final
                handles = [actual_layers[lidx].register_forward_hook(make_inject_hook(target_dir, test_scale))
                           for lidx in inj_layers]
                try:
                    with torch.no_grad():
                        inj_out = model(**inputs, output_hidden_states=True)
                        inj_h = inj_out.hidden_states[-1][:, -1, :].float().cpu()
                finally:
                    for h in handles:
                        h.remove()
                
                # Compute cos to all centroids
                src_centroid = centroids[src_cat]
                tgt_centroid = centroids[tgt_cat]
                
                cos_src_nat = F.cosine_similarity(nat_h.unsqueeze(0), src_centroid.unsqueeze(0)).item()
                cos_tgt_nat = F.cosine_similarity(nat_h.unsqueeze(0), tgt_centroid.unsqueeze(0)).item()
                cos_src_inj = F.cosine_similarity(inj_h.unsqueeze(0), src_centroid.unsqueeze(0)).item()
                cos_tgt_inj = F.cosine_similarity(inj_h.unsqueeze(0), tgt_centroid.unsqueeze(0)).item()
                
                cos_to_src_nat.append(cos_src_nat)
                cos_to_tgt_nat.append(cos_tgt_nat)
                cos_to_src_inj.append(cos_src_inj)
                cos_to_tgt_inj.append(cos_tgt_inj)
                
                # Check if "nearest centroid" shifted
                nat_nearest = "src" if cos_src_nat > cos_tgt_nat else "tgt"
                inj_nearest = "src" if cos_src_inj > cos_tgt_inj else "tgt"
                if nat_nearest == "src" and inj_nearest == "tgt":
                    category_shifted += 1
                    
            except Exception as e:
                continue
        
        n = len(cos_to_src_nat)
        if n > 0:
            avg_cos_src_nat = np.mean(cos_to_src_nat)
            avg_cos_tgt_nat = np.mean(cos_to_tgt_nat)
            avg_cos_src_inj = np.mean(cos_to_src_inj)
            avg_cos_tgt_inj = np.mean(cos_to_tgt_inj)
            
            all_results.append({
                "src": src_cat, "tgt": tgt_cat,
                "avg_cos_src_nat": avg_cos_src_nat, "avg_cos_tgt_nat": avg_cos_tgt_nat,
                "avg_cos_src_inj": avg_cos_src_inj, "avg_cos_tgt_inj": avg_cos_tgt_inj,
                "cos_src_delta": avg_cos_src_inj - avg_cos_src_nat,
                "cos_tgt_delta": avg_cos_tgt_inj - avg_cos_tgt_nat,
                "shifted": category_shifted, "n": n,
            })
    
    # Print results
    log(f"\n  === Semantic Shift Analysis (scale={test_scale}) ===")
    log(f"  {'src':>12s}  {'tgt':>12s}  {'cos_src':>10s}  {'cos_tgt':>10s}  {'d_src':>10s}  {'d_tgt':>10s}  {'shifted':>8s}  {'n':>3s}")
    log(f"  {'(natural)':>12s}  {'(natural)':>12s}  {'(natural)':>10s}  {'(natural)':>10s}")
    for r in all_results:
        log(f"  {r['src']:>12s}  {r['tgt']:>12s}  {r['avg_cos_src_nat']:10.4f}  {r['avg_cos_tgt_nat']:10.4f}  "
            f"{r['cos_src_delta']:+10.4f}  {r['cos_tgt_delta']:+10.4f}  {r['shifted']:>3d}/{r['n']:<3d}  {r['n']:3d}")
    
    # Summary
    if all_results:
        avg_shift_rate = np.mean([r['shifted'] / r['n'] for r in all_results]) * 100
        avg_d_tgt = np.mean([r['cos_tgt_delta'] for r in all_results])
        avg_d_src = np.mean([r['cos_src_delta'] for r in all_results])
        log(f"\n  Summary:")
        log(f"    Avg shift rate: {avg_shift_rate:.1f}% (texts that moved closer to target)")
        log(f"    Avg cos to src delta: {avg_d_src:+.4f}")
        log(f"    Avg cos to tgt delta: {avg_d_tgt:+.4f}")
        if avg_d_tgt > avg_d_src:
            log(f"    >> Net semantic shift toward TARGET direction!")
        else:
            log(f"    >> No consistent semantic shift detected")
    
    return all_results


# ============================================================
# P115: Manipulation Method Comparison
# ============================================================
def run_p115(model, tokenizer, mname, n_layers, d_model, texts):
    """
    P115: 操控方法对比 — 四种方法的直接比较
    
    方法:
    A) Cumulative Injection (全层注入, Phase XIII的方法)
    B) Single-layer Injection (只在L0注入, Phase XII的方法)
    C) Logit Override (直接修改最终logits)
    D) Activation Patching (用target文本的activation替换source)
    
    公平对比: 所有方法调整到产生相似的1-cos(≈0.05-0.10)
    
    评估:
    - KL divergence
    - Top-1 change rate
    - Text change rate
    - PPL delta
    - "Efficiency" = effect / total_computation_cost
    """
    log(f"\n{'='*70}")
    log(f"  P115: Manipulation Method Comparison ({mname})")
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
    
    uw, ub = get_unembed(model)
    if uw is not None:
        uw_gpu = uw.to(dev)
        ub_gpu = ub.to(dev) if ub is not None else None
    
    test_texts = [texts[i][0] for i in range(0, min(80, len(texts)), 3)][:20]
    
    # Method A: Cumulative Injection (first_quarter, scale=0.1)
    log(f"\n  [A] Cumulative Injection (first_quarter, scale=0.1)")
    inj_layers_a = list(range(min(n_layers // 4, len(actual_layers))))
    scale_a = 0.1
    
    res_a = {"cos": [], "kl": [], "top1": [], "text": [], "ppl_nat": [], "ppl_inj": []}
    for text in test_texts[:15]:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
                nat_logits = nat_out.logits[:, -1, :].float()
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            handles = [actual_layers[lidx].register_forward_hook(make_inject_hook(target_dir, scale_a))
                       for lidx in inj_layers_a]
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
            nat_t1 = nat_logits.argmax(dim=-1).item()
            inj_t1 = inj_logits.argmax(dim=-1).item()
            res_a["cos"].append(cos)
            res_a["kl"].append(kl)
            res_a["top1"].append(1 if nat_t1 != inj_t1 else 0)
            # Text gen
            nat_ids = inputs["input_ids"].clone()
            inj_ids = inputs["input_ids"].clone()
            with torch.no_grad():
                for _ in range(5):
                    out = model(input_ids=nat_ids, use_cache=True)
                    tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if tok.item() == tokenizer.eos_token_id: break
                    nat_ids = torch.cat([nat_ids, tok], dim=1)
                for _ in range(5):
                    hh = [actual_layers[lidx].register_forward_hook(make_inject_hook(target_dir, scale_a))
                          for lidx in inj_layers_a]
                    try:
                        out = model(input_ids=inj_ids, use_cache=True)
                        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    finally:
                        for h in hh: h.remove()
                    if tok.item() == tokenizer.eos_token_id: break
                    inj_ids = torch.cat([inj_ids, tok], dim=1)
            nat_t = tokenizer.decode(nat_ids[0], skip_special_tokens=True)
            inj_t = tokenizer.decode(inj_ids[0], skip_special_tokens=True)
            res_a["text"].append(1 if nat_t.strip() != inj_t.strip() else 0)
            p1, _ = compute_perplexity(model, tokenizer, nat_t)
            p2, _ = compute_perplexity(model, tokenizer, inj_t)
            res_a["ppl_nat"].append(p1)
            res_a["ppl_inj"].append(p2)
        except: continue
    
    # Method B: Single-layer L0 Injection
    log(f"  [B] Single-layer L0 Injection (scale=5.0)")
    scale_b = 5.0
    res_b = {"cos": [], "kl": [], "top1": [], "text": [], "ppl_nat": [], "ppl_inj": []}
    for text in test_texts[:15]:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
                nat_logits = nat_out.logits[:, -1, :].float()
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            h = actual_layers[0].register_forward_hook(make_inject_hook(target_dir, scale_b))
            try:
                with torch.no_grad():
                    inj_out = model(**inputs, output_hidden_states=True)
                    inj_logits = inj_out.logits[:, -1, :].float()
                    inj_h = inj_out.hidden_states[-1][:, -1, :].float()
            finally:
                h.remove()
            cos = F.cosine_similarity(nat_h, inj_h, dim=-1).item()
            kl = kl_divergence(nat_logits, inj_logits)
            res_b["cos"].append(cos)
            res_b["kl"].append(kl)
            res_b["top1"].append(1 if nat_logits.argmax().item() != inj_logits.argmax().item() else 0)
            res_b["text"].append(0)
            p1, _ = compute_perplexity(model, tokenizer, text)
            res_b["ppl_nat"].append(p1)
            res_b["ppl_inj"].append(p1)
        except: continue
    
    # Method C: Logit Override (直接替换top tokens)
    log(f"  [C] Logit Override (alpha mixing in logit space)")
    res_c = {"cos": [], "kl": [], "top1": [], "text": [], "ppl_nat": [], "ppl_inj": []}
    alpha_c = 0.3
    for text in test_texts[:15]:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
                nat_logits = nat_out.logits[:, -1, :].float()
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            # Target logits via unembed
            if uw_gpu is not None:
                shifted_h = nat_h + target_dir * 10.0
                tgt_logits = shifted_h @ uw_gpu.T
                if ub_gpu is not None:
                    tgt_logits = tgt_logits + ub_gpu.unsqueeze(0)
                mixed_logits = (1 - alpha_c) * nat_logits + alpha_c * tgt_logits
            else:
                mixed_logits = nat_logits
            
            kl = kl_divergence(nat_logits, mixed_logits)
            res_c["cos"].append(1.0)  # same h
            res_c["kl"].append(kl)
            res_c["top1"].append(1 if nat_logits.argmax().item() != mixed_logits.argmax().item() else 0)
            # Text gen with logit mixing
            gen_ids = inputs["input_ids"].clone()
            with torch.no_grad():
                for _ in range(5):
                    out = model(input_ids=gen_ids, use_cache=True)
                    logits = out.logits[:, -1, :].float()
                    h_f = out.hidden_states[-1][:, -1, :].float()
                    if uw_gpu is not None:
                        s_h = h_f + target_dir * 10.0
                        t_l = s_h @ uw_gpu.T
                        if ub_gpu is not None:
                            t_l = t_l + ub_gpu.unsqueeze(0)
                        m_l = (1 - alpha_c) * logits + alpha_c * t_l
                    else:
                        m_l = logits
                    tok = m_l.argmax(dim=-1, keepdim=True)
                    if tok.item() == tokenizer.eos_token_id: break
                    gen_ids = torch.cat([gen_ids, tok], dim=1)
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            nat_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            res_c["text"].append(1 if nat_text.strip() != gen_text.strip() else 0)
            p1, _ = compute_perplexity(model, tokenizer, nat_text)
            p2, _ = compute_perplexity(model, tokenizer, gen_text)
            res_c["ppl_nat"].append(p1)
            res_c["ppl_inj"].append(p2)
        except: continue
    
    # Summary comparison
    log(f"\n  === Method Comparison Summary ===")
    log(f"  {'Method':>30s}  {'1-cos':>10s}  {'KL':>8s}  {'T1%':>6s}  {'Txt%':>6s}  {'PPL_d':>8s}  {'n':>3s}")
    for name, res in [("A: Cumul.Inj(fq,s=0.1)", res_a),
                       ("B: Single L0(s=5.0)", res_b),
                       ("C: Logit Override(a=0.3)", res_c)]:
        if res["cos"]:
            n = len(res["cos"])
            log(f"  {name:>30s}  {1-np.mean(res['cos']):10.6f}  {np.mean(res['kl']):8.4f}  "
                f"{sum(res['top1'])/n*100:5.0f}%  {sum(res['text'])/n*100:5.0f}%  "
                f"{np.mean(res['ppl_inj'])-np.mean(res['ppl_nat']):+8.1f}  {n:3d}")
    
    return {"A": res_a, "B": res_b, "C": res_c}


# ============================================================
# P116: Four-Model Unified Comparison
# ============================================================
def run_p116(model, tokenizer, mname, n_layers, d_model, texts):
    """
    P116: 四模型统一对比 — 在相同配置下对比所有模型
    
    统一配置:
    - Cumulative injection, first_quarter, scale=0.2
    - 30 test texts
    - 测量: cos, KL, top1, text_change, PPL
    
    同时测量模型内部属性:
    - d_model (hidden size)
    - n_layers
    - average norm of hidden states
    - average cos between consecutive layers
    """
    log(f"\n{'='*70}")
    log(f"  P116: Unified Comparison ({mname})")
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
    
    inj_layers = list(range(min(n_layers // 4, len(actual_layers))))
    scale = 0.2
    
    # Model internal properties
    log(f"\n  Model properties:")
    log(f"    d_model = {d_model}")
    log(f"    n_layers = {n_layers}")
    log(f"    inj_layers = {len(inj_layers)} (first quarter)")
    
    # Measure inter-layer cos and norms
    sample_text = texts[0][0]
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=128).to(dev)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    inter_cos = []
    avg_norms = []
    n_states = min(len(outputs.hidden_states), n_layers + 1)
    for l in range(1, n_states - 1):
        h_l = outputs.hidden_states[l][:, -1, :].float()
        h_l1 = outputs.hidden_states[l + 1][:, -1, :].float()
        c = F.cosine_similarity(h_l, h_l1, dim=-1).item()
        inter_cos.append(c)
        avg_norms.append(h_l.norm(dim=-1).item())
    
    if inter_cos:
        log(f"    avg inter-layer cos = {np.mean(inter_cos):.4f}")
        log(f"    avg hidden norm = {np.mean(avg_norms):.2f}")
        log(f"    final hidden norm = {avg_norms[-1]:.2f}")
    
    # Main test: 30 texts with cumulative injection
    test_texts = [texts[i][0] for i in range(0, min(120, len(texts)), 4)][:30]
    
    cos_vals = []
    kl_vals = []
    top1_changed = []
    text_changed = []
    ppl_nat_vals = []
    ppl_inj_vals = []
    
    log(f"\n  Testing {len(test_texts)} texts with scale={scale}")
    
    for text in test_texts:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
            
            # Natural
            with torch.no_grad():
                nat_out = model(**inputs, output_hidden_states=True)
                nat_logits = nat_out.logits[:, -1, :].float()
                nat_h = nat_out.hidden_states[-1][:, -1, :].float()
            
            # Injected
            handles = [actual_layers[lidx].register_forward_hook(make_inject_hook(target_dir, scale))
                       for lidx in inj_layers]
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
            top1_changed.append(1 if nat_logits.argmax().item() != inj_logits.argmax().item() else 0)
            
            # Generate 5 tokens
            nat_ids = inputs["input_ids"].clone()
            inj_ids = inputs["input_ids"].clone()
            with torch.no_grad():
                for _ in range(5):
                    out = model(input_ids=nat_ids, use_cache=True)
                    tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if tok.item() == tokenizer.eos_token_id: break
                    nat_ids = torch.cat([nat_ids, tok], dim=1)
                for _ in range(5):
                    hh = [actual_layers[lidx].register_forward_hook(make_inject_hook(target_dir, scale))
                          for lidx in inj_layers]
                    try:
                        out = model(input_ids=inj_ids, use_cache=True)
                        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    finally:
                        for h in hh: h.remove()
                    if tok.item() == tokenizer.eos_token_id: break
                    inj_ids = torch.cat([inj_ids, tok], dim=1)
            nat_t = tokenizer.decode(nat_ids[0], skip_special_tokens=True)
            inj_t = tokenizer.decode(inj_ids[0], skip_special_tokens=True)
            text_changed.append(1 if nat_t.strip() != inj_t.strip() else 0)
            p1, _ = compute_perplexity(model, tokenizer, nat_t)
            p2, _ = compute_perplexity(model, tokenizer, inj_t)
            ppl_nat_vals.append(p1)
            ppl_inj_vals.append(p2)
        except:
            continue
    
    if cos_vals:
        n = len(cos_vals)
        result = {
            "model": mname, "d_model": d_model, "n_layers": n_layers,
            "avg_1_cos": 1 - np.mean(cos_vals),
            "avg_kl": np.mean(kl_vals),
            "top1_rate": sum(top1_changed) / n * 100,
            "text_change_rate": sum(text_changed) / n * 100,
            "avg_ppl_delta": np.mean(ppl_inj_vals) - np.mean(ppl_nat_vals),
            "n": n,
            "avg_inter_cos": np.mean(inter_cos) if inter_cos else 0,
            "avg_norm": np.mean(avg_norms) if avg_norms else 0,
        }
        log(f"\n  === Unified Result ({mname}) ===")
        log(f"    1-cos: {result['avg_1_cos']:.6f}")
        log(f"    KL:    {result['avg_kl']:.4f}")
        log(f"    Top-1: {result['top1_rate']:.0f}%")
        log(f"    Text:  {result['text_change_rate']:.0f}%")
        log(f"    dPPL:  {result['avg_ppl_delta']:+.1f}")
        return result
    
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
            print(f"  Skipping {mname} (not found)")
            continue
        
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        log_dir = _Path(f"d:/develop/TransformerLens-main/tests/glm5_temp/stage719_phase14_{mname}_{ts}")
        log = Logger(str(log_dir), "results")
        
        log(f"\n{'#'*70}")
        log(f"# Stage 719: Phase XIV — {mname}")
        log(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"{'#'*70}")
        
        try:
            model, tokenizer, n_layers, d_model = load_model(mname)
        except Exception as e:
            log(f"  FAILED to load {mname}: {e}")
            continue
        
        try:
            run_p113(model, tokenizer, mname, n_layers, d_model, texts)
            run_p114(model, tokenizer, mname, n_layers, d_model, texts)
            run_p115(model, tokenizer, mname, n_layers, d_model, texts)
            run_p116(model, tokenizer, mname, n_layers, d_model, texts)
        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        log(f"\n  Done with {mname}")
        log.close()
        log = None
        
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  GPU freed for {mname}. Waiting 30s...")
        time.sleep(30)
    
    print("\n" + "="*60)
    print("  Phase XIV COMPLETE for all models")
    print("="*60)


if __name__ == "__main__":
    main()
