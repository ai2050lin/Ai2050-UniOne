#!/usr/bin/env python3
"""
Stage 717: Phase XII — 操控有效性验证 + 非Logits操控 + 多模型独立测试
========================================================================
核心问题: Phase XI发现logits mixing操控完全无效(INV-403/404)，本阶段
1. 直接测量KL散度确认操控是否真的无效
2. 探索新操控方法: attention head干预、FFN输出偏移、layer norm缩放
3. 每个模型独立进程，大样本量(200+)

P105: Logits KL散度分析 — 直接测量操控前后logits分布差异
P106: Attention Head干预 — 直接修改attention权重实现方向控制
P107: FFN输出偏移 — 在FFN层注入语义方向
P108: 多模型交叉验证 — 四模型依次独立测试

关键原理:
- KL散度 = Σ p(x) log(p(x)/q(x))，量化两个概率分布的差异
- 如果操控真的无效，KL(logit_natural || logit_manipulated) ≈ 0
- Attention干预: 修改特定head的attn_output实现语义偏移
- FFN偏移: 在residual stream注入方向向量，利用层间传播放大效应

用法: python stage717_phase12.py --model qwen3
      python stage717_phase12.py --model deepseek7b
      python stage717_phase12.py --model glm4
      python stage717_phase12.py --model gemma4
      python stage717_phase12.py --model all  (依次四模型)
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

# ===== Text dataset (477 texts, 11 categories) =====
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
        "Beijing roast duck is a famous dish originating from the imperial kitchen.",
        "The Chinese writing system has over fifty thousand characters.",
        "The Red Cliff is a famous historical site from the Three Kingdoms period.",
        "The Terracotta Warriors were discovered by farmers in nineteen seventy four.",
        "Peking opera is one of the most influential forms of Chinese performing arts.",
        "The Grand Canal is the longest artificial waterway in the world.",
        "Chinese gardens seek to recreate natural landscapes in miniature.",
        "Confucianism has shaped Chinese society for over two thousand years.",
        "The Yellow River is known as the cradle of Chinese civilization.",
        "Chinese tea culture has a history spanning thousands of years.",
        "The Zhou dynasty lasted longer than any other dynasty in Chinese history.",
        "The ancient Chinese invented papermaking printing and gunpowder.",
        "The Chinese zodiac consists of twelve animal signs.",
        "Mahjong is a popular tile-based game that originated in China.",
        "The philosophy of yin and yang is fundamental to Chinese thought.",
        "The Temple of Heaven was where emperors prayed for good harvests.",
        "Sichuan cuisine is famous for its bold flavors and use of spices.",
        "Chinese silk was traded along the Silk Road for centuries.",
        "The Chinese education system emphasizes mathematics and science.",
        "The Mid-Autumn Festival celebrates the harvest moon.",
    ]
    for t in chinese: T.append((t, "chinese"))
    poetry = [
        "Roses are red violets are blue.", "Shall I compare thee to a summer's day.",
        "The road not taken by Robert Frost.", "I wandered lonely as a cloud that floats on high.",
        "Two roads diverged in a yellow wood and sorry I could not travel both.",
        "To be or not to be that is the question.", "Hope is the thing with feathers.",
        "Do not go gentle into that good night.", "In Xanadu did Kubla Khan a stately pleasure dome decree.",
        "The fog comes on little cat feet.", "Because I could not stop for death he kindly stopped for me.",
        "I think that I shall never see a poem lovely as a tree.",
        "The Waste Land is a landmark modernist poem by T.S. Eliot.",
        "Ode to a Nightingale explores the themes of mortality and beauty.",
        "The Raven by Edgar Allan Poe features the famous line nevermore.",
        "Emily Dickinson wrote over seventeen hundred poems in her lifetime.",
        "Sonnets traditionally have fourteen lines and a specific rhyme scheme.",
        "Free verse poetry does not follow a regular meter or rhyme pattern.",
        "Haiku is a Japanese poetic form consisting of three lines with a five seven five syllable structure.",
        "Blank verse is written in iambic pentameter but does not rhyme.",
        "An epic poem is a lengthy narrative poem that tells the story of a heroic figure.",
        "Metaphor compares two unlike things without using like or as.",
        "Alliteration is the repetition of initial consonant sounds in nearby words.",
        "Personification gives human qualities to nonhuman things.",
        "Imagery uses vivid language to appeal to the senses.",
        "The Lake Isle of Innisfree expresses a longing for peaceful rural life.",
        "The Love Song of J. Alfred Prufrock is a modernist masterpiece.",
        "William Wordsworth is considered a founder of the Romantic movement in English poetry.",
        "Langston Hughes was a leading figure of the Harlem Renaissance.",
        "Maya Angelou's Still I Rise is an anthem of resilience and empowerment.",
        "Dante's Divine Comedy describes a journey through Hell Purgatory and Paradise.",
        "Homer's Iliad and Odyssey are foundational works of Western literature.",
        "Virgil's Aeneid tells the legendary story of Aeneas.",
        "The Bhagavad Gita is a sacred Hindu scripture in verse form.",
        "Rumi's poetry explores themes of love spirituality and the human condition.",
        "Sappho of Lesbos is one of the earliest known female poets.",
        "Alexander Pope is known for his satirical verse and heroic couplets.",
        "John Keats wrote about beauty mortality and the imagination.",
        "Lord Byron was a leading figure of the Romantic movement.",
        "Percy Bysshe Shelley wrote Ode to the West Wind.",
        "Walt Whitman's Leaves of Grass revolutionized American poetry.",
        "Robert Frost often wrote about rural life in New England.",
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

# Gemma4 path
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
    """KL(p || q) for two probability distributions."""
    p = F.softmax(p.float(), dim=-1) + eps
    q = F.softmax(q.float(), dim=-1) + eps
    return (p * (p / q).log()).sum().item()


def js_divergence(p, q, eps=1e-10):
    """JS divergence between two distributions (symmetric)."""
    p = F.softmax(p.float(), dim=-1) + eps
    q = F.softmax(q.float(), dim=-1) + eps
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ===== P105: KL散度直接验证操控有效性 =====
def get_logits_on_gpu(model, tokenizer, text, max_length=128):
    """Get logits and h_final on GPU (native dtype), and unembed on GPU (float32)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        nat_logits = outputs.logits[:, -1, :].float()  # keep on GPU
        h_final = outputs.hidden_states[-1][:, -1, :].float()  # keep on GPU
    return nat_logits, h_final, inputs


def run_p105(model, tokenizer, mname, n_layers, texts):
    """
    P105: 直接对比操控前后的logits分布差异
    
    原理:
    - 对同一输入，分别计算 natural_logits 和 manipulated_logits
    - 用KL散度和JS散度量化差异
    - 如果操控有效，KL应该显著 > 0；如果操控无效，KL ≈ 0
    - 同时测量: top-k token变化率、logits L2距离
    """
    log(f"\n{'='*60}")
    log(f"  P105: Logits KL散度分析 ({mname})")
    log(f"{'='*60}")
    
    centroids = compute_centroids(model, tokenizer, texts, n_layers, n_samples=30)
    cats = list(centroids.keys())
    if len(cats) < 2:
        log("  Not enough categories")
        return {}
    
    uw, ub = get_unembed(model)
    # Move unembed to GPU for all-GPU computation
    if uw is not None:
        uw = uw.to(model.device)
        if ub is not None:
            ub = ub.to(model.device)
    dev = model.device
    
    # Pick 5 direction pairs
    dir_pairs = []
    for i in range(min(5, len(cats))):
        j = (i + 1) % len(cats)
        dir_pairs.append((cats[i], cats[j]))
    
    alphas = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    test_indices = list(range(0, min(100, len(texts)), 3))[:60]  # 60 test samples
    
    results = {"by_alpha": defaultdict(list), "details": []}
    
    log(f"\n  Testing {len(test_indices)} samples x {len(dir_pairs)} directions x {len(alphas)} alphas")
    
    for src_cat, tgt_cat in dir_pairs:
        if tgt_cat not in centroids or src_cat not in centroids:
            continue
        target_dir = centroids[tgt_cat] - centroids[src_cat]
        target_dir = F.normalize(target_dir.float(), dim=0).to(dev)
        
        for idx in test_indices:
            text = texts[idx][0]
            try:
                nat_logits, h_final, _ = get_logits_on_gpu(model, tokenizer, text)
                
                for alpha in alphas:
                    if uw is not None:
                        shifted = h_final + target_dir * alpha * 3.0
                        tgt_logits = shifted @ uw.T
                        if ub is not None:
                            tgt_logits = tgt_logits + ub.unsqueeze(0)
                        mixed = (1 - alpha) * nat_logits + alpha * tgt_logits
                    else:
                        mixed = nat_logits
                    
                    kl = kl_divergence(nat_logits, mixed)
                    js = js_divergence(nat_logits, mixed)
                    l2 = (nat_logits - mixed).norm().item()
                    
                    nat_top5 = nat_logits[0].topk(5).indices.tolist()
                    mix_top5 = mixed[0].topk(5).indices.tolist()
                    overlap = len(set(nat_top5) & set(mix_top5))
                    
                    results["by_alpha"][alpha].append({
                        "kl": kl, "js": js, "l2": l2, "overlap5": overlap,
                        "src": src_cat, "tgt": tgt_cat,
                    })
            except Exception as e:
                log(f"    [WARN] idx={idx}: {e}")
                continue
    
    # Summary
    log("\n  === Results by alpha (logits mixing) ===")
    log(f"  {'alpha':>6s}  {'avg_KL':>10s}  {'avg_JS':>10s}  {'avg_L2':>10s}  {'avg_overlap5':>14s}  {'n':>5s}")
    for alpha in alphas:
        data = results["by_alpha"][alpha]
        if data:
            avg_kl = np.mean([d["kl"] for d in data])
            avg_js = np.mean([d["js"] for d in data])
            avg_l2 = np.mean([d["l2"] for d in data])
            avg_ov = np.mean([d["overlap5"] for d in data])
            log(f"  {alpha:6.1f}  {avg_kl:10.4f}  {avg_js:10.4f}  {avg_l2:10.2f}  {avg_ov:14.2f}/5  {len(data):5d}")
    
    # Also test h-space injection with scaling (pure unembed, no mixing)
    log("\n  === Results by scaling factor (h-space injection, pure unembed) ===")
    scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    scale_results = defaultdict(list)
    
    for src_cat, tgt_cat in dir_pairs[:2]:
        if tgt_cat not in centroids or src_cat not in centroids:
            continue
        target_dir = centroids[tgt_cat] - centroids[src_cat]
        target_dir = F.normalize(target_dir.float(), dim=0).to(dev)
        
        for idx in test_indices[:30]:
            text = texts[idx][0]
            try:
                nat_logits, h_final, _ = get_logits_on_gpu(model, tokenizer, text)
                
                for scale in scales:
                    shifted = h_final + target_dir * scale
                    tgt_logits = shifted @ uw.T
                    if ub is not None:
                        tgt_logits = tgt_logits + ub.unsqueeze(0)
                    
                    kl = kl_divergence(nat_logits, tgt_logits)
                    js = js_divergence(nat_logits, tgt_logits)
                    l2 = (nat_logits - tgt_logits).norm().item()
                    
                    nat_top5 = nat_logits.topk(5).indices
                    tgt_top5 = tgt_logits.topk(5).indices
                    overlap = len(set(nat_top5.tolist()) & set(tgt_top5.tolist()))
                    
                    scale_results[scale].append({"kl": kl, "js": js, "l2": l2, "overlap5": overlap})
            except Exception as e:
                continue
    
    log(f"  {'scale':>6s}  {'avg_KL':>10s}  {'avg_JS':>10s}  {'avg_L2':>10s}  {'avg_ov5':>10s}  {'n':>5s}")
    for scale in scales:
        data = scale_results[scale]
        if data:
            avg_kl = np.mean([d["kl"] for d in data])
            avg_js = np.mean([d["js"] for d in data])
            avg_l2 = np.mean([d["l2"] for d in data])
            avg_ov = np.mean([d["overlap5"] for d in data])
            log(f"  {scale:6.1f}  {avg_kl:10.4f}  {avg_js:10.4f}  {avg_l2:10.2f}  {avg_ov:10.2f}/5  {len(data):5d}")
    
    return {"by_alpha": {str(k): len(v) for k, v in results["by_alpha"].items()},
            "by_scale": {str(k): len(v) for k, v in scale_results.items()}}


# ===== P106: Attention Head干预 =====
def run_p106(model, tokenizer, mname, n_layers, d_model, texts):
    """
    P106: 直接修改attention输出实现方向控制
    
    原理:
    - Transformer的attention输出是 residual stream 的主要更新源
    - 如果能在特定层的attention输出中注入方向向量，效果应比logits mixing更强
    - 方法: hook attention输出，在output上投影一个语义方向
    - 对比: 不改attn vs 改attn后生成的文本perplexity
    """
    log(f"\n{'='*60}")
    log(f"  P106: Attention Head干预 ({mname})")
    log(f"{'='*60}")
    
    centroids = compute_centroids(model, tokenizer, texts, n_layers, n_samples=30)
    cats = list(centroids.keys())
    if len(cats) < 2:
        log("  Not enough categories")
        return {}
    
    target_dir = centroids[cats[1]] - centroids[cats[0]]
    target_dir = F.normalize(target_dir, dim=0)
    
    # 找到transformer layers的位置
    model_ref = model
    layers_container = None
    for attr in ['model', 'transformer', 'language_model']:
        container = getattr(model_ref, attr, None)
        if container is not None:
            layers_container = container
            break
    if layers_container is None:
        log("  Cannot find transformer layers container")
        return {}
    
    # 找到实际层
    actual_layers = None
    for attr in ['layers', 'blocks', 'encoder', 'decoder', 'h']:
        l = getattr(layers_container, attr, None)
        if l is not None and hasattr(l, '__len__'):
            actual_layers = l
            break
    if actual_layers is None:
        log("  Cannot find transformer layers")
        return {}
    
    # 测试在不同层注入的效果
    test_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    prompts = [t[0] for t in texts[:30]]
    
    log(f"\n  Testing attention injection at layers: {test_layers}")
    log(f"  Using {len(prompts)} prompts, injection scale=1.0")
    
    injection_results = {}
    
    for inj_layer in test_layers:
        if inj_layer >= len(actual_layers):
            continue
        
        # 生成带attention hook的文本
        hooked_texts = []
        nat_texts = []
        delta_ppls = []
        
        for prompt in prompts[:15]:
            try:
                # Natural generation (no hook)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(model.device)
                with torch.no_grad():
                    nat_out = model.generate(
                        **inputs, max_new_tokens=15, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        output_hidden_states=False,
                    )
                nat_text = tokenizer.decode(nat_out[0], skip_special_tokens=True)
                nat_ppl, _ = compute_perplexity(model, tokenizer, nat_text)
                nat_texts.append(nat_text)
                
                # Hooked generation
                def make_hook(inj_layer_idx, direction, scale=1.0):
                    def hook_fn(module, input, output):
                        # output is typically a tuple, first element is hidden_states
                        if isinstance(output, tuple):
                            hs = output[0]
                        else:
                            hs = output
                        # Inject direction into last token position
                        dev = hs.device
                        dtyp = hs.dtype
                        hs_injected = hs.clone()
                        hs_injected[:, -1, :] = hs_injected[:, -1, :] + direction.to(dtyp).to(dev) * scale
                        if isinstance(output, tuple):
                            return (hs_injected,) + output[1:]
                        return hs_injected
                    return hook_fn
                
                hook = make_hook(inj_layer, target_dir, scale=1.0)
                handle = actual_layers[inj_layer].register_forward_hook(hook)
                
                try:
                    with torch.no_grad():
                        hook_out = model.generate(
                            **inputs, max_new_tokens=15, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    hooked_text = tokenizer.decode(hook_out[0], skip_special_tokens=True)
                    hook_ppl, _ = compute_perplexity(model, tokenizer, hooked_text)
                    hooked_texts.append(hooked_text)
                    delta_ppls.append(hook_ppl - nat_ppl)
                finally:
                    handle.remove()
                
            except Exception as e:
                log(f"    [WARN] L{inj_layer} prompt failed: {e}")
                continue
        
        if delta_ppls:
            avg_delta = np.mean(delta_ppls)
            injection_results[inj_layer] = {
                "avg_delta_ppl": float(avg_delta),
                "n_samples": len(delta_ppls),
                "delta_ppls": [float(d) for d in delta_ppls[:5]],
            }
            
            # Check text difference
            n_changed = sum(1 for n, h in zip(nat_texts, hooked_texts) if n != h)
            log(f"  L{inj_layer}: avg_delta_ppl={avg_delta:+.2f}, "
                f"texts_changed={n_changed}/{len(nat_texts)}, n={len(delta_ppls)}")
            
            # Show examples
            for i in range(min(3, len(nat_texts))):
                if nat_texts[i] != hooked_texts[i]:
                    log(f"    NAT: {nat_texts[i][:80]}")
                    log(f"    HOK: {hooked_texts[i][:80]}")
    
    # Summary: which layer is most effective?
    log("\n  === Injection effectiveness summary ===")
    if injection_results:
        best_layer = max(injection_results, key=lambda l: abs(injection_results[l]["avg_delta_ppl"]))
        worst_layer = min(injection_results, key=lambda l: abs(injection_results[l]["avg_delta_ppl"]))
        log(f"  Most effective layer: L{best_layer} (delta_ppl={injection_results[best_layer]['avg_delta_ppl']:+.2f})")
        log(f"  Least effective layer: L{worst_layer} (delta_ppl={injection_results[worst_layer]['avg_delta_ppl']:+.2f})")
        
        # Is there a pattern? Early vs late layers
        early = [injection_results[l]["avg_delta_ppl"] for l in test_layers[:3] if l in injection_results]
        late = [injection_results[l]["avg_delta_ppl"] for l in test_layers[3:] if l in injection_results]
        if early and late:
            log(f"  Early layers avg: {np.mean(early):+.2f}, Late layers avg: {np.mean(late):+.2f}")
    
    return injection_results


# ===== P107: FFN输出偏移 =====
def run_p107(model, tokenizer, mname, n_layers, d_model, texts):
    """
    P107: 在residual stream上注入语义方向（层输出hook）
    
    原理:
    - 直接hook每层输出(hidden_states)，在最后一个token位置注入方向
    - 测量注入后最终hidden state的变化程度(cos similarity)
    - 如果skip connection稀释了注入，cos(nat, inj)应≈1.0
    - 如果注入能传播，cos应显著<1.0
    
    关键: 使用层级hook而非子模块hook，避免维度不匹配
    """
    log(f"\n{'='*60}")
    log(f"  P107: Residual Stream方向注入 ({mname})")
    log(f"{'='*60}")
    
    centroids = compute_centroids(model, tokenizer, texts, n_layers, n_samples=30)
    cats = list(centroids.keys())
    if len(cats) < 2:
        log("  Not enough categories")
        return {}
    
    target_dir = centroids[cats[1]] - centroids[cats[0]]
    target_dir = F.normalize(target_dir.float(), dim=0).to(model.device)
    dev = model.device
    
    # Get model's internal module structure
    model_ref = model
    layers_container = None
    for attr in ['model', 'transformer', 'language_model']:
        container = getattr(model_ref, attr, None)
        if container is not None:
            layers_container = container
            break
    if layers_container is None:
        log("  Cannot find layers container")
        return {}
    
    actual_layers = None
    for attr in ['layers', 'blocks', 'encoder', 'decoder', 'h']:
        l = getattr(layers_container, attr, None)
        if l is not None and hasattr(l, '__len__'):
            actual_layers = l
            break
    if actual_layers is None:
        log("  Cannot find layers")
        return {}
    
    log(f"  Found {len(actual_layers)} transformer layers")
    
    # Test injection at different layers with different scales
    test_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    test_scales = [1.0, 5.0, 10.0, 50.0, 100.0]
    test_texts = [t[0] for t in texts[:15]]
    
    propagation_results = {}
    
    for inj_layer in test_layers:
        if inj_layer >= len(actual_layers):
            continue
        
        for scale in test_scales:
            layer_key = f"L{inj_layer}_s{scale}"
            cos_values = []
            l2_values = []
            
            for text in test_texts[:8]:
                try:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
                    
                    # Natural forward pass
                    with torch.no_grad():
                        nat_out = model(**inputs, output_hidden_states=True)
                        nat_h = nat_out.hidden_states[-1][:, -1, :].float()
                    
                    # Injected forward pass — hook at specific layer output
                    def make_layer_hook(hook_layer_idx, direction, s):
                        def hook_fn(module, input, output):
                            # For decoder-only models, output is typically hidden_states or a tuple
                            if isinstance(output, tuple):
                                hs = output[0]  # (batch, seq, d_model)
                            else:
                                hs = output
                            hs_inj = hs.clone()
                            hs_inj[:, -1, :] = hs_inj[:, -1, :] + direction.to(hs.dtype) * s
                            if isinstance(output, tuple):
                                return (hs_inj,) + output[1:]
                            return hs_inj
                        return hook_fn
                    
                    handle = actual_layers[inj_layer].register_forward_hook(
                        make_layer_hook(inj_layer, target_dir, scale)
                    )
                    try:
                        with torch.no_grad():
                            inj_out = model(**inputs, output_hidden_states=True)
                            inj_h = inj_out.hidden_states[-1][:, -1, :].float()
                    finally:
                        handle.remove()
                    
                    cos = F.cosine_similarity(nat_h, inj_h, dim=-1).item()
                    l2 = (nat_h - inj_h).norm(dim=-1).item()
                    cos_values.append(cos)
                    l2_values.append(l2)
                    
                except Exception as e:
                    log(f"    [WARN] L{inj_layer} s={scale}: {e}")
                    continue
            
            if cos_values:
                propagation_results[layer_key] = {
                    "layer": inj_layer, "scale": scale,
                    "avg_cos": float(np.mean(cos_values)),
                    "avg_l2": float(np.mean(l2_values)),
                    "n": len(cos_values),
                }
    
    log(f"\n  === Residual Stream Injection Results ===")
    log(f"  {'Layer':>6s}  {'Scale':>6s}  {'avg_cos':>10s}  {'avg_L2':>10s}  {'1-cos':>10s}  {'n':>5s}")
    for key in sorted(propagation_results.keys()):
        r = propagation_results[key]
        log(f"  L{r['layer']:>4d}  {r['scale']:6.1f}  {r['avg_cos']:10.6f}  {r['avg_l2']:10.4f}  "
            f"{1-r['avg_cos']:10.6f}  {r['n']:5d}")
    
    # Key analysis
    all_cos = [r["avg_cos"] for r in propagation_results.values()]
    if all_cos:
        log(f"\n  Overall: avg_cos = {np.mean(all_cos):.6f}, avg(1-cos) = {1-np.mean(all_cos):.6f}")
        
        # Per-layer summary (at scale=50.0)
        log(f"\n  Per-layer summary at scale=50.0:")
        for l in test_layers:
            key = f"L{l}_s50.0"
            if key in propagation_results:
                r = propagation_results[key]
                log(f"    L{l}: cos={r['avg_cos']:.6f}, 1-cos={1-r['avg_cos']:.6f}, L2={r['avg_l2']:.4f}")
    
    return propagation_results


# ===== P108: 多模型交叉验证 =====
def run_p108(model, tokenizer, mname, n_layers, d_model, texts):
    """
    P108: 综合操控验证 - 每个模型运行简化版P105+P106+P107
    
    原理:
    - Phase XI只测了Qwen3，本阶段验证四个模型
    - 重点关注: 操控效果是否跨模型一致？
    - 如果所有模型都显示操控无效，则问题在方法论而非模型特性
    """
    log(f"\n{'='*60}")
    log(f"  P108: 综合操控验证 ({mname})")
    log(f"{'='*60}")
    
    # 1. 快速KL验证
    centroids = compute_centroids(model, tokenizer, texts, n_layers, n_samples=20)
    cats = list(centroids.keys())
    if len(cats) < 2:
        return {}
    
    target_dir = centroids[cats[1]] - centroids[cats[0]]
    target_dir = F.normalize(target_dir, dim=0)
    uw, ub = get_unembed(model)
    
    # Quick logits mixing test (10 samples, 3 scales)
    log("\n  [1/3] Quick logits mixing KL test...")
    quick_kls = []
    quick_ppls_nat = []
    quick_ppls_manip = []
    
    uw_gpu = uw.to(model.device) if uw is not None else None
    ub_gpu = ub.to(model.device) if ub is not None else None
    dir_gpu = target_dir.to(model.device)
    
    for idx in range(0, min(30, len(texts)), 3):
        text = texts[idx][0]
        try:
            nat_logits, h_final, inputs = get_logits_on_gpu(model, tokenizer, text)
            
            # Scale 3.0
            shifted = h_final + dir_gpu * 3.0
            tgt_logits = shifted @ uw_gpu.T
            if ub_gpu is not None:
                tgt_logits = tgt_logits + ub_gpu.unsqueeze(0)
            mixed = 0.5 * nat_logits + 0.5 * tgt_logits
            
            kl = kl_divergence(nat_logits, mixed)
            quick_kls.append(kl)
            
            # PPL check
            nat_ppl, _ = compute_perplexity(model, tokenizer, text)
            quick_ppls_nat.append(nat_ppl)
            
            # Generate with manipulation
            gen_ids = inputs["input_ids"].clone()
            for step in range(5):
                with torch.no_grad():
                    out = model(input_ids=gen_ids, output_hidden_states=True, use_cache=True)
                    h = out.hidden_states[-1][:, -1, :].float()
                    logits = out.logits[:, -1, :].float()
                    shifted_h = h + dir_gpu * 3.0
                    tgt_l = shifted_h @ uw_gpu.T
                    if ub_gpu is not None:
                        tgt_l = tgt_l + ub_gpu.unsqueeze(0)
                    m_l = 0.5 * logits + 0.5 * tgt_l
                    next_tok = torch.argmax(m_l, dim=-1, keepdim=True)
                    if next_tok.item() == tokenizer.eos_token_id:
                        break
                    gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            manip_ppl, _ = compute_perplexity(model, tokenizer, gen_text)
            quick_ppls_manip.append(manip_ppl)
            
        except Exception as e:
            log(f"    [WARN] idx={idx}: {e}")
    
    if quick_kls:
        log(f"  Logits mixing KL divergence: avg={np.mean(quick_kls):.4f}, std={np.std(quick_kls):.4f}")
    if quick_ppls_nat and quick_ppls_manip:
        log(f"  PPL: natural_avg={np.mean(quick_ppls_nat):.1f}, manipulated_avg={np.mean(quick_ppls_manip):.1f}")
        log(f"  PPL delta: {np.mean(quick_ppls_manip) - np.mean(quick_ppls_nat):+.2f}")
    
    # 2. Natural generation baseline
    log("\n  [2/3] Natural generation baseline...")
    nat_gen_ppls = []
    for idx in range(0, min(30, len(texts)), 3):
        text = texts[idx][0]
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=15, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0], skip_special_tokens=True)
            ppl, _ = compute_perplexity(model, tokenizer, gen)
            nat_gen_ppls.append(ppl)
        except:
            continue
    
    if nat_gen_ppls:
        log(f"  Natural generation avg PPL: {np.mean(nat_gen_ppls):.1f} (n={len(nat_gen_ppls)})")
    
    # 3. Skip connection dilution analysis
    log("\n  [3/3] Skip connection dilution measurement...")
    dilution = measure_skip_dilution(model, tokenizer, texts, n_layers, target_dir)
    
    return {
        "avg_kl": float(np.mean(quick_kls)) if quick_kls else 0,
        "avg_ppl_nat": float(np.mean(quick_ppls_nat)) if quick_ppls_nat else 0,
        "avg_ppl_manip": float(np.mean(quick_ppls_manip)) if quick_ppls_manip else 0,
        "dilution": dilution,
    }


def measure_skip_dilution(model, tokenizer, texts, n_layers, direction):
    """
    测量skip connection对方向注入的稀释效应
    
    原理:
    Transformer每一层的输出 = LayerNorm(input + Sublayer(input))
    skip connection意味着: 注入 δ 到第l层输出，第l+1层接收的是:
      h_{l+1} = input_l + Sublayer(h_l + δ)
    由于Sublayer的非线性，δ会被部分保留但也会被重新映射
    
    测量方法:
    1. 获取自然hidden states h_0, h_1, ..., h_L
    2. 在第l层注入方向: h_l' = h_l + δ
    3. 用近似方法计算h_l'经过第l+1层后的方向保留率
    """
    text = texts[0][0]
    try:
        states, inputs = get_all_h(model, tokenizer, text, n_layers)
    except:
        return {}
    
    dilution_per_layer = {}
    test_layers = list(range(0, n_layers, max(1, n_layers // 10)))
    
    for l in test_layers:
        if l >= len(states) - 1:
            continue
        
        h_l = states[l]
        h_l1 = states[l + 1]
        
        # 注入方向
        h_l_inj = h_l + direction * 1.0
        
        # 计算方向保留: cos(h_l1 - h_l, h_l_inj - h_l)
        # 这是近似: 如果完全保留，cos(direction, layer_transform(direction)) 应该 > 0
        delta_natural = h_l1 - h_l
        cos_natural_dir = F.cosine_similarity(delta_natural.unsqueeze(0), direction.unsqueeze(0)).item()
        
        dilution_per_layer[l] = {
            "cos(delta_natural, direction)": cos_natural_dir,
            "||delta_natural||": delta_natural.norm().item(),
            "||direction||": direction.norm().item(),
        }
    
    log(f"  {'Layer':>6s}  {'cos(delta,dir)':>14s}  {'||delta||':>10s}  {'||dir||':>10s}")
    for l, v in dilution_per_layer.items():
        log(f"  L{l:>4d}  {v['cos(delta_natural, direction)']:14.4f}  {v['||delta_natural||']:10.2f}  {v['||direction||']:10.4f}")
    
    return dilution_per_layer


# ===== Main =====
def main():
    global log, log_dir
    
    parser = argparse.ArgumentParser(description="Stage 717: Phase XII")
    parser.add_argument("--model", type=str, default="qwen3",
                        choices=["qwen3", "deepseek7b", "glm4", "gemma4", "all"],
                        help="Model to test (default: qwen3)")
    args = parser.parse_args()
    
    log_dir = os.path.join(r"d:\develop\TransformerLens-main\tests\glm5_temp",
                           f"stage717_phase12_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    log = Logger(log_dir, "results")
    log(f"Log dir: {log_dir}")
    log(f"Model: {args.model}")
    
    log("=" * 70)
    log("  Stage 717: Phase XII - 操控有效性验证 + 非Logits操控 + 多模型独立测试")
    log("=" * 70)
    
    all_texts = build_texts()
    categories = sorted(set(t[1] for t in all_texts))
    log(f"Total texts: {len(all_texts)}, Categories: {len(categories)} - {categories}")
    
    models_to_test = MODEL_ORDER if args.model == "all" else [args.model]
    all_results = {}
    
    for mname in models_to_test:
        if mname not in MODEL_MAP:
            log(f"\n  {mname} not found in MODEL_MAP, skipping")
            continue
        
        log(f"\n{'#'*70}")
        log(f"  Processing: {mname}")
        log(f"{'#'*70}")
        t0 = time.time()
        
        model, tokenizer, n_layers, d_model = load_model(mname)
        
        # P105: KL散度验证
        try:
            p105 = run_p105(model, tokenizer, mname, n_layers, all_texts)
            all_results[f"{mname}_P105"] = p105
        except Exception as e:
            log(f"  P105 failed: {e}")
            import traceback; traceback.print_exc()
        
        # P106: Attention干预
        try:
            p106 = run_p106(model, tokenizer, mname, n_layers, d_model, all_texts)
            all_results[f"{mname}_P106"] = {"layers_tested": list(p106.keys()) if p106 else []}
        except Exception as e:
            log(f"  P106 failed: {e}")
            import traceback; traceback.print_exc()
        
        # P107: FFN偏移
        try:
            p107 = run_p107(model, tokenizer, mname, n_layers, d_model, all_texts)
            all_results[f"{mname}_P107"] = {"configs_tested": len(p107) if p107 else 0}
        except Exception as e:
            log(f"  P107 failed: {e}")
            import traceback; traceback.print_exc()
        
        # P108: 综合验证
        try:
            p108 = run_p108(model, tokenizer, mname, n_layers, d_model, all_texts)
            all_results[f"{mname}_P108"] = p108
        except Exception as e:
            log(f"  P108 failed: {e}")
            import traceback; traceback.print_exc()
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        elapsed = time.time() - t0
        log(f"\n  {mname} done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    
    # Save summary
    with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    log(f"\n{'='*70}")
    log(f"  Stage 717 Phase XII COMPLETE")
    log(f"  Results saved to: {log_dir}")
    log(f"{'='*70}")
    
    log.close()


if __name__ == "__main__":
    main()
