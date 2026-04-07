#!/usr/bin/env python3
"""
Stage 715: Phase X — 突破文本质量瓶颈 + 偏置语义 + 条件W_l
=============================================================
P96: KL散度引导生成 — 用目标类别logits分布作为KL约束, 非直接混合
P97: 逐步退火策略 — alpha从高到低衰减, 模型逐步接管
P98: Prompt对比实验 — prompt操控 vs h-space操控 vs 自然生成, 文本质量对比
P99: 偏置项语义分析 — b_l与类别标签的相关性, PCA可视化
P100: 条件W_l分析 — 不同类别文本的层变换矩阵差异量化

三模型串行: Qwen3 -> DS7B -> GLM4 (排除Gemma4)
设备: CUDA
样本: 417条文本(10类别)
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
}
MODEL_ORDER = ["qwen3", "deepseek7b", "glm4"]

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
        "The invention of gunpowder changed the course of history.",
        "The Yangtze River Delta is China's most economically developed region.",
        "Beijing opera is one of China's most iconic cultural traditions.",
        "The Chinese zodiac has twelve animal signs.",
        "Chinese tea culture dates back thousands of years.",
        "The Grand Canal is the longest artificial waterway in the world.",
        "China's high-speed rail network is the largest in the world.",
        "The Chinese education system emphasizes mathematics and science.",
        "Chinese gardens are designed to create miniature landscapes.",
        "The Mid-Autumn Festival celebrates family reunion.",
        "China has launched several space exploration missions.",
        "Kung Fu is a traditional Chinese martial art with ancient origins.",
        "The Chinese economy is the second largest in the world.",
    ]
    for t in chinese: T.append((t, "chinese"))
    poetry = [
        "Roses are red, violets are blue, sugar is sweet, and so are you.",
        "Shall I compare thee to a summer's day? Thou art more lovely and more temperate.",
        "The road not taken was the one less traveled by, and that has made all the difference.",
        "I wandered lonely as a cloud that floats on high over vales and hills.",
        "To be or not to be, that is the question.", "Hope is the thing with feathers that perches in the soul.",
        "Two roads diverged in a yellow wood, and sorry I could not travel both.",
        "In the middle of difficulty lies opportunity.", "The only thing we have to fear is fear itself.",
        "Ask not what your country can do for you, ask what you can do for your country.",
        "Do not go gentle into that good night, rage rage against the dying of the light.",
        "Because I could not stop for death, he kindly stopped for me.",
        "I think that I shall never see a poem lovely as a tree.",
        "The fog comes on little cat feet.", "Once upon a midnight dreary while I pondered weak and weary.",
        "Whose woods these are I think I know, his house is in the village though.",
        "A thing of beauty is a joy forever.", "Water water everywhere nor any drop to drink.",
        "My heart leaps up when I behold a rainbow in the sky.",
        "The world is too much with us, late and soon, getting and spending we lay waste our powers.",
        "Four seasons fill the measure of the year, here are four sons of different character.",
        "The moon was a ghostly galleon tossed upon cloudy seas.",
        "She walks in beauty like the night of cloudless climes and starry skies.",
        "The lady doth protest too much methinks.", "All that glitters is not gold.",
        "A rose by any other name would smell as sweet.", "To thine own self be true.",
        "Parting is such sweet sorrow.", "The quality of mercy is not strained.",
        "We are such stuff as dreams are made on.", "Life is but a walking shadow.",
        "Friends Romans countrymen lend me your ears.", "Brevity is the soul of wit.",
        "The course of true love never did run smooth.", "Love looks not with the eyes but with the mind.",
        "Now is the winter of our discontent.", "Uneasy lies the head that wears a crown.",
        "All the world is a stage and all the men and women merely players.",
        "If music be the food of love play on.", "Something is rotten in the state of Denmark.",
        "Cowards die many times before their deaths.", "The better part of valor is discretion.",
    ]
    for t in poetry: T.append((t, "poetry"))
    philosophy = [
        "I think therefore I am.", "The unexamined life is not worth living.",
        "The only true wisdom is in knowing you know nothing.",
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
    # 5 more categories with fewer texts
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
        "Michael Jordan is widely considered the greatest basketball player of all time.",
        "The FIFA World Cup is the most watched sporting event on Earth.",
        "Serena Williams won 23 Grand Slam singles titles in tennis.",
        "Usain Bolt holds the world record in the 100 meter sprint.",
        "The Super Bowl is the championship game of American football.",
        "Cricket is hugely popular in India, Australia, and England.",
        "Swimming is an excellent full body workout.",
        "The Tour de France is the most prestigious cycling race.",
        "Baseball is known as America's pastime.", "Tennis requires agility and precision.",
        "The marathon is a 26.2 mile race testing endurance.", "Boxing has produced legendary champions like Muhammad Ali.",
        "Golf is played on courses with 18 holes.", "Volleyball is popular at beaches worldwide.",
        "The NBA is the premier professional basketball league.",
        "Ice hockey is Canada's national winter sport.",
        "Formula One racing features the fastest cars in the world.",
        "Rugby is a physically demanding contact sport.",
        "Wrestling is one of the oldest forms of combat sport.",
        "The Paralympics showcase athletes with disabilities.",
        "Extreme sports include skydiving, rock climbing, and surfing.",
        "Yoga improves flexibility, strength, and mental well being.",
        "Table tennis is widely played in China and East Asia.",
        "The NBA playoffs determine the league champion each year.",
        "A hat trick in soccer means scoring three goals in one game.",
        "Home runs are a key statistic in baseball.",
        "Martial arts teach discipline and self defense.",
        "The Ironman triathlon is one of the most grueling endurance events.",
        "Figure skating combines athleticism with artistic expression.",
        "Badminton is the fastest racket sport in the world.",
        "Rowing is a demanding sport requiring teamwork and strength.",
        "Archery requires focus and precision.",
        "The UEFA Champions League is the pinnacle of European club soccer.",
        "Mixed martial arts has grown rapidly in popularity.",
        "Rock climbing challenges both physical and mental strength.",
        "The Commonwealth Games feature athletes from former British colonies.",
        "Weightlifting builds muscle strength and power.",
        "Cross country running tests endurance over varied terrain.",
        "Handball is a fast paced team sport popular in Europe.",
        "The Stanley Cup is the championship trophy of the National Hockey League.",
        "Lacrosse has Native American origins and is growing internationally.",
        "Skiing and snowboarding are popular winter sports worldwide.",
        "Fencing is one of the oldest Olympic sports.",
        "Parkour involves moving efficiently through urban environments.",
        "Polo is known as the sport of kings.",
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

# ===== Helpers =====
def safe_decode(tokenizer, tid):
    try:
        tokens = tokenizer.decode([tid], skip_special_tokens=True)
        return tokens.replace(" ", "_") if tokens.strip() else f"<{tid}>"
    except:
        return f"<{tid}>"

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

def load_model(mname):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = MODEL_MAP[mname]
    log(f"  Loading {mname} from {p.name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(p), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(p), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Detect architecture
    cfg = model.config
    n_layers = getattr(cfg, 'num_hidden_layers', None) or getattr(cfg, 'n_layers', None)
    if n_layers is None:
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
    d_model = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'd_model', None)
    if d_model is None:
        d_model = 2048
    log(f"  {mname}: n_layers={n_layers}, d_model={d_model}, device={model.device}")
    return model, tokenizer, n_layers, d_model

def compute_centroids(model, tokenizer, texts, n_layers):
    """Compute per-category centroids from hidden states."""
    from collections import defaultdict
    cat_h = defaultdict(list)
    n_samples = min(40, len(texts))
    indices = list(range(0, len(texts), max(1, len(texts) // n_samples)))[:n_samples]
    for idx in indices:
        states, _ = get_all_h(model, tokenizer, texts[idx][0], n_layers)
        h_final = states[-1]
        cat_h[texts[idx][1]].append(h_final)
    centroids = {}
    for cat, hs in cat_h.items():
        centroids[cat] = torch.stack(hs).mean(dim=0)
    return centroids

def evaluate_text_quality(text):
    """Simple heuristic for text quality: longer non-special tokens + fewer repeats."""
    if not text or len(text) < 5:
        return 0.0
    # Penalize special chars
    special = sum(1 for c in text if c in '<>[]{}|/\\`~')
    special_ratio = special / max(len(text), 1)
    # Reward meaningful words (alphabetic sequences)
    words = [w for w in text.split() if len(w) > 1 and w.isalpha()]
    word_ratio = len(words) / max(len(text.split()), 1)
    # Penalize excessive repetition
    chunks = [text[i:i+3] for i in range(len(text)-2)]
    if len(chunks) > 0:
        unique_ratio = len(set(chunks)) / len(chunks)
    else:
        unique_ratio = 1.0
    # Score
    score = word_ratio * 0.4 + unique_ratio * 0.3 + (1 - special_ratio) * 0.3
    return max(0, min(1, score))

log = None

# ===== P96: KL散度引导生成 =====
def P96(model, tokenizer, texts, centroids, n_layers, d_model, mname):
    """KL divergence guided generation: penalize logits that diverge from target category distribution."""
    log(f"\n{'='*60}")
    log(f"P96: KL散度引导生成 [{mname}]")
    log(f"{'='*60}")
    
    # Build target category logit distributions (average logits for each category)
    log("  Building category logit distributions...")
    cat_logits = {}
    for cat, centroid in centroids.items():
        # Compute h_ctrl @ W.T + b to get target logits
        h = centroid.unsqueeze(0)  # [1, d]
        if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
            W = model.lm_head.weight.float()  # [vocab, d]
            ctrl_logits = (h.to(W.device).to(W.dtype) @ W.T).float()
            if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
                ctrl_logits = ctrl_logits + model.lm_head.bias.float().unsqueeze(0)
            cat_logits[cat] = ctrl_logits[0]  # [vocab]
        else:
            # Fallback: use model output
            log(f"  WARNING: No lm_head found, using model output for {cat}")
            cat_logits[cat] = None
    
    # Test pairs: (src_text, src_cat, tgt_cat)
    test_pairs = [
        ("code", "poetry"),
        ("code", "math_sci"),
        ("chinese", "gen_en"),
        ("math_sci", "philosophy"),
    ]
    
    kl_weights = [0.1, 0.3, 0.5, 0.8]
    n_gen = 20
    
    log(f"\n  {'Pair':>18} {'KL_w':>6} {'avg_cos':>9} {'quality':>8} {'text_preview':>50}")
    log(f"  {'-'*18} {'-'*6} {'-'*9} {'-'*8} {'-'*50}")
    
    results = []
    for src_cat, tgt_cat in test_pairs:
        # Find a source text
        src_idx = next(i for i, t in enumerate(texts) if t[1] == src_cat)
        src_text = texts[src_idx][0]
        tgt_dist = cat_logits.get(tgt_cat)
        if tgt_dist is None:
            continue
            
        for kl_w in kl_weights:
            tokens_cl = []
            cos_list = []
            input_ids = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(model.device)
            tgt_centroid = centroids[tgt_cat]
            orig_len = input_ids.shape[1]
            
            for step in range(n_gen):
                with torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)
                h_final = outputs.hidden_states[-1][:, -1, :].float().squeeze(0)
                natural_logits = outputs.logits[:, -1, :].float()  # [1, vocab]
                
                cos_val = F.cosine_similarity(h_final.unsqueeze(0), tgt_centroid.unsqueeze(0).to(h_final.device), dim=-1).item()
                cos_list.append(cos_val)
                
                # KL-guided: penalize divergence from target distribution
                # KL(p || q) where p = target_dist (softmax), q = natural_dist (softmax)
                # We want to minimize KL, so add -kl_weight * log(q/p) to logits
                natural_probs = F.softmax(natural_logits, dim=-1)  # [1, vocab]
                target_probs = F.softmax(tgt_dist.unsqueeze(0).to(natural_logits.device), dim=-1)  # [1, vocab]
                
                # Adjusted logits: natural_logits + kl_w * (log(target) - log(natural))
                # This encourages the output to be closer to target distribution
                target_log = torch.log(target_probs + 1e-10)
                natural_log = torch.log(natural_probs + 1e-10)
                kl_adjustment = kl_w * (target_log - natural_log)
                
                guided_logits = natural_logits + kl_adjustment
                # Temperature sampling
                probs = F.softmax(guided_logits / 0.8, dim=-1)
                topk_vals, topk_ids = probs.topk(50)
                topk_probs = F.softmax(topk_vals / 0.8, dim=-1)
                sampled_idx = torch.multinomial(topk_probs, 1)
                token_id = topk_ids[0, sampled_idx[0, 0]].view(1, 1)
                
                tokens_cl.append(safe_decode(tokenizer, token_id.item()))
                input_ids = torch.cat([input_ids, token_id], dim=1)
            
            gen_text = tokenizer.decode(input_ids[0, orig_len:].tolist(), skip_special_tokens=True)
            avg_cos = sum(cos_list) / len(cos_list) if cos_list else 0
            quality = evaluate_text_quality(gen_text)
            preview = gen_text[:50].replace("\n", " ")
            results.append({
                "pair": f"{src_cat}->{tgt_cat}", "kl_w": kl_w,
                "avg_cos": avg_cos, "quality": quality, "text": gen_text[:100]
            })
            log(f"  {f'{src_cat}->{tgt_cat}':>18} {kl_w:>6.1f} {avg_cos:>9.4f} {quality:>8.3f} {preview:>50}")
    
    # Compare with P94 baseline (direct logits mixing)
    log(f"\n  === P96 vs P94(Baseline) Summary ===")
    for pair in [f"{s}->{t}" for s, t in test_pairs]:
        kl_results = [r for r in results if r["pair"] == pair]
        if kl_results:
            best_kl = max(kl_results, key=lambda r: r["quality"])
            avg_q = sum(r["quality"] for r in kl_results) / len(kl_results)
            log(f"  {pair:>18}: best_kl_w={best_kl['kl_w']:.1f}, best_quality={best_kl['quality']:.3f}, avg_quality={avg_q:.3f}")
            log(f"    best_text: {best_kl['text']}")
    return results

# ===== P97: 逐步退火策略 =====
def P97(model, tokenizer, texts, centroids, n_layers, d_model, mname):
    """Annealing strategy: alpha starts high, decays to 0. Model takes over gradually."""
    log(f"\n{'='*60}")
    log(f"P97: 逐步退火策略 [{mname}]")
    log(f"{'='*60}")
    
    test_pairs = [
        ("code", "poetry"),
        ("code", "math_sci"),
        ("chinese", "gen_en"),
    ]
    
    anneal_strategies = {
        "linear": lambda step, n: max(0.0, 0.8 * (1 - step / n)),
        "exp":    lambda step, n: 0.8 * (0.5 ** (step / (n * 0.5))),
        "cosine": lambda step, n: 0.8 * 0.5 * (1 + np.cos(np.pi * step / n)),
        "step":   lambda step, n: 0.8 if step < n * 0.3 else (0.3 if step < n * 0.6 else 0.0),
    }
    
    n_gen = 25
    
    log(f"\n  {'Pair':>18} {'Strategy':>10} {'avg_cos':>9} {'end_cos':>9} {'quality':>8} {'text_preview':>50}")
    log(f"  {'-'*18} {'-'*10} {'-'*9} {'-'*9} {'-'*8} {'-'*50}")
    
    results = []
    for src_cat, tgt_cat in test_pairs:
        src_idx = next(i for i, t in enumerate(texts) if t[1] == src_cat)
        src_text = texts[src_idx][0]
        tgt_centroid = centroids[tgt_cat]
        
        for strat_name, strat_fn in anneal_strategies.items():
            tokens_cl = []
            cos_list = []
            alpha_list = []
            input_ids = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(model.device)
            orig_len = input_ids.shape[1]
            
            for step in range(n_gen):
                alpha = strat_fn(step, n_gen)
                alpha_list.append(alpha)
                
                with torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)
                h_final = outputs.hidden_states[-1][:, -1, :].float().squeeze(0)
                natural_logits = outputs.logits[:, -1, :].float()
                
                cos_val = F.cosine_similarity(h_final.unsqueeze(0), tgt_centroid.unsqueeze(0).to(h_final.device), dim=-1).item()
                cos_list.append(cos_val)
                
                if alpha > 0.01:
                    h_ctrl = ((1 - alpha) * h_final + alpha * tgt_centroid.to(h_final.device)).unsqueeze(0)
                    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                        W = model.lm_head.weight.float()
                        ctrl_logits = (h_ctrl.to(W.device).to(W.dtype) @ W.T).float()
                        if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
                            ctrl_logits = ctrl_logits + model.lm_head.bias.float().unsqueeze(0)
                        mixed_logits = alpha * ctrl_logits.to(natural_logits.device) + (1 - alpha) * natural_logits
                    else:
                        mixed_logits = natural_logits
                else:
                    mixed_logits = natural_logits
                
                probs = F.softmax(mixed_logits / 0.8, dim=-1)
                topk_vals, topk_ids = probs.topk(50)
                topk_probs = F.softmax(topk_vals / 0.8, dim=-1)
                sampled_idx = torch.multinomial(topk_probs, 1)
                token_id = topk_ids[0, sampled_idx[0, 0]].view(1, 1)
                
                tokens_cl.append(safe_decode(tokenizer, token_id.item()))
                input_ids = torch.cat([input_ids, token_id], dim=1)
            
            gen_text = tokenizer.decode(input_ids[0, orig_len:].tolist(), skip_special_tokens=True)
            avg_cos = sum(cos_list) / len(cos_list)
            end_cos = cos_list[-1] if cos_list else 0
            quality = evaluate_text_quality(gen_text)
            preview = gen_text[:50].replace("\n", " ")
            results.append({
                "pair": f"{src_cat}->{tgt_cat}", "strategy": strat_name,
                "avg_cos": avg_cos, "end_cos": end_cos, "quality": quality,
                "alpha_decay": alpha_list, "text": gen_text[:120]
            })
            log(f"  {f'{src_cat}->{tgt_cat}':>18} {strat_name:>10} {avg_cos:>9.4f} {end_cos:>9.4f} {quality:>8.3f} {preview:>50}")
    
    # Summary: best strategy per pair
    log(f"\n  === P97 Best Strategy Summary ===")
    for pair in [f"{s}->{t}" for s, t in test_pairs]:
        pair_results = [r for r in results if r["pair"] == pair]
        if pair_results:
            best = max(pair_results, key=lambda r: r["quality"])
            log(f"  {pair:>18}: best={best['strategy']}, quality={best['quality']:.3f}, end_cos={best['end_cos']:.4f}")
            log(f"    text: {best['text']}")
    return results

# ===== P98: Prompt对比实验 =====
def P98(model, tokenizer, texts, centroids, n_layers, d_model, mname):
    """Compare: prompt-based vs h-space control vs natural generation."""
    log(f"\n{'='*60}")
    log(f"P98: Prompt对比实验 [{mname}]")
    log(f"{'='*60}")
    
    cat_prompts = {
        "poetry": "Write a beautiful poem:\n",
        "code": "Write a Python function:\n",
        "math_sci": "Explain a scientific concept:\n",
        "gen_en": "Write a paragraph in English:\n",
        "philosophy": "Discuss a philosophical idea:\n",
        "chinese": "Write about Chinese culture:\n",
    }
    
    test_configs = [
        # (base_text, src_cat, tgt_cat, prompt_text)
        ("def sort_array(arr): return sorted(arr)", "code", "poetry", cat_prompts["poetry"]),
        ("def sort_array(arr): return sorted(arr)", "code", "math_sci", cat_prompts["math_sci"]),
        ("Beijing is the capital of China.", "chinese", "gen_en", cat_prompts["gen_en"]),
        ("The Pythagorean theorem states", "math_sci", "philosophy", cat_prompts["philosophy"]),
    ]
    
    n_gen = 20
    
    log(f"\n  {'Config':>30} {'Method':>12} {'avg_cos':>9} {'quality':>8} {'text_preview':>55}")
    log(f"  {'-'*30} {'-'*12} {'-'*9} {'-'*8} {'-'*55}")
    
    results = []
    for base_text, src_cat, tgt_cat, prompt_text in test_configs:
        tgt_centroid = centroids.get(tgt_cat)
        if tgt_centroid is None:
            continue
        label = f"{src_cat[:4]}->{tgt_cat[:4]}"
        
        # Method 1: Natural generation (no control)
        input_ids = tokenizer(base_text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(model.device)
        orig_len = input_ids.shape[1]
        cos_natural = []
        for step in range(n_gen):
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][:, -1, :].float().squeeze(0)
            cos_natural.append(F.cosine_similarity(h.unsqueeze(0), tgt_centroid.unsqueeze(0).to(h.device), dim=-1).item())
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            topk_v, topk_i = probs.topk(50)
            topk_p = F.softmax(topk_v / 0.8, dim=-1)
            si = torch.multinomial(topk_p, 1)
            tid = topk_i[0, si[0, 0]].view(1, 1)
            input_ids = torch.cat([input_ids, tid], dim=1)
        text_natural = tokenizer.decode(input_ids[0, orig_len:].tolist(), skip_special_tokens=True)
        q_natural = evaluate_text_quality(text_natural)
        
        # Method 2: Prompt-based generation
        full_prompt = prompt_text + base_text
        input_ids = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=128).input_ids.to(model.device)
        orig_len = input_ids.shape[1]
        cos_prompt = []
        for step in range(n_gen):
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][:, -1, :].float().squeeze(0)
            cos_prompt.append(F.cosine_similarity(h.unsqueeze(0), tgt_centroid.unsqueeze(0).to(h.device), dim=-1).item())
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            topk_v, topk_i = probs.topk(50)
            topk_p = F.softmax(topk_v / 0.8, dim=-1)
            si = torch.multinomial(topk_p, 1)
            tid = topk_i[0, si[0, 0]].view(1, 1)
            input_ids = torch.cat([input_ids, tid], dim=1)
        text_prompt = tokenizer.decode(input_ids[0, orig_len:].tolist(), skip_special_tokens=True)
        q_prompt = evaluate_text_quality(text_prompt)
        
        # Method 3: H-space closed-loop (alpha=0.5)
        input_ids = tokenizer(base_text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(model.device)
        orig_len = input_ids.shape[1]
        cos_hctrl = []
        alpha = 0.5
        for step in range(n_gen):
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[-1][:, -1, :].float().squeeze(0)
            cos_hctrl.append(F.cosine_similarity(h.unsqueeze(0), tgt_centroid.unsqueeze(0).to(h.device), dim=-1).item())
            natural_logits = outputs.logits[:, -1, :].float()
            h_ctrl = ((1 - alpha) * h + alpha * tgt_centroid.to(h.device)).unsqueeze(0)
            if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                W = model.lm_head.weight.float()
                ctrl_logits = (h_ctrl.to(W.device).to(W.dtype) @ W.T).float()
                if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
                    ctrl_logits = ctrl_logits + model.lm_head.bias.float().unsqueeze(0)
                mixed = alpha * ctrl_logits.to(natural_logits.device) + (1 - alpha) * natural_logits
            else:
                mixed = natural_logits
            probs = F.softmax(mixed / 0.8, dim=-1)
            topk_v, topk_i = probs.topk(50)
            topk_p = F.softmax(topk_v / 0.8, dim=-1)
            si = torch.multinomial(topk_p, 1)
            tid = topk_i[0, si[0, 0]].view(1, 1)
            input_ids = torch.cat([input_ids, tid], dim=1)
        text_hctrl = tokenizer.decode(input_ids[0, orig_len:].tolist(), skip_special_tokens=True)
        q_hctrl = evaluate_text_quality(text_hctrl)
        
        for method, cos_l, text, q in [("Natural", cos_natural, text_natural, q_natural),
                                         ("Prompt", cos_prompt, text_prompt, q_prompt),
                                         ("H-Ctrl", cos_hctrl, text_hctrl, q_hctrl)]:
            avg_c = sum(cos_l) / len(cos_l)
            preview = text[:55].replace("\n", " ")
            results.append({"config": label, "method": method, "avg_cos": avg_c, "quality": q, "text": text[:120]})
            log(f"  {label:>30} {method:>12} {avg_c:>9.4f} {q:>8.3f} {preview:>55}")
    
    # Summary
    log(f"\n  === P98 Method Comparison ===")
    for config in set(r["config"] for r in results):
        cfg_results = [r for r in results if r["config"] == config]
        best = max(cfg_results, key=lambda r: r["quality"])
        log(f"  {config:>30}: best_method={best['method']}, quality={best['quality']:.3f}")
        for r in cfg_results:
            log(f"    {r['method']:>8}: cos={r['avg_cos']:.4f}, q={r['quality']:.3f}, text=\"{r['text'][:60]}\"")
    return results

# ===== P99: 偏置项语义分析 =====
def P99(model, tokenizer, texts, n_layers, d_model, mname):
    """Analyze b_l: correlation with categories, PCA visualization data."""
    log(f"\n{'='*60}")
    log(f"P99: 偏置项语义分析 [{mname}]")
    log(f"{'='*60}")
    
    # Estimate b_l for each layer by fitting delta = W @ h + b
    # b_l ≈ E[delta_h_l] - W_l @ E[h_{l-1}]
    # Or simply: b_l ≈ E[delta_h_l - W_l @ h_{l-1}] after fitting W_l
    
    n_analysis = min(60, len(texts))
    indices = list(range(0, len(texts), max(1, len(texts) // n_analysis)))[:n_analysis]
    
    log(f"  Extracting per-layer h for {n_analysis} texts...")
    all_states = []
    text_cats = []
    for idx in indices:
        states, _ = get_all_h(model, tokenizer, texts[idx][0], n_layers)
        all_states.append(states)
        text_cats.append(texts[idx][1])
    
    actual_layers = len(all_states[0]) - 1
    log(f"  Actual layers: {actual_layers}")
    
    # Compute b_l per layer
    log(f"\n  {'Layer':>6} {'||b_l||':>10} {'bias_top5_pc':>14} {'bias_cat_corr':>15} {'R2_residual':>12}")
    log(f"  {'-'*6} {'-'*10} {'-'*14} {'-'*15} {'-'*12}")
    
    categories = sorted(set(text_cats))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    
    layer_b_data = []
    for l in range(1, actual_layers + 1):
        H_prev = torch.stack([all_states[t][l-1] for t in range(n_analysis)])  # [n, d]
        H_curr = torch.stack([all_states[t][l] for t in range(n_analysis)])
        Delta = H_curr - H_prev
        
        # Fit W_l
        H_cpu = H_prev.cpu().numpy()
        D_cpu = Delta.cpu().numpy()
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(H_cpu, D_cpu)
        pred_D = ridge.predict(H_cpu)
        
        # b_l = residual mean
        residuals = D_cpu - pred_D
        b_l = torch.tensor(residuals.mean(axis=0))  # [d]
        b_norm = float(b_l.norm())
        
        # R2 of residuals (should be very low if W captures most)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((D_cpu - D_cpu.mean(axis=0))**2))
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        
        # Top 5 principal components of b_l direction
        b_np = b_l.numpy()
        pca = PCA(n_components=5)
        pca.fit(residuals.T)  # residuals: [n, d], transpose: [d, n]
        top_var_ratio = pca.explained_variance_ratio_
        
        # Category correlation: project b_l onto centroid directions
        # For each category, compute centroid of residuals
        cat_residuals = {}
        for t_idx in range(n_analysis):
            cat = text_cats[t_idx]
            if cat not in cat_residuals:
                cat_residuals[cat] = []
            cat_residuals[cat].append(residuals[t_idx])
        
        cat_means = {}
        for cat, res_list in cat_residuals.items():
            cat_means[cat] = np.mean(res_list, axis=0)
        
        # Correlation of b_l with category-specific mean residuals
        b_normalized = b_np / max(np.linalg.norm(b_np), 1e-10)
        cat_corrs = {}
        for cat, cm in cat_means.items():
            cm_norm = cm / max(np.linalg.norm(cm), 1e-10)
            corr = abs(float(np.dot(b_normalized, cm_norm)))
            cat_corrs[cat] = corr
        
        top5_cats = sorted(cat_corrs.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_str = ", ".join(f"{c}={v:.3f}" for c, v in top5_cats)
        
        layer_b_data.append({
            "layer": l, "b_norm": b_norm, "r2": r2,
            "top5_cats": top5_cats, "top_var": top_var_ratio.tolist()
        })
        
        if l <= 3 or l % max(1, actual_layers // 10) == 0 or l == actual_layers:
            log(f"  {l:>6} {b_norm:>10.3f} {top5_str[:40]:>14} {'...':>1} {r2:>15.6f}")
    
    # Summary: which layers have most category-specific bias?
    log(f"\n  === P99 Bias Category Analysis Summary ===")
    high_bias_layers = sorted(layer_b_data, key=lambda x: x["b_norm"], reverse=True)[:5]
    log(f"  Top 5 layers by bias norm:")
    for ld in high_bias_layers:
        log(f"    L{ld['layer']}: ||b||={ld['b_norm']:.3f}, top_cats={ld['top5_cats'][:3]}")
    
    # Simplified: just compute inter-layer bias direction similarity
    log(f"  Computing inter-layer bias direction cosine similarities...")
    layer_biases = []
    for l in range(1, actual_layers + 1):
        H_prev = torch.stack([all_states[t][l-1].cpu() for t in range(n_analysis)])
        H_curr = torch.stack([all_states[t][l].cpu() for t in range(n_analysis)])
        Delta = H_curr - H_prev
        mean_delta = Delta.mean(dim=0)  # approximates b_l direction
        layer_biases.append(mean_delta)
    
    log(f"\n  {'Layer':>6} {'cos_with_next':>14} {'cos_with_L1':>12} {'cos_with_last':>14}")
    for l in range(min(10, len(layer_biases))):
        cos_next = F.cosine_similarity(layer_biases[l].unsqueeze(0), layer_biases[l+1].unsqueeze(0), dim=-1).item() if l+1 < len(layer_biases) else 0
        cos_first = F.cosine_similarity(layer_biases[l].unsqueeze(0), layer_biases[0].unsqueeze(0), dim=-1).item()
        cos_last = F.cosine_similarity(layer_biases[l].unsqueeze(0), layer_biases[-1].unsqueeze(0), dim=-1).item()
        log(f"  {l+1:>6} {cos_next:>14.4f} {cos_first:>12.4f} {cos_last:>14.4f}")
    # Last few layers
    for l in range(max(10, len(layer_biases)-5), len(layer_biases)):
        cos_next = F.cosine_similarity(layer_biases[l].unsqueeze(0), layer_biases[l+1].unsqueeze(0), dim=-1).item() if l+1 < len(layer_biases) else 0
        cos_first = F.cosine_similarity(layer_biases[l].unsqueeze(0), layer_biases[0].unsqueeze(0), dim=-1).item()
        cos_last = F.cosine_similarity(layer_biases[l].unsqueeze(0), layer_biases[-1].unsqueeze(0), dim=-1).item()
        log(f"  {l+1:>6} {cos_next:>14.4f} {cos_first:>12.4f} {cos_last:>14.4f}")
    
    return layer_b_data

# ===== P100: 条件W_l分析 =====
def P100(model, tokenizer, texts, n_layers, d_model, mname):
    """Analyze whether W_l differs across categories: fit per-category W_l and compare."""
    log(f"\n{'='*60}")
    log(f"P100: 条件W_l分析 [{mname}]")
    log(f"{'='*60}")
    
    # For each layer, fit W_l separately for each category
    # Then compare: Frobenius norm of W_cat1 - W_cat2
    categories = sorted(set(t[1] for t in texts))
    
    n_per_cat = 15
    cat_texts = {}
    for cat in categories:
        cat_list = [t for t in texts if t[1] == cat]
        cat_texts[cat] = cat_list[:n_per_cat]
    
    log(f"  Categories: {len(categories)}, samples per cat: {n_per_cat}")
    
    # Extract states
    log(f"  Extracting hidden states...")
    cat_states = {}
    for cat, tlist in cat_texts.items():
        states_list = []
        for text, _ in tlist:
            states, _ = get_all_h(model, tokenizer, text, n_layers)
            states_list.append([s.cpu() for s in states])
        cat_states[cat] = states_list
    
    actual_layers = len(list(cat_states.values())[0][0]) - 1
    
    # Pick key category pairs to compare
    key_pairs = [
        ("code", "poetry"),
        ("code", "math_sci"),
        ("chinese", "gen_en"),
        ("poetry", "philosophy"),
        ("math_sci", "philosophy"),
    ]
    
    log(f"\n  {'Layer':>6} " + " ".join(f"{f'{c1[:3]}-{c2[:3]}':>14}" for c1, c2 in key_pairs))
    log(f"  {'-'*6} " + " ".join(f"{'-'*14}" for _ in key_pairs))
    
    layer_W_diffs = []
    for l in range(1, actual_layers + 1, max(1, actual_layers // 15)):
        cat_W = {}
        for cat, states_list in cat_states.items():
            H_prev = torch.stack([states_list[t][l-1] for t in range(len(states_list))])
            H_curr = torch.stack([states_list[t][l] for t in range(len(states_list))])
            Delta = H_curr - H_prev
            
            H_np = H_prev.numpy()
            D_np = Delta.numpy()
            
            ridge = Ridge(alpha=1.0)
            ridge.fit(H_np, D_np)
            cat_W[cat] = ridge.coef_.T  # [d, d]
        
        # Compute pairwise Frobenius norms of W differences
        diffs = []
        for c1, c2 in key_pairs:
            if c1 in cat_W and c2 in cat_W:
                W_diff = cat_W[c1] - cat_W[c2]
                frob = np.linalg.norm(W_diff, 'fro')
                # Normalize by the norm of W
                w_norm = max(np.linalg.norm(cat_W[c1], 'fro'), np.linalg.norm(cat_W[c2], 'fro'))
                rel_frob = frob / max(w_norm, 1e-10)
                diffs.append(rel_frob)
            else:
                diffs.append(0)
        
        layer_W_diffs.append({"layer": l, "diffs": dict(zip(key_pairs, diffs))})
        
        log_str = f"  {l:>6} "
        for d in diffs:
            log_str += f"{d:>14.4f} "
        log(log_str)
    
    # Summary
    log(f"\n  === P100 Conditional W_l Analysis Summary ===")
    if layer_W_diffs:
        # Find which layers have maximum W divergence
        for c1, c2 in key_pairs:
            pair_diffs = [(ld["layer"], ld["diffs"].get((c1, c2), 0)) for ld in layer_W_diffs]
            if pair_diffs:
                max_l, max_d = max(pair_diffs, key=lambda x: x[1])
                avg_d = sum(d for _, d in pair_diffs) / len(pair_diffs)
                log(f"  {c1[:4]}-{c2[:4]}: max_divergence={max_d:.4f} at L{max_l}, avg={avg_d:.4f}")
        
        # Find overall most differentiated layers
        log(f"\n  Top 5 most differentiated layers:")
        avg_diffs = []
        for ld in layer_W_diffs:
            avg = sum(ld["diffs"].values()) / len(ld["diffs"]) if ld["diffs"] else 0
            avg_diffs.append((ld["layer"], avg))
        for l, a in sorted(avg_diffs, key=lambda x: x[1], reverse=True)[:5]:
            log(f"    L{l}: avg_W_diff={a:.4f}")
    
    return layer_W_diffs

# ===== Main =====
def main():
    global log
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = _Path(f"d:/develop/TransformerLens-main/tests/glm5_temp/stage715_phase10_{ts}")
    log = Logger(str(log_dir), "results")
    log(f"Stage 715: Phase X — 突破文本质量瓶颈 + 偏置语义 + 条件W_l")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    texts = build_texts()
    log(f"Total texts: {len(texts)}")
    
    for mname in MODEL_ORDER:
        log(f"\n{'#'*70}")
        log(f"# Model: {mname}")
        log(f"{'#'*70}")
        t0 = time.time()
        
        model, tokenizer, n_layers, d_model = load_model(mname)
        
        # Compute centroids
        log(f"  Computing centroids...")
        centroids = compute_centroids(model, tokenizer, texts, n_layers)
        log(f"  Categories: {list(centroids.keys())}")
        
        # Run experiments
        P96(model, tokenizer, texts, centroids, n_layers, d_model, mname)
        
        P97(model, tokenizer, texts, centroids, n_layers, d_model, mname)
        
        P98(model, tokenizer, texts, centroids, n_layers, d_model, mname)
        
        P99(model, tokenizer, texts, n_layers, d_model, mname)
        
        P100(model, tokenizer, texts, n_layers, d_model, mname)
        
        elapsed = time.time() - t0
        log(f"\n  {mname} done in {elapsed:.1f}s")
        
        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    log(f"\n{'#'*70}")
    log(f"# ALL DONE")
    log(f"{'#'*70}")
    log.close()

if __name__ == "__main__":
    main()
