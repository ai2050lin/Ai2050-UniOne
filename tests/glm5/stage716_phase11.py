#!/usr/bin/env python3
"""
Stage 716: Phase XI — Perplexity评估重建 + GLM4免疫机制 + 跨模型W_l对齐
========================================================================
P101: Perplexity质量评估 — 用模型自身perplexity替代启发式cos/quality指标,
      大样本(400+文本)建立ppl-cos-质量的完整映射
P102: GLM4免疫机制分析 — 层级消融定位"自修复"层,
      attention vs skip connection贡献分离
P103: 跨模型W_l语义对齐 — CCA(Canonical Correlation Analysis)量化
      不同模型的W_l在语义空间中的对齐度

三模型串行: Qwen3 -> DS7B -> GLM4
设备: CUDA
样本: 477条文本(11类别)
"""

import sys, time, gc, json, os, math
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

log_dir = os.path.join(r"d:\develop\TransformerLens-main\tests\glm5_temp",
                       f"stage716_phase11_{datetime.now().strftime('%Y%m%d_%H%M')}")
log = Logger(log_dir, "results")
log(f"Log dir: {log_dir}")

# ===== Model Config =====
MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
}
MODEL_ORDER = ["qwen3", "deepseek7b", "glm4"]

# ===== Text dataset (reuse from stage715, 477 texts, 11 categories) =====
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

def evaluate_text_quality(text):
    """Simple heuristic for text quality."""
    words = text.split()
    if len(words) < 3:
        return 0.0
    unique_words = set(w.lower() for w in words if w.isalpha())
    word_ratio = min(1.0, len(unique_words) / max(1, len(words)))
    alpha_ratio = sum(1 for w in words if w.isalpha()) / max(1, len(words))
    return 0.6 * word_ratio + 0.4 * alpha_ratio

def compute_perplexity(model, tokenizer, text, max_length=128):
    """Compute perplexity of a text using the model itself."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    # loss is average negative log-likelihood over tokens
    avg_loss = outputs.loss.item()
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return ppl, avg_loss

def compute_centroids(model, tokenizer, texts, n_layers, n_samples=40):
    """Compute per-category centroids from hidden states."""
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

def get_unembed(model):
    """Get unembedding weight and bias as float32 CPU tensors."""
    if hasattr(model, 'lm_head'):
        um = model.lm_head
    elif hasattr(model, 'get_output_embeddings'):
        um = model.get_output_embeddings()
    else:
        return None, None
    w = um.weight.detach().to(torch.float32)  # keep on same device
    b = um.bias.detach().to(torch.float32) if (hasattr(um, 'bias') and um.bias is not None) else None
    return w, b

def generate_with_h_control(model, tokenizer, prompt, target_dir, alpha=0.5, max_new_tokens=20):
    """Generate text with h-space direction control. All ops in native model dtype."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(model.device)
    generated_ids = inputs["input_ids"].clone()
    uw, ub = get_unembed(model)
    dev = model.device
    dtype = next(model.parameters()).dtype
    
    td = target_dir.to(dtype).to(dev)
    
    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated_ids, output_hidden_states=True, use_cache=True)
            h_final = outputs.hidden_states[-1][:, -1, :]  # native dtype
            logits = outputs.logits[:, -1, :]  # native dtype
        
        if uw is not None:
            h_f32 = h_final.float()
            shifted_h = h_f32 + td.float()
            target_logits = shifted_h @ uw.T
            if ub is not None:
                target_logits = target_logits + ub
            mixed_logits = (1 - alpha) * logits.float() + alpha * target_logits
        else:
            mixed_logits = logits.float()
        
        next_token = torch.argmax(mixed_logits, dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated_ids = torch.cat([generated_ids, next_token.to(dev)], dim=1)
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def generate_natural(model, tokenizer, prompt, max_new_tokens=20):
    """Generate text naturally without control."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ===== P101: Perplexity-based Quality Assessment =====
def run_p101(model, tokenizer, mname, n_layers, texts):
    """P101: Build ppl-cos-quality mapping with large samples."""
    log(f"\n{'='*60}")
    log(f"  P101: Perplexity-based Quality Assessment ({mname})")
    log(f"{'='*60}")
    
    # 1. Compute centroids for direction control
    log("\n  [1/4] Computing centroids...")
    centroids = compute_centroids(model, tokenizer, texts, n_layers, n_samples=40)
    categories = list(centroids.keys())
    
    # 2. Generate text with different alpha values and measure all metrics
    log("\n  [2/4] Generating controlled text with multiple alpha values...")
    test_texts = texts[:80]  # Large sample
    directions_to_test = []
    for cat in categories[:3]:  # Test 3 target directions
        for src_cat in categories[:3]:
            if src_cat != cat:
                directions_to_test.append((src_cat, cat))
    directions_to_test = directions_to_test[:6]  # Limit to 6 directions
    
    alphas = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    results = []
    
    for src_cat, tgt_cat in directions_to_test:
        if tgt_cat not in centroids or src_cat not in centroids:
            continue
        target_dir = centroids[tgt_cat] - centroids[src_cat]
        target_dir = F.normalize(target_dir.float(), dim=0)
        
        for alpha in alphas:
            n_test = min(10, len([t for t in test_texts if t[1] == src_cat]))
            src_samples = [t for t in test_texts if t[1] == src_cat][:n_test]
            
            for text, _ in src_samples:
                try:
                    gen = generate_with_h_control(model, tokenizer, text, target_dir, alpha=alpha, max_new_tokens=20)
                    if not gen or len(gen.split()) < 3:
                        continue
                    ppl, _ = compute_perplexity(model, tokenizer, gen)
                    quality = evaluate_text_quality(gen)
                    
                    # Get cos similarity of generated h_final with target centroid
                    gen_states, _ = get_all_h(model, tokenizer, gen, n_layers)
                    gen_h = gen_states[-1].float()
                    cos_target = F.cosine_similarity(gen_h.unsqueeze(0), 
                                                      centroids[tgt_cat].unsqueeze(0)).item()
                    
                    results.append({
                        "src": src_cat, "tgt": tgt_cat, "alpha": alpha,
                        "ppl": ppl, "quality": quality,
                        "cos_target": cos_target,
                        "gen_len": len(gen.split()),
                        "gen_text": gen[:100],
                    })
                except Exception as e:
                    log(f"    [WARN] {src_cat}->{tgt_cat} alpha={alpha}: {e}")
                    continue
    
    log(f"  Collected {len(results)} data points")
    
    # 3. Correlation analysis
    log("\n  [3/4] Correlation analysis: ppl vs quality vs cos...")
    if len(results) > 10:
        ppls = [r["ppl"] for r in results if r["ppl"] < 1000]
        qualities = [r["quality"] for r in results if r["ppl"] < 1000]
        cos_targets = [r["cos_target"] for r in results if r["ppl"] < 1000]
        
        if len(ppls) > 5:
            # ppl-quality correlation
            corr_ppl_q = np.corrcoef(ppls, qualities)[0, 1] if len(set(qualities)) > 1 else 0
            # cos-quality correlation
            corr_cos_q = np.corrcoef(cos_targets, qualities)[0, 1] if len(set(cos_targets)) > 1 else 0
            # ppl-cos correlation
            corr_ppl_cos = np.corrcoef(ppls, cos_targets)[0, 1] if len(set(ppls)) > 1 else 0
            
            log(f"  ppl vs quality  : r = {corr_ppl_q:.4f}")
            log(f"  cos vs quality  : r = {corr_cos_q:.4f}")
            log(f"  ppl vs cos      : r = {corr_ppl_cos:.4f}")
            
            # By alpha
            log("\n  By alpha value:")
            for alpha in alphas:
                alpha_data = [r for r in results if r["alpha"] == alpha and r["ppl"] < 1000]
                if len(alpha_data) > 2:
                    avg_ppl = np.mean([r["ppl"] for r in alpha_data])
                    avg_q = np.mean([r["quality"] for r in alpha_data])
                    avg_cos = np.mean([r["cos_target"] for r in alpha_data])
                    log(f"    alpha={alpha:.1f}: avg_ppl={avg_ppl:.1f}, avg_quality={avg_q:.3f}, avg_cos={avg_cos:.3f}")
    
    # 4. Key question: is ppl a better indicator than cos?
    log("\n  [4/4] ppl vs cos as quality predictor:")
    if len(results) > 10:
        valid = [r for r in results if r["ppl"] < 1000]
        if len(valid) > 10:
            # Which has stronger correlation with quality?
            corr_ppl_q = abs(np.corrcoef([r["ppl"] for r in valid], [r["quality"] for r in valid])[0, 1])
            corr_cos_q = abs(np.corrcoef([r["cos_target"] for r in valid], [r["quality"] for r in valid])[0, 1])
            
            better = "perplexity" if corr_ppl_q > corr_cos_q else "cos"
            log(f"  |ppl-quality| = {corr_ppl_q:.4f}, |cos-quality| = {corr_cos_q:.4f}")
            log(f"  Better predictor: {better}")
            
            # Top-5 lowest ppl texts (best quality by ppl)
            sorted_by_ppl = sorted(valid, key=lambda x: x["ppl"])[:5]
            log(f"\n  Top-5 lowest ppl (best by model):")
            for i, r in enumerate(sorted_by_ppl):
                log(f"    {i+1}. ppl={r['ppl']:.1f}, q={r['quality']:.3f}, cos={r['cos_target']:.3f}, alpha={r['alpha']}")
                log(f"       {r['gen_text']}")
            
            # Top-5 highest quality texts
            sorted_by_q = sorted(valid, key=lambda x: -x["quality"])[:5]
            log(f"\n  Top-5 highest quality (by heuristic):")
            for i, r in enumerate(sorted_by_q):
                log(f"    {i+1}. q={r['quality']:.3f}, ppl={r['ppl']:.1f}, cos={r['cos_target']:.3f}, alpha={r['alpha']}")
                log(f"       {r['gen_text']}")
    
    return results


# ===== P102: GLM4 Immunity Mechanism Analysis =====
def run_p102(model, tokenizer, mname, n_layers, d_model, texts):
    """P102: Identify which layer(s) cause GLM4's 'self-repair' after h-space manipulation."""
    log(f"\n{'='*60}")
    log(f"  P102: GLM4 Immunity Mechanism Analysis ({mname})")
    log(f"{'='*60}")
    
    # Only detailed analysis for GLM4; quick check for others
    is_glm4 = "glm4" in mname
    
    # 1. Perturb h at different layers and see how fast it recovers
    log("\n  [1/3] Layer-by-layer perturbation recovery test...")
    test_sample = texts[0][0]
    states_orig, inputs_orig = get_all_h(model, tokenizer, test_sample, n_layers)
    h_orig_final = states_orig[-1].float().clone()
    
    # Get a target direction to inject
    centroids = compute_centroids(model, tokenizer, texts, n_layers, n_samples=20)
    cats = list(centroids.keys())
    if len(cats) < 2:
        log("  Not enough categories, skipping")
        return {}
    
    target_dir = centroids[cats[1]] - centroids[cats[0]]
    target_dir = F.normalize(target_dir.float(), dim=0)
    
    # Inject perturbation at each layer and measure recovery
    recovery_results = {}
    layers_to_test = list(range(1, n_layers, max(1, n_layers // 10))) + [n_layers - 1]
    
    for inj_layer in layers_to_test:
        cos_after = []
        
        for trial in range(3):  # 3 trials per layer
            try:
                states, _ = get_all_h(model, tokenizer, test_sample, n_layers)
                inj_h = states[inj_layer].float()
                final_h = states[-1].float()
                
                # Perturb at injection layer
                perturbed = inj_h + 0.5 * target_dir.float()
                
                # Cosine between perturbed final and target centroid
                # (approximate: how much would perturbation change the final direction)
                cos_orig_target = F.cosine_similarity(
                    final_h.unsqueeze(0), centroids[cats[1]].float().unsqueeze(0)
                ).item()
                # Approximate perturbed final by adding direction to final
                pert_final = final_h + 0.5 * target_dir.float()
                cos_pert_target = F.cosine_similarity(
                    pert_final.unsqueeze(0), centroids[cats[1]].float().unsqueeze(0)
                ).item()
                cos_after.append(cos_pert_target - cos_orig_target)
            except Exception as e:
                log(f"    [WARN] L{inj_layer} trial {trial}: {e}")
                continue
        
        if cos_after:
            recovery_results[inj_layer] = {
                "avg_delta_cos": np.mean(cos_after),
            }
            log(f"    L{inj_layer}: avg_delta_cos={np.mean(cos_after):.4f}")
    
    # 2. Step-by-step generation with perturbation tracking
    log("\n  [2/3] Step-by-step generation perturbation tracking...")
    prompt = texts[0][0]
    
    # Get natural generation hidden states step by step
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(model.device)
    
    gen_states = []
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        initial_h = out.hidden_states[-1][:, -1, :].float()
    
    # Now generate step by step, perturb initial h, track cos at each step
    perturbed_h = initial_h.float() + 0.5 * target_dir.float()
    gen_ids = inputs["input_ids"].clone()
    uw, ub = get_unembed(model)
    
    step_cos = []
    for step in range(15):
        with torch.no_grad():
            step_out = model(input_ids=gen_ids[:, -1:], output_hidden_states=True)
            nat_h = step_out.hidden_states[-1][:, -1, :].float().cpu()
            nat_logits = step_out.logits[:, -1, :].float().cpu()
            
            if uw is not None:
                pert_h_cpu = perturbed_h.float().cpu()
                pert_logits = pert_h_cpu.unsqueeze(0) @ uw.T
                if ub is not None:
                    pert_logits = pert_logits + ub.unsqueeze(0)
                mixed = 0.5 * nat_logits + 0.5 * pert_logits
            else:
                mixed = nat_logits
            
            next_tok = torch.argmax(mixed, dim=-1, keepdim=True)
            
            if next_tok.item() == tokenizer.eos_token_id:
                break
            
            gen_ids = torch.cat([gen_ids, next_tok.to(model.device)], dim=1)
            
            cos_target = F.cosine_similarity(perturbed_h.float().cpu().unsqueeze(0), 
                                              centroids[cats[1]].float().cpu().unsqueeze(0)).item()
            step_cos.append({"step": step, "cos_target": cos_target})
            
            # Decay perturbed_h towards natural
            perturbed_h = 0.8 * perturbed_h + 0.2 * nat_h.to(perturbed_h.device)
    
    log("  Step-by-step cos tracking (perturbed cos to target):")
    for s in step_cos:
        log(f"    step {s['step']:2d}: cos_target={s['cos_target']:.4f}")
    
    # 3. Attention pattern analysis
    log("\n  [3/3] Attention self-repair analysis...")
    if is_glm4:
        log("  GLM4 detailed attention analysis:")
        try:
            # Enable attention output for GLM4
            orig_config = model.config.output_attentions if hasattr(model.config, 'output_attentions') else None
            model.config.output_attentions = True
            with torch.no_grad():
                out_nat = model(**inputs, output_attentions=True, output_hidden_states=True)
                nat_attentions = getattr(out_nat, 'attentions', None)
            
            if nat_attentions is not None:
                n_attn_layers = len(nat_attentions)
                log(f"  Attention layers: {n_attn_layers}")
                
                # Analyze last-token attention distribution
                for layer_idx in [0, n_attn_layers//4, n_attn_layers//2, 3*n_attn_layers//4, n_attn_layers-1]:
                    try:
                        attn = nat_attentions[layer_idx][0].float()  # (heads, seq, seq)
                        last_tok_attn = attn[:, -1, :].mean(dim=0)  # avg over heads
                        
                        probs = F.softmax(last_tok_attn, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                        max_attn_pos = probs.argmax().item()
                        max_attn_val = probs[max_attn_pos].item()
                        
                        log(f"    Layer {layer_idx}: entropy={entropy:.3f}, "
                            f"max_attn_pos={max_attn_pos}, max_attn_val={max_attn_val:.3f}")
                    except Exception as e:
                        log(f"    Layer {layer_idx}: attn analysis failed: {e}")
            else:
                log("  No attention output available")
        except Exception as e:
            log(f"  Attention analysis failed: {e}")
    else:
        log("  Quick check for non-GLM4 models:")
        with torch.no_grad():
            out_nat = model(**inputs, output_attentions=True, output_hidden_states=True)
            if out_nat.attentions is not None:
                n_attn_layers = len(out_nat.attentions)
                try:
                    attn = out_nat.attentions[-1][0].float()
                    last_tok_attn = attn[:, -1, :].mean(dim=0)
                    probs = F.softmax(last_tok_attn, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                    log(f"  Last layer attn entropy: {entropy:.3f} (GLM4 typically has lower entropy = more focused)")
                except Exception as e:
                    log(f"  Attention analysis failed: {e}")
    
    return {"recovery": recovery_results, "step_cos": step_cos}


# ===== P103: Cross-model W_l Semantic Alignment (CCA) — Sequential version =====
def run_p103_sequential(texts):
    """P103: CCA alignment of W_l across models. Load one model at a time."""
    log(f"\n{'='*60}")
    log(f"  P103: Cross-model W_l Semantic Alignment (CCA)")
    log(f"{'='*60}")
    
    from sklearn.linear_model import Ridge
    from sklearn.cross_decomposition import CCA
    
    # Phase 1: Extract W_l for each model (load one at a time)
    log("\n  [1/3] Extracting W_l for each model...")
    model_wl = {}
    model_h_finals = {}  # Store h_final projections per model
    
    for mname in MODEL_ORDER:
        log(f"\n  Loading {mname} for W_l extraction...")
        model, tokenizer, n_layers, d_model = load_model(mname)
        
        wls = {}
        n_extract = 60
        sample_indices = list(range(0, len(texts), max(1, len(texts) // n_extract)))[:n_extract]
        
        for layer_idx in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
            H_prev = []
            H_curr = []
            
            for idx in sample_indices:
                try:
                    states, _ = get_all_h(model, tokenizer, texts[idx][0], n_layers)
                    if layer_idx < len(states) - 1:
                        H_prev.append(states[layer_idx].float().cpu().numpy())
                        H_curr.append(states[layer_idx + 1].float().cpu().numpy())
                except:
                    continue
            
            if len(H_prev) > 5:
                H_prev = np.array(H_prev)
                H_curr = np.array(H_curr)
                reg = Ridge(alpha=1.0)
                reg.fit(H_prev, H_curr)
                W = reg.coef_
                
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
                top_k = min(20, len(S))
                semantic_subspace = Vt[:top_k].T
                
                wls[layer_idx] = {
                    "W": W, "top_sv": S[:top_k],
                    "semantic_subspace": semantic_subspace,
                    "R2": reg.score(H_prev, H_curr),
                    "rank_approx": np.sum(S > 0.1 * S[0]),
                }
                log(f"    L{layer_idx}: R2={wls[layer_idx]['R2']:.4f}, "
                    f"rank_approx={wls[layer_idx]['rank_approx']}")
        
        model_wl[mname] = wls
        
        # Also compute h_final projections for CCA
        log(f"    Computing h_final projections...")
        proj_data = []
        proj_indices = list(range(0, len(texts), max(1, len(texts) // 40)))[:40]
        
        for layer_idx in [n_layers//2, n_layers-1]:
            if layer_idx in wls:
                sub = wls[layer_idx]["semantic_subspace"]
                k = min(sub.shape[1], 10)
                projs = []
                cats = []
                for idx in proj_indices:
                    try:
                        states, _ = get_all_h(model, tokenizer, texts[idx][0], n_layers)
                        h = states[-1].float().cpu().numpy()
                        proj = h @ sub[:, :k]
                        projs.append(proj)
                        cats.append(texts[idx][1])
                    except:
                        continue
                if projs:
                    proj_data.append({
                        "layer": layer_idx,
                        "projs": np.array(projs),
                        "cats": cats,
                        "subspace": sub[:, :k],
                    })
        
        model_h_finals[mname] = proj_data
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Phase 2: CCA between model pairs
    log("\n  [2/3] CCA between model pairs...")
    cca_results = {}
    
    for i in range(len(MODEL_ORDER)):
        for j in range(i + 1, len(MODEL_ORDER)):
            m1, m2 = MODEL_ORDER[i], MODEL_ORDER[j]
            projs1 = model_h_finals.get(m1, [])
            projs2 = model_h_finals.get(m2, [])
            
            log(f"\n  {m1} vs {m2}:")
            pair_results = {}
            
            # Match by layer position
            for p1 in projs1:
                for p2 in projs2:
                    if p1["layer"] != p2["layer"]:
                        continue
                    
                    k = min(p1["projs"].shape[1], p2["projs"].shape[1])
                    n_common = min(p1["projs"].shape[0], p2["projs"].shape[0])
                    
                    if n_common < 15:
                        continue
                    
                    X1 = p1["projs"][:n_common]
                    X2 = p2["projs"][:n_common]
                    
                    # Standardize
                    X1 = (X1 - X1.mean(axis=0)) / (X1.std(axis=0) + 1e-8)
                    X2 = (X2 - X2.mean(axis=0)) / (X2.std(axis=0) + 1e-8)
                    
                    n_components = min(k, n_common - 2)
                    if n_components > 0:
                        try:
                            cca = CCA(n_components=n_components)
                            X1_c, X2_c = cca.fit_transform(X1, X2)
                            
                            corrs = [np.corrcoef(X1_c[:, c], X2_c[:, c])[0, 1] for c in range(n_components)]
                            avg_corr = np.mean([abs(c) for c in corrs])
                            max_corr = max(abs(c) for c in corrs)
                            
                            pair_results[p1["layer"]] = {
                                "avg_corr": float(avg_corr),
                                "max_corr": float(max_corr),
                                "corrs": [float(c) for c in corrs[:5]],
                            }
                            log(f"    L{p1['layer']}: avg_CCA={avg_corr:.4f}, "
                                f"max_CCA={max_corr:.4f}, "
                                f"top3=[{corrs[0]:.3f}, {corrs[1]:.3f}, {corrs[2]:.3f}]")
                        except Exception as e:
                            log(f"    L{p1['layer']}: CCA failed: {e}")
            
            cca_results[f"{m1}_vs_{m2}"] = pair_results
    
    # Phase 3: Summary
    log("\n  [3/3] Summary:")
    for pair_name, pair_data in cca_results.items():
        if pair_data:
            avg_all = np.mean([v["avg_corr"] for v in pair_data.values()])
            max_all = np.mean([v["max_corr"] for v in pair_data.values()])
            log(f"  {pair_name}: avg_CCA={avg_all:.4f}, avg_max_CCA={max_all:.4f}")
    
    return cca_results


# ===== Main =====
def main():
    log("=" * 70)
    log("  Stage 716: Phase XI — Perplexity评估 + GLM4免疫 + 跨模型W_l对齐")
    log("=" * 70)
    
    all_texts = build_texts()
    categories = sorted(set(t[1] for t in all_texts))
    log(f"Total texts: {len(all_texts)}, Categories: {len(categories)} - {categories}")
    
    models_data = {}
    all_results = {}
    
    for mname in MODEL_ORDER:
        log(f"\n{'#'*70}")
        log(f"  Processing: {mname}")
        log(f"{'#'*70}")
        t0 = time.time()
        
        # Load model
        model, tokenizer, n_layers, d_model = load_model(mname)
        models_data[mname] = (model, tokenizer, n_layers, d_model)
        
        # P101: Perplexity assessment
        try:
            p101_results = run_p101(model, tokenizer, mname, n_layers, all_texts)
            all_results[f"{mname}_P101"] = {
                "n_data_points": len(p101_results),
                "summary": "see log"
            }
        except Exception as e:
            log(f"  P101 failed: {e}")
            import traceback; traceback.print_exc()
        
        # P102: GLM4 immunity
        try:
            p102_results = run_p102(model, tokenizer, mname, n_layers, d_model, all_texts)
            all_results[f"{mname}_P102"] = {"recovery_layers": list(p102_results.get("recovery", {}).keys())}
        except Exception as e:
            log(f"  P102 failed: {e}")
            import traceback; traceback.print_exc()
        
        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        elapsed = time.time() - t0
        log(f"  {mname} done in {elapsed:.0f}s")
    
    # P103: Cross-model W_l alignment (load models one at a time to save GPU)
    log(f"\n{'#'*70}")
    log(f"  P103: Cross-model W_l Semantic Alignment (CCA)")
    log(f"{'#'*70}")
    
    t0 = time.time()
    
    try:
        p103_results = run_p103_sequential(all_texts)
        all_results["P103"] = "completed"
    except Exception as e:
        log(f"  P103 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["P103"] = f"failed: {e}"
    
    elapsed = time.time() - t0
    log(f"  P103 done in {elapsed:.0f}s")
    
    # Save results summary
    with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    log(f"\n{'='*70}")
    log(f"  Stage 716 Phase XI COMPLETE")
    log(f"  Results saved to: {log_dir}")
    log(f"{'='*70}")
    
    log.close()

if __name__ == "__main__":
    main()
