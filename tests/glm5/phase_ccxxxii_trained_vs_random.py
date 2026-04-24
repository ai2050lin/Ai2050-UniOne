"""
Phase CCXXXII: 频分复用的来源 — 训练vs架构
================================================================
核心目标:
1. 对比训练模型vs随机初始化模型的特征PC1正交性
2. 验证频分复用(CDMA-like)是训练产生的还是架构固有的
3. 对比残差旋转分量(Δh⊥h比例)在训练vs随机模型中的差异
4. 分析PC1解释方差在训练vs随机模型中的差异

关键假设:
  如果频分复用是训练产生的:
    → 训练模型: 特征PC1高度正交(|cos|<0.10)
    → 随机模型: 特征PC1近似随机正交(|cos|≈1/√d)
  如果频分复用是架构固有的:
    → 训练模型和随机模型都高度正交
"""

import argparse
import json
import os
import time
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch

MODEL_CONFIGS = {
    'qwen3': {
        'name': 'Qwen3-4B',
        'path': 'D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',
        'n_layers': 36,
        'd_model': 2560,
        'dtype': 'bf16',
    },
    'deepseek7b': {
        'name': 'DeepSeek-R1-Distill-Qwen-7B',
        'path': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'n_layers': 28,
        'd_model': 3584,
        'dtype': '8bit',
    },
    'glm4': {
        'name': 'GLM4-9B-Chat',
        'path': 'D:/develop/model/hub/models--zai-org--glm-4-9b-chat-hf/download',
        'n_layers': 40,
        'd_model': 4096,
        'dtype': '8bit',
    }
}

# 5个核心特征
FEATURES = {
    'tense': {
        'type': 'SYN',
        'pairs': [
            ("The cat sleeps on the mat", "The cat slept on the mat"),
            ("She walks to school every day", "She walked to school every day"),
            ("He runs in the park", "He ran in the park"),
            ("They play football on Sundays", "They played football on Sundays"),
            ("The dog barks at strangers", "The dog barked at strangers"),
            ("We cook dinner together", "We cooked dinner together"),
            ("She sings beautifully", "She sang beautifully"),
            ("The baby cries loudly", "The baby cried loudly"),
            ("Birds fly south in winter", "Birds flew south in winter"),
            ("The teacher explains the lesson", "The teacher explained the lesson"),
            ("The wind blows gently", "The wind blew gently"),
            ("Children laugh and play", "Children laughed and played"),
            ("The river flows to the sea", "The river flowed to the sea"),
            ("She paints beautiful pictures", "She painted beautiful pictures"),
            ("The sun rises in the east", "The sun rose in the east"),
            ("He writes a letter home", "He wrote a letter home"),
            ("The bird builds a nest", "The bird built a nest"),
            ("She drives to work early", "She drove to work early"),
            ("The rain falls softly", "The rain fell softly"),
            ("He reads the newspaper", "He read the newspaper"),
            ("The bell rings loudly", "The bell rang loudly"),
            ("They dance at the party", "They danced at the party"),
            ("The train arrives on time", "The train arrived on time"),
            ("She wears a red dress", "She wore a red dress"),
            ("The garden grows quickly", "The garden grew quickly"),
            ("He speaks three languages", "He spoke three languages"),
            ("The water boils rapidly", "The water boiled rapidly"),
            ("She opens the window", "She opened the window"),
            ("The storm destroys the house", "The storm destroyed the house"),
            ("They travel abroad frequently", "They traveled abroad frequently"),
            ("The snow covers the ground", "The snow covered the ground"),
            ("She bakes a chocolate cake", "She baked a chocolate cake"),
            ("He fixes the broken chair", "He fixed the broken chair"),
            ("The clock strikes midnight", "The clock struck midnight"),
            ("They celebrate the victory", "They celebrated the victory"),
            ("The horse gallops across", "The horse galloped across"),
            ("She types the document", "She typed the document"),
            ("The machine operates smoothly", "The machine operated smoothly"),
            ("He delivers the package", "He delivered the package"),
            ("The ship sails across the ocean", "The ship sailed across the ocean"),
            ("The fire burns brightly", "The fire burned brightly"),
            ("She studies mathematics", "She studied mathematics"),
            ("The plane takes off early", "The plane took off early"),
            ("He draws a landscape", "He drew a landscape"),
            ("The music plays softly", "The music played softly"),
            ("They harvest the crops", "They harvested the crops"),
            ("The ice melts in spring", "The ice melted in spring"),
            ("She teaches the children", "She taught the children"),
            ("The frog jumps into the pond", "The frog jumped into the pond"),
            ("He drives carefully", "He drove carefully"),
            ("The stars shine brightly", "The stars shone brightly"),
            ("She chooses the blue one", "She chose the blue one"),
            ("The leaves fall from the tree", "The leaves fell from the tree"),
            ("He breaks the record", "He broke the record"),
            ("The dog catches the ball", "The dog caught the ball"),
            ("She spends the afternoon reading", "She spent the afternoon reading"),
            ("The boat crosses the lake", "The boat crossed the lake"),
            ("He brings the supplies", "He brought the supplies"),
            ("The wind carries the seeds", "The wind carried the seeds"),
            ("She holds the baby gently", "She held the baby gently"),
            ("The king rules the kingdom", "The king ruled the kingdom"),
            ("He stands by the door", "He stood by the door"),
            ("The snake bites the man", "The snake bit the man"),
            ("She feels the cold wind", "She felt the cold wind"),
            ("The light shines through", "The light shone through"),
            ("He throws the ball far", "He threw the ball far"),
            ("The fish swims upstream", "The fish swam upstream"),
            ("She leaves the room quietly", "She left the room quietly"),
            ("The thunder shakes the house", "The thunder shook the house"),
            ("He meets the president", "He met the president"),
            ("The flower blooms in spring", "The flower bloomed in spring"),
            ("She wins the prize", "She won the prize"),
            ("The dog hides under the bed", "The dog hid under the bed"),
            ("He tells the truth", "He told the truth"),
            ("The moon rises slowly", "The moon rose slowly"),
            ("She feeds the hungry cat", "She fed the hungry cat"),
            ("The car stops suddenly", "The car stopped suddenly"),
            ("He hangs the picture", "He hung the picture"),
            ("The bird sings a song", "The bird sang a song"),
            ("She keeps the secret", "She kept the secret"),
            ("The boy swings high", "The boy swung high"),
        ]
    },
    'polarity': {
        'type': 'SYN',
        'pairs': [
            ("She is happy about the result", "She is not happy about the result"),
            ("The project was successful", "The project was not successful"),
            ("He likes the new design", "He does not like the new design"),
            ("They enjoyed the concert", "They did not enjoy the concert"),
            ("The food tastes good", "The food does not taste good"),
            ("She can solve the problem", "She cannot solve the problem"),
            ("The weather is pleasant today", "The weather is not pleasant today"),
            ("He will attend the meeting", "He will not attend the meeting"),
            ("They have finished the work", "They have not finished the work"),
            ("The movie was entertaining", "The movie was not entertaining"),
            ("She knows the answer", "She does not know the answer"),
            ("The team won the match", "The team did not win the match"),
            ("He believes the story", "He does not believe the story"),
            ("They found the solution", "They did not find the solution"),
            ("The restaurant is open", "The restaurant is not open"),
            ("She understands the concept", "She does not understand the concept"),
            ("The machine works properly", "The machine does not work properly"),
            ("He remembers the event", "He does not remember the event"),
            ("They support the idea", "They do not support the idea"),
            ("The book is interesting", "The book is not interesting"),
            ("She wants to go home", "She does not want to go home"),
            ("The plan is feasible", "The plan is not feasible"),
            ("He trusts his colleagues", "He does not trust his colleagues"),
            ("They need more time", "They do not need more time"),
            ("The bridge is safe", "The bridge is not safe"),
            ("She accepts the offer", "She does not accept the offer"),
            ("The proposal makes sense", "The proposal does not make sense"),
            ("He agrees with the decision", "He does not agree with the decision"),
            ("They care about the environment", "They do not care about the environment"),
            ("The method is effective", "The method is not effective"),
            ("She enjoys reading books", "She does not enjoy reading books"),
            ("The product is reliable", "The product is not reliable"),
            ("He appreciates the help", "He does not appreciate the help"),
            ("They value the feedback", "They do not value the feedback"),
            ("The system is stable", "The system is not stable"),
            ("She follows the rules", "She does not follow the rules"),
            ("The evidence is convincing", "The evidence is not convincing"),
            ("He recognizes the voice", "He does not recognize the voice"),
            ("They respect the tradition", "They do not respect the tradition"),
            ("The answer is correct", "The answer is not correct"),
            ("She prefers tea over coffee", "She does not prefer tea over coffee"),
            ("The report is accurate", "The report is not accurate"),
            ("He deserves the award", "He does not deserve the award"),
            ("They own a house", "They do not own a house"),
            ("The suggestion is helpful", "The suggestion is not helpful"),
            ("She speaks German", "She does not speak German"),
            ("The experiment was successful", "The experiment was not successful"),
            ("He manages the team well", "He does not manage the team well"),
            ("They share the profits", "They do not share the profits"),
            ("The room is spacious", "The room is not spacious"),
            ("She considers the option", "She does not consider the option"),
            ("The dog is friendly", "The dog is not friendly"),
            ("He admits the mistake", "He does not admit the mistake"),
            ("They produce quality goods", "They do not produce quality goods"),
            ("The water is clean", "The water is not clean"),
            ("She expects an apology", "She does not expect an apology"),
            ("The road is passable", "The road is not passable"),
            ("He enjoys the view", "He does not enjoy the view"),
            ("They practice yoga daily", "They do not practice yoga daily"),
            ("The chair is comfortable", "The chair is not comfortable"),
            ("She represents the company", "She does not represent the company"),
            ("The food is fresh", "The food is not fresh"),
            ("He completes the task", "He does not complete the task"),
            ("They include everyone", "They do not include everyone"),
            ("The water is warm enough", "The water is not warm enough"),
            ("She believes in fairness", "She does not believe in fairness"),
            ("The engine runs smoothly", "The engine does not run smoothly"),
            ("He wears glasses", "He does not wear glasses"),
            ("They sell organic produce", "They do not sell organic produce"),
            ("The sky is clear today", "The sky is not clear today"),
            ("She likes classical music", "She does not like classical music"),
            ("The building is tall", "The building is not tall"),
            ("He drives a car", "He does not drive a car"),
            ("They speak English", "They do not speak English"),
            ("The soup is hot", "The soup is not hot"),
            ("She plays the piano", "She does not play the piano"),
            ("The door is locked", "The door is not locked"),
            ("He watches television", "He does not watch television"),
        ]
    },
    'voice': {
        'type': 'SYN',
        'pairs': [
            ("The cat chased the mouse", "The mouse was chased by the cat"),
            ("She wrote the report", "The report was written by her"),
            ("He fixed the car", "The car was fixed by him"),
            ("They built the house", "The house was built by them"),
            ("The teacher praised the student", "The student was praised by the teacher"),
            ("She cooked the meal", "The meal was cooked by her"),
            ("He painted the fence", "The fence was painted by him"),
            ("They discovered the treasure", "The treasure was discovered by them"),
            ("The chef prepared the dish", "The dish was prepared by the chef"),
            ("She delivered the speech", "The speech was delivered by her"),
            ("He solved the puzzle", "The puzzle was solved by him"),
            ("They cleaned the room", "The room was cleaned by them"),
            ("The company launched the product", "The product was launched by the company"),
            ("She wrote the novel", "The novel was written by her"),
            ("He designed the building", "The building was designed by him"),
            ("They organized the event", "The event was organized by them"),
            ("The scientist conducted the experiment", "The experiment was conducted by the scientist"),
            ("She directed the film", "The film was directed by her"),
            ("He composed the symphony", "The symphony was composed by him"),
            ("They published the article", "The article was published by them"),
            ("The artist painted the portrait", "The portrait was painted by the artist"),
            ("She recorded the song", "The song was recorded by her"),
            ("He programmed the software", "The software was programmed by him"),
            ("They manufactured the device", "The device was manufactured by them"),
            ("The author wrote the book", "The book was written by the author"),
            ("She managed the project", "The project was managed by her"),
            ("He repaired the roof", "The roof was repaired by him"),
            ("They developed the theory", "The theory was developed by them"),
            ("The baker made the bread", "The bread was made by the baker"),
            ("She translated the document", "The document was translated by her"),
            ("He directed the play", "The play was directed by him"),
            ("They renovated the kitchen", "The kitchen was renovated by them"),
            ("The gardener planted the tree", "The tree was planted by the gardener"),
            ("She performed the surgery", "The surgery was performed by her"),
            ("He invented the machine", "The machine was invented by him"),
            ("They adopted the policy", "The policy was adopted by them"),
            ("The judge dismissed the case", "The case was dismissed by the judge"),
            ("She edited the manuscript", "The manuscript was edited by her"),
            ("He chaired the meeting", "The meeting was chaired by him"),
            ("They approved the budget", "The budget was approved by them"),
            ("The nurse administered the medicine", "The medicine was administered by the nurse"),
            ("She founded the company", "The company was founded by her"),
            ("He coached the team", "The team was coached by him"),
            ("They installed the equipment", "The equipment was installed by them"),
            ("The pilot flew the plane", "The plane was flown by the pilot"),
            ("She led the expedition", "The expedition was led by her"),
            ("He brewed the coffee", "The coffee was brewed by him"),
            ("They hosted the conference", "The conference was hosted by them"),
            ("The committee rejected the proposal", "The proposal was rejected by the committee"),
            ("She painted the landscape", "The landscape was painted by her"),
            ("He drove the bus", "The bus was driven by him"),
            ("They drafted the constitution", "The constitution was drafted by them"),
            ("The student answered the question", "The question was answered by the student"),
            ("She baked the cake", "The cake was baked by her"),
            ("He caught the fish", "The fish was caught by him"),
            ("They cut the grass", "The grass was cut by them"),
            ("The waiter served the food", "The food was served by the waiter"),
            ("She typed the letter", "The letter was typed by her"),
            ("He mailed the package", "The package was mailed by him"),
            ("They washed the clothes", "The clothes were washed by them"),
            ("The referee blew the whistle", "The whistle was blown by the referee"),
            ("She wrapped the gift", "The gift was wrapped by her"),
            ("He tuned the piano", "The piano was tuned by him"),
            ("They sold the house", "The house was sold by them"),
            ("The doctor examined the patient", "The patient was examined by the doctor"),
            ("She sang the anthem", "The anthem was sung by her"),
            ("He built the bridge", "The bridge was built by him"),
            ("They planted the garden", "The garden was planted by them"),
            ("The captain steered the ship", "The ship was steered by the captain"),
            ("She wrote the poem", "The poem was written by her"),
            ("He mixed the chemicals", "The chemicals were mixed by him"),
            ("They filmed the scene", "The scene was filmed by them"),
            ("The manager hired the staff", "The staff was hired by the manager"),
            ("She knitted the sweater", "The sweater was knitted by her"),
            ("He brewed the tea", "The tea was brewed by him"),
            ("They painted the ceiling", "The ceiling was painted by them"),
            ("The chef seasoned the soup", "The soup was seasoned by the chef"),
            ("She folded the paper", "The paper was folded by her"),
            ("He polished the silver", "The silver was polished by him"),
            ("They decorated the hall", "The hall was decorated by them"),
        ]
    },
    'semantic_valence': {
        'type': 'SEM',
        'pairs': [
            ("She felt joy when she saw him", "She felt anger when she saw him"),
            ("The gift brought happiness", "The loss brought sadness"),
            ("He showed kindness to others", "He showed cruelty to others"),
            ("The garden was beautiful", "The wasteland was ugly"),
            ("She spoke with love", "She spoke with hatred"),
            ("The victory was glorious", "The defeat was shameful"),
            ("He gave a generous donation", "He made a selfish demand"),
            ("The sunrise was magnificent", "The disaster was devastating"),
            ("She expressed gratitude", "She expressed resentment"),
            ("The peace was comforting", "The war was terrifying"),
            ("He showed compassion for the poor", "He showed contempt for the poor"),
            ("The melody was delightful", "The noise was unbearable"),
            ("She felt pride in her work", "She felt shame in her work"),
            ("The feast was wonderful", "The famine was horrible"),
            ("He offered help willingly", "He refused help stubbornly"),
            ("The harmony was soothing", "The conflict was distressing"),
            ("She radiated warmth", "She radiated coldness"),
            ("The blessing was welcome", "The curse was feared"),
            ("He found pleasure in reading", "He found misery in reading"),
            ("The prosperity was encouraging", "The poverty was depressing"),
            ("She showed respect for elders", "She showed disrespect for elders"),
            ("The freedom was liberating", "The imprisonment was suffocating"),
            ("He felt hope for the future", "He felt despair for the future"),
            ("The success was rewarding", "The failure was punishing"),
            ("She experienced bliss", "She experienced agony"),
            ("The comfort was reassuring", "The pain was alarming"),
            ("He displayed courage in battle", "He displayed cowardice in battle"),
            ("The friendship was loyal", "The betrayal was devastating"),
            ("She found peace in meditation", "She found turmoil in meditation"),
            ("The justice was fair", "The injustice was outrageous"),
            ("He received praise from the boss", "He received criticism from the boss"),
            ("The beauty was breathtaking", "The ugliness was repulsive"),
            ("She felt satisfaction with life", "She felt dissatisfaction with life"),
            ("The harmony brought unity", "The discord brought division"),
            ("He demonstrated wisdom", "He demonstrated foolishness"),
            ("The truth was enlightening", "The lie was deceiving"),
            ("She showed patience with children", "She showed impatience with children"),
            ("The health was robust", "The illness was severe"),
            ("He gained honor through service", "He gained disgrace through crime"),
            ("The celebration was festive", "The mourning was somber"),
            ("She found strength in adversity", "She found weakness in adversity"),
            ("The trust was unbreakable", "The suspicion was corrosive"),
            ("He felt contentment at home", "He felt restlessness at home"),
            ("The miracle was awe-inspiring", "The tragedy was heartbreaking"),
            ("She demonstrated humility", "She demonstrated arrogance"),
            ("The progress was remarkable", "The decline was alarming"),
            ("He found comfort in friends", "He found loneliness in isolation"),
            ("The grace was elegant", "The clumsiness was awkward"),
            ("She showed mercy to prisoners", "She showed cruelty to prisoners"),
            ("The abundance was plentiful", "The scarcity was dire"),
            ("He experienced wonder", "He experienced boredom"),
            ("The clarity was refreshing", "The confusion was frustrating"),
            ("She felt enthusiasm for work", "She felt apathy for work"),
            ("The kindness was touching", "The malice was chilling"),
            ("He found inspiration in art", "He found despair in art"),
            ("The serenity was calming", "The chaos was overwhelming"),
            ("She experienced delight", "She experienced disgust"),
            ("The achievement was impressive", "The setback was disappointing"),
            ("He showed devotion to family", "He showed neglect of family"),
            ("The freshness was invigorating", "The staleness was depressing"),
            ("She found comfort in prayer", "She found anguish in prayer"),
            ("The innovation was exciting", "The stagnation was dull"),
            ("He displayed generosity", "He displayed greed"),
            ("The safety was reassuring", "The danger was terrifying"),
            ("She felt tenderness for animals", "She felt hostility for animals"),
            ("The recovery was miraculous", "The relapse was devastating"),
            ("He showed loyalty to friends", "He showed betrayal of friends"),
            ("The vitality was energizing", "The lethargy was draining"),
            ("She found joy in simplicity", "She found misery in complexity"),
            ("The excellence was outstanding", "The mediocrity was underwhelming"),
            ("He experienced gratitude", "He experienced resentment"),
            ("The wealth was abundant", "The poverty was extreme"),
            ("She showed forgiveness", "She showed vengeance"),
            ("The optimism was inspiring", "The pessimism was discouraging"),
            ("He found meaning in service", "He found emptiness in selfishness"),
            ("The resilience was admirable", "The fragility was concerning"),
            ("She experienced serenity", "She experienced anxiety"),
            ("The glory was triumphant", "The shame was humiliating"),
        ]
    },
    'semantic_topic': {
        'type': 'SEM',
        'pairs': [
            ("The doctor examined the patient carefully", "The teacher explained the lesson clearly"),
            ("The engine roared to life", "The orchestra played beautifully"),
            ("She studied chemistry at university", "She studied painting at university"),
            ("The stock market crashed today", "The river flooded the valley"),
            ("He repaired the broken computer", "He cooked the traditional meal"),
            ("The lawyer argued the case", "The farmer harvested the wheat"),
            ("The scientist discovered a new element", "The artist created a new sculpture"),
            ("She programmed the algorithm", "She composed the melody"),
            ("The bridge collapsed under pressure", "The cake rose in the oven"),
            ("He analyzed the financial data", "He analyzed the poetry"),
            ("The machine processed the information", "The chef prepared the ingredients"),
            ("She calculated the trajectory", "She painted the landscape"),
            ("The rocket launched successfully", "The ship sailed smoothly"),
            ("He diagnosed the medical condition", "He interpreted the dream"),
            ("The engineer designed the circuit", "The architect designed the garden"),
            ("She measured the radiation level", "She measured the emotional response"),
            ("The microscope revealed the bacteria", "The telescope revealed the galaxy"),
            ("He solved the mathematical equation", "He solved the philosophical puzzle"),
            ("The factory produced steel beams", "The bakery produced fresh bread"),
            ("She operated the surgical equipment", "She played the musical instrument"),
            ("The computer processed the data", "The brain processed the memory"),
            ("He invested in the technology company", "He invested in the art gallery"),
            ("The weather satellite tracked the storm", "The space telescope tracked the comet"),
            ("She wrote the research paper", "She wrote the novel"),
            ("The robot assembled the car parts", "The craftsman carved the wooden figure"),
            ("He prescribed the medication", "He recommended the restaurant"),
            ("The microscope magnified the cell", "The loudspeaker amplified the voice"),
            ("She coded the software application", "She choreographed the dance routine"),
            ("The thermometer measured the temperature", "The scale measured the weight"),
            ("He operated the heavy machinery", "He conducted the orchestra"),
            ("The battery powered the device", "The wind powered the mill"),
            ("She calibrated the scientific instrument", "She tuned the piano"),
            ("The laser cut through the metal", "The scissors cut through the fabric"),
            ("He designed the electronic circuit", "He designed the flower arrangement"),
            ("The reactor generated nuclear power", "The dam generated hydroelectric power"),
            ("She synthesized the chemical compound", "She blended the musical tracks"),
            ("The sensor detected the radiation", "The nose detected the fragrance"),
            ("He engineered the suspension bridge", "He composed the string quartet"),
            ("The algorithm sorted the database", "The librarian sorted the books"),
            ("She optimized the neural network", "She perfected the recipe"),
            ("The microscope revealed the microorganism", "The painting revealed the emotion"),
            ("He calculated the orbital mechanics", "He calculated the musical intervals"),
            ("The factory manufactured microchips", "The studio manufactured recordings"),
            ("She debugged the software code", "She edited the manuscript"),
            ("The vaccine prevented the disease", "The sunscreen prevented the burn"),
            ("He simulated the physical process", "He rehearsed the theatrical scene"),
            ("The transistor switched the current", "The switch turned on the light"),
            ("She analyzed the DNA sequence", "She analyzed the poem structure"),
            ("The laboratory tested the hypothesis", "The kitchen tested the recipe"),
            ("He programmed the robotic arm", "He trained the rescue dog"),
            ("The accelerator increased the speed", "The fertilizer increased the growth"),
            ("She developed the vaccine formula", "She developed the film script"),
            ("The computer stored the information", "The library stored the knowledge"),
            ("He diagnosed the engine problem", "He diagnosed the medical condition"),
            ("The telescope observed the distant star", "The microscope observed the tiny cell"),
            ("She generated the statistical report", "She generated the creative portfolio"),
            ("The machine learned the pattern", "The student learned the lesson"),
            ("He engineered the safety system", "He designed the security plan"),
            ("The processor executed the command", "The actor executed the performance"),
            ("She patented the invention", "She published the poem"),
            ("The satellite transmitted the signal", "The radio transmitted the broadcast"),
            ("He measured the quantum state", "He measured the emotional state"),
            ("The algorithm classified the image", "The critic classified the artwork"),
            ("She built the neural model", "She built the architectural model"),
            ("The computer rendered the scene", "The painter rendered the portrait"),
            ("He solved the optimization problem", "He resolved the conflict"),
            ("The network transmitted the data packet", "The mailman delivered the letter"),
            ("She calibrated the laser beam", "She adjusted the stage lighting"),
            ("The robot navigated the obstacle course", "The dancer navigated the stage"),
            ("He developed the mobile application", "He developed the photographic film"),
            ("The database stored the records", "The museum stored the artifacts"),
            ("She encrypted the digital message", "She encoded the secret meaning"),
            ("The processor computed the result", "The judge reached the verdict"),
            ("He analyzed the market trends", "He analyzed the literary themes"),
            ("The circuit carried the electrical current", "The river carried the water"),
            ("She modeled the physical system", "She modeled the fashion design"),
            ("The technology advanced rapidly", "The art evolved gradually"),
        ]
    }
}


def load_model(model_key, device='cuda'):
    """加载训练模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = MODEL_CONFIGS[model_key]
    path = config['path']

    if model_key in ["glm4", "deepseek7b"]:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map='cpu',
            trust_remote_code=True, local_files_only=True
        )
        model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_random_model(model_key, device='cuda'):
    """创建随机初始化模型(同架构) — 使用8bit加载后重新初始化"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = MODEL_CONFIGS[model_key]
    path = config['path']

    print(f"  创建随机模型(8bit加载后重初始化权重)...")

    # 加载模型(8bit)然后重新初始化权重
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    random_model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, local_files_only=True
    )
    
    # 重新初始化所有可训练的权重
    def init_weights(module):
        if hasattr(module, 'weight') and module.weight is not None:
            if hasattr(module.weight, 'data') and module.weight.data.is_floating_point():
                torch.nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            if hasattr(module.bias, 'data') and module.bias.data.is_floating_point():
                torch.nn.init.zeros_(module.bias.data)
    
    random_model.apply(init_weights)
    random_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return random_model, tokenizer


def get_feature_pc1(model, tokenizer, feature_name, feature_data, n_pairs, last_layer, device):
    """计算特征的PC1方向(最后一层)"""
    pairs = feature_data['pairs'][:n_pairs]
    all_diffs = []

    for s1, s2 in pairs:
        try:
            with torch.no_grad():
                toks1 = tokenizer(s1, return_tensors='pt').to(model.device)
                out1 = model(**toks1, output_hidden_states=True)
                h1 = out1.hidden_states[last_layer][0, -1, :].float().cpu().numpy()

                toks2 = tokenizer(s2, return_tensors='pt').to(model.device)
                out2 = model(**toks2, output_hidden_states=True)
                h2 = out2.hidden_states[last_layer][0, -1, :].float().cpu().numpy()

                diff = h1 - h2
                diff = diff / (np.linalg.norm(diff) + 1e-10)
                all_diffs.append(diff)

                del out1, out2
        except Exception as e:
            continue

    if len(all_diffs) < 10:
        return None

    diffs_matrix = np.array(all_diffs)
    mean = diffs_matrix.mean(axis=0)
    centered = diffs_matrix - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = Vt[0]
    explained = S[0]**2 / (S**2).sum()

    return {
        'pc1': pc1,
        'explained_ratio': float(explained),
        'n_samples': len(all_diffs),
        'type': feature_data['type']
    }


def compute_residue_rotation(model, tokenizer, n_layers, device):
    """计算残差旋转分量(Δh⊥h比例)"""
    test_sents = [
        "The cat sat on the mat",
        "She walked to the store yesterday",
        "The book was written by a famous author",
        "He does not like cold weather",
        "The students studied for the exam",
        "The garden grew beautiful flowers",
        "The car was repaired by the mechanic",
        "She felt joy when she heard the news",
        "The scientist conducted the experiment",
        "He painted the fence last weekend",
    ]

    rotation_ratios = []
    for sent in test_sents:
        try:
            with torch.no_grad():
                toks = tokenizer(sent, return_tensors='pt').to(model.device)
                out = model(**toks, output_hidden_states=True)

                for l in range(1, min(n_layers, len(out.hidden_states))):
                    h_prev = out.hidden_states[l-1][0, -1, :].float().cpu().numpy()
                    h_curr = out.hidden_states[l][0, -1, :].float().cpu().numpy()
                    delta = h_curr - h_prev

                    h_norm = np.linalg.norm(h_prev)
                    if h_norm < 1e-10:
                        continue
                    h_dir = h_prev / h_norm

                    parallel = np.dot(delta, h_dir) * h_dir
                    perp = delta - parallel

                    parallel_norm = np.linalg.norm(parallel)
                    perp_norm = np.linalg.norm(perp)
                    total = parallel_norm + perp_norm

                    if total > 1e-10:
                        rotation_ratios.append(perp_norm / total)

                del out
        except:
            continue

    return float(np.mean(rotation_ratios)) if rotation_ratios else 0.0


def run_analysis(model_key, n_pairs=80, device='cuda'):
    """运行完整分析"""
    from transformers import BitsAndBytesConfig

    t0 = time.time()
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    d_model = config['d_model']
    last_layer = n_layers - 1

    out_dir = f'results/causal_fiber/{model_key}_ccxxxii'
    os.makedirs(out_dir, exist_ok=True)
    log_path = f'{out_dir}/run.log'

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log("=" * 70)
    log(f"Phase CCXXXII: 频分复用的来源(训练vs架构) — {config['name']}")
    log(f"  n_pairs={n_pairs}, d_model={d_model}, n_layers={n_layers}")
    log(f"  时间={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # ===== S1: 加载训练模型 =====
    log(f"\n--- S1: 加载训练模型 ---")
    trained_model, tokenizer = load_model(model_key, device)
    log(f"  加载完成: n_layers={n_layers}, d_model={d_model}")

    # ===== S2: 训练模型特征PC1正交性 =====
    log(f"\n{'=' * 60}")
    log(f"S2: 训练模型 — 5特征PC1正交性(最后一层)")
    log(f"{'=' * 60}")

    trained_features = {}
    for fname, fdata in FEATURES.items():
        log(f"  收集 {fname} ({n_pairs}对, 类型={fdata['type']})...")
        result = get_feature_pc1(trained_model, tokenizer, fname, fdata, n_pairs, last_layer, device)
        if result:
            trained_features[fname] = result
            log(f"    PC1解释方差: {result['explained_ratio']:.3f}, 样本数: {result['n_samples']}")

    # 训练模型PC1正交性
    trained_cos_matrix = {}
    fnames = list(trained_features.keys())
    for i, f1 in enumerate(fnames):
        for j, f2 in enumerate(fnames):
            if i < j:
                cos = abs(np.dot(trained_features[f1]['pc1'], trained_features[f2]['pc1']))
                trained_cos_matrix[(f1, f2)] = cos

    trained_avg_cos = np.mean(list(trained_cos_matrix.values()))
    trained_max_cos = np.max(list(trained_cos_matrix.values()))

    log(f"\n  训练模型PC1×PC1 |cos|:")
    for (f1, f2), cos in sorted(trained_cos_matrix.items(), key=lambda x: -x[1]):
        log(f"    {f1} × {f2}: {cos:.4f}")
    log(f"  平均|cos|: {trained_avg_cos:.4f}, 最大|cos|: {trained_max_cos:.4f}")

    # ===== S3: 训练模型残差旋转 =====
    log(f"\n{'=' * 60}")
    log(f"S3: 训练模型残差旋转分量")
    log(f"{'=' * 60}")

    trained_rotation = compute_residue_rotation(trained_model, tokenizer, n_layers, device)
    log(f"  Δh⊥h平均比例: {trained_rotation:.4f}")

    # 释放训练模型
    del trained_model, tokenizer
    torch.cuda.empty_cache()
    log(f"  训练模型已释放")

    # ===== S4: 加载随机模型 =====
    log(f"\n{'=' * 60}")
    log(f"S4: 加载随机初始化模型(同架构)")
    log(f"{'=' * 60}")

    random_model, tokenizer = create_random_model(model_key, device)
    log(f"  随机模型创建完成")

    # ===== S5: 随机模型特征PC1正交性 =====
    log(f"\n{'=' * 60}")
    log(f"S5: 随机模型 — 5特征PC1正交性(最后一层)")
    log(f"{'=' * 60}")

    random_features = {}
    for fname, fdata in FEATURES.items():
        log(f"  收集 {fname} ({n_pairs}对, 类型={fdata['type']})...")
        result = get_feature_pc1(random_model, tokenizer, fname, fdata, n_pairs, last_layer, device)
        if result:
            random_features[fname] = result
            log(f"    PC1解释方差: {result['explained_ratio']:.3f}, 样本数: {result['n_samples']}")

    # 随机模型PC1正交性
    random_cos_matrix = {}
    rfnames = list(random_features.keys())
    for i, f1 in enumerate(rfnames):
        for j, f2 in enumerate(rfnames):
            if i < j:
                cos = abs(np.dot(random_features[f1]['pc1'], random_features[f2]['pc1']))
                random_cos_matrix[(f1, f2)] = cos

    random_avg_cos = np.mean(list(random_cos_matrix.values())) if random_cos_matrix else 0
    random_max_cos = np.max(list(random_cos_matrix.values())) if random_cos_matrix else 0

    log(f"\n  随机模型PC1×PC1 |cos|:")
    for (f1, f2), cos in sorted(random_cos_matrix.items(), key=lambda x: -x[1]):
        log(f"    {f1} × {f2}: {cos:.4f}")
    log(f"  平均|cos|: {random_avg_cos:.4f}, 最大|cos|: {random_max_cos:.4f}")

    # ===== S6: 随机模型残差旋转 =====
    log(f"\n{'=' * 60}")
    log(f"S6: 随机模型残差旋转分量")
    log(f"{'=' * 60}")

    random_rotation = compute_residue_rotation(random_model, tokenizer, n_layers, device)
    log(f"  Δh⊥h平均比例: {random_rotation:.4f}")

    # ===== S7: 随机基线 =====
    log(f"\n{'=' * 60}")
    log(f"S7: 理论随机基线与核心对比")
    log(f"{'=' * 60}")

    N = len(trained_features)
    expected_random_cos = np.sqrt(2 / (np.pi * d_model))
    log(f"  d_model={d_model}, N个特征={N}")
    log(f"  理论随机|cos| ≈ √(2/(π·d)) = {expected_random_cos:.4f}")
    log(f"  训练模型实际|cos| = {trained_avg_cos:.4f} (是随机的 {trained_avg_cos/expected_random_cos:.1f}x)")
    log(f"  随机模型实际|cos| = {random_avg_cos:.4f} (是随机的 {random_avg_cos/expected_random_cos:.1f}x)")

    # ===== S8: PC1解释方差对比 =====
    log(f"\n{'=' * 60}")
    log(f"S8: PC1解释方差对比(训练vs随机)")
    log(f"{'=' * 60}")

    log(f"  {'特征':<20} {'训练PC1%':>10} {'随机PC1%':>10} {'比值':>8}")
    log(f"  {'----':<20} {'-------':>10} {'-------':>10} {'----':>8}")
    for fname in FEATURES:
        if fname in trained_features and fname in random_features:
            t_ratio = trained_features[fname]['explained_ratio']
            r_ratio = random_features[fname]['explained_ratio']
            ratio = t_ratio / r_ratio if r_ratio > 0 else 0
            log(f"  {fname:<20} {t_ratio*100:>9.1f}% {r_ratio*100:>9.1f}% {ratio:>7.2f}x")

    # ===== S9: 逐对对比 =====
    log(f"\n{'=' * 60}")
    log(f"S9: 逐对|cos|对比(训练vs随机)")
    log(f"{'=' * 60}")

    log(f"  {'特征对':<35} {'训练|cos|':>10} {'随机|cos|':>10} {'训练/随机':>10}")
    log(f"  {'------':<35} {'--------':>10} {'--------':>10} {'--------':>10}")
    all_pairs = set(trained_cos_matrix.keys()) | set(random_cos_matrix.keys())
    for (f1, f2) in sorted(all_pairs):
        t_cos = trained_cos_matrix.get((f1, f2), 0)
        r_cos = random_cos_matrix.get((f1, f2), 0)
        ratio = t_cos / r_cos if r_cos > 0.001 else float('inf')
        log(f"  {f1} × {f2:<25} {t_cos:>10.4f} {r_cos:>10.4f} {ratio:>10.2f}x")

    # ===== S10: 总结 =====
    log(f"\n{'=' * 60}")
    log(f"S10: 核心对比总结")
    log(f"{'=' * 60}")

    log(f"\n  指标                     训练模型      随机模型      随机基线")
    log(f"  ----                     --------      --------      --------")
    log(f"  平均|cos|               {trained_avg_cos:.4f}        {random_avg_cos:.4f}        {expected_random_cos:.4f}")
    log(f"  最大|cos|               {trained_max_cos:.4f}        {random_max_cos:.4f}        -")
    log(f"  Δh⊥h比例               {trained_rotation:.4f}        {random_rotation:.4f}        -")

    # 判定
    log(f"\n  判定:")
    orth_ratio = random_avg_cos / trained_avg_cos if trained_avg_cos > 0 else 0
    if trained_avg_cos < random_avg_cos * 0.5:
        log(f"    → 训练模型正交性显著高于随机模型({orth_ratio:.1f}x) → **频分复用是训练产生的**")
    elif trained_avg_cos > random_avg_cos * 2:
        log(f"    → 训练模型正交性显著低于随机模型 → **训练降低了正交性**")
    else:
        log(f"    → 训练模型和随机模型正交性相近({orth_ratio:.1f}x) → **频分复用部分来自架构**")

    if trained_rotation > 0.7 and random_rotation > 0.7:
        log(f"    → 两者Δh⊥h比例都>0.7 → **旋转主导是架构属性**")
    elif trained_rotation > 0.7 and random_rotation < 0.5:
        log(f"    → 训练模型旋转主导, 随机模型不 → **旋转主导是训练产生的**")
    else:
        log(f"    → 旋转主导的程度需要进一步分析 (训练:{trained_rotation:.3f} vs 随机:{random_rotation:.3f})")

    # PC1解释方差判定
    trained_avg_explained = np.mean([v['explained_ratio'] for v in trained_features.values()])
    random_avg_explained = np.mean([v['explained_ratio'] for v in random_features.values()])
    log(f"\n  PC1解释方差:")
    log(f"    训练模型平均: {trained_avg_explained*100:.1f}%")
    log(f"    随机模型平均: {random_avg_explained*100:.1f}%")
    if trained_avg_explained > random_avg_explained * 2:
        log(f"    → 训练模型PC1解释方差显著高于随机 → **训练产生了语义方向一致性**")
    else:
        log(f"    → PC1解释方差相近 → 语义方向一致性可能是架构属性")

    # 释放随机模型
    del random_model, tokenizer
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    log(f"\n  总耗时: {elapsed/60:.1f}分钟")

    # 保存结果
    results = {
        'model': model_key,
        'd_model': d_model,
        'n_layers': n_layers,
        'n_pairs': n_pairs,
        'trained': {
            'avg_cos': float(trained_avg_cos),
            'max_cos': float(trained_max_cos),
            'rotation_ratio': float(trained_rotation),
            'avg_pc1_explained': float(trained_avg_explained),
            'cos_pairs': {f"{f1}_{f2}": float(v) for (f1, f2), v in trained_cos_matrix.items()},
            'pc1_explained': {f: float(v['explained_ratio']) for f, v in trained_features.items()},
        },
        'random': {
            'avg_cos': float(random_avg_cos),
            'max_cos': float(random_max_cos),
            'rotation_ratio': float(random_rotation),
            'avg_pc1_explained': float(random_avg_explained),
            'cos_pairs': {f"{f1}_{f2}": float(v) for (f1, f2), v in random_cos_matrix.items()},
            'pc1_explained': {f: float(v['explained_ratio']) for f, v in random_features.items()},
        },
        'baseline': {
            'expected_random_cos': float(expected_random_cos),
        }
    }

    with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"  结果已保存: {out_dir}/results.json")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=80)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    run_analysis(args.model, args.n_pairs, args.device)
