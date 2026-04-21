"""
Phase CCX: 语法/语义因果对偶验证 (大样本+多特征)
核心改进:
  1. number/sentiment/semantic_topic扩充到80-100模板
  2. 新增语法特征: negation(否定), question(疑问), modality(情态)
  3. 验证语法=积分算子 vs 语义=投影算子的对偶假设
  4. 增量保存 + 日志
"""
import os, sys, gc, time, json, argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'w', buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

PATHS = {
    "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
    "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

# ===== 语法特征 (逐层累积型) =====

TENSE = [
    ("The cat sat quietly on the mat", "The cat sits quietly on the mat"),
    ("She walked to the store yesterday", "She walks to the store today"),
    ("He played guitar every evening", "He plays guitar every evening"),
    ("They worked on the project", "They work on the project"),
    ("The dog ran across the field", "The dog runs across the field"),
    ("She wrote a long letter", "She writes a long letter"),
    ("He drove the car fast", "He drives the car fast"),
    ("They built a new house", "They build a new house"),
    ("The bird flew over the lake", "The bird flies over the lake"),
    ("She cooked dinner for us", "She cooks dinner for us"),
    ("He read many books last year", "He reads many books this year"),
    ("They sang beautiful songs", "They sing beautiful songs"),
    ("The train arrived late", "The train arrives late"),
    ("She taught the children math", "She teaches the children math"),
    ("He caught the ball easily", "He catches the ball easily"),
    ("They grew vegetables in spring", "They grow vegetables in spring"),
    ("The water froze overnight", "The water freezes overnight"),
    ("She drew a colorful picture", "She draws a colorful picture"),
    ("He held the baby gently", "He holds the baby gently"),
    ("They knew the answer quickly", "They know the answer quickly"),
    ("The sun rose early this morning", "The sun rises early every morning"),
    ("She chose the blue dress", "She chooses the blue dress"),
    ("He broke the window accidentally", "He breaks the window accidentally"),
    ("They spoke about the problem", "They speak about the problem"),
    ("The wind blew strongly today", "The wind blows strongly today"),
    ("I thought about the plan carefully", "I think about the plan carefully"),
    ("We found the solution quickly", "We find the solution quickly"),
    ("She kept the secret safe", "She keeps the secret safe"),
    ("He brought the documents yesterday", "He brings the documents today"),
    ("They fought for their rights", "They fight for their rights"),
]

POLARITY = [
    ("The cat is happy today", "The cat is not happy today"),
    ("She likes the new movie", "She does not like the new movie"),
    ("He can solve the problem", "He cannot solve the problem"),
    ("They will attend the meeting", "They will not attend the meeting"),
    ("The dog is friendly to strangers", "The dog is not friendly to strangers"),
    ("She has finished the work", "She has not finished the work"),
    ("He was available yesterday", "He was not available yesterday"),
    ("They are coming to the party", "They are not coming to the party"),
    ("The food was delicious", "The food was not delicious"),
    ("She does know the answer", "She does not know the answer"),
    ("He should go to school", "He should not go to school"),
    ("They would help with chores", "They would not help with chores"),
    ("The test was easy for me", "The test was not easy for me"),
    ("She could swim very well", "She could not swim very well"),
    ("He must leave right now", "He must not leave right now"),
    ("They had enough money saved", "They had not enough money saved"),
    ("The door was open this morning", "The door was not open this morning"),
    ("She did enjoy the concert", "She did not enjoy the concert"),
    ("He has been working hard", "He has not been working hard"),
    ("They were ready on time", "They were not ready on time"),
    ("The plan was successful", "The plan was not successful"),
    ("She will accept the offer", "She will not accept the offer"),
    ("He can drive the truck", "He cannot drive the truck"),
    ("They did win the game", "They did not win the game"),
    ("The idea was brilliant", "The idea was not brilliant"),
    ("I believe this is correct", "I do not believe this is correct"),
    ("We trust the new system", "We do not trust the new system"),
    ("She understands the material", "She does not understand the material"),
    ("He remembers the details", "He does not remember the details"),
    ("They appreciate the effort", "They do not appreciate the effort"),
]

# 扩充NUMBER到100对!
NUMBER = [
    ("The cat is sleeping now", "The cats are sleeping now"),
    ("A book was on the table", "The books were on the table"),
    ("The child plays outside", "The children play outside"),
    ("A dog runs in the park", "The dogs run in the park"),
    ("The woman walks to work", "The women walk to work"),
    ("A man drives the bus", "The men drive the bus"),
    ("The bird flies high above", "The birds fly high above"),
    ("A student reads the text", "The students read the text"),
    ("The teacher explains clearly", "The teachers explain clearly"),
    ("A flower grows in spring", "The flowers grow in spring"),
    ("The tree stands very tall", "The trees stand very tall"),
    ("A car drives down the road", "The cars drive down the road"),
    ("The house looks beautiful", "The houses look beautiful"),
    ("A river flows through town", "The rivers flow through town"),
    ("The mountain rises above clouds", "The mountains rise above clouds"),
    ("A star shines in the sky", "The stars shine in the sky"),
    ("The country borders two seas", "The countries border two seas"),
    ("A city grows very fast", "The cities grow very fast"),
    ("The church stands on the hill", "The churches stand on the hill"),
    ("A bridge crosses the river", "The bridges cross the river"),
    ("The knife cuts the bread", "The knives cut the bread"),
    ("A wolf howls at night", "The wolves howl at night"),
    ("The leaf falls from the tree", "The leaves fall from the tree"),
    ("A shelf holds many items", "The shelves hold many items"),
    ("The wolf hunts in the pack", "The wolves hunt in the pack"),
    ("A calf drinks the milk", "The calves drink the milk"),
    ("The half was given to each", "The halves were given to each"),
    ("A loaf sits on the counter", "The loaves sit on the counter"),
    ("The thief stole the jewel", "The thieves stole the jewel"),
    ("A wife prepares the meal", "The wives prepare the meal"),
    ("The life was saved by doctors", "The lives were saved by doctors"),
    ("A mouse ran under the couch", "The mice ran under the couch"),
    ("The goose swam in the pond", "The geese swam in the pond"),
    ("A foot stepped on the grass", "The feet stepped on the grass"),
    ("The tooth was cleaned carefully", "The teeth were cleaned carefully"),
    ("A person entered the room", "The people entered the room"),
    ("The ox pulled the cart", "The oxen pulled the cart"),
    ("A datum supports the theory", "The data support the theory"),
    ("The criterion was met perfectly", "The criteria were met perfectly"),
    ("A phenomenon occurred today", "The phenomena occurred today"),
    ("The sheep grazed quietly", "The sheep grazed quietly"),
    ("A fish swam upstream", "The fish swam upstream"),
    ("The deer ran through the forest", "The deer ran through the forest"),
    ("A species evolved over time", "The species evolved over time"),
    ("The aircraft landed safely", "The aircraft landed safely"),
    ("A series was broadcast", "The series were broadcast"),
    ("The headquarters moved downtown", "The headquarters moved downtown"),
    ("A means was found to help", "The means were found to help"),
    ("The offspring grew strong", "The offspring grew strong"),
    ("The boy plays the game", "The boys play the game"),
    ("A girl sings the song", "The girls sing the song"),
    ("The baby cries loudly", "The babies cry loudly"),
    ("A lady walks gracefully", "The ladies walk gracefully"),
    ("The gentleman speaks softly", "The gentlemen speak softly"),
    ("A hero saves the day", "The heroes save the day"),
    ("The potato grows underground", "The potatoes grow underground"),
    ("A tomato ripens on the vine", "The tomatoes ripen on the vine"),
    ("The echo bounced off the wall", "The echoes bounced off the wall"),
    ("A motto inspires the team", "The mottos inspire the team"),
    ("The volcano erupted violently", "The volcanoes erupted violently"),
    ("A ratio was calculated", "The ratios were calculated"),
    ("The studio produced the film", "The studios produced the film"),
    ("A piano plays the melody", "The pianos play the melody"),
    ("The radio broadcast the news", "The radios broadcast the news"),
    ("A video was shared online", "The videos were shared online"),
    ("The zoo housed the animal", "The zoos housed the animals"),
    ("A roof covers the building", "The roofs cover the buildings"),
    ("The proof supports the claim", "The proofs support the claims"),
    ("A chief leads the tribe", "The chiefs lead the tribe"),
    ("The cliff overlooks the sea", "The cliffs overlook the sea"),
    ("A scarf keeps you warm", "The scarves keep you warm"),
    ("The wharf extends into water", "The wharves extend into water"),
    ("A dwarf mined the gems", "The dwarves mined the gems"),
    ("The elf helped the traveler", "The elves helped the travelers"),
    ("A shelf stores the books", "The shelves store the books"),
    ("The self reflects on life", "The selves reflect on life"),
    ("A calf was born today", "The calves were born today"),
    ("The half remained unfinished", "The halves remained unfinished"),
    ("A loaf was baked fresh", "The loaves were baked fresh"),
    ("The wife cared for family", "The wives cared for family"),
    ("A knife sliced the fruit", "The knives sliced the fruit"),
    ("The life continued as normal", "The lives continued as normal"),
    ("A belief guided the action", "The beliefs guided the actions"),
    ("The grief was shared among all", "The griefs were shared among all"),
    ("A relief came after the storm", "The reliefs came after the storm"),
    ("The chief commanded respect", "The chiefs commanded respect"),
    ("A brief outlined the plan", "The briefs outlined the plan"),
    ("The thief escaped the scene", "The thieves escaped the scene"),
    ("A hand reached out to help", "The hands reached out to help"),
    ("The land stretched far ahead", "The lands stretched far ahead"),
    ("A band played the music", "The bands played the music"),
    ("The sand covered the beach", "The sands covered the beaches"),
    ("A friend helped with the task", "The friends helped with the tasks"),
    ("The trend continued upward", "The trends continued upward"),
    ("A blend created the flavor", "The blends created the flavors"),
    ("The island sat in the ocean", "The islands sat in the ocean"),
    ("A thousand attended the event", "Thousands attended the events"),
    ("The ground shook violently", "The grounds shook violently"),
]

# 新增: 否定句 (语法特征)
NEGATION = [
    ("She is going to the store", "She is not going to the store"),
    ("He was working late today", "He was not working late today"),
    ("They have seen the movie", "They have not seen the movie"),
    ("The cat will eat the food", "The cat will not eat the food"),
    ("I could hear the music", "I could not hear the music"),
    ("She did finish the report", "She did not finish the report"),
    ("He would agree to help", "He would not agree to help"),
    ("We should try the method", "We should not try the method"),
    ("The door was locked tight", "The door was not locked tight"),
    ("She had written the essay", "She had not written the essay"),
    ("He can speak the language", "He cannot speak the language"),
    ("They were watching the show", "They were not watching the show"),
    ("I am enjoying the book", "I am not enjoying the book"),
    ("The kids are playing outside", "The kids are not playing outside"),
    ("She has visited Paris", "She has not visited Paris"),
    ("He does understand the rules", "He does not understand the rules"),
    ("They might come tomorrow", "They might not come tomorrow"),
    ("We need to leave early", "We do not need to leave early"),
    ("The system is working well", "The system is not working well"),
    ("She knows the correct answer", "She does not know the correct answer"),
    ("He believes in the cause", "He does not believe in the cause"),
    ("They want to join the club", "They do not want to join the club"),
    ("I like the new design", "I do not like the new design"),
    ("The plan makes sense now", "The plan does not make sense now"),
    ("She enjoys cooking meals", "She does not enjoy cooking meals"),
    ("He owns a small boat", "He does not own a small boat"),
    ("They play tennis regularly", "They do not play tennis regularly"),
    ("We support the proposal", "We do not support the proposal"),
    ("The method works perfectly", "The method does not work perfectly"),
    ("She remembers the details", "She does not remember the details"),
]

# 新增: 疑问句 (语法特征)
QUESTION = [
    ("The cat is sleeping now", "Is the cat sleeping now"),
    ("She walked to the store", "Did she walk to the store"),
    ("He can solve the problem", "Can he solve the problem"),
    ("They will attend the meeting", "Will they attend the meeting"),
    ("The dog is friendly to strangers", "Is the dog friendly to strangers"),
    ("She has finished the work", "Has she finished the work"),
    ("He was available yesterday", "Was he available yesterday"),
    ("They are coming to the party", "Are they coming to the party"),
    ("The food was delicious", "Was the food delicious"),
    ("She should go to school", "Should she go to school"),
    ("He could swim very well", "Could he swim very well"),
    ("They would help with chores", "Would they help with chores"),
    ("The test was easy for me", "Was the test easy for me"),
    ("He must leave right now", "Must he leave right now"),
    ("The door was open this morning", "Was the door open this morning"),
    ("He has been working hard", "Has he been working hard"),
    ("They were ready on time", "Were they ready on time"),
    ("The plan was successful", "Was the plan successful"),
    ("She will accept the offer", "Will she accept the offer"),
    ("He can drive the truck", "Can he drive the truck"),
    ("The children are playing outside", "Are the children playing outside"),
    ("She knows the correct answer", "Does she know the correct answer"),
    ("They enjoy the new movie", "Do they enjoy the new movie"),
    ("He likes the restaurant", "Does he like the restaurant"),
    ("The system works efficiently", "Does the system work efficiently"),
    ("We need more time to finish", "Do we need more time to finish"),
    ("She wants to learn the skill", "Does she want to learn the skill"),
    ("They believe in the mission", "Do they believe in the mission"),
    ("The project requires funding", "Does the project require funding"),
    ("He plays guitar every day", "Does he play guitar every day"),
]

# ===== 语义特征 (直接编码型) =====

# 扩充SENTIMENT到100对!
SENTIMENT = [
    ("I love this amazing product", "I hate this terrible product"),
    ("The experience was wonderful", "The experience was awful"),
    ("She felt joyful and grateful", "She felt angry and resentful"),
    ("This is the best day ever", "This is the worst day ever"),
    ("The gift was thoughtful and kind", "The insult was cruel and mean"),
    ("We had a fantastic celebration", "We had a dreadful argument"),
    ("The performance was outstanding", "The performance was pathetic"),
    ("He spoke with warmth and care", "He spoke with coldness and spite"),
    ("The results exceeded expectations", "The results fell below expectations"),
    ("She smiled with genuine happiness", "She frowned with deep sorrow"),
    ("The music was uplifting and inspiring", "The noise was annoying and disturbing"),
    ("His advice was helpful and wise", "His advice was harmful and foolish"),
    ("The meal was a delightful treat", "The meal was a terrible disappointment"),
    ("They showed great courage and bravery", "They showed great fear and cowardice"),
    ("The story had a happy ending", "The story had a tragic ending"),
    ("I admire her dedication and effort", "I despise her laziness and neglect"),
    ("The garden was a paradise of beauty", "The dump was a wasteland of ugliness"),
    ("She embraced him with love", "She rejected him with hatred"),
    ("The news brought hope and relief", "The news brought despair and panic"),
    ("His actions showed generosity", "His actions showed greed"),
    ("The atmosphere was friendly and warm", "The atmosphere was hostile and cold"),
    ("She praised his excellent work", "She criticized his poor work"),
    ("The day was filled with laughter", "The day was filled with tears"),
    ("We celebrated our great victory", "We mourned our bitter defeat"),
    ("The future looks bright and promising", "The future looks bleak and hopeless"),
    ("He welcomed the guests warmly", "He rejected the guests coldly"),
    ("The team showed great teamwork", "The team showed great conflict"),
    ("She appreciated the beautiful gesture", "She resented the ugly gesture"),
    ("The weather was sunny and pleasant", "The weather was stormy and miserable"),
    ("He found peace in the garden", "He found chaos in the garden"),
    ("The book was fascinating and engaging", "The book was boring and tedious"),
    ("She expressed genuine gratitude", "She expressed deep resentment"),
    ("The movie was thrilling and exciting", "The movie was dull and uninspiring"),
    ("He delivered a brilliant performance", "He delivered a mediocre performance"),
    ("The community was united and strong", "The community was divided and weak"),
    ("She felt confident and empowered", "She felt insecure and powerless"),
    ("The garden bloomed with vibrant colors", "The garden withered in gray decay"),
    ("He offered sincere compliments", "He made harsh insults"),
    ("The child was joyful and playful", "The child was miserable and withdrawn"),
    ("She spoke with gentle kindness", "She spoke with bitter cruelty"),
    ("The gift brought great pleasure", "The injury brought great pain"),
    ("He showed deep compassion", "He showed cold indifference"),
    ("The evening was romantic and magical", "The evening was awkward and dreadful"),
    ("She radiated positive energy", "She radiated negative energy"),
    ("The speech was inspiring and moving", "The speech was demoralizing and offensive"),
    ("He felt proud of his achievement", "He felt ashamed of his failure"),
    ("The festival was lively and colorful", "The funeral was somber and gray"),
    ("She received warm congratulations", "She received harsh criticism"),
    ("The project was a great success", "The project was a total failure"),
    ("He experienced profound joy", "He experienced profound grief"),
    ("The painting was exquisite and beautiful", "The painting was ugly and crude"),
    ("She was enthusiastic and motivated", "She was apathetic and discouraged"),
    ("The concert was magnificent and memorable", "The concert was forgettable and unpleasant"),
    ("He demonstrated noble character", "He demonstrated despicable behavior"),
    ("The proposal was welcomed eagerly", "The proposal was rejected firmly"),
    ("She cherished the precious memory", "She despised the painful memory"),
    ("The journey was pleasant and smooth", "The journey was difficult and rough"),
    ("He shared his abundant blessings", "He hoarded his scarce resources"),
    ("The relationship was harmonious and loving", "The relationship was toxic and hateful"),
    ("She enjoyed the delightful surprise", "She endured the devastating shock"),
    ("The landscape was breathtaking and stunning", "The landscape was depressing and bleak"),
    ("He acted with honor and integrity", "He acted with deceit and malice"),
    ("The news was encouraging and positive", "The news was discouraging and negative"),
    ("She felt safe and protected", "She felt vulnerable and threatened"),
    ("The outcome was favorable and rewarding", "The outcome was unfavorable and punishing"),
    ("He displayed remarkable generosity", "He displayed remarkable selfishness"),
    ("The message was hopeful and uplifting", "The message was hopeless and devastating"),
    ("She found comfort in the familiar", "She found distress in the unfamiliar"),
    ("The experience was enriching and valuable", "The experience was depleting and worthless"),
    ("He gave his wholehearted support", "He gave his bitter opposition"),
    ("The situation was calm and peaceful", "The situation was tense and chaotic"),
    ("She received abundant praise", "She received severe condemnation"),
    ("The result was excellent and outstanding", "The result was terrible and shameful"),
    ("He showed great empathy and understanding", "He showed great apathy and ignorance"),
    ("The design was elegant and refined", "The design was crude and tasteless"),
    ("She maintained a positive attitude", "She maintained a negative attitude"),
    ("The performance was graceful and smooth", "The performance was clumsy and awkward"),
    ("He expressed genuine affection", "He expressed genuine hostility"),
    ("The atmosphere was cheerful and bright", "The atmosphere was gloomy and dark"),
    ("She felt liberated and free", "She felt trapped and confined"),
    ("The discovery was exciting and groundbreaking", "The discovery was disappointing and trivial"),
    ("He built trust through honesty", "He destroyed trust through deception"),
    ("The environment was healthy and vibrant", "The environment was toxic and decaying"),
    ("She experienced blissful contentment", "She experienced agonizing torment"),
    ("The achievement was remarkable and historic", "The failure was catastrophic and devastating"),
    ("He offered genuine friendship", "He offered false enmity"),
    ("The transition was smooth and seamless", "The transition was rough and problematic"),
    ("She felt respected and valued", "She felt disrespected and ignored"),
    ("The improvement was significant and meaningful", "The decline was significant and alarming"),
    ("He brought light into the room", "He brought darkness into the room"),
    ("The connection was deep and authentic", "The disconnection was deep and painful"),
    ("She showed remarkable resilience", "She showed remarkable fragility"),
    ("The solution was elegant and effective", "The problem was complex and intractable"),
    ("He earned well-deserved recognition", "He suffered unjust punishment"),
    ("The blessing was received with gratitude", "The curse was received with horror"),
    ("She celebrated the wonderful milestone", "She mourned the terrible loss"),
    ("The growth was steady and sustainable", "The decline was rapid and catastrophic"),
    ("He felt a sense of belonging", "He felt a sense of alienation"),
    ("The harmony was beautiful and perfect", "The discord was ugly and unbearable"),
    ("She expressed heartfelt appreciation", "She expressed bitter complaint"),
    ("The blessing brought inner peace", "The curse brought inner turmoil"),
    ("He found the experience rewarding", "He found the experience punishing"),
]

# 扩充SEMANTIC_TOPIC到100对!
SEMANTIC_TOPIC = [
    ("The doctor examined the patient carefully", "The chef prepared the meal carefully"),
    ("Scientists discovered a new particle", "Artists created a new painting"),
    ("The engine roared to life", "The orchestra played to life"),
    ("She solved the complex equation", "She wrote the complex poem"),
    ("The rocket launched into orbit", "The ship sailed into harbor"),
    ("He programmed the computer algorithm", "He composed the musical symphony"),
    ("The experiment yielded interesting data", "The novel yielded interesting insights"),
    ("The telescope observed distant galaxies", "The microscope observed tiny organisms"),
    ("She calculated the trajectory precisely", "She choreographed the dance precisely"),
    ("The reactor generated enormous power", "The storm generated enormous waves"),
    ("The mathematician proved the theorem", "The philosopher proved the argument"),
    ("The satellite orbited the planet", "The moon orbited the earth"),
    ("He analyzed the chemical compound", "He analyzed the literary work"),
    ("The bridge spanned the wide river", "The rainbow spanned the wide sky"),
    ("The laser cut through the metal", "The scissors cut through the fabric"),
    ("She researched the historical period", "She explored the geographical region"),
    ("The microscope revealed cell structures", "The telescope revealed star patterns"),
    ("The algorithm sorted the database", "The librarian sorted the collection"),
    ("The vaccine prevented the disease", "The umbrella prevented the soaking"),
    ("The code compiled without errors", "The speech delivered without pauses"),
    ("The surgeon performed the operation", "The conductor performed the symphony"),
    ("He measured the voltage precisely", "He measured the rhythm precisely"),
    ("The factory produced steel beams", "The bakery produced fresh bread"),
    ("She calibrated the microscope lens", "She tuned the piano keys"),
    ("The electron orbited the nucleus", "The dancer orbited the stage"),
    ("He synthesized the new compound", "He choreographed the new routine"),
    ("The database stored the records", "The library stored the manuscripts"),
    ("She optimized the neural network", "She perfected the painting technique"),
    ("The valve controlled the pressure", "The conductor controlled the tempo"),
    ("He diagnosed the mechanical fault", "He interpreted the literary symbol"),
    ("The circuit carried the current", "The melody carried the emotion"),
    ("She engineered the support structure", "She designed the artistic composition"),
    ("The catalyst accelerated the reaction", "The inspiration accelerated the creation"),
    ("He decoded the genetic sequence", "He decoded the poetic meaning"),
    ("The turbine generated electricity", "The orchestra generated emotion"),
    ("She verified the experimental results", "She validated the artistic vision"),
    ("The sensor detected the radiation", "The critic detected the symbolism"),
    ("He formulated the scientific theory", "He crafted the literary narrative"),
    ("The processor computed the solution", "The author composed the story"),
    ("She modeled the physical system", "She sketched the portrait"),
    ("The robot assembled the components", "The artist assembled the collage"),
    ("He translated the programming code", "He translated the foreign text"),
    ("The thermometer measured the heat", "The scale measured the harmony"),
    ("She debugged the software error", "She revised the draft essay"),
    ("The antenna received the signal", "The ear received the music"),
    ("He simulated the fluid dynamics", "He imagined the fictional world"),
    ("The experiment tested the hypothesis", "The audition tested the performance"),
    ("She published the research paper", "She exhibited the art installation"),
    ("The reactor sustained the chain reaction", "The choir sustained the harmony"),
    ("He specialized in quantum physics", "He specialized in classical music"),
    ("The telescope focused the light rays", "The camera focused the scene"),
    ("She developed the mathematical proof", "She developed the dramatic plot"),
    ("The microscope magnified the specimen", "The speaker amplified the message"),
    ("He investigated the crime scene", "He explored the fictional setting"),
    ("The computer processed the data stream", "The mind processed the sensory input"),
    ("She constructed the engineering model", "She built the theatrical set"),
    ("The formula predicted the outcome", "The script predicted the dialogue"),
    ("He interpreted the statistical results", "He interpreted the poetic imagery"),
    ("The spectrometer analyzed the spectrum", "The critic analyzed the artwork"),
    ("She mapped the genome sequence", "She mapped the story arc"),
    ("The satellite transmitted the image", "The broadcaster transmitted the program"),
    ("He repaired the broken circuit", "He restored the damaged painting"),
    ("The laboratory maintained strict protocols", "The studio maintained creative standards"),
    ("She patented the new invention", "She copyrighted the new composition"),
    ("The instrument recorded the measurements", "The journal recorded the observations"),
    ("He derived the mathematical equation", "He wrote the poetic verse"),
    ("The system processed the input signals", "The brain processed the sensory data"),
    ("She designed the control system", "She designed the stage lighting"),
    ("The battery stored the electrical energy", "The poem stored the emotional energy"),
    ("He programmed the robotic arm", "He directed the theatrical performance"),
    ("The microscope examined the tissue sample", "The magnifying glass examined the painting"),
    ("She calculated the orbital mechanics", "She calculated the dance choreography"),
    ("The network transmitted the data packets", "The story transmitted the cultural values"),
    ("He invented the new technology", "He imagined the new fictional world"),
    ("The centrifuge separated the mixture", "The editor separated the chapters"),
    ("She presented the scientific findings", "She performed the musical piece"),
    ("The protocol ensured the safety", "The tradition ensured the quality"),
    ("He operated the complex machinery", "He played the complex instrument"),
    ("The calculator performed the computation", "The author performed the narration"),
    ("She validated the simulation model", "She validated the artistic interpretation"),
    ("The radar tracked the moving target", "The narrator tracked the plot development"),
    ("He navigated the scientific literature", "He navigated the literary canon"),
    ("The specimen was preserved in formaldehyde", "The artifact was preserved in the museum"),
    ("She tested the chemical properties", "She tested the dramatic timing"),
    ("The oscilloscope displayed the waveform", "The screen displayed the film scene"),
    ("He classified the biological species", "He categorized the literary genres"),
    ("The formula described the physical law", "The poem described the human experience"),
    ("She generated the numerical solution", "She generated the creative inspiration"),
    ("The apparatus measured the force", "The instrument measured the pitch"),
    ("He observed the chemical reaction", "He witnessed the dramatic scene"),
    ("The data confirmed the prediction", "The review confirmed the excellence"),
    ("She extracted the pure compound", "She distilled the essential meaning"),
    ("The algorithm optimized the parameters", "The editor optimized the prose"),
    ("He operated the scanning device", "He operated the recording equipment"),
    ("The model simulated the real world", "The novel depicted the human condition"),
    ("She evaluated the experimental design", "She judged the artistic merit"),
    ("The frequency indicated the energy level", "The rhythm indicated the emotional state"),
    ("He controlled the experimental conditions", "He controlled the narrative pace"),
    ("The result supported the theoretical framework", "The performance supported the artistic vision"),
    ("She computed the probability distribution", "She composed the musical arrangement"),
    ("The device measured the physical quantity", "The poem captured the emotional truth"),
]

PREFIXES = ["Actually, ", "In fact, ", "Indeed, ", "Clearly, ", "Surely, ",
            "Perhaps, ", "Maybe, ", "Certainly, ", "Obviously, ", "Naturally, "]
SUFFIXES = [" right now", " today", " this time", " as expected", " for sure"]

def gen_pairs(templates, n, seed=42):
    np.random.seed(seed)
    pairs = list(templates)
    while len(pairs) < n:
        idx = np.random.randint(len(templates))
        a, b = templates[idx]
        aug = np.random.randint(4)
        if aug == 0:
            p = PREFIXES[np.random.randint(len(PREFIXES))]
            pairs.append((p+a, p+b))
        elif aug == 1:
            s = SUFFIXES[np.random.randint(len(SUFFIXES))]
            pairs.append((a+s, b+s))
        elif aug == 2:
            p = PREFIXES[np.random.randint(len(PREFIXES))]
            s = SUFFIXES[np.random.randint(len(SUFFIXES))]
            pairs.append((p+a+s, p+b+s))
        else:
            pairs.append((a, b))
    return pairs[:n]


def fit_algebraic_model(layers, l2_values):
    x = np.array(layers, dtype=float)
    y = np.array(l2_values, dtype=float)
    results = {}
    for name, fit_fn in [
        ("linear", lambda: _polyfit(x, y, 1)),
        ("exponential", lambda: _exp_fit(x, y)),
        ("power_law", lambda: _power_fit(x, y)),
        ("logarithmic", lambda: _log_fit(x, y)),
    ]:
        try:
            results[name] = fit_fn()
        except:
            results[name] = {"r2": 0}
    return results

def _polyfit(x, y, deg):
    coeffs = np.polyfit(x, y, deg)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"a": float(coeffs[0]), "b": float(coeffs[1]), "r2": float(r2)}

def _exp_fit(x, y):
    y_pos = np.maximum(y, 1e-6)
    coeffs = np.polyfit(x, np.log(y_pos), 1)
    y_pred = np.exp(coeffs[1]) * np.exp(coeffs[0] * x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"a": float(np.exp(coeffs[1])), "b": float(coeffs[0]), "r2": float(r2)}

def _power_fit(x, y):
    x_pos = np.maximum(x, 1.0)
    y_pos = np.maximum(y, 1e-6)
    log_x, log_y = np.log(x_pos), np.log(y_pos)
    valid = np.isfinite(log_x) & np.isfinite(log_y)
    if valid.sum() < 2: return {"r2": 0}
    coeffs = np.polyfit(log_x[valid], log_y[valid], 1)
    y_pred = np.exp(coeffs[1]) * x_pos ** coeffs[0]
    ss_res = np.sum((y[valid] - y_pred[valid])**2)
    ss_tot = np.sum((y[valid] - np.mean(y[valid]))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"a": float(np.exp(coeffs[1])), "b": float(coeffs[0]), "r2": float(r2)}

def _log_fit(x, y):
    log_x = np.log(x + 1)
    valid = np.isfinite(log_x) & np.isfinite(y)
    if valid.sum() < 2: return {"r2": 0}
    coeffs = np.polyfit(log_x[valid], y[valid], 1)
    y_pred = coeffs[0] * log_x + coeffs[1]
    ss_res = np.sum((y[valid] - y_pred[valid])**2)
    ss_tot = np.sum((y[valid] - np.mean(y[valid]))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"a": float(coeffs[0]), "b": float(coeffs[1]), "r2": float(r2)}


def do_patching(model, tokenizer, device, layer_idx, a_text, b_text, component=None):
    """Single pair patching. Returns l2 norm of logit difference."""
    try:
        src_ids = tokenizer(a_text, return_tensors='pt')['input_ids'].to(device)
        clean_ids = tokenizer(b_text, return_tensors='pt')['input_ids'].to(device)
        layer = model.model.layers[layer_idx]

        if component is None:
            target = layer
        elif component == 'attn':
            target = layer.self_attn
        else:
            target = layer.mlp

        # Capture source activation
        src_val = {}
        def cap(mod, inp, out):
            if isinstance(out, tuple):
                src_val['h'] = out[0][0, -1, :].detach().clone()
            else:
                src_val['h'] = out[0, -1, :].detach().clone()

        h = target.register_forward_hook(cap)
        with torch.no_grad():
            _ = model(src_ids)
        h.remove()
        if 'h' not in src_val:
            return None
        sv = src_val['h']

        # Clean logits
        with torch.no_grad():
            clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()

        # Patched logits
        def phook(mod, inp, out, _sv=sv):
            if isinstance(out, tuple):
                out[0][0, -1, :] = _sv.to(out[0].device)
            else:
                out[0, -1, :] = _sv.to(out.device)

        h = target.register_forward_hook(phook)
        with torch.no_grad():
            patched_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
        h.remove()

        diff = patched_logits - clean_logits
        return torch.norm(diff).item()
    except Exception as e:
        return None


def save_incremental(out_dir, all_results):
    with open(out_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=120)
    args = parser.parse_args()

    out_dir = Path(f"results/causal_fiber/{args.model}_ccx")
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(out_dir / 'run.log')

    path = PATHS[args.model]
    print(f"\n{'='*60}")
    print(f"Phase CCX: {args.model} (n_pairs={args.n_pairs})")
    print(f"Path: {path}")
    print(f"{'='*60}")

    # Load model
    print(f"[{time.strftime('%H:%M:%S')}] Loading model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model in ["glm4", "deepseek7b"]:
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
        model = model.to('cuda')
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    print(f"[{time.strftime('%H:%M:%S')}] Loaded in {time.time()-t0:.1f}s: n_layers={n_layers}, device={device}", flush=True)

    N = args.n_pairs
    # 8层均匀采样
    resid_layers = [int(i * (n_layers - 1) / 7) for i in range(8)]
    print(f"Resid layers: {resid_layers}", flush=True)

    # 语法特征 (应该逐层递增)
    syntactic_features = {
        'tense': gen_pairs(TENSE, N),
        'polarity': gen_pairs(POLARITY, N),
        'number': gen_pairs(NUMBER, N),  # 现在有100模板!
        'negation': gen_pairs(NEGATION, N),
        'question': gen_pairs(QUESTION, N),
    }

    # 语义特征 (应该L0就大，几乎恒定)
    semantic_features = {
        'sentiment': gen_pairs(SENTIMENT, N),  # 现在有100模板!
        'semantic_topic': gen_pairs(SEMANTIC_TOPIC, N),  # 现在有100模板!
    }

    all_results = {}

    # ===== S1: Residual全层扫描 - 所有特征 =====
    print(f"\n{'='*60}\nS1: Residual全层扫描 (7 features x 8 layers x {N} pairs)\n{'='*60}", flush=True)
    resid_results = {}

    for layer_idx in resid_layers:
        layer_name = f'L{layer_idx}'
        resid_results[layer_name] = {}
        t_layer = time.time()

        for feat_name, pairs in {**syntactic_features, **semantic_features}.items():
            n_test = min(len(pairs), N)
            l2s = []

            for i in range(n_test):
                a_text, b_text = pairs[i]
                l2 = do_patching(model, tokenizer, device, layer_idx, a_text, b_text)
                if l2 is not None:
                    l2s.append(l2)

            if l2s:
                resid_results[layer_name][feat_name] = {
                    'mean_l2': float(np.mean(l2s)),
                    'median_l2': float(np.median(l2s)),
                    'std_l2': float(np.std(l2s)),
                    'n': len(l2s),
                }

        elapsed = time.time() - t_layer
        # Print summary for this layer
        parts = []
        for feat_name in ['tense', 'polarity', 'number', 'negation', 'question', 'sentiment', 'semantic_topic']:
            if feat_name in resid_results[layer_name]:
                d = resid_results[layer_name][feat_name]
                parts.append(f"{feat_name}={d['median_l2']:.0f}(n={d['n']})")
        print(f"  {layer_name} [{elapsed:.0f}s]: {', '.join(parts)}", flush=True)

        all_results['resid'] = resid_results
        save_incremental(out_dir, all_results)

    # ===== S2: Attn vs MLP (首尾层) =====
    print(f"\n{'='*60}\nS2: Attn vs MLP\n{'='*60}", flush=True)
    comp_layers = [0, n_layers - 1]
    comp_results = {}

    for component in ['attn', 'mlp']:
        comp_results[component] = {}
        for layer_idx in comp_layers:
            ln = f'L{layer_idx}'
            comp_results[component][ln] = {}
            for feat_name in ['tense', 'polarity', 'number', 'sentiment']:
                pairs = {**syntactic_features, **semantic_features}[feat_name]
                n_test = min(len(pairs), 80)
                l2s = []
                for i in range(n_test):
                    l2 = do_patching(model, tokenizer, device, layer_idx, pairs[i][0], pairs[i][1], component=component)
                    if l2 is not None:
                        l2s.append(l2)
                if l2s:
                    comp_results[component][ln][feat_name] = {
                        'median_l2': float(np.median(l2s)), 'n': len(l2s)
                    }
            all_results['components'] = comp_results
            save_incremental(out_dir, all_results)

    # Contribution
    contrib = {}
    for layer_idx in comp_layers:
        ln = f'L{layer_idx}'
        contrib[ln] = {}
        for feat in ['tense', 'polarity', 'number', 'sentiment']:
            a = comp_results['attn'].get(ln, {}).get(feat, {}).get('median_l2', 0)
            m = comp_results['mlp'].get(ln, {}).get(feat, {}).get('median_l2', 0)
            total = a + m if (a + m) > 0 else 1
            contrib[ln][feat] = {'attn_pct': round(a/total*100,1), 'mlp_pct': round(m/total*100,1)}
            print(f"  {ln} {feat}: attn={a:.1f}({a/total*100:.0f}%), mlp={m:.1f}({m/total*100:.0f}%)")
    all_results['contribution'] = contrib
    save_incremental(out_dir, all_results)

    # ===== S3: 代数拟合 + 对偶验证 =====
    print(f"\n{'='*60}\nS3: 代数拟合 + 对偶验证\n{'='*60}", flush=True)
    algebraic_fits = {}
    duality_metrics = {}

    for feat_name in ['tense', 'polarity', 'number', 'negation', 'question', 'sentiment', 'semantic_topic']:
        layers_x, l2_medians = [], []
        for layer_idx in resid_layers:
            key = f'L{layer_idx}'
            if key in resid_results and feat_name in resid_results[key]:
                layers_x.append(float(layer_idx))
                l2_medians.append(resid_results[key][feat_name]['median_l2'])

        if len(layers_x) >= 3:
            fits = fit_algebraic_model(layers_x, l2_medians)
            algebraic_fits[feat_name] = fits
            best = max(fits.keys(), key=lambda k: fits[k].get('r2', 0))
            best_r2 = fits[best].get('r2', 0)

            # 对偶指标: L0值 vs 末层增长倍数
            l0_val = l2_medians[0]
            last_val = l2_medians[-1]
            growth_ratio = last_val / l0_val if l0_val > 0 else 0

            # 斜率 (线性拟合)
            slope = fits.get('linear', {}).get('a', 0)
            # 截距 (线性拟合)
            intercept = fits.get('linear', {}).get('b', 0)
            # 斜率/截距比 = 相对增长强度
            relative_growth = slope / intercept if intercept > 0 else 0

            duality_metrics[feat_name] = {
                'L0_median': float(l0_val),
                'L_final_median': float(last_val),
                'growth_ratio': float(growth_ratio),
                'best_fit': best,
                'best_r2': float(best_r2),
                'linear_slope': float(slope),
                'linear_intercept': float(intercept),
                'relative_growth': float(relative_growth),
            }

            # 分类: 语法(L0小+增长>1.5) vs 语义(L0大+增长<1.5)
            category = "SYNTACTIC" if growth_ratio > 1.5 else "SEMANTIC"

            print(f"  {feat_name}: {best} R2={best_r2:.3f}, L0={l0_val:.0f}, "
                  f"L_final={last_val:.0f}, growth={growth_ratio:.2f}x, "
                  f"relative_growth={relative_growth:.4f} => {category}")

    all_results['algebraic_fits'] = algebraic_fits
    all_results['duality_metrics'] = duality_metrics
    save_incremental(out_dir, all_results)

    # ===== S4: 对偶假设统计检验 =====
    print(f"\n{'='*60}\nS4: 对偶假设统计检验\n{'='*60}", flush=True)

    syntactic_names = ['tense', 'polarity', 'number', 'negation', 'question']
    semantic_names = ['sentiment', 'semantic_topic']

    syn_growth = [duality_metrics[f]['growth_ratio'] for f in syntactic_names if f in duality_metrics]
    sem_growth = [duality_metrics[f]['growth_ratio'] for f in semantic_names if f in duality_metrics]
    syn_l0 = [duality_metrics[f]['L0_median'] for f in syntactic_names if f in duality_metrics]
    sem_l0 = [duality_metrics[f]['L0_median'] for f in semantic_names if f in duality_metrics]
    syn_rel = [duality_metrics[f]['relative_growth'] for f in syntactic_names if f in duality_metrics]
    sem_rel = [duality_metrics[f]['relative_growth'] for f in semantic_names if f in duality_metrics]

    print(f"  Syntactic growth_ratios: {syn_growth}")
    print(f"  Semantic growth_ratios: {sem_growth}")
    print(f"  Syntactic L0: {syn_l0}")
    print(f"  Semantic L0: {sem_l0}")
    print(f"  Syntactic relative_growth: {syn_rel}")
    print(f"  Semantic relative_growth: {sem_rel}")

    if syn_growth and sem_growth:
        from scipy import stats as scipy_stats
        # Mann-Whitney U test (样本量小, 不假设正态)
        try:
            u_growth, p_growth = scipy_stats.mannwhitneyu(syn_growth, sem_growth, alternative='greater')
            print(f"  Growth ratio: U={u_growth:.1f}, p={p_growth:.4f} (syntactic > semantic?)")
        except:
            p_growth = 1.0
            print(f"  Growth ratio: test failed")

        try:
            u_l0, p_l0 = scipy_stats.mannwhitneyu(sem_l0, syn_l0, alternative='greater')
            print(f"  L0 value: U={u_l0:.1f}, p={p_l0:.4f} (semantic > syntactic?)")
        except:
            p_l0 = 1.0
            print(f"  L0 value: test failed")

        duality_test = {
            'growth_p_value': float(p_growth),
            'l0_p_value': float(p_l0),
            'syntactic_growth_mean': float(np.mean(syn_growth)),
            'semantic_growth_mean': float(np.mean(sem_growth)),
            'syntactic_l0_mean': float(np.mean(syn_l0)),
            'semantic_l0_mean': float(np.mean(sem_l0)),
        }
        all_results['duality_test'] = duality_test
        save_incremental(out_dir, all_results)

    # ===== FINAL =====
    print(f"\n{'='*60}\nFINAL: {args.model}\n{'='*60}", flush=True)
    print(f"\nResidual median_l2:")
    for ln in sorted(resid_results.keys()):
        parts = []
        for feat in ['tense', 'polarity', 'number', 'negation', 'question', 'sentiment', 'semantic_topic']:
            if feat in resid_results[ln]:
                d = resid_results[ln][feat]
                parts.append(f"{feat}={d['median_l2']:.0f}(n={d['n']})")
        print(f"  {ln}: {', '.join(parts)}")

    print(f"\nDuality Classification:")
    for feat, dm in duality_metrics.items():
        cat = "SYNTACTIC" if dm['growth_ratio'] > 1.5 else "SEMANTIC"
        print(f"  {feat}: growth={dm['growth_ratio']:.2f}x, L0={dm['L0_median']:.0f}, "
              f"best={dm['best_fit']}(R2={dm['best_r2']:.3f}) => {cat}")

    print(f"\nDONE! Saved to {out_dir}", flush=True)


if __name__ == '__main__':
    main()
