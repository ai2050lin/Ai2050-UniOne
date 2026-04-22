"""
Phase CCXIII: 微分向量几何分析 — 从L2范数到向量空间
核心突破:
  1. 收集实际的差分向量 (而非仅L2范数)
  2. PCA分析: 语法/语义差分向量的主成分
  3. 余弦相似度: 同类特征vs跨类特征
  4. 子空间正交性: 语法子空间与语义子空间是否正交?
  5. 纤维结构: 差分向量在层间的旋转/缩放轨迹

基于CCXII修正后的8语法+4语义分类:
  语法(8): tense, polarity, number, negation, question, person, definiteness, info_structure
  语义(4): sentiment, semantic_topic, voice, formality

目标样本: n=100 (向量收集开销大, 100足够)
层数: 4层 (L0, L33%, L66%, L_final)
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

# 精选模板 (每种30对, n=100时随机采样)
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
    ("The machine is working well", "The machine is not working well"),
]

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
]

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

PERSON = [
    ("I am going to the store", "She is going to the store"),
    ("I have finished the work", "She has finished the work"),
    ("I was running late today", "She was running late today"),
    ("I can solve the problem", "She can solve the problem"),
    ("I will attend the meeting", "She will attend the meeting"),
    ("I am happy with the result", "She is happy with the result"),
    ("I have seen the movie", "She has seen the movie"),
    ("I was working on the task", "She was working on the task"),
    ("I am reading the book", "She is reading the book"),
    ("I have written the essay", "She has written the essay"),
    ("I am enjoying the food", "She is enjoying the food"),
    ("I have visited the city", "She has visited the city"),
    ("I was studying the material", "She was studying the material"),
    ("I am learning the language", "She is learning the language"),
    ("I have completed the project", "She has completed the project"),
    ("I am cooking dinner now", "She is cooking dinner now"),
    ("I have bought the tickets", "She has bought the tickets"),
    ("I was watching the show", "She was watching the show"),
    ("I am feeling much better", "She is feeling much better"),
    ("I have heard the news", "She has heard the news"),
    ("I was playing the guitar", "She was playing the guitar"),
    ("I am taking the medicine", "She is taking the medicine"),
    ("I have read the report", "She has read the report"),
    ("I was driving the car", "She was driving the car"),
    ("I am building the model", "She is building the model"),
    ("I have made the decision", "She has made the decision"),
    ("I was painting the wall", "She was painting the wall"),
    ("I am writing the letter", "She is writing the letter"),
    ("I have found the solution", "She has found the solution"),
    ("I was teaching the class", "She was teaching the class"),
]

DEFINITENESS = [
    ("A cat sat on the mat", "The cat sat on the mat"),
    ("A student read the book", "The student read the book"),
    ("A doctor treated patients", "The doctor treated patients"),
    ("A teacher explained the rule", "The teacher explained the rule"),
    ("A bird flew over the lake", "The bird flew over the lake"),
    ("A car drove down the road", "The car drove down the road"),
    ("A dog chased the ball", "The dog chased the ball"),
    ("A child played in the park", "The child played in the park"),
    ("A woman walked to work", "The woman walked to work"),
    ("A man opened the door", "The man opened the door"),
    ("A flower grew in spring", "The flower grew in spring"),
    ("A river flowed through town", "The river flowed through town"),
    ("A star shone in the sky", "The star shone in the sky"),
    ("A house stood on the hill", "The house stood on the hill"),
    ("A tree fell in the storm", "The tree fell in the storm"),
    ("A book was on the shelf", "The book was on the shelf"),
    ("A song played on the radio", "The song played on the radio"),
    ("A train arrived at the station", "The train arrived at the station"),
    ("A boat sailed across the sea", "The boat sailed across the sea"),
    ("A plane flew through clouds", "The plane flew through clouds"),
    ("A cat slept on the sofa", "The cat slept on the sofa"),
    ("A horse ran around the track", "The horse ran around the track"),
    ("A fish swam in the pond", "The fish swam in the pond"),
    ("A rabbit hopped through the grass", "The rabbit hopped through the grass"),
    ("A fox crept through the forest", "The fox crept through the forest"),
    ("A bear climbed the mountain", "The bear climbed the mountain"),
    ("A snake slithered under the rock", "The snake slithered under the rock"),
    ("A deer grazed in the meadow", "The deer grazed in the meadow"),
    ("A wolf howled at the moon", "The wolf howled at the moon"),
    ("A hawk circled overhead", "The hawk circled overhead"),
]

INFO_STRUCTURE = [
    ("Mary broke the window", "It was Mary who broke the window"),
    ("John found the solution", "It was John who found the solution"),
    ("She wrote the report", "It was she who wrote the report"),
    ("They built the house", "It was they who built the house"),
    ("He solved the puzzle", "It was he who solved the puzzle"),
    ("We finished the project", "It was we who finished the project"),
    ("The dog chased the cat", "It was the dog that chased the cat"),
    ("The wind broke the branch", "It was the wind that broke the branch"),
    ("The rain ruined the crops", "It was the rain that ruined the crops"),
    ("The fire destroyed the building", "It was the fire that destroyed the building"),
    ("Tom designed the bridge", "It was Tom who designed the bridge"),
    ("Anna composed the music", "It was Anna who composed the music"),
    ("The team won the game", "It was the team that won the game"),
    ("The company launched the product", "It was the company that launched the product"),
    ("The teacher gave the assignment", "It was the teacher who gave the assignment"),
    ("Mike fixed the engine", "It was Mike who fixed the engine"),
    ("The student answered the question", "It was the student who answered the question"),
    ("Sarah painted the portrait", "It was Sarah who painted the portrait"),
    ("The manager approved the budget", "It was the manager who approved the budget"),
    ("The chef prepared the meal", "It was the chef who prepared the meal"),
    ("David wrote the software", "It was David who wrote the software"),
    ("The committee made the decision", "It was the committee that made the decision"),
    ("Lisa organized the event", "It was Lisa who organized the event"),
    ("The crew repaired the ship", "It was the crew that repaired the ship"),
    ("James discovered the error", "It was James who discovered the error"),
    ("The group completed the task", "It was the group that completed the task"),
    ("Emma translated the document", "It was Emma who translated the document"),
    ("The band performed the song", "It was the band that performed the song"),
    ("Robert managed the project", "It was Robert who managed the project"),
    ("The class passed the exam", "It was the class that passed the exam"),
]

SENTIMENT = [
    ("The happy child played outside", "The sad child played outside"),
    ("She gave a wonderful performance", "She gave a terrible performance"),
    ("He found a beautiful solution", "He found an ugly solution"),
    ("They enjoyed the delicious meal", "They suffered through the awful meal"),
    ("The warm sun brightened the day", "The cold rain darkened the day"),
    ("She received a generous gift", "She received a stingy gift"),
    ("He spoke with gentle words", "He spoke with harsh words"),
    ("The kind woman helped everyone", "The cruel woman hurt everyone"),
    ("They celebrated the joyful occasion", "They mourned the tragic occasion"),
    ("The peaceful garden felt calm", "The violent storm felt chaotic"),
    ("She wore a lovely dress", "She wore a hideous dress"),
    ("He told a fascinating story", "He told a boring story"),
    ("The clean room looked perfect", "The dirty room looked awful"),
    ("They built a magnificent castle", "They built a shabby shack"),
    ("The sweet music filled the air", "The harsh noise filled the air"),
    ("She made a brilliant decision", "She made a foolish decision"),
    ("He showed great courage today", "He showed great cowardice today"),
    ("The bright stars lit the sky", "The dark clouds covered the sky"),
    ("They shared a pleasant evening", "They endured a miserable evening"),
    ("The fresh flowers smelled nice", "The rotten flowers smelled bad"),
    ("She created an amazing painting", "She created a dreadful painting"),
    ("He gave an excellent presentation", "He gave a poor presentation"),
    ("The comfortable bed felt great", "The uncomfortable bed felt terrible"),
    ("They had a successful project", "They had a failed project"),
    ("The friendly dog wagged its tail", "The vicious dog bared its teeth"),
    ("She wrote an inspiring letter", "She wrote a depressing letter"),
    ("He made a wise choice", "He made a stupid choice"),
    ("The tasty cake was devoured", "The tasteless cake was rejected"),
    ("They achieved a remarkable victory", "They suffered a devastating defeat"),
    ("The cheerful song lifted spirits", "The sorrowful song brought tears"),
]

SEMANTIC_TOPIC = [
    ("The doctor examined the patient carefully", "The teacher examined the student carefully"),
    ("She cooked a delicious meal", "She programmed a complex algorithm"),
    ("The river flowed through the valley", "The current flowed through the circuit"),
    ("He planted seeds in the garden", "He invested money in the market"),
    ("The bird built a nest in spring", "The programmer built an app last year"),
    ("She painted a landscape with oils", "She designed a website with code"),
    ("The ship sailed across the ocean", "The data traveled across the network"),
    ("He climbed the mountain trail", "He solved the math equation"),
    ("The chef seasoned the soup perfectly", "The engineer calibrated the device perfectly"),
    ("She played violin in the orchestra", "She wrote code in the company"),
    ("The farmer harvested the wheat field", "The scientist analyzed the data field"),
    ("He rode his horse through the forest", "He drove his car through the city"),
    ("The artist sketched the portrait", "The architect drafted the blueprint"),
    ("She sang a beautiful melody", "She proved a beautiful theorem"),
    ("The fish swam in the clear lake", "The electron moved in the clear field"),
    ("He fixed the broken fence", "He debugged the broken software"),
    ("The baker kneaded the dough", "The writer drafted the chapter"),
    ("She grew roses in her garden", "She grew profits in her business"),
    ("The mechanic repaired the engine", "The doctor treated the patient"),
    ("He threw the ball to the catcher", "He sent the email to the manager"),
    ("The conductor led the symphony", "The manager led the project"),
    ("She wove the fabric on the loom", "She compiled the code on the server"),
    ("The sculptor carved the marble statue", "The developer built the mobile app"),
    ("He brewed the coffee every morning", "He reviewed the code every morning"),
    ("The dancer practiced the routine", "The researcher practiced the method"),
    ("She knitted the sweater by hand", "She configured the system by hand"),
    ("The pilot flew the passenger jet", "The analyst flew through the data"),
    ("He trimmed the hedge in the yard", "He optimized the query in the database"),
    ("The tailor sewed the custom suit", "The designer created the custom interface"),
    ("She polished the silver carefully", "She refined the model carefully"),
]

VOICE = [
    ("The cat chased the mouse", "The mouse was chased by the cat"),
    ("She wrote the report yesterday", "The report was written by her yesterday"),
    ("He fixed the broken window", "The broken window was fixed by him"),
    ("They built the new bridge", "The new bridge was built by them"),
    ("The teacher explained the lesson", "The lesson was explained by the teacher"),
    ("She cooked the delicious meal", "The delicious meal was cooked by her"),
    ("He painted the entire house", "The entire house was painted by him"),
    ("They discovered the ancient ruin", "The ancient ruin was discovered by them"),
    ("The company launched the product", "The product was launched by the company"),
    ("She translated the document", "The document was translated by her"),
    ("He designed the new logo", "The new logo was designed by him"),
    ("They organized the charity event", "The charity event was organized by them"),
    ("The chef prepared the special dish", "The special dish was prepared by the chef"),
    ("She composed the beautiful song", "The beautiful song was composed by her"),
    ("He directed the award-winning film", "The award-winning film was directed by him"),
    ("They published the research paper", "The research paper was published by them"),
    ("The artist created the sculpture", "The sculpture was created by the artist"),
    ("She managed the large project", "The large project was managed by her"),
    ("He developed the software tool", "The software tool was developed by him"),
    ("They renovated the old building", "The old building was renovated by them"),
    ("The team completed the mission", "The mission was completed by the team"),
    ("She wrote the best-selling novel", "The best-selling novel was written by her"),
    ("He solved the complex puzzle", "The complex puzzle was solved by him"),
    ("They implemented the new policy", "The new policy was implemented by them"),
    ("The scientist conducted the experiment", "The experiment was conducted by the scientist"),
    ("She recorded the hit song", "The hit song was recorded by her"),
    ("He engineered the suspension bridge", "The suspension bridge was engineered by him"),
    ("They manufactured the electric car", "The electric car was manufactured by them"),
    ("The council approved the budget", "The budget was approved by the council"),
    ("She delivered the keynote address", "The keynote address was delivered by her"),
]

FORMALITY = [
    ("It is imperative that we proceed", "We really need to get going"),
    ("I would like to request assistance", "I need some help"),
    ("The individual in question departed", "The guy left"),
    ("We shall commence the operation", "We'll start the job"),
    ("Please refrain from smoking herein", "Don't smoke here"),
    ("I am unable to attend the function", "I can't make it to the party"),
    ("The aforementioned document requires", "That paper needs"),
    ("It is recommended that you comply", "You should do it"),
    ("We regret to inform you that", "Sorry but we have to say"),
    ("The consumption of beverages is prohibited", "No drinks allowed"),
    ("I respectfully decline the invitation", "I'm passing on the invite"),
    ("Kindly submit your documentation", "Please send your stuff"),
    ("The procedure necessitates careful attention", "You gotta be careful with this"),
    ("We hereby acknowledge your correspondence", "We got your message"),
    ("It is advisable to consult a professional", "You should ask a pro"),
    ("The facility will be closed temporarily", "The place is closed for now"),
    ("I wish to express my gratitude", "Thanks a lot"),
    ("The meeting has been rescheduled", "The meeting got moved"),
    ("Please ensure punctuality for the event", "Don't be late"),
    ("We appreciate your cooperation in this matter", "Thanks for working with us on this"),
    ("The implementation requires additional resources", "We need more stuff to do this"),
    ("I must inform you of the changes", "Gotta tell you about the changes"),
    ("The circumstances dictate a different approach", "Things mean we need a new way"),
    ("Please verify the information provided", "Check the info you gave"),
    ("We anticipate significant improvements", "We expect things to get a lot better"),
    ("The analysis reveals several discrepancies", "The check found some mistakes"),
    ("It is essential to maintain accuracy", "You need to be right"),
    ("The organization is committed to excellence", "The company wants to do great"),
    ("I would appreciate your prompt response", "Please reply soon"),
    ("The current situation requires reassessment", "We need to rethink things"),
]

# 所有特征定义
ALL_FEATURES = {
    'tense': TENSE, 'polarity': POLARITY, 'number': NUMBER,
    'negation': NEGATION, 'question': QUESTION, 'person': PERSON,
    'definiteness': DEFINITENESS, 'info_structure': INFO_STRUCTURE,
    'sentiment': SENTIMENT, 'semantic_topic': SEMANTIC_TOPIC,
    'voice': VOICE, 'formality': FORMALITY,
}

SYNTACTIC = ['tense', 'polarity', 'number', 'negation', 'question', 'person', 'definiteness', 'info_structure']
SEMANTIC = ['sentiment', 'semantic_topic', 'voice', 'formality']

SUFFIXES = ["", " at this point", " in the end", " for sure", " without doubt",
            " as expected", " in reality", " on the whole", " after all", " by the way"]


def gen_pairs(templates, n):
    """Generate n pairs from templates with random suffix augmentation"""
    np.random.seed(42)
    pairs = []
    for i in range(n):
        idx = i % len(templates)
        a, b = templates[idx]
        if np.random.random() < 0.3:
            s = SUFFIXES[np.random.randint(len(SUFFIXES))]
            pairs.append((a + s, b + s))
        else:
            pairs.append((a, b))
    return pairs[:n]


def get_residual_vector(model, tokenizer, device, layer_idx, text):
    """Get the last-token residual stream vector at a specific layer"""
    try:
        input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
        activations = {}
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                activations['h'] = out[0][0, -1, :].detach().clone()
            else:
                activations['h'] = out[0, -1, :].detach().clone()

        layer = model.model.layers[layer_idx]
        h = layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(input_ids)
        h.remove()

        if 'h' in activations:
            return activations['h'].float().cpu().numpy()
        return None
    except:
        return None


def cosine_sim(v1, v2):
    """Cosine similarity between two vectors"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def subspace_overlap(V1, V2, k=5):
    """Measure overlap between two sets of vectors using PCA subspaces.
    V1: (n1, d) array, V2: (n2, d) array
    Returns: mean principal angle cosine (0=orthogonal, 1=identical)
    """
    from numpy.linalg import svd
    # PCA on each set
    V1c = V1 - V1.mean(axis=0, keepdims=True)
    V2c = V2 - V2.mean(axis=0, keepdims=True)
    
    try:
        U1, _, _ = svd(V1c, full_matrices=False)
        U2, _, _ = svd(V2c, full_matrices=False)
        k1 = min(k, U1.shape[1])
        k2 = min(k, U2.shape[1])
        # Subspace overlap = ||U1^T U2||_F / sqrt(k1*k2)
        overlap = np.linalg.norm(U1[:, :k1].T @ U2[:, :k2], 'fro') / np.sqrt(k1 * k2)
        return float(overlap)
    except:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(f"results/causal_fiber/{args.model}_ccxiii")
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(out_dir / 'run.log')

    path = PATHS[args.model]
    print(f"\n{'='*60}")
    print(f"Phase CCXIII: {args.model} (n_pairs={args.n_pairs}, Vector Geometry)")
    print(f"Path: {path}")
    print(f"{'='*60}")

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
    # 4层采样: L0, L33%, L66%, L_final
    sample_layers = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
    print(f"Sample layers (4): {sample_layers}", flush=True)

    # 生成所有特征对
    all_pairs = {}
    for feat_name, templates in ALL_FEATURES.items():
        all_pairs[feat_name] = gen_pairs(templates, N)

    # ===== S1: 收集差分向量 =====
    print(f"\n{'='*60}")
    print(f"S1: 收集差分向量 (12 features x 4 layers x {N} pairs)")
    print(f"{'='*60}", flush=True)

    # delta_vectors[feat_name][layer_idx] = (n, d) array of delta vectors
    delta_vectors = {}
    # resid_vectors[feat_name][layer_idx][a/b] = (n, d) array
    resid_vectors = {}

    for layer_idx in sample_layers:
        layer_name = f'L{layer_idx}'
        t_layer = time.time()

        for feat_name in list(ALL_FEATURES.keys()):
            if feat_name not in delta_vectors:
                delta_vectors[feat_name] = {}
                resid_vectors[feat_name] = {}

            pairs = all_pairs[feat_name]
            n_test = min(len(pairs), N)
            deltas = []
            a_vecs = []
            b_vecs = []

            for i in range(n_test):
                a_text, b_text = pairs[i]
                h_a = get_residual_vector(model, tokenizer, device, layer_idx, a_text)
                h_b = get_residual_vector(model, tokenizer, device, layer_idx, b_text)

                if h_a is not None and h_b is not None:
                    delta = h_a - h_b  # 差分向量
                    deltas.append(delta)
                    a_vecs.append(h_a)
                    b_vecs.append(h_b)

                # 每处理20对做一次GC
                if i > 0 and i % 20 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if deltas:
                delta_vectors[feat_name][layer_name] = np.array(deltas)
                resid_vectors[feat_name][layer_name] = {
                    'a': np.array(a_vecs),
                    'b': np.array(b_vecs),
                }

        elapsed = time.time() - t_layer
        parts = []
        for feat_name in list(ALL_FEATURES.keys()):
            if layer_name in delta_vectors.get(feat_name, {}):
                n = delta_vectors[feat_name][layer_name].shape[0]
                d = delta_vectors[feat_name][layer_name].shape[1]
                med_l2 = float(np.median(np.linalg.norm(delta_vectors[feat_name][layer_name], axis=1)))
                parts.append(f"{feat_name}={med_l2:.0f}(n={n},d={d})")
        print(f"  {layer_name} [{elapsed:.0f}s]: {', '.join(parts)}", flush=True)

    # 保存原始向量
    np.savez_compressed(
        out_dir / 'delta_vectors.npz',
        **{f"{feat}_{layer}": delta_vectors[feat][layer]
           for feat in delta_vectors for layer in delta_vectors[feat]}
    )
    print(f"\nSaved delta vectors to {out_dir / 'delta_vectors.npz'}", flush=True)

    # ===== S2: PCA分析 =====
    print(f"\n{'='*60}\nS2: PCA分析\n{'='*60}", flush=True)
    pca_results = {}

    for layer_name in [f'L{l}' for l in sample_layers]:
        pca_results[layer_name] = {}
        # 收集该层所有差分向量
        all_deltas = []
        labels = []  # 0=syntactic, 1=semantic
        feat_labels = []

        for feat_name in SYNTACTIC:
            if layer_name in delta_vectors.get(feat_name, {}):
                dv = delta_vectors[feat_name][layer_name]
                all_deltas.append(dv)
                labels.extend([0] * len(dv))
                feat_labels.extend([feat_name] * len(dv))

        for feat_name in SEMANTIC:
            if layer_name in delta_vectors.get(feat_name, {}):
                dv = delta_vectors[feat_name][layer_name]
                all_deltas.append(dv)
                labels.extend([1] * len(dv))
                feat_labels.extend([feat_name] * len(dv))

        if len(all_deltas) < 2:
            continue

        X = np.vstack(all_deltas)  # (total_n, d)
        y = np.array(labels)

        # PCA
        X_centered = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 方差解释比
        var_explained = (S ** 2) / (S ** 2).sum()
        cumvar = np.cumsum(var_explained)

        # PCA投影
        X_pca = X_centered @ Vt.T[:, :5]  # top 5 PCs

        # 语法vs语义在PC1上的分离度
        syn_mask = y == 0
        sem_mask = y == 1

        pc1_syn = X_pca[syn_mask, 0]
        pc1_sem = X_pca[sem_mask, 0]

        # Cohen's d
        if len(pc1_syn) > 1 and len(pc1_sem) > 1:
            pooled_std = np.sqrt((np.var(pc1_syn) + np.var(pc1_sem)) / 2)
            cohens_d = (np.mean(pc1_syn) - np.mean(pc1_sem)) / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0

        # 各特征在PC1-2上的质心
        centroids = {}
        for feat_name in list(ALL_FEATURES.keys()):
            mask = np.array([f == feat_name for f in feat_labels])
            if mask.any():
                centroids[feat_name] = {
                    'pc1': float(X_pca[mask, 0].mean()),
                    'pc2': float(X_pca[mask, 1].mean()),
                }

        pca_results[layer_name] = {
            'var_explained_top5': [float(v) for v in var_explained[:5]],
            'cumvar_top5': [float(v) for v in cumvar[:5]],
            'n_syn': int(syn_mask.sum()),
            'n_sem': int(sem_mask.sum()),
            'pc1_cohens_d': float(cohens_d),
            'centroids': centroids,
        }

        print(f"  {layer_name}: top5 var=[{', '.join(f'{v:.3f}' for v in var_explained[:5])}], "
              f"cumvar50 at PC{np.searchsorted(cumvar, 0.5)+1}, "
              f"PC1 Cohen d={cohens_d:.2f}")

    # ===== S3: 余弦相似度分析 =====
    print(f"\n{'='*60}\nS3: 余弦相似度分析\n{'='*60}", flush=True)
    cosine_results = {}

    for layer_name in [f'L{l}' for l in sample_layers]:
        # 计算各特征的均值差分向量
        mean_deltas = {}
        for feat_name in list(ALL_FEATURES.keys()):
            if layer_name in delta_vectors.get(feat_name, {}):
                mean_deltas[feat_name] = delta_vectors[feat_name][layer_name].mean(axis=0)

        if len(mean_deltas) < 2:
            continue

        # 同类余弦相似度
        syn_sims = []
        for i, f1 in enumerate(SYNTACTIC):
            for f2 in SYNTACTIC[i+1:]:
                if f1 in mean_deltas and f2 in mean_deltas:
                    sim = cosine_sim(mean_deltas[f1], mean_deltas[f2])
                    syn_sims.append(sim)

        sem_sims = []
        for i, f1 in enumerate(SEMANTIC):
            for f2 in SEMANTIC[i+1:]:
                if f1 in mean_deltas and f2 in mean_deltas:
                    sim = cosine_sim(mean_deltas[f1], mean_deltas[f2])
                    sem_sims.append(sim)

        # 跨类余弦相似度
        cross_sims = []
        for f1 in SYNTACTIC:
            for f2 in SEMANTIC:
                if f1 in mean_deltas and f2 in mean_deltas:
                    sim = cosine_sim(mean_deltas[f1], mean_deltas[f2])
                    cross_sims.append(sim)

        cosine_results[layer_name] = {
            'syn_within_mean': float(np.mean(syn_sims)) if syn_sims else 0,
            'syn_within_std': float(np.std(syn_sims)) if syn_sims else 0,
            'sem_within_mean': float(np.mean(sem_sims)) if sem_sims else 0,
            'sem_within_std': float(np.std(sem_sims)) if sem_sims else 0,
            'cross_mean': float(np.mean(cross_sims)) if cross_sims else 0,
            'cross_std': float(np.std(cross_sims)) if cross_sims else 0,
            'n_syn_pairs': len(syn_sims),
            'n_sem_pairs': len(sem_sims),
            'n_cross_pairs': len(cross_sims),
        }

        print(f"  {layer_name}: syn_within={np.mean(syn_sims):.3f}+-{np.std(syn_sims):.3f}, "
              f"sem_within={np.mean(sem_sims):.3f}+-{np.std(sem_sims):.3f}, "
              f"cross={np.mean(cross_sims):.3f}+-{np.std(cross_sims):.3f}")

    # ===== S4: 子空间正交性 =====
    print(f"\n{'='*60}\nS4: 子空间正交性\n{'='*60}", flush=True)
    subspace_results = {}

    for layer_name in [f'L{l}' for l in sample_layers]:
        # 收集语法和语义的差分向量矩阵
        syn_deltas = []
        for feat_name in SYNTACTIC:
            if layer_name in delta_vectors.get(feat_name, {}):
                syn_deltas.append(delta_vectors[feat_name][layer_name])

        sem_deltas = []
        for feat_name in SEMANTIC:
            if layer_name in delta_vectors.get(feat_name, {}):
                sem_deltas.append(delta_vectors[feat_name][layer_name])

        if not syn_deltas or not sem_deltas:
            continue

        syn_matrix = np.vstack(syn_deltas)  # (n_syn, d)
        sem_matrix = np.vstack(sem_deltas)  # (n_sem, d)

        # 子空间重叠度
        overlap_k3 = subspace_overlap(syn_matrix, sem_matrix, k=3)
        overlap_k5 = subspace_overlap(syn_matrix, sem_matrix, k=5)
        overlap_k10 = subspace_overlap(syn_matrix, sem_matrix, k=10)

        # 自身重叠度 (基线)
        syn_self_k5 = subspace_overlap(syn_matrix[:len(syn_matrix)//2], syn_matrix[len(syn_matrix)//2:], k=5)
        sem_self_k5 = subspace_overlap(sem_matrix[:len(sem_matrix)//2], sem_matrix[len(sem_matrix)//2:], k=5)

        subspace_results[layer_name] = {
            'cross_overlap_k3': overlap_k3,
            'cross_overlap_k5': overlap_k5,
            'cross_overlap_k10': overlap_k10,
            'syn_self_overlap_k5': syn_self_k5,
            'sem_self_overlap_k5': sem_self_k5,
        }

        print(f"  {layer_name}: cross_overlap(k=5)={overlap_k5:.3f}, "
              f"syn_self={syn_self_k5:.3f}, sem_self={sem_self_k5:.3f}")

    # ===== S5: 纤维轨迹分析 =====
    print(f"\n{'='*60}\nS5: 纤维轨迹分析 (层间旋转/缩放)\n{'='*60}", flush=True)
    fiber_results = {}

    for feat_name in list(ALL_FEATURES.keys()):
        mean_deltas_by_layer = {}
        for layer_idx in sample_layers:
            layer_name = f'L{layer_idx}'
            if layer_name in delta_vectors.get(feat_name, {}):
                mean_deltas_by_layer[layer_name] = delta_vectors[feat_name][layer_name].mean(axis=0)

        if len(mean_deltas_by_layer) < 2:
            continue

        layers_sorted = sorted(mean_deltas_by_layer.keys())
        rotations = []
        growths = []

        for i in range(len(layers_sorted) - 1):
            v1 = mean_deltas_by_layer[layers_sorted[i]]
            v2 = mean_deltas_by_layer[layers_sorted[i + 1]]

            # 余弦相似度 = 旋转程度
            sim = cosine_sim(v1, v2)
            rotations.append(sim)

            # L2范数比 = 缩放程度
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            growths.append(n2 / n1 if n1 > 1e-8 else 0)

        category = "SYN" if feat_name in SYNTACTIC else "SEM"

        fiber_results[feat_name] = {
            'category': category,
            'rotations': [float(r) for r in rotations],
            'growth_ratios': [float(g) for g in growths],
            'mean_rotation': float(np.mean(rotations)),
            'mean_growth': float(np.mean(growths)),
        }

        print(f"  {feat_name} [{category}]: "
              f"rotation={np.mean(rotations):.3f}, "
              f"growth={np.mean(growths):.2f}x")

    # ===== S6: 统计检验 =====
    print(f"\n{'='*60}\nS6: 语法vs语义纤维差异统计检验\n{'='*60}", flush=True)

    syn_rotations = [fiber_results[f]['mean_rotation'] for f in SYNTACTIC if f in fiber_results]
    sem_rotations = [fiber_results[f]['mean_rotation'] for f in SEMANTIC if f in fiber_results]
    syn_growths = [fiber_results[f]['mean_growth'] for f in SYNTACTIC if f in fiber_results]
    sem_growths = [fiber_results[f]['mean_growth'] for f in SEMANTIC if f in fiber_results]

    from scipy import stats as scipy_stats

    stat_results = {}
    if syn_rotations and sem_rotations:
        try:
            u_rot, p_rot = scipy_stats.mannwhitneyu(syn_rotations, sem_rotations, alternative='less')
            print(f"  Rotation (syn < sem?): U={u_rot:.1f}, p={p_rot:.4f}")
            stat_results['rotation_mw_p'] = float(p_rot)
        except:
            pass

    if syn_growths and sem_growths:
        try:
            u_grow, p_grow = scipy_stats.mannwhitneyu(syn_growths, sem_growths, alternative='greater')
            print(f"  Growth (syn > sem): U={u_grow:.1f}, p={p_grow:.4f}")
            stat_results['growth_mw_p'] = float(p_grow)
        except:
            pass

    print(f"  Syn rotation: {syn_rotations}")
    print(f"  Sem rotation: {sem_rotations}")
    print(f"  Syn growth: {[f'{g:.2f}' for g in syn_growths]}")
    print(f"  Sem growth: {[f'{g:.2f}' for g in sem_growths]}")

    # ===== 保存所有结果 =====
    all_results = {
        'pca': pca_results,
        'cosine': cosine_results,
        'subspace': subspace_results,
        'fiber': fiber_results,
        'statistics': stat_results,
    }

    # 处理numpy类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj

    with open(out_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    # ===== FINAL =====
    print(f"\n{'='*60}\nFINAL: {args.model}\n{'='*60}", flush=True)

    print(f"\n--- PCA: Variance Explained (top 5 PCs) ---")
    for ln in sorted(pca_results.keys()):
        r = pca_results[ln]
        print(f"  {ln}: {[f'{v:.3f}' for v in r['var_explained_top5']]}, PC1 Cohen d={r['pc1_cohens_d']:.2f}")

    print(f"\n--- Cosine Similarity ---")
    for ln in sorted(cosine_results.keys()):
        r = cosine_results[ln]
        print(f"  {ln}: syn_within={r['syn_within_mean']:.3f}, sem_within={r['sem_within_mean']:.3f}, cross={r['cross_mean']:.3f}")

    print(f"\n--- Subspace Overlap ---")
    for ln in sorted(subspace_results.keys()):
        r = subspace_results[ln]
        print(f"  {ln}: cross(k=5)={r['cross_overlap_k5']:.3f}, syn_self={r['syn_self_overlap_k5']:.3f}")

    print(f"\n--- Fiber Trajectories ---")
    for feat_name in SYNTACTIC + SEMANTIC:
        if feat_name in fiber_results:
            fr = fiber_results[feat_name]
            print(f"  {feat_name} [{fr['category']}]: rotation={fr['mean_rotation']:.3f}, growth={fr['mean_growth']:.2f}x")

    print(f"\nDONE! Saved to {out_dir}")


if __name__ == '__main__':
    main()
