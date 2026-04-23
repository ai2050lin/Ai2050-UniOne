"""
Phase CCXXX: 频分复用容量测试
================================================================
核心目标:
1. 扩展到10个语义特征, 验证正交性是否随特征数量增加而退化
2. 估计d_model维空间能支持多少个正交语义维度
3. 分析正交性退化的模式: 哪些特征对最容易重叠?
4. 验证: 句法特征(SYN) vs 语义特征(SEM)是否有不同的正交性?
5. 理论容量: 用随机向量基准对比, 10个正交方向是否超过随机期望?

关键假设:
  CCXXIX确认5个特征的频分复用(|cos|<0.10)。如果d_model=2560-4096,
  理论上可以支持d_model个正交方向。但实际能支持多少语义特征?
  如果10个特征仍然高度正交, 则频分复用容量远大于5。

样本量: 50对/特征, 10个特征, 30个测试句子
"""

import argparse
import json
import os
import time
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

import torch

# 扩展到10个语义特征
FEATURE_PAIRS = {
    # === 原有5个特征 (精简到50对) ===
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
            ("The clock strikes midnight", "The clock struck midnight"),
            ("She catches the morning bus", "She caught the morning bus"),
            ("The snow melts in spring", "The snow melted in spring"),
            ("He breaks the record", "He broke the record"),
            ("She chooses the blue pen", "She chose the blue pen"),
            ("The candle burns brightly", "The candle burned brightly"),
            ("He swims across the lake", "He swam across the lake"),
            ("The phone rings unexpectedly", "The phone rang unexpectedly"),
            ("She forgets the password", "She forgot the password"),
            ("The boat sails across the bay", "The boat sailed across the bay"),
            ("He understands the problem", "He understood the problem"),
            ("The door opens slowly", "The door opened slowly"),
            ("She feels happy today", "She felt happy today"),
            ("He leaves the office early", "He left the office early"),
            ("The engine starts smoothly", "The engine started smoothly"),
            ("She spends time with friends", "She spent time with friends"),
            ("The cat chases the mouse", "The cat chased the mouse"),
            ("He brings the groceries home", "He brought the groceries home"),
            ("The flower blooms in spring", "The flower bloomed in spring"),
            ("She teaches the students", "She taught the students"),
            ("The dog digs a hole", "The dog dug a hole"),
            ("He draws a picture", "He drew a picture"),
            ("The fire burns hot", "The fire burned hot"),
        ]
    },
    'polarity': {
        'type': 'SYN',
        'pairs': [
            ("The room was very bright", "The room was very dark"),
            ("She is extremely tall", "She is extremely short"),
            ("The movie was incredibly good", "The movie was incredibly bad"),
            ("He is always early", "He is always late"),
            ("The soup is very hot", "The soup is very cold"),
            ("She is very rich", "She is very poor"),
            ("The road was completely safe", "The road was completely dangerous"),
            ("He is deeply happy", "He is deeply sad"),
            ("The task was quite easy", "The task was quite hard"),
            ("She is very strong", "She is very weak"),
            ("He is incredibly fast", "He is incredibly slow"),
            ("The food was absolutely delicious", "The food was absolutely terrible"),
            ("She is extremely kind", "She is extremely cruel"),
            ("The building is very old", "The building is very new"),
            ("The story was really interesting", "The story was really boring"),
            ("She is very brave", "She is very cowardly"),
            ("He is quite generous", "He is quite selfish"),
            ("The city is very clean", "The city is very dirty"),
            ("She is extremely polite", "She is extremely rude"),
            ("The material is very soft", "The material is very hard"),
            ("He is very careful", "He is very careless"),
            ("She is very quiet", "She is very loud"),
            ("The room is very large", "The room is very small"),
            ("He is very patient", "He is very impatient"),
            ("The water is very deep", "The water is very shallow"),
            ("She is very humble", "She is very proud"),
            ("He is very honest", "He is very dishonest"),
            ("She is very calm", "She is very anxious"),
            ("The road is very wide", "The road is very narrow"),
            ("He is very healthy", "He is very sick"),
            ("She is very organized", "She is very messy"),
            ("The fabric is very thick", "The fabric is very thin"),
            ("He is very confident", "He is very timid"),
            ("She is very gentle", "She is very harsh"),
            ("The surface is very smooth", "The surface is very rough"),
            ("He is very loyal", "He is very treacherous"),
            ("She is very wise", "She is very foolish"),
            ("He is very optimistic", "He is very pessimistic"),
            ("She is very creative", "She is very conventional"),
            ("He is very frugal", "He is very wasteful"),
            ("She is very graceful", "She is very clumsy"),
            ("He is very forgiving", "He is very vengeful"),
            ("She is very active", "She is very passive"),
            ("The metal is very sharp", "The metal is very blunt"),
            ("The wood is very solid", "The wood is very hollow"),
            ("The sky is very clear", "The sky is very cloudy"),
            ("The star is very bright", "The star is very faint"),
            ("The light is very bright", "The light is very dim"),
            ("The valley is very deep", "The valley is very shallow"),
            ("The fabric is very tight", "The fabric is very loose"),
        ]
    },
    'voice': {
        'type': 'SYN',
        'pairs': [
            ("The cat chased the mouse", "The mouse was chased by the cat"),
            ("The chef cooked the meal", "The meal was cooked by the chef"),
            ("The teacher praised the student", "The student was praised by the teacher"),
            ("The company launched the product", "The product was launched by the company"),
            ("The artist painted the portrait", "The portrait was painted by the artist"),
            ("The scientist discovered the cure", "The cure was discovered by the scientist"),
            ("The author wrote the novel", "The novel was written by the author"),
            ("The builder constructed the house", "The house was constructed by the builder"),
            ("The musician composed the song", "The song was composed by the musician"),
            ("The director filmed the scene", "The scene was filmed by the director"),
            ("The mechanic repaired the engine", "The engine was repaired by the mechanic"),
            ("The doctor treated the patient", "The patient was treated by the doctor"),
            ("The judge sentenced the criminal", "The criminal was sentenced by the judge"),
            ("The editor published the article", "The article was published by the editor"),
            ("The designer created the logo", "The logo was created by the designer"),
            ("The baker made the bread", "The bread was made by the baker"),
            ("The farmer grew the crops", "The crops were grown by the farmer"),
            ("The police arrested the thief", "The thief was arrested by the police"),
            ("The team won the championship", "The championship was won by the team"),
            ("The committee approved the plan", "The plan was approved by the committee"),
            ("The government passed the law", "The law was passed by the government"),
            ("The school educated the children", "The children were educated by the school"),
            ("The nurse cared for the patient", "The patient was cared for by the nurse"),
            ("The manager hired the employee", "The employee was hired by the manager"),
            ("The pilot flew the airplane", "The airplane was flown by the pilot"),
            ("The chef prepared the dessert", "The dessert was prepared by the chef"),
            ("The writer translated the book", "The book was translated by the writer"),
            ("The coach trained the athlete", "The athlete was trained by the coach"),
            ("The engineer designed the bridge", "The bridge was designed by the engineer"),
            ("The researcher analyzed the data", "The data was analyzed by the researcher"),
        ]
    },
    'semantic_valence': {
        'type': 'SEM',
        'pairs': [
            ("She received a wonderful gift", "She received a terrible gift"),
            ("The news brought great joy", "The news brought great sorrow"),
            ("He achieved remarkable success", "He suffered devastating failure"),
            ("The garden looked absolutely stunning", "The garden looked absolutely dreadful"),
            ("She showed genuine kindness", "She showed genuine cruelty"),
            ("The performance was truly amazing", "The performance was truly awful"),
            ("He expressed deep gratitude", "He expressed deep resentment"),
            ("The experience was incredibly rewarding", "The experience was incredibly punishing"),
            ("She displayed exceptional courage", "She displayed exceptional cowardice"),
            ("The result was highly beneficial", "The result was highly harmful"),
            ("He felt overwhelming pride", "He felt overwhelming shame"),
            ("The outcome was completely positive", "The outcome was completely negative"),
            ("She demonstrated remarkable patience", "She demonstrated remarkable hostility"),
            ("The atmosphere was wonderfully peaceful", "The atmosphere was terribly hostile"),
            ("He showed extraordinary generosity", "He showed extraordinary greed"),
            ("The change brought immense relief", "The change brought immense distress"),
            ("She offered sincere appreciation", "She offered bitter criticism"),
            ("The event was tremendously exciting", "The event was tremendously depressing"),
            ("He experienced profound happiness", "He experienced profound misery"),
            ("The situation improved significantly", "The situation worsened significantly"),
            ("She found great comfort in friends", "She found great torment in enemies"),
            ("The discovery was extremely valuable", "The discovery was extremely worthless"),
            ("He gained tremendous respect", "He gained tremendous contempt"),
            ("The solution proved highly effective", "The solution proved highly useless"),
            ("She experienced pure delight", "She experienced pure anguish"),
            ("The journey was most pleasant", "The journey was most painful"),
            ("He received warm praise", "He received harsh condemnation"),
            ("The advice was incredibly helpful", "The advice was incredibly harmful"),
            ("She earned well-deserved recognition", "She earned well-deserved criticism"),
            ("He shared genuine love", "He shared genuine hatred"),
            ("She maintained strong hope", "She maintained strong despair"),
            ("The agreement brought lasting peace", "The agreement brought lasting conflict"),
            ("He provided essential support", "He provided severe opposition"),
            ("She showed boundless enthusiasm", "She showed boundless indifference"),
            ("He offered unconditional forgiveness", "He offered unconditional vengeance"),
            ("The memory brought gentle warmth", "The memory brought bitter coldness"),
            ("The creation was a magnificent triumph", "The creation was a miserable disaster"),
            ("The gift was deeply appreciated", "The insult was deeply resented"),
            ("The progress was remarkably swift", "The decline was remarkably steep"),
            ("The transformation was beautifully positive", "The transformation was horribly negative"),
        ]
    },
    'semantic_topic': {
        'type': 'SEM',
        'pairs': [
            ("The doctor treated the patient", "The chef treated the ingredients"),
            ("The lawyer argued the case", "The scientist argued the hypothesis"),
            ("The engineer built the bridge", "The artist built the sculpture"),
            ("The pilot flew the plane", "The sailor sailed the ship"),
            ("The teacher explained the theory", "The coach explained the strategy"),
            ("The farmer planted the seeds", "The programmer planted the ideas"),
            ("The soldier fought the battle", "The lawyer fought the case"),
            ("The nurse healed the wound", "The mechanic fixed the engine"),
            ("The judge delivered the verdict", "The professor delivered the lecture"),
            ("The banker managed the fund", "The conductor managed the orchestra"),
            ("The architect designed the building", "The poet designed the verse"),
            ("The detective investigated the crime", "The researcher investigated the phenomenon"),
            ("The surgeon performed the operation", "The musician performed the concert"),
            ("The driver navigated the road", "The captain navigated the ocean"),
            ("The programmer wrote the code", "The novelist wrote the story"),
            ("The fisherman caught the fish", "The photographer caught the moment"),
            ("The librarian organized the books", "The general organized the troops"),
            ("The painter decorated the wall", "The composer decorated the silence"),
            ("The mechanic tuned the engine", "The pianist tuned the piano"),
            ("The baker kneaded the dough", "The philosopher kneaded the concepts"),
            ("The tailor stitched the fabric", "The historian stitched the narrative"),
            ("The chemist mixed the solution", "The DJ mixed the tracks"),
            ("The judge ruled the court", "The king ruled the kingdom"),
            ("The coach trained the team", "The mentor trained the mind"),
            ("The builder laid the foundation", "The educator laid the groundwork"),
            ("The plumber fixed the pipe", "The diplomat fixed the relations"),
            ("The editor revised the manuscript", "The surgeon revised the procedure"),
            ("The accountant balanced the books", "The acrobat balanced the body"),
            ("The banker financed the startup", "The farmer financed the harvest"),
            ("The chef seasoned the soup", "The teacher seasoned the lecture"),
            ("The poet wrote the verse", "The lawyer wrote the contract"),
            ("The sailor navigated the ocean", "The driver navigated the traffic"),
            ("The biologist studied the cell", "The sociologist studied the group"),
            ("The musician composed the symphony", "The architect composed the skyline"),
            ("The novelist created the character", "The sculptor created the statue"),
            ("The philosopher questioned existence", "The scientist questioned the hypothesis"),
            ("The electrician wired the house", "The therapist wired the brain"),
        ]
    },
    # === 新增5个特征 ===
    'number': {
        'type': 'SYN',
        'pairs': [
            ("The cat runs fast", "The cats run fast"),
            ("A dog barks loudly", "Dogs bark loudly"),
            ("The child plays outside", "The children play outside"),
            ("A bird sings sweetly", "Birds sing sweetly"),
            ("The student reads the book", "The students read the book"),
            ("A flower blooms in spring", "Flowers bloom in spring"),
            ("The tree grows tall", "Trees grow tall"),
            ("A car drives down the street", "Cars drive down the street"),
            ("The house looks old", "Houses look old"),
            ("A fish swims upstream", "Fish swim upstream"),
            ("The star shines brightly", "Stars shine brightly"),
            ("A book lies on the table", "Books lie on the table"),
            ("The cloud drifts slowly", "Clouds drift slowly"),
            ("A river flows to the sea", "Rivers flow to the sea"),
            ("The mountain stands tall", "Mountains stand tall"),
            ("A leaf falls from the tree", "Leaves fall from the tree"),
            ("The candle burns steadily", "Candles burn steadily"),
            ("A bell rings in the distance", "Bells ring in the distance"),
            ("The door opens quietly", "Doors open quietly"),
            ("A window faces the garden", "Windows face the garden"),
            ("The baby smiles happily", "Babies smile happily"),
            ("A sheep grazes in the field", "Sheep graze in the field"),
            ("The wolf howls at night", "Wolves howl at night"),
            ("A mouse hides in the corner", "Mice hide in the corner"),
            ("The goose swims in the pond", "Geese swim in the pond"),
            ("A tooth hurts badly", "Teeth hurt badly"),
            ("The foot aches after running", "Feet ache after running"),
            ("A man walks down the road", "Men walk down the road"),
            ("The woman sings softly", "Women sing softly"),
            ("A child laughs at the joke", "Children laugh at the joke"),
            ("The person waits patiently", "People wait patiently"),
            ("A deer runs through the forest", "Deer run through the forest"),
            ("The knife cuts the bread", "Knives cut the bread"),
            ("A shelf holds the books", "Shelves hold the books"),
            ("The loaf sits on the counter", "Loaves sit on the counter"),
            ("A wolf prowls the territory", "Wolves prowl the territory"),
            ("The calf drinks the milk", "Calves drink the milk"),
            ("A half was given to each", "Halves were given to each"),
            ("The leaf turns brown", "Leaves turn brown"),
            ("A life was saved that day", "Lives were saved that day"),
        ]
    },
    'negation': {
        'type': 'SYN',
        'pairs': [
            ("She is happy with the result", "She is not happy with the result"),
            ("He understands the problem", "He does not understand the problem"),
            ("The project was successful", "The project was not successful"),
            ("They agree with the proposal", "They do not agree with the proposal"),
            ("The evidence supports the theory", "The evidence does not support the theory"),
            ("She likes the new design", "She does not like the new design"),
            ("He believes the story", "He does not believe the story"),
            ("The plan worked perfectly", "The plan did not work perfectly"),
            ("They trust the new leader", "They do not trust the new leader"),
            ("The medicine helped the patient", "The medicine did not help the patient"),
            ("She enjoys the concert", "She does not enjoy the concert"),
            ("He remembers the details", "He does not remember the details"),
            ("The system functions properly", "The system does not function properly"),
            ("They accept the invitation", "They do not accept the invitation"),
            ("The data confirms the hypothesis", "The data does not confirm the hypothesis"),
            ("She wants to continue", "She does not want to continue"),
            ("He knows the answer", "He does not know the answer"),
            ("The team won the match", "The team did not win the match"),
            ("They support the cause", "They do not support the cause"),
            ("The product meets the standards", "The product does not meet the standards"),
            ("She can solve the puzzle", "She cannot solve the puzzle"),
            ("He will attend the meeting", "He will not attend the meeting"),
            ("The dog likes the food", "The dog does not like the food"),
            ("They have finished the work", "They have not finished the work"),
            ("The car needs repair", "The car does not need repair"),
            ("She should apologize", "She should not apologize"),
            ("He could find the key", "He could not find the key"),
            ("The report includes the data", "The report does not include the data"),
            ("They must follow the rules", "They must not follow the rules"),
            ("She would recommend the book", "She would not recommend the book"),
            ("He always arrives early", "He never arrives early"),
            ("The store is open today", "The store is not open today"),
            ("She has seen the movie", "She has not seen the movie"),
            ("They were informed of the change", "They were not informed of the change"),
            ("The door was locked", "The door was not locked"),
            ("He had permission to enter", "He did not have permission to enter"),
            ("She got the promotion", "She did not get the promotion"),
            ("The machine is working", "The machine is not working"),
            ("They received the package", "They did not receive the package"),
            ("He found the solution", "He did not find the solution"),
        ]
    },
    'comparison': {
        'type': 'SYN',
        'pairs': [
            ("She is tall", "She is taller"),
            ("The building is large", "The building is larger"),
            ("He runs fast", "He runs faster"),
            ("The river is wide", "The river is wider"),
            ("The cake is sweet", "The cake is sweeter"),
            ("She is smart", "She is smarter"),
            ("The road is long", "The road is longer"),
            ("He is strong", "He is stronger"),
            ("The mountain is high", "The mountain is higher"),
            ("The book is thick", "The book is thicker"),
            ("She is brave", "She is braver"),
            ("The weather is warm", "The weather is warmer"),
            ("He is old", "He is older"),
            ("The light is bright", "The light is brighter"),
            ("She is kind", "She is kinder"),
            ("The water is deep", "The water is deeper"),
            ("He is rich", "He is richer"),
            ("The car is fast", "The car is faster"),
            ("She is young", "She is younger"),
            ("The house is big", "The house is bigger"),
            ("He is thin", "He is thinner"),
            ("The room is dark", "The room is darker"),
            ("She is pretty", "She is prettier"),
            ("The dog is small", "The dog is smaller"),
            ("He is heavy", "He is heavier"),
            ("The song is loud", "The song is louder"),
            ("She is gentle", "She is gentler"),
            ("The wind is soft", "The wind is softer"),
            ("He is fierce", "He is fiercer"),
            ("The star is bright", "The star is brighter"),
            ("She is clever", "She is cleverer"),
            ("The garden is beautiful", "The garden is more beautiful"),
            ("He is intelligent", "He is more intelligent"),
            ("The story is interesting", "The story is more interesting"),
            ("She is patient", "She is more patient"),
            ("The task is difficult", "The task is more difficult"),
            ("He is powerful", "He is more powerful"),
            ("The painting is valuable", "The painting is more valuable"),
            ("She is careful", "She is more careful"),
            ("The journey is dangerous", "The journey is more dangerous"),
        ]
    },
    'modality': {
        'type': 'SYN',
        'pairs': [
            ("She goes to work", "She must go to work"),
            ("He stays home", "He should stay home"),
            ("They finish the project", "They can finish the project"),
            ("The team wins the game", "The team will win the game"),
            ("She takes the exam", "She might take the exam"),
            ("He helps the neighbor", "He would help the neighbor"),
            ("They attend the meeting", "They could attend the meeting"),
            ("The student passes the test", "The student may pass the test"),
            ("She buys the car", "She will buy the car"),
            ("He accepts the offer", "He should accept the offer"),
            ("They solve the problem", "They can solve the problem"),
            ("The company expands", "The company must expand"),
            ("She writes the report", "She would write the report"),
            ("He joins the team", "He might join the team"),
            ("They complete the task", "They will complete the task"),
            ("The scientist publishes the paper", "The scientist can publish the paper"),
            ("She learns the language", "She should learn the language"),
            ("He finds the answer", "He could find the answer"),
            ("They win the competition", "They may win the competition"),
            ("The manager approves the plan", "The manager must approve the plan"),
            ("She visits the museum", "She will visit the museum"),
            ("He repairs the device", "He would repair the device"),
            ("They build the house", "They can build the house"),
            ("The chef creates the dish", "The chef might create the dish"),
            ("She delivers the speech", "She should deliver the speech"),
            ("He crosses the river", "He could cross the river"),
            ("They reach the summit", "They will reach the summit"),
            ("The artist paints the mural", "The artist may paint the mural"),
            ("She understands the concept", "She must understand the concept"),
            ("He remembers the date", "He would remember the date"),
            ("They discover the truth", "They can discover the truth"),
            ("The engine starts smoothly", "The engine will start smoothly"),
            ("She enjoys the meal", "She might enjoy the meal"),
            ("He follows the instructions", "He should follow the instructions"),
            ("They climb the mountain", "They could climb the mountain"),
            ("The bird flies south", "The bird will fly south"),
            ("She solves the equation", "She can solve the equation"),
            ("He reaches the destination", "He must reach the destination"),
            ("They finish the race", "They would finish the race"),
            ("The river overflows the bank", "The river may overflow the bank"),
        ]
    },
    'question': {
        'type': 'SYN',
        'pairs': [
            ("The cat sat on the mat", "Did the cat sit on the mat"),
            ("She likes chocolate", "Does she like chocolate"),
            ("He went to the store", "Did he go to the store"),
            ("They play tennis every week", "Do they play tennis every week"),
            ("The dog chased the ball", "Did the dog chase the ball"),
            ("She speaks French fluently", "Does she speak French fluently"),
            ("He finished the assignment", "Did he finish the assignment"),
            ("The children ate the cookies", "Did the children eat the cookies"),
            ("She understands the material", "Does she understand the material"),
            ("They built a sandcastle", "Did they build a sandcastle"),
            ("The train arrived on time", "Did the train arrive on time"),
            ("He wrote the letter yesterday", "Did he write the letter yesterday"),
            ("She won the competition", "Did she win the competition"),
            ("The birds flew south", "Did the birds fly south"),
            ("They found the treasure", "Did they find the treasure"),
            ("The teacher explained the rule", "Did the teacher explain the rule"),
            ("She bought a new car", "Did she buy a new car"),
            ("He solved the puzzle", "Did he solve the puzzle"),
            ("The rain stopped eventually", "Did the rain stop eventually"),
            ("They watched the sunset", "Did they watch the sunset"),
            ("The cake tastes delicious", "Does the cake taste delicious"),
            ("She reads every evening", "Does she read every evening"),
            ("He runs five miles daily", "Does he run five miles daily"),
            ("The flowers need water", "Do the flowers need water"),
            ("They study hard for exams", "Do they study hard for exams"),
            ("The movie starts at eight", "Does the movie start at eight"),
            ("She works at the hospital", "Does she work at the hospital"),
            ("He plays the guitar well", "Does he play the guitar well"),
            ("The soup needs more salt", "Does the soup need more salt"),
            ("They travel every summer", "Do they travel every summer"),
        ]
    },
}

feature_names = list(FEATURE_PAIRS.keys())

TEST_SENTENCES = [
    "The weather today is",
    "She looked at the",
    "He decided to",
    "The city was",
    "They found the",
    "After the storm the",
    "The scientist discovered",
    "She always wanted to",
    "The book describes",
    "He carefully opened the",
    "The river flows through",
    "She smiled at the",
    "The old man walked",
    "They built a new",
    "The music played softly",
    "He remembered the",
    "The garden was full of",
    "She wrote a long",
    "The children played in the",
    "He watched the",
    "The restaurant served",
    "She chose the",
    "The forest was dark and",
    "He finished the",
    "The road led to",
    "She noticed the",
    "The cat sat on the",
    "He finally understood the",
    "The flowers bloomed in the",
    "She carefully explained the",
]

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
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=50)
    parser.add_argument('--n_test', type=int, default=30)
    args = parser.parse_args()

    model_key = args.model
    n_pairs = args.n_pairs
    n_test = args.n_test
    config = MODEL_CONFIGS[model_key]

    out_dir = f'results/causal_fiber/{model_key}_ccxxx'
    os.makedirs(out_dir, exist_ok=True)
    log_path = f'{out_dir}/run.log'

    def log(msg):
        try:
            print(msg, flush=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
                f.flush()
        except Exception:
            pass

    log(f"=" * 70)
    log(f"Phase CCXXX: 频分复用容量测试 — {config['name']}")
    log(f"  n_features={len(feature_names)}, n_pairs={n_pairs}, n_test={n_test}")
    log(f"  时间={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sklearn.decomposition import PCA

    # ============================================================
    # S1: 加载模型
    # ============================================================
    log(f"\n--- S1: 加载模型 ---")

    path = config['path']
    t0 = time.time()

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
        model = model.to('cuda')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_time = time.time() - t0
    n_layers = config['n_layers']
    d_model = config['d_model']
    last_layer = n_layers - 1

    log(f"  加载完成: {load_time:.0f}s, n_layers={n_layers}, d_model={d_model}")

    # ============================================================
    # S2: 逐层收集10个特征的差分向量
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S2: 逐层收集10个特征差分向量")
    log(f"{'=' * 60}")

    sample_layers = list(range(0, n_layers, 4))  # 每4层采样, 减少计算量
    if last_layer not in sample_layers:
        sample_layers.append(last_layer)
    sample_layers.sort()

    log(f"  采样层 ({len(sample_layers)}层): {sample_layers}")

    feat_layer_diffs = defaultdict(lambda: defaultdict(list))

    for feat in feature_names:
        pairs = FEATURE_PAIRS[feat]['pairs'][:n_pairs]
        log(f"  收集 {feat} ({len(pairs)}对, 类型={FEATURE_PAIRS[feat]['type']})...")

        for i, (s_pos, s_neg) in enumerate(pairs):
            try:
                with torch.no_grad():
                    toks_pos = tokenizer(s_pos, return_tensors='pt').to(model.device)
                    out_pos = model(**toks_pos, output_hidden_states=True)

                    toks_neg = tokenizer(s_neg, return_tensors='pt').to(model.device)
                    out_neg = model(**toks_neg, output_hidden_states=True)

                    for L in sample_layers:
                        h_pos = out_pos.hidden_states[L][0, -1, :].float().cpu().numpy()
                        h_neg = out_neg.hidden_states[L][0, -1, :].float().cpu().numpy()
                        diff = h_pos - h_neg
                        feat_layer_diffs[feat][L].append(diff)

                    del out_pos, out_neg

                if (i + 1) % 25 == 0:
                    log(f"    {feat}: {i+1}/{len(pairs)}")
            except Exception as e:
                continue

    # 计算每特征每层的PC1
    feat_layer_pc1 = {}  # (feat, L) -> pc1_dir

    for feat in feature_names:
        for L in sample_layers:
            diffs = feat_layer_diffs[feat][L]
            if len(diffs) >= 10:
                diffs = np.array(diffs)
                pca = PCA(n_components=1)
                pca.fit(diffs)
                feat_layer_pc1[(feat, L)] = pca.components_[0]

    log(f"\n  计算完成: {len(feat_layer_pc1)} 个(特征,层)组合")

    # ============================================================
    # S3: 10特征间正交性分析 — 核心实验
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S3: 10特征间正交性分析")
    log(f"{'=' * 60}")

    # 在最后一层计算10×10 cos矩阵
    log(f"\n  --- 最后一层(L{last_layer})特征间|cos|矩阵 ---")

    feat_pc1s_last = {}
    for feat in feature_names:
        if (feat, last_layer) in feat_layer_pc1:
            feat_pc1s_last[feat] = feat_layer_pc1[(feat, last_layer)]

    feats_present = sorted(feat_pc1s_last.keys())
    log(f"  有数据的特征: {len(feats_present)}/{len(feature_names)}")

    # 打印矩阵
    header = f"  {'':>18}"
    for f2 in feats_present:
        header += f" {f2[:6]:>6}"
    log(header)

    cos_matrix = {}
    for f1 in feats_present:
        row = f"  {f1:>18}"
        for f2 in feats_present:
            if f1 == f2:
                row += f" {'1.00':>6}"
            else:
                cos_val = abs(np.dot(feat_pc1s_last[f1], feat_pc1s_last[f2]) /
                             (np.linalg.norm(feat_pc1s_last[f1]) * np.linalg.norm(feat_pc1s_last[f2])))
                cos_matrix[(f1, f2)] = cos_val
                row += f" {cos_val:>6.2f}"
        log(row)

    # 统计
    all_cos = list(cos_matrix.values())
    mean_cos = np.mean(all_cos)
    max_cos = np.max(all_cos)
    median_cos = np.median(all_cos)
    pct_above_03 = sum(1 for c in all_cos if c > 0.3) / len(all_cos) * 100

    log(f"\n  统计:")
    log(f"    平均|cos| = {mean_cos:.4f}")
    log(f"    中位数|cos| = {median_cos:.4f}")
    log(f"    最大|cos| = {max_cos:.4f}")
    log(f"    |cos|>0.3的比例 = {pct_above_03:.1f}%")

    # 找出最正交和最重叠的对
    sorted_pairs = sorted(cos_matrix.items(), key=lambda x: x[1])
    log(f"\n  最正交的5对:")
    for (f1, f2), c in sorted_pairs[:5]:
        log(f"    {f1} × {f2}: |cos|={c:.4f}")
    log(f"  最重叠的5对:")
    for (f1, f2), c in sorted_pairs[-5:]:
        log(f"    {f1} × {f2}: |cos|={c:.4f}")

    # ============================================================
    # S4: SYN vs SEM正交性对比
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S4: SYN vs SEM正交性对比")
    log(f"{'=' * 60}")

    syn_feats = [f for f in feature_names if FEATURE_PAIRS[f]['type'] == 'SYN']
    sem_feats = [f for f in feature_names if FEATURE_PAIRS[f]['type'] == 'SEM']

    log(f"  SYN特征: {syn_feats}")
    log(f"  SEM特征: {sem_feats}")

    syn_syn_cos = []
    sem_sem_cos = []
    syn_sem_cos = []

    for (f1, f2), c in cos_matrix.items():
        t1 = FEATURE_PAIRS[f1]['type']
        t2 = FEATURE_PAIRS[f2]['type']
        if t1 == 'SYN' and t2 == 'SYN':
            syn_syn_cos.append(c)
        elif t1 == 'SEM' and t2 == 'SEM':
            sem_sem_cos.append(c)
        else:
            syn_sem_cos.append(c)

    log(f"\n  句法特征间(SYN×SYN): n={len(syn_syn_cos)}, 平均|cos|={np.mean(syn_syn_cos):.4f}")
    log(f"  语义特征间(SEM×SEM): n={len(sem_sem_cos)}, 平均|cos|={np.mean(sem_sem_cos):.4f}")
    log(f"  句法×语义(SYN×SEM): n={len(syn_sem_cos)}, 平均|cos|={np.mean(syn_sem_cos):.4f}")

    # ============================================================
    # S5: 子空间维度分析 — 10特征
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S5: 10特征子空间维度分析")
    log(f"{'=' * 60}")

    if len(feat_pc1s_last) >= 5:
        M_last = np.array([feat_pc1s_last[f] for f in feats_present])
        U_f, S_f, Vt_f = np.linalg.svd(M_last, full_matrices=False)

        log(f"  10特征PC1的SVD奇异值:")
        for i, s in enumerate(S_f):
            log(f"    σ{i+1} = {s:.4f} (累计方差: {np.cumsum(S_f**2)[i]/(S_f**2).sum()*100:.1f}%)")

        n_eff_50 = sum(1 for s in S_f if s > 0.5 * S_f[0])
        n_eff_30 = sum(1 for s in S_f if s > 0.3 * S_f[0])
        log(f"\n  有效维度(σ>0.5σ_max): {n_eff_50}")
        log(f"  有效维度(σ>0.3σ_max): {n_eff_30}")

        # 与随机基准对比
        log(f"\n  --- 随机基准对比 ---")
        n_random_trials = 100
        random_eff_dims = []
        random_mean_cos = []

        for _ in range(n_random_trials):
            # 10个随机方向
            rand_dirs = np.random.randn(len(feats_present), d_model)
            rand_dirs = rand_dirs / np.linalg.norm(rand_dirs, axis=1, keepdims=True)

            # SVD
            _, S_rand, _ = np.linalg.svd(rand_dirs, full_matrices=False)
            n_eff_rand = sum(1 for s in S_rand if s > 0.5 * S_rand[0])
            random_eff_dims.append(n_eff_rand)

            # 平均|cos|
            cos_vals = []
            for i in range(len(rand_dirs)):
                for j in range(i+1, len(rand_dirs)):
                    cos_vals.append(abs(np.dot(rand_dirs[i], rand_dirs[j])))
            random_mean_cos.append(np.mean(cos_vals))

        log(f"  随机10方向在d={d_model}中:")
        log(f"    平均有效维度: {np.mean(random_eff_dims):.1f} ± {np.std(random_eff_dims):.1f}")
        log(f"    平均|cos|: {np.mean(random_mean_cos):.4f} ± {np.std(random_mean_cos):.4f}")
        log(f"  实际10特征:")
        log(f"    有效维度: {n_eff_50}")
        log(f"    平均|cos|: {mean_cos:.4f}")

        if mean_cos < np.mean(random_mean_cos) - 2 * np.std(random_mean_cos):
            log(f"  → 实际正交性显著高于随机 → 语义特征有意正交!")
        elif mean_cos > np.mean(random_mean_cos) + 2 * np.std(random_mean_cos):
            log(f"  → 实际正交性显著低于随机 → 语义特征有意对齐!")
        else:
            log(f"  → 实际正交性与随机相当 → 正交性可能是高维空间的自然结果")

    # ============================================================
    # S6: 跨层正交性演化
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S6: 跨层正交性演化")
    log(f"{'=' * 60}")

    layer_mean_cos = {}
    for L in sample_layers:
        feat_pc1s_L = {}
        for feat in feature_names:
            if (feat, L) in feat_layer_pc1:
                feat_pc1s_L[feat] = feat_layer_pc1[(feat, L)]

        if len(feat_pc1s_L) < 5:
            continue

        cos_vals = []
        feats_L = sorted(feat_pc1s_L.keys())
        for i, f1 in enumerate(feats_L):
            for f2 in feats_L[i+1:]:
                c = abs(np.dot(feat_pc1s_L[f1], feat_pc1s_L[f2]) /
                        (np.linalg.norm(feat_pc1s_L[f1]) * np.linalg.norm(feat_pc1s_L[f2])))
                cos_vals.append(c)

        layer_mean_cos[L] = np.mean(cos_vals)

    log(f"  {'层':>4} | {'平均|cos|':>10} | {'特征数':>4}")
    log(f"  {'-'*4} | {'-'*10} | {'-'*4}")
    for L in sorted(layer_mean_cos.keys()):
        n_feats = sum(1 for f in feature_names if (f, L) in feat_layer_pc1)
        log(f"  L{L:>3} | {layer_mean_cos[L]:>10.4f} | {n_feats:>4}")

    # ============================================================
    # S7: 容量估计
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S7: 频分复用容量估计")
    log(f"{'=' * 60}")

    # 如果10个特征平均|cos|=X, 那么理论上可以支持多少个特征?
    # 在d维空间中, N个随机方向的平均|cos| ≈ 1/√d
    # 如果实际|cos| < 随机|cos|, 则模型有意维持正交性

    random_expected = 1.0 / np.sqrt(d_model)
    log(f"  d_model = {d_model}")
    log(f"  随机期望|cos| ≈ 1/√d = {random_expected:.4f}")
    log(f"  实际10特征平均|cos| = {mean_cos:.4f}")

    if mean_cos < random_expected:
        log(f"  → 实际正交性强于随机 → 模型有意维持正交性")
        # 估计容量: 正交性退化的临界N
        # 在d维空间中, N个方向的平均|cos| ≈ N/d (当N<<d时)
        # 当|cos|≈random_expected时, N ≈ d * random_expected = √d
        capacity = int(d_model * random_expected)
        log(f"  估计容量(正交性退化为随机时的特征数) ≈ √d = {capacity}")
    else:
        log(f"  → 实际正交性与随机相当")

    log(f"\n  理论上限: d_model={d_model}维空间中最多{d_model}个完全正交方向")
    log(f"  实际测试: 10个特征平均|cos|={mean_cos:.4f}")

    # ============================================================
    # S8: 汇总
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S8: 汇总")
    log(f"{'=' * 60}")

    log(f"\n  1. 10特征正交性:")
    log(f"     平均|cos| = {mean_cos:.4f}")
    log(f"     最大|cos| = {max_cos:.4f}")
    log(f"     随机期望 = {random_expected:.4f}")

    if 'n_eff_50' in dir():
        log(f"\n  2. 子空间维度:")
        log(f"     有效维度(σ>0.5σ_max) = {n_eff_50}")

    if syn_syn_cos and sem_sem_cos:
        log(f"\n  3. SYN vs SEM:")
        log(f"     SYN×SYN |cos| = {np.mean(syn_syn_cos):.4f}")
        log(f"     SEM×SEM |cos| = {np.mean(sem_sem_cos):.4f}")
        log(f"     SYN×SEM |cos| = {np.mean(syn_sem_cos):.4f}")

    log(f"\n  4. 容量估计:")
    log(f"     d_model = {d_model}")
    log(f"     理论最大正交方向 = {d_model}")
    log(f"     估计可用特征数 ≈ √d = {int(np.sqrt(d_model))}")

    # 保存结果
    results = {
        'model': config['name'],
        'n_features': len(feature_names),
        'mean_cos_10feat': float(mean_cos),
        'max_cos_10feat': float(max_cos),
        'random_expected_cos': float(random_expected),
        'syn_syn_mean_cos': float(np.mean(syn_syn_cos)) if syn_syn_cos else 0,
        'sem_sem_mean_cos': float(np.mean(sem_sem_cos)) if sem_sem_cos else 0,
        'syn_sem_mean_cos': float(np.mean(syn_sem_cos)) if syn_sem_cos else 0,
    }

    with open(f'{out_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    log(f"\n  结果已保存到 {out_dir}/results.json")
    log(f"\n{'=' * 70}")
    log(f"CCXXX 完成 — {config['name']}")
    log(f"{'=' * 70}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
