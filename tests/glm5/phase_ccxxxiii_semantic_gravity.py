"""
Phase CCXXXIII: 语义引力定量验证 — 特征交互分析与条件PCA
================================================================
核心目标:
1. 验证"语义引力模型": PC1|cos|高的特征对是否有更强的交互效应?
2. 条件PCA: 特征A的PC1方向是否依赖于特征B的取值?
3. 交互分析: 2×2因子设计, 测量特征对在隐空间中的非线性交互
4. 量化"语义引力": 交互强度与PC1|cos|的相关性

关键假设:
  语义引力模型预测:
  - PC1|cos|高的特征对(如tense×question): 交互强, 条件PCA角度大
  - PC1|cos|低的特征对(如voice×polarity): 交互弱, 条件PCA角度小
  - 交互强度与PC1|cos|正相关

方法:
  对每个特征对(A, B), 创建4路最小对:
    A0B0: 基线 (如: present, statement)
    A1B0: 只变A (如: past, statement)
    A0B1: 只变B (如: present, question)
    A1B1: 都变   (如: past, question)
  
  交互效应 = Δh(A1B1) - [Δh(A1B0) + Δh(A0B1)]
  交互比 = |交互效应| / |Δh(A1B1)|
  
  条件PCA: 比较PC1(A|B=0)和PC1(A|B=1)的角度
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

# 6个特征对, 每对40个基础句子 → 4路最小对
# 设计原则: 特征A和B可以独立变化
FOUR_WAY_PAIRS = {
    # 1. tense × question — CCXXXI发现高重叠(|cos|>0.44)
    'tense_x_question': {
        'feature_A': 'tense',  # present→past
        'feature_B': 'question',  # statement→question
        'expected_cos': 'HIGH',
        'sentences': [
            # (A0B0: present+statement, A1B0: past+statement, A0B1: present+question, A1B1: past+question)
            ("She walks to school", "She walked to school", "Does she walk to school", "Did she walk to school"),
            ("He runs in the park", "He ran in the park", "Does he run in the park", "Did he run in the park"),
            ("They play football", "They played football", "Do they play football", "Did they play football"),
            ("The cat sleeps on the mat", "The cat slept on the mat", "Does the cat sleep on the mat", "Did the cat sleep on the mat"),
            ("She sings beautifully", "She sang beautifully", "Does she sing beautifully", "Did she sing beautifully"),
            ("He writes a letter", "He wrote a letter", "Does he write a letter", "Did he write a letter"),
            ("They travel abroad", "They traveled abroad", "Do they travel abroad", "Did they travel abroad"),
            ("The dog barks loudly", "The dog barked loudly", "Does the dog bark loudly", "Did the dog bark loudly"),
            ("She cooks dinner", "She cooked dinner", "Does she cook dinner", "Did she cook dinner"),
            ("He drives carefully", "He drove carefully", "Does he drive carefully", "Did he drive carefully"),
            ("The bird flies south", "The bird flew south", "Does the bird fly south", "Did the bird fly south"),
            ("She reads the book", "She read the book", "Does she read the book", "Did she read the book"),
            ("They build houses", "They built houses", "Do they build houses", "Did they build houses"),
            ("The river flows north", "The river flowed north", "Does the river flow north", "Did the river flow north"),
            ("She teaches mathematics", "She taught mathematics", "Does she teach mathematics", "Did she teach mathematics"),
            ("He fixes computers", "He fixed computers", "Does he fix computers", "Did he fix computers"),
            ("The train arrives early", "The train arrived early", "Does the train arrive early", "Did the train arrive early"),
            ("She paints landscapes", "She painted landscapes", "Does she paint landscapes", "Did she paint landscapes"),
            ("They sell fresh bread", "They sold fresh bread", "Do they sell fresh bread", "Did they sell fresh bread"),
            ("He speaks three languages", "He spoke three languages", "Does he speak three languages", "Did he speak three languages"),
            ("The wind blows gently", "The wind blew gently", "Does the wind blow gently", "Did the wind blow gently"),
            ("She types documents", "She typed documents", "Does she type documents", "Did she type documents"),
            ("They harvest crops", "They harvested crops", "Do they harvest crops", "Did they harvest crops"),
            ("He delivers packages", "He delivered packages", "Does he deliver packages", "Did he deliver packages"),
            ("The bell rings loudly", "The bell rang loudly", "Does the bell ring loudly", "Did the bell ring loudly"),
            ("She wears a dress", "She wore a dress", "Does she wear a dress", "Did she wear a dress"),
            ("They dance at parties", "They danced at parties", "Do they dance at parties", "Did they dance at parties"),
            ("He catches the ball", "He caught the ball", "Does he catch the ball", "Did he catch the ball"),
            ("The baby cries often", "The baby cried often", "Does the baby cry often", "Did the baby cry often"),
            ("She opens the window", "She opened the window", "Does she open the window", "Did she open the window"),
            ("They win prizes", "They won prizes", "Do they win prizes", "Did they win prizes"),
            ("He breaks records", "He broke records", "Does he break records", "Did he break records"),
            ("The water boils rapidly", "The water boiled rapidly", "Does the water boil rapidly", "Did the water boil rapidly"),
            ("She feeds the cat", "She fed the cat", "Does she feed the cat", "Did she feed the cat"),
            ("They watch television", "They watched television", "Do they watch television", "Did they watch television"),
            ("He grows vegetables", "He grew vegetables", "Does he grow vegetables", "Did he grow vegetables"),
            ("The sun rises early", "The sun rose early", "Does the sun rise early", "Did the sun rise early"),
            ("She keeps secrets", "She kept secrets", "Does she keep secrets", "Did she keep secrets"),
            ("They practice yoga", "They practiced yoga", "Do they practice yoga", "Did they practice yoga"),
            ("He meets the president", "He met the president", "Does he meet the president", "Did he meet the president"),
        ]
    },
    # 2. tense × voice — CCXXXI发现中等重叠(|cos|~0.10-0.30)
    'tense_x_voice': {
        'feature_A': 'tense',  # present→past
        'feature_B': 'voice',  # active→passive
        'expected_cos': 'MODERATE',
        'sentences': [
            # (A0B0: present+active, A1B0: past+active, A0B1: present+passive, A1B1: past+passive)
            ("The cat chases the mouse", "The cat chased the mouse", "The mouse is chased by the cat", "The mouse was chased by the cat"),
            ("She writes the report", "She wrote the report", "The report is written by her", "The report was written by her"),
            ("He fixes the car", "He fixed the car", "The car is fixed by him", "The car was fixed by him"),
            ("They build the house", "They built the house", "The house is built by them", "The house was built by them"),
            ("The chef cooks the meal", "The chef cooked the meal", "The meal is cooked by the chef", "The meal was cooked by the chef"),
            ("She delivers the speech", "She delivered the speech", "The speech is delivered by her", "The speech was delivered by her"),
            ("He paints the fence", "He painted the fence", "The fence is painted by him", "The fence was painted by him"),
            ("They discover the treasure", "They discovered the treasure", "The treasure is discovered by them", "The treasure was discovered by them"),
            ("The teacher explains the lesson", "The teacher explained the lesson", "The lesson is explained by the teacher", "The lesson was explained by the teacher"),
            ("She directs the film", "She directed the film", "The film is directed by her", "The film was directed by her"),
            ("He composes the music", "He composed the music", "The music is composed by him", "The music was composed by him"),
            ("They publish the article", "They published the article", "The article is published by them", "The article was published by them"),
            ("The company launches the product", "The company launched the product", "The product is launched by the company", "The product was launched by the company"),
            ("She records the song", "She recorded the song", "The song is recorded by her", "The song was recorded by her"),
            ("He designs the building", "He designed the building", "The building is designed by him", "The building was designed by him"),
            ("They organize the event", "They organized the event", "The event is organized by them", "The event was organized by them"),
            ("The artist paints the portrait", "The artist painted the portrait", "The portrait is painted by the artist", "The portrait was painted by the artist"),
            ("She programs the software", "She programmed the software", "The software is programmed by her", "The software was programmed by her"),
            ("He repairs the roof", "He repaired the roof", "The roof is repaired by him", "The roof was repaired by him"),
            ("They manufacture the device", "They manufactured the device", "The device is manufactured by them", "The device was manufactured by them"),
            ("The police arrest the thief", "The police arrested the thief", "The thief is arrested by the police", "The thief was arrested by the police"),
            ("She washes the dishes", "She washed the dishes", "The dishes are washed by her", "The dishes were washed by her"),
            ("He drives the bus", "He drove the bus", "The bus is driven by him", "The bus was driven by him"),
            ("They clean the room", "They cleaned the room", "The room is cleaned by them", "The room was cleaned by them"),
            ("The wind blows the leaves", "The wind blew the leaves", "The leaves are blown by the wind", "The leaves were blown by the wind"),
            ("She types the letter", "She typed the letter", "The letter is typed by her", "The letter was typed by her"),
            ("He catches the fish", "He caught the fish", "The fish is caught by him", "The fish was caught by him"),
            ("They sell the house", "They sold the house", "The house is sold by them", "The house was sold by them"),
            ("The doctor examines the patient", "The doctor examined the patient", "The patient is examined by the doctor", "The patient was examined by the doctor"),
            ("She bakes the cake", "She baked the cake", "The cake is baked by her", "The cake was baked by her"),
            ("He breaks the window", "He broke the window", "The window is broken by him", "The window was broken by him"),
            ("They cut the grass", "They cut the grass", "The grass is cut by them", "The grass was cut by them"),
            ("The judge dismisses the case", "The judge dismissed the case", "The case is dismissed by the judge", "The case was dismissed by the judge"),
            ("She translates the document", "She translated the document", "The document is translated by her", "The document was translated by her"),
            ("He chairs the meeting", "He chaired the meeting", "The meeting is chaired by him", "The meeting was chaired by him"),
            ("They approve the budget", "They approved the budget", "The budget is approved by them", "The budget was approved by them"),
            ("The nurse gives the medicine", "The nurse gave the medicine", "The medicine is given by the nurse", "The medicine was given by the nurse"),
            ("She founded the company", "She founded the company", "The company is founded by her", "The company was founded by her"),
            ("He coaches the team", "He coached the team", "The team is coached by him", "The team was coached by him"),
            ("They install the equipment", "They installed the equipment", "The equipment is installed by them", "The equipment was installed by them"),
        ]
    },
    # 3. tense × polarity — 不同模型|cos|差异大
    'tense_x_polarity': {
        'feature_A': 'tense',  # present→past
        'feature_B': 'polarity',  # positive→negative
        'expected_cos': 'VARIABLE',
        'sentences': [
            # (A0B0: present+positive, A1B0: past+positive, A0B1: present+negative, A1B1: past+negative)
            ("She likes the design", "She liked the design", "She does not like the design", "She did not like the design"),
            ("He enjoys the concert", "He enjoyed the concert", "He does not enjoy the concert", "He did not enjoy the concert"),
            ("They support the idea", "They supported the idea", "They do not support the idea", "They did not support the idea"),
            ("The method works well", "The method worked well", "The method does not work well", "The method did not work well"),
            ("She understands the concept", "She understood the concept", "She does not understand the concept", "She did not understand the concept"),
            ("He appreciates the help", "He appreciated the help", "He does not appreciate the help", "He did not appreciate the help"),
            ("They value the feedback", "They valued the feedback", "They do not value the feedback", "They did not value the feedback"),
            ("The plan makes sense", "The plan made sense", "The plan does not make sense", "The plan did not make sense"),
            ("She trusts his judgment", "She trusted his judgment", "She does not trust his judgment", "She did not trust his judgment"),
            ("He believes the story", "He believed the story", "He does not believe the story", "He did not believe the story"),
            ("They accept the offer", "They accepted the offer", "They do not accept the offer", "They did not accept the offer"),
            ("The food tastes good", "The food tasted good", "The food does not taste good", "The food did not taste good"),
            ("She wants to go home", "She wanted to go home", "She does not want to go home", "She did not want to go home"),
            ("He remembers the event", "He remembered the event", "He does not remember the event", "He did not remember the event"),
            ("They need more time", "They needed more time", "They do not need more time", "They did not need more time"),
            ("The system runs smoothly", "The system ran smoothly", "The system does not run smoothly", "The system did not run smoothly"),
            ("She follows the rules", "She followed the rules", "She does not follow the rules", "She did not follow the rules"),
            ("He deserves the award", "He deserved the award", "He does not deserve the award", "He did not deserve the award"),
            ("They own a house", "They owned a house", "They do not own a house", "They did not own a house"),
            ("The evidence convinces him", "The evidence convinced him", "The evidence does not convince him", "The evidence did not convince him"),
            ("She speaks German", "She spoke German", "She does not speak German", "She did not speak German"),
            ("He manages the team", "He managed the team", "He does not manage the team", "He did not manage the team"),
            ("They share the profits", "They shared the profits", "They do not share the profits", "They did not share the profits"),
            ("The bridge is safe", "The bridge was safe", "The bridge is not safe", "The bridge was not safe"),
            ("She prefers tea", "She preferred tea", "She does not prefer tea", "She did not prefer tea"),
            ("He recognizes the voice", "He recognized the voice", "He does not recognize the voice", "He did not recognize the voice"),
            ("They respect the tradition", "They respected the tradition", "They do not respect the tradition", "They did not respect the tradition"),
            ("The answer is correct", "The answer was correct", "The answer is not correct", "The answer was not correct"),
            ("She considers the option", "She considered the option", "She does not consider the option", "She did not consider the option"),
            ("He admits the mistake", "He admitted the mistake", "He does not admit the mistake", "He did not admit the mistake"),
            ("They produce quality goods", "They produced quality goods", "They do not produce quality goods", "They did not produce quality goods"),
            ("The water is clean", "The water was clean", "The water is not clean", "The water was not clean"),
            ("She expects an apology", "She expected an apology", "She does not expect an apology", "She did not expect an apology"),
            ("He completes the task", "He completed the task", "He does not complete the task", "He did not complete the task"),
            ("They include everyone", "They included everyone", "They do not include everyone", "They did not include everyone"),
            ("The soup is hot", "The soup was hot", "The soup is not hot", "The soup was not hot"),
            ("She plays the piano", "She played the piano", "She does not play the piano", "She did not play the piano"),
            ("He drives a car", "He drove a car", "He does not drive a car", "He did not drive a car"),
            ("They speak English", "They spoke English", "They do not speak English", "They did not speak English"),
            ("The room is spacious", "The room was spacious", "The room is not spacious", "The room was not spacious"),
        ]
    },
    # 4. voice × polarity — CCXXXI发现低重叠(|cos|<0.16)
    'voice_x_polarity': {
        'feature_A': 'voice',  # active→passive
        'feature_B': 'polarity',  # positive→negative
        'expected_cos': 'LOW',
        'sentences': [
            # (A0B0: active+positive, A1B0: passive+positive, A0B1: active+negative, A1B1: passive+negative)
            ("The cat chased the mouse", "The mouse was chased by the cat", "The cat did not chase the mouse", "The mouse was not chased by the cat"),
            ("She wrote the report", "The report was written by her", "She did not write the report", "The report was not written by her"),
            ("He fixed the car", "The car was fixed by him", "He did not fix the car", "The car was not fixed by him"),
            ("They built the house", "The house was built by them", "They did not build the house", "The house was not built by them"),
            ("The chef cooked the meal", "The meal was cooked by the chef", "The chef did not cook the meal", "The meal was not cooked by the chef"),
            ("She delivered the speech", "The speech was delivered by her", "She did not deliver the speech", "The speech was not delivered by her"),
            ("He painted the fence", "The fence was painted by him", "He did not paint the fence", "The fence was not painted by him"),
            ("They discovered the treasure", "The treasure was discovered by them", "They did not discover the treasure", "The treasure was not discovered by them"),
            ("The teacher explained the rule", "The rule was explained by the teacher", "The teacher did not explain the rule", "The rule was not explained by the teacher"),
            ("She cleaned the room", "The room was cleaned by her", "She did not clean the room", "The room was not cleaned by her"),
            ("He solved the puzzle", "The puzzle was solved by him", "He did not solve the puzzle", "The puzzle was not solved by him"),
            ("They published the article", "The article was published by them", "They did not publish the article", "The article was not published by them"),
            ("The company launched the product", "The product was launched by the company", "The company did not launch the product", "The product was not launched by the company"),
            ("She directed the film", "The film was directed by her", "She did not direct the film", "The film was not directed by her"),
            ("He composed the music", "The music was composed by him", "He did not compose the music", "The music was not composed by him"),
            ("They organized the event", "The event was organized by them", "They did not organize the event", "The event was not organized by them"),
            ("The police arrested the thief", "The thief was arrested by the police", "The police did not arrest the thief", "The thief was not arrested by the police"),
            ("She washed the dishes", "The dishes were washed by her", "She did not wash the dishes", "The dishes were not washed by her"),
            ("He caught the fish", "The fish was caught by him", "He did not catch the fish", "The fish was not caught by him"),
            ("They sold the house", "The house was sold by them", "They did not sell the house", "The house was not sold by them"),
            ("The storm damaged the roof", "The roof was damaged by the storm", "The storm did not damage the roof", "The roof was not damaged by the storm"),
            ("She opened the door", "The door was opened by her", "She did not open the door", "The door was not opened by her"),
            ("He repaired the engine", "The engine was repaired by him", "He did not repair the engine", "The engine was not repaired by him"),
            ("They approved the budget", "The budget was approved by them", "They did not approve the budget", "The budget was not approved by them"),
            ("The committee rejected the proposal", "The proposal was rejected by the committee", "The committee did not reject the proposal", "The proposal was not rejected by the committee"),
            ("She translated the document", "The document was translated by her", "She did not translate the document", "The document was not translated by her"),
            ("He chaired the meeting", "The meeting was chaired by him", "He did not chair the meeting", "The meeting was not chaired by him"),
            ("They completed the project", "The project was completed by them", "They did not complete the project", "The project was not completed by them"),
            ("The dog bit the man", "The man was bitten by the dog", "The dog did not bite the man", "The man was not bitten by the dog"),
            ("She fed the cat", "The cat was fed by her", "She did not feed the cat", "The cat was not fed by her"),
            ("He taught the class", "The class was taught by him", "He did not teach the class", "The class was not taught by him"),
            ("They cleaned the beach", "The beach was cleaned by them", "They did not clean the beach", "The beach was not cleaned by them"),
            ("The wind blew the leaves", "The leaves were blown by the wind", "The wind did not blow the leaves", "The leaves were not blown by the wind"),
            ("She baked the bread", "The bread was baked by her", "She did not bake the bread", "The bread was not baked by her"),
            ("He planted the tree", "The tree was planted by him", "He did not plant the tree", "The tree was not planted by him"),
            ("They found the solution", "The solution was found by them", "They did not find the solution", "The solution was not found by them"),
            ("The manager hired the staff", "The staff was hired by the manager", "The manager did not hire the staff", "The staff was not hired by the manager"),
            ("She cut the cake", "The cake was cut by her", "She did not cut the cake", "The cake was not cut by her"),
            ("He broke the window", "The window was broken by him", "He did not break the window", "The window was not broken by him"),
            ("They won the championship", "The championship was won by them", "They did not win the championship", "The championship was not won by them"),
        ]
    },
    # 5. voice × question — 预期低重叠
    'voice_x_question': {
        'feature_A': 'voice',  # active→passive
        'feature_B': 'question',  # statement→question
        'expected_cos': 'LOW',
        'sentences': [
            # (A0B0: active+statement, A1B0: passive+statement, A0B1: active+question, A1B1: passive+question)
            ("The company develops software", "Software is developed by the company", "Does the company develop software", "Is software developed by the company"),
            ("She writes the report", "The report is written by her", "Does she write the report", "Is the report written by her"),
            ("He fixes the car", "The car is fixed by him", "Does he fix the car", "Is the car fixed by him"),
            ("They build houses", "Houses are built by them", "Do they build houses", "Are houses built by them"),
            ("The chef cooks the meal", "The meal is cooked by the chef", "Does the chef cook the meal", "Is the meal cooked by the chef"),
            ("She delivers the mail", "The mail is delivered by her", "Does she deliver the mail", "Is the mail delivered by her"),
            ("He paints the fence", "The fence is painted by him", "Does he paint the fence", "Is the fence painted by him"),
            ("They sell fresh bread", "Fresh bread is sold by them", "Do they sell fresh bread", "Is fresh bread sold by them"),
            ("The teacher explains the rule", "The rule is explained by the teacher", "Does the teacher explain the rule", "Is the rule explained by the teacher"),
            ("She directs the film", "The film is directed by her", "Does she direct the film", "Is the film directed by her"),
            ("He composes the music", "The music is composed by him", "Does he compose the music", "Is the music composed by him"),
            ("They publish the article", "The article is published by them", "Do they publish the article", "Is the article published by them"),
            ("The police arrest thieves", "Thieves are arrested by the police", "Do the police arrest thieves", "Are thieves arrested by the police"),
            ("She washes the dishes", "The dishes are washed by her", "Does she wash the dishes", "Are the dishes washed by her"),
            ("He catches the fish", "The fish is caught by him", "Does he catch the fish", "Is the fish caught by him"),
            ("They clean the room", "The room is cleaned by them", "Do they clean the room", "Is the room cleaned by them"),
            ("The wind blows the leaves", "The leaves are blown by the wind", "Does the wind blow the leaves", "Are the leaves blown by the wind"),
            ("She bakes the bread", "The bread is baked by her", "Does she bake the bread", "Is the bread baked by her"),
            ("He repairs the roof", "The roof is repaired by him", "Does he repair the roof", "Is the roof repaired by him"),
            ("They manufacture devices", "Devices are manufactured by them", "Do they manufacture devices", "Are devices manufactured by them"),
            ("The doctor examines patients", "Patients are examined by the doctor", "Does the doctor examine patients", "Are patients examined by the doctor"),
            ("She translates documents", "Documents are translated by her", "Does she translate documents", "Are documents translated by her"),
            ("He drives the bus", "The bus is driven by him", "Does he drive the bus", "Is the bus driven by him"),
            ("They organize events", "Events are organized by them", "Do they organize events", "Are events organized by them"),
            ("The judge dismisses cases", "Cases are dismissed by the judge", "Does the judge dismiss cases", "Are cases dismissed by the judge"),
            ("She programs software", "Software is programmed by her", "Does she program software", "Is software programmed by her"),
            ("He chairs the meeting", "The meeting is chaired by him", "Does he chair the meeting", "Is the meeting chaired by him"),
            ("They approve the budget", "The budget is approved by them", "Do they approve the budget", "Is the budget approved by them"),
            ("The nurse gives medicine", "Medicine is given by the nurse", "Does the nurse give medicine", "Is medicine given by the nurse"),
            ("She founded the company", "The company was founded by her", "Did she found the company", "Was the company founded by her"),
            ("He coached the team", "The team was coached by him", "Did he coach the team", "Was the team coached by him"),
            ("They installed equipment", "Equipment was installed by them", "Did they install equipment", "Was equipment installed by them"),
            ("The cat chased the mouse", "The mouse was chased by the cat", "Did the cat chase the mouse", "Was the mouse chased by the cat"),
            ("She painted the portrait", "The portrait was painted by her", "Did she paint the portrait", "Was the portrait painted by her"),
            ("He caught the ball", "The ball was caught by him", "Did he catch the ball", "Was the ball caught by him"),
            ("They sold the house", "The house was sold by them", "Did they sell the house", "Was the house sold by them"),
            ("The storm damaged the roof", "The roof was damaged by the storm", "Did the storm damage the roof", "Was the roof damaged by the storm"),
            ("She wrote the poem", "The poem was written by her", "Did she write the poem", "Was the poem written by her"),
            ("He broke the record", "The record was broken by him", "Did he break the record", "Was the record broken by him"),
            ("They won the prize", "The prize was won by them", "Did they win the prize", "Was the prize won by them"),
        ]
    },
    # 6. question × polarity — 预期低-中等重叠
    'question_x_polarity': {
        'feature_A': 'question',  # statement→question
        'feature_B': 'polarity',  # positive→negative
        'expected_cos': 'LOW-MODERATE',
        'sentences': [
            # (A0B0: statement+positive, A1B0: question+positive, A0B1: statement+negative, A1B1: question+negative)
            ("She likes the design", "Does she like the design", "She does not like the design", "Does she not like the design"),
            ("He enjoys the concert", "Does he enjoy the concert", "He does not enjoy the concert", "Does he not enjoy the concert"),
            ("They support the idea", "Do they support the idea", "They do not support the idea", "Do they not support the idea"),
            ("The method works well", "Does the method work well", "The method does not work well", "Does the method not work well"),
            ("She understands the concept", "Does she understand the concept", "She does not understand the concept", "Does she not understand the concept"),
            ("He appreciates the help", "Does he appreciate the help", "He does not appreciate the help", "Does he not appreciate the help"),
            ("They value the feedback", "Do they value the feedback", "They do not value the feedback", "Do they not value the feedback"),
            ("The plan makes sense", "Does the plan make sense", "The plan does not make sense", "Does the plan not make sense"),
            ("She trusts his judgment", "Does she trust his judgment", "She does not trust his judgment", "Does she not trust his judgment"),
            ("He believes the story", "Does he believe the story", "He does not believe the story", "Does he not believe the story"),
            ("They accept the offer", "Do they accept the offer", "They do not accept the offer", "Do they not accept the offer"),
            ("The food tastes good", "Does the food taste good", "The food does not taste good", "Does the food not taste good"),
            ("She wants to go home", "Does she want to go home", "She does not want to go home", "Does she not want to go home"),
            ("He remembers the event", "Does he remember the event", "He does not remember the event", "Does he not remember the event"),
            ("They need more time", "Do they need more time", "They do not need more time", "Do they not need more time"),
            ("The system runs smoothly", "Does the system run smoothly", "The system does not run smoothly", "Does the system not run smoothly"),
            ("She follows the rules", "Does she follow the rules", "She does not follow the rules", "Does she not follow the rules"),
            ("He deserves the award", "Does he deserve the award", "He does not deserve the award", "Does he not deserve the award"),
            ("They own a house", "Do they own a house", "They do not own a house", "Do they not own a house"),
            ("The evidence convinces him", "Does the evidence convince him", "The evidence does not convince him", "Does the evidence not convince him"),
            ("She speaks German", "Does she speak German", "She does not speak German", "Does she not speak German"),
            ("He manages the team", "Does he manage the team", "He does not manage the team", "Does he not manage the team"),
            ("They share the profits", "Do they share the profits", "They do not share the profits", "Do they not share the profits"),
            ("The bridge is safe", "Is the bridge safe", "The bridge is not safe", "Is the bridge not safe"),
            ("She prefers tea", "Does she prefer tea", "She does not prefer tea", "Does she not prefer tea"),
            ("He recognizes the voice", "Does he recognize the voice", "He does not recognize the voice", "Does he not recognize the voice"),
            ("They respect the tradition", "Do they respect the tradition", "They do not respect the tradition", "Do they not respect the tradition"),
            ("The answer is correct", "Is the answer correct", "The answer is not correct", "Is the answer not correct"),
            ("She considers the option", "Does she consider the option", "She does not consider the option", "Does she not consider the option"),
            ("He admits the mistake", "Does he admit the mistake", "He does not admit the mistake", "Does he not admit the mistake"),
            ("They produce quality goods", "Do they produce quality goods", "They do not produce quality goods", "Do they not produce quality goods"),
            ("The water is clean", "Is the water clean", "The water is not clean", "Is the water not clean"),
            ("She expects an apology", "Does she expect an apology", "She does not expect an apology", "Does she not expect an apology"),
            ("He completes the task", "Does he complete the task", "He does not complete the task", "Does he not complete the task"),
            ("They include everyone", "Do they include everyone", "They do not include everyone", "Do they not include everyone"),
            ("The soup is hot", "Is the soup hot", "The soup is not hot", "Is the soup not hot"),
            ("She plays the piano", "Does she play the piano", "She does not play the piano", "Does she not play the piano"),
            ("He drives a car", "Does he drive a car", "He does not drive a car", "Does he not drive a car"),
            ("They speak English", "Do they speak English", "They do not speak English", "Do they not speak English"),
            ("The room is spacious", "Is the room spacious", "The room is not spacious", "Is the room not spacious"),
        ]
    },
}


def load_model(model_key, device='cuda'):
    """加载模型和tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = MODEL_CONFIGS[model_key]
    path = config['path']
    dtype = config['dtype']

    print(f"  加载模型: {config['name']} ({dtype})")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dtype == '8bit':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
    elif dtype == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto",
            trust_remote_code=True, local_files_only=True
        )

    model.eval()
    return model, tokenizer


def get_last_token_hidden(model, tokenizer, text, device='cuda'):
    """获取最后一个token的隐藏状态(所有层)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    # 所有层的隐藏状态
    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, d_model)

    # 取最后一个非padding token
    seq_len = attention_mask.sum().item()
    last_hidden = torch.stack([h[0, seq_len-1, :].cpu().float() for h in hidden_states])

    return last_hidden  # (n_layers+1, d_model)


def compute_interaction_analysis(h_A0B0, h_A1B0, h_A0B1, h_A1B1):
    """
    2×2因子设计交互分析
    
    Δh_A = h(A1B0) - h(A0B0)   # A的主效应
    Δh_B = h(A0B1) - h(A0B0)   # B的主效应
    Δh_AB = h(A1B1) - h(A0B0)  # 组合效应
    交互 = Δh_AB - (Δh_A + Δh_B)  # 非线性交互
    
    返回: 交互比, Δh_A范数, Δh_B范数, 交互范数
    """
    dh_A = h_A1B0 - h_A0B0
    dh_B = h_A0B1 - h_A0B0
    dh_AB = h_A1B1 - h_A0B0

    interaction = dh_AB - (dh_A + dh_B)

    norm_dh_AB = torch.norm(dh_AB).item()
    norm_interaction = torch.norm(interaction).item()
    norm_dh_A = torch.norm(dh_A).item()
    norm_dh_B = torch.norm(dh_B).item()

    interaction_ratio = norm_interaction / max(norm_dh_AB, 1e-10)

    # 交互方向与主效应方向的关系
    if norm_dh_A > 1e-10 and norm_interaction > 1e-10:
        cos_int_A = torch.nn.functional.cosine_similarity(interaction.unsqueeze(0), dh_A.unsqueeze(0)).item()
    else:
        cos_int_A = 0.0

    if norm_dh_B > 1e-10 and norm_interaction > 1e-10:
        cos_int_B = torch.nn.functional.cosine_similarity(interaction.unsqueeze(0), dh_B.unsqueeze(0)).item()
    else:
        cos_int_B = 0.0

    # 加法预测误差
    if norm_dh_AB > 1e-10:
        additive_pred = dh_A + dh_B
        pred_error = torch.norm(dh_AB - additive_pred).item() / norm_dh_AB
    else:
        pred_error = 0.0

    return {
        'interaction_ratio': interaction_ratio,
        'norm_dh_A': norm_dh_A,
        'norm_dh_B': norm_dh_B,
        'norm_interaction': norm_interaction,
        'norm_dh_AB': norm_dh_AB,
        'cos_int_A': cos_int_A,
        'cos_int_B': cos_int_B,
        'additive_pred_error': pred_error,
    }


def compute_conditional_pca(dh_A_given_B0_list, dh_A_given_B1_list):
    """
    条件PCA: 比较PC1(A|B=0)和PC1(A|B=1)的方向
    
    dh_A_given_B0: B=0时A的差分列表
    dh_A_given_B1: B=1时A的差分列表
    """
    # 堆叠成矩阵
    mat_B0 = torch.stack(dh_A_given_B0_list)  # (n_samples, d_model)
    mat_B1 = torch.stack(dh_A_given_B1_list)

    # SVD获取PC1
    _, _, Vt_B0 = torch.linalg.svd(mat_B0, full_matrices=False)
    _, _, Vt_B1 = torch.linalg.svd(mat_B1, full_matrices=False)

    pc1_B0 = Vt_B0[0]  # (d_model,)
    pc1_B1 = Vt_B1[0]

    # PC1方向的余弦相似度
    cos_pc1 = torch.nn.functional.cosine_similarity(pc1_B0.unsqueeze(0), pc1_B1.unsqueeze(0)).item()

    # PC1解释方差
    var_B0 = torch.var(mat_B0 @ pc1_B0) / torch.var(mat_B0)
    var_B1 = torch.var(mat_B1 @ pc1_B1) / torch.var(mat_B1)

    # PC1-5解释方差
    n_pcs = min(5, Vt_B0.shape[0], Vt_B1.shape[0])
    var_explained_B0 = []
    var_explained_B1 = []
    for i in range(n_pcs):
        pc_B0 = Vt_B0[i]
        pc_B1 = Vt_B1[i]
        proj_var_B0 = torch.var(mat_B0 @ pc_B0).item()
        proj_var_B1 = torch.var(mat_B1 @ pc_B1).item()
        total_var_B0 = torch.sum(torch.var(mat_B0, dim=0)).item()
        total_var_B1 = torch.sum(torch.var(mat_B1, dim=0)).item()
        var_explained_B0.append(proj_var_B0 / max(total_var_B0, 1e-10))
        var_explained_B1.append(proj_var_B1 / max(total_var_B1, 1e-10))

    return {
        'cos_pc1': cos_pc1,
        'angle_pc1_deg': np.degrees(np.arccos(min(abs(cos_pc1), 1.0))),
        'var_explained_pc1_B0': var_explained_B0[0] if var_explained_B0 else 0,
        'var_explained_pc1_B1': var_explained_B1[0] if var_explained_B1 else 0,
        'var_explained_B0': var_explained_B0[:5],
        'var_explained_B1': var_explained_B1[:5],
    }


def main():
    parser = argparse.ArgumentParser(description='Phase CCXXXIII: Semantic Gravity Verification')
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=40, help='每个特征对的句子数')
    parser.add_argument('--layers', type=str, default='last', help='分析哪些层: last, all, mid')
    args = parser.parse_args()

    model_key = args.model
    n_pairs = min(args.n_pairs, 40)  # 最多40对
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    d_model = config['d_model']

    # 输出目录
    out_dir = f'results/causal_fiber/{model_key}_ccxxxiii'
    os.makedirs(out_dir, exist_ok=True)

    log_file = open(f'{out_dir}/run.log', 'w', encoding='utf-8', errors='replace')
    def log(msg):
        try:
            print(msg, flush=True)
            log_file.write(msg + '\n')
            log_file.flush()
        except Exception:
            safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
            print(safe_msg, flush=True)
            log_file.write(safe_msg + '\n')
            log_file.flush()

    log(f"Phase CCXXXIII: 语义引力定量验证 — {config['name']}")
    log(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"特征对数: {len(FOUR_WAY_PAIRS)}, 每对句子数: {n_pairs}")
    log("=" * 70)

    # ===== S1: 加载模型 =====
    log("\nS1: 加载模型")
    model, tokenizer = load_model(model_key)
    log(f"  模型加载完成: n_layers={n_layers}, d_model={d_model}")

    # ===== S2: 收集4路隐藏状态 =====
    log("\nS2: 收集4路最小对的隐藏状态")

    all_results = {}

    for pair_name, pair_data in FOUR_WAY_PAIRS.items():
        log(f"\n--- 特征对: {pair_name} ---")
        log(f"  Feature A: {pair_data['feature_A']}, Feature B: {pair_data['feature_B']}")
        log(f"  Expected |cos|: {pair_data['expected_cos']}")

        sentences = pair_data['sentences'][:n_pairs]
        n_sents = len(sentences)

        # 收集所有4路的隐藏状态
        h_A0B0_all = []  # (n_sents, n_layers+1, d_model)
        h_A1B0_all = []
        h_A0B1_all = []
        h_A1B1_all = []

        for i, (s_A0B0, s_A1B0, s_A0B1, s_A1B1) in enumerate(sentences):
            h_A0B0 = get_last_token_hidden(model, tokenizer, s_A0B0)
            h_A1B0 = get_last_token_hidden(model, tokenizer, s_A1B0)
            h_A0B1 = get_last_token_hidden(model, tokenizer, s_A0B1)
            h_A1B1 = get_last_token_hidden(model, tokenizer, s_A1B1)

            h_A0B0_all.append(h_A0B0)
            h_A1B0_all.append(h_A1B0)
            h_A0B1_all.append(h_A0B1)
            h_A1B1_all.append(h_A1B1)

            if (i + 1) % 10 == 0:
                log(f"  已处理 {i+1}/{n_sents} 句")

        log(f"  完成: {n_sents} 句, 每句4个变体")

        # ===== S3: 交互分析 =====
        log("\nS3: 交互分析 (2×2因子设计)")

        # 分析最后一层和中间层
        if args.layers == 'last':
            target_layers = [n_layers]
        elif args.layers == 'mid':
            target_layers = [n_layers // 2, n_layers]
        else:
            target_layers = list(range(1, n_layers + 1))

        interaction_results = {}

        for layer_idx in target_layers:
            # 逐样本计算交互
            interaction_ratios = []
            additive_errors = []
            cos_int_As = []
            cos_int_Bs = []

            for i in range(n_sents):
                h0 = h_A0B0_all[i][layer_idx]
                h1 = h_A1B0_all[i][layer_idx]
                h2 = h_A0B1_all[i][layer_idx]
                h3 = h_A1B1_all[i][layer_idx]

                result = compute_interaction_analysis(h0, h1, h2, h3)
                interaction_ratios.append(result['interaction_ratio'])
                additive_errors.append(result['additive_pred_error'])
                cos_int_As.append(result['cos_int_A'])
                cos_int_Bs.append(result['cos_int_B'])

            interaction_results[layer_idx] = {
                'mean_interaction_ratio': np.mean(interaction_ratios),
                'std_interaction_ratio': np.std(interaction_ratios),
                'mean_additive_error': np.mean(additive_errors),
                'std_additive_error': np.std(additive_errors),
                'mean_cos_int_A': np.mean(cos_int_As),
                'mean_cos_int_B': np.mean(cos_int_Bs),
            }

        # 只打印最后一层的结果
        last_layer = n_layers
        ir = interaction_results[last_layer]
        log(f"\n  最后一层(L{last_layer})交互分析:")
        log(f"    交互比: {ir['mean_interaction_ratio']:.4f} ± {ir['std_interaction_ratio']:.4f}")
        log(f"    加法预测误差: {ir['mean_additive_error']:.4f} ± {ir['std_additive_error']:.4f}")
        log(f"    交互×A方向cos: {ir['mean_cos_int_A']:.4f}")
        log(f"    交互×B方向cos: {ir['mean_cos_int_B']:.4f}")

        # ===== S4: 条件PCA =====
        log("\nS4: 条件PCA分析")

        conditional_pca_results = {}

        for layer_idx in target_layers:
            # A的差分, 分别在B=0和B=1条件下
            dh_A_given_B0 = [h_A1B0_all[i][layer_idx] - h_A0B0_all[i][layer_idx] for i in range(n_sents)]
            dh_A_given_B1 = [h_A1B1_all[i][layer_idx] - h_A0B1_all[i][layer_idx] for i in range(n_sents)]

            # B的差分, 分别在A=0和A=1条件下
            dh_B_given_A0 = [h_A0B1_all[i][layer_idx] - h_A0B0_all[i][layer_idx] for i in range(n_sents)]
            dh_B_given_A1 = [h_A1B1_all[i][layer_idx] - h_A1B0_all[i][layer_idx] for i in range(n_sents)]

            pca_A = compute_conditional_pca(dh_A_given_B0, dh_A_given_B1)
            pca_B = compute_conditional_pca(dh_B_given_A0, dh_B_given_A1)

            conditional_pca_results[layer_idx] = {
                'A_conditional_on_B': pca_A,
                'B_conditional_on_A': pca_B,
            }

        # 打印最后一层结果
        cp = conditional_pca_results[last_layer]
        pca_A_last = cp['A_conditional_on_B']
        pca_B_last = cp['B_conditional_on_A']

        log(f"\n  最后一层(L{last_layer})条件PCA:")
        log(f"    {pair_data['feature_A']}|B=0 vs {pair_data['feature_A']}|B=1:")
        log(f"      PC1 cos = {pca_A_last['cos_pc1']:.4f}, angle = {pca_A_last['angle_pc1_deg']:.1f}°")
        log(f"      PC1方差解释: B0={pca_A_last['var_explained_pc1_B0']:.4f}, B1={pca_A_last['var_explained_pc1_B1']:.4f}")
        log(f"    {pair_data['feature_B']}|A=0 vs {pair_data['feature_B']}|A=1:")
        log(f"      PC1 cos = {pca_B_last['cos_pc1']:.4f}, angle = {pca_B_last['angle_pc1_deg']:.1f}°")
        log(f"      PC1方差解释: A0={pca_B_last['var_explained_pc1_B0']:.4f}, A1={pca_B_last['var_explained_pc1_B1']:.4f}")

        # ===== S5: PC1|cos|基线 (从当前数据计算) =====
        log("\nS5: 特征间PC1|cos|基线")

        # 用A0B0和A1B0计算A的PC1
        # 用A0B0和A0B1计算B的PC1
        dh_A_all = [h_A1B0_all[i][last_layer] - h_A0B0_all[i][last_layer] for i in range(n_sents)]
        dh_B_all = [h_A0B1_all[i][last_layer] - h_A0B0_all[i][last_layer] for i in range(n_sents)]

        mat_A = torch.stack(dh_A_all)
        mat_B = torch.stack(dh_B_all)

        _, _, Vt_A = torch.linalg.svd(mat_A, full_matrices=False)
        _, _, Vt_B = torch.linalg.svd(mat_B, full_matrices=False)

        pc1_A = Vt_A[0]
        pc1_B = Vt_B[0]

        cos_pc1_AB = abs(torch.nn.functional.cosine_similarity(pc1_A.unsqueeze(0), pc1_B.unsqueeze(0)).item())

        log(f"  PC1({pair_data['feature_A']}) × PC1({pair_data['feature_B']}): |cos| = {cos_pc1_AB:.4f}")

        # 存储结果
        all_results[pair_name] = {
            'feature_A': pair_data['feature_A'],
            'feature_B': pair_data['feature_B'],
            'expected_cos': pair_data['expected_cos'],
            'n_pairs': n_sents,
            'pc1_cos_AB': cos_pc1_AB,
            'interaction_last_layer': interaction_results[last_layer],
            'conditional_pca_last_layer': conditional_pca_results[last_layer],
            'interaction_all_layers': {str(k): v for k, v in interaction_results.items()},
            'conditional_pca_all_layers': {},
        }

        # 序列化conditional_pca (不能直接json化tensor)
        for layer_idx, cp_data in conditional_pca_results.items():
            all_results[pair_name]['conditional_pca_all_layers'][str(layer_idx)] = {
                'A_conditional_on_B': {
                    'cos_pc1': cp_data['A_conditional_on_B']['cos_pc1'],
                    'angle_pc1_deg': cp_data['A_conditional_on_B']['angle_pc1_deg'],
                    'var_explained_pc1_B0': cp_data['A_conditional_on_B']['var_explained_pc1_B0'],
                    'var_explained_pc1_B1': cp_data['A_conditional_on_B']['var_explained_pc1_B1'],
                    'var_explained_B0': cp_data['A_conditional_on_B']['var_explained_B0'],
                    'var_explained_B1': cp_data['A_conditional_on_B']['var_explained_B1'],
                },
                'B_conditional_on_A': {
                    'cos_pc1': cp_data['B_conditional_on_A']['cos_pc1'],
                    'angle_pc1_deg': cp_data['B_conditional_on_A']['angle_pc1_deg'],
                    'var_explained_pc1_B0': cp_data['B_conditional_on_A']['var_explained_pc1_B0'],
                    'var_explained_pc1_B1': cp_data['B_conditional_on_A']['var_explained_pc1_B1'],
                    'var_explained_B0': cp_data['B_conditional_on_A']['var_explained_B0'],
                    'var_explained_B1': cp_data['B_conditional_on_A']['var_explained_B1'],
                },
            }

    # ===== S6: 汇总分析 =====
    log("\n" + "=" * 70)
    log("S6: 汇总分析 — 语义引力验证")
    log("=" * 70)

    # 收集所有特征对的指标
    summary = []
    for pair_name, result in all_results.items():
        ir = result['interaction_last_layer']
        cp_A = result['conditional_pca_last_layer']['A_conditional_on_B']
        cp_B = result['conditional_pca_last_layer']['B_conditional_on_A']

        summary.append({
            'pair': pair_name,
            'feature_A': result['feature_A'],
            'feature_B': result['feature_B'],
            'expected_cos': result['expected_cos'],
            'pc1_cos': result['pc1_cos_AB'],
            'interaction_ratio': ir['mean_interaction_ratio'],
            'additive_error': ir['mean_additive_error'],
            'cond_pca_angle_A': cp_A['angle_pc1_deg'],
            'cond_pca_cos_A': cp_A['cos_pc1'],
            'cond_pca_angle_B': cp_B['angle_pc1_deg'],
            'cond_pca_cos_B': cp_B['cos_pc1'],
        })

    log("\n6个特征对汇总:")
    log(f"{'特征对':<25} {'PC1|cos|':>10} {'交互比':>10} {'加法误差':>10} {'条件PCA角度A':>14} {'条件PCA角度B':>14}")
    log("-" * 90)

    pc1_coses = []
    interaction_ratios = []
    cond_pca_angles = []

    for s in summary:
        log(f"{s['pair']:<25} {s['pc1_cos']:>10.4f} {s['interaction_ratio']:>10.4f} {s['additive_error']:>10.4f} {s['cond_pca_angle_A']:>14.1f} {s['cond_pca_angle_B']:>14.1f}")
        pc1_coses.append(s['pc1_cos'])
        interaction_ratios.append(s['interaction_ratio'])
        cond_pca_angles.append((s['cond_pca_angle_A'] + s['cond_pca_angle_B']) / 2)

    # 相关性分析
    log("\n相关性分析 (Pearson r):")
    pc1_coses = np.array(pc1_coses)
    interaction_ratios = np.array(interaction_ratios)
    cond_pca_angles = np.array(cond_pca_angles)

    if len(pc1_coses) > 2:
        r_int = np.corrcoef(pc1_coses, interaction_ratios)[0, 1]
        r_angle = np.corrcoef(pc1_coses, cond_pca_angles)[0, 1]
        r_int_angle = np.corrcoef(interaction_ratios, cond_pca_angles)[0, 1]
        log(f"  PC1|cos| × 交互比: r = {r_int:.4f}")
        log(f"  PC1|cos| × 条件PCA角度: r = {r_angle:.4f}")
        log(f"  交互比 × 条件PCA角度: r = {r_int_angle:.4f}")

        # 语义引力验证
        log("\n语义引力模型验证:")
        if r_int > 0.5:
            log(f"  [SUPPORTED] PC1|cos|与交互比正相关(r={r_int:.2f}) -> 语义引力假设得到支持!")
            log(f"     高重叠特征对有更强的非线性交互")
        elif r_int > 0.2:
            log(f"  [PARTIAL] PC1|cos|与交互比弱正相关(r={r_int:.2f}) -> 语义引力假设部分支持")
        else:
            log(f"  [REJECTED] PC1|cos|与交互比无正相关(r={r_int:.2f}) -> 语义引力假设不被支持")

        if r_angle > 0.5:
            log(f"  [SUPPORTED] PC1|cos|与条件PCA角度正相关(r={r_angle:.2f}) -> 特征表示确实相互依赖!")
        elif r_angle > 0.2:
            log(f"  [PARTIAL] PC1|cos|与条件PCA角度弱正相关(r={r_angle:.2f}) -> 部分依赖")
        else:
            log(f"  [REJECTED] PC1|cos|与条件PCA角度无正相关(r={r_angle:.2f}) -> 特征表示独立")

    # 保存结果
    results_json = {
        'model': model_key,
        'model_name': config['name'],
        'timestamp': datetime.now().isoformat(),
        'n_pairs': n_pairs,
        'n_feature_pairs': len(FOUR_WAY_PAIRS),
        'summary': summary,
        'correlations': {
            'pc1_cos_vs_interaction_ratio': float(np.corrcoef(pc1_coses, interaction_ratios)[0, 1]) if len(pc1_coses) > 2 else 0,
            'pc1_cos_vs_cond_pca_angle': float(np.corrcoef(pc1_coses, cond_pca_angles)[0, 1]) if len(pc1_coses) > 2 else 0,
            'interaction_ratio_vs_cond_pca_angle': float(np.corrcoef(interaction_ratios, cond_pca_angles)[0, 1]) if len(pc1_coses) > 2 else 0,
        },
        'detailed_results': all_results,
    }

    # 清理不能json化的内容
    for pair_name in results_json['detailed_results']:
        del results_json['detailed_results'][pair_name]['interaction_all_layers']
        del results_json['detailed_results'][pair_name]['conditional_pca_all_layers']
        del results_json['detailed_results'][pair_name]['interaction_last_layer']
        del results_json['detailed_results'][pair_name]['conditional_pca_last_layer']

    with open(f'{out_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    log(f"\n结果已保存: {out_dir}/results.json")
    log(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 释放模型
    del model
    torch.cuda.empty_cache()
    log("模型已释放")

    log_file.close()


if __name__ == '__main__':
    main()
