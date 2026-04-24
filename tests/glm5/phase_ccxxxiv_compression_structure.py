"""
Phase CCXXXIV: 压缩的数学结构 — 交互项SVD分析与压缩子空间假设
================================================================
核心目标:
1. 交互项矩阵的SVD分析: 压缩是低秩的吗?
2. 主效应 vs 交互项的秩比较
3. 不同特征对的压缩子空间是否重叠? (共享压缩子空间假设)
4. 层级动力学: 压缩在哪些层最强?
5. 压缩方向的一致性: 交互项方向在不同样本间是否一致?

关键假设:
  亚加性压缩模型预测:
  - 交互项矩阵的秩 < 主效应矩阵的秩 (压缩是低秩的)
  - 不同特征对的交互项共享压缩子空间 (通用压缩机制)
  - 压缩在中间层最强 (特征交互最丰富的层)
  - 交互项方向在样本间一致 (系统性压缩, 不是噪声)

方法:
  对CCXXXIII的6个特征对, 逐层收集4路最小对:
  1. 交互矩阵: [Δh(A1B1) - Δh(A1B0) - Δh(A0B1)] × 40样本
  2. SVD分解 → 奇异值谱, 有效秩, 累积方差
  3. 主效应矩阵: [Δh(A)] × 40 和 [Δh(B)] × 40 做同样分析
  4. 跨对Grassmann距离: 不同特征对交互子空间的重叠度
  5. 关键层采样: L1/4, L1/2, L3/4, L_last
"""

import argparse
import json
import os
import time
import numpy as np
from datetime import datetime

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

# 复用CCXXXIII的特征对设计
FOUR_WAY_PAIRS = {
    'tense_x_question': {
        'feature_A': 'tense',
        'feature_B': 'question',
        'expected_cos': 'HIGH',
        'sentences': [
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
    'tense_x_voice': {
        'feature_A': 'tense',
        'feature_B': 'voice',
        'expected_cos': 'MODERATE',
        'sentences': [
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
    'tense_x_polarity': {
        'feature_A': 'tense',
        'feature_B': 'polarity',
        'expected_cos': 'VARIABLE',
        'sentences': [
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
    'voice_x_polarity': {
        'feature_A': 'voice',
        'feature_B': 'polarity',
        'expected_cos': 'LOW',
        'sentences': [
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
    'voice_x_question': {
        'feature_A': 'voice',
        'feature_B': 'question',
        'expected_cos': 'LOW',
        'sentences': [
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
    'question_x_polarity': {
        'feature_A': 'question',
        'feature_B': 'polarity',
        'expected_cos': 'LOW-MODERATE',
        'sentences': [
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
            ("He remembers the event", "Does he remember the event", "He does not remember the event", "He did not remember the event"),
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    config = MODEL_CONFIGS[model_key]
    path = config['path']
    dtype = config['dtype']

    print(f"  Loading: {config['name']} ({dtype})")

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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    seq_len = attention_mask.sum().item()
    last_hidden = torch.stack([h[0, seq_len-1, :].cpu().float() for h in hidden_states])

    return last_hidden


def effective_rank(singular_values, threshold=0.01):
    """Entropy-based effective rank"""
    sv = singular_values[singular_values > 0]
    if len(sv) == 0:
        return 0
    p = sv / sv.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)


def cumulative_variance(singular_values, k):
    """Variance explained by top-k components"""
    total = singular_values.sum()
    if total == 0:
        return 0
    return singular_values[:k].sum() / total


def compute_grassmann_distance(U1, U2, k):
    """
    Grassmann distance between two subspaces spanned by top-k left singular vectors
    Returns: principal angles, mean angle, subspace overlap
    """
    # Take top-k left singular vectors
    V1 = U1[:, :k]  # (d, k)
    V2 = U2[:, :k]  # (d, k)

    # Compute principal angles via SVD of V1^T V2
    M = V1.T @ V2  # (k, k)
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1, 1)
    angles = np.arccos(s)

    mean_angle = np.mean(angles)
    # Subspace overlap = mean of cos^2 of principal angles
    overlap = np.mean(s**2)

    return {
        'principal_angles_deg': np.degrees(angles).tolist(),
        'mean_angle_deg': float(np.degrees(mean_angle)),
        'subspace_overlap': float(overlap),
    }


def main():
    parser = argparse.ArgumentParser(description='Phase CCXXXIV: Compression Structure Analysis')
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=40)
    args = parser.parse_args()

    model_key = args.model
    n_pairs = min(args.n_pairs, 40)
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    d_model = config['d_model']

    out_dir = f'results/causal_fiber/{model_key}_ccxxxiv'
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

    log(f"Phase CCXXXIV: Compression Structure Analysis - {config['name']}")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Pairs: {len(FOUR_WAY_PAIRS)}, sentences/pair: {n_pairs}")
    log("=" * 70)

    # Key layers for analysis
    key_layers = [1, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
    # Also include embedding (layer 0)
    key_layers = [0] + key_layers
    log(f"Key layers: {key_layers}")

    # S1: Load model
    log("\nS1: Loading model")
    model, tokenizer = load_model(model_key)
    log(f"  Model loaded: n_layers={n_layers}, d_model={d_model}")

    # S2: Collect 4-way hidden states
    log("\nS2: Collecting 4-way minimal pair hidden states")

    all_pair_data = {}

    for pair_name, pair_data in FOUR_WAY_PAIRS.items():
        log(f"\n--- Pair: {pair_name} ---")
        sentences = pair_data['sentences'][:n_pairs]
        n_sents = len(sentences)

        h_A0B0_all = []
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
                log(f"  Processed {i+1}/{n_sents}")

        log(f"  Done: {n_sents} sentences")

        all_pair_data[pair_name] = {
            'h_A0B0': h_A0B0_all,
            'h_A1B0': h_A1B0_all,
            'h_A0B1': h_A0B1_all,
            'h_A1B1': h_A1B1_all,
            'feature_A': pair_data['feature_A'],
            'feature_B': pair_data['feature_B'],
        }

    # S3: SVD analysis of interaction and main effects
    log("\n" + "=" * 70)
    log("S3: SVD Analysis - Interaction vs Main Effects")
    log("=" * 70)

    svd_results = {}

    for pair_name, pdata in all_pair_data.items():
        log(f"\n--- {pair_name} ---")
        n_sents = len(pdata['h_A0B0'])
        pair_svd = {}

        for layer_idx in key_layers:
            # Compute main effect A: h(A1B0) - h(A0B0)
            dh_A = torch.stack([pdata['h_A1B0'][i][layer_idx] - pdata['h_A0B0'][i][layer_idx] for i in range(n_sents)])
            # Compute main effect B: h(A0B1) - h(A0B0)
            dh_B = torch.stack([pdata['h_A0B1'][i][layer_idx] - pdata['h_A0B0'][i][layer_idx] for i in range(n_sents)])
            # Compute combined effect: h(A1B1) - h(A0B0)
            dh_AB = torch.stack([pdata['h_A1B1'][i][layer_idx] - pdata['h_A0B0'][i][layer_idx] for i in range(n_sents)])
            # Compute interaction: dh_AB - (dh_A + dh_B)
            interaction = dh_AB - (dh_A + dh_B)

            # SVD of each matrix (n_sents x d_model)
            sv_A = torch.linalg.svdvals(dh_A).numpy()
            sv_B = torch.linalg.svdvals(dh_B).numpy()
            sv_AB = torch.linalg.svdvals(dh_AB).numpy()
            sv_int = torch.linalg.svdvals(interaction).numpy()

            # Effective ranks
            erank_A = effective_rank(sv_A)
            erank_B = effective_rank(sv_B)
            erank_AB = effective_rank(sv_AB)
            erank_int = effective_rank(sv_int)

            # Cumulative variance at k=1,3,5,10
            cumvar_A = {k: cumulative_variance(sv_A, k) for k in [1, 3, 5, 10]}
            cumvar_int = {k: cumulative_variance(sv_int, k) for k in [1, 3, 5, 10]}

            # Interaction ratio (mean across samples)
            int_ratio = torch.norm(interaction, dim=1) / torch.norm(dh_AB, dim=1)
            mean_int_ratio = int_ratio.mean().item()

            # Cosine of interaction with main effects (mean across samples)
            cos_int_A = torch.nn.functional.cosine_similarity(interaction, dh_A, dim=1).mean().item()
            cos_int_B = torch.nn.functional.cosine_similarity(interaction, dh_B, dim=1).mean().item()

            # Full SVD for Grassmann distance (need U matrices)
            U_A, _, _ = torch.linalg.svd(dh_A, full_matrices=False)
            U_int, _, _ = torch.linalg.svd(interaction, full_matrices=False)

            pair_svd[layer_idx] = {
                'erank_A': erank_A,
                'erank_B': erank_B,
                'erank_AB': erank_AB,
                'erank_int': erank_int,
                'cumvar_A': cumvar_A,
                'cumvar_int': cumvar_int,
                'mean_int_ratio': mean_int_ratio,
                'cos_int_A': cos_int_A,
                'cos_int_B': cos_int_B,
                'sv_spectrum_A': sv_A[:10].tolist(),
                'sv_spectrum_int': sv_int[:10].tolist(),
                'U_A': U_A.numpy(),  # (min(n,d), d_model) -- for Grassmann
                'U_int': U_int.numpy(),
            }

        # Print last layer results
        last = n_layers
        p = pair_svd[last]
        log(f"  Layer L{last}:")
        log(f"    Effective rank: A={p['erank_A']:.1f}, B={p['erank_B']:.1f}, AB={p['erank_AB']:.1f}, int={p['erank_int']:.1f}")
        log(f"    CumVar(A): k1={p['cumvar_A'][1]:.3f}, k3={p['cumvar_A'][3]:.3f}, k5={p['cumvar_A'][5]:.3f}")
        log(f"    CumVar(int): k1={p['cumvar_int'][1]:.3f}, k3={p['cumvar_int'][3]:.3f}, k5={p['cumvar_int'][5]:.3f}")
        log(f"    Int ratio: {p['mean_int_ratio']:.4f}")
        log(f"    cos(int, dhA)={p['cos_int_A']:.4f}, cos(int, dhB)={p['cos_int_B']:.4f}")

        svd_results[pair_name] = pair_svd

    # S4: Cross-pair Grassmann distance
    log("\n" + "=" * 70)
    log("S4: Cross-pair Grassmann Distance - Shared Compression Subspace?")
    log("=" * 70)

    pair_names = list(all_pair_data.keys())
    last = n_layers
    k_sub = 5  # Compare top-5 subspaces

    grassmann_matrix = {}
    for i, p1 in enumerate(pair_names):
        for j, p2 in enumerate(pair_names):
            if j <= i:
                continue
            U1 = svd_results[p1][last]['U_int']  # (n_sents, d_model) but we want (d_model, n_sents)
            U2 = svd_results[p2][last]['U_int']
            # U matrices are (n_sents, d_model) from SVD, need transpose for Grassmann
            gd = compute_grassmann_distance(U1.T, U2.T, min(k_sub, U1.shape[1], U2.shape[1]))
            grassmann_matrix[f"{p1}_x_{p2}"] = gd
            log(f"  {p1} x {p2}: mean_angle={gd['mean_angle_deg']:.1f} deg, overlap={gd['subspace_overlap']:.3f}")

    # Also compute Grassmann distance between main effects and interactions
    log("\n  Main effect vs interaction subspace overlap:")
    for p1 in pair_names:
        U_A = svd_results[p1][last]['U_A']
        U_int = svd_results[p1][last]['U_int']
        gd = compute_grassmann_distance(U_A.T, U_int.T, min(k_sub, U_A.shape[1], U_int.shape[1]))
        log(f"    {p1}: A vs int overlap={gd['subspace_overlap']:.3f}, angle={gd['mean_angle_deg']:.1f}")

    # S5: Layer dynamics
    log("\n" + "=" * 70)
    log("S5: Layer Dynamics - Where is compression strongest?")
    log("=" * 70)

    for pair_name in pair_names:
        log(f"\n  {pair_name}:")
        for layer_idx in key_layers:
            p = svd_results[pair_name][layer_idx]
            rank_ratio = p['erank_int'] / max(p['erank_A'], 0.1)
            log(f"    L{layer_idx}: erank_A={p['erank_A']:.1f}, erank_int={p['erank_int']:.1f}, "
                f"ratio={rank_ratio:.2f}, int_ratio={p['mean_int_ratio']:.3f}, "
                f"cos_int_A={p['cos_int_A']:.3f}")

    # S6: Summary
    log("\n" + "=" * 70)
    log("S6: Summary - Compression Structure")
    log("=" * 70)

    # Collect last-layer stats across all pairs
    last = n_layers
    erank_A_list = []
    erank_int_list = []
    cumvar_int_k1_list = []
    cumvar_int_k3_list = []
    int_ratio_list = []
    cos_int_A_list = []
    rank_ratios = []

    for pair_name in pair_names:
        p = svd_results[pair_name][last]
        erank_A_list.append(p['erank_A'])
        erank_int_list.append(p['erank_int'])
        cumvar_int_k1_list.append(p['cumvar_int'][1])
        cumvar_int_k3_list.append(p['cumvar_int'][3])
        int_ratio_list.append(p['mean_int_ratio'])
        cos_int_A_list.append(p['cos_int_A'])
        rank_ratios.append(p['erank_int'] / max(p['erank_A'], 0.1))

    log(f"\nLast layer (L{last}) across {len(pair_names)} pairs:")
    log(f"  Effective rank A: {np.mean(erank_A_list):.1f} +/- {np.std(erank_A_list):.1f}")
    log(f"  Effective rank int: {np.mean(erank_int_list):.1f} +/- {np.std(erank_int_list):.1f}")
    log(f"  Rank ratio (int/A): {np.mean(rank_ratios):.2f} +/- {np.std(rank_ratios):.2f}")
    log(f"  CumVar(int) k=1: {np.mean(cumvar_int_k1_list):.3f}")
    log(f"  CumVar(int) k=3: {np.mean(cumvar_int_k3_list):.3f}")
    log(f"  Interaction ratio: {np.mean(int_ratio_list):.3f}")
    log(f"  cos(int, dhA): {np.mean(cos_int_A_list):.3f}")

    # Cross-pair subspace overlap statistics
    overlaps = [v['subspace_overlap'] for v in grassmann_matrix.values()]
    angles = [v['mean_angle_deg'] for v in grassmann_matrix.values()]
    log(f"\n  Cross-pair Grassmann (k={k_sub}):")
    log(f"    Mean overlap: {np.mean(overlaps):.3f} +/- {np.std(overlaps):.3f}")
    log(f"    Mean angle: {np.mean(angles):.1f} +/- {np.std(angles):.1f} deg")

    # Low-rank hypothesis test
    log(f"\n  Low-rank hypothesis:")
    if np.mean(cumvar_int_k1_list) > 0.5:
        log(f"    [SUPPORTED] PC1 explains >50% of interaction variance ({np.mean(cumvar_int_k1_list):.1%})")
        log(f"    -> Interaction is approximately 1-dimensional (anti-parallel to main effect)")
    elif np.mean(cumvar_int_k3_list) > 0.7:
        log(f"    [PARTIAL] Top-3 PCs explain >70% ({np.mean(cumvar_int_k3_list):.1%})")
        log(f"    -> Interaction is low-rank (3-5 dimensions)")
    else:
        log(f"    [REJECTED] Interaction is NOT low-rank")
        log(f"    -> Compression uses many dimensions")

    # Shared subspace hypothesis
    log(f"\n  Shared compression subspace hypothesis:")
    if np.mean(overlaps) > 0.5:
        log(f"    [SUPPORTED] High cross-pair overlap ({np.mean(overlaps):.2f}) -> shared subspace")
    elif np.mean(overlaps) > 0.2:
        log(f"    [PARTIAL] Moderate overlap ({np.mean(overlaps):.2f}) -> partially shared")
    else:
        log(f"    [REJECTED] Low overlap ({np.mean(overlaps):.2f}) -> pair-specific compression")

    # Save results
    results_json = {
        'model': model_key,
        'model_name': config['name'],
        'timestamp': datetime.now().isoformat(),
        'n_pairs': n_pairs,
        'key_layers': key_layers,
        'summary': {
            'erank_A_mean': float(np.mean(erank_A_list)),
            'erank_int_mean': float(np.mean(erank_int_list)),
            'rank_ratio_mean': float(np.mean(rank_ratios)),
            'cumvar_int_k1_mean': float(np.mean(cumvar_int_k1_list)),
            'cumvar_int_k3_mean': float(np.mean(cumvar_int_k3_list)),
            'interaction_ratio_mean': float(np.mean(int_ratio_list)),
            'cos_int_A_mean': float(np.mean(cos_int_A_list)),
            'cross_pair_overlap_mean': float(np.mean(overlaps)),
            'cross_pair_angle_mean': float(np.mean(angles)),
        },
        'per_pair_last_layer': {},
        'layer_dynamics': {},
        'grassmann_distances': {k: {'mean_angle_deg': v['mean_angle_deg'], 'subspace_overlap': v['subspace_overlap']}
                                for k, v in grassmann_matrix.items()},
    }

    # Per-pair last layer
    for pair_name in pair_names:
        p = svd_results[pair_name][last]
        results_json['per_pair_last_layer'][pair_name] = {
            'erank_A': p['erank_A'],
            'erank_B': p['erank_B'],
            'erank_int': p['erank_int'],
            'cumvar_A_k1': p['cumvar_A'][1],
            'cumvar_int_k1': p['cumvar_int'][1],
            'cumvar_int_k3': p['cumvar_int'][3],
            'mean_int_ratio': p['mean_int_ratio'],
            'cos_int_A': p['cos_int_A'],
            'cos_int_B': p['cos_int_B'],
            'sv_spectrum_A_top10': p['sv_spectrum_A'],
            'sv_spectrum_int_top10': p['sv_spectrum_int'],
        }

    # Layer dynamics (without U matrices)
    for pair_name in pair_names:
        results_json['layer_dynamics'][pair_name] = {}
        for layer_idx in key_layers:
            p = svd_results[pair_name][layer_idx]
            results_json['layer_dynamics'][pair_name][str(layer_idx)] = {
                'erank_A': p['erank_A'],
                'erank_int': p['erank_int'],
                'mean_int_ratio': p['mean_int_ratio'],
                'cos_int_A': p['cos_int_A'],
                'cumvar_int_k1': p['cumvar_int'][1],
            }

    with open(f'{out_dir}/results.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    log(f"\nResults saved: {out_dir}/results.json")
    log(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    del model
    torch.cuda.empty_cache()
    log("Model released")

    log_file.close()


if __name__ == '__main__':
    main()
