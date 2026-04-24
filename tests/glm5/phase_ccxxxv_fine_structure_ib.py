"""
Phase CCXXXV: 交互项精细结构 + 信息瓶颈验证 + 拓扑分析
=====================================================
核心目标:
1. 样本级精细结构: 每个样本的cos(int_i, dhA_i)分布 — 反平行是统计性的还是样本级一致的?
2. 信息瓶颈验证: 压缩是否最小化互信息损失? 交互项中保留了多少A和B的信息?
3. 持续同调拓扑分析: 残差流的拓扑结构 — 是否存在语言特征的拓扑签名?

关键假设:
  H1: 反平行是统计性的 — 逐样本cos分布宽, 有些样本甚至正平行
  H2: 信息瓶颈 — 压缩后的表示I(h_AB; A,B)最大化, 压缩项I(h_AB; A|B)最小化
  H3: 拓扑签名 — 不同语言特征的流形有不同的Betti数/持续条形码

方法:
  对CCXXXIV的6个特征对×50个4路最小对:
  1. 逐样本cos分析: cos(int_i, dhA_i)的分布(均值,方差,正比例)
  2. 信息论指标: 
     - 线性互信息估计 I(h; A) ∝ log(det(Cov_h|A=0)/det(Cov_h))
     - 压缩效率 = I(h_AB; A,B) / I(h_A+h_B; A,B) (线性近似的加性基线)
  3. 持续同调: 
     - 对每类4路条件(A0B0, A1B0, A0B1, A1B1)构建点云(50×d_model)
     - Vietoris-Rips复形的持续同调, dim=1, max_edge=0.5
     - 比较不同条件的Betti数和持续图
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

# Extended to 50 sentences per pair for better statistics
FOUR_WAY_PAIRS = {
    'tense_x_question': {
        'feature_A': 'tense',
        'feature_B': 'question',
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
            ("The door closes slowly", "The door closed slowly", "Does the door close slowly", "Did the door close slowly"),
            ("She borrows books", "She borrowed books", "Does she borrow books", "Did she borrow books"),
            ("They swim in the lake", "They swam in the lake", "Do they swim in the lake", "Did they swim in the lake"),
            ("He rides a bicycle", "He rode a bicycle", "Does he ride a bicycle", "Did he ride a bicycle"),
            ("The plane takes off", "The plane took off", "Does the plane take off", "Did the plane take off"),
            ("She draws portraits", "She drew portraits", "Does she draw portraits", "Did she draw portraits"),
            ("They climb mountains", "They climbed mountains", "Do they climb mountains", "Did they climb mountains"),
            ("He drinks coffee", "He drank coffee", "Does he drink coffee", "Did he drink coffee"),
            ("The flower blooms early", "The flower bloomed early", "Does the flower bloom early", "Did the flower bloom early"),
        ]
    },
    'tense_x_voice': {
        'feature_A': 'tense',
        'feature_B': 'voice',
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
            ("She founded the company", "The company is founded by her", "The company was founded by her", "The company had been founded by her"),
            ("He coaches the team", "The team is coached by him", "The team was coached by him", "The team had been coached by him"),
            ("They install the equipment", "The equipment is installed by them", "The equipment was installed by them", "The equipment had been installed by them"),
            ("The cat kills the mouse", "The mouse is killed by the cat", "The cat killed the mouse", "The mouse was killed by the cat"),
            ("She draws the map", "The map is drawn by her", "She drew the map", "The map was drawn by her"),
            ("He signs the contract", "The contract is signed by him", "He signed the contract", "The contract was signed by him"),
            ("They deliver the package", "The package is delivered by them", "They delivered the package", "The package was delivered by them"),
            ("The storm destroys the crop", "The crop is destroyed by the storm", "The storm destroyed the crop", "The crop was destroyed by the storm"),
            ("She writes the essay", "The essay is written by her", "She wrote the essay", "The essay was written by her"),
            ("He leads the team", "The team is led by him", "He led the team", "The team was led by him"),
            ("They paint the wall", "The wall is painted by them", "They painted the wall", "The wall was painted by them"),
            ("The fire damages the building", "The building is damaged by the fire", "The fire damaged the building", "The building was damaged by the fire"),
        ]
    },
    'tense_x_polarity': {
        'feature_A': 'tense',
        'feature_B': 'polarity',
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
            ("She loves the movie", "She loved the movie", "She does not love the movie", "She did not love the movie"),
            ("He knows the answer", "He knew the answer", "He does not know the answer", "He did not know the answer"),
            ("They enjoy the game", "They enjoyed the game", "They do not enjoy the game", "They did not enjoy the game"),
            ("The project succeeds", "The project succeeded", "The project does not succeed", "The project did not succeed"),
            ("She understands the theory", "She understood the theory", "She does not understand the theory", "She did not understand the theory"),
            ("He believes the rumor", "He believed the rumor", "He does not believe the rumor", "He did not believe the rumor"),
            ("They trust the process", "They trusted the process", "They do not trust the process", "They did not trust the process"),
            ("The plan works", "The plan worked", "The plan does not work", "The plan did not work"),
            ("She likes the color", "She liked the color", "She does not like the color", "She did not like the color"),
        ]
    },
    'voice_x_polarity': {
        'feature_A': 'voice',
        'feature_B': 'polarity',
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
            ("The artist drew the picture", "The picture was drawn by the artist", "The artist did not draw the picture", "The picture was not drawn by the artist"),
            ("She signed the letter", "The letter was signed by her", "She did not sign the letter", "The letter was not signed by her"),
            ("He delivered the package", "The package was delivered by him", "He did not deliver the package", "The package was not delivered by him"),
            ("They painted the wall", "The wall was painted by them", "They did not paint the wall", "The wall was not painted by them"),
            ("The fire destroyed the forest", "The forest was destroyed by the fire", "The fire did not destroy the forest", "The forest was not destroyed by the fire"),
            ("She wrote the essay", "The essay was written by her", "She did not write the essay", "The essay was not written by her"),
            ("He led the expedition", "The expedition was led by him", "He did not lead the expedition", "The expedition was not led by him"),
            ("They built the bridge", "The bridge was built by them", "They did not build the bridge", "The bridge was not built by them"),
            ("The company developed the vaccine", "The vaccine was developed by the company", "The company did not develop the vaccine", "The vaccine was not developed by the company"),
        ]
    },
    'voice_x_question': {
        'feature_A': 'voice',
        'feature_B': 'question',
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
            ("She paints portraits", "Portraits are painted by her", "Does she paint portraits", "Are portraits painted by her"),
            ("He writes novels", "Novels are written by him", "Does he write novels", "Are novels written by him"),
            ("They design buildings", "Buildings are designed by them", "Do they design buildings", "Are buildings designed by them"),
            ("The school educates children", "Children are educated by the school", "Does the school educate children", "Are children educated by the school"),
            ("She sells flowers", "Flowers are sold by her", "Does she sell flowers", "Are flowers sold by her"),
            ("He delivers newspapers", "Newspapers are delivered by him", "Does he deliver newspapers", "Are newspapers delivered by him"),
            ("They serve lunch", "Lunch is served by them", "Do they serve lunch", "Is lunch served by them"),
            ("The chef prepares dinner", "Dinner is prepared by the chef", "Does the chef prepare dinner", "Is dinner prepared by the chef"),
            ("She types letters", "Letters are typed by her", "Does she type letters", "Are letters typed by her"),
            ("He repairs bicycles", "Bicycles are repaired by him", "Does he repair bicycles", "Are bicycles repaired by him"),
            ("They bake cookies", "Cookies are baked by them", "Do they bake cookies", "Are cookies baked by them"),
            ("The city maintains parks", "Parks are maintained by the city", "Does the city maintain parks", "Are parks maintained by the city"),
            ("She designs websites", "Websites are designed by her", "Does she design websites", "Are websites designed by her"),
            ("He grows tomatoes", "Tomatoes are grown by him", "Does he grow tomatoes", "Are tomatoes grown by him"),
            ("They print books", "Books are printed by them", "Do they print books", "Are books printed by them"),
            ("The company produces cars", "Cars are produced by the company", "Does the company produce cars", "Are cars produced by the company"),
            ("She teaches physics", "Physics is taught by her", "Does she teach physics", "Is physics taught by her"),
            ("He records songs", "Songs are recorded by him", "Does he record songs", "Are songs recorded by him"),
            ("They brew coffee", "Coffee is brewed by them", "Do they brew coffee", "Is coffee brewed by them"),
            ("The lab tests samples", "Samples are tested by the lab", "Does the lab test samples", "Are samples tested by the lab"),
        ]
    },
    'question_x_polarity': {
        'feature_A': 'question',
        'feature_B': 'polarity',
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
            ("She loves the music", "Does she love the music", "She does not love the music", "Does she not love the music"),
            ("He knows the truth", "Does he know the truth", "He does not know the truth", "Does he not know the truth"),
            ("They enjoy the movie", "Do they enjoy the movie", "They do not enjoy the movie", "Do they not enjoy the movie"),
            ("The project succeeds", "Does the project succeed", "The project does not succeed", "Does the project not succeed"),
            ("She understands the theory", "Does she understand the theory", "She does not understand the theory", "Does she not understand the theory"),
            ("He believes the rumor", "Does he believe the rumor", "He does not believe the rumor", "Does he not believe the rumor"),
            ("They trust the process", "Do they trust the process", "They do not trust the process", "Do they not trust the process"),
            ("The plan works", "Does the plan work", "The plan does not work", "Does the plan not work"),
            ("She likes the color", "Does she like the color", "She does not like the color", "Does she not like the color"),
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


def linear_mutual_info(X, labels):
    """
    Estimate mutual information I(X; Y) using linear Gaussian assumption.
    I(X; Y) = 0.5 * log(det(Cov(X)) / det(Cov(X|Y)))
    For binary Y: I(X; Y) = 0.5 * log(det(Cov(X)) / (det(Cov_X0) * det(Cov_X1))^(1/2))
    Simplified: use trace ratio as a robust estimate.
    """
    X = X.numpy() if isinstance(X, torch.Tensor) else X
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    # Overall covariance
    cov_all = np.cov(X, rowvar=False)
    if cov_all.ndim == 0:
        cov_all = cov_all.reshape(1, 1)
    
    # Conditional covariance for each label
    unique_labels = np.unique(labels)
    log_det_cond = 0.0
    for lab in unique_labels:
        mask = labels == lab
        X_cond = X[mask]
        if len(X_cond) < 2:
            return 0.0
        cov_cond = np.cov(X_cond, rowvar=False)
        if cov_cond.ndim == 0:
            cov_cond = cov_cond.reshape(1, 1)
        # Regularize
        cov_cond += 1e-6 * np.eye(cov_cond.shape[0])
        try:
            sign, logdet = np.linalg.slogdet(cov_cond)
            log_det_cond += logdet * len(X_cond) / len(X)
        except:
            return 0.0
    
    cov_all += 1e-6 * np.eye(cov_all.shape[0])
    try:
        sign, log_det_all = np.linalg.slogdet(cov_all)
    except:
        return 0.0
    
    mi = 0.5 * (log_det_all - log_det_cond)
    return max(mi, 0.0)


def compute_persistence_diagram_gpu(distance_matrix, max_dim=1, max_edge=None):
    """
    Compute persistence homology using a simple approach.
    For efficiency with 50 points in high-dim space, we use sub-sampled PCA projection.
    Returns simplified persistence statistics.
    """
    n = distance_matrix.shape[0]
    
    if max_edge is None:
        # Use 90th percentile of distances as max edge
        upper_tri = distance_matrix[np.triu_indices(n, k=1)]
        max_edge = np.percentile(upper_tri, 90)
    
    # Build filtration using distance threshold
    # Simplified Betti number computation: count connected components and loops
    # at various distance thresholds
    
    thresholds = np.linspace(0, max_edge, 20)
    betti_0 = []  # Connected components
    betti_1_est = []  # Estimated loops (edges - vertices + components)
    
    for t in thresholds:
        # Build adjacency at threshold t
        adj = (distance_matrix <= t).astype(int)
        np.fill_diagonal(adj, 0)
        
        # Count connected components using union-find
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        n_edges = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    n_edges += 1
                    pi, pj = find(i), find(j)
                    if pi != pj:
                        parent[pi] = pj
        
        n_components = len(set(find(x) for x in range(n)))
        betti_0.append(n_components)
        
        # Estimate H1: edges - vertices + components
        # This is an upper bound on H1
        h1_est = max(0, n_edges - n + n_components)
        betti_1_est.append(h1_est)
    
    return {
        'betti_0': betti_0,
        'betti_1_est': betti_1_est,
        'thresholds': thresholds.tolist(),
        'n_components_final': betti_0[-1],
        'max_h1': max(betti_1_est) if betti_1_est else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Phase CCXXXV: Fine Structure + Info Bottleneck + Topology')
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=50)
    parser.add_argument('--skip_topology', action='store_true', help='Skip topology analysis (faster)')
    args = parser.parse_args()

    model_key = args.model
    n_pairs = min(args.n_pairs, 50)
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    d_model = config['d_model']

    out_dir = f'results/causal_fiber/{model_key}_ccxxxv'
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

    log(f"Phase CCXXXV: Fine Structure + Info Bottleneck + Topology - {config['name']}")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Pairs: {len(FOUR_WAY_PAIRS)}, sentences/pair: {n_pairs}")
    log("=" * 70)

    # Key layers: last layer only for detailed analysis, plus L1 and L_mid for comparison
    key_layers = [1, n_layers // 2, n_layers]
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

    # ============================================================
    # S3: Sample-level fine structure analysis
    # ============================================================
    log("\n" + "=" * 70)
    log("S3: Sample-Level Fine Structure Analysis")
    log("=" * 70)

    fine_structure_results = {}

    for pair_name, pdata in all_pair_data.items():
        log(f"\n--- {pair_name} ---")
        n_sents = len(pdata['h_A0B0'])
        pair_fs = {}

        for layer_idx in key_layers:
            # Compute per-sample interaction vectors
            dh_A = torch.stack([pdata['h_A1B0'][i][layer_idx] - pdata['h_A0B0'][i][layer_idx] for i in range(n_sents)])
            dh_B = torch.stack([pdata['h_A0B1'][i][layer_idx] - pdata['h_A0B0'][i][layer_idx] for i in range(n_sents)])
            dh_AB = torch.stack([pdata['h_A1B1'][i][layer_idx] - pdata['h_A0B0'][i][layer_idx] for i in range(n_sents)])
            interaction = dh_AB - (dh_A + dh_B)

            # Per-sample cosine similarities
            cos_int_A_per_sample = torch.nn.functional.cosine_similarity(interaction, dh_A, dim=1).numpy()
            cos_int_B_per_sample = torch.nn.functional.cosine_similarity(interaction, dh_B, dim=1).numpy()
            cos_int_AB_per_sample = torch.nn.functional.cosine_similarity(interaction, dh_AB, dim=1).numpy()

            # Per-sample interaction ratio
            int_norms = torch.norm(interaction, dim=1).numpy()
            ab_norms = torch.norm(dh_AB, dim=1).numpy()
            a_norms = torch.norm(dh_A, dim=1).numpy()
            b_norms = torch.norm(dh_B, dim=1).numpy()
            int_ratio_per_sample = int_norms / np.maximum(ab_norms, 1e-10)

            # Statistics
            pair_fs[layer_idx] = {
                'cos_int_A': {
                    'mean': float(cos_int_A_per_sample.mean()),
                    'std': float(cos_int_A_per_sample.std()),
                    'median': float(np.median(cos_int_A_per_sample)),
                    'min': float(cos_int_A_per_sample.min()),
                    'max': float(cos_int_A_per_sample.max()),
                    'frac_negative': float((cos_int_A_per_sample < 0).mean()),
                    'frac_positive': float((cos_int_A_per_sample > 0).mean()),
                    'q10': float(np.percentile(cos_int_A_per_sample, 10)),
                    'q90': float(np.percentile(cos_int_A_per_sample, 90)),
                },
                'cos_int_B': {
                    'mean': float(cos_int_B_per_sample.mean()),
                    'std': float(cos_int_B_per_sample.std()),
                    'median': float(np.median(cos_int_B_per_sample)),
                    'min': float(cos_int_B_per_sample.min()),
                    'max': float(cos_int_B_per_sample.max()),
                    'frac_negative': float((cos_int_B_per_sample < 0).mean()),
                    'frac_positive': float((cos_int_B_per_sample > 0).mean()),
                },
                'cos_int_AB': {
                    'mean': float(cos_int_AB_per_sample.mean()),
                    'std': float(cos_int_AB_per_sample.std()),
                    'frac_negative': float((cos_int_AB_per_sample < 0).mean()),
                },
                'int_ratio': {
                    'mean': float(int_ratio_per_sample.mean()),
                    'std': float(int_ratio_per_sample.std()),
                    'median': float(np.median(int_ratio_per_sample)),
                    'min': float(int_ratio_per_sample.min()),
                    'max': float(int_ratio_per_sample.max()),
                },
                # Norm ratio: |dh_A| / |dh_B| (asymmetry measure)
                'norm_ratio_AB': {
                    'mean': float((a_norms / np.maximum(b_norms, 1e-10)).mean()),
                    'std': float((a_norms / np.maximum(b_norms, 1e-10)).std()),
                },
                # Correlation between cos_int_A and int_ratio (is stronger compression more anti-parallel?)
                'corr_cos_intA_int_ratio': float(np.corrcoef(cos_int_A_per_sample, int_ratio_per_sample)[0, 1]),
                # Correlation between |dh_A| and cos_int_A (does effect size affect anti-parallelism?)
                'corr_normA_cos_intA': float(np.corrcoef(a_norms, cos_int_A_per_sample)[0, 1]),
            }

            log(f"  L{layer_idx}: cos(int,A)={cos_int_A_per_sample.mean():.3f}+-{cos_int_A_per_sample.std():.3f} "
                f"neg={float((cos_int_A_per_sample < 0).mean()):.2f} "
                f"int_ratio={int_ratio_per_sample.mean():.3f}+-{int_ratio_per_sample.std():.3f} "
                f"corr(cosA,ratio)={pair_fs[layer_idx]['corr_cos_intA_int_ratio']:.3f}")

        fine_structure_results[pair_name] = pair_fs

    # ============================================================
    # S4: Information Bottleneck Analysis
    # ============================================================
    log("\n" + "=" * 70)
    log("S4: Information Bottleneck Analysis")
    log("=" * 70)

    ib_results = {}

    for pair_name, pdata in all_pair_data.items():
        log(f"\n--- {pair_name} ---")
        n_sents = len(pdata['h_A0B0'])
        pair_ib = {}

        for layer_idx in key_layers:
            # Build feature matrices for each condition
            h_A0B0 = torch.stack([pdata['h_A0B0'][i][layer_idx] for i in range(n_sents)])
            h_A1B0 = torch.stack([pdata['h_A1B0'][i][layer_idx] for i in range(n_sents)])
            h_A0B1 = torch.stack([pdata['h_A0B1'][i][layer_idx] for i in range(n_sents)])
            h_A1B1 = torch.stack([pdata['h_A1B1'][i][layer_idx] for i in range(n_sents)])

            # All hidden states pooled
            h_all = torch.cat([h_A0B0, h_A1B0, h_A0B1, h_A1B1], dim=0)  # (4N, d)

            # Labels for feature A (0/1)
            labels_A = np.array([0]*n_sents + [1]*n_sents + [0]*n_sents + [1]*n_sents)
            # Labels for feature B (0/1)
            labels_B = np.array([0]*n_sents + [0]*n_sents + [1]*n_sents + [1]*n_sents)
            # Joint labels (A,B)
            labels_AB = np.array([0]*n_sents + [1]*n_sents + [2]*n_sents + [3]*n_sents)

            # PCA to reduce dimension for MI estimation (top 50 components)
            from sklearn.decomposition import PCA
            n_pca = min(50, h_all.shape[0] - 1, h_all.shape[1])
            pca = PCA(n_components=n_pca)
            h_pca = pca.fit_transform(h_all.numpy())
            
            # Explained variance
            cumvar_50 = float(pca.explained_variance_ratio_[:50].sum()) if n_pca >= 50 else float(pca.explained_variance_ratio_.sum())

            # MI estimates using linear Gaussian assumption
            mi_A = linear_mutual_info(h_pca, labels_A)
            mi_B = linear_mutual_info(h_pca, labels_B)
            mi_AB = linear_mutual_info(h_pca, labels_AB)

            # Information decomposition
            # I(h; A,B) = I(h; A) + I(h; B) - I(h; A; B) [interaction information]
            # where I(h; A; B) = I(h; A) + I(h; B) - I(h; A,B)
            interaction_info = mi_A + mi_B - mi_AB  # Negative = redundancy, Positive = synergy

            # Compression efficiency: how much of the theoretical max info is preserved?
            # Theoretical max: if A and B were independent, I(h; A,B) = I(h; A) + I(h; B)
            # Efficiency = I(h; A,B) / (I(h; A) + I(h; B))
            additive_mi = mi_A + mi_B
            efficiency = mi_AB / max(additive_mi, 1e-10)

            # Redundancy vs Synergy
            # Redundancy = I(h; A; B) if negative (same info about both)
            # Synergy = I(h; A; B) if positive (joint info > sum)
            redundancy = max(0, -interaction_info)
            synergy = max(0, interaction_info)

            pair_ib[layer_idx] = {
                'mi_A': float(mi_A),
                'mi_B': float(mi_B),
                'mi_AB': float(mi_AB),
                'interaction_info': float(interaction_info),
                'additive_mi': float(additive_mi),
                'efficiency': float(efficiency),
                'redundancy': float(redundancy),
                'synergy': float(synergy),
                'pca_cumvar_50': float(cumvar_50),
                'n_pca': n_pca,
            }

            log(f"  L{layer_idx}: MI(A)={mi_A:.3f} MI(B)={mi_B:.3f} MI(A,B)={mi_AB:.3f} "
                f"InteractInfo={interaction_info:.3f} Eff={efficiency:.3f} "
                f"Redund={redundancy:.3f} Synergy={synergy:.3f} PCA50={cumvar_50:.3f}")

        ib_results[pair_name] = pair_ib

    # ============================================================
    # S5: Topological Analysis (Persistence Homology)
    # ============================================================
    if not args.skip_topology:
        log("\n" + "=" * 70)
        log("S5: Topological Analysis (Simplified Persistence)")
        log("=" * 70)

        topo_results = {}

        for pair_name, pdata in all_pair_data.items():
            log(f"\n--- {pair_name} ---")
            n_sents = len(pdata['h_A0B0'])
            pair_topo = {}

            for layer_idx in key_layers:
                # Build point clouds for each condition
                h_A0B0 = torch.stack([pdata['h_A0B0'][i][layer_idx] for i in range(n_sents)]).numpy()
                h_A1B0 = torch.stack([pdata['h_A1B0'][i][layer_idx] for i in range(n_sents)]).numpy()
                h_A0B1 = torch.stack([pdata['h_A0B1'][i][layer_idx] for i in range(n_sents)]).numpy()
                h_A1B1 = torch.stack([pdata['h_A1B1'][i][layer_idx] for i in range(n_sents)]).numpy()

                # PCA to 20 dims for distance computation
                from sklearn.decomposition import PCA
                h_all = np.vstack([h_A0B0, h_A1B0, h_A0B1, h_A1B1])
                pca = PCA(n_components=min(20, n_sents - 1))
                h_all_pca = pca.fit_transform(h_all)

                n_pca = h_all_pca.shape[1]
                h0_pca = h_all_pca[:n_sents]
                h1_pca = h_all_pca[n_sents:2*n_sents]
                h2_pca = h_all_pca[2*n_sents:3*n_sents]
                h3_pca = h_all_pca[3*n_sents:]

                # Compute pairwise distance matrices
                from scipy.spatial.distance import pdist, squareform
                dist_00 = squareform(pdist(h0_pca, 'cosine'))
                dist_10 = squareform(pdist(h1_pca, 'cosine'))
                dist_01 = squareform(pdist(h2_pca, 'cosine'))
                dist_11 = squareform(pdist(h3_pca, 'cosine'))

                # Compute persistence diagrams for each condition
                persp_00 = compute_persistence_diagram_gpu(dist_00)
                persp_10 = compute_persistence_diagram_gpu(dist_10)
                persp_01 = compute_persistence_diagram_gpu(dist_01)
                persp_11 = compute_persistence_diagram_gpu(dist_11)

                # Compare topological signatures
                # Key metrics:
                # 1. Betti-0 profile (how components merge as threshold increases)
                # 2. Max H1 (loop count)
                # 3. Mean pairwise distance (overall spread)
                mean_dist = {
                    'A0B0': float(dist_00[np.triu_indices(n_sents, k=1)].mean()),
                    'A1B0': float(dist_10[np.triu_indices(n_sents, k=1)].mean()),
                    'A0B1': float(dist_01[np.triu_indices(n_sents, k=1)].mean()),
                    'A1B1': float(dist_11[np.triu_indices(n_sents, k=1)].mean()),
                }

                pair_topo[layer_idx] = {
                    'persistence_A0B0': persp_00,
                    'persistence_A1B0': persp_10,
                    'persistence_A0B1': persp_01,
                    'persistence_A1B1': persp_11,
                    'mean_cosine_dist': mean_dist,
                    'betti0_A0B0_final': persp_00['n_components_final'],
                    'betti0_A1B0_final': persp_10['n_components_final'],
                    'betti0_A0B1_final': persp_01['n_components_final'],
                    'betti0_A1B1_final': persp_11['n_components_final'],
                    'max_h1_A0B0': persp_00['max_h1'],
                    'max_h1_A1B0': persp_10['max_h1'],
                    'max_h1_A0B1': persp_01['max_h1'],
                    'max_h1_A1B1': persp_11['max_h1'],
                }

                log(f"  L{layer_idx}: Betti0(A0B0={persp_00['n_components_final']}, A1B0={persp_10['n_components_final']}, "
                    f"A0B1={persp_01['n_components_final']}, A1B1={persp_11['n_components_final']}) "
                    f"MaxH1(A0B0={persp_00['max_h1']}, A1B0={persp_10['max_h1']}, "
                    f"A0B1={persp_01['max_h1']}, A1B1={persp_11['max_h1']}) "
                    f"MeanDist(A0B0={mean_dist['A0B0']:.4f}, A1B1={mean_dist['A1B1']:.4f})")

            topo_results[pair_name] = pair_topo
    else:
        log("\nS5: Topology analysis skipped (--skip_topology)")
        topo_results = {}

    # ============================================================
    # S6: Cross-model summary
    # ============================================================
    log("\n" + "=" * 70)
    log("S6: Summary")
    log("=" * 70)

    # Aggregate fine structure
    log("\nFine Structure Summary (last layer):")
    all_cos_int_A = []
    all_frac_neg = []
    all_int_ratio = []
    for pair_name, fs in fine_structure_results.items():
        last_layer = max(fs.keys())
        d = fs[last_layer]
        all_cos_int_A.append(d['cos_int_A']['mean'])
        all_frac_neg.append(d['cos_int_A']['frac_negative'])
        all_int_ratio.append(d['int_ratio']['mean'])
    log(f"  cos(int,A): mean={np.mean(all_cos_int_A):.3f}, range=[{np.min(all_cos_int_A):.3f}, {np.max(all_cos_int_A):.3f}]")
    log(f"  frac_negative(cos_int_A): mean={np.mean(all_frac_neg):.3f}")
    log(f"  int_ratio: mean={np.mean(all_int_ratio):.3f}, range=[{np.min(all_int_ratio):.3f}, {np.max(all_int_ratio):.3f}]")

    # Aggregate IB
    log("\nInformation Bottleneck Summary (last layer):")
    all_mi_A = []
    all_mi_B = []
    all_mi_AB = []
    all_efficiency = []
    all_synergy = []
    for pair_name, ib in ib_results.items():
        last_layer = max(ib.keys())
        d = ib[last_layer]
        all_mi_A.append(d['mi_A'])
        all_mi_B.append(d['mi_B'])
        all_mi_AB.append(d['mi_AB'])
        all_efficiency.append(d['efficiency'])
        all_synergy.append(d['synergy'])
    log(f"  MI(A): mean={np.mean(all_mi_A):.3f}")
    log(f"  MI(B): mean={np.mean(all_mi_B):.3f}")
    log(f"  MI(A,B): mean={np.mean(all_mi_AB):.3f}")
    log(f"  Efficiency: mean={np.mean(all_efficiency):.3f}")
    log(f"  Synergy: mean={np.mean(all_synergy):.3f}")

    # Aggregate topology
    if topo_results:
        log("\nTopology Summary (last layer):")
        for pair_name, topo in topo_results.items():
            last_layer = max(topo.keys())
            d = topo[last_layer]
            log(f"  {pair_name}: Betti0={d['betti0_A0B0_final']}/{d['betti0_A1B1_final']}, "
                f"MaxH1={d['max_h1_A0B0']}/{d['max_h1_A1B1']}, "
                f"MeanDist(A0B0)={d['mean_cosine_dist']['A0B0']:.4f}, "
                f"MeanDist(A1B1)={d['mean_cosine_dist']['A1B1']:.4f}")

    # Save results
    results = {
        'model': config['name'],
        'model_key': model_key,
        'n_pairs': n_pairs,
        'timestamp': datetime.now().isoformat(),
        'fine_structure': fine_structure_results,
        'information_bottleneck': ib_results,
        'topology': topo_results,
    }

    with open(f'{out_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\nResults saved to {out_dir}/results.json")
    log(f"Total time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
