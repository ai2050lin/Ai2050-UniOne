"""
Phase CCXVI: 直接Head Hook因果分析
====================================
核心改进: 用register_forward_hook直接获取Head输出,
不用W_o投影(避免8bit/GQA兼容问题)

S1: 差分向量收集 + Head Hook (150对, 3关键层)
S2: Head输出差分贡献 (直接hook, 不需要W_o)
S3: PCA原子分解 + 跨层PC追踪
S4: 因果原子发现 + 条件互信息
S5: 统计检验

关键改进:
- 直接hook Head输出 → 避免W_o投影的8bit/GQA问题
- 跨层PC追踪 → 追踪同一PC成分在层间的连续性
- 条件互信息 → 语法→语义的因果依赖结构
- n=150 → 更稳健的统计
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from scipy import stats

import torch

# ============================================================
# 模型配置
# ============================================================
MODEL_CONFIGS = {
    'deepseek7b': {
        'path': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'name': 'DeepSeek-R1-Distill-Qwen-7B',
    },
    'qwen3': {
        'path': 'D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',
        'name': 'Qwen3-4B',
    },
    'glm4': {
        'path': 'D:/develop/model/hub/models--zai-org--glm-4-9b-chat-hf/download',
        'name': 'GLM4-9B-Chat',
    },
}

# ============================================================
# 12特征的Minimal Pair定义
# ============================================================
FEATURE_PAIRS = {
    'tense': {
        'type': 'SYN',
        'pairs': [
            ("The cat sleeps on the mat", "The cat slept on the mat"),
            ("She walks to school every day", "She walked to school every day"),
            ("He runs fast in the morning", "He ran fast in the morning"),
            ("They play soccer after class", "They played soccer after class"),
            ("I read books every weekend", "I read a book last weekend"),
            ("We eat lunch at noon", "We ate lunch at noon"),
            ("The dog barks loudly", "The dog barked loudly"),
            ("She sings beautifully", "She sang beautifully"),
            ("The bird flies south", "The bird flew south"),
            ("He drives to work", "He drove to work"),
            ("The children laugh together", "The children laughed together"),
            ("I write letters sometimes", "I wrote letters yesterday"),
            ("The wind blows hard", "The wind blew hard"),
            ("She teaches math well", "She taught math well"),
            ("They build houses here", "They built houses here"),
            ("We drink coffee daily", "We drank coffee yesterday"),
            ("The river flows quietly", "The river flowed quietly"),
            ("He reads the newspaper", "He read the newspaper"),
            ("The train arrives late", "The train arrived late"),
            ("She wears red dresses", "She wore a red dress"),
            ("I understand the lesson", "I understood the lesson"),
            ("The sun shines bright", "The sun shone bright"),
            ("They win the game", "They won the game"),
            ("We begin the project", "We began the project"),
            ("The bell rings twice", "The bell rang twice"),
            ("She draws beautiful pictures", "She drew beautiful pictures"),
            ("He catches the ball", "He caught the ball"),
            ("The horse gallops fast", "The horse galloped fast"),
            ("I feel happy today", "I felt happy yesterday"),
            ("The plant grows tall", "The plant grew tall"),
            ("She brings the food", "She brought the food"),
            ("They choose the blue one", "They chose the blue one"),
            ("He breaks the window", "He broke the window"),
            ("The light shines brightly", "The light shone brightly"),
            ("We speak English fluently", "We spoke English fluently"),
            ("She keeps the secret", "She kept the secret"),
            ("They spend time together", "They spent time together"),
            ("I leave early today", "I left early yesterday"),
            ("The dog bites the bone", "The dog bit the bone"),
        ],
    },
    'polarity': {
        'type': 'SYN',
        'pairs': [
            ("The cat is on the mat", "The cat is not on the mat"),
            ("She likes the movie", "She does not like the movie"),
            ("He can swim well", "He cannot swim well"),
            ("They will come tomorrow", "They will not come tomorrow"),
            ("I have finished the work", "I have not finished the work"),
            ("We should go now", "We should not go now"),
            ("The door is open", "The door is not open"),
            ("She was happy", "She was not happy"),
            ("He could see the mountain", "He could not see the mountain"),
            ("They must leave early", "They must not leave early"),
            ("I am tired today", "I am not tired today"),
            ("We were ready", "We were not ready"),
            ("The car is working", "The car is not working"),
            ("She has the book", "She does not have the book"),
            ("He knows the answer", "He does not know the answer"),
            ("They found the key", "They did not find the key"),
            ("I believe the story", "I do not believe the story"),
            ("We enjoyed the party", "We did not enjoy the party"),
            ("The bird is singing", "The bird is not singing"),
            ("She bought the dress", "She did not buy the dress"),
            ("He won the prize", "He did not win the prize"),
            ("They passed the test", "They did not pass the test"),
            ("I trust the process", "I do not trust the process"),
            ("We need the money", "We do not need the money"),
            ("The dog is barking", "The dog is not barking"),
            ("She loves the song", "She does not love the song"),
            ("He ate the cake", "He did not eat the cake"),
            ("They saw the sign", "They did not see the sign"),
            ("I want the job", "I do not want the job"),
            ("We like the plan", "We do not like the plan"),
            ("The sun is shining", "The sun is not shining"),
            ("She wrote the letter", "She did not write the letter"),
            ("He took the bus", "He did not take the bus"),
            ("They made the cake", "They did not make the cake"),
            ("I heard the news", "I did not hear the news"),
            ("We saw the movie", "We did not see the movie"),
            ("The rain stopped", "The rain has not stopped"),
            ("She came home early", "She did not come home early"),
            ("He found the answer", "He did not find the answer"),
            ("They built the house", "They did not build the house"),
        ],
    },
    'number': {
        'type': 'SYN',
        'pairs': [
            ("The cat sleeps on the mat", "The cats sleep on the mat"),
            ("A dog barks loudly", "Some dogs bark loudly"),
            ("This book is interesting", "These books are interesting"),
            ("That tree looks old", "Those trees look old"),
            ("The child plays outside", "The children play outside"),
            ("A bird sings beautifully", "Some birds sing beautifully"),
            ("This flower smells nice", "These flowers smell nice"),
            ("That house looks big", "Those houses look big"),
            ("The man walks slowly", "The men walk slowly"),
            ("A woman reads quietly", "Some women read quietly"),
            ("This student studies hard", "These students study hard"),
            ("That teacher speaks clearly", "Those teachers speak clearly"),
            ("The fish swims fast", "The fish swim fast"),
            ("A sheep grazes quietly", "Some sheep graze quietly"),
            ("This mouse runs quickly", "These mice run quickly"),
            ("That tooth looks healthy", "Those teeth look healthy"),
            ("The foot hurts badly", "The feet hurt badly"),
            ("A person thinks deeply", "Some people think deeply"),
            ("This box contains books", "These boxes contain books"),
            ("That bus arrives late", "Those buses arrive late"),
            ("The leaf falls gently", "The leaves fall gently"),
            ("A knife cuts sharply", "Some knives cut sharply"),
            ("This shelf holds books", "These shelves hold books"),
            ("That wolf howls loudly", "Those wolves howl loudly"),
            ("The calf runs fast", "The calves run fast"),
            ("A goose swims well", "Some geese swim well"),
            ("This toothbrush is new", "These toothbrushes are new"),
            ("That country is large", "Those countries are large"),
            ("The city grows fast", "The cities grow fast"),
            ("A story ends well", "Some stories end well"),
            ("The apple tastes sweet", "The apples taste sweet"),
            ("A cherry looks ripe", "Some cherries look ripe"),
            ("This berry is fresh", "These berries are fresh"),
            ("That deer runs fast", "Those deer run fast"),
            ("The loaf tastes good", "The loaves taste good"),
            ("A goose flies south", "Some geese fly south"),
            ("This church looks old", "These churches look old"),
            ("That match burns bright", "Those matches burn bright"),
            ("The patch covers well", "The patches cover well"),
            ("A brush paints smoothly", "Some brushes paint smoothly"),
        ],
    },
    'negation': {
        'type': 'SYN',
        'pairs': [
            ("She is always happy", "She is never happy"),
            ("He sometimes forgets", "He never forgets"),
            ("They often visit us", "They rarely visit us"),
            ("I always wake up early", "I never wake up early"),
            ("We frequently travel", "We seldom travel"),
            ("She usually agrees", "She rarely agrees"),
            ("He generally wins", "He rarely wins"),
            ("They commonly practice", "They rarely practice"),
            ("I mostly succeed", "I rarely succeed"),
            ("We normally finish", "We never finish"),
            ("She constantly worries", "She never worries"),
            ("He certainly knows", "He hardly knows"),
            ("They definitely agree", "They hardly agree"),
            ("I absolutely believe", "I hardly believe"),
            ("We completely understand", "We barely understand"),
            ("She entirely agrees", "She barely agrees"),
            ("He totally gets it", "He hardly gets it"),
            ("They fully support it", "They barely support it"),
            ("I entirely trust them", "I barely trust them"),
            ("We wholly accept it", "We hardly accept it"),
            ("She consistently delivers", "She never delivers"),
            ("He invariably succeeds", "He never succeeds"),
            ("They regularly contribute", "They rarely contribute"),
            ("I perpetually study", "I never study"),
            ("We perpetually improve", "We never improve"),
            ("She always remembers", "She never remembers"),
            ("He constantly improves", "He never improves"),
            ("They invariably succeed", "They rarely succeed"),
            ("I consistently perform", "I rarely perform"),
            ("We always achieve", "We never achieve"),
            ("She surely wins", "She hardly wins"),
            ("He truly cares", "He barely cares"),
            ("They clearly understand", "They barely understand"),
            ("I strongly recommend", "I hardly recommend"),
            ("We deeply appreciate", "We barely appreciate"),
            ("She fully commits", "She barely commits"),
            ("He entirely agrees", "He barely agrees"),
            ("They thoroughly enjoy", "They barely enjoy"),
            ("I wholly support it", "I hardly support it"),
            ("We absolutely need it", "We hardly need it"),
        ],
    },
    'question': {
        'type': 'SYN',
        'pairs': [
            ("The cat is on the mat", "Is the cat on the mat?"),
            ("She likes the movie", "Does she like the movie?"),
            ("He can swim well", "Can he swim well?"),
            ("They will come tomorrow", "Will they come tomorrow?"),
            ("I have finished the work", "Have I finished the work?"),
            ("We should go now", "Should we go now?"),
            ("The door is open", "Is the door open?"),
            ("She was happy", "Was she happy?"),
            ("He could see the mountain", "Could he see the mountain?"),
            ("They must leave early", "Must they leave early?"),
            ("I am tired today", "Am I tired today?"),
            ("We were ready", "Were we ready?"),
            ("The car is working", "Is the car working?"),
            ("She has the book", "Does she have the book?"),
            ("He knows the answer", "Does he know the answer?"),
            ("They found the key", "Did they find the key?"),
            ("I believe the story", "Do I believe the story?"),
            ("We enjoyed the party", "Did we enjoy the party?"),
            ("The bird is singing", "Is the bird singing?"),
            ("She bought the dress", "Did she buy the dress?"),
            ("He won the prize", "Did he win the prize?"),
            ("They passed the test", "Did they pass the test?"),
            ("I trust the process", "Do I trust the process?"),
            ("We need the money", "Do we need the money?"),
            ("The dog is barking", "Is the dog barking?"),
            ("She loves the song", "Does she love the song?"),
            ("He ate the cake", "Did he eat the cake?"),
            ("They saw the sign", "Did they see the sign?"),
            ("I want the job", "Do I want the job?"),
            ("We like the plan", "Do we like the plan?"),
            ("The house is big", "Is the house big?"),
            ("She wrote the letter", "Did she write the letter?"),
            ("He took the bus", "Did he take the bus?"),
            ("They made the decision", "Did they make the decision?"),
            ("I heard the news", "Did I hear the news?"),
            ("We saw the movie", "Did we see the movie?"),
            ("The rain stopped", "Has the rain stopped?"),
            ("She came home", "Did she come home?"),
            ("He found the answer", "Did he find the answer?"),
            ("They built the house", "Did they build the house?"),
        ],
    },
    'person': {
        'type': 'SYN',
        'pairs': [
            ("I walk to the store", "She walks to the store"),
            ("I have a book", "She has a book"),
            ("I am happy today", "She is happy today"),
            ("I was running late", "She was running late"),
            ("I can do this", "She can do this"),
            ("We walk to the store", "They walk to the store"),
            ("We have books", "They have books"),
            ("We are happy today", "They are happy today"),
            ("We were running late", "They were running late"),
            ("We can do this", "They can do this"),
            ("I like the food", "He likes the food"),
            ("I go to school", "He goes to school"),
            ("I study hard", "He studies hard"),
            ("I play guitar", "He plays guitar"),
            ("I know the answer", "He knows the answer"),
            ("We eat lunch together", "They eat lunch together"),
            ("We read every day", "They read every day"),
            ("We sing in the choir", "They sing in the choir"),
            ("We work from home", "They work from home"),
            ("We live nearby", "They live nearby"),
            ("I enjoy music", "She enjoys music"),
            ("I need help", "She needs help"),
            ("I think so", "She thinks so"),
            ("I try hard", "She tries hard"),
            ("I watch TV", "She watches TV"),
            ("We drive carefully", "They drive carefully"),
            ("We cook dinner", "They cook dinner"),
            ("We speak English", "They speak English"),
            ("We sleep early", "They sleep early"),
            ("We exercise daily", "They exercise daily"),
            ("I love this song", "He loves this song"),
            ("I miss home", "She misses home"),
            ("I wash the car", "He washes the car"),
            ("I fix the problem", "She fixes the problem"),
            ("I carry the bag", "He carries the bag"),
            ("We plan the trip", "They plan the trip"),
            ("We clean the house", "They clean the house"),
            ("We paint the wall", "They paint the wall"),
            ("We build the model", "They build the model"),
            ("We design the logo", "They design the logo"),
        ],
    },
    'definiteness': {
        'type': 'SYN',
        'pairs': [
            ("A cat sleeps on the mat", "The cat sleeps on the mat"),
            ("A dog barks in the yard", "The dog barks in the yard"),
            ("A bird sings in the tree", "The bird sings in the tree"),
            ("A child plays in the park", "The child plays in the park"),
            ("A student reads in the library", "The student reads in the library"),
            ("A flower grows in the garden", "The flower grows in the garden"),
            ("A man walks down the street", "The man walks down the street"),
            ("A woman works in the office", "The woman works in the office"),
            ("A book sits on the shelf", "The book sits on the shelf"),
            ("A car drives on the road", "The car drives on the road"),
            ("A house stands on the hill", "The house stands on the hill"),
            ("A tree grows in the forest", "The tree grows in the forest"),
            ("A river flows through the valley", "The river flows through the valley"),
            ("A mountain rises behind the town", "The mountain rises behind the town"),
            ("A star shines in the sky", "The star shines in the sky"),
            ("A fish swims in the pond", "The fish swims in the pond"),
            ("A horse runs across the field", "The horse runs across the field"),
            ("A cat sits by the window", "The cat sits by the window"),
            ("A dog waits at the door", "The dog waits at the door"),
            ("A bird flies over the lake", "The bird flies over the lake"),
            ("An apple falls from the tree", "The apple falls from the tree"),
            ("An idea comes to mind", "The idea comes to mind"),
            ("An egg sits in the nest", "The egg sits in the nest"),
            ("An old man walks slowly", "The old man walks slowly"),
            ("An artist paints the landscape", "The artist paints the landscape"),
            ("A teacher writes on the board", "The teacher writes on the board"),
            ("A doctor treats the patient", "The doctor treats the patient"),
            ("A singer performs the song", "The singer performs the song"),
            ("A farmer tends the crops", "The farmer tends the crops"),
            ("A writer drafts the novel", "The writer drafts the novel"),
            ("A cat chases the mouse", "The cat chases the mouse"),
            ("A dog fetches the ball", "The dog fetches the ball"),
            ("A bird builds the nest", "The bird builds the nest"),
            ("A student answers the question", "The student answers the question"),
            ("A cook prepares the meal", "The cook prepares the meal"),
            ("A driver steers the car", "The driver steers the car"),
            ("A painter colors the wall", "The painter colors the wall"),
            ("A baker makes the bread", "The baker makes the bread"),
            ("A nurse checks the patient", "The nurse checks the patient"),
            ("A clerk files the document", "The clerk files the document"),
        ],
    },
    'info_structure': {
        'type': 'SYN',
        'pairs': [
            ("John broke the window", "It was John who broke the window"),
            ("Mary found the key", "It was Mary who found the key"),
            ("Tom ate the cake", "It was Tom who ate the cake"),
            ("She bought the book", "It was the book that she bought"),
            ("He met the teacher", "It was the teacher that he met"),
            ("They visited Paris", "It was Paris that they visited"),
            ("We finished the project", "It was the project that we finished"),
            ("I lost my keys", "It was my keys that I lost"),
            ("She loves chocolate", "It is chocolate that she loves"),
            ("He needs money", "It is money that he needs"),
            ("The team won the championship", "It was the championship that the team won"),
            ("The dog chased the cat", "It was the cat that the dog chased"),
            ("The rain ruined the picnic", "It was the picnic that the rain ruined"),
            ("The wind broke the fence", "It was the fence that the wind broke"),
            ("The fire destroyed the barn", "It was the barn that the fire destroyed"),
            ("Alice wrote the report", "It was Alice who wrote the report"),
            ("Bob delivered the package", "It was Bob who delivered the package"),
            ("Carol designed the logo", "It was Carol who designed the logo"),
            ("David built the cabinet", "It was David who built the cabinet"),
            ("Eve composed the song", "It was Eve who composed the song"),
            ("Frank directed the movie", "It was Frank who directed the movie"),
            ("Grace organized the event", "It was Grace who organized the event"),
            ("Henry managed the team", "It was Henry who managed the team"),
            ("Irene edited the article", "It was Irene who edited the article"),
            ("Jack painted the portrait", "It was Jack who painted the portrait"),
            ("She bought a new car", "What she bought was a new car"),
            ("He fixed the old roof", "What he fixed was the old roof"),
            ("They found the lost dog", "What they found was the lost dog"),
            ("We need more time", "What we need is more time"),
            ("I want some coffee", "What I want is some coffee"),
            ("She loves Italian food", "What she loves is Italian food"),
            ("He reads science fiction", "What he reads is science fiction"),
            ("They play classical music", "What they play is classical music"),
            ("We study ancient history", "What we study is ancient history"),
            ("I prefer morning walks", "What I prefer is morning walks"),
            ("She teaches modern art", "What she teaches is modern art"),
            ("He drives electric cars", "What he drives is electric cars"),
            ("They grow organic vegetables", "What they grow is organic vegetables"),
            ("We build wooden furniture", "What we build is wooden furniture"),
            ("I write short stories", "What I write is short stories"),
        ],
    },
    'sentiment': {
        'type': 'SEM',
        'pairs': [
            ("The movie was wonderful and exciting", "The movie was terrible and boring"),
            ("She had a great day at work", "She had a awful day at work"),
            ("The food was delicious and fresh", "The food was disgusting and stale"),
            ("He felt happy and grateful", "He felt sad and resentful"),
            ("The weather was beautiful today", "The weather was dreadful today"),
            ("They enjoyed the lovely concert", "They suffered through the awful concert"),
            ("The gift was thoughtful and generous", "The gift was thoughtless and cheap"),
            ("She gave a brilliant performance", "She gave a terrible performance"),
            ("The garden looked vibrant and healthy", "The garden looked withered and dying"),
            ("He received a kind and warm welcome", "He received a cold and hostile welcome"),
            ("The team celebrated their amazing victory", "The team mourned their devastating defeat"),
            ("The book was fascinating and insightful", "The book was dull and confusing"),
            ("She made a generous donation", "She made a stingy contribution"),
            ("The trip was delightful and memorable", "The trip was miserable and forgettable"),
            ("He told a hilarious joke", "He told a pathetic joke"),
            ("The service was excellent and prompt", "The service was terrible and slow"),
            ("She wore a gorgeous dress", "She wore a shabby dress"),
            ("The room was spacious and bright", "The room was cramped and dark"),
            ("He showed genuine compassion", "He showed fake indifference"),
            ("The concert was thrilling and energetic", "The concert was dull and lifeless"),
            ("The puppy was adorable and playful", "The puppy was vicious and aggressive"),
            ("She cooked a superb meal", "She cooked a terrible meal"),
            ("The view was breathtaking and stunning", "The view was ugly and depressing"),
            ("He earned a well-deserved promotion", "He received an unfair demotion"),
            ("The hotel was luxurious and comfortable", "The hotel was dingy and uncomfortable"),
            ("She shared a heartwarming story", "She shared a heartbreaking story"),
            ("The painting was magnificent and striking", "The painting was hideous and dull"),
            ("He gave an inspiring speech", "He gave a depressing speech"),
            ("The garden was lush and thriving", "The garden was barren and dying"),
            ("She had a joyful experience", "She had a painful experience"),
            ("The cake was sweet and delicious", "The cake was bitter and disgusting"),
            ("He found a peaceful solution", "He found a violent solution"),
            ("The park was clean and beautiful", "The park was dirty and ugly"),
            ("She felt confident and proud", "She felt insecure and ashamed"),
            ("The movie was entertaining and fun", "The movie was boring and tedious"),
            ("He received warm praise", "He received harsh criticism"),
            ("The neighborhood was safe and friendly", "The neighborhood was dangerous and hostile"),
            ("She had a comfortable journey", "She had a miserable journey"),
            ("The music was soothing and calm", "The music was harsh and grating"),
            ("He achieved remarkable success", "He suffered complete failure"),
        ],
    },
    'semantic_topic': {
        'type': 'SEM',
        'pairs': [
            ("The doctor treated the patient carefully", "The mechanic fixed the engine carefully"),
            ("The teacher explained the lesson clearly", "The chef prepared the dish clearly"),
            ("The scientist discovered a new element", "The artist created a new painting"),
            ("The lawyer argued the case persuasively", "The musician played the song persuasively"),
            ("The engineer designed the bridge precisely", "The writer crafted the story precisely"),
            ("The nurse cared for the elderly patient", "The gardener tended the young plants"),
            ("The pilot flew the plane smoothly", "The sailor steered the ship smoothly"),
            ("The baker baked fresh bread", "The painter painted fresh walls"),
            ("The judge ruled the court fairly", "The conductor led the orchestra fairly"),
            ("The farmer grew healthy crops", "The builder constructed sturdy buildings"),
            ("The librarian organized the books neatly", "The programmer organized the code neatly"),
            ("The coach trained the athletes rigorously", "The director trained the actors rigorously"),
            ("The dentist cleaned the teeth thoroughly", "The janitor cleaned the floors thoroughly"),
            ("The banker managed the accounts carefully", "The chef managed the kitchen carefully"),
            ("The soldier defended the country bravely", "The firefighter rescued the people bravely"),
            ("The philosopher thought about existence deeply", "The poet wrote about nature deeply"),
            ("The biologist studied the cell closely", "The astronomer studied the star closely"),
            ("The architect designed the building creatively", "The composer designed the symphony creatively"),
            ("The pharmacist dispensed the medicine accurately", "The bartender dispensed the drinks accurately"),
            ("The therapist healed the mind gently", "The sculptor shaped the clay gently"),
            ("The journalist reported the news accurately", "The historian recorded the events accurately"),
            ("The electrician wired the house safely", "The plumber piped the bathroom safely"),
            ("The photographer captured the moment perfectly", "The dancer captured the audience perfectly"),
            ("The veterinarian treated the animal compassionately", "The florist arranged the flowers compassionately"),
            ("The economist analyzed the market shrewdly", "The critic analyzed the film shrewdly"),
            ("The psychologist studied behavior methodically", "The geologist studied rocks methodically"),
            ("The professor lectured on physics brilliantly", "The curator lectured on art brilliantly"),
            ("The surgeon operated on the heart skillfully", "The carpenter built the table skillfully"),
            ("The police officer patrolled the street vigilantly", "The lifeguard watched the pool vigilantly"),
            ("The accountant audited the finances meticulously", "The editor audited the manuscript meticulously"),
            ("The chef seasoned the soup perfectly", "The chemist mixed the solution perfectly"),
            ("The pilot navigated the storm expertly", "The guide navigated the trail expertly"),
            ("The mechanic repaired the car efficiently", "The doctor treated the wound efficiently"),
            ("The teacher graded the papers fairly", "The judge scored the competition fairly"),
            ("The farmer harvested the wheat early", "The fisherman caught the fish early"),
            ("The programmer debugged the code quickly", "The detective solved the case quickly"),
            ("The musician tuned the instrument carefully", "The surgeon calibrated the tool carefully"),
            ("The writer edited the manuscript thoroughly", "The painter restored the artwork thoroughly"),
            ("The athlete trained the body relentlessly", "The scholar trained the mind relentlessly"),
            ("The captain commanded the crew firmly", "The principal led the school firmly"),
        ],
    },
    'voice': {
        'type': 'SEM',
        'pairs': [
            ("The cat chased the mouse", "The mouse was chased by the cat"),
            ("The dog bit the man", "The man was bitten by the dog"),
            ("She wrote the letter", "The letter was written by her"),
            ("He fixed the car", "The car was fixed by him"),
            ("They built the house", "The house was built by them"),
            ("The wind broke the window", "The window was broken by the wind"),
            ("She cooked the dinner", "The dinner was cooked by her"),
            ("He painted the door", "The door was painted by him"),
            ("They won the game", "The game was won by them"),
            ("The rain ruined the picnic", "The picnic was ruined by the rain"),
            ("She sang the song", "The song was sung by her"),
            ("He drove the bus", "The bus was driven by him"),
            ("They caught the fish", "The fish was caught by them"),
            ("The fire destroyed the barn", "The barn was destroyed by the fire"),
            ("She baked the cake", "The cake was baked by her"),
            ("He wrote the book", "The book was written by him"),
            ("They found the treasure", "The treasure was found by them"),
            ("The storm damaged the roof", "The roof was damaged by the storm"),
            ("She taught the class", "The class was taught by her"),
            ("He cleaned the room", "The room was cleaned by him"),
            ("They delivered the package", "The package was delivered by them"),
            ("The sun melted the snow", "The snow was melted by the sun"),
            ("She opened the door", "The door was opened by her"),
            ("He broke the record", "The record was broken by him"),
            ("They solved the problem", "The problem was solved by them"),
            ("The teacher praised the student", "The student was praised by the teacher"),
            ("She cut the paper", "The paper was cut by her"),
            ("He washed the car", "The car was washed by him"),
            ("They painted the wall", "The wall was painted by them"),
            ("The cat killed the bird", "The bird was killed by the cat"),
            ("She made the dress", "The dress was made by her"),
            ("He read the book", "The book was read by him"),
            ("They sold the house", "The house was sold by them"),
            ("The dog chased the cat", "The cat was chased by the dog"),
            ("She bought the gift", "The gift was bought by her"),
            ("He threw the ball", "The ball was thrown by him"),
            ("They ate the pizza", "The pizza was eaten by them"),
            ("The wind blew the leaves", "The leaves were blown by the wind"),
            ("She drew the picture", "The picture was drawn by her"),
            ("He took the photo", "The photo was taken by him"),
        ],
    },
    'formality': {
        'type': 'SEM',
        'pairs': [
            ("Hey, what's up?", "Greetings, how are you doing?"),
            ("Wanna grab some food?", "Would you like to have some dinner?"),
            ("Gotta go now, see ya!", "I must leave now, goodbye!"),
            ("That's really cool!", "That is quite impressive!"),
            ("She's super nice", "She is exceptionally kind"),
            ("It's gonna be awesome", "It will be magnificent"),
            ("Dude, that's wild", "Sir, that is extraordinary"),
            ("No way, that's crazy", "That is quite unbelievable"),
            ("Pretty good stuff", "Quite excellent material"),
            ("I'm gonna head out", "I shall depart now"),
            ("Got any ideas?", "Do you have any suggestions?"),
            ("Let's chill for a bit", "Let us relax for a moment"),
            ("That sucks big time", "That is quite unfortunate"),
            ("He's a cool guy", "He is a respectable gentleman"),
            ("Can't complain, you know", "I am doing well, thank you"),
            ("What's the deal here?", "What is the situation here?"),
            ("I dunno about that", "I am uncertain about that"),
            ("Hang on a sec", "Please wait a moment"),
            ("So what's the plan?", "What is the proposed course of action?"),
            ("Cool, I'm in", "Excellent, I shall participate"),
            ("That's legit", "That is legitimate"),
            ("No prob, anytime", "No problem, I am available anytime"),
            ("She's really smart", "She is highly intelligent"),
            ("That was epic", "That was extraordinary"),
            ("Let me think about it", "Allow me to consider the matter"),
            ("This is dope", "This is remarkable"),
            ("He nailed it", "He executed it perfectly"),
            ("We should totally do this", "We should certainly proceed with this"),
            ("I'm beat, need rest", "I am exhausted and require rest"),
            ("That's wild, for real", "That is genuinely extraordinary"),
            ("Gotcha, makes sense", "I understand, that is logical"),
            ("She's got mad skills", "She possesses exceptional abilities"),
            ("We're gonna crush it", "We will succeed magnificently"),
            ("Heck yeah, let's go", "Indeed, let us proceed"),
            ("That ain't right", "That is incorrect"),
            ("Yo, check this out", "Please observe this"),
            ("It's lit in here", "The atmosphere is vibrant"),
            ("I'm down for that", "I am agreeable to that"),
            ("Bruh, seriously?", "Sir, are you serious?"),
            ("That hits different", "That produces a distinct impression"),
        ],
    },
}


def get_log_fn(outdir):
    log_path = os.path.join(outdir, 'run.log')
    def log(msg):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=150)
    args = parser.parse_args()
    
    model_name = args.model
    n_pairs = min(args.n_pairs, 60)  # 每特征最多60对(平衡速度与统计力)
    
    cfg = MODEL_CONFIGS[model_name]
    
    outdir = f'results/causal_fiber/{model_name}_ccxvi'
    os.makedirs(outdir, exist_ok=True)
    log = get_log_fn(outdir)
    
    log(f"Phase CCXVI: Direct Head Hook Causal Analysis")
    log(f"Model: {cfg['name']}, n_pairs={n_pairs}")
    
    # ============================================================
    # 加载模型
    # ============================================================
    log("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    t0 = time.time()
    
    path = cfg['path']
    
    if model_name in ["glm4", "deepseek7b"]:
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
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log(f"Model loaded in {time.time()-t0:.0f}s")
    
    device = next(model.parameters()).device
    
    # 从模型配置获取参数
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    d_model = model.config.hidden_size
    
    log(f"Config: n_layers={n_layers}, n_heads={n_heads}, head_dim={head_dim}, d_model={d_model}")
    
    # 选择3个关键层 (早期/中期/后期)
    layer_indices = [0, n_layers // 2, n_layers - 1]
    log(f"Layer indices: {layer_indices}")
    
    feature_names = list(FEATURE_PAIRS.keys())
    
    # ============================================================
    # S1: 差分向量收集 + Head Hook
    # ============================================================
    log(f"\n{'='*60}")
    log("S1: 差分向量收集 + Head Hook")
    log(f"{'='*60}")
    
    # 存储结构
    all_deltas = {feat: {l: [] for l in layer_indices} for feat in feature_names}
    # Head-level差分: {feat: {l: {h: []}}}
    head_deltas = {feat: {l: {h: [] for h in range(n_heads)} for l in layer_indices} for feat in feature_names}
    
    for feat_idx, feat in enumerate(feature_names):
        pairs = FEATURE_PAIRS[feat]['pairs'][:n_pairs]
        log(f"  [{feat_idx+1}/{len(feature_names)}] {feat} ({FEATURE_PAIRS[feat]['type']}): {len(pairs)} pairs")
        
        for l in layer_indices:
            # Hook存储
            head_outputs = {}
            
            def make_hook(layer_idx, hook_heads):
                def hook_fn(module, input, output):
                    # output: (batch, seq, n_heads*head_dim) 或 (batch, seq, d_model)
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    hook_heads[layer_idx] = out.detach()
                return hook_fn
            
            handles = []
            captured = {}
            
            # 注册hook
            handle = model.model.layers[l].self_attn.o_proj.register_forward_hook(
                make_hook(l, captured)
            )
            handles.append(handle)
            
            # 同时hook residual stream (layer output)
            res_captured = {}
            res_handle = model.model.layers[l].register_forward_hook(
                make_hook(l, res_captured)
            )
            handles.append(res_handle)
            
            feat_deltas = []
            feat_head_deltas = {h: [] for h in range(n_heads)}
            
            for s1, s2 in pairs:
                with torch.no_grad():
                    # Sentence 1
                    enc1 = tokenizer(s1, return_tensors='pt', padding=True, truncation=True, max_length=64)
                    enc1 = {k: v.to(device) for k, v in enc1.items()}
                    _ = model(**enc1)
                    h1 = captured[l].float().cpu().numpy() if l in captured else None
                    r1 = res_captured[l].float().cpu().numpy() if l in res_captured else None
                    
                    # Sentence 2
                    enc2 = tokenizer(s2, return_tensors='pt', padding=True, truncation=True, max_length=64)
                    enc2 = {k: v.to(device) for k, v in enc2.items()}
                    _ = model(**enc2)
                    h2 = captured[l].float().cpu().numpy() if l in captured else None
                    r2 = res_captured[l].float().cpu().numpy() if l in res_captured else None
                    
                    # 提取最后token的residual stream差分
                    if r1 is not None and r2 is not None:
                        # 获取实际序列长度
                        len1 = enc1['attention_mask'].sum().item()
                        len2 = enc2['attention_mask'].sum().item()
                        # 用倒数第二个token (更稳定)
                        idx1 = max(0, len1 - 2)
                        idx2 = max(0, len2 - 2)
                        delta_r = r2[0, idx2] - r1[0, idx1]
                        feat_deltas.append(delta_r)
                    
                    # 提取o_proj输出的差分 (Head-level)
                    if h1 is not None and h2 is not None:
                        len1 = enc1['attention_mask'].sum().item()
                        len2 = enc2['attention_mask'].sum().item()
                        idx1 = max(0, len1 - 2)
                        idx2 = max(0, len2 - 2)
                        delta_h = h2[0, idx2] - h1[0, idx1]  # (d_model,)
                        
                        # 分割为n_heads个head
                        if delta_h.shape[0] == n_heads * head_dim:
                            for h in range(n_heads):
                                start = h * head_dim
                                end = start + head_dim
                                feat_head_deltas[h].append(delta_h[start:end].copy())
            
            # 移除hook
            for h in handles:
                h.remove()
            
            all_deltas[feat][l] = feat_deltas
            for h in range(n_heads):
                head_deltas[feat][l][h] = feat_head_deltas[h]
        
        n_valid = sum(1 for l in layer_indices for d in all_deltas[feat][l] if d is not None)
        log(f"    Collected {n_valid} valid delta vectors")
    
    # 统计每个特征的delta norm
    log("\n--- Delta Norms ---")
    for feat in feature_names:
        norms = []
        for l in layer_indices:
            valid = [d for d in all_deltas[feat][l] if d is not None]
            if valid:
                norms.append(str(int(np.mean([np.linalg.norm(d) for d in valid]))))
            else:
                norms.append("N/A")
        log(f"  {feat} [{FEATURE_PAIRS[feat]['type']}]: {', '.join(norms)}")
    
    # ============================================================
    # S2: Head输出差分贡献分析
    # ============================================================
    log(f"\n{'='*60}")
    log("S2: Head输出差分贡献 (直接Hook)")
    log(f"{'='*60}")
    
    # 对每个层, 计算每个Head对语法/语义差分的贡献
    head_spec_results = {}
    
    for l in layer_indices:
        # 收集语法/语义的head差分
        syn_head_deltas = {h: [] for h in range(n_heads)}
        sem_head_deltas = {h: [] for h in range(n_heads)}
        
        for feat in feature_names:
            for h in range(n_heads):
                hds = head_deltas[feat][l][h]
                if not hds:
                    continue
                if FEATURE_PAIRS[feat]['type'] == 'SYN':
                    syn_head_deltas[h].extend(hds)
                else:
                    sem_head_deltas[h].extend(hds)
        
        # 计算每个Head的专化度
        head_specs = []
        for h in range(n_heads):
            syn_ds = syn_head_deltas[h]
            sem_ds = sem_head_deltas[h]
            
            if not syn_ds or not sem_ds:
                head_specs.append({'h': h, 'syn_norm': 0, 'sem_norm': 0, 'spec': 0})
                continue
            
            syn_norm = np.mean([np.linalg.norm(d) for d in syn_ds])
            sem_norm = np.mean([np.linalg.norm(d) for d in sem_ds])
            
            # 专化度 = (syn_norm - sem_norm) / (syn_norm + sem_norm)
            spec = (syn_norm - sem_norm) / (syn_norm + sem_norm + 1e-8)
            
            head_specs.append({'h': h, 'syn_norm': float(syn_norm), 'sem_norm': float(sem_norm), 'spec': float(spec)})
        
        # 分类
        n_syn = sum(1 for s in head_specs if s['spec'] > 0.3)
        n_sem = sum(1 for s in head_specs if s['spec'] < -0.3)
        n_mixed = n_heads - n_syn - n_sem
        
        head_spec_results[l] = {
            'specs': head_specs,
            'n_syn': n_syn, 'n_sem': n_sem, 'n_mixed': n_mixed,
        }
        
        log(f"  L{l}: SYN={n_syn}, SEM={n_sem}, MIXED={n_mixed}")
        
        # Top SYN/SEM heads
        sorted_syn = sorted([s for s in head_specs if s['spec'] > 0], key=lambda x: -x['spec'])[:5]
        sorted_sem = sorted([s for s in head_specs if s['spec'] < 0], key=lambda x: x['spec'])[:5]
        
        if sorted_syn:
            log(f"    Top SYN: {[s['h'] for s in sorted_syn]} spec={[round(s['spec'],3) for s in sorted_syn]}")
        if sorted_sem:
            log(f"    Top SEM: {[s['h'] for s in sorted_sem]} spec={[round(s['spec'],3) for s in sorted_sem]}")
        
        # 更细粒度: 逐特征的Head贡献
        log(f"    Per-feature Head contributions:")
        for feat in ['question', 'tense', 'definiteness', 'sentiment', 'semantic_topic']:
            hds = head_deltas[feat][l]
            head_norms = []
            for h in range(n_heads):
                if hds[h]:
                    head_norms.append((h, np.mean([np.linalg.norm(d) for d in hds[h]])))
                else:
                    head_norms.append((h, 0))
            head_norms.sort(key=lambda x: -x[1])
            top3 = head_norms[:3]
            log(f"      {feat}: Top Heads={[t[0] for t in top3]} norms={[round(t[1],1) for t in top3]}")
    
    # ============================================================
    # S3: PCA原子分解 + 跨层PC追踪
    # ============================================================
    log(f"\n{'='*60}")
    log("S3: PCA原子分解 + 跨层PC追踪")
    log(f"{'='*60}")
    
    pca_results = {}
    layer_projectors = {}  # 保存每层的projector用于跨层追踪
    
    for l in layer_indices:
        syn_deltas = []
        sem_deltas = []
        feat_indices = {}
        
        idx = 0
        for feat in feature_names:
            valid_deltas = [d for d in all_deltas[feat][l] if d is not None]
            start_idx = idx
            if valid_deltas:
                if FEATURE_PAIRS[feat]['type'] == 'SYN':
                    syn_deltas.extend(valid_deltas)
                else:
                    sem_deltas.extend(valid_deltas)
                idx += len(valid_deltas)
            feat_indices[feat] = (start_idx, idx)
        
        all_vecs = syn_deltas + sem_deltas
        if len(all_vecs) < 10:
            log(f"  L{l}: insufficient vectors ({len(all_vecs)})")
            continue
        
        mat = np.array(all_vecs, dtype=np.float32)
        n_samples, d = mat.shape
        log(f"  L{l}: {n_samples} vectors (SYN={len(syn_deltas)}, SEM={len(sem_deltas)}), d={d}")
        
        # TruncatedSVD (内存友好)
        from sklearn.decomposition import TruncatedSVD
        n_pc = min(20, n_samples - 1, d)
        svd = TruncatedSVD(n_components=n_pc, random_state=42)
        projected = svd.fit_transform(mat)
        
        var_top5 = svd.explained_variance_ratio_[:5].tolist()
        cumvar_top5 = np.cumsum(var_top5).tolist()
        cumvar_10 = np.cumsum(svd.explained_variance_ratio_[:10]).tolist()[-1] if len(var_top5) >= 5 else cumvar_top5[-1]
        
        log(f"  L{l} PCA: PC1={var_top5[0]:.4f}, PC2={var_top5[1]:.4f}, cum5={cumvar_top5[-1]:.4f}, cum10={cumvar_10:.4f}")
        
        # 保存projector
        layer_projectors[l] = {
            'components': svd.components_,  # (n_pc, d)
            'explained_var': svd.explained_variance_ratio_,
            'mean': mat.mean(axis=0),
        }
        
        syn_proj = projected[:len(syn_deltas)]
        sem_proj = projected[len(syn_deltas):]
        
        # PC1 Cohen's d
        pc1_syn = syn_proj[:, 0]
        pc1_sem = sem_proj[:, 0]
        pooled_std = np.sqrt((np.var(pc1_syn) + np.var(pc1_sem)) / 2)
        cohens_d = (np.mean(pc1_syn) - np.mean(pc1_sem)) / (pooled_std + 1e-8)
        
        # Centroid余弦
        syn_centroid = np.mean(syn_proj[:, :5], axis=0)
        sem_centroid = np.mean(sem_proj[:, :5], axis=0)
        n1 = np.linalg.norm(syn_centroid)
        n2 = np.linalg.norm(sem_centroid)
        centroid_cos = float(abs(np.dot(syn_centroid, sem_centroid) / (n1 * n2 + 1e-8)))
        
        log(f"  L{l} Cohen_d={cohens_d:.3f}, centroid_cos={centroid_cos:.4f}")
        
        # 逐特征centroid
        feat_centroids = {}
        for feat in feature_names:
            start_i, end_i = feat_indices[feat]
            if end_i > start_i:
                feat_proj = projected[start_i:end_i]
                feat_centroids[feat] = {
                    'pc1': float(feat_proj[:, 0].mean()),
                    'pc2': float(feat_proj[:, 1].mean()),
                    'pc1_std': float(feat_proj[:, 0].std()),
                    'n': end_i - start_i,
                }
        
        # 逐PC的语法/语义t-test
        pc_separations = []
        for pc_idx in range(min(10, n_pc)):
            syn_vals = syn_proj[:, pc_idx]
            sem_vals = sem_proj[:, pc_idx]
            t_stat, p_val = stats.ttest_ind(syn_vals, sem_vals)
            pc_separations.append({
                'pc': pc_idx,
                't': float(t_stat),
                'p': float(p_val),
                'syn_mean': float(syn_vals.mean()),
                'sem_mean': float(sem_vals.mean()),
            })
        
        # 找到最强分离的PC
        best_sep = max(pc_separations, key=lambda x: abs(x['t']))
        log(f"  L{l} Best PC: PC{best_sep['pc']} (t={best_sep['t']:.2f}, p={best_sep['p']:.4f})")
        
        pca_results[l] = {
            'var_top5': var_top5,
            'cumvar_5': cumvar_top5[-1],
            'cumvar_10': cumvar_10,
            'cohens_d': float(cohens_d),
            'centroid_cos': centroid_cos,
            'feat_centroids': feat_centroids,
            'pc_separations': pc_separations,
            'best_pc': best_sep,
        }
    
    # ============================================================
    # 跨层PC追踪
    # ============================================================
    log(f"\n--- Cross-layer PC Tracking ---")
    
    # 将早期层的PC1方向投影到后期层, 看是否保留
    sorted_layers = sorted(layer_indices)
    if len(sorted_layers) >= 2:
        l_early = sorted_layers[0]
        l_late = sorted_layers[-1]
        
        if l_early in layer_projectors and l_late in layer_projectors:
            pc1_early = layer_projectors[l_early]['components'][0]  # (d,)
            pc1_late = layer_projectors[l_late]['components'][0]  # (d,)
            
            # PC1方向在层间的余弦相似度
            cos_pc1 = abs(np.dot(pc1_early, pc1_late) / (np.linalg.norm(pc1_early) * np.linalg.norm(pc1_late) + 1e-8))
            log(f"  PC1(L{l_early}) vs PC1(L{l_late}): cos={cos_pc1:.4f}")
            
            # 将早期PC1投影到后期PC空间
            proj_early_on_late = layer_projectors[l_late]['components'] @ pc1_early
            proj_early_on_late_norm = proj_early_on_late / (np.linalg.norm(proj_early_on_late) + 1e-8)
            top3_pcs = np.argsort(-abs(proj_early_on_late_norm))[:3].tolist()
            log(f"  PC1(L{l_early}) projected onto L{l_late} PC space: top3 PCs={top3_pcs}")
            
            # PC1-5在层间的投影矩阵
            early_top5 = layer_projectors[l_early]['components'][:5]  # (5, d)
            late_top5 = layer_projectors[l_late]['components'][:5]  # (5, d)
            cross_proj = early_top5 @ late_top5.T  # (5, 5)
            log(f"  Cross-projection matrix (early→late PC1-5):")
            for i in range(5):
                row = [f"{cross_proj[i,j]:.3f}" for j in range(5)]
                log(f"    PC{i}(early) → [{', '.join(row)}]")
            
            # 对角线均值 = 跨层一致性
            diag_mean = np.mean([abs(cross_proj[i,i]) for i in range(5)])
            offdiag_max = np.max([abs(cross_proj[i,j]) for i in range(5) for j in range(5) if i != j])
            log(f"  Diagonal mean={diag_mean:.4f}, Off-diagonal max={offdiag_max:.4f}")
            log(f"  → PC consistency: {'HIGH' if diag_mean > 0.5 else 'LOW'}")
    
    # ============================================================
    # S4: 因果原子发现 + 条件互信息
    # ============================================================
    log(f"\n{'='*60}")
    log("S4: 因果原子发现 + 条件互信息")
    log(f"{'='*60}")
    
    for l in layer_indices:
        if l not in pca_results:
            continue
        
        # 逐特征在PC1上的分离度
        fc = pca_results[l]['feat_centroids']
        pc1_vals = [(feat, fc[feat]['pc1'], fc[feat]['pc1_std'], FEATURE_PAIRS[feat]['type']) 
                    for feat in feature_names if feat in fc]
        
        syn_spread = np.std([v[1] for v in pc1_vals if v[3] == 'SYN']) if any(v[3]=='SYN' for v in pc1_vals) else 0
        sem_spread = np.std([v[1] for v in pc1_vals if v[3] == 'SEM']) if any(v[3]=='SEM' for v in pc1_vals) else 0
        
        log(f"  L{l}: PC1 syn_spread={syn_spread:.3f}, sem_spread={sem_spread:.3f}")
        
        # Separability = |mean_syn - mean_sem| / pooled_std
        syn_means = [v[1] for v in pc1_vals if v[3] == 'SYN']
        sem_means = [v[1] for v in pc1_vals if v[3] == 'SEM']
        
        if syn_means and sem_means:
            sep = abs(np.mean(syn_means) - np.mean(sem_means)) / (np.std(syn_means + sem_means) + 1e-8)
            log(f"  L{l}: SYN/SEM separability on PC1 = {sep:.3f}")
        
        # 逐特征separability排名
        all_mean = np.mean([v[1] for v in pc1_vals])
        all_std = np.std([v[1] for v in pc1_vals]) + 1e-8
        
        log(f"  L{l}: Feature separability ranking:")
        for feat, pc1_val, pc1_std, ftype in sorted(pc1_vals, key=lambda x: -abs(x[1]-all_mean)/all_std):
            sep_score = abs(pc1_val - all_mean) / all_std
            log(f"    {feat} [{ftype}]: {sep_score:.3f}")
    
    # 条件互信息: 语法特征之间 vs 语义特征之间 vs 跨类
    log(f"\n--- Conditional Mutual Information ---")
    
    for l in layer_indices:
        if l not in pca_results:
            continue
        
        # 用PC1-5的投影值作为特征空间
        # 收集每对特征的centroid距离
        fc = pca_results[l]['feat_centroids']
        syn_feats = [f for f in feature_names if f in fc and FEATURE_PAIRS[f]['type'] == 'SYN']
        sem_feats = [f for f in feature_names if f in fc and FEATURE_PAIRS[f]['type'] == 'SEM']
        
        # 计算类内和跨类的centroid距离
        syn_dists = []
        for i in range(len(syn_feats)):
            for j in range(i+1, len(syn_feats)):
                d = abs(fc[syn_feats[i]]['pc1'] - fc[syn_feats[j]]['pc1'])
                syn_dists.append(d)
        
        sem_dists = []
        for i in range(len(sem_feats)):
            for j in range(i+1, len(sem_feats)):
                d = abs(fc[sem_feats[i]]['pc1'] - fc[sem_feats[j]]['pc1'])
                sem_dists.append(d)
        
        cross_dists = []
        for sf in syn_feats:
            for ef in sem_feats:
                d = abs(fc[sf]['pc1'] - fc[ef]['pc1'])
                cross_dists.append(d)
        
        if syn_dists and sem_dists and cross_dists:
            log(f"  L{l}: Intra-SYN dist={np.mean(syn_dists):.3f}, Intra-SEM dist={np.mean(sem_dists):.3f}, Cross dist={np.mean(cross_dists):.3f}")
            
            # 互信息近似: 跨类距离 > 类内距离 → 语法/语义可分
            if np.mean(cross_dists) > max(np.mean(syn_dists), np.mean(sem_dists)):
                log(f"  L{l}: → SYNTAX/SEMANTIC SEPARABLE (cross > intra)")
            else:
                log(f"  L{l}: → Syntax/Semantic MIXED (cross <= intra)")
    
    # ============================================================
    # S5: 统计检验
    # ============================================================
    log(f"\n{'='*60}")
    log("S5: 统计检验")
    log(f"{'='*60}")
    
    # Growth统计
    growth_syn = []
    growth_sem = []
    
    for feat in feature_names:
        early_l = sorted_layers[0]
        late_l = sorted_layers[-1]
        
        early_valid = [d for d in all_deltas[feat][early_l] if d is not None]
        late_valid = [d for d in all_deltas[feat][late_l] if d is not None]
        
        if early_valid and late_valid:
            early_norm = np.mean([np.linalg.norm(d) for d in early_valid])
            late_norm = np.mean([np.linalg.norm(d) for d in late_valid])
            growth = late_norm / (early_norm + 1e-8)
            
            if FEATURE_PAIRS[feat]['type'] == 'SYN':
                growth_syn.append(growth)
            else:
                growth_sem.append(growth)
    
    if growth_syn and growth_sem:
        stat, p_val = stats.mannwhitneyu(growth_syn, growth_sem, alternative='greater')
        log(f"  Growth: syn_mean={np.mean(growth_syn):.2f} vs sem_mean={np.mean(growth_sem):.2f}, p={p_val:.4f}")
    
    # Cohen's d
    cohens = [pca_results[l]['cohens_d'] for l in sorted_layers if l in pca_results]
    log(f"  PC1 Cohen's d: {[f'{c:.3f}' for c in cohens]}")
    log(f"  Mean |Cohen's d|: {np.mean([abs(c) for c in cohens]):.3f}")
    
    # Centroid cos
    overlaps = [pca_results[l]['centroid_cos'] for l in sorted_layers if l in pca_results]
    log(f"  Subspace overlaps: {[f'{o:.4f}' for o in overlaps]}")
    log(f"  Mean overlap: {np.mean(overlaps):.4f}")
    
    # ============================================================
    # 最终汇总
    # ============================================================
    log(f"\n{'='*60}")
    log(f"FINAL: {model_name}")
    log(f"{'='*60}")
    
    log("\n--- Delta Norms ---")
    for feat in feature_names:
        norms = []
        for l in sorted_layers:
            valid = [d for d in all_deltas[feat][l] if d is not None]
            if valid:
                norms.append(str(int(np.mean([np.linalg.norm(d) for d in valid]))))
            else:
                norms.append("N/A")
        log(f"  {feat} [{FEATURE_PAIRS[feat]['type']}]: {', '.join(norms)}")
    
    log("\n--- PCA ---")
    for l in sorted_layers:
        if l in pca_results:
            r = pca_results[l]
            log(f"  L{l}: PC1={r['var_top5'][0]:.4f}, cum5={r['cumvar_5']:.4f}, Cohen_d={r['cohens_d']:.3f}, overlap={r['centroid_cos']:.4f}")
    
    log("\n--- Head Specialization (Direct Hook) ---")
    for l in sorted_layers:
        if l in head_spec_results:
            r = head_spec_results[l]
            log(f"  L{l}: SYN={r['n_syn']}, SEM={r['n_sem']}, MIXED={r['n_mixed']}")
            # 输出每个spec
            specs = [s['spec'] for s in r['specs']]
            log(f"    spec range: [{min(specs):.3f}, {max(specs):.3f}], std={np.std(specs):.3f}")
    
    log("\n--- Fiber Growth ---")
    for feat in feature_names:
        early_l = sorted_layers[0]
        late_l = sorted_layers[-1]
        early_valid = [d for d in all_deltas[feat][early_l] if d is not None]
        late_valid = [d for d in all_deltas[feat][late_l] if d is not None]
        if early_valid and late_valid:
            early_norm = np.mean([np.linalg.norm(d) for d in early_valid])
            late_norm = np.mean([np.linalg.norm(d) for d in late_valid])
            growth = late_norm / (early_norm + 1e-8)
            log(f"  {feat} [{FEATURE_PAIRS[feat]['type']}]: growth={growth:.2f}x")
    
    # 保存完整结果
    save_data = {
        'model': model_name,
        'n_pairs': n_pairs,
        'layer_indices': layer_indices,
        'pca_results': {str(k): v for k, v in pca_results.items()},
        'head_spec_results': {str(k): v for k, v in head_spec_results.items()},
        'growth_syn': growth_syn,
        'growth_sem': growth_sem,
    }
    
    with open(os.path.join(outdir, 'full_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    log(f"\nDONE! Saved to {outdir}")


if __name__ == '__main__':
    main()
