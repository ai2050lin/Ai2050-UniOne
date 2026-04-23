"""
Phase CCXXIII: 输出汇聚验证 — 验证1D流形是否与lm_head对齐
================================================================
核心目标:
1. 计算lm_head的SVD主方向 (前10个奇异向量)
2. 在最后一层收集per-feature PCA的PC1方向
3. 检查PC1是否与lm_head的某个主奇异方向对齐
4. 在最后一层做Interchange Intervention验证因果效应
5. 分析"输出汇聚"的层间演化 (最后3层)

关键假设:
  "输出汇聚": 最后一层的1D流形方向 = lm_head的主奇异方向
  → 如果PC1与SVD第1方向对齐 → "输出汇聚"成立
  → 如果PC1与SVD其他方向对齐 → 部分汇聚
  → 如果PC1与所有SVD方向都不对齐 → 1D流形有独立语义意义

样本量: 150对/特征 (增加样本量提高统计可靠性)
特征: tense, polarity, voice, semantic_valence, semantic_topic
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

# 复用特征对定义 (排除formality)
FEATURE_PAIRS = {
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
            ("The sun shines brightly", "The sun shone brightly"),
            ("We study English at school", "We studied English at school"),
            ("He builds model airplanes", "He built model airplanes"),
            ("The train arrives at noon", "The train arrived at noon"),
            ("The flowers bloom in spring", "The flowers bloomed in spring"),
            ("She teaches mathematics", "She taught mathematics"),
            ("The cat jumps over the fence", "The cat jumped over the fence"),
            ("He knows the answer", "He knew the answer"),
            ("The bell rings loudly", "The bell rang loudly"),
            ("I understand the problem", "I understood the problem"),
            ("The bird builds a nest", "The bird built a nest"),
            ("She wears a red dress", "She wore a red dress"),
            ("They swim in the lake", "They swam in the lake"),
            ("The clock strikes midnight", "The clock struck midnight"),
            ("He throws the ball far", "He threw the ball far"),
            ("We celebrate his birthday", "We celebrated his birthday"),
            ("The horse gallops across the field", "The horse galloped across the field"),
            ("She grows tomatoes in her garden", "She grew tomatoes in her garden"),
            ("The children sing carols", "The children sang carols"),
            ("I begin my homework early", "I began my homework early"),
            ("The phone rings constantly", "The phone rang constantly"),
            ("A cat sleeps peacefully", "A cat slept peacefully"),
            ("My sister walks her dog", "My sister walked her dog"),
            ("The boy runs very fast", "The boy ran very fast"),
            ("Some kids play hide and seek", "Some kids played hide and seek"),
            ("John reads the newspaper", "John read the newspaper"),
            ("Our dog barks at the mailman", "Our dog barked at the mailman"),
            ("Mom cooks breakfast for us", "Mom cooked breakfast for us"),
            ("Anna sings in the choir", "Anna sang in the choir"),
            ("The infant cries for milk", "The infant cried for milk"),
            ("Dad drives the kids to school", "Dad drove the kids to school"),
            ("These birds fly very high", "These birds flew very high"),
            ("Professor Smith explains the theory", "Professor Smith explained the theory"),
            ("Mary writes in her diary", "Mary wrote in her diary"),
            ("A strong wind blows from the north", "A strong wind blew from the north"),
            ("The little ones laugh happily", "The little ones laughed happily"),
            ("This river flows through the valley", "This river flowed through the valley"),
            ("Linda paints with watercolors", "Linda painted with watercolors"),
            ("The morning sun shines on the hills", "The morning sun shone on the hills"),
            ("Students study hard for exams", "Students studied hard for exams"),
            ("My brother builds sandcastles", "My brother built sandcastles"),
            ("The express train arrives on time", "The express train arrived on time"),
            ("Wild flowers bloom along the path", "Wild flowers bloomed along the path"),
            ("Ms Lee teaches us chemistry", "Ms Lee taught us chemistry"),
            ("A black cat jumps onto the wall", "A black cat jumped onto the wall"),
            ("She finally knows the truth", "She finally knew the truth"),
            ("The family visits the museum", "The family visited the museum"),
            ("The church bell rings every hour", "The church bell rang every hour"),
            ("Now I understand your point", "Now I understood your point"),
            ("The robin builds a nest in the tree", "The robin built a nest in the tree"),
            ("Emma wears her favorite scarf", "Emma wore her favorite scarf"),
            ("The teenagers swim at the beach", "The teenagers swam at the beach"),
            ("The tower clock strikes twelve", "The tower clock struck twelve"),
            ("He throws the frisbee to the dog", "He threw the frisbee to the dog"),
            ("We celebrate their anniversary", "We celebrated their anniversary"),
            ("The wild horse gallops freely", "The wild horse galloped freely"),
            ("Grandma grows roses in her yard", "Grandma grew roses in her yard"),
            ("The choir sings holiday songs", "The choir sang holiday songs"),
            ("I begin to see the pattern", "I began to see the pattern"),
            ("The alarm rings at dawn", "The alarm rang at dawn"),
            ("The kitten sleeps all day", "The kitten slept all day"),
            ("Mr Brown walks to the office", "Mr Brown walked to the office"),
            ("The rabbit runs through the garden", "The rabbit ran through the garden"),
            ("Those boys play soccer after school", "Those boys played soccer after school"),
            ("The puppy barks at the mirror", "The puppy barked at the mirror"),
            ("We bake cookies on weekends", "We baked cookies last weekend"),
            ("The soprano sings the aria", "The soprano sang the aria"),
            ("Geese fly in formation", "Geese flew in formation"),
            ("The professor explains the paradox", "The professor explained the paradox"),
            ("He writes poetry at night", "He wrote poetry last night"),
            ("A breeze blows through the window", "A breeze blew through the window"),
            ("The audience laughs at the joke", "The audience laughed at the joke"),
            ("A stream flows down the mountain", "A stream flowed down the mountain"),
            ("The artist paints landscapes", "The artist painted landscapes"),
            ("The moon shines over the lake", "The moon shone over the lake"),
            ("They study philosophy at college", "They studied philosophy at college"),
            ("The girl builds a treehouse", "The girl built a treehouse"),
            ("The soldier fights bravely", "The soldier fought bravely"),
            ("She spends money wisely", "She spent money wisely"),
            ("The fire burns brightly", "The fire burned brightly"),
            ("We eat lunch at noon", "We ate lunch at noon"),
            ("The boy catches the ball", "The boy caught the ball"),
            ("The girl draws a picture", "The girl drew a picture"),
            ("He swims across the river", "He swam across the river"),
            ("The horse jumps the fence", "The horse jumped the fence"),
            ("The chef cooks the meal", "The chef cooked the meal"),
            ("She breaks the record", "She broke the record"),
            ("The team wins the match", "The team won the match"),
            ("The girl sings a song", "The girl sang a song"),
            ("He drives the car fast", "He drove the car fast"),
            ("The wind shakes the trees", "The wind shook the trees"),
            ("She chooses the blue one", "She chose the blue one"),
            ("The bear eats the honey", "The bear ate the honey"),
            ("I feel happy today", "I felt happy yesterday"),
            ("The king rules the kingdom", "The king ruled the kingdom"),
            ("They find the treasure", "They found the treasure"),
            ("The dog protects the house", "The dog protected the house"),
            ("The girl writes a letter", "The girl wrote a letter"),
            ("We hear the music", "We heard the music"),
            ("The boy throws the stone", "The boy threw the stone"),
            ("She keeps the secret", "She kept the secret"),
            ("The sun warms the earth", "The sun warmed the earth"),
            ("He meets his friends", "He met his friends"),
            ("The woman leads the team", "The woman led the team"),
            ("We leave early tomorrow", "We left early yesterday"),
            ("The dog sits on the porch", "The dog sat on the porch"),
            ("She loses her keys", "She lost her keys"),
            ("The boy holds the kite", "The boy held the kite"),
            ("They bring the food", "They brought the food"),
            ("The cat catches the mouse", "The cat caught the mouse"),
            ("I give him the book", "I gave him the book"),
            ("The wind pushes the clouds", "The wind pushed the clouds"),
            ("She tells the truth", "She told the truth"),
            ("The man stands at the door", "The man stood at the door"),
            ("We sell the old car", "We sold the old car"),
            ("The rain falls gently", "The rain fell gently"),
            ("He thinks about the problem", "He thought about the problem"),
            ("The boy takes the bus", "The boy took the bus"),
            ("She wears a hat", "She wore a hat"),
            ("The team loses the game", "The team lost the game"),
            ("The dog hides under the bed", "The dog hid under the bed"),
            ("The bird lays eggs", "The bird laid eggs"),
            ("I wake up early", "I woke up early"),
            ("The snow melts in spring", "The snow melted in spring"),
            ("The man speaks softly", "The man spoke softly"),
            ("She sends the email", "She sent the email"),
            ("The child draws a flower", "The child drew a flower"),
            ("They drink cold water", "They drank cold water"),
            ("The boat sails across the lake", "The boat sailed across the lake"),
            ("He cuts the paper", "He cut the paper"),
            ("The girl feeds the cat", "The girl fed the cat"),
            ("We pick the apples", "We picked the apples"),
            ("The bell sounds at noon", "The bell sounded at noon"),
            ("She reads the letter", "She read the letter"),
            ("The tree grows tall", "The tree grew tall"),
            ("They hang the picture", "They hung the picture"),
            ("The light shines in the dark", "The light shone in the dark"),
            ("I mean what I say", "I meant what I said"),
            ("The dog deals with the situation", "The dog dealt with the situation"),
        ]
    },
    'polarity': {
        'type': 'SEM',
        'pairs': [
            ("The cat is happy", "The cat is not happy"),
            ("He likes the movie", "He does not like the movie"),
            ("The food is delicious", "The food is not delicious"),
            ("She can swim well", "She cannot swim well"),
            ("The test was easy", "The test was not easy"),
            ("The car works fine", "The car does not work fine"),
            ("I believe the story", "I do not believe the story"),
            ("The weather is warm", "The weather is not warm"),
            ("They enjoy the party", "They do not enjoy the party"),
            ("The room is clean", "The room is not clean"),
            ("He understands the concept", "He does not understand the concept"),
            ("The machine works properly", "The machine does not work properly"),
            ("She enjoys cooking", "She does not enjoy cooking"),
            ("He plays the guitar", "He does not play the guitar"),
            ("The water is clean", "The water is not clean"),
            ("We agree with the plan", "We do not agree with the plan"),
            ("The food tastes good", "The food does not taste good"),
            ("She remembers the date", "She does not remember the date"),
            ("He drives carefully", "He does not drive carefully"),
            ("The room is spacious", "The room is not spacious"),
            ("I trust his judgment", "I do not trust his judgment"),
            ("The book is interesting", "The book is not interesting"),
            ("She sings professionally", "She does not sing professionally"),
            ("He works on weekends", "He does not work on weekends"),
            ("The movie is scary", "The movie is not scary"),
            ("We support the cause", "We do not support the cause"),
            ("The test is easy", "The test is not easy"),
            ("She writes poetry", "She does not write poetry"),
            ("He exercises daily", "He does not exercise daily"),
            ("The garden is beautiful", "The garden is not beautiful"),
            ("I recognize that face", "I do not recognize that face"),
            ("The coffee is hot", "The coffee is not hot"),
            ("She watches television", "She does not watch television"),
            ("He reads the newspaper", "He does not read the newspaper"),
            ("The problem is simple", "The problem is not simple"),
            ("We accept the offer", "We do not accept the offer"),
            ("The dress fits perfectly", "The dress does not fit perfectly"),
            ("She loves animals", "She does not love animals"),
            ("He plays chess", "He does not play chess"),
            ("The weather is nice", "The weather is not nice"),
            ("I understand the rules", "I do not understand the rules"),
            ("The house is large", "The house is not large"),
            ("She speaks loudly", "She does not speak loudly"),
            ("He arrives early", "He does not arrive early"),
            ("The river is deep", "The river is not deep"),
            ("We enjoy the concert", "We do not enjoy the concert"),
            ("The soup is salty", "The soup is not salty"),
            ("She paints portraits", "She does not paint portraits"),
            ("He fixes computers", "He does not fix computers"),
            ("The cake is sweet", "The cake is not sweet"),
            ("I like the design", "I do not like the design"),
            ("The road is wide", "The road is not wide"),
            ("She teaches children", "She does not teach children"),
            ("He wears glasses", "He does not wear glasses"),
            ("The sky is clear", "The sky is not clear"),
            ("We need permission", "We do not need permission"),
            ("The dog is friendly", "The dog is not friendly"),
            ("She cooks Italian food", "She does not cook Italian food"),
            ("He reads novels", "He does not read novels"),
            ("The music is loud", "The music is not loud"),
            ("I trust the results", "I do not trust the results"),
            ("The room is bright", "The room is not bright"),
            ("She practices yoga", "She does not practice yoga"),
            ("He drinks coffee", "He does not drink coffee"),
            ("The water is cold", "The water is not cold"),
            ("We have evidence", "We do not have evidence"),
            ("The movie is funny", "The movie is not funny"),
            ("She speaks German", "She does not speak German"),
            ("He plays tennis", "He does not play tennis"),
            ("The building is tall", "The building is not tall"),
            ("I remember the event", "I do not remember the event"),
            ("The food is fresh", "The food is not fresh"),
            ("She enjoys hiking", "She does not enjoy hiking"),
            ("He drives a truck", "He does not drive a truck"),
            ("The lake is frozen", "The lake is not frozen"),
            ("We want to help", "We do not want to help"),
            ("The exam is hard", "The exam is not hard"),
            ("She writes code", "She does not write code"),
            ("He studies chemistry", "He does not study chemistry"),
            ("The wind blows strongly", "The wind does not blow strongly"),
            ("She practices piano", "She does not practice piano"),
            ("The river freezes in winter", "The river does not freeze in winter"),
            ("He visits the museum", "He does not visit the museum"),
            ("The bird sings sweetly", "The bird does not sing sweetly"),
            ("She sews her own clothes", "She does not sew her own clothes"),
            ("The baby sleeps peacefully", "The baby does not sleep peacefully"),
            ("He mows the lawn", "He does not mow the lawn"),
            ("The fruit is ripe", "The fruit is not ripe"),
            ("We celebrate holidays", "We do not celebrate holidays"),
            ("The fish is fresh", "The fish is not fresh"),
            ("She irons the shirts", "She does not iron the shirts"),
            ("He repairs the roof", "He does not repair the roof"),
            ("The grass is green", "The grass is not green"),
            ("I speak Japanese", "I do not speak Japanese"),
            ("The bread is fresh", "The bread is not fresh"),
            ("She knits sweaters", "She does not knit sweaters"),
            ("He brews beer at home", "He does not brew beer at home"),
            ("The wind is warm", "The wind is not warm"),
            ("We recycle paper", "We do not recycle paper"),
            ("The apple is sweet", "The apple is not sweet"),
            ("She bakes bread", "She does not bake bread"),
            ("He jogs every morning", "He does not jog every morning"),
            ("The star is bright", "The star is not bright"),
            ("I write letters", "I do not write letters"),
            ("The soup is hot", "The soup is not hot"),
            ("She paints landscapes", "She does not paint landscapes"),
            ("He climbs mountains", "He does not climb mountains"),
            ("The coffee is strong", "The coffee is not strong"),
            ("We grow vegetables", "We do not grow vegetables"),
            ("The shirt is clean", "The shirt is not clean"),
            ("She collects stamps", "She does not collect stamps"),
            ("He rides a bicycle", "He does not ride a bicycle"),
            ("The ice is thick", "The ice is not thick"),
            ("I swim laps", "I do not swim laps"),
            ("The sky is blue", "The sky is not blue"),
            ("She loves music", "She does not love music"),
            ("He works hard", "He does not work hard"),
            ("The cake is delicious", "The cake is not delicious"),
            ("We need more time", "We do not need more time"),
            ("The cat is friendly", "The cat is not friendly"),
            ("She runs fast", "She does not run fast"),
            ("He remembers the details", "He does not remember the details"),
            ("The picture is clear", "The picture is not clear"),
            ("I feel confident", "I do not feel confident"),
            ("The water is safe", "The water is not safe"),
            ("She speaks French", "She does not speak French"),
            ("The plan works", "The plan does not work"),
            ("He enjoys reading", "He does not enjoy reading"),
            ("The movie is exciting", "The movie is not exciting"),
            ("We have enough food", "We do not have enough food"),
            ("The door is open", "The door is not open"),
            ("She can drive", "She cannot drive"),
            ("The answer is correct", "The answer is not correct"),
            ("He likes vegetables", "He does not like vegetables"),
            ("The garden is large", "The garden is not large"),
            ("I trust the process", "I do not trust the process"),
            ("She sings well", "She does not sing well"),
            ("The road is safe", "The road is not safe"),
            ("He can swim", "He cannot swim"),
            ("The story is true", "The story is not true"),
            ("We like the music", "We do not like the music"),
            ("The boy is tall", "The boy is not tall"),
            ("She dances beautifully", "She does not dance beautifully"),
            ("The coffee tastes good", "The coffee does not taste good"),
            ("He can cook", "He cannot cook"),
        ]
    },
    'voice': {
        'type': 'SEM',
        'pairs': [
            ("The cat chased the mouse", "The mouse was chased by the cat"),
            ("The dog bit the man", "The man was bitten by the dog"),
            ("The wind destroyed the house", "The house was destroyed by the wind"),
            ("The chef prepared the meal", "The meal was prepared by the chef"),
            ("The teacher punished the student", "The student was punished by the teacher"),
            ("The fire damaged the building", "The building was damaged by the fire"),
            ("The company developed the product", "The product was developed by the company"),
            ("The artist created the sculpture", "The sculpture was created by the artist"),
            ("The scientist discovered the cure", "The cure was discovered by the scientist"),
            ("The author published the book", "The book was published by the author"),
            ("The player scored the goal", "The goal was scored by the player"),
            ("The singer performed the song", "The song was performed by the singer"),
            ("The police arrested the thief", "The thief was arrested by the police"),
            ("The engineer designed the bridge", "The bridge was designed by the engineer"),
            ("The farmer grew the crops", "The crops were grown by the farmer"),
            ("The doctor treated the patient", "The patient was treated by the doctor"),
            ("The worker built the wall", "The wall was built by the worker"),
            ("The driver delivered the package", "The package was delivered by the driver"),
            ("The manager approved the request", "The request was approved by the manager"),
            ("The committee selected the winner", "The winner was selected by the committee"),
            ("The baker made the bread", "The bread was made by the baker"),
            ("The mechanic fixed the car", "The car was fixed by the mechanic"),
            ("The cleaner washed the floor", "The floor was washed by the cleaner"),
            ("The judge sentenced the criminal", "The criminal was sentenced by the judge"),
            ("The photographer took the picture", "The picture was taken by the photographer"),
            ("The librarian organized the books", "The books were organized by the librarian"),
            ("The nurse cared for the patient", "The patient was cared for by the nurse"),
            ("The pilot flew the plane", "The plane was flown by the pilot"),
            ("The plumber repaired the pipe", "The pipe was repaired by the plumber"),
            ("The student wrote the essay", "The essay was written by the student"),
            ("The tailor made the suit", "The suit was made by the tailor"),
            ("The waiter served the food", "The food was served by the waiter"),
            ("The painter painted the house", "The house was painted by the painter"),
            ("The director filmed the movie", "The movie was filmed by the director"),
            ("The gardener planted the tree", "The tree was planted by the gardener"),
            ("The chef cooked the dinner", "The dinner was cooked by the chef"),
            ("The editor revised the article", "The article was revised by the editor"),
            ("The architect designed the building", "The building was designed by the architect"),
            ("The coach trained the team", "The team was trained by the coach"),
            ("The scientist conducted the experiment", "The experiment was conducted by the scientist"),
            ("The writer composed the poem", "The poem was composed by the writer"),
            ("The programmer developed the software", "The software was developed by the programmer"),
            ("The musician played the symphony", "The symphony was played by the musician"),
            ("The king ruled the kingdom", "The kingdom was ruled by the king"),
            ("The general commanded the army", "The army was commanded by the general"),
            ("The teacher explained the theory", "The theory was explained by the teacher"),
            ("The mother hugged the child", "The child was hugged by the mother"),
            ("The boss fired the employee", "The employee was fired by the boss"),
            ("The cat caught the fish", "The fish was caught by the cat"),
            ("The dog chased the rabbit", "The rabbit was chased by the dog"),
            ("The wind blew the leaves", "The leaves were blown by the wind"),
            ("The rain flooded the street", "The street was flooded by the rain"),
            ("The sun melted the ice", "The ice was melted by the sun"),
            ("The fire burned the forest", "The forest was burned by the fire"),
            ("The wave hit the shore", "The shore was hit by the wave"),
            ("The storm destroyed the village", "The village was destroyed by the storm"),
            ("The earthquake shook the city", "The city was shaken by the earthquake"),
            ("The flood washed away the bridge", "The bridge was washed away by the flood"),
            ("The lightning struck the tree", "The tree was struck by the lightning"),
            ("The tornado lifted the car", "The car was lifted by the tornado"),
            ("The volcano buried the town", "The town was buried by the volcano"),
            ("The tsunami hit the coast", "The coast was hit by the tsunami"),
            ("The hurricane damaged the roof", "The roof was damaged by the hurricane"),
            ("The avalanche buried the skier", "The skier was buried by the avalanche"),
            ("The glacier carved the valley", "The valley was carved by the glacier"),
            ("The river eroded the bank", "The bank was eroded by the river"),
            ("The tide washed the sand", "The sand was washed by the tide"),
            ("The frost killed the flowers", "The flowers were killed by the frost"),
            ("The hail damaged the crops", "The crops were damaged by the hail"),
            ("The drought dried the lake", "The lake was dried by the drought"),
            ("The mist covered the mountain", "The mountain was covered by the mist"),
            ("The snow blocked the road", "The road was blocked by the snow"),
            ("The ice covered the lake", "The lake was covered by the ice"),
            ("The fog delayed the flight", "The flight was delayed by the fog"),
            ("The heat melted the chocolate", "The chocolate was melted by the heat"),
            ("The cold froze the pipes", "The pipes were frozen by the cold"),
            ("The rain soaked the clothes", "The clothes were soaked by the rain"),
            ("The wind toppled the tree", "The tree was toppled by the wind"),
            ("The sun warmed the room", "The room was warmed by the sun"),
            ("The moon lit the path", "The path was lit by the moon"),
            ("The stars guided the sailors", "The sailors were guided by the stars"),
            ("The dog guards the house", "The house is guarded by the dog"),
            ("The boy opens the door", "The door is opened by the boy"),
            ("The girl reads the book", "The book is read by the girl"),
            ("The man drives the truck", "The truck is driven by the man"),
            ("The woman paints the wall", "The wall is painted by the woman"),
            ("The child throws the ball", "The ball is thrown by the child"),
            ("The teacher grades the test", "The test is graded by the teacher"),
            ("The doctor examines the patient", "The patient is examined by the doctor"),
            ("The chef prepares the sauce", "The sauce is prepared by the chef"),
            ("The musician plays the piano", "The piano is played by the musician"),
            ("The worker paints the fence", "The fence is painted by the worker"),
            ("The girl feeds the birds", "The birds are fed by the girl"),
            ("The boy kicks the ball", "The ball is kicked by the boy"),
            ("The man lifts the box", "The box is lifted by the man"),
            ("The woman carries the bag", "The bag is carried by the woman"),
            ("The student solves the problem", "The problem is solved by the student"),
            ("The artist draws the portrait", "The portrait is drawn by the artist"),
            ("The writer tells the story", "The story is told by the writer"),
            ("The singer sings the song", "The song is sung by the singer"),
            ("The builder constructs the house", "The house is constructed by the builder"),
            ("The designer creates the logo", "The logo is created by the designer"),
            ("The engineer builds the robot", "The robot is built by the engineer"),
            ("The farmer plants the seeds", "The seeds are planted by the farmer"),
            ("The baker bakes the cake", "The cake is baked by the baker"),
            ("The driver delivers the mail", "The mail is delivered by the driver"),
            ("The clerk sells the tickets", "The tickets are sold by the clerk"),
            ("The guard watches the gate", "The gate is watched by the guard"),
            ("The host welcomes the guests", "The guests are welcomed by the host"),
            ("The judge hears the case", "The case is heard by the judge"),
            ("The mayor leads the city", "The city is led by the mayor"),
            ("The nurse treats the wound", "The wound is treated by the nurse"),
            ("The officer issues the permit", "The permit is issued by the officer"),
            ("The priest blesses the people", "The people are blessed by the priest"),
            ("The queen rules the nation", "The nation is ruled by the queen"),
            ("The referee calls the foul", "The foul is called by the referee"),
            ("The secretary types the letter", "The letter is typed by the secretary"),
            ("The soldier defends the fort", "The fort is defended by the soldier"),
            ("The teacher guides the class", "The class is guided by the teacher"),
            ("The vet treats the animal", "The animal is treated by the vet"),
            ("The chef seasons the soup", "The soup is seasoned by the chef"),
            ("The maid cleans the room", "The room is cleaned by the maid"),
            ("The pilot steers the plane", "The plane is steered by the pilot"),
            ("The poet writes the verse", "The verse is written by the poet"),
            ("The boss manages the team", "The team is managed by the boss"),
            ("The child breaks the toy", "The toy is broken by the child"),
            ("The dog fetches the stick", "The stick is fetched by the dog"),
            ("The girl braids her hair", "Her hair is braided by the girl"),
            ("The boy builds the model", "The model is built by the boy"),
            ("The man chops the wood", "The wood is chopped by the man"),
            ("The woman knits the scarf", "The scarf is knitted by the woman"),
            ("The student reads the chapter", "The chapter is read by the student"),
            ("The artist sculpts the statue", "The statue is sculpted by the artist"),
            ("The writer drafts the essay", "The essay is drafted by the writer"),
            ("The doctor cures the illness", "The illness is cured by the doctor"),
            ("The scientist proves the theory", "The theory is proved by the scientist"),
            ("The programmer codes the app", "The app is coded by the programmer"),
            ("The coach leads the practice", "The practice is led by the coach"),
            ("The mechanic tunes the engine", "The engine is tuned by the mechanic"),
            ("The librarian lends the book", "The book is lent by the librarian"),
            ("The baker decorates the cake", "The cake is decorated by the baker"),
            ("The painter frames the picture", "The picture is framed by the painter"),
            ("The singer records the album", "The album is recorded by the singer"),
            ("The editor publishes the article", "The article is published by the editor"),
            ("The farmer harvests the wheat", "The wheat is harvested by the farmer"),
            ("The builder installs the window", "The window is installed by the builder"),
            ("The clerk processes the order", "The order is processed by the clerk"),
            ("The driver transports the goods", "The goods are transported by the driver"),
            ("The teacher corrects the paper", "The paper is corrected by the teacher"),
        ]
    },
    'semantic_valence': {
        'type': 'SEM',
        'pairs': [
            ("The brave soldier fought", "The cruel soldier fought"),
            ("She gave a generous gift", "She gave a stingy gift"),
            ("The kind teacher helped", "The mean teacher helped"),
            ("He told the honest truth", "He told the deceptive truth"),
            ("The warm fire glowed", "The cold fire glowed"),
            ("She showed genuine love", "She showed fake love"),
            ("The bright sun rose", "The dark sun rose"),
            ("He felt pure joy", "He felt deep sorrow"),
            ("The gentle rain fell", "The violent rain fell"),
            ("She spoke with wisdom", "She spoke with foolishness"),
            ("The sweet fruit ripened", "The bitter fruit ripened"),
            ("He won a glorious victory", "He won a shameful victory"),
            ("The clean water flowed", "The dirty water flowed"),
            ("She gave a warm smile", "She gave a cold smile"),
            ("The peaceful village slept", "The violent village slept"),
            ("He made a noble sacrifice", "He made a cowardly sacrifice"),
            ("The beautiful garden bloomed", "The ugly garden bloomed"),
            ("She felt proud confidence", "She felt shameful doubt"),
            ("The strong hero stood", "The weak hero stood"),
            ("He showed deep compassion", "He showed cruel indifference"),
            ("The pleasant music played", "The harsh music played"),
            ("She received a warm welcome", "She received a cold welcome"),
            ("The fresh air blew", "The stale air blew"),
            ("He made a wise decision", "He made a foolish decision"),
            ("The cheerful child laughed", "The gloomy child laughed"),
            ("She had a positive attitude", "She had a negative attitude"),
            ("The safe neighborhood was quiet", "The dangerous neighborhood was quiet"),
            ("He gave a sincere apology", "He gave a fake apology"),
            ("The delicious meal was served", "The disgusting meal was served"),
            ("She experienced true happiness", "She experienced deep misery"),
            ("The loyal dog stayed", "The treacherous dog stayed"),
            ("He earned an honest living", "He earned a dishonest living"),
            ("The comfortable bed was soft", "The uncomfortable bed was soft"),
            ("She expressed genuine gratitude", "She expressed fake gratitude"),
            ("The healthy plant grew", "The sickly plant grew"),
            ("He received a fair reward", "He received an unfair reward"),
            ("The calm sea was still", "The rough sea was still"),
            ("She told a fascinating story", "She told a boring story"),
            ("The elegant dress shimmered", "The shabby dress shimmered"),
            ("He showed great courage", "He showed great cowardice"),
            ("The loving mother cared", "The hateful mother cared"),
            ("She made a generous donation", "She made a selfish donation"),
            ("The virtuous leader ruled", "The corrupt leader ruled"),
            ("He performed a heroic deed", "He performed a villainous deed"),
            ("The fragrant flower bloomed", "The foul flower bloomed"),
            ("She offered sincere praise", "She offered sarcastic praise"),
            ("The harmonious choir sang", "The discordant choir sang"),
            ("He had a brilliant idea", "He had a stupid idea"),
            ("The prosperous city thrived", "The impoverished city thrived"),
            ("She felt immense relief", "She felt immense anxiety"),
            ("The righteous judge decided", "The biased judge decided"),
            ("He showed tender affection", "He showed brutal hostility"),
            ("The splendid palace gleamed", "The dismal palace gleamed"),
            ("She gave an enthusiastic speech", "She gave an apathetic speech"),
            ("The benevolent king ruled", "The malevolent king ruled"),
            ("He made a prudent choice", "He made a reckless choice"),
            ("The vibrant painting glowed", "The dull painting glowed"),
            ("She felt serene peace", "She felt restless turmoil"),
            ("The wholesome food nourished", "The toxic food nourished"),
            ("He demonstrated steadfast loyalty", "He demonstrated fickle loyalty"),
            ("The magnificent cathedral stood", "The humble cathedral stood"),
            ("She displayed remarkable patience", "She displayed extreme impatience"),
            ("The graceful dancer moved", "The clumsy dancer moved"),
            ("He shared a profound insight", "He shared a trivial insight"),
            ("The charming village attracted", "The dreary village attracted"),
            ("She maintained quiet dignity", "She maintained loud arrogance"),
            ("The soothing melody calmed", "The jarring melody calmed"),
            ("He possessed extraordinary talent", "He possessed mediocre talent"),
            ("The pristine lake reflected", "The polluted lake reflected"),
            ("She exuded quiet confidence", "She exuded nervous insecurity"),
            ("The luxurious hotel impressed", "The shoddy hotel impressed"),
            ("He exhibited genuine kindness", "He exhibited fake kindness"),
            ("The cheerful melody uplifted", "The melancholy melody uplifted"),
            ("She demonstrated clear wisdom", "She demonstrated foolish ignorance"),
            ("The noble knight fought", "The wicked knight fought"),
            ("He expressed heartfelt thanks", "He expressed insincere thanks"),
            ("The vibrant garden flourished", "The withered garden flourished"),
            ("She maintained calm composure", "She maintained wild panic"),
            ("The gentle breeze whispered", "The fierce gale whispered"),
            ("He showed unwavering faith", "He showed constant doubt"),
            ("The radiant sunrise appeared", "The gloomy sunrise appeared"),
            ("She offered genuine friendship", "She offered false friendship"),
            ("The delicious aroma filled", "The nauseating aroma filled"),
            ("He spoke with quiet authority", "He spoke with blustering weakness"),
            ("The crystal stream sparkled", "The muddy stream sparkled"),
            ("She displayed elegant simplicity", "She displayed tacky excess"),
            ("The warm embrace comforted", "The cold embrace comforted"),
            ("He made a selfless choice", "He made a selfish choice"),
            ("The beautiful sunset faded", "The bleak sunset faded"),
            ("She felt deep contentment", "She felt bitter resentment"),
            ("The precious gem sparkled", "The worthless gem sparkled"),
            ("He gave a thoughtful gift", "He gave a thoughtless gift"),
            ("The sweet melody lingered", "The harsh melody lingered"),
            ("She showed quiet strength", "She showed loud weakness"),
            ("The clean sheet was crisp", "The dirty sheet was crisp"),
            ("He earned deep respect", "He earned utter contempt"),
            ("The pleasant scent drifted", "The unpleasant scent drifted"),
            ("She maintained graceful poise", "She maintained awkward clumsiness"),
            ("The bright morning dawned", "The dark morning dawned"),
            ("He felt warm affection", "He felt cold hatred"),
            ("The fresh bread smelled", "The stale bread smelled"),
            ("She gave a heartfelt compliment", "She gave a backhanded compliment"),
            ("The sharp knife cut", "The dull knife cut"),
            ("He showed polite respect", "He showed rude disrespect"),
            ("The smooth road stretched", "The bumpy road stretched"),
            ("She felt boundless energy", "She felt crushing fatigue"),
            ("The rich soil produced", "The barren soil produced"),
            ("He demonstrated quiet competence", "He demonstrated loud incompetence"),
            ("The sweet honey dripped", "The sour honey dripped"),
            ("She shared hopeful optimism", "She shared hopeless pessimism"),
            ("The warm blanket covered", "The thin blanket covered"),
            ("He earned well-deserved praise", "He earned unjustified criticism"),
            ("The clear sky appeared", "The cloudy sky appeared"),
            ("She showed natural grace", "She showed forced awkwardness"),
            ("The soft pillow supported", "The hard pillow supported"),
            ("He made a meaningful contribution", "He made a meaningless contribution"),
            ("The bright light shone", "The dim light shone"),
            ("She expressed genuine concern", "She expressed fake concern"),
            ("The rich flavor delighted", "The bland flavor delighted"),
            ("He displayed quiet courage", "He displayed noisy fear"),
            ("The warm hearth glowed", "The cold hearth glowed"),
            ("She felt pure bliss", "She felt pure agony"),
            ("The lively party continued", "The dull party continued"),
            ("He gave an honest answer", "He gave a dishonest answer"),
            ("The fresh paint gleamed", "The peeling paint gleamed"),
            ("She maintained steadfast resolve", "She maintained wavering uncertainty"),
            ("The bright future awaited", "The bleak future awaited"),
            ("He showed genuine remorse", "He showed fake remorse"),
            ("The beautiful melody played", "The ugly melody played"),
            ("She gave a warm hug", "She gave a cold shrug"),
            ("The safe harbor protected", "The dangerous harbor protected"),
            ("He felt sincere gratitude", "He felt bitter envy"),
            ("The pleasant evening passed", "The unpleasant evening passed"),
            ("She demonstrated natural talent", "She demonstrated learned incompetence"),
            ("The comfortable chair welcomed", "The uncomfortable chair welcomed"),
            ("He made a wise investment", "He made a foolish investment"),
            ("The sweet victory tasted", "The bitter defeat tasted"),
            ("She showed deep empathy", "She showed cold apathy"),
            ("The bright stars twinkled", "The dim stars twinkled"),
            ("He experienced genuine surprise", "He experienced fake surprise"),
            ("The clean house impressed", "The messy house impressed"),
            ("She felt secure confidence", "She felt anxious worry"),
            ("The gentle slope descended", "The steep slope descended"),
            ("He spoke with clear logic", "He spoke with confused illogic"),
        ]
    },
    'semantic_topic': {
        'type': 'SEM',
        'pairs': [
            ("The doctor treated the patient", "The chef treated the customer"),
            ("The student read the textbook", "The student read the novel"),
            ("The scientist ran the experiment", "The athlete ran the marathon"),
            ("The pilot flew the airplane", "The driver drove the car"),
            ("The farmer planted the seeds", "The builder laid the bricks"),
            ("The judge made the ruling", "The chef made the sauce"),
            ("The nurse gave the injection", "The teacher gave the lecture"),
            ("The engineer built the bridge", "The painter painted the portrait"),
            ("The lawyer argued the case", "The comedian told the joke"),
            ("The soldier fired the weapon", "The photographer fired the flash"),
            ("The baker baked the bread", "The potter shaped the clay"),
            ("The mechanic fixed the engine", "The doctor fixed the bone"),
            ("The fisherman caught the fish", "The police caught the thief"),
            ("The author wrote the book", "The programmer wrote the code"),
            ("The singer sang the aria", "The preacher sang the hymn"),
            ("The architect drew the blueprint", "The artist drew the landscape"),
            ("The banker counted the money", "The referee counted the points"),
            ("The teacher graded the paper", "The critic graded the restaurant"),
            ("The surgeon operated on the heart", "The mechanic operated the machine"),
            ("The plumber fixed the pipe", "The therapist fixed the relationship"),
            ("The carpenter built the table", "The composer built the symphony"),
            ("The driver delivered the package", "The speaker delivered the speech"),
            ("The miner dug the tunnel", "The detective dug the clues"),
            ("The electrician wired the house", "The journalist wired the report"),
            ("The chemist mixed the solution", "The DJ mixed the track"),
            ("The tailor sewed the suit", "The writer sewed the plot"),
            ("The guard protected the vault", "The sunscreen protected the skin"),
            ("The coach trained the athlete", "The teacher trained the student"),
            ("The chef seasoned the dish", "The comedian seasoned the joke"),
            ("The artist painted the canvas", "The politician painted the picture"),
            ("The doctor prescribed the medicine", "The judge prescribed the sentence"),
            ("The professor lectured on physics", "The priest lectured on theology"),
            ("The banker managed the portfolio", "The coach managed the team"),
            ("The programmer debugged the code", "The doctor debugged the diagnosis"),
            ("The architect designed the museum", "The fashion designer designed the dress"),
            ("The scientist discovered the element", "The explorer discovered the island"),
            ("The engineer calculated the stress", "The accountant calculated the tax"),
            ("The librarian cataloged the books", "The astronomer cataloged the stars"),
            ("The chef prepared the feast", "The general prepared the battle"),
            ("The musician tuned the violin", "The mechanic tuned the engine"),
            ("The dentist filled the cavity", "The construction worker filled the trench"),
            ("The pharmacist dispensed the drug", "The banker dispensed the cash"),
            ("The botanist studied the flower", "The historian studied the war"),
            ("The astronaut explored the planet", "The detective explored the crime scene"),
            ("The psychologist analyzed the dream", "The critic analyzed the film"),
            ("The surgeon removed the tumor", "The editor removed the paragraph"),
            ("The pilot navigated the storm", "The manager navigated the crisis"),
            ("The farmer harvested the wheat", "The company harvested the data"),
            ("The jeweler polished the diamond", "The student polished the essay"),
            ("The meteorologist predicted the weather", "The economist predicted the market"),
            ("The carpenter measured the wood", "The scientist measured the radiation"),
            ("The baker kneaded the dough", "The politician kneaded the crowd"),
            ("The fireman extinguished the fire", "The auditor extinguished the fraud"),
            ("The translator converted the text", "The chemist converted the compound"),
            ("The waiter served the meal", "The soldier served the country"),
            ("The cashier handled the money", "The diplomat handled the negotiation"),
            ("The conductor led the orchestra", "The mayor led the city"),
            ("The diver explored the wreck", "The researcher explored the hypothesis"),
            ("The sculptor carved the marble", "The chef carved the turkey"),
            ("The tailor measured the fabric", "The surveyor measured the land"),
            ("The nurse monitored the patient", "The security guard monitored the building"),
            ("The painter mixed the colors", "The chemist mixed the reagents"),
            ("The editor revised the manuscript", "The surgeon revised the procedure"),
            ("The coach motivated the player", "The teacher motivated the student"),
            ("The driver navigated the road", "The captain navigated the ship"),
            ("The programmer compiled the code", "The author compiled the anthology"),
            ("The electrician installed the wiring", "The curator installed the exhibit"),
            ("The lawyer filed the motion", "The accountant filed the return"),
            ("The pharmacist compounded the prescription", "The musician compounded the harmony"),
            ("The therapist counseled the patient", "The consultant counseled the business"),
            ("The photographer captured the image", "The historian captured the event"),
            ("The reporter covered the story", "The blanket covered the bed"),
            ("The governor governed the state", "The referee governed the match"),
            ("The surgeon sutured the wound", "The tailor sutured the seam"),
            ("The broker traded the stock", "The merchant traded the goods"),
            ("The inspector inspected the factory", "The critic inspected the restaurant"),
            ("The mechanic overhauled the engine", "The doctor overhauled the treatment"),
            ("The director directed the film", "The principal directed the school"),
            ("The analyst analyzed the data", "The critic analyzed the performance"),
            ("The manager managed the project", "The conductor managed the orchestra"),
            ("The engineer optimized the design", "The athlete optimized the routine"),
            ("The researcher published the paper", "The author published the novel"),
            ("The doctor diagnosed the disease", "The mechanic diagnosed the problem"),
            ("The professor taught the class", "The trainer taught the exercise"),
            ("The chef cooked the meal", "The chemist cooked the compound"),
            ("The judge delivered the verdict", "The mailman delivered the letter"),
            ("The programmer coded the algorithm", "The lawyer coded the contract"),
            ("The scientist tested the hypothesis", "The athlete tested the equipment"),
            ("The designer created the layout", "The composer created the melody"),
            ("The builder constructed the house", "The writer constructed the argument"),
            ("The investor invested the capital", "The teacher invested the time"),
            ("The banker financed the project", "The director financed the film"),
            ("The doctor healed the wound", "Time healed the pain"),
            ("The student learned the lesson", "The monk learned the teaching"),
            ("The engineer solved the equation", "The detective solved the case"),
            ("The artist expressed the emotion", "The scientist expressed the theory"),
            ("The banker approved the loan", "The manager approved the request"),
            ("The researcher observed the phenomenon", "The tourist observed the landmark"),
            ("The editor published the article", "The developer published the app"),
            ("The pilot landed the plane", "The fisherman landed the fish"),
            ("The doctor examined the patient", "The auditor examined the records"),
            ("The teacher explained the concept", "The guide explained the route"),
            ("The scientist wrote the report", "The journalist wrote the article"),
            ("The manager organized the event", "The librarian organized the collection"),
            ("The programmer developed the app", "The photographer developed the film"),
            ("The architect planned the building", "The general planned the campaign"),
            ("The chef created the recipe", "The artist created the masterpiece"),
            ("The engineer designed the system", "The fashion designer designed the collection"),
            ("The doctor treated the disease", "The teacher treated the student"),
            ("The worker operated the machine", "The surgeon operated the patient"),
            ("The author published the story", "The scientist published the findings"),
            ("The coach trained the champion", "The teacher trained the scholar"),
            ("The musician performed the concerto", "The actor performed the monologue"),
            ("The painter finished the mural", "The author finished the manuscript"),
            ("The scientist researched the topic", "The historian researched the period"),
            ("The builder constructed the tower", "The writer constructed the narrative"),
            ("The dentist cleaned the teeth", "The cleaner cleaned the floor"),
            ("The florist arranged the flowers", "The director arranged the scene"),
            ("The programmer debugged the software", "The doctor debugged the illness"),
            ("The editor corrected the errors", "The teacher corrected the homework"),
            ("The manager supervised the staff", "The officer supervised the patrol"),
            ("The chef seasoned the soup", "The comedian seasoned the routine"),
            ("The artist sketched the portrait", "The architect sketched the plan"),
            ("The doctor prescribed the treatment", "The judge prescribed the penalty"),
            ("The researcher conducted the study", "The conductor conducted the symphony"),
            ("The engineer calculated the load", "The accountant calculated the budget"),
            ("The lawyer presented the evidence", "The artist presented the exhibition"),
            ("The teacher assessed the student", "The critic assessed the performance"),
            ("The scientist measured the effect", "The surveyor measured the distance"),
            ("The builder laid the foundation", "The writer laid the groundwork"),
            ("The doctor administered the vaccine", "The teacher administered the test"),
            ("The programmer implemented the feature", "The government implemented the policy"),
            ("The chef garnished the plate", "The decorator garnished the room"),
            ("The musician rehearsed the piece", "The actor rehearsed the scene"),
            ("The editor formatted the document", "The designer formatted the layout"),
            ("The architect reviewed the plans", "The manager reviewed the proposal"),
            ("The scientist validated the model", "The accountant validated the books"),
            ("The teacher instructed the class", "The coach instructed the team"),
            ("The engineer maintained the system", "The nurse maintained the equipment"),
            ("The chef adjusted the seasoning", "The musician adjusted the tuning"),
            ("The artist chose the colors", "The designer chose the materials"),
            ("The doctor monitored the recovery", "The scientist monitored the experiment"),
            ("The teacher evaluated the progress", "The manager evaluated the performance"),
        ]
    },
}

feature_names = list(FEATURE_PAIRS.keys())

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
    parser.add_argument('--n_pairs', type=int, default=150)
    args = parser.parse_args()
    
    model_key = args.model
    n_pairs = args.n_pairs
    config = MODEL_CONFIGS[model_key]
    
    # 输出目录
    out_dir = f'results/causal_fiber/{model_key}_ccxxiii'
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
    log(f"Phase CCXXIII: 输出汇聚验证 — {config['name']}")
    log(f"  n_pairs={n_pairs}, 时间={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"=" * 70)
    
    # ============================================================
    # 加载模型
    # ============================================================
    log(f"\n--- 加载模型 ---")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sklearn.decomposition import PCA
    
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
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    log(f"  加载完成: {time.time()-t0:.0f}s")
    log(f"  n_layers={n_layers}, d_model={d_model}, device={device}")
    
    last_layer = n_layers - 1
    # 分析最后3层 + 1个中间层做对比
    analysis_layers = [last_layer - 2, last_layer - 1, last_layer]
    if n_layers >= 10:
        mid_layer = n_layers // 2
        analysis_layers.append(mid_layer)
    analysis_layers = sorted(set(analysis_layers))
    log(f"  分析层: {analysis_layers}")
    
    # ============================================================
    # S1: 提取lm_head权重矩阵并做SVD
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S1: 提取lm_head权重矩阵并做SVD")
    log(f"{'='*60}")
    
    # 找到lm_head (不同模型名字不同)
    lm_head = None
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
        lm_head = model.model.lm_head
    elif hasattr(model, 'output'):
        lm_head = model.output
    
    if lm_head is None:
        # 搜索
        for name, module in model.named_modules():
            if 'lm_head' in name or 'embed_out' in name:
                lm_head = module
                log(f"  找到lm_head: {name}")
                break
    
    if lm_head is None:
        log("  !!! 无法找到lm_head, 尝试get_output_embeddings")
        lm_head = model.get_output_embeddings()
    
    if lm_head is not None:
        try:
            # 分步提取权重，避免内存峰值
            log(f"  正在提取lm_head权重...")
            w = lm_head.weight.detach()
            # 先移到CPU，再转float32
            w = w.cpu()
            W = w.float().numpy()
            del w
            log(f"  lm_head权重形状: {W.shape}")
            
            # 使用截断SVD (只计算前20个分量, 节省内存)
            from sklearn.decomposition import TruncatedSVD
            n_svd_comp = 20
            t_svd = time.time()
            svd_model = TruncatedSVD(n_components=n_svd_comp)
            svd_model.fit(W)  # W: (vocab, d_model)
            S = svd_model.singular_values_
            Vt = svd_model.components_  # (n_svd_comp, d_model)
            del W, svd_model  # 释放内存
            log(f"  截断SVD完成: {time.time()-t_svd:.0f}s, S={S.shape}, Vt={Vt.shape}")
            
            # 前20个奇异值 (方差比例用S^2近似)
            total_energy = np.sum(S**2) * (d_model / n_svd_comp)  # 近似总能量
            log(f"  前20个奇异值:")
            cum_var = 0
            for i in range(min(20, len(S))):
                cum_var += S[i]**2
                log(f"    S[{i}] = {S[i]:.1f}")
            log(f"  前20分量累积能量: {np.sum(S**2):.0f}")
            
            # 保存前20个右奇异向量 (d_model维)
            n_svd = 20
            svd_directions = Vt[:n_svd]  # (20, d_model)
            svd_singular = S[:n_svd]
            log(f"  保存前{n_svd}个SVD主方向")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log(f"  !!! lm_head权重提取/SVD失败: {e}")
            svd_directions = None
            svd_singular = None
    else:
        log("  !!! 无法找到lm_head! 跳过SVD分析")
        svd_directions = None
    
    # ============================================================
    # S2: 在最后3层+中间层收集差分向量并做PCA
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S2: 收集差分向量并做PCA ({n_pairs}对/特征 × {len(analysis_layers)}层)")
    log(f"{'='*60}")
    
    layer_deltas = {l: {f: [] for f in feature_names} for l in analysis_layers}
    layer_pca = {}  # layer -> feat -> {pc1_dir, pc1_var, explained_var}
    
    def make_hook(layer_idx, store):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            store[layer_idx] = out.detach().cpu().clone()
        return hook_fn
    
    cap_store = {}
    handles = []
    for layer in analysis_layers:
        h = model.model.layers[layer].register_forward_hook(make_hook(layer, cap_store))
        handles.append(h)
    
    for feat in feature_names:
        all_pairs = FEATURE_PAIRS[feat]['pairs'][:n_pairs]
        log(f"  收集 {feat} ({len(all_pairs)}对)...")
        
        for i, (s1, s2) in enumerate(all_pairs):
            if (i+1) % 30 == 0:
                log(f"    {feat}: {i+1}/{len(all_pairs)}")
            
            with torch.no_grad():
                # s1
                tokens1 = tokenizer(s1, return_tensors='pt').to(device)
                out1 = model(**tokens1)
                
                act1_by_layer = {}
                for layer in analysis_layers:
                    if layer in cap_store:
                        act1_by_layer[layer] = cap_store[layer][0, -1].float().numpy()
                
                # s2
                tokens2 = tokenizer(s2, return_tensors='pt').to(device)
                out2 = model(**tokens2)
                
                act2_by_layer = {}
                for layer in analysis_layers:
                    if layer in cap_store:
                        act2_by_layer[layer] = cap_store[layer][0, -1].float().numpy()
            
            for layer in analysis_layers:
                if layer in act1_by_layer and layer in act2_by_layer:
                    delta = act2_by_layer[layer] - act1_by_layer[layer]
                    layer_deltas[layer][feat].append(delta)
            
            del out1, out2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    for h in handles:
        h.remove()
    
    log(f"\n  PCA计算...")
    for layer in analysis_layers:
        layer_pca[layer] = {}
        for feat in feature_names:
            deltas = np.array(layer_deltas[layer][feat])
            if len(deltas) < 5:
                continue
            pca = PCA(n_components=min(5, len(deltas)))
            pca.fit(deltas)
            layer_pca[layer][feat] = {
                'pc1_dir': pca.components_[0],  # PC1方向
                'pc1_var': pca.explained_variance_ratio_[0],
                'explained_var': pca.explained_variance_ratio_[:5],
                'n_samples': len(deltas),
            }
            log(f"  L{layer} {feat}: PC1方差={pca.explained_variance_ratio_[0]*100:.1f}%, "
                f"PC1-5={np.sum(pca.explained_variance_ratio_[:5])*100:.1f}%")
    
    # ============================================================
    # S3: PC1方向 vs lm_head SVD主方向的对齐度
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S3: PC1方向 vs lm_head SVD主方向的对齐度")
    log(f"{'='*60}")
    
    if svd_directions is not None:
        for layer in analysis_layers:
            log(f"\n  --- Layer {layer} ---")
            for feat in feature_names:
                if feat not in layer_pca[layer]:
                    continue
                pc1 = layer_pca[layer][feat]['pc1_dir']
                
                # 计算与每个SVD主方向的cosine对齐度
                cosines = []
                for k in range(n_svd):
                    cos_val = abs(np.dot(pc1, svd_directions[k]) / 
                                (np.linalg.norm(pc1) * np.linalg.norm(svd_directions[k])))
                    cosines.append(cos_val)
                
                # 找最大对齐的SVD方向
                best_k = np.argmax(cosines)
                best_cos = cosines[best_k]
                
                # 计算PC1在前K个SVD方向上的投影比例
                proj_top1 = cosines[0]**2
                proj_top3 = sum(c**2 for c in cosines[:3])
                proj_top5 = sum(c**2 for c in cosines[:5])
                proj_top10 = sum(c**2 for c in cosines[:10])
                proj_top20 = sum(c**2 for c in cosines[:20])
                
                log(f"  {feat}: PC1方差={layer_pca[layer][feat]['pc1_var']*100:.1f}%")
                log(f"    最对齐SVD方向: SVD[{best_k}] (cos={best_cos:.4f})")
                log(f"    SVD[0] cos={cosines[0]:.4f}, SVD[1] cos={cosines[1]:.4f}, "
                    f"SVD[2] cos={cosines[2]:.4f}")
                log(f"    投影比例: top1={proj_top1:.4f}, top3={proj_top3:.4f}, "
                    f"top5={proj_top5:.4f}, top10={proj_top10:.4f}, top20={proj_top20:.4f}")
        
        # 跨特征PC1对齐度 (确认CCXXII的发现)
        log(f"\n  --- 跨特征PC1对齐度 ---")
        for layer in analysis_layers:
            log(f"  Layer {layer}:")
            pc1_dirs = {}
            for feat in feature_names:
                if feat in layer_pca[layer]:
                    pc1_dirs[feat] = layer_pca[layer][feat]['pc1_dir']
            
            for f1 in feature_names:
                for f2 in feature_names:
                    if f1 >= f2:
                        continue
                    if f1 not in pc1_dirs or f2 not in pc1_dirs:
                        continue
                    cos_val = abs(np.dot(pc1_dirs[f1], pc1_dirs[f2]) / 
                             (np.linalg.norm(pc1_dirs[f1]) * np.linalg.norm(pc1_dirs[f2]) + 1e-10))
                    star = "★" if cos_val > 0.8 else ""
                    log(f"    {f1}↔{f2}: cos={cos_val:.4f} {star}")
    else:
        log("  跳过 (无lm_head SVD)")
    
    # ============================================================
    # S4: 差分向量与SVD方向的投影分析 (轻量版, 避免OOM)
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S4: 差分向量与SVD方向的投影分析")
    log(f"{'='*60}")
    
    layer = last_layer
    for feat in feature_names:
        if feat not in layer_pca[layer] or svd_directions is None:
            continue
        
        deltas = np.array(layer_deltas[layer][feat])
        pc1 = layer_pca[layer][feat]['pc1_dir']
        
        # 每个差分向量在PC1方向的投影
        pc1_projs = np.array([np.dot(d, pc1) for d in deltas])
        
        # 每个差分向量在SVD[0]和SVD[1]方向的投影
        svd0_projs = np.array([np.dot(d, svd_directions[0]) for d in deltas])
        svd1_projs = np.array([np.dot(d, svd_directions[1]) for d in deltas])
        
        # 差分向量范数
        norms = np.array([np.linalg.norm(d) for d in deltas])
        
        # 计算投影比的统计量
        # |proj_pc1| / norm vs |proj_svd0| / norm vs |proj_svd1| / norm
        pc1_ratio = np.mean(np.abs(pc1_projs) / (norms + 1e-10))
        svd0_ratio = np.mean(np.abs(svd0_projs) / (norms + 1e-10))
        svd1_ratio = np.mean(np.abs(svd1_projs) / (norms + 1e-10))
        
        log(f"  {feat} (L{layer}):")
        log(f"    差分向量在PC1方向投影比: {pc1_ratio:.4f}")
        log(f"    差分向量在SVD[0]方向投影比: {svd0_ratio:.4f}")
        log(f"    差分向量在SVD[1]方向投影比: {svd1_ratio:.4f}")
        log(f"    PC1投影比 / SVD[0]投影比 = {pc1_ratio/svd0_ratio:.2f}x")
        log(f"    PC1投影比 / SVD[1]投影比 = {pc1_ratio/svd1_ratio:.2f}x")
        log(f"    SVD[1]投影比 / SVD[0]投影比 = {svd1_ratio/svd0_ratio:.2f}x")
    
    # ============================================================
    # S5: 控制分析 — 随机方向 vs PC1 vs SVD
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S5: 控制分析 — 随机方向 vs PC1 vs SVD")
    log(f"{'='*60}")
    
    n_random = 100
    layer = last_layer
    
    for feat in feature_names:
        if feat not in layer_pca[layer]:
            continue
        
        pc1 = layer_pca[layer][feat]['pc1_dir']
        deltas = np.array(layer_deltas[layer][feat])
        
        # PC1方向的投影比
        pc1_projs = [abs(np.dot(d, pc1)) for d in deltas]
        
        # 随机方向的投影比
        rand_projs_all = []
        for _ in range(n_random):
            rand_dir = np.random.randn(d_model)
            rand_dir /= np.linalg.norm(rand_dir)
            rand_projs = [abs(np.dot(d, rand_dir)) for d in deltas]
            rand_projs_all.append(np.mean(rand_projs))
        
        # SVD前3方向的投影
        if svd_directions is not None:
            svd_projs = {}
            for k in [0, 1, 2]:
                svd_projs[k] = [abs(np.dot(d, svd_directions[k])) for d in deltas]
        
        log(f"\n  {feat} (L{layer}):")
        log(f"    PC1方向: mean_proj={np.mean(pc1_projs):.6f}")
        log(f"    随机方向: mean_proj={np.mean(rand_projs_all):.6f} ± {np.std(rand_projs_all):.6f}")
        log(f"    PC1/Random = {np.mean(pc1_projs)/np.mean(rand_projs_all):.2f}x")
        
        if svd_directions is not None:
            for k in [0, 1, 2]:
                log(f"    SVD[{k}]方向: mean_proj={np.mean(svd_projs[k]):.6f}, "
                    f"SVD[{k}]/Random = {np.mean(svd_projs[k])/np.mean(rand_projs_all):.2f}x")
            
            # PC1方向与SVD方向的投影比对比
            # 如果PC1/Random >> SVD[0]/Random, 说明PC1不是简单地被SVD[0]主导
            # 如果PC1/Random ≈ SVD[0]/Random, 说明PC1 ≈ SVD[0]
            log(f"    PC1/Random vs SVD[0]/Random: "
                f"{np.mean(pc1_projs)/np.mean(rand_projs_all):.2f}x vs "
                f"{np.mean(svd_projs[0])/np.mean(rand_projs_all):.2f}x")
    
    # ============================================================
    # S6: 汇总与判断
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S6: 汇总与判断")
    log(f"{'='*60}")
    
    if svd_directions is not None:
        log(f"\n  === PC1 vs SVD主方向对齐度汇总 (Layer {last_layer}) ===")
        header = f"  {'Feature':>20} {'PC1_var':>8} {'Best_SVD':>8} {'Best_cos':>8} {'SVD0_cos':>8} {'Proj_top1':>10} {'Proj_top5':>10} {'Proj_top20':>10}"
        log(header)
        
        for feat in feature_names:
            if feat not in layer_pca[last_layer]:
                continue
            pc1 = layer_pca[last_layer][feat]['pc1_dir']
            cosines = []
            for k in range(n_svd):
                cos_val = abs(np.dot(pc1, svd_directions[k]) / 
                            (np.linalg.norm(pc1) * np.linalg.norm(svd_directions[k])))
                cosines.append(cos_val)
            best_k = np.argmax(cosines)
            
            proj_top1 = cosines[0]**2
            proj_top5 = sum(c**2 for c in cosines[:5])
            proj_top20 = sum(c**2 for c in cosines[:20])
            
            log(f"  {feat:>20} {layer_pca[last_layer][feat]['pc1_var']*100:>7.1f}% "
                f"{'SVD['+str(best_k)+']':>8} {cosines[best_k]:>8.4f} {cosines[0]:>8.4f} "
                f"{proj_top1:>10.4f} {proj_top5:>10.4f} {proj_top20:>10.4f}")
        
        log(f"\n  === 层间对比: PC1→SVD[0]对齐度 ===")
        header = f"  {'Feature':>20}"
        for layer in analysis_layers:
            header += f" {'L'+str(layer):>8}"
        log(header)
        
        for feat in feature_names:
            line = f"  {feat:>20}"
            for layer in analysis_layers:
                if feat in layer_pca[layer]:
                    pc1 = layer_pca[layer][feat]['pc1_dir']
                    cos0 = abs(np.dot(pc1, svd_directions[0]) / 
                              (np.linalg.norm(pc1) * np.linalg.norm(svd_directions[0])))
                    line += f" {cos0:>8.4f}"
                else:
                    line += f" {'N/A':>8}"
            log(line)
        
        # 判断输出汇聚假设
        log(f"\n  === 输出汇聚假设判断 ===")
        
        # 收集最后一层的对齐数据
        last_aligns = []
        for feat in feature_names:
            if feat in layer_pca[last_layer]:
                pc1 = layer_pca[last_layer][feat]['pc1_dir']
                cos0 = abs(np.dot(pc1, svd_directions[0]) / 
                          (np.linalg.norm(pc1) * np.linalg.norm(svd_directions[0])))
                last_aligns.append(cos0)
        
        mean_last_align = np.mean(last_aligns) if last_aligns else 0
        
        # 收集中间层的对齐数据
        mid_aligns = []
        for layer in analysis_layers[:-1]:
            for feat in feature_names:
                if feat in layer_pca[layer]:
                    pc1 = layer_pca[layer][feat]['pc1_dir']
                    cos0 = abs(np.dot(pc1, svd_directions[0]) / 
                              (np.linalg.norm(pc1) * np.linalg.norm(svd_directions[0])))
                    mid_aligns.append(cos0)
        
        mean_mid_align = np.mean(mid_aligns) if mid_aligns else 0
        
        log(f"  最后一层PC1→SVD[0]平均对齐度: {mean_last_align:.4f}")
        log(f"  前面层PC1→SVD[0]平均对齐度: {mean_mid_align:.4f}")
        log(f"  对齐度提升: {mean_last_align - mean_mid_align:.4f}")
        
        if mean_last_align > 0.7:
            log(f"  ★★★ 强输出汇聚: 最后一层PC1高度对齐SVD[0]!")
            log(f"  → 1D流形方向 = lm_head的主奇异方向")
            log(f"  → '输出汇聚'假设成立: 网络主动将信息对齐到lm_head")
        elif mean_last_align > 0.4:
            log(f"  ★★ 部分输出汇聚: 最后一层PC1中等对齐SVD[0]")
            log(f"  → 1D流形方向与lm_head有关, 但不完全是SVD[0]")
        else:
            log(f"  ★ 无输出汇聚: 最后一层PC1与SVD[0]不对齐")
            log(f"  → 1D流形方向独立于lm_head → 有语义意义")
        
        # 投影比对比
        log(f"\n  === PC1/Random vs SVD[0]/Random ===")
        log(f"  如果PC1/Random ≈ SVD[0]/Random → PC1方向被SVD[0]主导 (trivial)")
        log(f"  如果PC1/Random >> SVD[0]/Random → PC1方向有独立因果效应 (非trivial)")
    else:
        log("  跳过汇总 (无lm_head SVD)")
    
    # 保存结果
    results = {
        'model': model_key,
        'n_pairs': n_pairs,
        'analysis_layers': analysis_layers,
        'last_layer': last_layer,
        'svd_singular_top20': svd_singular.tolist() if svd_directions is not None else None,
        'layer_pca_summary': {},
    }
    
    for layer in analysis_layers:
        results['layer_pca_summary'][str(layer)] = {}
        for feat in feature_names:
            if feat in layer_pca[layer]:
                results['layer_pca_summary'][str(layer)][feat] = {
                    'pc1_var': float(layer_pca[layer][feat]['pc1_var']),
                    'explained_var_top5': layer_pca[layer][feat]['explained_var'].tolist(),
                }
    
    with open(f'{out_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    log(f"\n结果已保存到 {out_dir}/results.json")
    log(f"\n{'='*70}")
    log(f"CCXXIII 完成!")
    log(f"{'='*70}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
