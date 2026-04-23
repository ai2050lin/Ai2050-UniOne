"""
Phase CCXIX: 因果干预实验 — 验证瓶颈层PC1是否因果控制输出
================================================================
核心目标:
1. PC投影比例分析: 差分信号在各PC方向上的承载比例
2. Per-feature PCA: 每个特征独立PCA, 比较PC1占比
3. 残差分析: 消除PC1后, 剩余差分信号是否还能区分语法/语义
4. PC1方向跨特征一致性: 不同特征的PC1是否对齐
5. PC1缩放对logit的影响(间接方式: 通过激活编辑)

瓶颈层位置(来自CCXVIII):
  Qwen3: L6 (PC1=99.90%)
  DS7B:  L4 (PC1=99.84%)
  GLM4:  无明确瓶颈, 选取L30(最大PC旋转处)作为对照

样本量: 100对/特征
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

# 复用CCXVIII的特征对定义
FEATURE_PAIRS_BASE = {
    'tense': {
        'type': 'SYN',
        'pairs': [
            ("The cat sleeps on the mat", "The cat slept on the mat"),
            ("She walks to school every day", "She walked to school every day"),
            ("He runs in the park", "He ran in the park"),
            ("They play football on Sundays", "They played football on Sundays"),
            ("I read books every night", "I read books last night"),
            ("The dog barks at strangers", "The dog barked at strangers"),
            ("We cook dinner together", "We cooked dinner together"),
            ("She sings beautifully", "She sang beautifully"),
            ("The baby cries loudly", "The baby cried loudly"),
            ("He drives to work", "He drove to work"),
            ("Birds fly south in winter", "Birds flew south in winter"),
            ("The teacher explains the lesson", "The teacher explained the lesson"),
            ("I write letters to friends", "I wrote letters to friends"),
            ("The wind blows gently", "The wind blew gently"),
            ("Children laugh and play", "Children laughed and played"),
            ("The river flows to the sea", "The river flowed to the sea"),
            ("She paints beautiful pictures", "She painted beautiful pictures"),
            ("The sun shines brightly", "The sun shone brightly"),
            ("We study English at school", "We studied English at school"),
            ("He builds model airplanes", "He built model airplanes"),
            ("The train arrives at noon", "The train arrived at noon"),
            ("I drink coffee every morning", "I drank coffee this morning"),
            ("The flowers bloom in spring", "The flowers bloomed in spring"),
            ("She teaches mathematics", "She taught mathematics"),
            ("The cat jumps over the fence", "The cat jumped over the fence"),
            ("He knows the answer", "He knew the answer"),
            ("We visit grandma on weekends", "We visited grandma last weekend"),
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
            ("Grandpa drinks tea every afternoon", "Grandpa drank tea every afternoon"),
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
            ("I read the report carefully", "I read the report yesterday"),
            ("The puppy barks at the mirror", "The puppy barked at the mirror"),
            ("We bake cookies on weekends", "We baked cookies last weekend"),
            ("The soprano sings the aria", "The soprano sang the aria"),
            ("The infant cries for attention", "The infant cried for attention"),
            ("The courier drives a van", "The courier drove a van"),
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
        ]
    },
    'polarity': {
        'type': 'SYN',
        'pairs': [
            ("The cat is on the mat", "The cat is not on the mat"),
            ("She likes chocolate", "She does not like chocolate"),
            ("He can swim well", "He cannot swim well"),
            ("They have finished the project", "They have not finished the project"),
            ("I understand the question", "I do not understand the question"),
            ("The door is open", "The door is not open"),
            ("We need more time", "We do not need more time"),
            ("She was at the party", "She was not at the party"),
            ("He will come tomorrow", "He will not come tomorrow"),
            ("The answer is correct", "The answer is not correct"),
            ("Birds can fly", "Birds cannot fly"),
            ("The store is open today", "The store is not open today"),
            ("I have seen this movie", "I have not seen this movie"),
            ("She speaks French fluently", "She does not speak French fluently"),
            ("The train has arrived", "The train has not arrived"),
            ("He knows the password", "He does not know the password"),
            ("We enjoyed the concert", "We did not enjoy the concert"),
            ("The cake tastes good", "The cake does not taste good"),
            ("They found the treasure", "They did not find the treasure"),
            ("I believe your story", "I do not believe your story"),
            ("The weather is nice today", "The weather is not nice today"),
            ("She plays the piano", "She does not play the piano"),
            ("He finished his homework", "He did not finish his homework"),
            ("The team won the match", "The team did not win the match"),
            ("We received the package", "We did not receive the package"),
            ("The room is clean", "The room is not clean"),
            ("I remember that day", "I do not remember that day"),
            ("She agreed with the plan", "She did not agree with the plan"),
            ("The test was easy", "The test was not easy"),
            ("He passed the exam", "He did not pass the exam"),
            ("The food is ready", "The food is not ready"),
            ("They own a house", "They do not own a house"),
            ("The road is safe", "The road is not safe"),
            ("I trust his judgment", "I do not trust his judgment"),
            ("We saw the rainbow", "We did not see the rainbow"),
            ("The water is warm", "The water is not warm"),
            ("She won the prize", "She did not win the prize"),
            ("The plan worked perfectly", "The plan did not work perfectly"),
            ("He told the truth", "He did not tell the truth"),
            ("The dog is friendly", "The dog is not friendly"),
            ("A bird is in the tree", "A bird is not in the tree"),
            ("My friend likes ice cream", "My friend does not like ice cream"),
            ("The child can read already", "The child cannot read already"),
            ("Some people have finished eating", "Some people have not finished eating"),
            ("John understands the rules", "John does not understand the rules"),
            ("Our window is open wide", "Our window is not open wide"),
            ("Mom needs some help", "Mom does not need some help"),
            ("Anna was at the library", "Anna was not at the library"),
            ("Dad will call tonight", "Dad will not call tonight"),
            ("This answer is right", "This answer is not right"),
            ("Parrots can talk", "Parrots cannot talk"),
            ("The shop is open late", "The shop is not open late"),
            ("I have tried that dish", "I have not tried that dish"),
            ("Linda speaks German well", "Linda does not speak German well"),
            ("The bus has come yet", "The bus has not come yet"),
            ("She knows the secret", "She does not know the secret"),
            ("We loved the show", "We did not love the show"),
            ("The soup smells delicious", "The soup does not smell delicious"),
            ("They discovered the truth", "They did not discover the truth"),
            ("I accept your apology", "I do not accept your apology"),
            ("Today is a sunny day", "Today is not a sunny day"),
            ("He plays guitar", "He does not play guitar"),
            ("Mary completed the task", "Mary did not complete the task"),
            ("Our team scored a goal", "Our team did not score a goal"),
            ("I got the message", "I did not get the message"),
            ("The kitchen is spotless", "The kitchen is not spotless"),
            ("I recall that moment", "I do not recall that moment"),
            ("Tom supported the idea", "Tom did not support the idea"),
            ("The exam was simple", "The exam was not simple"),
            ("She qualified for finals", "She did not qualify for finals"),
            ("Dinner is served", "Dinner is not served"),
            ("They possess a car", "They do not possess a car"),
            ("The bridge is sturdy", "The bridge is not sturdy"),
            ("I admire her courage", "I do not admire her courage"),
            ("We witnessed the event", "We did not witness the event"),
            ("The pool is heated", "The pool is not heated"),
            ("He earned the award", "He did not earn the award"),
            ("The strategy paid off", "The strategy did not pay off"),
            ("She kept her promise", "She did not keep her promise"),
            ("The kitten is playful", "The kitten is not playful"),
            ("The sky is clear tonight", "The sky is not clear tonight"),
            ("My mother loves gardening", "My mother does not love gardening"),
            ("The student can solve it", "The student cannot solve it"),
            ("The gate is locked", "The gate is not locked"),
            ("The team needs a break", "The team does not need a break"),
            ("She will attend the meeting", "She will not attend the meeting"),
            ("The solution is optimal", "The solution is not optimal"),
            ("Eagles can soar high", "Eagles cannot soar high"),
            ("The bank is open on Saturday", "The bank is not open on Saturday"),
            ("He speaks Japanese fluently", "He does not speak Japanese fluently"),
            ("The package has arrived safely", "The package has not arrived safely"),
            ("She remembers the password", "She does not remember the password"),
            ("The music sounds pleasant", "The music does not sound pleasant"),
            ("They completed the mission", "They did not complete the mission"),
            ("I recognize that face", "I do not recognize that face"),
            ("The committee approved the budget", "The committee did not approve the budget"),
            ("This theory explains the data", "This theory does not explain the data"),
            ("The machine works properly", "The machine does not work properly"),
            ("The project requires funding", "The project does not require funding"),
            ("The concert begins at seven", "The concert does not begin at seven"),
        ]
    },
    'number': {
        'type': 'SYN',
        'pairs': [
            ("The cat sits on the mat", "The cats sit on the mat"),
            ("A dog barks loudly", "Dogs bark loudly"),
            ("The child plays in the garden", "The children play in the garden"),
            ("This book is interesting", "These books are interesting"),
            ("The woman walks to work", "The women walk to work"),
            ("A bird sings in the tree", "Birds sing in the tree"),
            ("The student studies hard", "The students study hard"),
            ("That flower is beautiful", "Those flowers are beautiful"),
            ("The man reads the newspaper", "The men read the newspaper"),
            ("A fish swims in the pond", "Fish swim in the pond"),
            ("The leaf falls from the tree", "The leaves fall from the tree"),
            ("This country is large", "These countries are large"),
            ("The baby cries for milk", "The babies cry for milk"),
            ("A mouse runs across the floor", "Mice run across the floor"),
            ("The person enters the room", "The people enter the room"),
            ("That knife is sharp", "Those knives are sharp"),
            ("The wolf howls at the moon", "The wolves howl at the moon"),
            ("A tooth needs cleaning", "Teeth need cleaning"),
            ("The foot hurts badly", "The feet hurt badly"),
            ("This goose swims in the lake", "These geese swim in the lake"),
            ("The bus arrives at the station", "The buses arrive at the station"),
            ("A box contains old photos", "Boxes contain old photos"),
            ("The match lights the fire", "The matches light the fire"),
            ("That dish tastes delicious", "Those dishes taste delicious"),
            ("The glass breaks easily", "Glasses break easily"),
            ("A brush paints the wall", "Brushes paint the wall"),
            ("The wish comes true", "Wishes come true"),
            ("This watch keeps perfect time", "These watches keep perfect time"),
            ("The class begins at nine", "The classes begin at nine"),
            ("A bush grows near the fence", "Bushes grow near the fence"),
            ("The church stands on the hill", "The churches stand on the hill"),
            ("A fox hides in the den", "Foxes hide in the den"),
            ("The calf drinks milk", "Calves drink milk"),
            ("This half is larger", "These halves are larger"),
            ("The calf runs to its mother", "The calves run to their mother"),
            ("A loaf sits on the table", "Loaves sit on the table"),
            ("The thief steals quietly", "Thieves steal quietly"),
            ("This shelf holds many books", "These shelves hold many books"),
            ("The wolf hunts at night", "The wolves hunt at night"),
            ("A scarf wraps around the neck", "Scarves wrap around the neck"),
            ("The cat chases a mouse", "The cats chase mice"),
            ("A dog guards the house", "Dogs guard the house"),
            ("The boy kicks the ball", "The boys kick the ball"),
            ("This tree loses its leaves", "These trees lose their leaves"),
            ("The lady walks her dog", "The ladies walk their dogs"),
            ("A goose honks loudly", "Geese honk loudly"),
            ("The child learns to read", "The children learn to read"),
            ("That house looks old", "Those houses look old"),
            ("The man drives a truck", "The men drive trucks"),
            ("A sheep grazes in the field", "Sheep graze in the field"),
            ("The tooth aches terribly", "The teeth ache terribly"),
            ("This city is very crowded", "These cities are very crowded"),
            ("The mouse eats the cheese", "The mice eat the cheese"),
            ("A person waits at the bus stop", "People wait at the bus stop"),
            ("That knife cuts well", "Those knives cut well"),
            ("The wolf howls in the forest", "The wolves howl in the forest"),
            ("A foot steps in the puddle", "Feet step in the puddle"),
            ("The goose swims across the pond", "The geese swim across the pond"),
            ("This class studies mathematics", "These classes study mathematics"),
            ("The bus stops at the corner", "The buses stop at the corner"),
            ("A box holds the supplies", "Boxes hold the supplies"),
            ("The match starts at three", "The matches start at three"),
            ("That dish needs salt", "Those dishes need salt"),
            ("The glass holds water", "Glasses hold water"),
            ("A brush cleans the surface", "Brushes clean the surface"),
            ("The wish is granted", "Wishes are granted"),
            ("This watch costs a lot", "These watches cost a lot"),
            ("The church bells ring", "The church bells ring"),
            ("A fox approaches cautiously", "Foxes approach cautiously"),
            ("The calf stays close", "The calves stay close"),
            ("This half is enough", "These halves are enough"),
            ("A loaf is freshly baked", "Loaves are freshly baked"),
            ("The thief runs away", "Thieves run away"),
            ("This shelf is sturdy", "These shelves are sturdy"),
            ("The scarf blows in the wind", "Scarves blow in the wind"),
            ("The ox pulls the cart", "The oxen pull the cart"),
            ("A louse crawls on the skin", "Lice crawl on the skin"),
            ("The criterion is strict", "The criteria are strict"),
            ("This phenomenon is rare", "These phenomena are rare"),
            ("The datum suggests a trend", "The data suggest a trend"),
            ("A stimulus triggers a response", "Stimuli trigger responses"),
            ("The nucleus contains DNA", "The nuclei contain DNA"),
            ("This syllabus covers the topic", "These syllabi cover the topic"),
            ("The analysis reveals the pattern", "The analyses reveal the pattern"),
            ("A basis for the argument exists", "Bases for the argument exist"),
            ("The thesis is compelling", "The theses are compelling"),
            ("This crisis demands attention", "These crises demand attention"),
            ("The index lists the entries", "The indices list the entries"),
            ("A formula solves the problem", "Formulas solve the problem"),
            ("The vertex points upward", "The vertices point upward"),
            ("This axis runs horizontally", "These axes run horizontally"),
            ("The appendix provides details", "The appendices provide details"),
            ("A medium conveys the message", "Media convey the message"),
            ("The focus is sharp", "The foci are sharp"),
            ("This genus includes many species", "These genera include many species"),
            ("The corpus is extensive", "The corpora are extensive"),
            ("A memorandum records the event", "Memoranda record the event"),
        ]
    },
    'question': {
        'type': 'SYN',
        'pairs': [
            ("The cat is on the mat", "Is the cat on the mat"),
            ("She likes chocolate", "Does she like chocolate"),
            ("He can swim well", "Can he swim well"),
            ("They have finished the project", "Have they finished the project"),
            ("The door is open", "Is the door open"),
            ("We need more time", "Do we need more time"),
            ("She was at the party", "Was she at the party"),
            ("He will come tomorrow", "Will he come tomorrow"),
            ("The answer is correct", "Is the answer correct"),
            ("Birds can fly", "Can birds fly"),
            ("The store is open today", "Is the store open today"),
            ("I have seen this movie", "Have I seen this movie"),
            ("She speaks French fluently", "Does she speak French fluently"),
            ("The train has arrived", "Has the train arrived"),
            ("He knows the password", "Does he know the password"),
            ("The cake tastes good", "Does the cake taste good"),
            ("They found the treasure", "Did they find the treasure"),
            ("The weather is nice today", "Is the weather nice today"),
            ("She plays the piano", "Does she play the piano"),
            ("He finished his homework", "Did he finish his homework"),
            ("The team won the match", "Did the team win the match"),
            ("We received the package", "Did we receive the package"),
            ("The room is clean", "Is the room clean"),
            ("She agreed with the plan", "Did she agree with the plan"),
            ("The test was easy", "Was the test easy"),
            ("He passed the exam", "Did he pass the exam"),
            ("The food is ready", "Is the food ready"),
            ("They own a house", "Do they own a house"),
            ("The road is safe", "Is the road safe"),
            ("The water is warm", "Is the water warm"),
            ("She won the prize", "Did she win the prize"),
            ("He told the truth", "Did he tell the truth"),
            ("The dog is friendly", "Is the dog friendly"),
            ("The cat is sleeping", "Is the cat sleeping"),
            ("The children are playing", "Are the children playing"),
            ("This book belongs to Mary", "Does this book belong to Mary"),
            ("The sun rises in the east", "Does the sun rise in the east"),
            ("The movie starts at eight", "Does the movie start at eight"),
            ("The flowers need water", "Do the flowers need water"),
            ("The car needs fuel", "Does the car need fuel"),
            ("A bird is in the tree", "Is a bird in the tree"),
            ("My friend likes ice cream", "Does my friend like ice cream"),
            ("The child can read already", "Can the child read already"),
            ("John understands the rules", "Does John understand the rules"),
            ("Our window is open wide", "Is our window open wide"),
            ("Mom needs some help", "Does Mom need some help"),
            ("Anna was at the library", "Was Anna at the library"),
            ("Dad will call tonight", "Will Dad call tonight"),
            ("This answer is right", "Is this answer right"),
            ("Parrots can talk", "Can parrots talk"),
            ("The shop is open late", "Is the shop open late"),
            ("She knows the secret", "Does she know the secret"),
            ("The soup smells delicious", "Does the soup smell delicious"),
            ("Today is a sunny day", "Is today a sunny day"),
            ("He plays guitar", "Does he play guitar"),
            ("Mary completed the task", "Did Mary complete the task"),
            ("Our team scored a goal", "Did our team score a goal"),
            ("The kitchen is spotless", "Is the kitchen spotless"),
            ("The exam was simple", "Was the exam simple"),
            ("Dinner is served", "Is dinner served"),
            ("They possess a car", "Do they possess a car"),
            ("The bridge is sturdy", "Is the bridge sturdy"),
            ("The pool is heated", "Is the pool heated"),
            ("He earned the award", "Did he earn the award"),
            ("She kept her promise", "Did she keep her promise"),
            ("The kitten is playful", "Is the kitten playful"),
            ("The sky is clear tonight", "Is the sky clear tonight"),
            ("My mother loves gardening", "Does my mother love gardening"),
            ("The student can solve it", "Can the student solve it"),
            ("The gate is locked", "Is the gate locked"),
            ("The team needs a break", "Does the team need a break"),
            ("She will attend the meeting", "Will she attend the meeting"),
            ("The solution is optimal", "Is the solution optimal"),
            ("Eagles can soar high", "Can eagles soar high"),
            ("The bank is open on Saturday", "Is the bank open on Saturday"),
            ("He speaks Japanese fluently", "Does he speak Japanese fluently"),
            ("The package has arrived safely", "Has the package arrived safely"),
            ("She remembers the password", "Does she remember the password"),
            ("The music sounds pleasant", "Does the music sound pleasant"),
            ("They completed the mission", "Did they complete the mission"),
            ("I recognize that face", "Do I recognize that face"),
            ("The committee approved the budget", "Did the committee approve the budget"),
            ("This theory explains the data", "Does this theory explain the data"),
            ("The machine works properly", "Does the machine work properly"),
            ("The project requires funding", "Does the project require funding"),
            ("The concert begins at seven", "Does the concert begin at seven"),
            ("The report contains errors", "Does the report contain errors"),
            ("This method produces results", "Does this method produce results"),
            ("The system handles requests", "Does the system handle requests"),
            ("The experiment confirms the hypothesis", "Does the experiment confirm the hypothesis"),
            ("The device measures temperature", "Does the device measure temperature"),
            ("The algorithm processes data", "Does the algorithm process data"),
            ("The model predicts outcomes", "Does the model predict outcomes"),
            ("This technique improves accuracy", "Does this technique improve accuracy"),
            ("The program supports multiple formats", "Does the program support multiple formats"),
            ("The network connects devices", "Does the network connect devices"),
            ("The software updates automatically", "Does the software update automatically"),
        ]
    },
    'info_structure': {
        'type': 'SYN',
        'pairs': [
            ("John broke the window", "The window was broken by John"),
            ("Mary ate the cake", "The cake was eaten by Mary"),
            ("Tom wrote the letter", "The letter was written by Tom"),
            ("She painted the house", "The house was painted by her"),
            ("He fixed the car", "The car was fixed by him"),
            ("They built the bridge", "The bridge was built by them"),
            ("The chef cooked the meal", "The meal was cooked by the chef"),
            ("The dog chased the cat", "The cat was chased by the dog"),
            ("The wind blew the door open", "The door was blown open by the wind"),
            ("The company launched the product", "The product was launched by the company"),
            ("The teacher graded the papers", "The papers were graded by the teacher"),
            ("The storm destroyed the crops", "The crops were destroyed by the storm"),
            ("The artist painted a mural", "A mural was painted by the artist"),
            ("The committee approved the plan", "The plan was approved by the committee"),
            ("The scientist discovered the element", "The element was discovered by the scientist"),
            ("The musician composed the symphony", "The symphony was composed by the musician"),
            ("The author wrote the novel", "The novel was written by the author"),
            ("The architect designed the building", "The building was designed by the architect"),
            ("The programmer coded the software", "The software was coded by the programmer"),
            ("The farmer harvested the wheat", "The wheat was harvested by the farmer"),
            ("The police arrested the suspect", "The suspect was arrested by the police"),
            ("The nurse treated the patient", "The patient was treated by the nurse"),
            ("The driver delivered the package", "The package was delivered by the driver"),
            ("The coach trained the team", "The team was trained by the coach"),
            ("The judge sentenced the criminal", "The criminal was sentenced by the judge"),
            ("The baker made the bread", "The bread was made by the baker"),
            ("The mechanic repaired the engine", "The engine was repaired by the mechanic"),
            ("The cleaner washed the floor", "The floor was washed by the cleaner"),
            ("The librarian organized the books", "The books were organized by the librarian"),
            ("The editor revised the article", "The article was revised by the editor"),
            ("The pilot flew the plane", "The plane was flown by the pilot"),
            ("The waiter served the food", "The food was served by the waiter"),
            ("The director filmed the scene", "The scene was filmed by the director"),
            ("The doctor prescribed the medicine", "The medicine was prescribed by the doctor"),
            ("The student solved the equation", "The equation was solved by the student"),
            ("The president signed the bill", "The bill was signed by the president"),
            ("The goalkeeper saved the penalty", "The penalty was saved by the goalkeeper"),
            ("The conductor led the orchestra", "The orchestra was led by the conductor"),
            ("The manager hired the employee", "The employee was hired by the manager"),
            ("The general commanded the army", "The army was commanded by the general"),
            ("The king ruled the kingdom", "The kingdom was ruled by the king"),
            ("The queen governed the nation", "The nation was governed by the queen"),
            ("The mayor led the city council", "The city council was led by the mayor"),
            ("The captain steered the ship", "The ship was steered by the captain"),
            ("The sheriff protected the town", "The town was protected by the sheriff"),
            ("The professor taught the course", "The course was taught by the professor"),
            ("The researcher published the findings", "The findings were published by the researcher"),
            ("The engineer tested the prototype", "The prototype was tested by the engineer"),
            ("The designer created the logo", "The logo was created by the designer"),
            ("The analyst prepared the report", "The report was prepared by the analyst"),
            ("The inspector checked the equipment", "The equipment was checked by the inspector"),
            ("The supervisor reviewed the proposal", "The proposal was reviewed by the supervisor"),
            ("The technician calibrated the instrument", "The instrument was calibrated by the technician"),
            ("The operator ran the machine", "The machine was run by the operator"),
            ("The consultant advised the client", "The client was advised by the consultant"),
            ("The counselor guided the student", "The student was guided by the counselor"),
            ("The mediator resolved the conflict", "The conflict was resolved by the mediator"),
            ("The trainer coached the athlete", "The athlete was coached by the trainer"),
            ("The instructor demonstrated the method", "The method was demonstrated by the instructor"),
            ("The guide showed the route", "The route was shown by the guide"),
            ("The leader inspired the group", "The group was inspired by the leader"),
            ("The pioneer explored the territory", "The territory was explored by the pioneer"),
            ("The inventor designed the device", "The device was designed by the inventor"),
            ("The scholar wrote the treatise", "The treatise was written by the scholar"),
            ("The specialist handled the case", "The case was handled by the specialist"),
            ("The expert analyzed the data", "The data was analyzed by the expert"),
            ("The developer built the application", "The application was built by the developer"),
            ("The planner organized the event", "The event was organized by the planner"),
            ("The coordinator managed the project", "The project was managed by the coordinator"),
            ("The administrator processed the request", "The request was processed by the administrator"),
            ("The receptionist handled the call", "The call was handled by the receptionist"),
            ("The assistant helped the manager", "The manager was helped by the assistant"),
            ("The deputy represented the official", "The official was represented by the deputy"),
            ("The volunteer supported the cause", "The cause was supported by the volunteer"),
            ("The activist championed the movement", "The movement was championed by the activist"),
            ("The philosopher questioned the assumption", "The assumption was questioned by the philosopher"),
            ("The historian documented the era", "The era was documented by the historian"),
            ("The chemist synthesized the compound", "The compound was synthesized by the chemist"),
            ("The biologist studied the organism", "The organism was studied by the biologist"),
            ("The physicist measured the force", "The force was measured by the physicist"),
            ("The astronomer observed the galaxy", "The galaxy was observed by the astronomer"),
            ("The geologist examined the rock", "The rock was examined by the geologist"),
            ("The meteorologist tracked the storm", "The storm was tracked by the meteorologist"),
            ("The ecologist monitored the habitat", "The habitat was monitored by the ecologist"),
            ("The geneticist mapped the genome", "The genome was mapped by the geneticist"),
            ("The neurologist studied the brain", "The brain was studied by the neurologist"),
            ("The cardiologist examined the heart", "The heart was examined by the cardiologist"),
            ("The surgeon performed the operation", "The operation was performed by the surgeon"),
            ("The pharmacist dispensed the medication", "The medication was dispensed by the pharmacist"),
            ("The dentist cleaned the teeth", "The teeth were cleaned by the dentist"),
            ("The optometrist checked the vision", "The vision was checked by the optometrist"),
            ("The psychologist assessed the patient", "The patient was assessed by the psychologist"),
            ("The therapist treated the injury", "The injury was treated by the therapist"),
            ("The dietitian planned the meals", "The meals were planned by the dietitian"),
            ("The veterinarian treated the animal", "The animal was treated by the veterinarian"),
            ("The botanist identified the plant", "The plant was identified by the botanist"),
            ("The zoologist studied the species", "The species was studied by the zoologist"),
        ]
    },
    'formality': {
        'type': 'SEM',
        'pairs': [
            ("Hey, what's up", "Greetings, how are you doing"),
            ("It's really cool", "It is quite impressive"),
            ("She's gonna help", "She is going to assist"),
            ("That's awesome", "That is remarkable"),
            ("Cool, thanks a lot", "Thank you very much indeed"),
            ("What's the deal", "What is the situation"),
            ("He's a great guy", "He is an excellent individual"),
            ("Let's grab some food", "Let us obtain some nourishment"),
            ("I dunno about that", "I am uncertain about that matter"),
            ("She's super smart", "She is highly intelligent"),
            ("That's dope", "That is exceptional"),
            ("We gotta go now", "We must depart immediately"),
            ("What's happening", "What is occurring"),
            ("He's kinda weird", "He is somewhat unusual"),
            ("That's legit", "That is legitimate"),
            ("She's really nice", "She is genuinely kind"),
            ("Got any ideas", "Do you have any suggestions"),
            ("That's wild", "That is extraordinary"),
            ("He's totally right", "He is absolutely correct"),
            ("Let me think", "Allow me to contemplate"),
            ("That's crazy", "That is astonishing"),
            ("She's pretty good", "She is fairly competent"),
            ("Wanna hang out", "Would you like to socialize"),
            ("He's been busy", "He has been occupied"),
            ("That's so funny", "That is quite amusing"),
            ("I'm real tired", "I am genuinely exhausted"),
            ("She's my buddy", "She is my companion"),
            ("Got it done", "Completed the task"),
            ("That's sick", "That is impressive"),
            ("He's a big deal", "He is a significant figure"),
            ("Yo, check this out", "Please observe this"),
            ("She's super cool", "She is exceptionally pleasant"),
            ("That's messed up", "That is problematic"),
            ("He's outta here", "He has departed"),
            ("What's the plan", "What is the strategy"),
            ("That's sweet", "That is delightful"),
            ("She's a blast", "She is thoroughly enjoyable"),
            ("Gimme a break", "Grant me a respite"),
            ("He's got chops", "He possesses considerable skill"),
            ("That's fire", "That is outstanding"),
            ("Sup, how's it going", "Hello, how are you faring"),
            ("Yo, that's lit", "I say, that is splendid"),
            ("She's chill", "She is composed"),
            ("Dude, no way", "Sir, that is improbable"),
            ("That's bougie", "That is luxurious"),
            ("He's basic", "He is conventional"),
            ("I'm lowkey excited", "I am subtly enthusiastic"),
            ("That's legit amazing", "That is genuinely remarkable"),
            ("She's highkey talented", "She is evidently gifted"),
            ("We vibing", "We are enjoying ourselves"),
            ("That's a flex", "That is an impressive display"),
            ("He's salty", "He is resentful"),
            ("I'm shook", "I am astonished"),
            ("That's bussin", "That is delicious"),
            ("She's slaying", "She is excelling"),
            ("No cap, that's real", "Truly, that is authentic"),
            ("He's goated", "He is the greatest of all time"),
            ("That's giving main character", "That exhibits protagonist energy"),
            ("I'm dead, that's so funny", "I am extremely amused"),
            ("She understood the assignment", "She performed admirably"),
            ("That hits different", "That has a unique impact"),
            ("He's got rizz", "He possesses charisma"),
            ("It is what it is", "The situation remains unchanged"),
            ("She's in her era", "She is thriving presently"),
            ("That's underrated", "That deserves more recognition"),
            ("He fell off", "He declined in quality"),
            ("I'm here for it", "I enthusiastically support this"),
            ("That's iconic", "That is legendary"),
            ("She's mother", "She is a paragon"),
            ("Rent free in my head", "Perpetually in my thoughts"),
            ("He understood the memo", "He comprehended the instructions"),
            ("That's giving", "That is impressive"),
            ("She ate that up", "She performed superbly"),
            ("That's a mood", "That resonates emotionally"),
            ("He's valid", "He is worthy of respect"),
            ("Period, no notes", "Definitively, without critique"),
            ("That's sending me", "That is hilarious"),
            ("She's the moment", "She is the highlight"),
            ("He cooked", "He performed exceptionally"),
            ("This slaps", "This is excellent"),
            ("W or L", "Victory or defeat"),
            ("She snapped", "She performed brilliantly"),
            ("That's facts", "That is accurate"),
            ("He's him", "He is exemplary"),
            ("I oop", "I am surprised"),
            ("That's tea", "That is gossip"),
            ("She's serving looks", "She appears stunning"),
            ("We been knew", "We were already aware"),
            ("That's a whole vibe", "That is an experience"),
            ("He's doing the most", "He is exceeding expectations"),
            ("Slay queen", "Excel magnificently"),
            ("That's big brain", "That is intelligent"),
            ("I'm built different", "I am uniquely capable"),
            ("She's an icon", "She is a legend"),
            ("That's based", "That is principled"),
            ("He's cracked", "He is highly skilled"),
            ("No shot", "No possibility"),
            ("She's built different", "She is exceptional"),
            ("That's mid", "That is mediocre"),
        ]
    },
    'semantic_valence': {
        'type': 'SEM',
        'pairs': [
            ("The gift brought immense joy", "The disaster caused terrible grief"),
            ("Her smile radiated warmth", "Her scowl projected hostility"),
            ("The victory inspired pride", "The defeat brought shame"),
            ("He showed great courage", "He displayed extreme cowardice"),
            ("The garden bloomed with beauty", "The wasteland reeked of decay"),
            ("She offered sincere praise", "She delivered harsh criticism"),
            ("The concert was magnificent", "The performance was dreadful"),
            ("He achieved remarkable success", "He suffered utter failure"),
            ("The news brought hope", "The news brought despair"),
            ("She expressed deep gratitude", "She expressed bitter resentment"),
            ("The victory was glorious", "The defeat was humiliating"),
            ("He maintained a positive attitude", "He maintained a negative attitude"),
            ("The transformation was remarkable", "The deterioration was alarming"),
            ("She possessed extraordinary beauty", "She possessed extreme ugliness"),
            ("The invention was groundbreaking", "The invention was pointless"),
            ("He demonstrated unwavering loyalty", "He demonstrated complete betrayal"),
            ("The ceremony was solemn and dignified", "The ceremony was chaotic and disgraceful"),
            ("She exhibited profound insight", "She exhibited shallow thinking"),
            ("The discovery was revolutionary", "The discovery was trivial"),
            ("He pursued a righteous cause", "He pursued a wicked scheme"),
            ("The restoration was meticulous", "The damage was extensive"),
            ("She preserved her dignity", "She lost her dignity"),
            ("The tradition is venerable", "The tradition is obsolete"),
            ("He showed exemplary conduct", "He showed disgraceful conduct"),
            ("The institution is reputable", "The institution is disreputable"),
            ("She maintained impeccable standards", "She maintained dreadful standards"),
            ("The relationship was harmonious", "The relationship was hostile"),
            ("He attained legendary status", "He attained notorious status"),
            ("The craftsmanship was superb", "The craftsmanship was shoddy"),
            ("She achieved lasting fame", "She achieved lasting shame"),
            ("The culture is vibrant", "The culture is stagnant"),
            ("He preserved his integrity", "He compromised his integrity"),
            ("The heritage is priceless", "The heritage is worthless"),
            ("She inspired universal admiration", "She inspired universal contempt"),
            ("The architecture is splendid", "The architecture is grotesque"),
            ("He left a glorious legacy", "He left a disgraceful legacy"),
            ("The blessing was divine", "The curse was infernal"),
            ("She showed boundless generosity", "She showed infinite greed"),
            ("The creation was sublime", "The destruction was horrific"),
            ("He earned profound respect", "He earned deep contempt"),
            ("The dawn brought promise", "The dusk brought foreboding"),
            ("She displayed elegant grace", "She displayed clumsy awkwardness"),
            ("The harmony was perfect", "The discord was intolerable"),
            ("He spoke with eloquent wisdom", "He spoke with foolish ignorance"),
            ("The masterpiece was exquisite", "The forgery was pathetic"),
            ("She radiated serene tranquility", "She radiated anxious turmoil"),
            ("The blessing brought comfort", "The affliction brought suffering"),
            ("He embodied noble virtue", "He embodied base vice"),
            ("The revelation was enlightening", "The deception was misleading"),
            ("She demonstrated compassionate mercy", "She demonstrated ruthless cruelty"),
            ("The miracle was wondrous", "The tragedy was devastating"),
            ("He possessed genuine humility", "He possessed arrogant pride"),
            ("The melody was enchanting", "The noise was deafening"),
            ("She shared abundant generosity", "She hoarded selfish greed"),
            ("The prosperity was flourishing", "The poverty was crushing"),
            ("He extended genuine friendship", "He extended bitter enmity"),
            ("The paradise was heavenly", "The abyss was hellish"),
            ("She conveyed tender affection", "She conveyed cold indifference"),
            ("The triumph was exalted", "The downfall was ignominious"),
            ("He exhibited steadfast devotion", "He exhibited fickle abandonment"),
            ("The oasis was refreshing", "The desert was desolate"),
            ("She offered gentle consolation", "She offered harsh condemnation"),
            ("The sanctuary was peaceful", "The battleground was violent"),
            ("He displayed virtuous restraint", "He displayed wanton excess"),
            ("The garden was flourishing", "The ruin was decaying"),
            ("She expressed heartfelt appreciation", "She expressed venomous spite"),
            ("The fellowship was warm", "The isolation was cold"),
            ("He showed benevolent kindness", "He showed malevolent cruelty"),
            ("The sunrise was glorious", "The nightmare was terrifying"),
            ("She provided steadfast support", "She provided treacherous opposition"),
            ("The blessing was sacred", "The profanation was sacrilegious"),
            ("He manifested pure innocence", "He manifested corrupt guilt"),
            ("The harmony was celestial", "The chaos was infernal"),
            ("She bestowed abundant blessings", "She inflicted terrible curses"),
            ("The salvation was redemptive", "The damnation was condemnatory"),
            ("He demonstrated heroic bravery", "He demonstrated cowardly timidity"),
            ("The paradise was blissful", "The torment was agonizing"),
            ("She channeled divine inspiration", "She channeled demonic possession"),
            ("The light was illuminating", "The darkness was obscuring"),
            ("He practiced righteous justice", "He practiced corrupt injustice"),
            ("The creation was miraculous", "The destruction was catastrophic"),
            ("She embodied saintly purity", "She embodied sinful corruption"),
            ("The ascent was triumphant", "The descent was calamitous"),
            ("He pursued noble aspirations", "He pursued base desires"),
            ("The spring was renewing", "The winter was withering"),
            ("She demonstrated loving compassion", "She demonstrated hateful malice"),
            ("The victory was complete", "The defeat was total"),
            ("He maintained steadfast faith", "He maintained cynical doubt"),
            ("The bloom was radiant", "The wilt was dismal"),
            ("She offered selfless sacrifice", "She offered selfish exploitation"),
            ("The sanctuary was holy", "The profanation was unholy"),
            ("He exhibited dignified honor", "He exhibited disgraceful dishonor"),
            ("The blessing was merciful", "The judgment was merciless"),
            ("She pursued virtuous excellence", "She pursued wicked debauchery"),
            ("The dawn was hopeful", "The twilight was despairing"),
            ("He demonstrated faithful loyalty", "He demonstrated treacherous betrayal"),
            ("The rebirth was glorious", "The demise was tragic"),
            ("She conveyed joyous celebration", "She conveyed mournful lamentation"),
            ("The consecration was sacred", "The desecration was profane"),
            ("He showed steadfast perseverance", "He showed fickle abandonment"),
        ]
    },
    'semantic_topic': {
        'type': 'SEM',
        'pairs': [
            ("The doctor examined the patient", "The chef prepared the meal"),
            ("The scientist conducted the experiment", "The artist painted the canvas"),
            ("The programmer wrote the code", "The musician composed the melody"),
            ("The engineer designed the bridge", "The farmer harvested the crops"),
            ("The teacher explained the theory", "The athlete trained for the marathon"),
            ("The lawyer argued the case", "The baker kneaded the dough"),
            ("The nurse cared for the sick", "The carpenter built the cabinet"),
            ("The pilot flew the airplane", "The fisherman cast the net"),
            ("The mechanic fixed the engine", "The tailor sewed the garment"),
            ("The architect planned the building", "The gardener planted the flowers"),
            ("The judge delivered the verdict", "The painter mixed the colors"),
            ("The soldier marched forward", "The dancer practiced the routine"),
            ("The banker approved the loan", "The sculptor chiseled the stone"),
            ("The reporter covered the story", "The poet wrote the verse"),
            ("The professor published the paper", "The actor rehearsed the scene"),
            ("The manager led the meeting", "The florist arranged the bouquet"),
            ("The accountant balanced the books", "The potter shaped the clay"),
            ("The detective investigated the crime", "The weaver threaded the loom"),
            ("The surgeon performed the operation", "The brewer fermented the beer"),
            ("The pharmacist dispensed the medicine", "The jeweler polished the diamond"),
            ("The librarian catalogued the books", "The chef seasoned the soup"),
            ("The dentist cleaned the teeth", "The barber trimmed the hair"),
            ("The electrician wired the house", "The plumber fixed the pipe"),
            ("The geologist studied the rocks", "The botanist examined the plants"),
            ("The historian researched the period", "The chemist analyzed the compound"),
            ("The psychologist counseled the patient", "The veterinarian treated the animal"),
            ("The economist forecast the market", "The meteorologist predicted the weather"),
            ("The philosopher pondered existence", "The theologian interpreted scripture"),
            ("The sociologist studied the community", "The anthropologist observed the culture"),
            ("The linguist analyzed the grammar", "The mathematician proved the theorem"),
            ("The biologist examined the cell", "The physicist measured the force"),
            ("The astronomer observed the stars", "The oceanographer mapped the currents"),
            ("The ecologist monitored the habitat", "The geneticist sequenced the DNA"),
            ("The neurologist studied the brain", "The cardiologist examined the heart"),
            ("The dermatologist treated the skin", "The orthopedist set the bone"),
            ("The pediatrician examined the child", "The geriatrician cared for the elderly"),
            ("The optometrist checked the vision", "The audiologist tested the hearing"),
            ("The radiologist read the scan", "The pathologist examined the tissue"),
            ("The anesthesiologist administered the drugs", "The physiotherapist massaged the muscle"),
            ("The coach trained the team", "The conductor led the orchestra"),
            ("The mayor governed the city", "The captain steered the ship"),
            ("The principal ran the school", "The curator managed the museum"),
            ("The sheriff protected the town", "The shepherd guided the flock"),
            ("The admiral commanded the fleet", "The maestro directed the symphony"),
            ("The ambassador represented the nation", "The merchant traded the goods"),
            ("The CEO directed the company", "The abbot led the monastery"),
            ("The general commanded the troops", "The dean administered the college"),
            ("The governor ruled the state", "The warden managed the prison"),
            ("The commissioner oversaw the department", "The provost governed the university"),
            ("The chairman led the board", "The prior headed the convent"),
            ("The director managed the agency", "The rector governed the parish"),
            ("The inspector examined the facility", "The ranger patrolled the forest"),
            ("The auditor reviewed the accounts", "The referee judged the match"),
            ("The examiner tested the students", "The appraiser valued the property"),
            ("The analyst studied the data", "The critic reviewed the film"),
            ("The consultant advised the client", "The counselor guided the student"),
            ("The mediator resolved the conflict", "The arbitrator settled the dispute"),
            ("The facilitator guided the workshop", "The mentor coached the protege"),
            ("The trainer taught the course", "The tutor helped the learner"),
            ("The instructor demonstrated the technique", "The coach refined the strategy"),
            ("The guide showed the way", "The leader inspired the group"),
            ("The pioneer explored the territory", "The visionary imagined the future"),
            ("The innovator created the product", "The inventor designed the device"),
            ("The researcher discovered the pattern", "The scholar wrote the treatise"),
            ("The expert provided the analysis", "The specialist handled the case"),
            ("The technician calibrated the instrument", "The operator ran the machine"),
            ("The engineer optimized the process", "The designer improved the layout"),
            ("The developer built the application", "The programmer implemented the algorithm"),
            ("The architect structured the system", "The planner organized the project"),
            ("The strategist formulated the plan", "The coordinator managed the schedule"),
            ("The supervisor monitored the work", "The overseer directed the labor"),
            ("The foreman organized the crew", "The manager allocated the resources"),
            ("The executive made the decision", "The administrator processed the request"),
            ("The clerk filed the document", "The registrar recorded the entry"),
            ("The secretary prepared the agenda", "The receptionist handled the call"),
            ("The assistant helped the boss", "The deputy represented the official"),
            ("The intern learned the trade", "The apprentice mastered the craft"),
            ("The volunteer helped the cause", "The activist championed the movement"),
            ("The soldier patrolled the border", "The chef prepared the feast"),
            ("The tailor hemmed the pants", "The farmer planted the seeds"),
            ("The baker kneaded the bread", "The sailor navigated the ship"),
            ("The painter brushed the canvas", "The driver delivered the mail"),
            ("The singer performed the aria", "The builder laid the bricks"),
            ("The dancer leaped across the stage", "The welder joined the metal"),
            ("The author drafted the manuscript", "The mechanic tuned the engine"),
            ("The poet composed the sonnet", "The pilot flew the helicopter"),
            ("The sculptor carved the marble", "The programmer debugged the code"),
            ("The actor rehearsed the monologue", "The engineer tested the bridge"),
            ("The musician tuned the violin", "The doctor diagnosed the illness"),
            ("The florist arranged the roses", "The lawyer filed the motion"),
            ("The plumber installed the faucet", "The teacher graded the essay"),
            ("The carpenter sanded the wood", "The nurse administered the injection"),
            ("The barber trimmed the beard", "The scientist collected the samples"),
            ("The electrician replaced the fuse", "The journalist wrote the article"),
            ("The welder sealed the joint", "The pharmacist filled the prescription"),
            ("The mason laid the stones", "The judge heard the appeal"),
            ("The tailor stitched the hem", "The dentist filled the cavity"),
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
            ("The government approved the law", "The law was approved by the government"),
            ("The team won the championship", "The championship was won by the team"),
            ("The artist created the sculpture", "The sculpture was created by the artist"),
            ("The scientist discovered the cure", "The cure was discovered by the scientist"),
            ("The author published the book", "The book was published by the author"),
            ("The committee rejected the proposal", "The proposal was rejected by the committee"),
            ("The court upheld the verdict", "The verdict was upheld by the court"),
            ("The council approved the budget", "The budget was approved by the council"),
            ("The king conquered the territory", "The territory was conquered by the king"),
            ("The general defeated the enemy", "The enemy was defeated by the general"),
            ("The player scored the goal", "The goal was scored by the player"),
            ("The singer performed the song", "The song was performed by the singer"),
            ("The director filmed the movie", "The movie was filmed by the director"),
            ("The engineer built the machine", "The machine was built by the engineer"),
            ("The designer created the logo", "The logo was created by the designer"),
            ("The researcher found the solution", "The solution was found by the researcher"),
            ("The manager hired the employee", "The employee was hired by the manager"),
            ("The president signed the treaty", "The treaty was signed by the president"),
            ("The editor published the article", "The article was published by the editor"),
            ("The farmer grew the crops", "The crops were grown by the farmer"),
            ("The baker made the bread", "The bread was made by the baker"),
            ("The mechanic repaired the car", "The car was repaired by the mechanic"),
            ("The nurse treated the wound", "The wound was treated by the nurse"),
            ("The police caught the thief", "The thief was caught by the police"),
            ("The driver delivered the pizza", "The pizza was delivered by the driver"),
            ("The waiter served the dessert", "The dessert was served by the waiter"),
            ("The cleaner washed the windows", "The windows were washed by the cleaner"),
            ("The builder constructed the wall", "The wall was constructed by the builder"),
            ("The painter decorated the room", "The room was decorated by the painter"),
            ("The plumber fixed the sink", "The sink was fixed by the plumber"),
            ("The electrician installed the light", "The light was installed by the electrician"),
            ("The gardener pruned the hedge", "The hedge was pruned by the gardener"),
            ("The tailor altered the suit", "The suit was altered by the tailor"),
            ("The cat caught the fish", "The fish was caught by the cat"),
            ("The boy kicked the ball", "The ball was kicked by the boy"),
            ("The rain ruined the picnic", "The picnic was ruined by the rain"),
            ("The girl broke the vase", "The vase was broken by the girl"),
            ("The man opened the door", "The door was opened by the man"),
            ("The woman closed the window", "The window was closed by the woman"),
            ("The child dropped the glass", "The glass was dropped by the child"),
            ("The dog found the bone", "The bone was found by the dog"),
            ("The bird built the nest", "The nest was built by the bird"),
            ("The horse jumped the fence", "The fence was jumped by the horse"),
            ("The cow produced the milk", "The milk was produced by the cow"),
            ("The bee made the honey", "The honey was made by the bee"),
            ("The spider spun the web", "The web was spun by the spider"),
            ("The ant carried the leaf", "The leaf was carried by the ant"),
            ("The worm turned the soil", "The soil was turned by the worm"),
            ("The rabbit dug the burrow", "The burrow was dug by the rabbit"),
            ("The fox stole the chicken", "The chicken was stolen by the fox"),
            ("The wolf attacked the sheep", "The sheep was attacked by the wolf"),
            ("The bear caught the salmon", "The salmon was caught by the bear"),
            ("The eagle hunted the rabbit", "The rabbit was hunted by the eagle"),
            ("The snake bit the frog", "The frog was bitten by the snake"),
            ("The lion chased the gazelle", "The gazelle was chased by the lion"),
            ("The tiger attacked the deer", "The deer was attacked by the tiger"),
            ("The shark bit the surfer", "The surfer was bitten by the shark"),
            ("The whale swallowed the fish", "The fish was swallowed by the whale"),
            ("The dolphin rescued the swimmer", "The swimmer was rescued by the dolphin"),
            ("The monkey picked the banana", "The banana was picked by the monkey"),
            ("The elephant drank the water", "The water was drunk by the elephant"),
            ("The giraffe ate the leaves", "The leaves were eaten by the giraffe"),
            ("The penguin caught the fish", "The fish was caught by the penguin"),
            ("The owl hunted the mouse", "The mouse was hunted by the owl"),
            ("The bat caught the insect", "The insect was caught by the bat"),
            ("The frog swallowed the fly", "The fly was swallowed by the frog"),
            ("The turtle laid the eggs", "The eggs were laid by the turtle"),
            ("The crocodile attacked the zebra", "The zebra was attacked by the crocodile"),
            ("The gorilla beat the chest", "The chest was beaten by the gorilla"),
            ("The parrot repeated the word", "The word was repeated by the parrot"),
            ("The dolphin jumped the wave", "The wave was jumped by the dolphin"),
            ("The cheetah chased the antelope", "The antelope was chased by the cheetah"),
            ("The hawk hunted the squirrel", "The squirrel was hunted by the hawk"),
            ("The crane lifted the steel", "The steel was lifted by the crane"),
            ("The tractor pulled the plow", "The plow was pulled by the tractor"),
            ("The wind blew the leaves", "The leaves were blown by the wind"),
            ("The sun melted the ice", "The ice was melted by the sun"),
            ("The river carried the boat", "The boat was carried by the river"),
            ("The earthquake shook the city", "The city was shaken by the earthquake"),
            ("The volcano erupted lava", "Lava was erupted by the volcano"),
            ("The storm sank the ship", "The ship was sunk by the storm"),
            ("The flood destroyed the village", "The village was destroyed by the flood"),
            ("The fire burned the forest", "The forest was burned by the fire"),
            ("The snow covered the mountain", "The mountain was covered by the snow"),
            ("The tide washed the sand", "The sand was washed by the tide"),
            ("The frost damaged the crops", "The crops were damaged by the frost"),
            ("The drought killed the plants", "The plants were killed by the drought"),
            ("The hurricane destroyed the pier", "The pier was destroyed by the hurricane"),
            ("The tornado lifted the roof", "The roof was lifted by the tornado"),
            ("The glacier carved the valley", "The valley was carved by the glacier"),
            ("The lightning struck the tree", "The tree was struck by the lightning"),
            ("The avalanche buried the cabin", "The cabin was buried by the avalanche"),
        ]
    },
}

# 限制对数
for feat in FEATURE_PAIRS_BASE:
    FEATURE_PAIRS_BASE[feat]['pairs'] = FEATURE_PAIRS_BASE[feat]['pairs'][:100]

FEATURE_PAIRS = FEATURE_PAIRS_BASE

# ============================================================
# 模型配置
# ============================================================
MODEL_CONFIGS = {
    'deepseek7b': {
        'path': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'name': 'DeepSeek-R1-Distill-Qwen-7B',
        'bottleneck_layer': 4,
    },
    'qwen3': {
        'path': 'D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',
        'name': 'Qwen3-4B',
        'bottleneck_layer': 6,
    },
    'glm4': {
        'path': 'D:/develop/model/hub/models--zai-org--glm-4-9b-chat-hf/download',
        'name': 'GLM4-9B-Chat',
        'bottleneck_layer': 30,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "deepseek7b", "glm4"])
    parser.add_argument("--n_pairs", type=int, default=100)
    args = parser.parse_args()
    
    model_name = args.model
    n_pairs_max = args.n_pairs
    cfg = MODEL_CONFIGS[model_name]
    bottleneck_layer = cfg['bottleneck_layer']
    
    out_dir = f"results/causal_fiber/{model_name}_ccxix"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = open(f"{out_dir}/run.log", "w", encoding="utf-8")
    
    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()
    
    log(f"Phase CCXIX: 因果干预实验 — 瓶颈层PC1因果效应验证")
    log(f"Model: {cfg['name']}, bottleneck_layer={bottleneck_layer}, n_pairs={n_pairs_max}")
    
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
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    
    log(f"n_layers={n_layers}, d_model={d_model}, device={device}")
    
    feature_names = list(FEATURE_PAIRS.keys())
    
    # ============================================================
    # S1: 收集瓶颈层+所有层的差分向量
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S1: 收集瓶颈层L{bottleneck_layer}+关键层的差分向量")
    log(f"{'='*60}")
    
    # 采样层: 瓶颈层 + 每4层 + 首尾层
    layer_indices = sorted(set([0, bottleneck_layer, n_layers-1] + list(range(0, n_layers, 4))))
    if (n_layers - 1) not in layer_indices:
        layer_indices.append(n_layers - 1)
    layer_indices = sorted(layer_indices)
    log(f"Sampled layers ({len(layer_indices)}): {layer_indices}")
    
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
    for l in layer_indices:
        h = model.model.layers[l].register_forward_hook(make_hook(l, cap_store))
        handles.append(h)
    
    all_deltas = {feat: {l: [] for l in layer_indices} for feat in feature_names}
    
    for feat_idx, feat in enumerate(feature_names):
        pairs = FEATURE_PAIRS[feat]['pairs'][:n_pairs_max]
        log(f"  [{feat_idx+1}/{len(feature_names)}] {feat} ({FEATURE_PAIRS[feat]['type']}): {len(pairs)} pairs")
        
        for s1, s2 in pairs:
            with torch.no_grad():
                enc1 = tokenizer(s1, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc1 = {k: v.to(device) for k, v in enc1.items()}
                _ = model(**enc1)
                len1 = enc1['attention_mask'].sum().item()
                idx1 = max(0, len1 - 2)
                
                s1_acts = {}
                for l in layer_indices:
                    if l in cap_store:
                        s1_acts[l] = cap_store[l][0, idx1].float().numpy()
                
                enc2 = tokenizer(s2, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc2 = {k: v.to(device) for k, v in enc2.items()}
                _ = model(**enc2)
                len2 = enc2['attention_mask'].sum().item()
                idx2 = max(0, len2 - 2)
                
                for l in layer_indices:
                    if l in cap_store and l in s1_acts:
                        s2_vec = cap_store[l][0, idx2].float().numpy()
                        delta = s2_vec - s1_acts[l]
                        all_deltas[feat][l].append(delta)
                
                if model_name in ["glm4", "deepseek7b"]:
                    torch.cuda.empty_cache()
        
        n_valid = len(all_deltas[feat][bottleneck_layer])
        log(f"    Collected {n_valid} deltas at bottleneck layer")
    
    for h in handles:
        h.remove()
    
    # ============================================================
    # S2: 全局PCA + Per-feature PCA
    # ============================================================
    log(f"\n{'='*60}")
    log("S2: PCA分析 — 全局 + Per-feature")
    log(f"{'='*60}")
    
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    
    syn_features = [f for f in feature_names if FEATURE_PAIRS[f]['type'] == 'SYN']
    sem_features = [f for f in feature_names if FEATURE_PAIRS[f]['type'] == 'SEM']
    
    # 瓶颈层全局PCA
    all_bn_deltas = np.concatenate([all_deltas[f][bottleneck_layer] for f in feature_names 
                                     if len(all_deltas[f][bottleneck_layer]) > 0], axis=0)
    log(f"  Bottleneck layer deltas: {all_bn_deltas.shape}")
    
    pca_global = PCA(n_components=10)
    pca_global.fit(all_bn_deltas)
    pc_dirs = pca_global.components_  # (10, d_model)
    pc_vars = pca_global.explained_variance_ratio_
    
    log(f"  Global PC variance (top 10): {np.round(pc_vars, 4)}")
    for i in range(min(5, len(pc_vars))):
        log(f"    PC{i+1}: {pc_vars[i]*100:.2f}%")
    
    # Per-feature PCA (瓶颈层)
    log(f"\n  Per-feature PCA at bottleneck layer:")
    feat_pc1_var = {}
    for feat in feature_names:
        deltas = np.array(all_deltas[feat][bottleneck_layer])
        if len(deltas) < 5:
            continue
        pca_feat = PCA(n_components=5)
        pca_feat.fit(deltas)
        feat_pc1_var[feat] = pca_feat.explained_variance_ratio_[0]
        log(f"    {feat}: PC1={pca_feat.explained_variance_ratio_[0]*100:.2f}%, PC2={pca_feat.explained_variance_ratio_[1]*100:.2f}%")
    
    # ============================================================
    # S3: PC投影比例分析 (间接因果证据)
    # ============================================================
    log(f"\n{'='*60}")
    log("S3: PC投影比例分析 — 差分信号在PC1 vs PC2-5上的承载")
    log(f"{'='*60}")
    
    pc1_proj_ratios = {feat: [] for feat in feature_names}
    pc2_5_proj_ratios = {feat: [] for feat in feature_names}
    random_proj_ratios = {feat: [] for feat in feature_names}
    
    for feat in feature_names:
        deltas = all_deltas[feat][bottleneck_layer]
        if not deltas:
            continue
        
        pc1_dir = pc_dirs[0]
        pc2_5_dirs = pc_dirs[1:5]
        
        # 随机方向(归一化)
        np.random.seed(42)
        random_dir = np.random.randn(d_model)
        random_dir = random_dir / np.linalg.norm(random_dir)
        
        for delta in deltas:
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-8:
                continue
            
            # PC1投影比例
            pc1_proj = abs(np.dot(delta, pc1_dir)) / delta_norm
            pc1_proj_ratios[feat].append(pc1_proj)
            
            # PC2-5投影比例(总和)
            pc2_5_proj = sum(abs(np.dot(delta, d)) for d in pc2_5_dirs) / delta_norm
            pc2_5_proj_ratios[feat].append(pc2_5_proj)
            
            # 随机方向投影比例
            rand_proj = abs(np.dot(delta, random_dir)) / delta_norm
            random_proj_ratios[feat].append(rand_proj)
    
    log(f"\n  投影比例汇总:")
    log(f"  {'Feature':<20} {'PC1_ratio':>10} {'PC2-5_ratio':>12} {'Random_ratio':>13} {'PC1/PC2-5_per':>14} {'PC1/Random':>11}")
    log(f"  {'-'*80}")
    
    all_pc1_ratios = []
    all_pc2_5_ratios = []
    all_random_ratios = []
    
    for feat in feature_names:
        if not pc1_proj_ratios[feat]:
            continue
        pc1_mean = np.mean(pc1_proj_ratios[feat])
        pc2_5_mean = np.mean(pc2_5_proj_ratios[feat])
        rand_mean = np.mean(random_proj_ratios[feat])
        
        pc2_5_per = pc2_5_mean / 4 if pc2_5_mean > 0 else 0
        pc1_vs_pc2_5 = pc1_mean / pc2_5_per if pc2_5_per > 0 else float('inf')
        pc1_vs_rand = pc1_mean / rand_mean if rand_mean > 0 else float('inf')
        
        log(f"  {feat:<20} {pc1_mean:>10.4f} {pc2_5_mean:>12.4f} {rand_mean:>13.4f} {pc1_vs_pc2_5:>14.2f}x {pc1_vs_rand:>10.2f}x")
        
        all_pc1_ratios.extend(pc1_proj_ratios[feat])
        all_pc2_5_ratios.extend(pc2_5_proj_ratios[feat])
        all_random_ratios.extend(random_proj_ratios[feat])
    
    if all_pc1_ratios:
        overall_pc1 = np.mean(all_pc1_ratios)
        overall_pc2_5 = np.mean(all_pc2_5_ratios)
        overall_rand = np.mean(all_random_ratios)
        pc2_5_per = overall_pc2_5 / 4
        log(f"\n  === 全局 ===")
        log(f"  PC1投影比: {overall_pc1:.4f}")
        log(f"  PC2-5投影比(总): {overall_pc2_5:.4f}, (per PC): {pc2_5_per:.4f}")
        log(f"  Random投影比: {overall_rand:.4f}")
        log(f"  PC1/PC2-5_per: {overall_pc1/pc2_5_per:.2f}x" if pc2_5_per > 0 else "  PC1/PC2-5_per: inf")
        log(f"  PC1/Random: {overall_pc1/overall_rand:.2f}x" if overall_rand > 0 else "  PC1/Random: inf")
        
        # t检验
        t1, p1 = stats.ttest_ind(all_pc1_ratios, [v/4 for v in all_pc2_5_ratios])
        t2, p2 = stats.ttest_ind(all_pc1_ratios, all_random_ratios)
        log(f"  t-test PC1 vs PC2-5_per: t={t1:.3f}, p={p1:.6f}")
        log(f"  t-test PC1 vs Random: t={t2:.3f}, p={p2:.6f}")
    
    # ============================================================
    # S4: PC1跨特征一致性分析
    # ============================================================
    log(f"\n{'='*60}")
    log("S4: PC1跨特征一致性 — 不同特征的PC1方向是否对齐?")
    log(f"{'='*60}")
    
    feat_pc1_dirs = {}
    for feat in feature_names:
        deltas = np.array(all_deltas[feat][bottleneck_layer])
        if len(deltas) < 5:
            continue
        pca_feat = PCA(n_components=1)
        pca_feat.fit(deltas)
        feat_pc1_dirs[feat] = pca_feat.components_[0]
    
    # 计算所有特征对之间的PC1余弦相似度
    feat_names_with_pc1 = list(feat_pc1_dirs.keys())
    log(f"  Features with PC1: {len(feat_names_with_pc1)}")
    
    syn_syn_cos = []
    sem_sem_cos = []
    syn_sem_cos = []
    
    for i, f1 in enumerate(feat_names_with_pc1):
        for j, f2 in enumerate(feat_names_with_pc1):
            if i >= j:
                continue
            cos_val = abs(np.dot(feat_pc1_dirs[f1], feat_pc1_dirs[f2]))
            t1 = FEATURE_PAIRS[f1]['type']
            t2 = FEATURE_PAIRS[f2]['type']
            if t1 == 'SYN' and t2 == 'SYN':
                syn_syn_cos.append(cos_val)
            elif t1 == 'SEM' and t2 == 'SEM':
                sem_sem_cos.append(cos_val)
            else:
                syn_sem_cos.append(cos_val)
    
    log(f"  PC1方向对齐度 (|cos|):")
    log(f"    SYN-SYN: mean={np.mean(syn_syn_cos):.4f}, std={np.std(syn_syn_cos):.4f}, n={len(syn_syn_cos)}")
    log(f"    SEM-SEM: mean={np.mean(sem_sem_cos):.4f}, std={np.std(sem_sem_cos):.4f}, n={len(sem_sem_cos)}")
    log(f"    SYN-SEM: mean={np.mean(syn_sem_cos):.4f}, std={np.std(syn_sem_cos):.4f}, n={len(syn_sem_cos)}")
    
    # 各特征的PC1与全局PC1的对齐度
    global_pc1 = pc_dirs[0]
    log(f"\n  各特征PC1与全局PC1的对齐度:")
    for feat in feat_names_with_pc1:
        cos_val = abs(np.dot(feat_pc1_dirs[feat], global_pc1))
        log(f"    {feat}: |cos|={cos_val:.4f}")
    
    # ============================================================
    # S5: 残差分析 — 消除PC1后的可分性
    # ============================================================
    log(f"\n{'='*60}")
    log("S5: 残差分析 — 消除PC1后, 差分信号是否仍可区分语法/语义?")
    log(f"{'='*60}")
    
    # 对每对特征, 计算PC1消除前后的可分性
    # 可分性指标: logistic regression准确率
    
    pc1_dir_np = pc_dirs[0]
    
    for feat in feature_names[:6]:  # 测试前6个特征
        deltas = np.array(all_deltas[feat][bottleneck_layer])
        if len(deltas) < 10:
            continue
        
        # 原始差分: 计算每对差分的L2范数(代表信号强度)
        orig_norms = np.linalg.norm(deltas, axis=1)
        
        # 消除PC1后的残差
        pc1_components = np.outer(deltas @ pc1_dir_np, pc1_dir_np)
        residuals = deltas - pc1_components
        residual_norms = np.linalg.norm(residuals, axis=1)
        
        # 信号保留比
        retention_ratio = residual_norms / (orig_norms + 1e-10)
        
        log(f"  {feat}:")
        log(f"    原始范数: mean={np.mean(orig_norms):.4f}, std={np.std(orig_norms):.4f}")
        log(f"    残差范数: mean={np.mean(residual_norms):.4f}, std={np.std(residual_norms):.4f}")
        log(f"    信号保留比: mean={np.mean(retention_ratio):.4f} ({np.mean(retention_ratio)*100:.1f}%保留)")
        log(f"    PC1解释方差: {(1-np.mean(retention_ratio**2))*100:.1f}%")
    
    # 语法vs语义可分性(使用logistic regression)
    log(f"\n  语法/语义可分性测试 (logistic regression):")
    
    # 准备数据: 语法差分 vs 语义差分
    syn_deltas_bn = np.concatenate([all_deltas[f][bottleneck_layer] for f in syn_features 
                                     if len(all_deltas[f][bottleneck_layer]) > 0], axis=0)
    sem_deltas_bn = np.concatenate([all_deltas[f][bottleneck_layer] for f in sem_features 
                                     if len(all_deltas[f][bottleneck_layer]) > 0], axis=0)
    
    if len(syn_deltas_bn) > 10 and len(sem_deltas_bn) > 10:
        # 原始
        X_orig = np.concatenate([syn_deltas_bn, sem_deltas_bn], axis=0)
        y = np.concatenate([np.zeros(len(syn_deltas_bn)), np.ones(len(sem_deltas_bn))])
        
        # 消除PC1
        pc1_comps = np.outer(X_orig @ pc1_dir_np, pc1_dir_np)
        X_no_pc1 = X_orig - pc1_comps
        
        # 只保留PC1
        X_only_pc1 = pc1_comps
        
        # 消除PC1-5
        pc1_5_comps = np.zeros_like(X_orig)
        for i in range(min(5, len(pc_dirs))):
            pc1_5_comps += np.outer(X_orig @ pc_dirs[i], pc_dirs[i])
        X_no_pc1_5 = X_orig - pc1_5_comps
        
        # 5-fold交叉验证
        from sklearn.model_selection import cross_val_score
        
        for name, X_test in [("Original", X_orig), ("No PC1", X_no_pc1), 
                              ("Only PC1", X_only_pc1), ("No PC1-5", X_no_pc1_5)]:
            clf = LogisticRegression(max_iter=1000, C=1.0)
            scores = cross_val_score(clf, X_test, y, cv=5, scoring='accuracy')
            log(f"    {name}: accuracy={np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # ============================================================
    # S6: Hook干预 — PC1消藻对logit的因果效应
    # ============================================================
    log(f"\n{'='*60}")
    log("S6: Hook干预 — PC1消藻对输出logit的因果效应")
    log(f"{'='*60}")
    
    # 安全的hook干预方式: 修改hidden_states而非output
    # 使用register_forward_hook在层输出后修改
    
    n_hook_test = 15
    hook_features = feature_names[:6]
    
    pc1_dir_t = torch.tensor(pc_dirs[0], dtype=torch.float32)
    pc2_5_dirs_t = torch.tensor(pc_dirs[1:5], dtype=torch.float32) if len(pc_dirs) >= 5 else None
    
    hook_results = {
        'baseline_logit_diff': {feat: [] for feat in hook_features},
        'pc1_ablate_logit_diff': {feat: [] for feat in hook_features},
        'pc2_5_ablate_logit_diff': {feat: [] for feat in hook_features},
    }
    
    for feat_idx, feat in enumerate(hook_features):
        pairs = FEATURE_PAIRS[feat]['pairs'][:n_hook_test]
        log(f"  [{feat_idx+1}/{len(hook_features)}] {feat}: {len(pairs)} pairs for hook test")
        
        for s1, s2 in pairs:
            # --- Baseline: 获取s1和s2的logit差 ---
            with torch.no_grad():
                enc1 = tokenizer(s1, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc1 = {k: v.to(device) for k, v in enc1.items()}
                out1 = model(**enc1)
                logits_s1 = out1.logits[0, -1, :].float().cpu().numpy()
                
                enc2 = tokenizer(s2, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc2 = {k: v.to(device) for k, v in enc2.items()}
                out2 = model(**enc2)
                logits_s2 = out2.logits[0, -1, :].float().cpu().numpy()
            
            baseline_diff = np.linalg.norm(logits_s2 - logits_s1)
            if baseline_diff < 1e-6:
                continue
            
            # --- PC1消融干预(s1) ---
            # 使用model.generate方式不行, 需要直接在forward中修改
            # 安全方式: 创建一个wrapper在forward后修改hidden_states
            
            try:
                def make_ablation_hook(pc_dir_tensor, n_pcs=1):
                    """创建消融hook, 移除指定PC方向的分量"""
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            hidden = output[0]
                        else:
                            hidden = output
                        
                        # 只修改最后一个token位置
                        seq_len = hidden.shape[1]
                        idx = seq_len - 2 if seq_len > 2 else 0
                        
                        # 获取原始激活
                        act = hidden[0, idx].float()
                        
                        # 移除PC分量
                        if n_pcs == 1:
                            proj = torch.dot(act, pc_dir_tensor.to(act.device)) * pc_dir_tensor.to(act.device)
                            new_act = act - proj
                        else:
                            proj = torch.zeros_like(act)
                            for i in range(min(n_pcs, pc_dir_tensor.shape[0])):
                                d = pc_dir_tensor[i].to(act.device)
                                proj += torch.dot(act, d) * d
                            new_act = act - proj
                        
                        # 创建新tensor (避免in-place修改)
                        new_hidden = hidden.clone()
                        new_hidden[0, idx] = new_act.to(hidden.dtype)
                        
                        if isinstance(output, tuple):
                            return (new_hidden,) + output[1:]
                        return new_hidden
                    return hook_fn
                
                # PC1消融
                with torch.no_grad():
                    h1 = model.model.layers[bottleneck_layer].register_forward_hook(
                        make_ablation_hook(pc1_dir_t, n_pcs=1)
                    )
                    out_int1 = model(**enc1)
                    logits_pc1_ablated = out_int1.logits[0, -1, :].float().cpu().numpy()
                    h1.remove()
                
                # PC2-5消融
                if pc2_5_dirs_t is not None:
                    with torch.no_grad():
                        h2 = model.model.layers[bottleneck_layer].register_forward_hook(
                            make_ablation_hook(pc2_5_dirs_t, n_pcs=4)
                        )
                        out_int2 = model(**enc1)
                        logits_pc2_5_ablated = out_int2.logits[0, -1, :].float().cpu().numpy()
                        h2.remove()
                else:
                    logits_pc2_5_ablated = logits_s1  # fallback
                
                pc1_effect = np.linalg.norm(logits_pc1_ablated - logits_s1)
                pc2_5_effect = np.linalg.norm(logits_pc2_5_ablated - logits_s1)
                
                hook_results['baseline_logit_diff'][feat].append(baseline_diff)
                hook_results['pc1_ablate_logit_diff'][feat].append(pc1_effect)
                hook_results['pc2_5_ablate_logit_diff'][feat].append(pc2_5_effect)
                
            except Exception as e:
                log(f"    Hook failed for pair: {e}")
                continue
            
            if model_name in ["glm4", "deepseek7b"]:
                torch.cuda.empty_cache()
    
    # Hook结果汇总
    log(f"\n  Hook干预结果汇总:")
    log(f"  {'Feature':<20} {'Baseline':>10} {'PC1_ablate':>11} {'PC2-5_ablate':>13} {'Causal_ratio':>13}")
    log(f"  {'-'*70}")
    
    all_pc1_effects = []
    all_pc2_5_effects = []
    all_baselines_h = []
    
    for feat in hook_features:
        bl = hook_results['baseline_logit_diff'][feat]
        pc1 = hook_results['pc1_ablate_logit_diff'][feat]
        pc2_5 = hook_results['pc2_5_ablate_logit_diff'][feat]
        
        if bl and pc1 and pc2_5:
            bl_m = np.mean(bl)
            pc1_m = np.mean(pc1)
            pc2_5_m = np.mean(pc2_5)
            pc2_5_per = pc2_5_m / 4
            causal_r = pc1_m / pc2_5_per if pc2_5_per > 0 else float('inf')
            
            log(f"  {feat:<20} {bl_m:>10.4f} {pc1_m:>11.4f} {pc2_5_m:>13.4f} {causal_r:>12.2f}x")
            
            all_pc1_effects.extend(pc1)
            all_pc2_5_effects.extend(pc2_5)
            all_baselines_h.extend(bl)
    
    if all_pc1_effects:
        mean_pc1 = np.mean(all_pc1_effects)
        mean_pc2_5 = np.mean(all_pc2_5_effects)
        mean_bl = np.mean(all_baselines_h)
        pc2_5_per = mean_pc2_5 / 4
        causal_r = mean_pc1 / pc2_5_per if pc2_5_per > 0 else float('inf')
        
        # 相对于baseline的比例
        pc1_pct = mean_pc1 / mean_bl * 100 if mean_bl > 0 else 0
        pc2_5_pct = mean_pc2_5 / mean_bl * 100 if mean_bl > 0 else 0
        
        t_stat, p_val = stats.ttest_ind(all_pc1_effects, [v/4 for v in all_pc2_5_effects])
        
        log(f"\n  === 全局因果效应 ===")
        log(f"  Baseline logit diff: {mean_bl:.4f}")
        log(f"  PC1 ablation effect: {mean_pc1:.4f} ({pc1_pct:.1f}% of baseline)")
        log(f"  PC2-5 ablation effect: {mean_pc2_5:.4f} ({pc2_5_pct:.1f}% of baseline)")
        log(f"  PC2-5 per PC: {pc2_5_per:.4f}")
        log(f"  Causal ratio (PC1/PC2-5_per): {causal_r:.2f}x")
        log(f"  t-test: t={t_stat:.3f}, p={p_val:.6f}")
        
        if p_val < 0.01:
            log(f"  *** PC1因果效应显著大于PC2-5 (p<0.01) ***")
        elif p_val < 0.05:
            log(f"  ** PC1因果效应显著大于PC2-5 (p<0.05) **")
        else:
            log(f"  * PC1因果效应不显著大于PC2-5 (p>0.05) *")
    
    # ============================================================
    # 保存结果
    # ============================================================
    results = {
        'model': cfg['name'],
        'bottleneck_layer': bottleneck_layer,
        'global_pc_variance': pc_vars.tolist(),
        'per_feature_pc1_variance': {f: float(v) for f, v in feat_pc1_var.items()},
        'projection_ratios': {
            'pc1_mean': float(np.mean(all_pc1_ratios)) if all_pc1_ratios else 0,
            'pc2_5_mean': float(np.mean(all_pc2_5_ratios)) if all_pc2_5_ratios else 0,
            'random_mean': float(np.mean(all_random_ratios)) if all_random_ratios else 0,
        },
        'pc1_alignment': {
            'syn_syn': float(np.mean(syn_syn_cos)) if syn_syn_cos else 0,
            'sem_sem': float(np.mean(sem_sem_cos)) if sem_sem_cos else 0,
            'syn_sem': float(np.mean(syn_sem_cos)) if syn_sem_cos else 0,
        },
        'hook_intervention': {
            'pc1_ablation_effect': float(np.mean(all_pc1_effects)) if all_pc1_effects else 0,
            'pc2_5_ablation_effect': float(np.mean(all_pc2_5_effects)) if all_pc2_5_effects else 0,
            'causal_ratio': float(causal_r) if all_pc1_effects and all_pc2_5_effects else 0,
            'p_value': float(p_val) if all_pc1_effects and all_pc2_5_effects else 1.0,
        }
    }
    
    with open(f"{out_dir}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    np.save(f"{out_dir}/pc_directions.npy", pc_dirs)
    
    log(f"\nResults saved to {out_dir}/")
    log(f"Phase CCXIX complete!")
    
    log_file.close()


if __name__ == "__main__":
    main()
