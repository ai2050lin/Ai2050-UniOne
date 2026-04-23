"""
Phase CCXXII: 全层因果扫描 — 1D流形的层间演化
================================================================
核心目标:
1. 在每一层做per-feature PCA + 差分投影比 (PC1/Random)
2. 在每一层计算跨特征PC1对齐度
3. 绘制PC1因果效应的"层曲线", 找到峰值层
4. 追踪1D流形从输入到输出的演化路径

关键问题:
  - DS7B的1D流形只在L4存在, 还是在所有层都有?
  - PC1因果效应的峰值层在哪里?
  - 1D流形方向是否随层旋转? (PC1方向的层间对齐度)

样本量: 120对/特征 (平衡速度与统计可靠性)
特征: tense, polarity, voice, semantic_valence, semantic_topic (排除formality)
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

# 复用特征对定义 (排除formality以避免方差陷阱)
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
            ("The cat chases the mouse", "The cat chased the mouse"),
            ("She cuts the paper carefully", "She cut the paper carefully"),
            ("The boy throws the stone", "The boy threw the stone"),
            ("The dog digs a hole", "The dog dug a hole"),
            ("We ride the bus to town", "We rode the bus to town"),
            ("The girl feeds the cat", "The girl fed the cat"),
            ("He sits by the window", "He sat by the window"),
            ("The ice melts in the sun", "The ice melted in the sun"),
            ("The rain falls heavily", "The rain fell heavily"),
            ("The bird flies over the house", "The bird flew over the house"),
            ("He carries the box inside", "He carried the box inside"),
            ("The plant grows very tall", "The plant grew very tall"),
            ("She tells a funny story", "She told a funny story"),
            ("The boat sails across the lake", "The boat sailed across the lake"),
            ("He finds the lost key", "He found the lost key"),
            ("The bell chimes every hour", "The bell chimed every hour"),
            ("The cake bakes in the oven", "The cake baked in the oven"),
            ("She keeps the door open", "She kept the door open"),
            ("The light shines through the window", "The light shone through the window"),
            ("The man drives a taxi", "The man drove a taxi"),
            ("She sends a letter home", "She sent a letter home"),
            ("The girl holds the baby", "The girl held the baby"),
            ("The fog covers the valley", "The fog covered the valley"),
            ("He meets his friend at noon", "He met his friend at noon"),
            ("The snow falls softly", "The snow fell softly"),
            ("She chooses the blue dress", "She chose the blue dress"),
            ("The dog catches the frisbee", "The dog caught the frisbee"),
            ("He teaches the young students", "He taught the young students"),
            ("The plane flies above the clouds", "The plane flew above the clouds"),
            ("She feels the cold wind", "She felt the cold wind"),
            ("The ship crosses the ocean", "The ship crossed the ocean"),
            ("He leaves the room quietly", "He left the room quietly"),
            ("The dog hides under the bed", "The dog hid under the bed"),
            ("She brings fresh flowers", "She brought fresh flowers"),
            ("The rain stops suddenly", "The rain stopped suddenly"),
            ("He loses his wallet", "He lost his wallet"),
            ("The children wake up early", "The children woke up early"),
            ("She stands by the door", "She stood by the door"),
            ("The car stops at the light", "The car stopped at the light"),
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
            ("The door was open", "The door was not open"),
            ("She will attend the meeting", "She will not attend the meeting"),
            ("He has seen the movie", "He has not seen the movie"),
            ("The train is on time", "The train is not on time"),
            ("We need more time", "We do not need more time"),
            ("The answer is correct", "The answer is not correct"),
            ("She knows the password", "She does not know the password"),
            ("He owns a car", "He does not own a car"),
            ("The shop is open today", "The shop is not open today"),
            ("They speak French", "They do not speak French"),
            ("I believe your story", "I do not believe your story"),
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
            ("The director filmed the movie", "The movie was filmed by the director"),
            ("The engineer built the machine", "The machine was built by the engineer"),
            ("The designer created the logo", "The logo was created by the designer"),
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
            ("The builder constructed the wall", "The wall was constructed by the builder"),
            ("The painter decorated the room", "The room was decorated by the painter"),
            ("The plumber fixed the sink", "The sink was fixed by the plumber"),
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
            ("The rabbit dug the burrow", "The burrow was dug by the rabbit"),
            ("The fox stole the chicken", "The chicken was stolen by the fox"),
            ("The wolf attacked the sheep", "The sheep was attacked by the wolf"),
            ("The bear caught the salmon", "The salmon was caught by the bear"),
            ("The eagle hunted the rabbit", "The rabbit was hunted by the eagle"),
            ("The snake bit the frog", "The frog was bitten by the snake"),
            ("The lion chased the gazelle", "The gazelle was chased by the lion"),
            ("The tiger attacked the deer", "The deer was attacked by the tiger"),
            ("The monkey picked the banana", "The banana was picked by the monkey"),
            ("The elephant drank the water", "The water was drunk by the elephant"),
            ("The giraffe ate the leaves", "The leaves were eaten by the giraffe"),
            ("The wind blew the leaves", "The leaves were blown by the wind"),
            ("The sun melted the ice", "The ice was melted by the sun"),
            ("The river carried the boat", "The boat was carried by the river"),
            ("The storm sank the ship", "The ship was sunk by the storm"),
            ("The flood destroyed the village", "The village was destroyed by the flood"),
            ("The fire burned the forest", "The forest was burned by the fire"),
            ("The snow covered the mountain", "The mountain was covered by the snow"),
            ("The frost damaged the crops", "The crops were damaged by the frost"),
            ("The drought killed the plants", "The plants were killed by the drought"),
            ("The hurricane destroyed the pier", "The pier was destroyed by the hurricane"),
            ("The lightning struck the tree", "The tree was struck by the lightning"),
            ("The avalanche buried the cabin", "The cabin was buried by the avalanche"),
            ("The earthquake cracked the wall", "The wall was cracked by the earthquake"),
            ("The wind knocked down the tree", "The tree was knocked down by the wind"),
            ("The rain flooded the street", "The street was flooded by the rain"),
            ("The hail damaged the car", "The car was damaged by the hail"),
            ("The frost killed the flowers", "The flowers were killed by the frost"),
            ("The tide eroded the cliff", "The cliff was eroded by the tide"),
            ("The sun dried the river", "The river was dried by the sun"),
            ("The ice covered the lake", "The lake was covered by the ice"),
            ("The mud buried the path", "The path was buried by the mud"),
            ("The heat melted the snow", "The snow was melted by the heat"),
            ("The storm uprooted the tree", "The tree was uprooted by the storm"),
            ("The rain soaked the ground", "The ground was soaked by the rain"),
            ("The wind scattered the leaves", "The leaves were scattered by the wind"),
            ("The current swept the debris", "The debris was swept by the current"),
            ("The fire charred the wood", "The wood was charred by the fire"),
            ("The earthquake leveled the building", "The building was leveled by the earthquake"),
            ("The sun warmed the earth", "The earth was warmed by the sun"),
            ("The wind bent the trees", "The trees were bent by the wind"),
            ("The rain filled the pond", "The pond was filled by the rain"),
            ("The snow blanketed the hills", "The hills were blanketed by the snow"),
            ("The storm darkened the sky", "The sky was darkened by the storm"),
            ("The ice sealed the lake", "The lake was sealed by the ice"),
            ("The wave eroded the beach", "The beach was eroded by the wave"),
            ("The heat cracked the pavement", "The pavement was cracked by the heat"),
            ("The drought dried the well", "The well was dried by the drought"),
            ("The lightning ignited the forest", "The forest was ignited by the lightning"),
            ("The avalanche blocked the road", "The road was blocked by the avalanche"),
            ("The glacier shaped the landscape", "The landscape was shaped by the glacier"),
            ("The current carved the canyon", "The canyon was carved by the current"),
            ("The wind shaped the dunes", "The dunes were shaped by the wind"),
            ("The rain nourished the soil", "The soil was nourished by the rain"),
            ("The sun ripened the fruit", "The fruit was ripened by the sun"),
            ("The frost nipped the buds", "The buds were nipped by the frost"),
            ("The tide reshaped the shore", "The shore was reshaped by the tide"),
            ("The fire blackened the field", "The field was blackened by the fire"),
            ("The earthquake split the rock", "The rock was split by the earthquake"),
            ("The wind polished the stone", "The stone was polished by the wind"),
            ("The rain cleansed the air", "The air was cleansed by the rain"),
            ("The snow insulated the ground", "The ground was insulated by the snow"),
            ("The storm revealed the wreckage", "The wreckage was revealed by the storm"),
            ("The sun bleached the fabric", "The fabric was bleached by the sun"),
            ("The ice preserved the specimen", "The specimen was preserved by the ice"),
            ("The flood deposited the silt", "The silt was deposited by the flood"),
            ("The wind dispersed the seeds", "The seeds were dispersed by the wind"),
            ("The rain revived the garden", "The garden was revived by the rain"),
            ("The frost hardened the ground", "The ground was hardened by the frost"),
            ("The heat expanded the metal", "The metal was expanded by the heat"),
            ("The drought withered the grass", "The grass was withered by the drought"),
            ("The hurricane flattened the crops", "The crops were flattened by the hurricane"),
        ]
    },
    'semantic_valence': {
        'type': 'SEM',
        'pairs': [
            ("The gift brought immense joy", "The disaster caused terrible grief"),
            ("Her smile radiated warmth", "Her scowl projected hostility"),
            ("The victory filled us with pride", "The defeat filled us with shame"),
            ("The garden bloomed with beauty", "The wasteland reeked of decay"),
            ("The gentle rain nourished the earth", "The toxic waste poisoned the soil"),
            ("The kind stranger helped the child", "The cruel thief robbed the elderly"),
            ("The peaceful meadow soothed the soul", "The battlefield horrified the witnesses"),
            ("The delicious feast delighted everyone", "The spoiled food disgusted the guests"),
            ("The brilliant sunrise inspired hope", "The devastating earthquake sparked fear"),
            ("The warm embrace comforted the baby", "The cold rejection wounded the teenager"),
            ("The harmonious choir uplifted spirits", "The discordant noise grated on nerves"),
            ("The fragrant rose pleased the senses", "The foul odor repulsed the crowd"),
            ("The wise mentor guided the student", "The corrupt official deceived the public"),
            ("The generous donation helped the poor", "The greedy scam exploited the vulnerable"),
            ("The clean water refreshed the hikers", "The polluted stream sickened the villagers"),
            ("The cheerful song brightened the mood", "The mournful dirge deepened the sorrow"),
            ("The sturdy bridge connected the towns", "The crumbling wall isolated the community"),
            ("The tender moment touched the heart", "The brutal attack scarred the victim"),
            ("The sweet honey delighted the palate", "The bitter medicine distressed the patient"),
            ("The bright light illuminated the room", "The dense fog obscured the path"),
            ("The faithful dog protected the family", "The wild beast threatened the campers"),
            ("The soft blanket comforted the child", "The rough surface scratched the skin"),
            ("The clear stream quenched the thirst", "The muddy puddle contaminated the water"),
            ("The gentle breeze cooled the skin", "The scorching heat burned the crops"),
            ("The honest answer satisfied the judge", "The lying witness angered the court"),
            ("The beautiful painting amazed the critics", "The ugly graffiti appalled the residents"),
            ("The loyal friend supported the cause", "The treacherous spy betrayed the team"),
            ("The healthy baby thrived and grew", "The sick patient suffered and weakened"),
            ("The calm ocean relaxed the swimmers", "The raging storm terrified the sailors"),
            ("The fresh bread filled the kitchen with aroma", "The rotten fruit filled the room with stench"),
            ("The brave soldier defended the country", "The cowardly deserter abandoned the post"),
            ("The lucky winner celebrated the prize", "The unfortunate victim mourned the loss"),
            ("The clean city impressed the tourists", "The filthy slum shocked the visitors"),
            ("The safe harbor sheltered the boats", "The dangerous reef wrecked the ships"),
            ("The wise decision benefited everyone", "The foolish mistake harmed many people"),
            ("The loving mother cared for the baby", "The neglectful parent ignored the child"),
            ("The successful project earned praise", "The failed experiment caused disappointment"),
            ("The peaceful protest demanded justice", "The violent riot destroyed property"),
            ("The helpful guide assisted the travelers", "The misleading sign confused the drivers"),
            ("The fair judge upheld the law", "The biased referee favored the home team"),
            ("The rich soil produced abundant crops", "The barren desert yielded nothing"),
            ("The warm fire heated the cabin", "The bitter cold froze the pipes"),
            ("The clear sky revealed the stars", "The thick smoke hid the mountains"),
            ("The smooth road allowed fast travel", "The bumpy trail slowed the hikers"),
            ("The sharp knife cut cleanly", "The dull blade tore the paper"),
            ("The fresh air invigorated the runners", "The stale air drowsed the audience"),
            ("The bright student earned the scholarship", "The lazy worker lost the job"),
            ("The clean house welcomed the guests", "The messy room embarrassed the host"),
            ("The kind words encouraged the child", "The harsh criticism crushed the spirit"),
            ("The strong bridge withstood the flood", "The weak dam burst under pressure"),
            ("The sweet melody charmed the audience", "The harsh noise annoyed the neighbors"),
            ("The gentle rain refreshed the garden", "The acid rain killed the fish"),
            ("The honest politician served the people", "The corrupt leader stole from the treasury"),
            ("The efficient machine saved time", "The broken tool wasted effort"),
            ("The comfortable bed eased the pain", "The hard floor worsened the ache"),
            ("The delicious meal satisfied the hunger", "The tasteless porridge left them wanting"),
            ("The beautiful garden attracted butterflies", "The weed patch repelled the visitors"),
            ("The competent doctor cured the illness", "The inept surgeon worsened the injury"),
            ("The loyal dog guarded the house", "The stray cat knocked over the vase"),
            ("The bright candle lit the room", "The power outage darkened the city"),
            ("The smooth silk felt luxurious", "The coarse burlap scratched the skin"),
            ("The pure spring water tasted fresh", "The contaminated well made people ill"),
            ("The generous host welcomed the guests", "The stingy landlord evicted the tenants"),
            ("The creative solution impressed the boss", "The lazy approach disappointed the team"),
            ("The warm sweater provided comfort", "The thin jacket offered no protection"),
            ("The clean beach attracted swimmers", "The oil spill killed the wildlife"),
            ("The honest merchant gave fair prices", "The scammer cheated the customers"),
            ("The skilled craftsman built quality furniture", "The amateur carpenter made shaky chairs"),
            ("The sweet fruit satisfied the sweet tooth", "The sour lemon made her pucker"),
            ("The calm lake reflected the mountains", "The turbulent rapids capsized the canoe"),
            ("The peaceful village slept quietly", "The war zone echoed with explosions"),
            ("The healthy forest supported wildlife", "The logged clearing left animals homeless"),
            ("The safe neighborhood raised children", "The dangerous street bred criminals"),
            ("The fragrant perfume attracted compliments", "The pungent odor drove people away"),
            ("The brilliant scientist made discoveries", "The ignorant fool spread misinformation"),
            ("The reliable car started every morning", "The clunker broke down on the highway"),
            ("The spacious room accommodated everyone", "The cramped closet felt suffocating"),
            ("The wise owl watched silently", "The foolish parrot squawked noisily"),
            ("The soft pillow cradled the head", "The hard stone bruised the knee"),
            ("The clean river teemed with fish", "The polluted canal was lifeless"),
            ("The happy couple celebrated their anniversary", "The grieving widow mourned her loss"),
            ("The fresh snow blanketed the ground", "The dirty slush ruined the shoes"),
            ("The clever fox outsmarted the hunter", "The dim wolf fell into the trap"),
            ("The gentle giant helped the villagers", "The cruel tyrant oppressed the people"),
            ("The bright diamond sparkled in the light", "The dull pebble sat in the dirt"),
            ("The nutritious meal fed the family", "The junk food left them hungry"),
            ("The steady hand painted the portrait", "The trembling brush ruined the canvas"),
            ("The clear explanation clarified the concept", "The confusing lecture baffled the students"),
            ("The smooth landing pleased the passengers", "The crash landing injured the crew"),
            ("The fair trial ensured justice", "The kangaroo court convicted the innocent"),
            ("The warm sunlight nurtured the plants", "The freezing wind killed the flowers"),
            ("The honest report stated the facts", "The fake news spread lies"),
            ("The comfortable chair relaxed the reader", "The stiff bench ached the back"),
            ("The skilled chef prepared the feast", "The amateur cook burned the dinner"),
            ("The clean air filled the lungs", "The smog choked the city dwellers"),
            ("The reliable witness told the truth", "The perjurer lied under oath"),
            ("The spacious park provided recreation", "The cramped cell caused claustrophobia"),
            ("The gentle teacher encouraged the pupil", "The harsh master beat the apprentice"),
            ("The beautiful music moved the audience", "The cacophony hurt the ears"),
            ("The efficient system processed the data", "The buggy software crashed the computer"),
            ("The safe car protected the passengers", "The defective vehicle endangered the driver"),
            ("The clean water nourished the village", "The contaminated supply poisoned the town"),
            ("The bright future awaited the graduate", "The bleak outlook discouraged the youth"),
            ("The warm coat kept out the chill", "The torn jacket let in the cold"),
            ("The smooth sailing pleased the crew", "The rough seas terrified the passengers"),
            ("The kind gesture warmed the heart", "The cruel joke humiliated the victim"),
            ("The fresh produce nourished the body", "The expired food caused illness"),
            ("The competent leader inspired confidence", "The inept manager caused chaos"),
            ("The clear signal reached the receiver", "The static noise disrupted the broadcast"),
        ]
    },
    'semantic_topic': {
        'type': 'SEM',
        'pairs': [
            ("The chef seasoned the soup carefully", "The programmer debugged the code patiently"),
            ("The painter mixed colors on the palette", "The engineer calibrated the instrument precisely"),
            ("The musician tuned the violin beautifully", "The scientist analyzed the data thoroughly"),
            ("The gardener pruned the roses skillfully", "The lawyer argued the case persuasively"),
            ("The baker kneaded the dough gently", "The doctor examined the patient carefully"),
            ("The dancer practiced the routine gracefully", "The architect designed the building innovatively"),
            ("The poet composed verses elegantly", "The mechanic repaired the engine efficiently"),
            ("The actor rehearsed the lines passionately", "The accountant balanced the books accurately"),
            ("The fisherman cast the net skillfully", "The teacher explained the lesson clearly"),
            ("The sculptor carved the marble masterfully", "The nurse administered the medicine gently"),
            ("The singer rehearsed the aria beautifully", "The electrician wired the house safely"),
            ("The weaver threaded the loom carefully", "The chemist mixed the solution precisely"),
            ("The potter shaped the clay artistically", "The plumber fixed the pipe quickly"),
            ("The conductor led the orchestra magnificently", "The judge presided over the court fairly"),
            ("The farmer planted the seeds methodically", "The pilot navigated the aircraft smoothly"),
            ("The chef prepared the sauce delicately", "The programmer optimized the algorithm cleverly"),
            ("The painter sketched the portrait swiftly", "The engineer tested the bridge rigorously"),
            ("The musician played the sonata flawlessly", "The scientist published the paper successfully"),
            ("The gardener watered the plants diligently", "The lawyer filed the motion promptly"),
            ("The baker decorated the cake beautifully", "The doctor prescribed the medication wisely"),
            ("The dancer performed the waltz elegantly", "The architect planned the city comprehensively"),
            ("The poet wrote the ode movingly", "The mechanic overhauled the transmission thoroughly"),
            ("The actor delivered the monologue powerfully", "The accountant audited the records meticulously"),
            ("The fisherman caught the salmon expertly", "The teacher graded the exams fairly"),
            ("The sculptor chiseled the statue precisely", "The nurse dressed the wound gently"),
            ("The soprano sang the aria sublimely", "The technician serviced the equipment reliably"),
            ("The artist painted the landscape vividly", "The researcher conducted the study rigorously"),
            ("The cook prepared the feast lavishly", "The developer built the application efficiently"),
            ("The musician composed the symphony brilliantly", "The analyst evaluated the report critically"),
            ("The florist arranged the bouquet artistically", "The pharmacist dispensed the prescription accurately"),
            ("The chef garnished the dish elegantly", "The programmer refactored the code cleanly"),
            ("The painter applied the glaze smoothly", "The engineer designed the circuit cleverly"),
            ("The musician practiced the concerto diligently", "The scientist replicated the experiment carefully"),
            ("The gardener harvested the vegetables timely", "The lawyer negotiated the settlement skillfully"),
            ("The baker measured the ingredients precisely", "The doctor diagnosed the condition accurately"),
            ("The dancer choreographed the ballet creatively", "The architect rendered the blueprint clearly"),
            ("The poet crafted the haiku meticulously", "The mechanic aligned the wheels perfectly"),
            ("The actor portrayed the character convincingly", "The accountant calculated the taxes correctly"),
            ("The fisherman navigated the river confidently", "The teacher inspired the students effectively"),
            ("The sculptor polished the bronze brilliantly", "The nurse monitored the patient attentively"),
            ("The vocalist performed the ballad emotionally", "The technician calibrated the sensor exactly"),
            ("The illustrator drew the comic expressively", "The researcher proved the theorem rigorously"),
            ("The pastry chef frosted the cupcake beautifully", "The software engineer optimized the database efficiently"),
            ("The cellist played the suite passionately", "The data analyst visualized the trends clearly"),
            ("The florist wrapped the bouquet neatly", "The pharmacist verified the dosage carefully"),
            ("The chef plated the appetizer artistically", "The programmer implemented the feature correctly"),
            ("The watercolorist washed the sky subtly", "The engineer simulated the model accurately"),
            ("The violinist performed the caprice virtuosically", "The scientist validated the hypothesis convincingly"),
            ("The horticulturist grafted the branch skillfully", "The attorney cross-examined the witness sharply"),
            ("The chocolatier tempered the chocolate perfectly", "The physician treated the infection effectively"),
            ("The ballerina executed the pirouette flawlessly", "The urban planner designed the park thoughtfully"),
            ("The lyricist penned the chorus catchily", "The technician troubleshot the problem systematically"),
            ("The thespian delivered the soliloquy movingly", "The auditor verified the compliance thoroughly"),
            ("The angler landed the trout expertly", "The educator assessed the progress fairly"),
            ("The ceramicist glazed the vase beautifully", "The clinician evaluated the symptoms carefully"),
            ("The mezzo-soprano sang the aria expressively", "The specialist diagnosed the fault accurately"),
            ("The muralist painted the wall vibrantly", "The investigator gathered the evidence methodically"),
            ("The saucier reduced the sauce perfectly", "The developer deployed the update seamlessly"),
            ("The pianist played the nocturne tenderly", "The statistician analyzed the variance precisely"),
            ("The bonsai master pruned the tree artistically", "The barrister presented the argument persuasively"),
            ("The patissier layered the pastry delicately", "The clinician prescribed the treatment appropriately"),
            ("The contemporary dancer moved fluidly", "The structural engineer calculated the loads accurately"),
            ("The sonneteer crafted the poem exquisitely", "The auto mechanic diagnosed the issue correctly"),
            ("The improviser delivered the line spontaneously", "The bookkeeper reconciled the accounts precisely"),
            ("The fly fisherman cast the line gracefully", "The tutor explained the concept patiently"),
            ("The glassblower shaped the vase skillfully", "The pharmacist compounded the medication accurately"),
            ("The jazz singer scatted the melody creatively", "The programmer debugged the script efficiently"),
            ("The portraitist captured the likeness perfectly", "The engineer tested the prototype rigorously"),
            ("The sommelier paired the wine expertly", "The researcher conducted the survey methodically"),
            ("The topiary artist trimmed the hedge precisely", "The attorney drafted the contract carefully"),
            ("The sushi chef sliced the fish masterfully", "The surgeon performed the operation successfully"),
            ("The figure skater landed the jump cleanly", "The architect specified the materials appropriately"),
            ("The librettist wrote the opera eloquently", "The technician maintained the system reliably"),
            ("The character actor embodied the role completely", "The financial analyst forecast the market accurately"),
            ("The fly tier dressed the hook meticulously", "The professor explained the theory clearly"),
            ("The potter fired the kiln carefully", "The anesthesiologist monitored the vitals closely"),
            ("The coloratura sang the cadenza brilliantly", "The network admin secured the firewall properly"),
            ("The street artist sprayed the mural boldly", "The biochemist purified the protein successfully"),
            ("The saucier emulsified the dressing smoothly", "The coder reviewed the pull request thoroughly"),
            ("The harpist plucked the strings delicately", "The metrologist measured the standard precisely"),
            ("The ikebana master arranged the flowers harmoniously", "The barrister cross-examined the witness incisively"),
            ("The pastry chef folded the croissant perfectly", "The geriatrician cared for the elderly compassionately"),
            ("The modern dancer interpreted the music expressively", "The civil engineer designed the overpass safely"),
            ("The epigrammatist composed the quip wittily", "The mechanic rebuilt the engine completely"),
            ("The stand-up comedian delivered the punchline perfectly", "The accountant reconciled the ledger accurately"),
            ("The fly fisherman released the trout gently", "The mentor guided the student patiently"),
            ("The ceramicist carved the pattern intricately", "The nurse practitioner assessed the patient thoroughly"),
            ("The cabaret singer performed the number dazzlingly", "The IT specialist resolved the ticket promptly"),
            ("The landscape painter captured the light beautifully", "The researcher published the findings convincingly"),
            ("The grill chef seared the steak perfectly", "The developer refactored the module cleanly"),
            ("The organist played the fugue masterfully", "The data scientist trained the model effectively"),
            ("The ikebana practitioner arranged the branches elegantly", "The legal advisor reviewed the contract meticulously"),
            ("The pizzaiolo stretched the dough expertly", "The pediatrician examined the child gently"),
            ("The tap dancer executed the routine rhythmically", "The project manager coordinated the team efficiently"),
            ("The versifier composed the limerick cleverly", "The electrician installed the panel safely"),
            ("The impressionist painted the haystack luminously", "The pharmacologist studied the interaction carefully"),
            ("The barista pulled the espresso perfectly", "The programmer wrote the test comprehensively"),
            ("The calligrapher penned the invitation elegantly", "The engineer validated the design thoroughly"),
            ("The mime performed the sketch silently", "The auditor examined the records diligently"),
        ]
    },
}

MODEL_CONFIGS = {
    'qwen3': {
        'name': 'Qwen3-4B',
        'path': 'D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',
        'n_layers': 36,
        'd_model': 2560,
        'bottleneck_layer': 6,
        'dtype': 'bf16',
    },
    'deepseek7b': {
        'name': 'DeepSeek-R1-Distill-Qwen-7B',
        'path': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'n_layers': 28,
        'd_model': 3584,
        'bottleneck_layer': 4,
        'dtype': '8bit',
    },
    'glm4': {
        'name': 'GLM4-9B-Chat',
        'path': 'D:/develop/model/hub/models--zai-org--glm-4-9b-chat-hf/download',
        'n_layers': 40,
        'd_model': 4096,
        'bottleneck_layer': 30,
        'dtype': '8bit',
    },
}

# 采样层: 不需要每层都做, 采样关键层
SAMPLED_LAYERS = {
    'qwen3': list(range(0, 36, 3)) + [35],  # 0,3,6,9,...,33,35 → ~13层
    'deepseek7b': list(range(0, 30, 2)) + [29],  # 0,2,4,...,28,29 → ~16层
    'glm4': list(range(0, 40, 4)) + [39],  # 0,4,8,...,36,39 → ~11层
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=120)
    args = parser.parse_args()
    
    model_key = args.model
    config = MODEL_CONFIGS[model_key]
    n_pairs = args.n_pairs
    feature_names = list(FEATURE_PAIRS.keys())
    sampled_layers = SAMPLED_LAYERS[model_key]
    
    # 输出目录
    out_dir = f'results/causal_fiber/{model_key}_ccxxii'
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'run.log')
    
    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    log(f"\n{'='*70}")
    log(f"Phase CCXXII: 全层因果扫描 — {model_key}")
    log(f"{'='*70}")
    log(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  模型: {config['name']}")
    log(f"  采样层: {sampled_layers}")
    log(f"  特征: {feature_names}")
    log(f"  样本对数: {n_pairs}")
    
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
    
    # 重新计算采样层 (基于实际n_layers)
    if model_key == 'qwen3':
        sampled_layers = sorted(set(list(range(0, n_layers, 3)) + [n_layers-1]))
    elif model_key == 'deepseek7b':
        sampled_layers = sorted(set(list(range(0, n_layers, 2)) + [n_layers-1]))
    else:
        sampled_layers = sorted(set(list(range(0, n_layers, 4)) + [n_layers-1]))
    
    log(f"  实际采样层: {sampled_layers}")
    
    # ============================================================
    # S1: 逐层收集差分向量
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S1: 逐层收集差分向量 ({n_pairs}对/特征 × {len(sampled_layers)}层)")
    log(f"{'='*60}")
    
    # 数据结构: layer_deltas[layer][feat] = [delta_vectors]
    layer_deltas = {l: {f: [] for f in feature_names} for l in sampled_layers}
    layer_logits = {l: {f: {'s1': [], 's2': []} for f in feature_names} for l in sampled_layers}
    
    def make_hook(layer_idx, store):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            store[layer_idx] = out.detach().cpu().clone()
        return hook_fn
    
    # 注册hook到所有采样层
    cap_store = {}
    handles = []
    for layer in sampled_layers:
        h = model.model.layers[layer].register_forward_hook(make_hook(layer, cap_store))
        handles.append(h)
    
    for feat in feature_names:
        all_pairs = FEATURE_PAIRS[feat]['pairs'][:n_pairs]
        log(f"  收集 {feat} ({len(all_pairs)}对)...")
        
        for i, (s1, s2) in enumerate(all_pairs):
            if (i+1) % 20 == 0:
                log(f"    {feat}: {i+1}/{len(all_pairs)}")
            
            with torch.no_grad():
                # s1
                tokens1 = tokenizer(s1, return_tensors='pt').to(device)
                out1 = model(**tokens1)
                logit1 = out1.logits[0, -1].detach().cpu().float().numpy()
                
                # 收集s1的各层激活
                act1_by_layer = {}
                for layer in sampled_layers:
                    if layer in cap_store:
                        act1_by_layer[layer] = cap_store[layer][0, -1].float().numpy()
                
                # s2
                tokens2 = tokenizer(s2, return_tensors='pt').to(device)
                out2 = model(**tokens2)
                logit2 = out2.logits[0, -1].detach().cpu().float().numpy()
                
                # 收集s2的各层激活
                act2_by_layer = {}
                for layer in sampled_layers:
                    if layer in cap_store:
                        act2_by_layer[layer] = cap_store[layer][0, -1].float().numpy()
            
            for layer in sampled_layers:
                if layer in act1_by_layer and layer in act2_by_layer:
                    delta = act2_by_layer[layer] - act1_by_layer[layer]
                    layer_deltas[layer][feat].append(delta)
                    layer_logits[layer][feat]['s1'].append(logit1)
                    layer_logits[layer][feat]['s2'].append(logit2)
            
            del out1, out2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 移除hook
    for h in handles:
        h.remove()
    
    log(f"\n  收集完成!")
    for layer in sampled_layers:
        counts = [len(layer_deltas[layer][f]) for f in feature_names]
        log(f"    L{layer}: {dict(zip(feature_names, counts))}")
    
    # ============================================================
    # S2: 逐层Per-feature PCA + 差分投影比
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S2: 逐层Per-feature PCA + 差分投影比")
    log(f"{'='*60}")
    
    from sklearn.decomposition import PCA
    
    # 结果存储
    layer_results = {}
    
    for layer in sampled_layers:
        log(f"\n  === Layer {layer} ===")
        layer_results[layer] = {}
        
        for feat in feature_names:
            deltas = np.array(layer_deltas[layer][feat])
            if len(deltas) < 10:
                continue
            
            # PCA
            pca = PCA(n_components=min(10, len(deltas)-1))
            pca.fit(deltas)
            pc1_dir = pca.components_[0]
            pc1_var = pca.explained_variance_ratio_[0]
            
            # 随机方向
            np.random.seed(42)
            rand_dir = np.random.randn(d_model)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            
            # 差分投影比
            pc1_projs = []
            rand_projs = []
            for delta in deltas:
                delta_norm = np.linalg.norm(delta)
                if delta_norm < 1e-8:
                    continue
                pc1_projs.append(abs(np.dot(delta, pc1_dir)) / delta_norm)
                rand_projs.append(abs(np.dot(delta, rand_dir)) / delta_norm)
            
            pc1_proj_mean = np.mean(pc1_projs)
            rand_proj_mean = np.mean(rand_projs)
            pc1_vs_random = pc1_proj_mean / rand_proj_mean if rand_proj_mean > 0 else 0
            
            layer_results[layer][feat] = {
                'pc1_variance': float(pc1_var),
                'pc1_proj_pct': float(pc1_proj_mean),
                'random_proj_pct': float(rand_proj_mean),
                'pc1_vs_random': float(pc1_vs_random),
                'pc1_dir': pc1_dir,
                'n_pairs': len(pc1_projs),
            }
        
        # 打印该层结果
        log(f"  {'Feature':<18} {'PC1_var%':>9} {'PC1_proj%':>9} {'Rand%':>7} {'PC1/R':>7}")
        log(f"  {'-'*55}")
        for feat in feature_names:
            if feat not in layer_results[layer]:
                continue
            r = layer_results[layer][feat]
            log(f"  {feat:<18} {r['pc1_variance']*100:>8.1f}% {r['pc1_proj_pct']*100:>8.2f}% {r['random_proj_pct']*100:>6.2f}% {r['pc1_vs_random']:>6.2f}x")
        
        # 该层平均
        avg_pc1_var = np.mean([r['pc1_variance'] for r in layer_results[layer].values()])
        avg_pc1_vs_r = np.mean([r['pc1_vs_random'] for r in layer_results[layer].values()])
        log(f"  {'AVG':<18} {avg_pc1_var*100:>8.1f}% {'':>9} {'':>7} {avg_pc1_vs_r:>6.2f}x")
    
    # ============================================================
    # S3: 逐层跨特征PC1对齐度
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S3: 逐层跨特征PC1对齐度")
    log(f"{'='*60}")
    
    for layer in sampled_layers:
        log(f"\n  === Layer {layer} ===")
        
        # 收集该层所有特征的PC1方向
        dirs = {}
        for feat in feature_names:
            if feat in layer_results[layer]:
                dirs[feat] = layer_results[layer][feat]['pc1_dir']
        
        if len(dirs) < 2:
            continue
        
        # 计算两两对齐度
        feat_list = list(dirs.keys())
        log(f"  跨特征PC1对齐度 (cosine):")
        alignments = []
        for i in range(len(feat_list)):
            for j in range(i+1, len(feat_list)):
                cos = abs(np.dot(dirs[feat_list[i]], dirs[feat_list[j]]))
                alignments.append(cos)
                if cos > 0.9 or cos < 0.1:
                    log(f"    {feat_list[i]:<16} <-> {feat_list[j]:<16}: {cos:.4f} {'★' if cos>0.9 else ''}")
        
        avg_align = np.mean(alignments) if alignments else 0
        min_align = np.min(alignments) if alignments else 0
        max_align = np.max(alignments) if alignments else 0
        log(f"  对齐度统计: mean={avg_align:.4f}, min={min_align:.4f}, max={max_align:.4f}")
        
        # 保存
        for feat in feature_names:
            if feat in layer_results[layer]:
                layer_results[layer][feat]['cross_feat_alignment_mean'] = float(avg_align)
                layer_results[layer][feat]['cross_feat_alignment_min'] = float(min_align)
    
    # ============================================================
    # S4: 层间PC1方向旋转分析
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S4: 层间PC1方向旋转分析")
    log(f"{'='*60}")
    
    for feat in feature_names:
        log(f"\n  === {feat} ===")
        log(f"  {'Layer':>6} -> {'Next':>6} {'cos(PC1)':>10} {'rotation°':>10}")
        log(f"  {'-'*40}")
        
        prev_layer = None
        for layer in sampled_layers:
            if feat not in layer_results[layer]:
                continue
            if prev_layer is not None and feat in layer_results[prev_layer]:
                dir_curr = layer_results[layer][feat]['pc1_dir']
                dir_prev = layer_results[prev_layer][feat]['pc1_dir']
                cos = abs(np.dot(dir_curr, dir_prev))
                angle = np.degrees(np.arccos(np.clip(cos, 0, 1)))
                log(f"  L{prev_layer:>4} -> L{layer:>4} {cos:>10.4f} {angle:>9.1f}°")
            prev_layer = layer
    
    # ============================================================
    # S5: 全层汇总 — PC1因果效应曲线
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S5: 全层汇总 — PC1因果效应曲线")
    log(f"{'='*60}")
    
    log(f"\n  === Per-feature PC1方差曲线 ===")
    header = f"  {'Layer':>6}"
    for feat in feature_names:
        header += f" {feat[:6]:>7}"
    header += f" {'AVG':>7}"
    log(header)
    
    for layer in sampled_layers:
        line = f"  L{layer:>4}"
        vals = []
        for feat in feature_names:
            if feat in layer_results[layer]:
                v = layer_results[layer][feat]['pc1_variance'] * 100
                vals.append(v)
                line += f" {v:>7.1f}"
            else:
                line += f" {'N/A':>7}"
        if vals:
            line += f" {np.mean(vals):>7.1f}"
        log(line)
    
    log(f"\n  === PC1/Random 因果效应曲线 ===")
    header = f"  {'Layer':>6}"
    for feat in feature_names:
        header += f" {feat[:6]:>7}"
    header += f" {'AVG':>7}"
    log(header)
    
    for layer in sampled_layers:
        line = f"  L{layer:>4}"
        vals = []
        for feat in feature_names:
            if feat in layer_results[layer]:
                v = layer_results[layer][feat]['pc1_vs_random']
                vals.append(v)
                line += f" {v:>7.2f}"
            else:
                line += f" {'N/A':>7}"
        if vals:
            line += f" {np.mean(vals):>7.2f}"
        log(line)
    
    log(f"\n  === 跨特征PC1对齐度曲线 ===")
    log(f"  {'Layer':>6} {'mean_cos':>10} {'min_cos':>10}")
    log(f"  {'-'*30}")
    
    for layer in sampled_layers:
        if feature_names[0] in layer_results[layer]:
            mean_cos = layer_results[layer][feature_names[0]].get('cross_feat_alignment_mean', 0)
            min_cos = layer_results[layer][feature_names[0]].get('cross_feat_alignment_min', 0)
            log(f"  L{layer:>4} {mean_cos:>10.4f} {min_cos:>10.4f}")
    
    # ============================================================
    # S6: 关键发现汇总
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S6: 关键发现汇总")
    log(f"{'='*60}")
    
    # 找PC1方差峰值层
    peak_var_layer = None
    peak_var_val = 0
    avg_var_by_layer = {}
    for layer in sampled_layers:
        vals = [layer_results[layer][f]['pc1_variance'] for f in feature_names if f in layer_results[layer]]
        if vals:
            avg_var_by_layer[layer] = np.mean(vals)
            if avg_var_by_layer[layer] > peak_var_val:
                peak_var_val = avg_var_by_layer[layer]
                peak_var_layer = layer
    
    log(f"\n  PC1方差峰值层: L{peak_var_layer} (avg={peak_var_val*100:.1f}%)")
    
    # 找PC1/Random峰值层
    peak_causal_layer = None
    peak_causal_val = 0
    avg_causal_by_layer = {}
    for layer in sampled_layers:
        vals = [layer_results[layer][f]['pc1_vs_random'] for f in feature_names if f in layer_results[layer]]
        if vals:
            avg_causal_by_layer[layer] = np.mean(vals)
            if avg_causal_by_layer[layer] > peak_causal_val:
                peak_causal_val = avg_causal_by_layer[layer]
                peak_causal_layer = layer
    
    log(f"  PC1/Random峰值层: L{peak_causal_layer} (avg={peak_causal_val:.2f}x)")
    
    # 找对齐度峰值层
    peak_align_layer = None
    peak_align_val = 0
    avg_align_by_layer = {}
    for layer in sampled_layers:
        if feature_names[0] in layer_results[layer]:
            mean_cos = layer_results[layer][feature_names[0]].get('cross_feat_alignment_mean', 0)
            avg_align_by_layer[layer] = mean_cos
            if mean_cos > peak_align_val:
                peak_align_val = mean_cos
                peak_align_layer = layer
    
    log(f"  跨特征对齐度峰值层: L{peak_align_layer} (mean_cos={peak_align_val:.4f})")
    
    # 判断1D流形是否只在瓶颈层
    log(f"\n  === 1D流形判断 ===")
    threshold_var = 0.5  # PC1方差>50%视为1D压缩
    threshold_causal = 10  # PC1/Random>10x视为有因果效应
    threshold_align = 0.8  # 对齐度>0.8视为共享方向
    
    layers_with_1d = []
    for layer in sampled_layers:
        has_1d = (
            avg_var_by_layer.get(layer, 0) > threshold_var and
            avg_causal_by_layer.get(layer, 0) > threshold_causal and
            avg_align_by_layer.get(layer, 0) > threshold_align
        )
        if has_1d:
            layers_with_1d.append(layer)
    
    if layers_with_1d:
        log(f"  1D流形层: {layers_with_1d}")
        log(f"  1D流形层范围: L{min(layers_with_1d)} - L{max(layers_with_1d)}")
        log(f"  1D流形是否只在瓶颈层: {'是' if len(layers_with_1d) <= 3 else '否'}")
    else:
        log(f"  无层满足1D流形条件 (var>{threshold_var}, causal>{threshold_causal}x, align>{threshold_align})")
    
    # 保存数值结果
    save_data = {
        'model': model_key,
        'n_pairs': n_pairs,
        'sampled_layers': sampled_layers,
        'feature_names': feature_names,
        'peak_var_layer': peak_var_layer,
        'peak_causal_layer': peak_causal_layer,
        'peak_align_layer': peak_align_layer,
        'layers_with_1d': layers_with_1d,
        'layer_summary': {},
    }
    
    for layer in sampled_layers:
        save_data['layer_summary'][str(layer)] = {
            'avg_pc1_var': float(avg_var_by_layer.get(layer, 0)),
            'avg_pc1_vs_random': float(avg_causal_by_layer.get(layer, 0)),
            'avg_alignment': float(avg_align_by_layer.get(layer, 0)),
            'per_feat': {
                f: {
                    'pc1_var': layer_results[layer][f]['pc1_variance'],
                    'pc1_vs_random': layer_results[layer][f]['pc1_vs_random'],
                    'cross_feat_alignment_mean': layer_results[layer][f].get('cross_feat_alignment_mean', 0),
                }
                for f in feature_names if f in layer_results[layer]
            }
        }
    
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    log(f"\n  结果已保存到 {out_dir}/results.json")
    log(f"\n{'='*70}")
    log(f"CCXXII 完成! {model_key}")
    log(f"{'='*70}")


if __name__ == '__main__':
    main()
