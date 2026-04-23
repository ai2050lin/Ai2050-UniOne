"""
Phase CCXXI: Interchange Intervention — 1D流形因果效应的金标准验证
================================================================
核心目标:
1. 真正的激活替换干预: 在瓶颈层替换per-feature PC1分量, 测量输出变化
2. 同特征干预: 用同特征另一对的PC1分量替换 → 验证PC1的因果效应
3. 跨特征干预: 用特征A的PC1分量替换特征B → 验证共享流形
4. 随机方向控制: 用随机方向替换 → 基线对照
5. DS7B重点验证(共享PC1=0.94+) + Qwen3(方差陷阱对照) + GLM4(无瓶颈对照)

实验方法:
  对每对句子(s1, s2), 在瓶颈层:
  1. 提取s1和s2在per-feature PC1方向上的投影值 proj1, proj2
  2. 将s2的PC1投影替换为s1的: act_ablated = act2 + (proj1 - proj2) * pc1_dir
  3. 从瓶颈层重新前向传播, 获取输出logit
  4. 比较: 原始logit变化 vs 干预后logit变化

  注意: 8bit模型无法in-place修改激活, 改用间接方法:
  - 只替换PC1分量: 计算 ablated_delta = delta - proj_along_pc1 * pc1_dir
  - 用线性近似: logit_change ≈ W_out @ ablated_delta
  - 或更安全: 用model的最终层lm_head直接映射

样本量: 150对/特征 (更大量样本确保统计可靠性)
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
            ("The storm comes from the east", "The storm came from the east"),
            ("The soldier fights bravely", "The soldier fought bravely"),
            ("She spends money wisely", "She spent money wisely"),
            ("The fire burns brightly", "The fire burned brightly"),
            ("The eagle soars above the clouds", "The eagle soared above the clouds"),
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
            ("He spends the afternoon reading", "He spent the afternoon reading"),
            ("The girl dreams about flying", "The girl dreamed about flying"),
            ("The bird lands on the branch", "The bird landed on the branch"),
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
            ("John understands the rules", "John does not understand the rules"),
            ("Our window is open wide", "Our window is not open wide"),
            ("Mom needs some help", "Mom does not need some help"),
            ("Anna was at the library", "Anna was not at the library"),
            ("Dad will call tonight", "Dad will not call tonight"),
            ("This answer is right", "This answer is not right"),
            ("The shop is open late", "The shop is not open late"),
            ("I have tried that dish", "I have not tried that dish"),
            ("She knows the secret", "She does not know the secret"),
            ("We loved the show", "We did not love the show"),
            ("The soup smells delicious", "The soup does not smell delicious"),
            ("They discovered the truth", "They did not discover the truth"),
            ("I accept your apology", "I do not accept your apology"),
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
            ("The cake tastes wonderful", "The cake does not taste wonderful"),
            ("He speaks the truth always", "He does not speak the truth always"),
            ("The movie was entertaining", "The movie was not entertaining"),
            ("The result matches expectations", "The result does not match expectations"),
            ("She enjoys classical music", "She does not enjoy classical music"),
            ("The plan makes sense", "The plan does not make sense"),
            ("We support the decision", "We do not support the decision"),
            ("The experiment succeeded", "The experiment did not succeed"),
            ("He solves the puzzle", "He does not solve the puzzle"),
            ("The flower smells sweet", "The flower does not smell sweet"),
            ("She remembers the details", "She does not remember the details"),
            ("The dog obeys commands", "The dog does not obey commands"),
            ("We follow the rules", "We do not follow the rules"),
            ("The child behaves well", "The child does not behave well"),
            ("He drives carefully", "He does not drive carefully"),
            ("The artist creates beauty", "The artist does not create beauty"),
            ("She speaks the language", "She does not speak the language"),
            ("The system functions correctly", "The system does not function correctly"),
            ("They respect the law", "They do not respect the law"),
            ("The student understands math", "The student does not understand math"),
            ("He appreciates the effort", "He does not appreciate the effort"),
            ("The tree provides shade", "The tree does not provide shade"),
            ("She owns a bicycle", "She does not own a bicycle"),
            ("The river flows north", "The river does not flow north"),
            ("They attend the ceremony", "They do not attend the ceremony"),
            ("He delivers the package", "He does not deliver the package"),
            ("The bird migrates south", "The bird does not migrate south"),
            ("She participates actively", "She does not participate actively"),
            ("The cat hunts mice", "The cat does not hunt mice"),
            ("We celebrate the holiday", "We do not celebrate the holiday"),
            ("He repairs the engine", "He does not repair the engine"),
            ("The dog guards the house", "The dog does not guard the house"),
            ("She collects stamps", "She does not collect stamps"),
            ("The plant survives winter", "The plant does not survive winter"),
            ("They speak English", "They do not speak English"),
            ("He recognizes the pattern", "He does not recognize the pattern"),
            ("The rocket reaches orbit", "The rocket does not reach orbit"),
            ("She bakes bread", "She does not bake bread"),
            ("The team cooperates well", "The team does not cooperate well"),
            ("He attends the lecture", "He does not attend the lecture"),
            ("The water boils quickly", "The water does not boil quickly"),
            ("She writes poetry", "She does not write poetry"),
            ("The car starts easily", "The car does not start easily"),
            ("He teaches physics", "He does not teach physics"),
            ("The flower blooms annually", "The flower does not bloom annually"),
            ("She exercises daily", "She does not exercise daily"),
            ("The computer processes data", "The computer does not process data"),
            ("He wears glasses", "He does not wear glasses"),
            ("The boat floats steadily", "The boat does not float steadily"),
            ("She plays tennis", "She does not play tennis"),
            ("The machine produces goods", "The machine does not produce goods"),
            ("He studies chemistry", "He does not study chemistry"),
            ("The wind blows strongly", "The wind does not blow strongly"),
            ("She practices piano", "She does not practice piano"),
            ("The river freezes in winter", "The river does not freeze in winter"),
            ("He visits the museum", "He does not visit the museum"),
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
            ("The storm damaged the roof", "The roof was damaged by the storm"),
            ("The flood covered the field", "The field was covered by the flood"),
            ("The earthquake cracked the wall", "The wall was cracked by the earthquake"),
            ("The wind knocked down the tree", "The tree was knocked down by the wind"),
            ("The rain flooded the street", "The street was flooded by the rain"),
            ("The hail damaged the car", "The car was damaged by the hail"),
            ("The frost killed the flowers", "The flowers were killed by the frost"),
            ("The tide eroded the cliff", "The cliff was eroded by the tide"),
            ("The sun dried the river", "The river was dried by the sun"),
            ("The ice covered the lake", "The lake was covered by the ice"),
            ("The mud buried the path", "The path was buried by the mud"),
            ("The wave crashed the boat", "The boat was crashed by the wave"),
            ("The heat melted the snow", "The snow was melted by the heat"),
            ("The storm uprooted the tree", "The tree was uprooted by the storm"),
            ("The rain soaked the ground", "The ground was soaked by the rain"),
            ("The wind scattered the leaves", "The leaves were scattered by the wind"),
            ("The current swept the debris", "The debris was swept by the current"),
            ("The fire charred the wood", "The wood was charred by the fire"),
            ("The flood washed away the bridge", "The bridge was washed away by the flood"),
            ("The frost coated the windows", "The windows were coated by the frost"),
            ("The earthquake leveled the building", "The building was leveled by the earthquake"),
            ("The sun warmed the earth", "The earth was warmed by the sun"),
            ("The wind bent the trees", "The trees were bent by the wind"),
            ("The rain filled the pond", "The pond was filled by the rain"),
            ("The snow blanketed the hills", "The hills were blanketed by the snow"),
            ("The storm darkened the sky", "The sky was darkened by the storm"),
            ("The hail pitted the roof", "The roof was pitted by the hail"),
            ("The ice sealed the lake", "The lake was sealed by the ice"),
            ("The mud clogged the drain", "The drain was clogged by the mud"),
            ("The wave eroded the beach", "The beach was eroded by the wave"),
            ("The heat cracked the pavement", "The pavement was cracked by the heat"),
            ("The drought dried the well", "The well was dried by the drought"),
            ("The hurricane flooded the town", "The town was flooded by the hurricane"),
            ("The tornado ripped the roof", "The roof was ripped by the tornado"),
            ("The lightning ignited the forest", "The forest was ignited by the lightning"),
            ("The avalanche blocked the road", "The road was blocked by the avalanche"),
            ("The glacier shaped the landscape", "The landscape was shaped by the glacier"),
            ("The volcano buried the village", "The village was buried by the volcano"),
            ("The current carved the canyon", "The canyon was carved by the current"),
            ("The wind shaped the dunes", "The dunes were shaped by the wind"),
            ("The rain nourished the soil", "The soil was nourished by the rain"),
            ("The sun ripened the fruit", "The fruit was ripened by the sun"),
            ("The frost nipped the buds", "The buds were nipped by the frost"),
            ("The tide reshaped the shore", "The shore was reshaped by the tide"),
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
            ("The garden thrived beautifully", "The wasteland decayed horribly"),
            ("She expressed joyful delight", "She expressed bitter sorrow"),
            ("The blessing uplifted spirits", "The curse crushed souls"),
            ("He radiated confident assurance", "He exuded fearful anxiety"),
            ("The symphony sounded magnificent", "The cacophony sounded terrible"),
            ("She demonstrated virtuous integrity", "She demonstrated corrupt dishonesty"),
            ("The miracle inspired awe", "The tragedy provoked disgust"),
            ("He maintained cheerful optimism", "He maintained gloomy pessimism"),
            ("The garden flourished wonderfully", "The ruin crumbled miserably"),
            ("She showed gentle compassion", "She showed harsh cruelty"),
            ("The blessing healed wounds", "The curse inflicted pain"),
            ("He displayed courageous resolve", "He displayed cowardly hesitation"),
            ("The victory celebrated life", "The defeat mourned death"),
            ("She offered warm friendship", "She offered cold hostility"),
            ("The dawn brought peace", "The dusk brought conflict"),
            ("He demonstrated wise leadership", "He demonstrated foolish governance"),
            ("The blessing nurtured growth", "The curse stunted development"),
            ("She displayed serene confidence", "She displayed anxious uncertainty"),
            ("The harmony united the community", "The discord divided the community"),
            ("He exhibited generous philanthropy", "He exhibited selfish greed"),
            ("The miracle restored faith", "The tragedy shattered trust"),
            ("She showed tender mercy", "She showed ruthless vengeance"),
            ("The blessing granted freedom", "The curse imposed bondage"),
            ("He pursued virtuous ambition", "He pursued wicked exploitation"),
            ("The creation enriched lives", "The destruction impoverished souls"),
            ("She demonstrated loyal devotion", "She demonstrated treacherous deceit"),
            ("The harmony soothed the spirit", "The discord agitated the mind"),
            ("He maintained steadfast courage", "He maintained craven fear"),
            ("The blessing illuminated the path", "The curse obscured the way"),
            ("She expressed sincere appreciation", "She expressed venomous contempt"),
            ("The miracle healed the wounded", "The tragedy injured the innocent"),
            ("He showed noble restraint", "He showed base indulgence"),
            ("The victory united the nation", "The defeat divided the people"),
            ("She offered compassionate understanding", "She offered callous indifference"),
            ("The dawn awakened hope", "The dusk extinguished dreams"),
            ("He demonstrated ethical integrity", "He demonstrated moral corruption"),
            ("The blessing purified the soul", "The curse defiled the spirit"),
            ("She displayed gracious humility", "She displayed arrogant vanity"),
            ("The harmony brought prosperity", "The discord brought ruin"),
            ("He exhibited selfless dedication", "He exhibited narcissistic obsession"),
            ("The miracle transformed lives", "The catastrophe destroyed futures"),
            ("She showed forgiving mercy", "She showed unforgiving wrath"),
            ("The blessing inspired greatness", "The curse condemned mediocrity"),
            ("He pursued honorable goals", "He pursued dishonorable schemes"),
            ("The creation elevated humanity", "The destruction degraded civilization"),
            ("She demonstrated faithful commitment", "She demonstrated faithless betrayal"),
            ("The harmony restored balance", "The discord created chaos"),
            ("He maintained dignified composure", "He maintained disgraceful panic"),
            ("The blessing granted salvation", "The curse brought damnation"),
            ("She offered wise counsel", "She offered foolish advice"),
            ("The miracle brought redemption", "The tragedy brought ruin"),
            ("He showed compassionate justice", "He showed cruel oppression"),
            ("The victory earned respect", "The defeat earned contempt"),
            ("She demonstrated noble sacrifice", "She demonstrated base selfishness"),
            ("The dawn revealed beauty", "The dusk concealed horror"),
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
            ("The baker decorated the cake", "The surgeon removed the tumor"),
            ("The painter hung the wallpaper", "The mechanic changed the oil"),
            ("The chef seasoned the dish", "The doctor prescribed the medicine"),
            ("The sculptor polished the bronze", "The programmer updated the software"),
            ("The musician rehearsed the piece", "The lawyer drafted the contract"),
            ("The dancer stretched her legs", "The engineer calculated the load"),
            ("The author revised the chapter", "The scientist wrote the report"),
            ("The poet recited the verses", "The doctor examined the x-ray"),
            ("The actor memorized the lines", "The teacher prepared the lesson"),
            ("The singer warmed up her voice", "The farmer inspected the crops"),
            ("The florist watered the plants", "The banker reviewed the portfolio"),
            ("The plumber unclogged the drain", "The mechanic balanced the tires"),
            ("The carpenter measured the wood", "The chef tasted the sauce"),
            ("The barber swept the floor", "The nurse checked the vitals"),
            ("The electrician wired the outlet", "The journalist interviewed the witness"),
            ("The welder adjusted the torch", "The pharmacist counted the pills"),
            ("The mason mixed the cement", "The judge reviewed the evidence"),
            ("The tailor pressed the fabric", "The dentist cleaned the instrument"),
            ("The baker measured the flour", "The surgeon scrubbed the hands"),
            ("The painter primed the surface", "The mechanic inspected the brakes"),
            ("The chef chopped the vegetables", "The doctor listened to the heartbeat"),
            ("The sculptor selected the stone", "The programmer designed the interface"),
            ("The musician tuned the piano", "The lawyer presented the argument"),
            ("The dancer practiced the steps", "The engineer drew the blueprint"),
            ("The author outlined the plot", "The scientist formed the hypothesis"),
            ("The poet chose the words", "The doctor ordered the test"),
            ("The actor applied the makeup", "The teacher wrote the syllabus"),
            ("The singer read the notes", "The farmer plowed the field"),
            ("The florist cut the stems", "The banker signed the check"),
            ("The plumber turned the valve", "The mechanic replaced the filter"),
            ("The carpenter nailed the boards", "The chef garnished the plate"),
            ("The barber combed the hair", "The nurse applied the bandage"),
            ("The electrician connected the wires", "The journalist took the photograph"),
            ("The welder wore the mask", "The pharmacist labeled the bottle"),
            ("The mason leveled the bricks", "The judge entered the chamber"),
            ("The tailor pinned the pattern", "The dentist positioned the drill"),
            ("The baker set the timer", "The surgeon marked the incision"),
            ("The painter cleaned the brush", "The mechanic drained the oil"),
            ("The chef minced the garlic", "The doctor felt the pulse"),
            ("The sculptor shaped the clay", "The programmer compiled the code"),
            ("The musician read the score", "The lawyer filed the brief"),
            ("The dancer tied the shoes", "The engineer checked the calculations"),
            ("The author wrote the conclusion", "The scientist analyzed the results"),
            ("The poet found the rhythm", "The doctor reviewed the chart"),
            ("The actor took the stage", "The teacher entered the classroom"),
            ("The singer hit the note", "The farmer harvested the wheat"),
            ("The florist wrapped the bouquet", "The banker approved the mortgage"),
            ("The plumber soldered the pipe", "The mechanic aligned the wheels"),
            ("The carpenter sanded the surface", "The chef plated the dish"),
            ("The barber styled the hair", "The nurse took the temperature"),
            ("The electrician tested the circuit", "The journalist wrote the headline"),
            ("The welder struck the arc", "The pharmacist measured the dose"),
            ("The mason pointed the mortar", "The judge delivered the sentence"),
            ("The tailor cut the cloth", "The dentist injected the anesthetic"),
            ("The baker kneaded the dough", "The surgeon closed the wound"),
        ]
    },
}

# 限制对数
for feat in FEATURE_PAIRS_BASE:
    FEATURE_PAIRS_BASE[feat]['pairs'] = FEATURE_PAIRS_BASE[feat]['pairs'][:150]

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
    parser.add_argument("--n_pairs", type=int, default=150)
    args = parser.parse_args()
    
    model_name = args.model
    n_pairs_max = args.n_pairs
    cfg = MODEL_CONFIGS[model_name]
    bottleneck_layer = cfg['bottleneck_layer']
    
    out_dir = f"results/causal_fiber/{model_name}_ccxxi"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = open(f"{out_dir}/run.log", "w", encoding="utf-8")
    
    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()
    
    log(f"Phase CCXXI: Interchange Intervention — 1D流形因果效应验证")
    log(f"Model: {cfg['name']}, bottleneck_layer={bottleneck_layer}, n_pairs={n_pairs_max}")
    
    # ============================================================
    # 加载模型
    # ============================================================
    log("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sklearn.decomposition import PCA
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
    # S1: 收集瓶颈层差分向量 + Per-feature PCA
    # ============================================================
    log(f"\n{'='*60}")
    log(f"S1: 收集瓶颈层L{bottleneck_layer}的差分向量")
    log(f"{'='*60}")
    
    def make_hook(layer_idx, store):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            store[layer_idx] = out.detach().cpu().clone()
        return hook_fn
    
    cap_store = {}
    handle = model.model.layers[bottleneck_layer].register_forward_hook(make_hook(bottleneck_layer, cap_store))
    
    all_deltas = {feat: [] for feat in feature_names}
    all_acts = {feat: {'s1': [], 's2': []} for feat in feature_names}
    all_logits = {feat: {'s1': [], 's2': []} for feat in feature_names}
    
    for feat_idx, feat in enumerate(feature_names):
        pairs = FEATURE_PAIRS[feat]['pairs'][:n_pairs_max]
        log(f"  [{feat_idx+1}/{len(feature_names)}] {feat} ({FEATURE_PAIRS[feat]['type']}): {len(pairs)} pairs")
        
        for s1, s2 in pairs:
            with torch.no_grad():
                # s1
                enc1 = tokenizer(s1, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc1 = {k: v.to(device) for k, v in enc1.items()}
                cap_store.clear()
                out1 = model(**enc1)
                len1 = enc1['attention_mask'].sum().item()
                idx1 = max(0, len1 - 2)
                s1_act = cap_store[bottleneck_layer][0, idx1].float().numpy().copy()
                s1_logit = out1.logits[0, -1].detach().cpu().float().numpy().copy()
                
                # s2
                enc2 = tokenizer(s2, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc2 = {k: v.to(device) for k, v in enc2.items()}
                cap_store.clear()
                out2 = model(**enc2)
                len2 = enc2['attention_mask'].sum().item()
                idx2 = max(0, len2 - 2)
                s2_act = cap_store[bottleneck_layer][0, idx2].float().numpy().copy()
                s2_logit = out2.logits[0, -1].detach().cpu().float().numpy().copy()
                
                delta = s2_act - s1_act
                all_deltas[feat].append(delta)
                all_acts[feat]['s1'].append(s1_act)
                all_acts[feat]['s2'].append(s2_act)
                all_logits[feat]['s1'].append(s1_logit)
                all_logits[feat]['s2'].append(s2_logit)
                
                if model_name in ["glm4", "deepseek7b"]:
                    torch.cuda.empty_cache()
        
        n_valid = len(all_deltas[feat])
        log(f"    Collected {n_valid} pairs")
    
    handle.remove()
    
    # ============================================================
    # S2: Per-feature PCA + 全局PCA (排除formality)
    # ============================================================
    log(f"\n{'='*60}")
    log("S2: PCA分析 — Per-feature + 排除formality全局")
    log(f"{'='*60}")
    
    feat_pc1_dirs = {}
    feat_pc1_vars = {}
    for feat in feature_names:
        deltas = np.array(all_deltas[feat])
        if len(deltas) < 10:
            continue
        pca_feat = PCA(n_components=5)
        pca_feat.fit(deltas)
        feat_pc1_dirs[feat] = pca_feat.components_[0]
        feat_pc1_vars[feat] = pca_feat.explained_variance_ratio_[0]
        log(f"  {feat}: PC1={pca_feat.explained_variance_ratio_[0]*100:.2f}%, PC2={pca_feat.explained_variance_ratio_[1]*100:.2f}%")
    
    # 排除formality的全局PCA
    non_form_feats = [f for f in feature_names if f != 'formality']
    no_form_deltas = np.concatenate([all_deltas[f] for f in non_form_feats], axis=0)
    pca_no_form = PCA(n_components=10)
    pca_no_form.fit(no_form_deltas)
    no_form_pc1 = pca_no_form.components_[0]
    no_form_pc1_var = pca_no_form.explained_variance_ratio_[0]
    log(f"\n  排除formality全局PCA: PC1={no_form_pc1_var*100:.2f}%")
    
    # ============================================================
    # S3: Interchange Intervention — 差分投影比 + Top-k logit效应
    # ============================================================
    log(f"\n{'='*60}")
    log("S3: Interchange Intervention — 差分投影比 + Top-k logit效应")
    log(f"{'='*60}")
    
    # 方法1: 差分投影比 (不需要lm_head, 不会OOM)
    # PC1分量在差分中的占比 = |delta·pc1| / ||delta||
    # 方法2: Top-k logit效应 (只计算top-k token, 避免全vocab矩阵乘法)
    
    # 先计算投影比
    log(f"\n  === 差分投影比 ===")
    log(f"  {'Feature':<20} {'PC1_proj%':>10} {'PC2_proj%':>10} {'Rand_proj%':>10} {'PC1/R':>8} {'PC2/R':>8}")
    log(f"  {'-'*70}")
    
    results_per_feat = {}
    
    for feat in feature_names:
        deltas = np.array(all_deltas[feat])
        
        if len(deltas) < 10:
            continue
        
        pc1_dir = feat_pc1_dirs[feat]
        
        pca_full = PCA(n_components=5)
        pca_full.fit(deltas)
        pc2_dir = pca_full.components_[1]
        
        np.random.seed(42)
        rand_dir = np.random.randn(d_model)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        
        pc1_projs = []
        pc2_projs = []
        rand_projs = []
        
        for delta in deltas:
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-8:
                continue
            
            pc1_proj = abs(np.dot(delta, pc1_dir)) / delta_norm
            pc2_proj = abs(np.dot(delta, pc2_dir)) / delta_norm
            rand_proj = abs(np.dot(delta, rand_dir)) / delta_norm
            
            pc1_projs.append(pc1_proj)
            pc2_projs.append(pc2_proj)
            rand_projs.append(rand_proj)
        
        if not pc1_projs:
            continue
        
        pc1_mean = np.mean(pc1_projs)
        pc2_mean = np.mean(pc2_projs)
        rand_mean = np.mean(rand_projs)
        pc1_vs_rand = pc1_mean / rand_mean if rand_mean > 0 else 0
        pc2_vs_rand = pc2_mean / rand_mean if rand_mean > 0 else 0
        
        log(f"  {feat:<20} {pc1_mean*100:>9.2f}% {pc2_mean*100:>9.2f}% {rand_mean*100:>9.2f}% {pc1_vs_rand:>7.2f}x {pc2_vs_rand:>7.2f}x")
        
        results_per_feat[feat] = {
            'pc1_proj_pct': float(pc1_mean),
            'pc2_proj_pct': float(pc2_mean),
            'random_proj_pct': float(rand_mean),
            'pc1_vs_random': float(pc1_vs_rand),
            'pc2_vs_random': float(pc2_vs_rand),
            'n_pairs': len(pc1_projs),
        }
    
    # 方法2: Top-k logit效应 (只对top-50 token计算, 避免OOM)
    log(f"\n  === Top-50 logit效应 (lm_head线性近似) ===")
    
    try:
        lm_head = model.lm_head
        W_lm = lm_head.weight.detach().cpu().float().numpy()  # (vocab, d_model)
        log(f"  lm_head weight shape: {W_lm.shape}")
        
        log(f"  {'Feature':<20} {'PC1_eff%':>10} {'PC2_eff%':>10} {'Rand_eff%':>10} {'PC1/R':>8} {'PC2/R':>8}")
        log(f"  {'-'*70}")
        
        for feat in feature_names:
            deltas = np.array(all_deltas[feat])
            logits_s1 = np.array(all_logits[feat]['s1'])
            logits_s2 = np.array(all_logits[feat]['s2'])
            
            if len(deltas) < 10:
                continue
            
            pc1_dir = feat_pc1_dirs[feat]
            pca_full = PCA(n_components=5)
            pca_full.fit(deltas)
            pc2_dir = pca_full.components_[1]
            
            np.random.seed(42)
            rand_dir = np.random.randn(d_model)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            
            pc1_effects = []
            pc2_effects = []
            random_effects = []
            
            for i in range(len(deltas)):
                delta = deltas[i]
                orig_logit_diff = logits_s2[i] - logits_s1[i]
                
                # 找top-50变化的token
                top_idx = np.argsort(np.abs(orig_logit_diff))[-50:]
                orig_top_norm = np.linalg.norm(orig_logit_diff[top_idx])
                
                if orig_top_norm < 1e-6:
                    continue
                
                # 只用top-50行做矩阵乘法
                W_top = W_lm[top_idx]  # (50, d_model)
                
                pc1_proj = np.dot(delta, pc1_dir)
                pc1_effect = np.linalg.norm(W_top @ (pc1_proj * pc1_dir))
                pc1_effect_pct = pc1_effect / orig_top_norm
                pc1_effects.append(pc1_effect_pct)
                
                pc2_proj = np.dot(delta, pc2_dir)
                pc2_effect = np.linalg.norm(W_top @ (pc2_proj * pc2_dir))
                pc2_effect_pct = pc2_effect / orig_top_norm
                pc2_effects.append(pc2_effect_pct)
                
                rand_proj = np.dot(delta, rand_dir)
                rand_effect = np.linalg.norm(W_top @ (rand_proj * rand_dir))
                rand_effect_pct = rand_effect / orig_top_norm
                random_effects.append(rand_effect_pct)
            
            if not pc1_effects:
                continue
            
            pc1_mean = np.mean(pc1_effects)
            pc2_mean = np.mean(pc2_effects)
            rand_mean = np.mean(random_effects)
            pc1_vs_rand = pc1_mean / rand_mean if rand_mean > 0 else 0
            pc2_vs_rand = pc2_mean / rand_mean if rand_mean > 0 else 0
            
            log(f"  {feat:<20} {pc1_mean*100:>9.2f}% {pc2_mean*100:>9.2f}% {rand_mean*100:>9.2f}% {pc1_vs_rand:>7.2f}x {pc2_vs_rand:>7.2f}x")
            
            if feat in results_per_feat:
                results_per_feat[feat]['logit_pc1_pct'] = float(pc1_mean)
                results_per_feat[feat]['logit_pc2_pct'] = float(pc2_mean)
                results_per_feat[feat]['logit_random_pct'] = float(rand_mean)
                results_per_feat[feat]['logit_pc1_vs_random'] = float(pc1_vs_rand)
    except Exception as e:
        log(f"  lm_head计算失败: {e}")
        log(f"  (使用差分投影比作为替代)")
    
    # 释放lm_head内存
    if 'W_lm' in dir():
        del W_lm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ============================================================
    # S4: 排除formality的全局PC1干预
    # ============================================================
    log(f"\n{'='*60}")
    log("S4: 排除formality的全局PC1干预 — 验证共享方向")
    log(f"{'='*60}")
    
    # 使用差分投影比 (不需要lm_head)
    log(f"  {'Feature':<20} {'NoFormPC1_proj%':>15} {'PerFeatPC1_proj%':>16} {'Ratio':>8}")
    log(f"  {'-'*60}")
    
    for feat in feature_names:
        deltas = np.array(all_deltas[feat])
        
        if len(deltas) < 10:
            continue
        
        pc1_dir = feat_pc1_dirs[feat]
        
        noform_projs = []
        perfeat_projs = []
        
        for delta in deltas:
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-8:
                continue
            
            nf_proj = abs(np.dot(delta, no_form_pc1)) / delta_norm
            pf_proj = abs(np.dot(delta, pc1_dir)) / delta_norm
            noform_projs.append(nf_proj)
            perfeat_projs.append(pf_proj)
        
        if not noform_projs:
            continue
        
        nf_mean = np.mean(noform_projs)
        pf_mean = np.mean(perfeat_projs)
        ratio = nf_mean / pf_mean if pf_mean > 0 else 0
        
        log(f"  {feat:<20} {nf_mean*100:>14.2f}% {pf_mean*100:>15.2f}% {ratio:>7.2f}")
    
    # ============================================================
    # S5: 跨特征Interchange — 用特征A的PC1方向干预特征B
    # ============================================================
    log(f"\n{'='*60}")
    log("S5: 跨特征Interchange — PC1方向的迁移效应")
    log(f"{'='*60}")
    
    # 用差分投影比衡量跨特征迁移
    cross_results = {}
    
    for feat_a in feature_names:
        for feat_b in feature_names:
            if feat_a == feat_b:
                continue
            if feat_a not in feat_pc1_dirs or feat_b not in feat_pc1_dirs:
                continue
            
            pc1_a = feat_pc1_dirs[feat_a]
            deltas_b = np.array(all_deltas[feat_b])
            
            if len(deltas_b) < 10:
                continue
            
            cross_projs = []
            for delta in deltas_b:
                delta_norm = np.linalg.norm(delta)
                if delta_norm < 1e-8:
                    continue
                proj_a = abs(np.dot(delta, pc1_a)) / delta_norm
                cross_projs.append(proj_a)
            
            if cross_projs:
                cross_results[(feat_a, feat_b)] = np.mean(cross_projs)
    
    # 展示: 对每个特征B, 哪个特征A的PC1方向对它最有效
    log(f"\n  对每个目标特征, 其他特征PC1方向的干预效应:")
    log(f"  {'Target':<20} {'Best_source':>14} {'Best_effect%':>13} {'Self_effect%':>13} {'Transfer_ratio':>15}")
    log(f"  {'-'*75}")
    
    for feat_b in feature_names:
        if feat_b not in feat_pc1_dirs:
            continue
        
        # 找到对feat_b最有效的源特征
        best_src = None
        best_eff = 0
        for feat_a in feature_names:
            if feat_a == feat_b:
                continue
            key = (feat_a, feat_b)
            if key in cross_results and cross_results[key] > best_eff:
                best_eff = cross_results[key]
                best_src = feat_a
        
        # 自身PC1效应
        self_eff = results_per_feat[feat_b]['pc1_effect_pct'] if feat_b in results_per_feat else 0
        
        if best_src and self_eff > 0:
            transfer_ratio = best_eff / self_eff
            log(f"  {feat_b:<20} {best_src:>14} {best_eff*100:>12.2f}% {self_eff*100:>12.2f}% {transfer_ratio:>14.2f}")
    
    # 跨特征迁移统计
    syn_feats = [f for f in feature_names if FEATURE_PAIRS[f]['type'] == 'SYN']
    sem_feats = [f for f in feature_names if FEATURE_PAIRS[f]['type'] == 'SEM']
    
    syn_to_syn = [cross_results[(a, b)] for a in syn_feats for b in syn_feats if (a, b) in cross_results]
    sem_to_sem = [cross_results[(a, b)] for a in sem_feats for b in sem_feats if (a, b) in cross_results]
    syn_to_sem = [cross_results[(a, b)] for a in syn_feats for b in sem_feats if (a, b) in cross_results]
    sem_to_syn = [cross_results[(a, b)] for a in sem_feats for b in syn_feats if (a, b) in cross_results]
    
    log(f"\n  跨特征迁移统计:")
    if syn_to_syn:
        log(f"    SYN→SYN: mean={np.mean(syn_to_syn)*100:.2f}%, n={len(syn_to_syn)}")
    if sem_to_sem:
        log(f"    SEM→SEM: mean={np.mean(sem_to_sem)*100:.2f}%, n={len(sem_to_sem)}")
    if syn_to_sem:
        log(f"    SYN→SEM: mean={np.mean(syn_to_sem)*100:.2f}%, n={len(syn_to_sem)}")
    if sem_to_syn:
        log(f"    SEM→SYN: mean={np.mean(sem_to_syn)*100:.2f}%, n={len(sem_to_syn)}")
    
    # ============================================================
    # S6: 完整Interchange — 激活替换验证 (BF16模型)
    # ============================================================
    log(f"\n{'='*60}")
    log("S6: 激活替换验证 (仅BF16模型支持in-place修改)")
    log(f"{'='*60}")
    
    if model_name == "qwen3":
        # 对Qwen3做真正的激活替换
        n_interv = min(30, min(len(all_deltas[f]) for f in feature_names))
        log(f"  使用 {n_interv} 对/特征 做激活替换")
        
        # 重新注册hook
        cap_store3 = {}
        
        for feat in feature_names:
            if feat not in feat_pc1_dirs:
                continue
            
            pc1_dir = feat_pc1_dirs[feat]
            pc1_torch = torch.tensor(pc1_dir, dtype=torch.bfloat16)
            
            pairs = FEATURE_PAIRS[feat]['pairs'][:n_interv]
            
            orig_logit_diffs = []
            ablated_logit_diffs = []
            swapped_logit_diffs = []
            
            log(f"\n  [{feat}] 激活替换干预 ({n_interv} pairs):")
            
            for pair_idx, (s1, s2) in enumerate(pairs):
                with torch.no_grad():
                    # s1 baseline
                    enc1 = tokenizer(s1, return_tensors='pt', padding=True, truncation=True, max_length=64)
                    enc1 = {k: v.to(device) for k, v in enc1.items()}
                    cap_store3.clear()
                    h = model.model.layers[bottleneck_layer].register_forward_hook(make_hook(bottleneck_layer, cap_store3))
                    out1 = model(**enc1)
                    h.remove()
                    logits1_orig = out1.logits[0, -1].detach().cpu().float()
                    len1 = enc1['attention_mask'].sum().item()
                    idx1 = max(0, len1 - 2)
                    act1 = cap_store3[bottleneck_layer][0, idx1].clone()
                    
                    # s2 baseline
                    enc2 = tokenizer(s2, return_tensors='pt', padding=True, truncation=True, max_length=64)
                    enc2 = {k: v.to(device) for k, v in enc2.items()}
                    cap_store3.clear()
                    h = model.model.layers[bottleneck_layer].register_forward_hook(make_hook(bottleneck_layer, cap_store3))
                    out2 = model(**enc2)
                    h.remove()
                    logits2_orig = out2.logits[0, -1].detach().cpu().float()
                    len2 = enc2['attention_mask'].sum().item()
                    idx2 = max(0, len2 - 2)
                    act2 = cap_store3[bottleneck_layer][0, idx2].clone()
                    
                    orig_diff = torch.norm(logits2_orig - logits1_orig).item()
                    orig_logit_diffs.append(orig_diff)
                    
                    # PC1消藻: 将s2在PC1方向上的投影替换为s1的
                    # act2_swapped = act2 - (act2·pc1 - act1·pc1) * pc1
                    proj2 = torch.dot(act2.flatten().float(), pc1_torch.float().to(device))
                    proj1 = torch.dot(act1.flatten().float(), pc1_torch.float().to(device))
                    delta_proj = proj2 - proj1
                    
                    # 用hook替换: 在s2的前向中, 消除PC1差分
                    swap_amount = delta_proj * pc1_torch.to(device)
                    
                    def make_swap_hook(swap_amt, position_idx):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                out = output[0].clone()
                                out[0, position_idx] = out[0, position_idx] - swap_amt.to(out.dtype)
                                return (out,) + output[1:]
                            else:
                                out = output.clone()
                                out[0, position_idx] = out[0, position_idx] - swap_amt.to(out.dtype)
                                return out
                        return hook_fn
                    
                    # s2 with PC1 swapped
                    cap_store3.clear()
                    h = model.model.layers[bottleneck_layer].register_forward_hook(make_swap_hook(swap_amount, idx2))
                    out2_swapped = model(**enc2)
                    h.remove()
                    logits2_swapped = out2_swapped.logits[0, -1].detach().cpu().float()
                    
                    swapped_diff = torch.norm(logits2_swapped - logits1_orig).item()
                    swapped_logit_diffs.append(swapped_diff)
            
            if orig_logit_diffs:
                orig_mean = np.mean(orig_logit_diffs)
                swapped_mean = np.mean(swapped_logit_diffs)
                reduction = (orig_mean - swapped_mean) / orig_mean * 100 if orig_mean > 0 else 0
                log(f"    原始logit变化: {orig_mean:.4f}")
                log(f"    PC1替换后:     {swapped_mean:.4f}")
                log(f"    PC1效应(减少): {reduction:.2f}%")
                
                results_per_feat[feat]['activation_swap_effect'] = float(reduction)
    else:
        log(f"  8bit模型不支持in-place激活修改, 跳过")
        log(f"  (线性近似方法已足够验证因果效应)")
    
    # ============================================================
    # S7: 最终汇总
    # ============================================================
    log(f"\n{'='*60}")
    log("S7: Interchange Intervention — 最终汇总")
    log(f"{'='*60}")
    
    log(f"\n  === Per-feature PC1因果效应 (差分投影比) ===")
    log(f"  {'Feature':<20} {'PC1_proj%':>10} {'PC2_proj%':>10} {'Rand_proj%':>10} {'PC1/R':>8} {'PC2/R':>8}")
    log(f"  {'-'*70}")
    
    for feat in feature_names:
        if feat not in results_per_feat:
            continue
        r = results_per_feat[feat]
        log(f"  {feat:<20} {r['pc1_proj_pct']*100:>9.2f}% {r['pc2_proj_pct']*100:>9.2f}% {r['random_proj_pct']*100:>9.2f}% {r['pc1_vs_random']:>7.2f}x {r['pc2_vs_random']:>7.2f}x")
    
    # 平均效应
    avg_pc1 = np.mean([r['pc1_proj_pct'] for r in results_per_feat.values()])
    avg_pc2 = np.mean([r['pc2_proj_pct'] for r in results_per_feat.values()])
    avg_rand = np.mean([r['random_proj_pct'] for r in results_per_feat.values()])
    avg_pc1_vs_rand = avg_pc1 / avg_rand if avg_rand > 0 else 0
    
    log(f"\n  === 平均效应 ===")
    log(f"  PC1投影比: {avg_pc1*100:.2f}%")
    log(f"  PC2投影比: {avg_pc2*100:.2f}%")
    log(f"  随机投影比: {avg_rand*100:.2f}%")
    log(f"  PC1/Random: {avg_pc1_vs_rand:.2f}x")
    
    # 排除formality全局PC1效应
    log(f"\n  === 排除formality的全局PC1 ===")
    log(f"  全局PC1方差: {no_form_pc1_var*100:.2f}%")
    
    # 判断
    log(f"\n  === 因果判断 ===")
    
    if avg_pc1_vs_rand > 10:
        log(f"  PC1/Random={avg_pc1_vs_rand:.1f}x > 10 → PC1有强因果效应!")
    elif avg_pc1_vs_rand > 3:
        log(f"  PC1/Random={avg_pc1_vs_rand:.1f}x > 3 → PC1有中等因果效应")
    else:
        log(f"  PC1/Random={avg_pc1_vs_rand:.1f}x → PC1因果效应弱")
    
    if no_form_pc1_var > 0.5:
        log(f"  排除formalityPC1={no_form_pc1_var*100:.0f}% > 50% → 1D流形真实存在!")
    else:
        log(f"  排除formalityPC1={no_form_pc1_var*100:.0f}% → 信息分散, 无1D流形")
    
    log(f"\n  ======== CCXXI 完成 ========")
    
    # 保存结果JSON
    result_json = {
        'model': model_name,
        'bottleneck_layer': bottleneck_layer,
        'n_pairs': n_pairs_max,
        'no_form_pc1_var': float(no_form_pc1_var),
        'avg_pc1_effect_pct': float(avg_pc1),
        'avg_pc2_effect_pct': float(avg_pc2),
        'avg_random_effect_pct': float(avg_rand),
        'avg_pc1_vs_random': float(avg_pc1_vs_rand),
        'per_feat_results': {f: {k: float(v) for k, v in r.items()} for f, r in results_per_feat.items()},
    }
    
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    
    log(f"Results saved to {out_dir}/results.json")
    
    log_file.close()


if __name__ == "__main__":
    main()
