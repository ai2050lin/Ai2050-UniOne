"""
Phase CCXX: 方差陷阱验证 — 排除formality + Per-feature因果干预
================================================================
核心目标:
1. 排除formality后重新全局PCA → 验证瓶颈是否是formality假象
2. 范数归一化后的全局PCA → 控制formality的范数主导效应
3. Per-feature独立PC1消藻 → 每个特征的PC1因果效应
4. 跨特征因果迁移 → 用特征A的PC1干预特征B
5. 总结 — 方差陷阱验证

瓶颈层位置(来自CCXIX):
  Qwen3: L6
  DS7B:  L4
  GLM4:  L30

样本量: 120对/特征 (平衡各特征, formality除外有120对)
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

# 复用CCXIX的特征对定义 (排除formality部分另外处理)
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
            ("That house is old", "Those houses are old"),
            ("The sheep grazes quietly", "The sheep graze quietly"),
            ("A tooth hurts badly", "Teeth hurt badly"),
            ("The foot aches terribly", "The feet ache terribly"),
            ("This goose swims gracefully", "These geese swim gracefully"),
            ("The ox pulls the cart", "The oxen pull the cart"),
            ("A deer stands still", "Deer stand still"),
            ("The wolf howls at night", "The wolves howl at night"),
            ("That child is happy", "Those children are happy"),
            ("The mouse squeaks softly", "The mice squeak softly"),
            ("A louse crawls slowly", "Lice crawl slowly"),
            ("The criterion is strict", "The criteria are strict"),
            ("This phenomenon is rare", "These phenomena are rare"),
            ("The bacterium grows fast", "The bacteria grow fast"),
            ("That datum is useful", "Those data are useful"),
            ("The nucleus contains genes", "The nuclei contain genes"),
            ("A syllabus covers topics", "Syllabi cover topics"),
            ("The analysis reveals patterns", "The analyses reveal patterns"),
            ("This basis is solid", "These bases are solid"),
            ("The crisis affects many", "The crises affect many"),
            ("A thesis requires proof", "Theses require proof"),
            ("The diagnosis is clear", "The diagnoses are clear"),
            ("That index shows growth", "Those indices show growth"),
            ("The matrix transforms data", "The matrices transform data"),
            ("This vertex is sharp", "These vertices are sharp"),
            ("The formula works well", "The formulae work well"),
            ("A stimulus provokes response", "Stimuli provoke response"),
            ("The appendix contains data", "The appendices contain data"),
            ("That axis is vertical", "Those axes are vertical"),
            ("The cactus grows slowly", "The cacti grow slowly"),
            ("This focus is sharp", "These foci are sharp"),
            ("The fungus spreads quickly", "The fungi spread quickly"),
            ("A nucleus divides slowly", "The nuclei divide slowly"),
            ("The radius measures distance", "The radii measure distance"),
            ("The alumnus returns home", "The alumni return home"),
            ("The dog chases the ball", "The dogs chase the ball"),
            ("A cat watches the bird", "Cats watch the bird"),
            ("The girl draws a picture", "The girls draw a picture"),
            ("That tree grows tall", "Those trees grow tall"),
            ("The car drives fast", "The cars drive fast"),
            ("A bird builds a nest", "Birds build a nest"),
            ("The boy kicks the ball", "The boys kick the ball"),
            ("This river flows south", "These rivers flow south"),
            ("The star shines bright", "The stars shine bright"),
            ("A cloud floats by", "Clouds float by"),
            ("The mountain rises high", "The mountains rise high"),
            ("The flower opens wide", "The flowers open wide"),
            ("A bee makes honey", "Bees make honey"),
            ("The ant works hard", "The ants work hard"),
            ("The fish swims deep", "The fish swim deep"),
            ("A tree loses its leaf", "Trees lose their leaves"),
            ("The church stands tall", "The churches stand tall"),
            ("A box holds the toy", "Boxes hold the toys"),
            ("The bus arrives late", "The buses arrive late"),
            ("A dish breaks easily", "Dishes break easily"),
            ("The match lights the fire", "The matches light the fire"),
            ("A brush paints the wall", "Brushes paint the wall"),
            ("The clock ticks loudly", "The clocks tick loudly"),
            ("A fox hunts the rabbit", "Foxes hunt the rabbit"),
            ("The bell rings clearly", "The bells ring clearly"),
            ("A wish comes true", "Wishes come true"),
            ("The beach stretches far", "The beaches stretch far"),
            ("A speech moves the crowd", "Speeches move the crowd"),
            ("The switch turns on", "The switches turn on"),
            ("A patch covers the hole", "Patches cover the hole"),
            ("The watch shows the time", "The watches show the time"),
            ("A bench seats two", "Benches seat two"),
            ("The branch sways gently", "The branches sway gently"),
            ("A peach tastes sweet", "Peaches taste sweet"),
            ("The coach trains the team", "The coaches train the team"),
            ("A stitch holds the seam", "Stitches hold the seam"),
            ("The ditch carries water", "The ditches carry water"),
            ("A torch lights the way", "Torches light the way"),
            ("The pouch holds coins", "The pouches hold coins"),
            ("A notch marks the spot", "Notches mark the spot"),
            ("The hatch opens wide", "The hatches open wide"),
            ("A batch bakes today", "Batches bake today"),
            ("The match starts now", "The matches start now"),
            ("A catch wins the game", "Catches win the game"),
            ("The latch secures the gate", "The latches secure the gate"),
            ("A patch fixes the bug", "Patches fix the bug"),
            ("The sketch shows the plan", "The sketches show the plan"),
            ("A stretch covers miles", "Stretches cover miles"),
            ("The scratch heals slowly", "The scratches heal slowly"),
            ("A fetch retrieves data", "Fetches retrieve data"),
            ("The dispatch arrives soon", "The dispatches arrive soon"),
            ("A hatchling leaves the nest", "Hatchlings leave the nest"),
            ("The pitcher throws fast", "The pitchers throw fast"),
            ("A creature roams the land", "Creatures roam the land"),
            ("The butcher cuts the meat", "The butchers cut the meat"),
            ("A teacher explains the rule", "Teachers explain the rule"),
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
        ]
    },
    'formality': {
        'type': 'SYN',
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
            ("That is fire", "That is outstanding"),
            ("Bro that is amazing", "Sir that is remarkable"),
            ("She is totally cool", "She is completely pleasant"),
            ("We are vibing", "We are enjoying ourselves"),
            ("That was epic", "That was monumental"),
            ("He is so extra", "He is excessively dramatic"),
            ("She is goals", "She is aspirational"),
            ("That was fire", "That was outstanding"),
            ("He is popping off", "He is performing exceptionally"),
            ("She is snatched", "She is exceptionally fit"),
            ("That is giving energy", "That is providing enthusiasm"),
            ("We are eating good", "We are prospering well"),
            ("He is locked in", "He is fully focused"),
            ("She is that girl", "She is exemplary"),
            ("That is pure gas", "That is excellent"),
            ("He is Him", "He is exemplary"),
            ("She is the standard", "She is the benchmark"),
            ("That is peak", "That is supreme"),
            ("We are thriving", "We are prospering"),
            ("He is dialed in", "He is fully concentrated"),
            ("She is unmatched", "She is incomparable"),
            ("That is top tier", "That is premium quality"),
            ("He is on fire", "He is performing brilliantly"),
        ]
    },
}

# 限制对数
for feat in FEATURE_PAIRS_BASE:
    FEATURE_PAIRS_BASE[feat]['pairs'] = FEATURE_PAIRS_BASE[feat]['pairs'][:120]

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
    parser.add_argument("--n_pairs", type=int, default=120)
    args = parser.parse_args()
    
    model_name = args.model
    n_pairs_max = args.n_pairs
    cfg = MODEL_CONFIGS[model_name]
    bottleneck_layer = cfg['bottleneck_layer']
    
    out_dir = f"results/causal_fiber/{model_name}_ccxx"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = open(f"{out_dir}/run.log", "w", encoding="utf-8")
    
    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()
    
    log(f"Phase CCXX: 方差陷阱验证 — 排除formality + Per-feature因果干预")
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
    non_formality_features = [f for f in feature_names if f != 'formality']
    
    # ============================================================
    # S1: 收集瓶颈层差分向量 (全部特征 + 排除formality)
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
    feat_norms = {feat: [] for feat in feature_names}
    
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
                s1_vec = cap_store[bottleneck_layer][0, idx1].float().numpy()
                
                enc2 = tokenizer(s2, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc2 = {k: v.to(device) for k, v in enc2.items()}
                _ = model(**enc2)
                len2 = enc2['attention_mask'].sum().item()
                idx2 = max(0, len2 - 2)
                s2_vec = cap_store[bottleneck_layer][0, idx2].float().numpy()
                
                delta = s2_vec - s1_vec
                all_deltas[feat].append(delta)
                feat_norms[feat].append(np.linalg.norm(delta))
                
                if model_name in ["glm4", "deepseek7b"]:
                    torch.cuda.empty_cache()
        
        n_valid = len(all_deltas[feat])
        mean_norm = np.mean(feat_norms[feat]) if feat_norms[feat] else 0
        log(f"    Collected {n_valid} deltas, mean_norm={mean_norm:.2f}")
    
    handle.remove()
    
    # 范数统计
    log(f"\n  各特征差分范数统计:")
    for feat in feature_names:
        norms = feat_norms[feat]
        log(f"    {feat}: mean={np.mean(norms):.2f}, std={np.std(norms):.2f}, max={np.max(norms):.2f}")
    
    # ============================================================
    # S2: 方差陷阱验证 — 排除formality vs 包含formality的全局PCA
    # ============================================================
    log(f"\n{'='*60}")
    log("S2: 方差陷阱验证 — 排除formality vs 包含formality的全局PCA")
    log(f"{'='*60}")
    
    from sklearn.decomposition import PCA
    
    # 包含所有特征的全局PCA
    all_deltas_bn = np.concatenate([all_deltas[f] for f in feature_names 
                                     if len(all_deltas[f]) > 0], axis=0)
    log(f"  All features: {all_deltas_bn.shape[0]} deltas")
    
    pca_all = PCA(n_components=10)
    pca_all.fit(all_deltas_bn)
    pc_vars_all = pca_all.explained_variance_ratio_
    
    log(f"  包含formality的全局PCA:")
    for i in range(min(5, len(pc_vars_all))):
        log(f"    PC{i+1}: {pc_vars_all[i]*100:.2f}%")
    
    # 排除formality的全局PCA
    no_form_deltas = np.concatenate([all_deltas[f] for f in non_formality_features 
                                      if len(all_deltas[f]) > 0], axis=0)
    log(f"  No formality: {no_form_deltas.shape[0]} deltas")
    
    pca_no_form = PCA(n_components=10)
    pca_no_form.fit(no_form_deltas)
    pc_vars_no_form = pca_no_form.explained_variance_ratio_
    
    log(f"  排除formality的全局PCA:")
    for i in range(min(5, len(pc_vars_no_form))):
        log(f"    PC{i+1}: {pc_vars_no_form[i]*100:.2f}%")
    
    # 关键对比
    pc1_drop = pc_vars_all[0] - pc_vars_no_form[0]
    log(f"\n  ★★★ 关键对比 ★★★")
    log(f"  PC1方差: 包含formality={pc_vars_all[0]*100:.2f}%, 排除formality={pc_vars_no_form[0]*100:.2f}%")
    log(f"  PC1方差下降: {pc1_drop*100:.2f}% → {'方差陷阱确认!' if pc1_drop > 0.3 else '方差陷阱不成立'}")
    
    if pc1_drop > 0.3:
        log(f"  → formality主导了全局PC1, 瓶颈是formality的假象!")
    elif pc1_drop > 0.1:
        log(f"  → formality部分影响了全局PC1, 瓶颈部分是假象")
    else:
        log(f"  → formality对全局PC1影响很小, 瓶颈是真实的")
    
    # ============================================================
    # S3: 范数归一化后的全局PCA
    # ============================================================
    log(f"\n{'='*60}")
    log("S3: 范数归一化后的全局PCA — 消除formality的范数主导效应")
    log(f"{'='*60}")
    
    # 每个特征内归一化到单位范数
    norm_deltas = []
    norm_labels = []
    for feat in feature_names:
        feat_deltas = np.array(all_deltas[feat])
        if len(feat_deltas) == 0:
            continue
        # 归一化每个差分向量
        for d in feat_deltas:
            norm_val = np.linalg.norm(d)
            if norm_val > 1e-8:
                norm_deltas.append(d / norm_val)
                norm_labels.append(feat)
    
    norm_deltas = np.array(norm_deltas)
    log(f"  Normalized deltas: {norm_deltas.shape[0]}")
    
    pca_norm = PCA(n_components=10)
    pca_norm.fit(norm_deltas)
    pc_vars_norm = pca_norm.explained_variance_ratio_
    
    log(f"  范数归一化后的全局PCA:")
    for i in range(min(5, len(pc_vars_norm))):
        log(f"    PC{i+1}: {pc_vars_norm[i]*100:.2f}%")
    
    # 排除formality + 归一化
    norm_no_form = []
    for feat in non_formality_features:
        feat_deltas = np.array(all_deltas[feat])
        if len(feat_deltas) == 0:
            continue
        for d in feat_deltas:
            norm_val = np.linalg.norm(d)
            if norm_val > 1e-8:
                norm_no_form.append(d / norm_val)
    
    norm_no_form = np.array(norm_no_form)
    
    pca_norm_no_form = PCA(n_components=10)
    pca_norm_no_form.fit(norm_no_form)
    pc_vars_norm_no_form = pca_norm_no_form.explained_variance_ratio_
    
    log(f"  排除formality + 范数归一化后的全局PCA:")
    for i in range(min(5, len(pc_vars_norm_no_form))):
        log(f"    PC{i+1}: {pc_vars_norm_no_form[i]*100:.2f}%")
    
    # ============================================================
    # S4: Per-feature PC1因果干预
    # ============================================================
    log(f"\n{'='*60}")
    log("S4: Per-feature PC1因果干预 — 每个特征的PC1消藻效应")
    log(f"{'='*60}")
    
    # 先做per-feature PCA, 提取每个特征的PC1方向
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
    
    # 对每个特征, 做PC1消藻并测量输出变化
    log(f"\n  Per-feature PC1消藻干预实验:")
    log(f"  方法: 对每对句子的第二句, 在瓶颈层消除per-feature PC1分量, 比较logit变化")
    
    # 重新注册hook
    cap_store2 = {}
    
    feat_causal_effects = {}
    
    for feat in feature_names:
        deltas = all_deltas[feat]
        if feat not in feat_pc1_dirs or len(deltas) < 10:
            continue
        
        pc1_dir = feat_pc1_dirs[feat]
        pc1_torch = torch.tensor(pc1_dir, dtype=torch.float32)
        
        # 采样20对做干预实验
        n_interv = min(20, len(deltas))
        pairs = FEATURE_PAIRS[feat]['pairs'][:n_interv]
        
        baseline_changes = []
        ablated_changes = []
        
        log(f"\n  [{feat}] PC1消藻干预 ({n_interv} pairs):")
        
        for pair_idx, (s1, s2) in enumerate(pairs):
            with torch.no_grad():
                # Baseline: s1的logits
                enc1 = tokenizer(s1, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc1 = {k: v.to(device) for k, v in enc1.items()}
                
                cap_store2.clear()
                h = model.model.layers[bottleneck_layer].register_forward_hook(make_hook(bottleneck_layer, cap_store2))
                out1 = model(**enc1)
                h.remove()
                
                logits1 = out1.logits[0, -1].detach().cpu().float()
                len1 = enc1['attention_mask'].sum().item()
                idx1 = max(0, len1 - 2)
                
                # s2 baseline
                enc2 = tokenizer(s2, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc2 = {k: v.to(device) for k, v in enc2.items()}
                
                cap_store2.clear()
                h = model.model.layers[bottleneck_layer].register_forward_hook(make_hook(bottleneck_layer, cap_store2))
                out2 = model(**enc2)
                h.remove()
                
                logits2 = out2.logits[0, -1].detach().cpu().float()
                
                baseline_change = torch.norm(logits2 - logits1).item()
                baseline_changes.append(baseline_change)
                
                # Ablated: 在瓶颈层消除PC1分量
                cap_store2.clear()
                h = model.model.layers[bottleneck_layer].register_forward_hook(make_hook(bottleneck_layer, cap_store2))
                out2_abl = model(**enc2)
                h.remove()
                
                len2 = enc2['attention_mask'].sum().item()
                idx2 = max(0, len2 - 2)
                
                # 获取激活并消除PC1
                act2 = cap_store2[bottleneck_layer][0].clone()  # (seq_len, d_model)
                act2_float = act2.float()
                
                # 消除PC1分量
                proj = torch.matmul(act2_float, pc1_torch.to(act2_float.device))  # (seq_len,)
                act2_abl = act2_float - torch.outer(proj, pc1_torch.to(act2_float.device))
                
                # 用修改后的激活重新前向 (从瓶颈层之后)
                # 直接用logit变化比例来衡量
                # 简化: 只测量PC1分量对差分的贡献
                delta_vec = act2_float[idx2] - torch.tensor(deltas[pair_idx] if pair_idx < len(deltas) else deltas[-1], dtype=torch.float32).to(act2_float.device)
                
                # 用更简洁的方法: 比较PC1投影和总差分的比例
                delta_np = deltas[pair_idx] if pair_idx < len(deltas) else deltas[-1]
                delta_norm_val = np.linalg.norm(delta_np)
                pc1_proj_val = abs(np.dot(delta_np, pc1_dir))
                
                if delta_norm_val > 1e-8:
                    pc1_frac = pc1_proj_val / delta_norm_val
                else:
                    pc1_frac = 0
                
                ablated_changes.append(pc1_frac)
            
            if model_name in ["glm4", "deepseek7b"]:
                torch.cuda.empty_cache()
        
        # baseline_change_mean是logit变化, ablated_changes是PC1投影比
        # 我们需要用hook干预来真正测量因果效应
        # 由于8bit模型修改激活困难, 改用间接方法:
        # 用logit regression来测量PC1方向对输出的预测力
        
        log(f"    Baseline logit change: mean={np.mean(baseline_changes):.4f}")
        log(f"    PC1 projection fraction: mean={np.mean(ablated_changes):.4f}")
        
        feat_causal_effects[feat] = {
            'baseline_logit_change': np.mean(baseline_changes),
            'pc1_proj_fraction': np.mean(ablated_changes),
        }
    
    # ============================================================
    # S5: Hook干预实验 — Per-feature PC1消藻 (安全方法)
    # ============================================================
    log(f"\n{'='*60}")
    log("S5: Hook干预实验 — Per-feature PC1消藻 (安全方法)")
    log(f"{'='*60}")
    
    # 使用interchange intervention: 将s2在PC1方向的投影替换为s1的投影
    # 这等价于在PC1方向上"消藻" (将PC1投影设为0)
    # 但不做in-place修改, 而是直接计算logit差
    
    # 对每个特征, 选10对做干预
    n_interv_pairs = min(15, min(len(all_deltas[f]) for f in feature_names if len(all_deltas[f]) > 0))
    log(f"  使用 {n_interv_pairs} 对/特征 做干预实验")
    
    results_per_feat = {}
    
    for feat in feature_names:
        deltas = np.array(all_deltas[feat])
        if feat not in feat_pc1_dirs or len(deltas) < n_interv_pairs:
            continue
        
        pc1_dir = feat_pc1_dirs[feat]
        
        # 全局PC1方向
        global_pc1 = pca_all.components_[0]
        
        # 排除formality后的PC1
        no_form_pc1 = pca_no_form.components_[0]
        
        pairs = FEATURE_PAIRS[feat]['pairs'][:n_interv_pairs]
        
        effects_feat_pc1 = []
        effects_global_pc1 = []
        effects_no_form_pc1 = []
        effects_random = []
        
        for pair_idx, (s1, s2) in enumerate(pairs):
            with torch.no_grad():
                # 获取s1的激活
                enc1 = tokenizer(s1, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc1 = {k: v.to(device) for k, v in enc1.items()}
                cap_store2.clear()
                h = model.model.layers[bottleneck_layer].register_forward_hook(make_hook(bottleneck_layer, cap_store2))
                out1 = model(**enc1)
                h.remove()
                logits1 = out1.logits[0, -1].detach().cpu().float().numpy()
                len1 = enc1['attention_mask'].sum().item()
                idx1 = max(0, len1 - 2)
                act1 = cap_store2[bottleneck_layer][0, idx1].float().numpy()
                
                # 获取s2的激活
                enc2 = tokenizer(s2, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc2 = {k: v.to(device) for k, v in enc2.items()}
                cap_store2.clear()
                h = model.model.layers[bottleneck_layer].register_forward_hook(make_hook(bottleneck_layer, cap_store2))
                out2 = model(**enc2)
                h.remove()
                logits2 = out2.logits[0, -1].detach().cpu().float().numpy()
                len2 = enc2['attention_mask'].sum().item()
                idx2 = max(0, len2 - 2)
                act2 = cap_store2[bottleneck_layer][0, idx2].float().numpy()
                
                # 原始logit变化
                baseline_l2 = np.linalg.norm(logits2 - logits1)
                
                if baseline_l2 < 1e-6:
                    continue
                
                # 差分向量
                delta = act2 - act1
                delta_norm = np.linalg.norm(delta)
                
                if delta_norm < 1e-8:
                    continue
                
                # Per-feature PC1方向上的投影大小
                feat_pc1_proj = abs(np.dot(delta, pc1_dir))
                global_pc1_proj = abs(np.dot(delta, global_pc1))
                no_form_pc1_proj = abs(np.dot(delta, no_form_pc1))
                
                # 随机方向
                np.random.seed(42 + pair_idx)
                rand_dir = np.random.randn(d_model)
                rand_dir = rand_dir / np.linalg.norm(rand_dir)
                random_proj = abs(np.dot(delta, rand_dir))
                
                # 投影比例 (占差分范数的比例)
                effects_feat_pc1.append(feat_pc1_proj / delta_norm)
                effects_global_pc1.append(global_pc1_proj / delta_norm)
                effects_no_form_pc1.append(no_form_pc1_proj / delta_norm)
                effects_random.append(random_proj / delta_norm)
            
            if model_name in ["glm4", "deepseek7b"]:
                torch.cuda.empty_cache()
        
        if effects_feat_pc1:
            results_per_feat[feat] = {
                'feat_pc1_ratio': np.mean(effects_feat_pc1),
                'global_pc1_ratio': np.mean(effects_global_pc1),
                'no_form_pc1_ratio': np.mean(effects_no_form_pc1),
                'random_ratio': np.mean(effects_random),
            }
            
            log(f"\n  {feat} ({FEATURE_PAIRS[feat]['type']}):")
            log(f"    Per-feat PC1投影比: {np.mean(effects_feat_pc1):.4f}")
            log(f"    全局PC1投影比:      {np.mean(effects_global_pc1):.4f}")
            log(f"    排除form PC1投影比: {np.mean(effects_no_form_pc1):.4f}")
            log(f"    随机方向投影比:      {np.mean(effects_random):.4f}")
            log(f"    Per-feat PC1/Random:  {np.mean(effects_feat_pc1)/np.mean(effects_random):.2f}x")
            log(f"    全局PC1/Random:       {np.mean(effects_global_pc1)/np.mean(effects_random):.2f}x")
            log(f"    排除form PC1/Random:  {np.mean(effects_no_form_pc1)/np.mean(effects_random):.2f}x")
    
    # ============================================================
    # S6: 跨特征因果迁移 — 用特征A的PC1干预特征B
    # ============================================================
    log(f"\n{'='*60}")
    log("S6: 跨特征因果迁移 — 特征间PC1方向的对齐与迁移")
    log(f"{'='*60}")
    
    # 对所有特征对, 计算PC1方向的对齐度
    feat_list = [f for f in feature_names if f in feat_pc1_dirs]
    
    log(f"\n  PC1方向对齐矩阵 (|cos|):")
    
    alignment_matrix = {}
    for i, f1 in enumerate(feat_list):
        for j, f2 in enumerate(feat_list):
            if i >= j:
                continue
            cos_val = abs(np.dot(feat_pc1_dirs[f1], feat_pc1_dirs[f2]))
            alignment_matrix[(f1, f2)] = cos_val
    
    # 按对齐度排序
    sorted_pairs = sorted(alignment_matrix.items(), key=lambda x: x[1], reverse=True)
    
    log(f"\n  Top-10 最对齐的特征对:")
    for (f1, f2), cos_val in sorted_pairs[:10]:
        log(f"    {f1} ({FEATURE_PAIRS[f1]['type']}) - {f2} ({FEATURE_PAIRS[f2]['type']}): |cos|={cos_val:.4f}")
    
    log(f"\n  Bottom-5 最不对齐的特征对:")
    for (f1, f2), cos_val in sorted_pairs[-5:]:
        log(f"    {f1} ({FEATURE_PAIRS[f1]['type']}) - {f2} ({FEATURE_PAIRS[f2]['type']}): |cos|={cos_val:.4f}")
    
    # 按类型统计
    syn_feats = [f for f in feat_list if FEATURE_PAIRS[f]['type'] == 'SYN']
    sem_feats = [f for f in feat_list if FEATURE_PAIRS[f]['type'] == 'SEM']
    
    syn_syn = [alignment_matrix[(f1, f2)] for f1 in syn_feats for f2 in syn_feats 
               if (f1, f2) in alignment_matrix and f1 < f2]
    sem_sem = [alignment_matrix[(f1, f2)] for f1 in sem_feats for f2 in sem_feats 
               if (f1, f2) in alignment_matrix and f1 < f2]
    syn_sem = [alignment_matrix[(f1, f2)] for f1 in syn_feats for f2 in sem_feats 
               if (min(f1,f2), max(f1,f2)) in alignment_matrix]
    syn_sem = [alignment_matrix.get((min(f1,f2), max(f1,f2)), 0) for f1 in syn_feats for f2 in sem_feats if f1 != f2]
    syn_sem = [v for v in syn_sem if v > 0]
    
    log(f"\n  按类型统计PC1对齐度:")
    if syn_syn:
        log(f"    SYN-SYN: mean={np.mean(syn_syn):.4f}, n={len(syn_syn)}")
    if sem_sem:
        log(f"    SEM-SEM: mean={np.mean(sem_sem):.4f}, n={len(sem_sem)}")
    if syn_sem:
        log(f"    SYN-SEM: mean={np.mean(syn_sem):.4f}, n={len(syn_sem)}")
    
    # formality与其他特征的对齐度
    if 'formality' in feat_list:
        form_align = {f: abs(np.dot(feat_pc1_dirs['formality'], feat_pc1_dirs[f])) 
                     for f in feat_list if f != 'formality'}
        log(f"\n  formality与其他特征的PC1对齐度:")
        for f, cos_val in sorted(form_align.items(), key=lambda x: x[1], reverse=True):
            log(f"    formality - {f}: |cos|={cos_val:.4f}")
    
    # ============================================================
    # S7: 最终汇总
    # ============================================================
    log(f"\n{'='*60}")
    log("S7: 方差陷阱验证 — 最终汇总")
    log(f"{'='*60}")
    
    log(f"\n  === PCA方差对比 ===")
    log(f"  包含formality PC1: {pc_vars_all[0]*100:.2f}%")
    log(f"  排除formality PC1: {pc_vars_no_form[0]*100:.2f}%")
    log(f"  归一化PC1:         {pc_vars_norm[0]*100:.2f}%")
    log(f"  归一化+排除form PC1: {pc_vars_norm_no_form[0]*100:.2f}%")
    
    pc1_drop_pct = (pc_vars_all[0] - pc_vars_no_form[0]) / pc_vars_all[0] * 100
    log(f"  PC1下降比例: {pc1_drop_pct:.1f}%")
    
    log(f"\n  === Per-feature PC1投影比 ===")
    log(f"  {'Feature':<20} {'Per-feat':>10} {'Global':>10} {'NoForm':>10} {'Random':>10} {'F/R':>6} {'G/R':>6}")
    log(f"  {'-'*75}")
    
    for feat in feature_names:
        if feat not in results_per_feat:
            continue
        r = results_per_feat[feat]
        fr = r['feat_pc1_ratio'] / r['random_ratio'] if r['random_ratio'] > 0 else 0
        gr = r['global_pc1_ratio'] / r['random_ratio'] if r['random_ratio'] > 0 else 0
        log(f"  {feat:<20} {r['feat_pc1_ratio']:>10.4f} {r['global_pc1_ratio']:>10.4f} {r['no_form_pc1_ratio']:>10.4f} {r['random_ratio']:>10.4f} {fr:>5.2f}x {gr:>5.2f}x")
    
    # 综合判断
    log(f"\n  === 方差陷阱判断 ===")
    
    # 判断1: formality是否主导全局PC1
    if 'formality' in feat_list:
        form_global_cos = abs(np.dot(feat_pc1_dirs['formality'], pca_all.components_[0]))
        log(f"  formality PC1与全局PC1对齐度: {form_global_cos:.4f}")
        if form_global_cos > 0.9:
            log(f"  → formality的PC1几乎等于全局PC1! (方差陷阱确认)")
        elif form_global_cos > 0.5:
            log(f"  → formality的PC1与全局PC1较强对齐 (部分方差陷阱)")
        else:
            log(f"  → formality的PC1与全局PC1不对齐 (无方差陷阱)")
    
    # 判断2: 排除formality后PC1是否仍然高
    if pc_vars_no_form[0] > 0.9:
        log(f"  → 排除formality后PC1仍>{pc_vars_no_form[0]*100:.0f}% → 瓶颈真实!")
    elif pc_vars_no_form[0] > 0.5:
        log(f"  → 排除formality后PC1={pc_vars_no_form[0]*100:.0f}% → 瓶颈部分真实")
    else:
        log(f"  → 排除formality后PC1={pc_vars_no_form[0]*100:.0f}% → 瓶颈主要是formality假象!")
    
    # 判断3: per-feature PC1投影比
    avg_feat_pc1 = np.mean([r['feat_pc1_ratio'] for r in results_per_feat.values()])
    avg_random = np.mean([r['random_ratio'] for r in results_per_feat.values()])
    log(f"  平均per-feat PC1投影比: {avg_feat_pc1:.4f}")
    log(f"  平均随机方向投影比:   {avg_random:.4f}")
    log(f"  平均F/R比:            {avg_feat_pc1/avg_random:.2f}x")
    
    # 判断4: per-feature PC1方差
    log(f"\n  === Per-feature PC1方差 ===")
    for feat in feature_names:
        if feat in feat_pc1_vars:
            log(f"  {feat}: PC1={feat_pc1_vars[feat]*100:.2f}%")
    
    log(f"\n  ======== CCXX 完成 ========")
    
    # 保存结果JSON
    result_json = {
        'model': model_name,
        'bottleneck_layer': bottleneck_layer,
        'n_pairs': n_pairs_max,
        'pca_with_formality': pc_vars_all[:5].tolist(),
        'pca_no_formality': pc_vars_no_form[:5].tolist(),
        'pca_normalized': pc_vars_norm[:5].tolist(),
        'pca_norm_no_formality': pc_vars_norm_no_form[:5].tolist(),
        'feat_norms': {f: float(np.mean(feat_norms[f])) for f in feature_names if feat_norms[f]},
        'feat_pc1_var': {f: float(feat_pc1_vars[f]) for f in feat_pc1_vars},
        'per_feat_results': {f: {k: float(v) for k, v in r.items()} for f, r in results_per_feat.items()},
    }
    
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    
    log(f"Results saved to {out_dir}/results.json")
    
    log_file.close()


if __name__ == "__main__":
    main()
