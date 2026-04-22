"""
Phase CCXVIII: 精细层扫描 + CCA子空间度量 + 跨层旋转追踪
================================================================
核心目标:
1. 每2层扫描(而非4层), 发现更精确的信息瓶颈位置
2. CCA子空间度量替代不稳定的centroid_cos
3. 跨层PC旋转追踪: PC_i在相邻层间的余弦相似度
4. 样本量增到80对(通过数据增强)

S1: 全层差分向量收集 (12特征 × 每2层 × 80对)
S2: 精细PCA演变 + 信息瓶颈精确定位
S3: CCA子空间度量 (语法/语义子空间的相关性)
S4: 跨层PC旋转追踪 (PC_i在层间的旋转角)
S5: Per-feature Head贡献矩阵 + 聚类
S6: 因果原子词典 + 统计总结
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

import torch

# ============================================================
# 特征对定义 — 扩展到80对(通过原始40对+变体)
# ============================================================
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
            # 变体(41-80): 通过添加不同主语/副词构造
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
            # 变体(41-80)
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
            ("The city grows rapidly", "The cities grow rapidly"),
            ("A story captures attention", "Stories capture attention"),
            ("The glass breaks easily", "Glasses break easily"),
            ("This photo shows the sunset", "These photos show the sunset"),
            ("The teacher explains the concept", "The teachers explain the concept"),
            ("A tree provides shade", "Trees provide shade"),
            ("The house stands on the hill", "The houses stand on the hill"),
            ("That watch keeps good time", "Those watches keep good time"),
            ("The fox hunts at night", "The foxes hunt at night"),
            ("A church stands on the corner", "Churches stand on the corner"),
            ("The potato grows underground", "The potatoes grow underground"),
            ("This life is precious", "These lives are precious"),
            ("The shelf holds many books", "The shelves hold many books"),
            ("A calf follows its mother", "Calves follow their mothers"),
            ("The thief steals the painting", "The thieves steal the painting"),
            ("That loaf is fresh", "Those loaves are fresh"),
            # 变体(41-80)
            ("The kitten chases the ball", "The kittens chase the ball"),
            ("A panda eats bamboo shoots", "Pandas eat bamboo shoots"),
            ("The butterfly lands on the flower", "The butterflies land on the flower"),
            ("This essay is well written", "These essays are well written"),
            ("The sheep grazes in the meadow", "The sheep graze in the meadow"),
            ("A deer runs through the forest", "Deer run through the forest"),
            ("The branch falls in the storm", "The branches fall in the storm"),
            ("That building is very tall", "Those buildings are very tall"),
            ("The hero saves the village", "The heroes save the village"),
            ("A cherry ripens in summer", "Cherries ripen in summer"),
            ("The child learns to read", "The children learn to read"),
            ("This berry is sweet and juicy", "These berries are sweet and juicy"),
            ("The lady walks her dog", "The ladies walk their dogs"),
            ("A goose crosses the road", "Geese cross the road"),
            ("The tooth aches terribly", "The teeth ache terribly"),
            ("That path leads to the river", "Those paths lead to the river"),
            ("The ox pulls the cart", "The oxen pull the cart"),
            ("A criterion determines quality", "Criteria determine quality"),
            ("The phenomenon puzzles scientists", "The phenomena puzzle scientists"),
            ("This basis is solid", "These bases are solid"),
            ("The louse crawls on the skin", "The lice crawl on the skin"),
            ("A fungus grows in the dark", "Fungi grow in the dark"),
            ("The nucleus contains DNA", "The nuclei contain DNA"),
            ("That syllabus covers grammar", "Those syllabi cover grammar"),
            ("The analysis reveals trends", "The analyses reveal trends"),
            ("A crisis demands attention", "Crises demand attention"),
            ("The thesis argues for reform", "The theses argue for reform"),
            ("This index helps navigation", "These indices help navigation"),
            ("The appendix provides data", "The appendices provide data"),
            ("A formula solves the equation", "Formulae solve the equation"),
            ("The stimulus triggers response", "The stimuli trigger response"),
            ("That datum is significant", "Those data are significant"),
            ("The corpus contains texts", "The corpora contain texts"),
            ("A genus includes species", "Genera include species"),
            ("The stratum reveals history", "The strata reveal history"),
            ("This axis shows values", "These axes show values"),
            ("The medium conveys information", "The media convey information"),
            ("A curriculum guides learning", "Curricula guide learning"),
            ("The memorandum explains policy", "The memoranda explain policy"),
            ("That vertex connects edges", "Those vertices connect edges"),
        ]
    },
    'negation': {
        'type': 'SYN',
        'pairs': [
            ("She always arrives on time", "She never arrives on time"),
            ("He likes the movie", "He dislikes the movie"),
            ("The door is open", "The door is closed"),
            ("They agree with the proposal", "They disagree with the proposal"),
            ("I trust his judgment", "I distrust his judgment"),
            ("The test was fair", "The test was unfair"),
            ("She is honest", "She is dishonest"),
            ("We approve the plan", "We disapprove the plan"),
            ("The action was legal", "The action was illegal"),
            ("He appeared at the meeting", "He disappeared from the meeting"),
            ("The information is accurate", "The information is inaccurate"),
            ("She behaved properly", "She misbehaved"),
            ("The patient is comfortable", "The patient is uncomfortable"),
            ("They connect the wires", "They disconnect the wires"),
            ("I believe the report", "I disbelieve the report"),
            ("The result was expected", "The result was unexpected"),
            ("He obeyed the rules", "He disobeyed the rules"),
            ("The material is readable", "The material is unreadable"),
            ("She tied the rope", "She untied the rope"),
            ("The task is possible", "The task is impossible"),
            ("The room is tidy", "The room is untidy"),
            ("He locked the door", "He unlocked the door"),
            ("The code is correct", "The code is incorrect"),
            ("They loaded the truck", "They unloaded the truck"),
            ("The answer is logical", "The answer is illogical"),
            ("She wrapped the gift", "She unwrapped the gift"),
            ("The event was regular", "The event was irregular"),
            ("We packed the bags", "We unpacked the bags"),
            ("The document is valid", "The document is invalid"),
            ("He did the work", "He undid the work"),
            ("The system is stable", "The system is unstable"),
            ("She covered the furniture", "She uncovered the furniture"),
            ("The choice was rational", "The choice was irrational"),
            ("They folded the clothes", "They unfolded the clothes"),
            ("The behavior was appropriate", "The behavior was inappropriate"),
            ("I fastened the seatbelt", "I unfastened the seatbelt"),
            ("The decision was responsible", "The decision was irresponsible"),
            ("She buttoned her coat", "She unbuttoned her coat"),
            ("The claim is believable", "The claim is unbelievable"),
            ("He dressed quickly", "He undressed quickly"),
            # 变体(41-80)
            ("The dog is friendly", "The dog is unfriendly"),
            ("She obeyed the order", "She disobeyed the order"),
            ("The water is drinkable", "The water is undrinkable"),
            ("They engaged the clutch", "They disengaged the clutch"),
            ("The student is literate", "The student is illiterate"),
            ("He mounted the horse", "He dismounted the horse"),
            ("The place is accessible", "The place is inaccessible"),
            ("She wrapped the sandwich", "She unwrapped the sandwich"),
            ("The evidence is admissible", "The evidence is inadmissible"),
            ("They plugged the device", "They unplugged the device"),
            ("The situation is tolerable", "The situation is intolerable"),
            ("He tied his shoelaces", "He untied his shoelaces"),
            ("The claim is justifiable", "The claim is unjustifiable"),
            ("She covered her eyes", "She uncovered her eyes"),
            ("The action was legitimate", "The action was illegitimate"),
            ("They armed the soldiers", "They disarmed the soldiers"),
            ("The method is efficient", "The method is inefficient"),
            ("He folded the letter", "He unfolded the letter"),
            ("The person is reliable", "The person is unreliable"),
            ("She buttoned the shirt", "She unbuttoned the shirt"),
            ("The argument is coherent", "The argument is incoherent"),
            ("I locked the gate", "I unlocked the gate"),
            ("The system is functional", "The system is dysfunctional"),
            ("They loaded the ship", "They unloaded the ship"),
            ("The answer is adequate", "The answer is inadequate"),
            ("She dressed the child", "She undressed the child"),
            ("The behavior is acceptable", "The behavior is unacceptable"),
            ("He fastened the rope", "He unfastened the rope"),
            ("The statement is consistent", "The statement is inconsistent"),
            ("They covered the pool", "They uncovered the pool"),
            ("The decision was ethical", "The decision was unethical"),
            ("I wrapped the bandage", "I unwrapped the bandage"),
            ("The person is capable", "The person is incapable"),
            ("She engaged the gear", "She disengaged the gear"),
            ("The method is practical", "The method is impractical"),
            ("He tied the knot", "He untied the knot"),
            ("The result is definite", "The result is indefinite"),
            ("They plugged the leak", "They unplugged the leak"),
            ("The theory is plausible", "The theory is implausible"),
            ("She mounted the display", "She dismounted the display"),
        ]
    },
    'question': {
        'type': 'SYN',
        'pairs': [
            ("The cat is on the mat", "Is the cat on the mat?"),
            ("She likes chocolate", "Does she like chocolate?"),
            ("He can swim well", "Can he swim well?"),
            ("They have finished the work", "Have they finished the work?"),
            ("The door is closed", "Is the door closed?"),
            ("We need more time", "Do we need more time?"),
            ("She was at the party", "Was she at the party?"),
            ("He will come tomorrow", "Will he come tomorrow?"),
            ("The answer is correct", "Is the answer correct?"),
            ("I understand the problem", "Do you understand the problem?"),
            ("The store is open", "Is the store open?"),
            ("She speaks French", "Does she speak French?"),
            ("The train has arrived", "Has the train arrived?"),
            ("He knows the answer", "Does he know the answer?"),
            ("We enjoyed the movie", "Did we enjoy the movie?"),
            ("The cake tastes good", "Does the cake taste good?"),
            ("They found the key", "Did they find the key?"),
            ("I believe your story", "Do you believe your story?"),
            ("The weather is nice", "Is the weather nice?"),
            ("She plays the piano", "Does she play the piano?"),
            ("He finished his homework", "Did he finish his homework?"),
            ("The team won the game", "Did the team win the game?"),
            ("We received the letter", "Did we receive the letter?"),
            ("The room is clean", "Is the room clean?"),
            ("I remember that day", "Do you remember that day?"),
            ("She agreed with the plan", "Did she agree with the plan?"),
            ("The test was easy", "Was the test easy?"),
            ("He passed the exam", "Did he pass the exam?"),
            ("The food is ready", "Is the food ready?"),
            ("They own a house", "Do they own a house?"),
            ("The road is safe", "Is the road safe?"),
            ("I trust his judgment", "Do you trust his judgment?"),
            ("We saw the sunset", "Did we see the sunset?"),
            ("The water is warm", "Is the water warm?"),
            ("She won the prize", "Did she win the prize?"),
            ("The plan worked well", "Did the plan work well?"),
            ("He told the truth", "Did he tell the truth?"),
            ("The dog is friendly", "Is the dog friendly?"),
            ("The bird can fly", "Can the bird fly?"),
            ("They left early", "Did they leave early?"),
            # 变体(41-80)
            ("The book is on the table", "Is the book on the table?"),
            ("Mary likes ice cream", "Does Mary like ice cream?"),
            ("The child can read", "Can the child read?"),
            ("People have finished lunch", "Have people finished lunch?"),
            ("This window is open", "Is this window open?"),
            ("Tom needs some help", "Does Tom need some help?"),
            ("Anna was at home", "Was Anna at home?"),
            ("Dad will call soon", "Will Dad call soon?"),
            ("This answer is right", "Is this answer right?"),
            ("You understand the rules", "Do you understand the rules?"),
            ("The shop is open now", "Is the shop open now?"),
            ("Linda speaks German", "Does Linda speak German?"),
            ("The bus has come", "Has the bus come?"),
            ("She knows the secret", "Does she know the secret?"),
            ("We loved the concert", "Did we love the concert?"),
            ("The soup smells nice", "Does the soup smell nice?"),
            ("They discovered gold", "Did they discover gold?"),
            ("You accept the offer", "Do you accept the offer?"),
            ("It is sunny today", "Is it sunny today?"),
            ("He plays guitar well", "Does he play guitar well?"),
            ("She completed the task", "Did she complete the task?"),
            ("Our team scored", "Did our team score?"),
            ("You got my message", "Did you get my message?"),
            ("The kitchen is clean", "Is the kitchen clean?"),
            ("You recall the event", "Do you recall the event?"),
            ("John supported the idea", "Did John support the idea?"),
            ("The exam was hard", "Was the exam hard?"),
            ("She qualified easily", "Did she qualify easily?"),
            ("Dinner is served now", "Is dinner served now?"),
            ("They possess a car", "Do they possess a car?"),
            ("The bridge is sturdy", "Is the bridge sturdy?"),
            ("You admire her work", "Do you admire her work?"),
            ("We witnessed the accident", "Did we witness the accident?"),
            ("The pool is warm", "Is the pool warm?"),
            ("He earned the medal", "Did he earn the medal?"),
            ("The strategy worked", "Did the strategy work?"),
            ("She kept her word", "Did she keep her word?"),
            ("The kitten is cute", "Is the kitten cute?"),
            ("The parrot can talk", "Can the parrot talk?"),
            ("You enjoyed the trip", "Did you enjoy the trip?"),
        ]
    },
    'person': {
        'type': 'SYN',
        'pairs': [
            ("I walk to school", "You walk to school"),
            ("She runs every morning", "They run every morning"),
            ("He likes pizza", "We like pizza"),
            ("I am happy today", "You are happy today"),
            ("She has a cat", "They have a cat"),
            ("He goes to the park", "We go to the park"),
            ("I can swim well", "You can swim well"),
            ("She was tired yesterday", "They were tired yesterday"),
            ("He will arrive soon", "We will arrive soon"),
            ("I study mathematics", "You study mathematics"),
            ("She works at the hospital", "They work at the hospital"),
            ("He knows the answer", "We know the answer"),
            ("I play the guitar", "You play the guitar"),
            ("She reads many books", "They read many books"),
            ("He drives a red car", "We drive a red car"),
            ("I need some help", "You need some help"),
            ("She sings in the choir", "They sing in the choir"),
            ("He watches television", "We watch television"),
            ("I believe in justice", "You believe in justice"),
            ("She teaches young children", "They teach young children"),
            ("He speaks three languages", "We speak three languages"),
            ("I enjoy outdoor activities", "You enjoy outdoor activities"),
            ("She writes poetry", "They write poetry"),
            ("He builds model planes", "We build model planes"),
            ("I understand the concept", "You understand the concept"),
            ("She visits her grandmother", "They visit their grandmother"),
            ("He prepares dinner", "We prepare dinner"),
            ("I remember the event", "You remember the event"),
            ("She appreciates good music", "They appreciate good music"),
            ("He manages the team", "We manage the team"),
            ("I prefer tea over coffee", "You prefer tea over coffee"),
            ("She designs websites", "They design websites"),
            ("He repairs old clocks", "We repair old clocks"),
            ("I recognize the melody", "You recognize the melody"),
            ("She organizes the event", "They organize the event"),
            ("He delivers the mail", "We deliver the mail"),
            ("I celebrate my birthday", "You celebrate your birthday"),
            ("She paints landscapes", "They paint landscapes"),
            ("He analyzes the data", "We analyze the data"),
            ("I recommend this book", "You recommend this book"),
            # 变体(41-80)
            ("I cook breakfast daily", "You cook breakfast daily"),
            ("She exercises every week", "They exercise every week"),
            ("He travels frequently", "We travel frequently"),
            ("I clean the house", "You clean the house"),
            ("She practices piano", "They practice piano"),
            ("He fixes the computer", "We fix the computer"),
            ("I plant flowers", "You plant flowers"),
            ("She walks the dog", "They walk the dog"),
            ("He paints the fence", "We paint the fence"),
            ("I bake cookies", "You bake cookies"),
            ("She sews clothes", "They sew clothes"),
            ("He washes the car", "We wash the car"),
            ("I milk the cow", "You milk the cow"),
            ("She feeds the cat", "They feed the cat"),
            ("He mows the lawn", "We mow the lawn"),
            ("I iron the shirts", "You iron the shirts"),
            ("She waters the garden", "They water the garden"),
            ("He tunes the guitar", "We tune the guitar"),
            ("I polish the silver", "You polish the silver"),
            ("She knits scarves", "They knit scarves"),
            ("He chops the wood", "We chop the wood"),
            ("I fold the laundry", "You fold the laundry"),
            ("She sweeps the floor", "They sweep the floor"),
            ("He repairs the roof", "We repair the roof"),
            ("I grind the coffee", "You grind the coffee"),
            ("She braids her hair", "They braid their hair"),
            ("He sharpens the knife", "We sharpen the knife"),
            ("I wrap the presents", "You wrap the presents"),
            ("She stirs the soup", "They stir the soup"),
            ("He measures the wood", "We measure the wood"),
            ("I hang the picture", "You hang the picture"),
            ("She threads the needle", "They thread the needle"),
            ("He stokes the fire", "We stoke the fire"),
            ("I dust the shelves", "You dust the shelves"),
            ("She rolls the dough", "They roll the dough"),
            ("He nails the boards", "We nail the boards"),
            ("I pour the wine", "You pour the wine"),
            ("She peels the apples", "They peel the apples"),
            ("He sands the table", "We sand the table"),
            ("I slice the bread", "You slice the bread"),
        ]
    },
    'definiteness': {
        'type': 'SYN',
        'pairs': [
            ("A cat sleeps on the mat", "The cat sleeps on the mat"),
            ("Some dogs bark at night", "The dogs bark at night"),
            ("An apple a day keeps the doctor away", "The apple a day keeps the doctor away"),
            ("A student raised their hand", "The student raised their hand"),
            ("Some flowers bloom in winter", "The flowers bloom in winter"),
            ("An old man sat on the bench", "The old man sat on the bench"),
            ("A bird sang in the tree", "The bird sang in the tree"),
            ("Some children play outside", "The children play outside"),
            ("A book changed my life", "The book changed my life"),
            ("Some students passed the test", "The students passed the test"),
            ("A teacher entered the room", "The teacher entered the room"),
            ("An idea came to mind", "The idea came to mind"),
            ("Some people like coffee", "The people like coffee"),
            ("A doctor examined the patient", "The doctor examined the patient"),
            ("Some birds migrate south", "The birds migrate south"),
            ("A scientist made a discovery", "The scientist made a discovery"),
            ("Some workers finished early", "The workers finished early"),
            ("A river flows through the city", "The river flows through the city"),
            ("Some trees lose their leaves", "The trees lose their leaves"),
            ("A car stopped at the light", "The car stopped at the light"),
            ("Some houses have gardens", "The houses have gardens"),
            ("A cat chased the mouse", "The cat chased the mouse"),
            ("Some clouds covered the sun", "The clouds covered the sun"),
            ("A key opened the door", "The key opened the door"),
            ("Some fish live in the pond", "The fish live in the pond"),
            ("A star twinkled in the sky", "The star twinkled in the sky"),
            ("Some roads lead to Rome", "The roads lead to Rome"),
            ("A letter arrived today", "The letter arrived today"),
            ("Some ships sail at dawn", "The ships sail at dawn"),
            ("A horse won the race", "The horse won the race"),
            ("Some cats purr when happy", "The cats purr when happy"),
            ("A bell rang in the tower", "The bell rang in the tower"),
            ("Some dogs chase squirrels", "The dogs chase squirrels"),
            ("A candle lit the room", "The candle lit the room"),
            ("Some birds sing at dawn", "The birds sing at dawn"),
            ("A bridge crossed the river", "The bridge crossed the river"),
            ("Some kids ride bicycles", "The kids ride bicycles"),
            ("A fire warmed the house", "The fire warmed the house"),
            ("Some ants build colonies", "The ants build colonies"),
            ("A storm damaged the roof", "The storm damaged the roof"),
            # 变体(41-80)
            ("A fox stole the chicken", "The fox stole the chicken"),
            ("Some bees make honey", "The bees make honey"),
            ("A lock secured the gate", "The lock secured the gate"),
            ("Some trees bear fruit", "The trees bear fruit"),
            ("A boat crossed the lake", "The boat crossed the lake"),
            ("Some wolves hunt in packs", "The wolves hunt in packs"),
            ("A clock struck midnight", "The clock struck midnight"),
            ("Some leaves turn red", "The leaves turn red"),
            ("A king ruled the land", "The king ruled the land"),
            ("Some stones form walls", "The stones form walls"),
            ("A snake hid in the grass", "The snake hid in the grass"),
            ("Some maps show trails", "The maps show trails"),
            ("A lamp lit the path", "The lamp lit the path"),
            ("Some cats chase mice", "The cats chase mice"),
            ("A drum beat loudly", "The drum beat loudly"),
            ("Some bees buzz around flowers", "The bees buzz around flowers"),
            ("A kite flew high", "The kite flew high"),
            ("Some roots grow deep", "The roots grow deep"),
            ("A coin fell to the floor", "The coin fell to the floor"),
            ("Some mice like cheese", "The mice like cheese"),
            ("A flag waved in the wind", "The flag waved in the wind"),
            ("Some stars shine bright", "The stars shine bright"),
            ("A horn sounded the alarm", "The horn sounded the alarm"),
            ("Some eggs hatch in spring", "The eggs hatch in spring"),
            ("A rope held the tent", "The rope held the tent"),
            ("Some frogs jump high", "The frogs jump high"),
            ("A shell washed ashore", "The shell washed ashore"),
            ("Some birds build nests", "The birds build nests"),
            ("A sword cut the rope", "The sword cut the rope"),
            ("Some fish swim upstream", "The fish swim upstream"),
            ("A wheel turned slowly", "The wheel turned slowly"),
            ("Some deer graze at dusk", "The deer graze at dusk"),
            ("A bell chimed noon", "The bell chimed noon"),
            ("Some owls hunt at night", "The owls hunt at night"),
            ("A train left the station", "The train left the station"),
            ("Some pigs roll in mud", "The pigs roll in mud"),
            ("A feather floated down", "The feather floated down"),
            ("Some ants carry food", "The ants carry food"),
            ("A mirror reflected light", "The mirror reflected light"),
            ("Some cows give milk", "The cows give milk"),
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
            # 变体(41-80)
            ("Alice opened the door", "The door was opened by Alice"),
            ("Bob closed the window", "The window was closed by Bob"),
            ("Carol baked the cake", "The cake was baked by Carol"),
            ("David washed the dishes", "The dishes were washed by David"),
            ("Eve found the key", "The key was found by Eve"),
            ("Frank cut the grass", "The grass was cut by Frank"),
            ("Grace lit the candle", "The candle was lit by Grace"),
            ("Henry fed the cat", "The cat was fed by Henry"),
            ("Irene painted the fence", "The fence was painted by Irene"),
            ("Jack repaired the roof", "The roof was repaired by Jack"),
            ("Karen cooked the dinner", "The dinner was cooked by Karen"),
            ("Leo swept the floor", "The floor was swept by Leo"),
            ("Mia watered the plants", "The plants were watered by Mia"),
            ("Noah built the shed", "The shed was built by Noah"),
            ("Olivia cleaned the window", "The window was cleaned by Olivia"),
            ("Paul polished the car", "The car was polished by Paul"),
            ("Quinn delivered the mail", "The mail was delivered by Quinn"),
            ("Rachel organized the party", "The party was organized by Rachel"),
            ("Sam fixed the leak", "The leak was fixed by Sam"),
            ("Tina wrapped the gift", "The gift was wrapped by Tina"),
            ("Victor wrote the report", "The report was written by Victor"),
            ("Wendy designed the logo", "The logo was designed by Wendy"),
            ("Xavier tuned the piano", "The piano was tuned by Xavier"),
            ("Yara planted the tree", "The tree was planted by Yara"),
            ("Zack brewed the coffee", "The coffee was brewed by Zack"),
            ("Anna ironed the shirt", "The shirt was ironed by Anna"),
            ("Ben folded the laundry", "The laundry was folded by Ben"),
            ("Clara made the bed", "The bed was made by Clara"),
            ("Derek vacuumed the carpet", "The carpet was vacuumed by Derek"),
            ("Elena arranged the flowers", "The flowers were arranged by Elena"),
            ("Felix hung the picture", "The picture was hung by Felix"),
            ("Gina sorted the mail", "The mail was sorted by Gina"),
            ("Hugo mowed the lawn", "The lawn was mowed by Hugo"),
            ("Isla packed the boxes", "The boxes were packed by Isla"),
            ("James sharpened the pencil", "The pencil was sharpened by James"),
            ("Kate chopped the onions", "The onions were chopped by Kate"),
            ("Liam stirred the soup", "The soup was stirred by Liam"),
            ("Maya squeezed the lemon", "The lemon was squeezed by Maya"),
            ("Nathan poured the juice", "The juice was poured by Nathan"),
            ("Oscar cracked the egg", "The egg was cracked by Oscar"),
        ]
    },
    'sentiment': {
        'type': 'SEM',
        'pairs': [
            ("The movie was excellent", "The movie was terrible"),
            ("She is a wonderful person", "She is an awful person"),
            ("The food tasted amazing", "The food tasted disgusting"),
            ("We had a fantastic time", "We had a miserable time"),
            ("The weather is beautiful", "The weather is dreadful"),
            ("He gave a brilliant performance", "He gave a dreadful performance"),
            ("The garden looks lovely", "The garden looks ugly"),
            ("She has a beautiful voice", "She has a harsh voice"),
            ("The book is fascinating", "The book is boring"),
            ("They live in a gorgeous house", "They live in a rundown house"),
            ("The sunset was breathtaking", "The smog was suffocating"),
            ("He is a generous person", "He is a stingy person"),
            ("The hotel was luxurious", "The hotel was squalid"),
            ("She received a warm welcome", "She received a cold reception"),
            ("The concert was thrilling", "The concert was dull"),
            ("We enjoyed a pleasant walk", "We endured a grueling march"),
            ("The children are joyful", "The children are miserable"),
            ("He told a hilarious joke", "He made an offensive remark"),
            ("The gift was thoughtful", "The gift was thoughtless"),
            ("She showed great kindness", "She showed great cruelty"),
            ("The news was encouraging", "The news was devastating"),
            ("The landscape is stunning", "The landscape is desolate"),
            ("He has a charming personality", "He has an obnoxious personality"),
            ("The meal was delicious", "The meal was revolting"),
            ("She made a wise decision", "She made a foolish decision"),
            ("The team showed great courage", "The team showed great cowardice"),
            ("He gave a sincere apology", "He gave a fake apology"),
            ("The atmosphere was festive", "The atmosphere was gloomy"),
            ("She has an optimistic outlook", "She has a pessimistic outlook"),
            ("The result was satisfying", "The result was disappointing"),
            ("He displayed remarkable talent", "He displayed remarkable incompetence"),
            ("The experience was enriching", "The experience was draining"),
            ("She spoke with confidence", "She spoke with hesitation"),
            ("The proposal was innovative", "The proposal was outdated"),
            ("He showed genuine compassion", "He showed utter indifference"),
            ("The celebration was lively", "The celebration was lifeless"),
            ("She demonstrated great skill", "She demonstrated great clumsiness"),
            ("The achievement was impressive", "The achievement was unremarkable"),
            ("He offered valuable advice", "He offered harmful advice"),
            ("The outcome was successful", "The outcome was disastrous"),
            # 变体(41-80)
            ("The view was magnificent", "The view was hideous"),
            ("She wore an elegant dress", "She wore a shabby dress"),
            ("The music was harmonious", "The music was discordant"),
            ("He gave a compelling speech", "He gave a tedious speech"),
            ("The solution was elegant", "The solution was crude"),
            ("She showed deep wisdom", "She showed deep ignorance"),
            ("The project was profitable", "The project was costly"),
            ("He made a noble sacrifice", "He made a selfish demand"),
            ("The performance was flawless", "The performance was sloppy"),
            ("She has a gentle nature", "She has a violent nature"),
            ("The review was glowing", "The review was scathing"),
            ("He earned a prestigious award", "He received a harsh penalty"),
            ("The garden was flourishing", "The garden was withering"),
            ("She expressed heartfelt gratitude", "She expressed bitter resentment"),
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
            # 变体(41-80)
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
            # 变体(41-80)
            ("The cat caught the fish", "The fish was caught by the cat"),
            ("The boy kicked the ball", "The ball was kicked by the boy"),
            ("The rain ruined the picnic", "The picnic was ruined by the rain"),
            ("The girl broke the vase", "The vase was broken by the girl"),
            ("The man opened the door", "The door was opened by the man"),
            ("The woman closed the window", "The window was closed by the woman"),
            ("The child dropped the glass", "The glass was dropped by the child"),
            ("The dog found the bone", "The bone was found by the dog"),
            ("The bird built the nest", "The nest was built by the bird"),
            ("The fish swam the river", "The river was swum by the fish"),
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
        ]
    },
    'formality': {
        'type': 'SEM',
        'pairs': [
            ("Hey, what's up?", "Good morning, how are you?"),
            ("Wanna grab some food?", "Would you like to have dinner?"),
            ("Gotta go now, see ya!", "I must leave now, goodbye."),
            ("That's cool, dude!", "That is impressive, sir."),
            ("No way, that's awesome!", "I find that remarkable."),
            ("Let's hang out later", "Shall we meet this evening?"),
            ("What's the deal with this?", "Could you explain this matter?"),
            ("I dunno what to do", "I am uncertain how to proceed."),
            ("This stuff is pretty good", "This material is quite satisfactory."),
            ("Can you help me out?", "Would you be able to assist me?"),
            ("I'm gonna try it", "I intend to attempt it."),
            ("She's really nice", "She is genuinely kind."),
            ("It's kinda weird", "It is somewhat unusual."),
            ("We should fix this thing", "We ought to resolve this matter."),
            ("He messed up big time", "He made a significant error."),
            ("They're doing okay", "They are performing adequately."),
            ("That was a close call", "That was a narrow escape."),
            ("I need to figure this out", "I must determine the solution."),
            ("She's got a lot of stuff", "She possesses numerous items."),
            ("We ran into some problems", "We encountered certain difficulties."),
            ("He's always messing around", "He consistently behaves inappropriately."),
            ("This place is really cool", "This establishment is quite impressive."),
            ("I'm not sure about that", "I harbor doubts regarding that."),
            ("They're gonna be late", "They will arrive tardily."),
            ("She came up with a plan", "She devised a strategy."),
            ("We need to step it up", "We must increase our efforts."),
            ("That doesn't make sense", "That lacks logical coherence."),
            ("He's been working hard", "He has been diligent in his work."),
            ("I'm looking for a job", "I am seeking employment."),
            ("They got a new place", "They acquired a new residence."),
            ("She's going through a lot", "She is experiencing considerable hardship."),
            ("We should talk about it", "We ought to discuss the matter."),
            ("He blew his chance", "He forfeited his opportunity."),
            ("I can't deal with this", "I am unable to manage this situation."),
            ("They're pretty much done", "They are essentially finished."),
            ("She found a way around it", "She discovered an alternative approach."),
            ("We gotta get going", "We must depart promptly."),
            ("He's not feeling great", "He is experiencing discomfort."),
            ("I'll think about it", "I shall consider the matter."),
            ("They made it work somehow", "They succeeded through uncertain means."),
            # 变体(41-80)
            ("What's going on?", "What is happening?"),
            ("I'm beat", "I am exhausted"),
            ("Got it", "I understand"),
            ("Sounds good to me", "That is acceptable to me"),
            ("Hold on a sec", "Please wait a moment"),
            ("Take care", "I wish you well"),
            ("Give me a hand", "Please assist me"),
            ("Hang in there", "Persevere through this"),
            ("Cut it out", "Cease that behavior"),
            ("Keep it up", "Continue your efforts"),
            ("Way to go", "Congratulations on your achievement"),
            ("So long", "Farewell"),
            ("Right on", "I agree completely"),
            ("How come?", "For what reason?"),
            ("No biggie", "It is of no consequence"),
            ("My bad", "I apologize for my error"),
            ("Chill out", "Please calm yourself"),
            ("Watch out", "Exercise caution"),
            ("Knock it off", "Desist immediately"),
            ("Cheer up", "Improve your disposition"),
            ("Get a move on", "Proceed with haste"),
            ("Hold your horses", "Exercise patience"),
            ("Break a leg", "I wish you success"),
            ("Hit the road", "Commence your journey"),
            ("Bite the bullet", "Endure the hardship"),
            ("Under the weather", "Experiencing illness"),
            ("Piece of cake", "A simple endeavor"),
            ("Spill the beans", "Reveal the information"),
            ("Cost an arm and a leg", "Exceedingly expensive"),
            ("Beat around the bush", "Avoid the central point"),
            ("Cut corners", "Compromise on quality"),
            ("Go the extra mile", "Exceed expectations"),
            ("Let the cat out of the bag", "Reveal the secret"),
            ("On thin ice", "In a precarious situation"),
            ("Play it by ear", "Improvise as needed"),
            ("Read between the lines", "Discern the implicit meaning"),
            ("The ball is in your court", "The decision is yours"),
            ("Through thick and thin", "Under all circumstances"),
            ("Up in the air", "Remains unresolved"),
            ("Barking up the wrong tree", "Pursuing a mistaken course"),
        ]
    },
}

FEATURE_PAIRS = FEATURE_PAIRS_BASE

# ============================================================
# 模型配置 (本地路径, 同CCXVII)
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


def cca_subspace_similarity(X1, X2, n_components=5):
    """CCA子空间相似度 — 替代不稳定的centroid_cos"""
    from sklearn.cross_decomposition import CCA
    n = min(X1.shape[0], X2.shape[0])
    X1 = X1[:n]
    X2 = X2[:n]
    n_comp = min(n_components, n - 1, X1.shape[1], X2.shape[1])
    if n_comp < 1:
        return 0.0, []
    cca = CCA(n_components=n_comp)
    try:
        X1_c, X2_c = cca.fit_transform(X1, X2)
        correlations = []
        for i in range(n_comp):
            c = np.corrcoef(X1_c[:, i], X2_c[:, i])[0, 1]
            correlations.append(abs(c))
        mean_corr = np.mean(correlations)
        return float(mean_corr), correlations
    except:
        return 0.0, []


def cka_similarity(X1, X2):
    """Linear CKA (Centered Kernel Alignment)"""
    X1 = X1 - X1.mean(axis=0, keepdims=True)
    X2 = X2 - X2.mean(axis=0, keepdims=True)
    
    dot = np.sum(X1 @ X2.T @ X2 @ X1.T)
    norm1 = np.sum(X1 @ X1.T @ X1 @ X1.T)
    norm2 = np.sum(X2 @ X2.T @ X2 @ X2.T)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(dot / np.sqrt(norm1 * norm2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "deepseek7b", "glm4"])
    parser.add_argument("--n_pairs", type=int, default=80)
    args = parser.parse_args()
    
    model_name = args.model
    n_pairs_max = args.n_pairs
    cfg = MODEL_CONFIGS[model_name]
    
    # 输出目录
    out_dir = f"results/causal_fiber/{model_name}_ccxviii"
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = open(f"{out_dir}/run.log", "w", encoding="utf-8")
    
    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()
    
    log(f"Phase CCXVIII: 精细层扫描 + CCA子空间度量 + 跨层旋转追踪")
    log(f"Model: {cfg['name']}, n_pairs={n_pairs_max}")
    
    # 加载模型 (同CCXVII方式)
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
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    d_model = model.config.hidden_size
    
    # 采样层: 每2层采样(精细扫描)
    layer_indices = list(range(0, n_layers, 2))
    if (n_layers - 1) not in layer_indices:
        layer_indices.append(n_layers - 1)
    log(f"n_layers={n_layers}, n_heads={n_heads}, head_dim={head_dim}, d_model={d_model}")
    log(f"Sampled layers ({len(layer_indices)}): {layer_indices}")
    
    feature_names = list(FEATURE_PAIRS.keys())
    
    # 实际可用的对数
    for feat in feature_names:
        available = len(FEATURE_PAIRS[feat]['pairs'])
        if available < n_pairs_max:
            log(f"  Warning: {feat} has only {available} pairs (requested {n_pairs_max})")
    n_pairs = n_pairs_max
    
    # ============================================================
    # S1: 全层差分向量收集 (Hook方式)
    # ============================================================
    log(f"\n{'='*60}")
    log("S1: 全层差分向量收集 (精细每2层扫描)")
    log(f"{'='*60}")
    
    all_deltas = {feat: {l: [] for l in layer_indices} for feat in feature_names}
    key_layers = [0, n_layers // 2, n_layers - 1]
    head_deltas = {feat: {l: {h: [] for h in range(n_heads)} for l in key_layers} for feat in feature_names}
    
    def make_hook(layer_idx, store):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            store[layer_idx] = out.detach().clone()
        return hook_fn
    
    # 一次性注册hooks
    layer_cap = {}
    head_cap = {}
    handles = []
    
    for l in layer_indices:
        layer_cap[l] = {}
        h = model.model.layers[l].register_forward_hook(make_hook(l, layer_cap[l]))
        handles.append(h)
    for l in key_layers:
        head_cap[l] = {}
        h = model.model.layers[l].self_attn.o_proj.register_forward_hook(make_hook(l, head_cap[l]))
        handles.append(h)
    
    log(f"  Registered {len(handles)} hooks")
    
    for feat_idx, feat in enumerate(feature_names):
        pairs = FEATURE_PAIRS[feat]['pairs'][:n_pairs]
        log(f"  [{feat_idx+1}/{len(feature_names)}] {feat} ({FEATURE_PAIRS[feat]['type']}): {len(pairs)} pairs")
        
        for s1, s2 in pairs:
            with torch.no_grad():
                # Sentence 1
                enc1 = tokenizer(s1, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc1 = {k: v.to(device) for k, v in enc1.items()}
                _ = model(**enc1)
                len1 = enc1['attention_mask'].sum().item()
                idx1 = max(0, len1 - 2)
                
                s1_layer = {l: layer_cap[l][l][0, idx1].float().cpu().numpy() for l in layer_indices if l in layer_cap[l]}
                s1_head = {l: head_cap[l][l][0, idx1].float().cpu().numpy() for l in key_layers if l in head_cap[l]}
                
                # Sentence 2
                enc2 = tokenizer(s2, return_tensors='pt', padding=True, truncation=True, max_length=64)
                enc2 = {k: v.to(device) for k, v in enc2.items()}
                _ = model(**enc2)
                len2 = enc2['attention_mask'].sum().item()
                idx2 = max(0, len2 - 2)
                
                # 计算差分
                for l in layer_indices:
                    if l in layer_cap[l] and l in s1_layer:
                        s2_vec = layer_cap[l][l][0, idx2].float().cpu().numpy()
                        delta = s2_vec - s1_layer[l]
                        all_deltas[feat][l].append(delta)
                
                for l in key_layers:
                    if l in head_cap[l] and l in s1_head:
                        s2_h = head_cap[l][l][0, idx2].float().cpu().numpy()
                        delta_h = s2_h - s1_head[l]
                        if delta_h.shape[0] == n_heads * head_dim:
                            for h in range(n_heads):
                                start = h * head_dim
                                end = start + head_dim
                                head_deltas[feat][l][h].append(delta_h[start:end].copy())
                
                if model_name in ["glm4", "deepseek7b"]:
                    torch.cuda.empty_cache()
        
        n_valid = sum(1 for l in layer_indices for d in all_deltas[feat][l] if d is not None)
        log(f"    Collected {n_valid} valid delta vectors (all layers)")
    
    for h in handles:
        h.remove()
    
    # ============================================================
    # S2: 精细PCA演变
    # ============================================================
    log(f"\n{'='*60}")
    log("S2: 精细PCA演变 + 信息瓶颈精确定位")
    log(f"{'='*60}")
    
    from sklearn.decomposition import TruncatedSVD
    
    layer_pca = {}
    pca_components = {}  # 保存PC方向用于旋转追踪
    
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
        
        n_pc = min(20, n_samples - 1, d)
        svd = TruncatedSVD(n_components=n_pc, random_state=42)
        projected = svd.fit_transform(mat)
        
        # 保存PC方向
        pca_components[l] = svd.components_[:10]  # 保存前10个PC
        
        var_top5 = svd.explained_variance_ratio_[:5].tolist()
        cumvar_top5 = np.cumsum(var_top5).tolist()
        cumvar_10 = np.cumsum(svd.explained_variance_ratio_[:10]).tolist()[-1] if len(var_top5) >= 5 else cumvar_top5[-1]
        
        # centroid cosine similarity
        syn_proj = projected[:len(syn_deltas)]
        sem_proj = projected[len(syn_deltas):]
        if len(syn_proj) > 0 and len(sem_proj) > 0:
            syn_centroid = np.mean(syn_proj[:, :5], axis=0)
            sem_centroid = np.mean(sem_proj[:, :5], axis=0)
            n1 = np.linalg.norm(syn_centroid)
            n2 = np.linalg.norm(sem_centroid)
            centroid_cos = float(abs(np.dot(syn_centroid, sem_centroid) / (n1 * n2 + 1e-8)))
        else:
            centroid_cos = 0.0
        
        # CCA子空间相似度
        if len(syn_proj) > 5 and len(sem_proj) > 5:
            syn_mat = syn_proj[:, :10]
            sem_mat = sem_proj[:, :10]
            cca_corr, cca_corrs = cca_subspace_similarity(syn_mat, sem_mat, n_components=5)
            cka_score = cka_similarity(syn_mat, sem_mat)
        else:
            cca_corr = 0.0
            cka_score = 0.0
        
        # Per-feature centroid on PC1
        feat_pc1_centroids = {}
        for feat in feature_names:
            si, ei = feat_indices[feat]
            if ei > si:
                feat_pc1_centroids[feat] = float(projected[si:ei, 0].mean())
        
        # Mean delta norm
        syn_norms = [np.linalg.norm(d) for d in syn_deltas]
        sem_norms = [np.linalg.norm(d) for d in sem_deltas]
        
        layer_pca[l] = {
            'pc1': var_top5[0],
            'cum5': cumvar_top5[-1],
            'cum10': cumvar_10,
            'centroid_cos': centroid_cos,
            'cca_corr': cca_corr,
            'cka_score': cka_score,
            'syn_mean_norm': float(np.mean(syn_norms)) if syn_norms else 0,
            'sem_mean_norm': float(np.mean(sem_norms)) if sem_norms else 0,
            'n_syn': len(syn_deltas),
            'n_sem': len(sem_deltas),
            'feat_pc1': feat_pc1_centroids,
        }
        
        log(f"  L{l}: PC1={var_top5[0]:.4f}, cum5={cumvar_top5[-1]:.4f}, "
            f"cos={centroid_cos:.4f}, CCA={cca_corr:.4f}, CKA={cka_score:.4f}, "
            f"norm(S/E)={np.mean(syn_norms):.0f}/{np.mean(sem_norms):.0f}")
    
    # 信息瓶颈 + 最正交层
    log("\n--- Information Bottleneck & Orthogonality ---")
    pc1_evolution = [(l, layer_pca[l]['pc1']) for l in layer_indices if l in layer_pca]
    cos_evolution = [(l, layer_pca[l]['centroid_cos']) for l in layer_indices if l in layer_pca]
    cca_evolution = [(l, layer_pca[l]['cca_corr']) for l in layer_indices if l in layer_pca]
    
    if pc1_evolution:
        bottleneck_l, bottleneck_v = max(pc1_evolution, key=lambda x: x[1])
        log(f"  Information bottleneck: L{bottleneck_l} (PC1={bottleneck_v:.4f})")
    if cos_evolution:
        ortho_l, ortho_v = min(cos_evolution, key=lambda x: x[1])
        log(f"  Most orthogonal (centroid_cos): L{ortho_l} (cos={ortho_v:.4f})")
    if cca_evolution:
        cca_ortho_l, cca_ortho_v = min(cca_evolution, key=lambda x: x[1])
        log(f"  Most orthogonal (CCA): L{cca_ortho_l} (CCA={cca_ortho_v:.4f})")
    
    # ============================================================
    # S3: 跨层PC旋转追踪
    # ============================================================
    log(f"\n{'='*60}")
    log("S3: 跨层PC旋转追踪")
    log(f"{'='*60}")
    
    rotation_data = []
    sorted_layers = sorted([l for l in pca_components.keys()])
    
    for i in range(len(sorted_layers) - 1):
        l1 = sorted_layers[i]
        l2 = sorted_layers[i + 1]
        pc1 = pca_components[l1]
        pc2 = pca_components[l2]
        
        # 计算前5个PC的旋转余弦
        cos_matrix = np.abs(pc1[:5] @ pc2[:5].T)
        pc1_rot = float(cos_matrix[0, 0])  # PC1的旋转
        
        # 平均旋转
        avg_rot = float(np.mean(np.diag(cos_matrix)))
        
        rotation_data.append({
            'l1': l1, 'l2': l2,
            'pc1_rotation': pc1_rot,
            'avg_rotation_5pc': avg_rot,
        })
        
        log(f"  L{l1}->L{l2}: PC1_rot={pc1_rot:.4f}, avg_rot(5PC)={avg_rot:.4f}")
    
    # 找最大旋转点
    if rotation_data:
        max_rot = max(rotation_data, key=lambda x: 1 - x['pc1_rotation'])
        log(f"  Max PC1 rotation: L{max_rot['l1']}->L{max_rot['l2']} (rot={1-max_rot['pc1_rotation']:.4f})")
    
    # ============================================================
    # S4: Per-feature Head贡献矩阵
    # ============================================================
    log(f"\n{'='*60}")
    log("S4: Per-feature Head贡献矩阵")
    log(f"{'='*60}")
    
    head_feature_matrix = {}
    
    for l in key_layers:
        matrix = np.zeros((n_heads, len(feature_names)))
        
        for f_idx, feat in enumerate(feature_names):
            for h in range(n_heads):
                hds = head_deltas[feat][l][h]
                if hds:
                    matrix[h, f_idx] = np.mean([np.linalg.norm(d) for d in hds])
        
        head_feature_matrix[l] = matrix
        
        log(f"  L{l}: Head-Feature contribution matrix ({n_heads}x{len(feature_names)})")
        
        log(f"  Per-feature top Heads:")
        for f_idx, feat in enumerate(feature_names):
            col = matrix[:, f_idx]
            top3 = np.argsort(-col)[:3]
            log(f"    {feat}: Top Heads={top3.tolist()}, norms={[f'{col[h]:.1f}' for h in top3]}")
        
        log(f"  Per-head top Features:")
        for h in range(min(10, n_heads)):
            row = matrix[h]
            top3 = np.argsort(-row)[:3]
            log(f"    H{h}: Top Features={[feature_names[i] for i in top3]}, norms={[f'{row[i]:.1f}' for i in top3]}")
    
    # ============================================================
    # S5: Head聚类分析
    # ============================================================
    log(f"\n{'='*60}")
    log("S5: Head聚类分析")
    log(f"{'='*60}")
    
    for l in key_layers:
        matrix = head_feature_matrix[l]
        
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix_norm = matrix / row_sums
        
        if n_heads > 2:
            dist_matrix = pdist(matrix_norm, metric='correlation')
            Z = linkage(dist_matrix, method='ward')
            
            for n_clusters in [2, 3, 4]:
                clusters = fcluster(Z, n_clusters, criterion='maxclust')
                cluster_sizes = [np.sum(clusters == c) for c in range(1, n_clusters + 1)]
                
                syn_cols = [i for i, f in enumerate(feature_names) if FEATURE_PAIRS[f]['type'] == 'SYN']
                sem_cols = [i for i, f in enumerate(feature_names) if FEATURE_PAIRS[f]['type'] == 'SEM']
                
                cluster_syn_prefs = []
                for c in range(1, n_clusters + 1):
                    mask = clusters == c
                    cluster_matrix = matrix_norm[mask]
                    syn_pref = cluster_matrix[:, syn_cols].mean() / (cluster_matrix[:, sem_cols].mean() + 1e-8)
                    cluster_syn_prefs.append(syn_pref)
                
                max_ratio = max(cluster_syn_prefs) / (min(cluster_syn_prefs) + 1e-8)
                
                log(f"  L{l} clusters={n_clusters}: sizes={cluster_sizes}, "
                    f"syn_prefs={[f'{r:.2f}' for r in cluster_syn_prefs]}, max_ratio={max_ratio:.2f}")
            
            clusters_2 = fcluster(Z, 2, criterion='maxclust')
            mask1 = clusters_2 == 1
            mask2 = clusters_2 == 2
            c1_syn = matrix_norm[mask1][:, syn_cols].mean() / (matrix_norm[mask1][:, sem_cols].mean() + 1e-8)
            c2_syn = matrix_norm[mask2][:, syn_cols].mean() / (matrix_norm[mask2][:, sem_cols].mean() + 1e-8)
            
            if c1_syn > c2_syn:
                syn_cluster, sem_cluster = 1, 2
            else:
                syn_cluster, sem_cluster = 2, 1
            
            syn_heads = np.where(clusters_2 == syn_cluster)[0].tolist()
            sem_heads = np.where(clusters_2 == sem_cluster)[0].tolist()
            
            log(f"  L{l} Best 2-cluster: SYN heads={syn_heads}, SEM heads={sem_heads}")
            
            if len(syn_heads) > 0 and len(sem_heads) > 0:
                intra_syn = np.mean(pdist(matrix_norm[mask1], 'correlation')) if len(syn_heads) > 1 else 0
                intra_sem = np.mean(pdist(matrix_norm[mask2], 'correlation')) if len(sem_heads) > 1 else 0
                inter = np.mean(pdist(np.vstack([matrix_norm[mask1], matrix_norm[mask2]]), 'correlation'))
                log(f"  L{l} Cluster quality: intra_syn={intra_syn:.3f}, intra_sem={intra_sem:.3f}, inter={inter:.3f}")
    
    # ============================================================
    # S6: 因果原子词典 + 统计总结
    # ============================================================
    log(f"\n{'='*60}")
    log("S6: 因果原子词典 + 统计总结")
    log(f"{'='*60}")
    
    # Growth计算
    growth_data = {}
    for feat in feature_names:
        norms_by_layer = {}
        for l in layer_indices:
            valid = [d for d in all_deltas[feat][l] if d is not None]
            if valid:
                norms_by_layer[l] = float(np.mean([np.linalg.norm(d) for d in valid]))
        
        early_layers = [l for l in layer_indices if l <= n_layers // 4]
        late_layers = [l for l in layer_indices if l >= 3 * n_layers // 4]
        
        early_norm = np.mean([norms_by_layer[l] for l in early_layers if l in norms_by_layer]) if early_layers else 0
        late_norm = np.mean([norms_by_layer[l] for l in late_layers if l in norms_by_layer]) if late_layers else 0
        
        growth = late_norm / (early_norm + 1e-8)
        growth_data[feat] = {
            'growth': growth,
            'early_norm': early_norm,
            'late_norm': late_norm,
        }
    
    # 语法 vs 语义 growth 统计检验
    syn_growths = [growth_data[f]['growth'] for f in feature_names if FEATURE_PAIRS[f]['type'] == 'SYN']
    sem_growths = [growth_data[f]['growth'] for f in feature_names if FEATURE_PAIRS[f]['type'] == 'SEM']
    
    if len(syn_growths) > 1 and len(sem_growths) > 1:
        u_stat, p_val = stats.mannwhitneyu(syn_growths, sem_growths, alternative='two-sided')
        log(f"  Growth: syn_mean={np.mean(syn_growths):.2f}x vs sem_mean={np.mean(sem_growths):.2f}x, p={p_val:.4f}")
    
    # 最终层因果原子词典
    final_layer = n_layers - 1
    if final_layer in layer_pca:
        feat_pc1 = layer_pca[final_layer].get('feat_pc1', {})
        log(f"\n  Causal Atom Dictionary (L{final_layer}, ranked by |PC1 centroid|):")
        
        sorted_feats = sorted(feat_pc1.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, centroid in sorted_feats:
            ftype = FEATURE_PAIRS[feat]['type']
            growth = growth_data[feat]['growth']
            log(f"    {feat} [{ftype}]: PC1 centroid={centroid:.3f}, growth={growth:.1f}x")
    
    # ============================================================
    # FINAL输出
    # ============================================================
    log(f"\n{'='*60}")
    log(f"FINAL: {model_name}")
    log(f"{'='*60}")
    
    # Delta Norms
    log("\n--- Delta Norms (sampled layers) ---")
    for feat in feature_names:
        norms = []
        for l in [0, n_layers // 2, n_layers - 1]:
            if l in layer_pca:
                if FEATURE_PAIRS[feat]['type'] == 'SYN':
                    norms.append(f"{layer_pca[l].get('syn_mean_norm', 0):.0f}")
                else:
                    norms.append(f"{layer_pca[l].get('sem_mean_norm', 0):.0f}")
            else:
                norms.append("N/A")
        log(f"  {feat} [{FEATURE_PAIRS[feat]['type']}]: {', '.join(norms)}")
    
    # PCA Evolution
    log("\n--- PCA Evolution ---")
    for l in layer_indices:
        if l in layer_pca:
            log(f"  L{l}: PC1={layer_pca[l]['pc1']:.4f}, cum5={layer_pca[l]['cum5']:.4f}, "
                f"cos={layer_pca[l]['centroid_cos']:.4f}, CCA={layer_pca[l]['cca_corr']:.4f}")
    
    # Rotation
    log("\n--- PC Rotation ---")
    for rd in rotation_data:
        log(f"  L{rd['l1']}->L{rd['l2']}: PC1_rot={rd['pc1_rotation']:.4f}")
    
    # Fiber Growth
    log("\n--- Fiber Growth ---")
    for feat in feature_names:
        log(f"  {feat} [{FEATURE_PAIRS[feat]['type']}]: growth={growth_data[feat]['growth']:.2f}x")
    
    # Head Feature Matrix (最终层)
    log(f"\n--- Head Feature Matrix (L{n_layers-1}) ---")
    if n_layers - 1 in head_feature_matrix:
        matrix = head_feature_matrix[n_layers - 1]
        for feat_idx, feat in enumerate(feature_names):
            col = matrix[:, feat_idx]
            top5 = np.argsort(-col)[:5]
            log(f"  {feat}: Top5 Heads={top5.tolist()}, weights={[f'{col[h]:.1f}' for h in top5]}")
    
    log(f"\nDONE! Saved to {out_dir}")
    
    # 保存JSON结果
    results = {
        'model': model_name,
        'n_pairs': n_pairs,
        'n_layers': n_layers,
        'layer_indices': layer_indices,
        'layer_pca': {str(k): v for k, v in layer_pca.items()},
        'rotation_data': rotation_data,
        'growth_data': growth_data,
        'head_feature_matrix': {str(k): v.tolist() for k, v in head_feature_matrix.items()},
    }
    
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log_file.close()


if __name__ == "__main__":
    main()
