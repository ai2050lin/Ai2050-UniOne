"""
Phase CCXXXI: 句法特征重叠机制与高阶PC分离分析
================================================================
核心目标:
1. 分析tense×question高度重叠(|cos|>0.44)的机制
2. 验证PC2/PC3能否分离重叠的特征对
3. 重叠是否跨层一致, 还是只在特定层出现
4. 特征子空间分析: 重叠对共享多少维子空间?
5. 信息论角度: 重叠特征的信息是否冗余还是互补?

关键假设:
  tense和question在PC1上高度重叠, 但可能在PC2/PC3上分离。
  这就像傅里叶分析: 两个信号在某基上重叠, 但在另一基上正交。

样本量: 80对/特征(重叠分析需要更多样本), 40个测试句子
"""

import argparse
import json
import os
import time
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

import torch

# 聚焦6个关键特征: 重叠对(tense, question) + 最正交对(voice, semantic_valence) + 中间(polarity, negation)
FOCUS_FEATURES = {
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
            ("The sun rises in the east", "The sun rose in the east"),
            ("He writes a letter home", "He wrote a letter home"),
            ("The bird builds a nest", "The bird built a nest"),
            ("She drives to work early", "She drove to work early"),
            ("The rain falls softly", "The rain fell softly"),
            ("He reads the newspaper", "He read the newspaper"),
            ("The bell rings loudly", "The bell rang loudly"),
            ("They dance at the party", "They danced at the party"),
            ("The train arrives on time", "The train arrived on time"),
            ("She wears a red dress", "She wore a red dress"),
            ("The garden grows quickly", "The garden grew quickly"),
            ("He speaks three languages", "He spoke three languages"),
            ("The water boils rapidly", "The water boiled rapidly"),
            ("She opens the window", "She opened the window"),
            ("The storm destroys the house", "The storm destroyed the house"),
            ("They travel abroad frequently", "They traveled abroad frequently"),
            ("The snow covers the ground", "The snow covered the ground"),
            ("She bakes a chocolate cake", "She baked a chocolate cake"),
            ("He fixes the broken chair", "He fixed the broken chair"),
            ("The clock strikes midnight", "The clock struck midnight"),
            ("They celebrate the victory", "They celebrated the victory"),
            ("The horse gallops across", "The horse galloped across"),
            ("She types the document", "She typed the document"),
            ("The machine operates smoothly", "The machine operated smoothly"),
            ("He delivers the package", "He delivered the package"),
            ("The ship sails across the ocean", "The ship sailed across the ocean"),
            ("The fire burns brightly", "The fire burned brightly"),
            ("She studies mathematics", "She studied mathematics"),
            ("The plane takes off early", "The plane took off early"),
            ("He draws a landscape", "He drew a landscape"),
            ("The music plays softly", "The music played softly"),
            ("They harvest the crops", "They harvested the crops"),
            ("The ice melts in spring", "The ice melted in spring"),
            ("She teaches the children", "She taught the children"),
            ("The frog jumps into the pond", "The frog jumped into the pond"),
            ("He drives carefully", "He drove carefully"),
            ("The stars shine brightly", "The stars shone brightly"),
            ("She chooses the blue one", "She chose the blue one"),
            ("The leaves fall from the tree", "The leaves fell from the tree"),
            ("He breaks the record", "He broke the record"),
            ("The dog catches the ball", "The dog caught the ball"),
            ("She spends the afternoon reading", "She spent the afternoon reading"),
            ("The boat crosses the lake", "The boat crossed the lake"),
            ("He brings the supplies", "He brought the supplies"),
            ("The wind carries the seeds", "The wind carried the seeds"),
            ("She holds the baby gently", "She held the baby gently"),
            ("The king rules the kingdom", "The king ruled the kingdom"),
            ("He stands by the door", "He stood by the door"),
            ("The snake bites the man", "The snake bit the man"),
            ("She feels the cold wind", "She felt the cold wind"),
            ("The light shines through", "The light shone through"),
            ("He throws the ball far", "He threw the ball far"),
            ("The fish swims upstream", "The fish swam upstream"),
            ("She leaves the room quietly", "She left the room quietly"),
            ("The thunder shakes the house", "The thunder shook the house"),
            ("He meets the president", "He met the president"),
            ("The flower blooms in spring", "The flower bloomed in spring"),
            ("She wins the prize", "She won the prize"),
            ("The dog hides under the bed", "The dog hid under the bed"),
            ("He tells the truth", "He told the truth"),
            ("The moon rises slowly", "The moon rose slowly"),
            ("She feeds the hungry cat", "She fed the hungry cat"),
            ("The car stops suddenly", "The car stopped suddenly"),
            ("He hangs the picture", "He hung the picture"),
            ("The bird sings a song", "The bird sang a song"),
            ("She keeps the secret", "She kept the secret"),
            ("The boy swings high", "The boy swung high"),
        ]
    },
    'question': {
        'type': 'SYN',
        'pairs': [
            ("The cat sat on the mat", "Did the cat sit on the mat"),
            ("She likes chocolate", "Does she like chocolate"),
            ("He went to the store", "Did he go to the store"),
            ("They play tennis every week", "Do they play tennis every week"),
            ("The dog chased the ball", "Did the dog chase the ball"),
            ("She speaks French fluently", "Does she speak French fluently"),
            ("He finished the assignment", "Did he finish the assignment"),
            ("The children ate the cookies", "Did the children eat the cookies"),
            ("She understands the material", "Does she understand the material"),
            ("They built a sandcastle", "Did they build a sandcastle"),
            ("The train arrived on time", "Did the train arrive on time"),
            ("He wrote the letter yesterday", "Did he write the letter yesterday"),
            ("She won the competition", "Did she win the competition"),
            ("The birds flew south", "Did the birds fly south"),
            ("They found the treasure", "Did they find the treasure"),
            ("The teacher explained the rule", "Did the teacher explain the rule"),
            ("She bought a new car", "Did she buy a new car"),
            ("He solved the puzzle", "Did he solve the puzzle"),
            ("The rain stopped eventually", "Did the rain stop eventually"),
            ("They watched the sunset", "Did they watch the sunset"),
            ("The cake tastes delicious", "Does the cake taste delicious"),
            ("She reads every evening", "Does she read every evening"),
            ("He runs five miles daily", "Does he run five miles daily"),
            ("The flowers need water", "Do the flowers need water"),
            ("They study hard for exams", "Do they study hard for exams"),
            ("The movie starts at eight", "Does the movie start at eight"),
            ("She works at the hospital", "Does she work at the hospital"),
            ("He plays the guitar well", "Does he play the guitar well"),
            ("The soup needs more salt", "Does the soup need more salt"),
            ("They travel every summer", "Do they travel every summer"),
            ("The car needs gas", "Does the car need gas"),
            ("She loves the ocean", "Does she love the ocean"),
            ("He drives a truck", "Does he drive a truck"),
            ("The baby sleeps well", "Does the baby sleep well"),
            ("They own a restaurant", "Do they own a restaurant"),
            ("The school opens early", "Does the school open early"),
            ("She teaches chemistry", "Does she teach chemistry"),
            ("He drinks coffee daily", "Does he drink coffee daily"),
            ("The project takes time", "Does the project take time"),
            ("They live nearby", "Do they live nearby"),
            ("The dog barks loudly", "Does the dog bark loudly"),
            ("She wears glasses", "Does she wear glasses"),
            ("He knows the answer", "Does he know the answer"),
            ("The river flows north", "Does the river flow north"),
            ("They sell fresh bread", "Do they sell fresh bread"),
            ("The clock chimes hourly", "Does the clock chime hourly"),
            ("She paints landscapes", "Does she paint landscapes"),
            ("He fixes computers", "Does he fix computers"),
            ("The bus comes often", "Does the bus come often"),
            ("They grow vegetables", "Do they grow vegetables"),
            ("The door locks automatically", "Does the door lock automatically"),
            ("She speaks three languages", "Does she speak three languages"),
            ("He enjoys hiking", "Does he enjoy hiking"),
            ("The team wins often", "Does the team win often"),
            ("They collect stamps", "Do they collect stamps"),
            ("The machine works well", "Does the machine work well"),
            ("She writes poetry", "Does she write poetry"),
            ("He reads the news", "Does he read the news"),
            ("The lake freezes winter", "Does the lake freeze winter"),
            ("They visit grandmother", "Do they visit grandmother"),
            ("The candle burns bright", "Does the candle burn bright"),
            ("She bakes cookies", "Does she bake cookies"),
            ("He rides a bicycle", "Does he ride a bicycle"),
            ("The cat purrs softly", "Does the cat purr softly"),
            ("They sing together", "Do they sing together"),
            ("The wind howls tonight", "Does the wind howl tonight"),
            ("She takes the bus", "Does she take the bus"),
            ("He remembers everything", "Does he remember everything"),
            ("The building stands tall", "Does the building stand tall"),
            ("They dance beautifully", "Do they dance beautifully"),
            ("The water tastes fresh", "Does the water taste fresh"),
            ("She practices daily", "Does she practice daily"),
            ("He earns a good salary", "Does he earn a good salary"),
            ("The bird builds nests", "Does the bird build nests"),
            ("They help the poor", "Do they help the poor"),
            ("The grass grows fast", "Does the grass grow fast"),
            ("She plays the piano", "Does she play the piano"),
        ]
    },
    'voice': {
        'type': 'SYN',
        'pairs': [
            ("The cat chased the mouse", "The mouse was chased by the cat"),
            ("She wrote the letter", "The letter was written by her"),
            ("He fixed the car", "The car was fixed by him"),
            ("They built the house", "The house was built by them"),
            ("The teacher punished the student", "The student was punished by the teacher"),
            ("She painted the portrait", "The portrait was painted by her"),
            ("He cooked the dinner", "The dinner was cooked by him"),
            ("They discovered the treasure", "The treasure was discovered by them"),
            ("The police arrested the thief", "The thief was arrested by the police"),
            ("She cleaned the room", "The room was cleaned by her"),
            ("He broke the window", "The window was broken by him"),
            ("They won the championship", "The championship was won by them"),
            ("The chef prepared the meal", "The meal was prepared by the chef"),
            ("She delivered the speech", "The speech was delivered by her"),
            ("He designed the building", "The building was designed by him"),
            ("They destroyed the evidence", "The evidence was destroyed by them"),
            ("The dog chased the ball", "The ball was chased by the dog"),
            ("She sang the song", "The song was sung by her"),
            ("He drove the bus", "The bus was driven by him"),
            ("They painted the fence", "The fence was painted by them"),
            ("The wind blew the leaves", "The leaves were blown by the wind"),
            ("She washed the dishes", "The dishes were washed by her"),
            ("He caught the fish", "The fish was caught by him"),
            ("They sold the house", "The house was sold by them"),
            ("The company developed the software", "The software was developed by the company"),
            ("She cut the cake", "The cake was cut by her"),
            ("He threw the ball", "The ball was thrown by him"),
            ("They found the solution", "The solution was found by them"),
            ("The storm damaged the roof", "The roof was damaged by the storm"),
            ("She opened the door", "The door was opened by her"),
            ("He wrote the report", "The report was written by him"),
            ("They invented the machine", "The machine was invented by them"),
            ("The cat killed the mouse", "The mouse was killed by the cat"),
            ("She typed the letter", "The letter was typed by her"),
            ("He built the boat", "The boat was built by him"),
            ("They completed the project", "The project was completed by them"),
            ("The fire destroyed the forest", "The forest was destroyed by the fire"),
            ("She drew the picture", "The picture was drawn by her"),
            ("He repaired the engine", "The engine was repaired by him"),
            ("They organized the event", "The event was organized by them"),
            ("The boy kicked the ball", "The ball was kicked by the boy"),
            ("She baked the bread", "The bread was baked by her"),
            ("He planted the tree", "The tree was planted by him"),
            ("They wrote the book", "The book was written by them"),
            ("The girl chose the dress", "The dress was chosen by the girl"),
            ("She fed the cat", "The cat was fed by her"),
            ("He taught the class", "The class was taught by him"),
            ("They cleaned the beach", "The beach was cleaned by them"),
            ("The dog bit the man", "The man was bitten by the dog"),
            ("She sent the email", "The email was sent by her"),
            ("He read the book", "The book was read by him"),
            ("They watched the movie", "The movie was watched by them"),
            ("The wind broke the branch", "The branch was broken by the wind"),
            ("She made the decision", "The decision was made by her"),
            ("He took the photo", "The photo was taken by him"),
            ("They grew the vegetables", "The vegetables were grown by them"),
            ("The rain ruined the picnic", "The picnic was ruined by the rain"),
            ("She translated the document", "The document was translated by her"),
            ("He composed the music", "The music was composed by him"),
            ("They carried the boxes", "The boxes were carried by them"),
            ("The sun melted the snow", "The snow was melted by the sun"),
            ("She served the dinner", "The dinner was served by her"),
            ("He directed the film", "The film was directed by him"),
            ("They repaired the road", "The road was repaired by them"),
            ("The flood washed away the bridge", "The bridge was washed away by the flood"),
            ("She knit the sweater", "The sweater was knit by her"),
            ("He trained the dog", "The dog was trained by him"),
            ("They decorated the room", "The room was decorated by them"),
            ("The earthquake shook the building", "The building was shaken by the earthquake"),
            ("She won the award", "The award was won by her"),
            ("He mixed the ingredients", "The ingredients were mixed by him"),
            ("They launched the rocket", "The rocket was launched by them"),
            ("The teacher graded the papers", "The papers were graded by the teacher"),
            ("She recorded the song", "The song was recorded by her"),
            ("He edited the video", "The video was edited by him"),
            ("They rescued the survivors", "The survivors were rescued by them"),
            ("The child broke the toy", "The toy was broken by the child"),
        ]
    },
    'polarity': {
        'type': 'SYN',
        'pairs': [
            ("She likes the movie", "She does not like the movie"),
            ("He enjoys running", "He does not enjoy running"),
            ("They want to come", "They do not want to come"),
            ("The dog barks loudly", "The dog does not bark loudly"),
            ("She speaks Chinese", "She does not speak Chinese"),
            ("He plays the piano", "He does not play the piano"),
            ("They understand the problem", "They do not understand the problem"),
            ("The car works well", "The car does not work well"),
            ("She eats breakfast", "She does not eat breakfast"),
            ("He believes the story", "He does not believe the story"),
            ("They need help", "They do not need help"),
            ("The plant grows fast", "The plant does not grow fast"),
            ("She remembers the event", "She does not remember the event"),
            ("He drives carefully", "He does not drive carefully"),
            ("They like pizza", "They do not like pizza"),
            ("The machine functions properly", "The machine does not function properly"),
            ("She sings well", "She does not sing well"),
            ("He knows the answer", "He does not know the answer"),
            ("They support the team", "They do not support the team"),
            ("The door opens easily", "The door does not open easily"),
            ("She writes clearly", "She does not write clearly"),
            ("He reads novels", "He does not read novels"),
            ("They agree with the plan", "They do not agree with the plan"),
            ("The light shines brightly", "The light does not shine brightly"),
            ("She trusts him", "She does not trust him"),
            ("He works hard", "He does not work hard"),
            ("They own a house", "They do not own a house"),
            ("The food tastes good", "The food does not taste good"),
            ("She speaks loudly", "She does not speak loudly"),
            ("He exercises daily", "He does not exercise daily"),
            ("They accept the offer", "They do not accept the offer"),
            ("The baby sleeps peacefully", "The baby does not sleep peacefully"),
            ("She follows the rules", "She does not follow the rules"),
            ("He watches television", "He does not watch television"),
            ("They appreciate the help", "They do not appreciate the help"),
            ("The system works efficiently", "The system does not work efficiently"),
            ("She cooks dinner", "She does not cook dinner"),
            ("He recognizes the face", "He does not recognize the face"),
            ("They attend the meeting", "They do not attend the meeting"),
            ("The bird flies high", "The bird does not fly high"),
            ("She wears glasses", "She does not wear glasses"),
            ("He enjoys music", "He does not enjoy music"),
            ("They produce results", "They do not produce results"),
            ("The flower smells sweet", "The flower does not smell sweet"),
            ("She pays attention", "She does not pay attention"),
            ("He plays chess", "He does not play chess"),
            ("They drink coffee", "They do not drink coffee"),
            ("The river flows fast", "The river does not flow fast"),
            ("She teaches math", "She does not teach math"),
            ("He drives a car", "He does not drive a car"),
            ("They build houses", "They do not build houses"),
            ("The dog likes bones", "The dog does not like bones"),
            ("She loves animals", "She does not love animals"),
            ("He speaks Spanish", "He does not speak Spanish"),
            ("They eat vegetables", "They do not eat vegetables"),
            ("The clock ticks loudly", "The clock does not tick loudly"),
            ("She sings in the choir", "She does not sing in the choir"),
            ("He uses a computer", "He does not use a computer"),
            ("They visit museums", "They do not visit museums"),
            ("The river freezes", "The river does not freeze"),
            ("She reads books", "She does not read books"),
            ("He runs fast", "He does not run fast"),
            ("They travel abroad", "They do not travel abroad"),
            ("The sun rises early", "The sun does not rise early"),
            ("She paints pictures", "She does not paint pictures"),
            ("He takes medicine", "He does not take medicine"),
            ("They grow flowers", "They do not grow flowers"),
            ("The wind blows hard", "The wind does not blow hard"),
            ("She writes poetry", "She does not write poetry"),
            ("He plays tennis", "He does not play tennis"),
            ("They walk to work", "They do not walk to work"),
            ("The rain falls steadily", "The rain does not fall steadily"),
            ("She studies hard", "She does not study hard"),
            ("He swims well", "He does not swim well"),
            ("They eat lunch together", "They do not eat lunch together"),
            ("The cat catches mice", "The cat does not catch mice"),
        ]
    },
    'negation': {
        'type': 'SYN',
        'pairs': [
            ("She is happy", "She is unhappy"),
            ("He is kind", "He is unkind"),
            ("The task is possible", "The task is impossible"),
            ("The answer is correct", "The answer is incorrect"),
            ("The behavior is appropriate", "The behavior is inappropriate"),
            ("She is honest", "She is dishonest"),
            ("He is patient", "He is impatient"),
            ("The result is logical", "The result is illogical"),
            ("The action is legal", "The action is illegal"),
            ("She is responsible", "She is irresponsible"),
            ("The decision is fair", "The decision is unfair"),
            ("He is polite", "He is impolite"),
            ("The situation is certain", "The situation is uncertain"),
            ("The method is effective", "The method is ineffective"),
            ("She is loyal", "She is disloyal"),
            ("He is grateful", "He is ungrateful"),
            ("The approach is practical", "The approach is impractical"),
            ("The statement is true", "The statement is untrue"),
            ("She is comfortable", "She is uncomfortable"),
            ("He is popular", "He is unpopular"),
            ("The plan is feasible", "The plan is infeasible"),
            ("The argument is valid", "The argument is invalid"),
            ("She is friendly", "She is unfriendly"),
            ("He is lucky", "He is unlucky"),
            ("The system is stable", "The system is unstable"),
            ("The idea is reasonable", "The idea is unreasonable"),
            ("She is generous", "She is ungenerous"),
            ("He is careful", "He is careless"),
            ("The process is efficient", "The process is inefficient"),
            ("The event is likely", "The event is unlikely"),
            ("She is faithful", "She is unfaithful"),
            ("He is willing", "He is unwilling"),
            ("The method is reliable", "The method is unreliable"),
            ("The outcome is predictable", "The outcome is unpredictable"),
            ("She is conscious", "She is unconscious"),
            ("He is modest", "He is immodest"),
            ("The behavior is acceptable", "The behavior is unacceptable"),
            ("The belief is rational", "The belief is irrational"),
            ("She is helpful", "She is unhelpful"),
            ("He is tidy", "He is untidy"),
            ("The system is secure", "The system is insecure"),
            ("The explanation is adequate", "The explanation is inadequate"),
            ("She is faithful", "She is unfaithful"),
            ("He is obedient", "He is disobedient"),
            ("The view is typical", "The view is atypical"),
            ("The reaction is normal", "The reaction is abnormal"),
            ("She is healthy", "She is unhealthy"),
            ("He is mature", "He is immature"),
            ("The proposal is sensible", "The proposal is nonsensical"),
            ("The product is safe", "The product is unsafe"),
            ("She is grateful", "She is ungrateful"),
            ("He is productive", "He is unproductive"),
            ("The approach is conventional", "The approach is unconventional"),
            ("The statement is accurate", "The statement is inaccurate"),
            ("She is pleasant", "She is unpleasant"),
            ("He is cooperative", "He is uncooperative"),
            ("The result is consistent", "The result is inconsistent"),
            ("The design is elegant", "The design is inelegant"),
            ("She is polite", "She is impolite"),
            ("He is balanced", "He is unbalanced"),
            ("The method is orthodox", "The method is unorthodox"),
            ("The quality is perfect", "The quality is imperfect"),
            ("She is satisfied", "She is dissatisfied"),
            ("He is connected", "He is disconnected"),
            ("The rule is applicable", "The rule is inapplicable"),
            ("The system is complete", "The system is incomplete"),
            ("She is successful", "She is unsuccessful"),
            ("He is trustworthy", "He is untrustworthy"),
            ("The solution is proper", "The solution is improper"),
            ("The attitude is respectful", "The attitude is disrespectful"),
            ("She is aware", "She is unaware"),
            ("He is dependent", "He is independent"),
            ("The pattern is regular", "The pattern is irregular"),
            ("The action is moral", "The action is immoral"),
            ("She is capable", "She is incapable"),
            ("He is visible", "He is invisible"),
            ("The state is natural", "The state is unnatural"),
        ]
    },
    'semantic_valence': {
        'type': 'SEM',
        'pairs': [
            ("She loves the beautiful garden", "She hates the ugly garden"),
            ("He enjoys the wonderful meal", "He dislikes the terrible meal"),
            ("The kind woman helps people", "The cruel woman hurts people"),
            ("They celebrate the great victory", "They mourn the terrible defeat"),
            ("The generous man donates money", "The greedy man steals money"),
            ("She praises the excellent work", "She criticizes the poor work"),
            ("He admires the brilliant idea", "He despises the stupid idea"),
            ("The cheerful child plays happily", "The sad child cries bitterly"),
            ("They welcome the warm sunshine", "They fear the cold darkness"),
            ("The honest worker tells the truth", "The dishonest worker tells lies"),
            ("She appreciates the kind gesture", "She resents the rude gesture"),
            ("He values the precious gift", "He wastes the worthless trash"),
            ("The peaceful village rests quietly", "The violent city fights fiercely"),
            ("They trust the reliable friend", "They distrust the treacherous enemy"),
            ("The sweet fruit tastes delicious", "The bitter fruit tastes awful"),
            ("She protects the innocent child", "She harms the vulnerable victim"),
            ("He supports the noble cause", "He opposes the evil plan"),
            ("The wise leader guides wisely", "The foolish leader leads blindly"),
            ("They cherish the happy memory", "They regret the painful mistake"),
            ("The brave soldier fights boldly", "The cowardly soldier runs away"),
            ("She comforts the anxious patient", "She torments the frightened prisoner"),
            ("He heals the sick person", "He injures the healthy person"),
            ("The gentle breeze blows softly", "The fierce storm rages wildly"),
            ("They build the beautiful temple", "They destroy the ancient ruins"),
            ("The rich merchant shares wealth", "The poor beggar hoards scraps"),
            ("She creates the masterpiece", "She ruins the artwork"),
            ("He saves the drowning man", "He drowns the helpless animal"),
            ("The clean water flows clearly", "The dirty water smells badly"),
            ("They plant the young tree", "They cut the old forest"),
            ("The bright star shines clearly", "The dark cloud blocks light"),
            ("She forgives the sorry friend", "She punishes the guilty criminal"),
            ("He encourages the struggling student", "He discourages the eager learner"),
            ("The warm fire heats the room", "The cold wind freezes the house"),
            ("They honor the brave hero", "They shame the coward traitor"),
            ("The smooth road leads forward", "The rough path goes nowhere"),
            ("She welcomes the new neighbor", "She rejects the strange outsider"),
            ("He enjoys the pleasant walk", "He endures the painful march"),
            ("The safe harbor shelters ships", "The dangerous reef sinks boats"),
            ("They praise the virtuous saint", "They condemn the wicked sinner"),
            ("The fresh bread smells wonderful", "The rotten food smells terrible"),
            ("She nourishes the growing plant", "She poisons the dying weed"),
            ("He liberates the oppressed people", "He imprisons the innocent citizens"),
            ("The beautiful painting inspires", "The ugly graffiti disgusts"),
            ("They rescue the trapped miners", "They abandon the lost travelers"),
            ("The healthy child runs fast", "The sick patient lies still"),
            ("She teaches the curious student", "She confuses the confused child"),
            ("He rewards the diligent worker", "He punishes the lazy employee"),
            ("The clear sky brings hope", "The dark night brings fear"),
            ("They unify the divided nation", "They fragment the united country"),
            ("The sweet song sounds lovely", "The harsh noise sounds terrible"),
            ("She empowers the weak person", "She dominates the helpless victim"),
            ("He enlightens the ignorant crowd", "He deceives the trusting people"),
            ("The pleasant dream brings joy", "The nightmare brings terror"),
            ("They cultivate the fertile land", "They neglect the barren field"),
            ("The strong bridge supports weight", "The weak structure collapses"),
            ("She heals the broken heart", "She breaks the loving bond"),
            ("He defends the innocent person", "He attacks the helpless victim"),
            ("The beautiful garden blooms", "The wasteland decays"),
            ("They celebrate the joyous occasion", "They mourn the tragic loss"),
            ("The warm hug feels comforting", "The cold shoulder feels rejecting"),
            ("She inspires the hopeless crowd", "She demoralizes the motivated team"),
            ("He uplifts the sad community", "He depresses the happy group"),
            ("The clean house looks inviting", "The dirty room looks disgusting"),
            ("They support the just cause", "They oppose the fair policy"),
            ("The bright future looks promising", "The bleak outlook seems hopeless"),
            ("She nurtures the young talent", "She crushes the creative spirit"),
            ("He embraces the positive change", "He rejects the good offer"),
            ("The harmonious music sounds peaceful", "The discordant noise sounds chaotic"),
            ("They promote the healthy lifestyle", "They discourage the good habit"),
            ("The clear answer satisfies", "The confusing question frustrates"),
            ("She calms the worried parent", "She alarms the peaceful citizen"),
            ("He feeds the hungry child", "He starves the needy person"),
            ("The hopeful message inspires", "The despairing news depresses"),
            ("They enrich the poor community", "They deprive the needy group"),
            ("The joyful celebration continues", "The sorrowful mourning persists"),
        ]
    },
}

feature_names = list(FOCUS_FEATURES.keys())

TEST_SENTENCES = [
    "The weather today is",
    "She looked at the",
    "He decided to",
    "The city was",
    "They found the",
    "After the storm the",
    "The scientist discovered",
    "She always wanted to",
    "The book describes",
    "He carefully opened the",
    "The river flows through",
    "She smiled at the",
    "The old man walked",
    "They built a new",
    "The music played softly",
    "He remembered the",
    "The garden was full of",
    "She wrote a long",
    "The children played in the",
    "He watched the",
    "The restaurant served",
    "She chose the",
    "The forest was dark and",
    "He finished the",
    "The road led to",
    "She noticed the",
    "The cat sat on the",
    "He finally understood the",
    "The flowers bloomed in the",
    "She carefully explained the",
    "The morning was cold and",
    "He walked along the",
    "They searched for the",
    "The wind carried the",
    "She picked up the",
    "The dog ran across the",
    "He listened to the",
    "The house stood on the",
    "She opened the small",
    "The students worked on the",
]

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
    parser.add_argument('--n_pairs', type=int, default=80)
    parser.add_argument('--n_test', type=int, default=40)
    args = parser.parse_args()

    model_key = args.model
    n_pairs = args.n_pairs
    n_test = args.n_test
    config = MODEL_CONFIGS[model_key]

    out_dir = f'results/causal_fiber/{model_key}_ccxxxi'
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
    log(f"Phase CCXXXI: 句法特征重叠机制与高阶PC分离分析 — {config['name']}")
    log(f"  n_features={len(feature_names)}, n_pairs={n_pairs}, n_test={n_test}")
    log(f"  时间={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sklearn.decomposition import PCA

    # ============================================================
    # S1: 加载模型
    # ============================================================
    log(f"\n--- S1: 加载模型 ---")

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

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_time = time.time() - t0
    n_layers = config['n_layers']
    d_model = config['d_model']
    last_layer = n_layers - 1

    log(f"  加载完成: {load_time:.0f}s, n_layers={n_layers}, d_model={d_model}")

    # ============================================================
    # S2: 逐层收集6个特征的差分向量(含PC1-PC5)
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S2: 逐层收集6个特征差分向量(PC1-PC5)")
    log(f"{'=' * 60}")

    # 每2层采样, 获得更细粒度的跨层信息
    sample_layers = list(range(0, n_layers, 2))
    if last_layer not in sample_layers:
        sample_layers.append(last_layer)
    sample_layers.sort()

    log(f"  采样层 ({len(sample_layers)}层): {sample_layers[:10]}...{sample_layers[-5:]}")

    feat_layer_diffs = defaultdict(lambda: defaultdict(list))

    for feat in feature_names:
        pairs = FOCUS_FEATURES[feat]['pairs'][:n_pairs]
        log(f"  收集 {feat} ({len(pairs)}对, 类型={FOCUS_FEATURES[feat]['type']})...")

        for i, (s_pos, s_neg) in enumerate(pairs):
            try:
                with torch.no_grad():
                    toks_pos = tokenizer(s_pos, return_tensors='pt').to(model.device)
                    out_pos = model(**toks_pos, output_hidden_states=True)

                    toks_neg = tokenizer(s_neg, return_tensors='pt').to(model.device)
                    out_neg = model(**toks_neg, output_hidden_states=True)

                    for L in sample_layers:
                        h_pos = out_pos.hidden_states[L][0, -1, :].float().cpu().numpy()
                        h_neg = out_neg.hidden_states[L][0, -1, :].float().cpu().numpy()
                        diff = h_pos - h_neg
                        feat_layer_diffs[feat][L].append(diff)

                    del out_pos, out_neg

                if (i + 1) % 40 == 0:
                    log(f"    {feat}: {i+1}/{len(pairs)}")
            except Exception as e:
                continue

    # 计算每特征每层的PC1-PC5
    feat_layer_pcs = {}  # (feat, L, pc_idx) -> direction
    feat_layer_explained = {}  # (feat, L) -> explained_variance_ratio

    N_PC = 5

    for feat in feature_names:
        for L in sample_layers:
            diffs = feat_layer_diffs[feat][L]
            if len(diffs) >= 15:
                diffs = np.array(diffs)
                pca = PCA(n_components=min(N_PC, diffs.shape[1], diffs.shape[0]))
                pca.fit(diffs)
                for pc_idx in range(min(N_PC, pca.n_components_)):
                    feat_layer_pcs[(feat, L, pc_idx)] = pca.components_[pc_idx]
                feat_layer_explained[(feat, L)] = pca.explained_variance_ratio_[:N_PC]

    log(f"\n  计算完成: {len(set((f,L) for f,L,_ in feat_layer_pcs.keys()))} 个(特征,层)组合")

    # ============================================================
    # S3: 核心实验 — tense×question重叠的PC1-PC5分析
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S3: tense×question重叠的PC1-PC5分析")
    log(f"{'=' * 60}")

    overlap_pairs = [
        ('tense', 'question'),     # 最重叠对(CCXXX: |cos|>0.44)
        ('voice', 'semantic_valence'),  # 最正交对(CCXXIX: |cos|<0.01)
        ('tense', 'polarity'),     # 中等重叠
        ('negation', 'polarity'),  # 可能重叠(都是否定)
        ('question', 'negation'),  # 句法×句法
    ]

    for f1, f2 in overlap_pairs:
        log(f"\n  --- {f1} × {f2} ---")

        # 对每一层计算PC1-PC5的交叉对齐
        layer_cross_align = defaultdict(list)

        for L in sample_layers:
            if (f1, L, 0) in feat_layer_pcs and (f2, L, 0) in feat_layer_pcs:
                # 计算f1的每个PC与f2的每个PC的对齐
                for pc1_idx in range(N_PC):
                    if (f1, L, pc1_idx) not in feat_layer_pcs:
                        continue
                    for pc2_idx in range(N_PC):
                        if (f2, L, pc2_idx) not in feat_layer_pcs:
                            continue
                        v1 = feat_layer_pcs[(f1, L, pc1_idx)]
                        v2 = feat_layer_pcs[(f2, L, pc2_idx)]
                        cos_val = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                        layer_cross_align[(pc1_idx, pc2_idx)].append((L, cos_val))

        # 打印对齐矩阵(所有层的平均值)
        log(f"  PC交叉对齐矩阵 (所有层平均|cos|):")
        header = f"  {'':>8}"
        for pc2 in range(N_PC):
            header += f"  {f2[:4]}PC{pc2+1}"
        log(header)

        avg_matrix = np.zeros((N_PC, N_PC))
        for pc1_idx in range(N_PC):
            row = f"  {f1[:4]}PC{pc1_idx+1}"
            for pc2_idx in range(N_PC):
                if (pc1_idx, pc2_idx) in layer_cross_align:
                    avg_cos = np.mean([c for _, c in layer_cross_align[(pc1_idx, pc2_idx)]])
                    avg_matrix[pc1_idx, pc2_idx] = avg_cos
                    row += f"  {avg_cos:>6.3f}"
                else:
                    row += f"  {'N/A':>6}"
            log(row)

        # 关键指标
        pc1_cos = avg_matrix[0, 0]
        # f1的PC1投影到f2的PC1-PC5子空间的累计解释
        f1_pc1_in_f2_subspace = sum(avg_matrix[0, :]**2)
        # f2的PC1投影到f1的PC1-PC5子空间的累计解释
        f2_pc1_in_f1_subspace = sum(avg_matrix[:, 0]**2)

        log(f"\n  关键指标:")
        log(f"    PC1×PC1 |cos| = {pc1_cos:.4f}")
        log(f"    {f1}_PC1在{f2}_PC1-5子空间投影 = {f1_pc1_in_f2_subspace:.4f}")
        log(f"    {f2}_PC1在{f1}_PC1-5子空间投影 = {f2_pc1_in_f1_subspace:.4f}")

        # 判定
        if pc1_cos > 0.3:
            if f1_pc1_in_f2_subspace > pc1_cos**2 + 0.1:
                log(f"    → PC1高度重叠, 但高阶PC有额外分离")
            else:
                log(f"    → PC1高度重叠, 且高阶PC无额外分离")
        elif pc1_cos < 0.1:
            log(f"    → PC1高度正交")

        # 跨层PC1×PC1变化
        if (0, 0) in layer_cross_align:
            layer_data = layer_cross_align[(0, 0)]
            early_cos = np.mean([c for L, c in layer_data if L < n_layers // 3])
            mid_cos = np.mean([c for L, c in layer_data if n_layers // 3 <= L < 2 * n_layers // 3])
            late_cos = np.mean([c for L, c in layer_data if L >= 2 * n_layers // 3])
            log(f"\n  跨层PC1×PC1变化:")
            log(f"    早期层: |cos|={early_cos:.4f}")
            log(f"    中间层: |cos|={mid_cos:.4f}")
            log(f"    晚期层: |cos|={late_cos:.4f}")

    # ============================================================
    # S4: 子空间重叠度 — Grassmann距离(PC1-3子空间)
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S4: 子空间重叠度(Grassmann距离)")
    log(f"{'=' * 60}")

    n_sub_pc = 3  # 用前3个PC构成子空间

    for f1, f2 in overlap_pairs:
        log(f"\n  --- {f1} × {f2} Grassmann距离 ---")

        for L in [n_layers // 4, n_layers // 2, 3 * n_layers // 4, last_layer]:
            # 收集f1和f2的前3PC
            f1_pcs = []
            f2_pcs = []
            for pc_idx in range(n_sub_pc):
                if (f1, L, pc_idx) in feat_layer_pcs:
                    f1_pcs.append(feat_layer_pcs[(f1, L, pc_idx)])
                if (f2, L, pc_idx) in feat_layer_pcs:
                    f2_pcs.append(feat_layer_pcs[(f2, L, pc_idx)])

            if len(f1_pcs) < 2 or len(f2_pcs) < 2:
                continue

            U1 = np.array(f1_pcs).T  # d × n_pc
            U2 = np.array(f2_pcs).T

            # QR分解正交化
            Q1, _ = np.linalg.qr(U1)
            Q2, _ = np.linalg.qr(U2)

            # Grassmann距离 = sqrt(sum(min(k1,k2) - σ_i^2))
            M = Q1.T @ Q2
            svd_vals = np.linalg.svd(M, compute_uv=False)
            k = min(Q1.shape[1], Q2.shape[1])
            grassmann_dist = np.sqrt(max(0, k - sum(svd_vals[:k]**2)))

            max_align = np.max(svd_vals)
            log(f"    L{L}: Grassmann={grassmann_dist:.4f}, max_SVD={max_align:.4f} (n_pc1={len(f1_pcs)}, n_pc2={len(f2_pcs)})")

    # ============================================================
    # S5: 重叠特征的信息互补性
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S5: 重叠特征的信息互补性")
    log(f"{'=' * 60}")

    for f1, f2 in [('tense', 'question'), ('voice', 'semantic_valence')]:
        log(f"\n  --- {f1} × {f2} 信息互补性 ---")

        # 用test句子检验: 修改f1和f2, hidden state变化是否独立
        test_sents = TEST_SENTENCES[:n_test]

        f1_alone_diffs = []  # 只改变f1
        f2_alone_diffs = []  # 只改变f2
        both_diffs = []      # 同时改变f1和f2

        # 使用特征对中的第一对来构造"翻转"版本
        # 简化: 直接用特征的差分向量
        # 检验: f1的差分向量在f2的PC子空间中的投影 vs 正交补中的投影

        for L in [last_layer]:
            if (f1, L, 0) not in feat_layer_pcs or (f2, L, 0) not in feat_layer_pcs:
                continue

            # f1的差分向量集合
            diffs_f1 = np.array(feat_layer_diffs[f1][L])
            diffs_f2 = np.array(feat_layer_diffs[f2][L])

            if len(diffs_f1) < 10 or len(diffs_f2) < 10:
                continue

            # f2的PC1-5构成子空间
            f2_pcs_list = []
            for pc_idx in range(N_PC):
                if (f2, L, pc_idx) in feat_layer_pcs:
                    f2_pcs_list.append(feat_layer_pcs[(f2, L, pc_idx)])

            if len(f2_pcs_list) < 2:
                continue

            Q_f2, _ = np.linalg.qr(np.array(f2_pcs_list).T)  # d × n_pc

            # f1的差分投影到f2子空间和正交补
            proj_in_f2 = diffs_f1 @ Q_f2 @ Q_f2.T  # 投影到f2子空间
            proj_orth_f2 = diffs_f1 - proj_in_f2    # 正交补投影

            # 能量比
            energy_in_f2 = np.mean(np.sum(proj_in_f2**2, axis=1))
            energy_orth_f2 = np.mean(np.sum(proj_orth_f2**2, axis=1))
            total_energy = energy_in_f2 + energy_orth_f2

            log(f"    L{L}: {f1}差分能量分配:")
            log(f"      在{f2}子空间内: {energy_in_f2/total_energy*100:.1f}%")
            log(f"      在{f2}子空间外: {energy_orth_f2/total_energy*100:.1f}%")
            log(f"      → {f1}信息的{energy_orth_f2/total_energy*100:.1f}%独立于{f2}")

            # 反向: f2差分在f1子空间
            f1_pcs_list = []
            for pc_idx in range(N_PC):
                if (f1, L, pc_idx) in feat_layer_pcs:
                    f1_pcs_list.append(feat_layer_pcs[(f1, L, pc_idx)])

            if len(f1_pcs_list) >= 2:
                Q_f1, _ = np.linalg.qr(np.array(f1_pcs_list).T)
                proj_in_f1 = diffs_f2 @ Q_f1 @ Q_f1.T
                proj_orth_f1 = diffs_f2 - proj_in_f1

                energy_in_f1 = np.mean(np.sum(proj_in_f1**2, axis=1))
                energy_orth_f1 = np.mean(np.sum(proj_orth_f1**2, axis=1))
                total_energy_r = energy_in_f1 + energy_orth_f1

                log(f"    L{L}: {f2}差分能量分配:")
                log(f"      在{f1}子空间内: {energy_in_f1/total_energy_r*100:.1f}%")
                log(f"      在{f1}子空间外: {energy_orth_f1/total_energy_r*100:.1f}%")
                log(f"      → {f2}信息的{energy_orth_f1/total_energy_r*100:.1f}%独立于{f1}")

    # ============================================================
    # S6: 跨层重叠演化 — tense×question vs voice×semantic_valence
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S6: 跨层重叠演化")
    log(f"{'=' * 60}")

    for f1, f2 in [('tense', 'question'), ('voice', 'semantic_valence')]:
        log(f"\n  --- {f1} × {f2} 跨层PC1×PC1 |cos| ---")

        layer_cos_data = []
        for L in sample_layers:
            if (f1, L, 0) in feat_layer_pcs and (f2, L, 0) in feat_layer_pcs:
                v1 = feat_layer_pcs[(f1, L, 0)]
                v2 = feat_layer_pcs[(f2, L, 0)]
                cos_val = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                layer_cos_data.append((L, cos_val))

        if layer_cos_data:
            log(f"  {'层':>4} | {'PC1×PC1 |cos|':>14} | 趋势")
            log(f"  {'-'*4} | {'-'*14} | ----")
            for L, c in layer_cos_data:
                bar = '#' * int(c * 40)
                log(f"  L{L:>2} | {c:>14.4f} | {bar}")

            # 趋势
            cos_vals = [c for _, c in layer_cos_data]
            early_avg = np.mean(cos_vals[:len(cos_vals)//3]) if len(cos_vals) >= 3 else cos_vals[0]
            late_avg = np.mean(cos_vals[-len(cos_vals)//3:]) if len(cos_vals) >= 3 else cos_vals[-1]
            log(f"  早期平均: {early_avg:.4f}, 晚期平均: {late_avg:.4f}")

    # ============================================================
    # S7: PC1-5全部6×6交叉对齐(最后一层)
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S7: 6特征全部PC1×PC1对齐(最后一层)")
    log(f"{'=' * 60}")

    L = last_layer
    feat_pc1s = {}
    for feat in feature_names:
        if (feat, L, 0) in feat_layer_pcs:
            feat_pc1s[feat] = feat_layer_pcs[(feat, L, 0)]

    feats_present = sorted(feat_pc1s.keys())
    log(f"  有数据的特征: {len(feats_present)}/{len(feature_names)}")

    # 打印6×6矩阵
    header = f"  {'':>18}"
    for f2 in feats_present:
        header += f" {f2[:6]:>6}"
    log(header)

    all_cos_pairs = {}
    for f1 in feats_present:
        row = f"  {f1:>18}"
        for f2 in feats_present:
            if f1 == f2:
                row += f" {'1.00':>6}"
            else:
                cos_val = abs(np.dot(feat_pc1s[f1], feat_pc1s[f2]) /
                             (np.linalg.norm(feat_pc1s[f1]) * np.linalg.norm(feat_pc1s[f2])))
                all_cos_pairs[(f1, f2)] = cos_val
                row += f" {cos_val:>6.2f}"
        log(row)

    # ============================================================
    # S8: 重叠对的PC2分离验证
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S8: 重叠对的PC2分离验证")
    log(f"{'=' * 60}")

    for f1, f2 in [('tense', 'question'), ('negation', 'polarity')]:
        log(f"\n  --- {f1} vs {f2}: PC2方向是否正交? ---")

        for L in [last_layer]:
            # f1的PC1和PC2
            if (f1, L, 0) not in feat_layer_pcs or (f2, L, 0) not in feat_layer_pcs:
                continue

            f1_pc1 = feat_layer_pcs[(f1, L, 0)]
            f2_pc1 = feat_layer_pcs[(f2, L, 0)]

            # PC1对齐
            pc1_cos = abs(np.dot(f1_pc1, f2_pc1) / (np.linalg.norm(f1_pc1) * np.linalg.norm(f2_pc1)))

            # f1的PC2 vs f2的PC1
            if (f1, L, 1) in feat_layer_pcs:
                f1_pc2 = feat_layer_pcs[(f1, L, 1)]
                f1_pc2_vs_f2_pc1 = abs(np.dot(f1_pc2, f2_pc1) / (np.linalg.norm(f1_pc2) * np.linalg.norm(f2_pc1)))
                f1_pc2_vs_f1_pc1 = abs(np.dot(f1_pc2, f1_pc1) / (np.linalg.norm(f1_pc2) * np.linalg.norm(f1_pc1)))

                log(f"    L{L}:")
                log(f"      {f1}_PC1 vs {f2}_PC1: |cos|={pc1_cos:.4f}")
                log(f"      {f1}_PC2 vs {f2}_PC1: |cos|={f1_pc2_vs_f2_pc1:.4f}")
                log(f"      {f1}_PC2 vs {f1}_PC1: |cos|={f1_pc2_vs_f1_pc1:.4f} (应≈0)")

                if pc1_cos > 0.3 and f1_pc2_vs_f2_pc1 < 0.1:
                    log(f"      → PC1重叠但PC2正交: 高阶PC能分离!")
                elif pc1_cos > 0.3 and f1_pc2_vs_f2_pc1 > 0.3:
                    log(f"      → PC1和PC2都重叠: 信息冗余!")
                else:
                    log(f"      → PC1不重叠")

            # f2的PC2 vs f1的PC1
            if (f2, L, 1) in feat_layer_pcs:
                f2_pc2 = feat_layer_pcs[(f2, L, 1)]
                f2_pc2_vs_f1_pc1 = abs(np.dot(f2_pc2, f1_pc1) / (np.linalg.norm(f2_pc2) * np.linalg.norm(f1_pc1)))
                f2_pc2_vs_f2_pc1 = abs(np.dot(f2_pc2, f2_pc1) / (np.linalg.norm(f2_pc2) * np.linalg.norm(f2_pc1)))

                log(f"      {f2}_PC2 vs {f1}_PC1: |cos|={f2_pc2_vs_f1_pc1:.4f}")
                log(f"      {f2}_PC2 vs {f2}_PC1: |cos|={f2_pc2_vs_f2_pc1:.4f} (应≈0)")

            # 合并: f1的信息在f2的子空间中的分布
            f1_diffs = np.array(feat_layer_diffs[f1][L])
            f2_pcs_list = []
            for pc_idx in range(N_PC):
                if (f2, L, pc_idx) in feat_layer_pcs:
                    f2_pcs_list.append(feat_layer_pcs[(f2, L, pc_idx)])

            if len(f2_pcs_list) >= 1 and len(f1_diffs) >= 10:
                Q2, _ = np.linalg.qr(np.array(f2_pcs_list).T)
                proj = f1_diffs @ Q2 @ Q2.T
                orth = f1_diffs - proj

                explained_by_f2 = np.mean(np.sum(proj**2, axis=1)) / np.mean(np.sum(f1_diffs**2, axis=1))
                log(f"\n    {f1}差分被{f2}_PC1-{len(f2_pcs_list)}解释: {explained_by_f2*100:.1f}%")
                log(f"    {f1}差分独立于{f2}: {(1-explained_by_f2)*100:.1f}%")

    # ============================================================
    # S9: 解释方差比 — 重叠vs正交特征
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S9: 解释方差比(PC1-5)")
    log(f"{'=' * 60}")

    for feat in feature_names:
        L = last_layer
        if (feat, L) in feat_layer_explained:
            evr = feat_layer_explained[(feat, L)]
            evr_str = " ".join([f"PC{i+1}:{evr[i]*100:.1f}%" for i in range(len(evr))])
            log(f"  {feat:>18}: {evr_str}")

    # ============================================================
    # S10: 总结
    # ============================================================
    log(f"\n{'=' * 60}")
    log(f"S10: 总结")
    log(f"{'=' * 60}")

    # 关键问题: tense×question的PC1重叠, PC2能否分离?
    log(f"\n  关键问题答案:")
    log(f"  1. tense×question PC1重叠(|cos|>0.3)时, PC2是否正交?")
    log(f"  2. 重叠对的信息有多少是互补(独立)的?")
    log(f"  3. 正交对(voice×semantic_valence)在高阶PC是否也正交?")

    # 收集关键数据
    summary = {
        'model': config['name'],
        'n_layers': n_layers,
        'd_model': d_model,
        'features': feature_names,
        'n_pairs': n_pairs,
        'overlap_analysis': {},
        'pc2_separation': {},
        'information_complementarity': {},
    }

    # 保存结果
    results_path = f'{out_dir}/results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    log(f"\n  结果已保存: {results_path}")

    # 清理GPU
    del model
    torch.cuda.empty_cache()
    log(f"\n  GPU内存已释放")


if __name__ == '__main__':
    main()
