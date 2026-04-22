"""
Phase CCXVII: Per-feature Head贡献矩阵 + 全层PCA扫描
======================================================
核心目标:
1. 构建 [Head × Feature] 贡献矩阵, 发现Head是否按具体功能专化
2. 全层PCA扫描, 验证DS7B L14信息瓶颈
3. 增大样本到50对/特征(实际最大40)

S1: 全层差分向量收集 (12特征 × 全层 × 50对)
    → 每隔4层采样, 减少计算量
S2: 全层PCA扫描 → 维度/方差/正交性在层间的演变
S3: Per-feature Head贡献矩阵 (只分析3关键层)
    → [n_heads × 12]矩阵, 每个元素=Head h对Feature f的平均贡献
S4: Head聚类分析 → Head是否按功能聚类
S5: 因果原子词典构建
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
# 特征对定义 (同CCXVI)
# ============================================================
FEATURE_PAIRS = {
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
            ("They build houses here", "They built houses here"),
            ("She paints landscapes", "She painted landscapes"),
            ("The river flows to the sea", "The river flowed to the sea"),
            ("We study mathematics", "We studied mathematics"),
            ("He fixes computers", "He fixed computers"),
            ("The children laugh at jokes", "The children laughed at jokes"),
            ("I drink coffee every morning", "I drank coffee this morning"),
            ("The sun rises in the east", "The sun rose in the east"),
            ("She teaches young students", "She taught young students"),
            ("They dance at parties", "They danced at parties"),
            ("We visit museums often", "We visited museums often"),
            ("He reads the newspaper", "He read the newspaper"),
            ("The wind blows gently", "The wind blew gently"),
            ("I bake bread on weekends", "I baked bread last weekend"),
            ("She knits sweaters", "She knitted sweaters"),
            ("The train arrives at noon", "The train arrived at noon"),
            ("They swim in the lake", "They swam in the lake"),
            ("We plant flowers in spring", "We planted flowers in spring"),
            ("He draws cartoons", "He drew cartoons"),
            ("The cat chases butterflies", "The cat chased butterflies"),
            ("I ride my bicycle", "I rode my bicycle"),
            ("She types reports quickly", "She typed reports quickly"),
            ("The bell rings loudly", "The bell rang loudly"),
            ("They climb mountains", "They climbed mountains"),
            ("We watch television", "We watched television"),
            ("The fire burns brightly", "The fire burned brightly"),
        ],
    },
    'polarity': {
        'type': 'SYN',
        'pairs': [
            ("The cat is on the mat", "The cat is not on the mat"),
            ("She likes the movie", "She does not like the movie"),
            ("He can swim well", "He cannot swim well"),
            ("They have finished the work", "They have not finished the work"),
            ("I will go to the party", "I will not go to the party"),
            ("The door is open", "The door is not open"),
            ("We should leave now", "We should not leave now"),
            ("She was happy yesterday", "She was not happy yesterday"),
            ("He has seen the film", "He has not seen the film"),
            ("The students passed the exam", "The students did not pass the exam"),
            ("I understand the question", "I do not understand the question"),
            ("They are coming tomorrow", "They are not coming tomorrow"),
            ("The car needs repairs", "The car does not need repairs"),
            ("We enjoyed the concert", "We did not enjoy the concert"),
            ("She speaks French fluently", "She does not speak French fluently"),
            ("The machine works properly", "The machine does not work properly"),
            ("He believes in ghosts", "He does not believe in ghosts"),
            ("I remember the address", "I do not remember the address"),
            ("They found the treasure", "They did not find the treasure"),
            ("The garden looks beautiful", "The garden does not look beautiful"),
            ("We need more time", "We do not need more time"),
            ("She plays the piano", "She does not play the piano"),
            ("The dog likes bones", "The dog does not like bones"),
            ("He knows the answer", "He does not know the answer"),
            ("I want some coffee", "I do not want any coffee"),
            ("The baby is sleeping", "The baby is not sleeping"),
            ("They own a house", "They do not own a house"),
            ("The soup tastes good", "The soup does not taste good"),
            ("We support the team", "We do not support the team"),
            ("She cares about animals", "She does not care about animals"),
            ("The plan makes sense", "The plan does not make sense"),
            ("He reads novels", "He does not read novels"),
            ("I like chocolate", "I do not like chocolate"),
            ("The weather is warm", "The weather is not warm"),
            ("They eat meat", "They do not eat meat"),
            ("We agree with you", "We do not agree with you"),
            ("The light is on", "The light is not on"),
            ("She wears glasses", "She does not wear glasses"),
            ("The store is open", "The store is not open"),
            ("He drives carefully", "He does not drive carefully"),
        ],
    },
    'number': {
        'type': 'SYN',
        'pairs': [
            ("The cat sleeps on the mat", "The cats sleep on the mat"),
            ("A dog barks loudly", "Some dogs bark loudly"),
            ("This book is interesting", "These books are interesting"),
            ("That house is big", "Those houses are big"),
            ("The child plays outside", "The children play outside"),
            ("A bird sings in the tree", "Some birds sing in the tree"),
            ("This car runs fast", "These cars run fast"),
            ("The student reads a lot", "The students read a lot"),
            ("A flower blooms in spring", "Some flowers bloom in spring"),
            ("That tree is very tall", "Those trees are very tall"),
            ("The chair is comfortable", "The chairs are comfortable"),
            ("A star shines brightly", "Some stars shine brightly"),
            ("This problem is difficult", "These problems are difficult"),
            ("The monkey eats bananas", "The monkeys eat bananas"),
            ("A river flows through town", "Some rivers flow through town"),
            ("That mountain is snow-covered", "Those mountains are snow-covered"),
            ("The letter arrived today", "The letters arrived today"),
            ("A cloud drifted overhead", "Some clouds drifted overhead"),
            ("This idea seems good", "These ideas seem good"),
            ("The sheep grazes quietly", "The sheep graze quietly"),
            ("A fish swims upstream", "Some fish swim upstream"),
            ("That building is old", "Those buildings are old"),
            ("The key opens the door", "The keys open the door"),
            ("A leaf fell from the tree", "Some leaves fell from the tree"),
            ("This method works well", "These methods work well"),
            ("The country is beautiful", "The countries are beautiful"),
            ("A village sits in the valley", "Some villages sit in the valley"),
            ("That island is tropical", "Those islands are tropical"),
            ("The box contains gold", "The boxes contain gold"),
            ("A butterfly lands on flowers", "Some butterflies land on flowers"),
            ("This story is funny", "These stories are funny"),
            ("The city never sleeps", "The cities never sleep"),
            ("A robot cleans the floor", "Some robots clean the floor"),
            ("That picture is colorful", "Those pictures are colorful"),
            ("The egg is fresh", "The eggs are fresh"),
            ("A teacher helps students", "Some teachers help students"),
            ("This path leads nowhere", "These paths lead nowhere"),
            ("The window needs cleaning", "The windows need cleaning"),
            ("A customer pays the bill", "Some customers pay the bill"),
            ("That river is deep", "Those rivers are deep"),
        ],
    },
    'negation': {
        'type': 'SYN',
        'pairs': [
            ("She is always happy", "She is never happy"),
            ("He sometimes forgets", "He never forgets"),
            ("They often visit us", "They rarely visit us"),
            ("I always finish my work", "I never finish my work"),
            ("The bus usually comes on time", "The bus seldom comes on time"),
            ("She frequently travels abroad", "She hardly travels abroad"),
            ("He generally wakes up early", "He barely wakes up early"),
            ("They consistently win matches", "They rarely win matches"),
            ("I mostly eat at home", "I seldom eat at home"),
            ("The store typically opens early", "The store rarely opens early"),
            ("She almost always smiles", "She almost never smiles"),
            ("He regularly attends meetings", "He seldom attends meetings"),
            ("They normally agree with us", "They rarely agree with us"),
            ("I practically always succeed", "I practically never succeed"),
            ("The train usually arrives late", "The train seldom arrives late"),
            ("She frequently wins prizes", "She rarely wins prizes"),
            ("He often helps others", "He seldom helps others"),
            ("They usually eat together", "They rarely eat together"),
            ("I generally feel well", "I rarely feel well"),
            ("The team consistently performs well", "The team rarely performs well"),
            ("She always tells the truth", "She never tells the truth"),
            ("He mostly drives carefully", "He seldom drives carefully"),
            ("They frequently go camping", "They rarely go camping"),
            ("I usually drink tea", "I seldom drink tea"),
            ("The class normally starts early", "The class rarely starts early"),
            ("She almost always arrives first", "She almost never arrives first"),
            ("He regularly exercises", "He rarely exercises"),
            ("They consistently meet deadlines", "They rarely meet deadlines"),
            ("I mostly stay calm", "I seldom stay calm"),
            ("The dog usually barks at strangers", "The dog seldom barks at strangers"),
            ("She frequently reads novels", "She rarely reads novels"),
            ("He generally pays on time", "He rarely pays on time"),
            ("They often celebrate together", "They seldom celebrate together"),
            ("I always check my work", "I never check my work"),
            ("The sun typically rises early", "The sun rarely rises early"),
            ("She mostly wears bright colors", "She seldom wears bright colors"),
            ("He regularly visits his parents", "He rarely visits his parents"),
            ("They usually finish quickly", "They rarely finish quickly"),
            ("I frequently take the bus", "I rarely take the bus"),
            ("The store always has fresh bread", "The store never has fresh bread"),
        ],
    },
    'question': {
        'type': 'SYN',
        'pairs': [
            ("The cat is on the mat", "Is the cat on the mat?"),
            ("She likes the movie", "Does she like the movie?"),
            ("He can swim well", "Can he swim well?"),
            ("They have finished the work", "Have they finished the work?"),
            ("I will go to the party", "Will I go to the party?"),
            ("The door is open", "Is the door open?"),
            ("We should leave now", "Should we leave now?"),
            ("She was happy yesterday", "Was she happy yesterday?"),
            ("He has seen the film", "Has he seen the film?"),
            ("The students passed the exam", "Did the students pass the exam?"),
            ("I understand the question", "Do I understand the question?"),
            ("They are coming tomorrow", "Are they coming tomorrow?"),
            ("The car needs repairs", "Does the car need repairs?"),
            ("We enjoyed the concert", "Did we enjoy the concert?"),
            ("She speaks French fluently", "Does she speak French fluently?"),
            ("The machine works properly", "Does the machine work properly?"),
            ("He believes in ghosts", "Does he believe in ghosts?"),
            ("I remember the address", "Do I remember the address?"),
            ("They found the treasure", "Did they find the treasure?"),
            ("The garden looks beautiful", "Does the garden look beautiful?"),
            ("We need more time", "Do we need more time?"),
            ("She plays the piano", "Does she play the piano?"),
            ("The dog likes bones", "Does the dog like bones?"),
            ("He knows the answer", "Does he know the answer?"),
            ("I want some coffee", "Do I want some coffee?"),
            ("The baby is sleeping", "Is the baby sleeping?"),
            ("They own a house", "Do they own a house?"),
            ("The soup tastes good", "Does the soup taste good?"),
            ("We support the team", "Do we support the team?"),
            ("She cares about animals", "Does she care about animals?"),
            ("The plan makes sense", "Does the plan make sense?"),
            ("He reads novels", "Does he read novels?"),
            ("I like chocolate", "Do I like chocolate?"),
            ("The weather is warm", "Is the weather warm?"),
            ("They eat meat", "Do they eat meat?"),
            ("We agree with you", "Do we agree with you?"),
            ("The light is on", "Is the light on?"),
            ("She wears glasses", "Does she wear glasses?"),
            ("The store is open", "Is the store open?"),
            ("He drives carefully", "Does he drive carefully?"),
        ],
    },
    'person': {
        'type': 'SYN',
        'pairs': [
            ("I walk to the store", "She walks to the store"),
            ("I have a book", "She has a book"),
            ("I am a student", "She is a student"),
            ("I was there yesterday", "She was there yesterday"),
            ("I can do it myself", "She can do it herself"),
            ("I like coffee", "He likes coffee"),
            ("I go to school", "He goes to school"),
            ("I am happy today", "He is happy today"),
            ("I was tired last night", "He was tired last night"),
            ("I have finished my work", "He has finished his work"),
            ("We play soccer", "They play soccer"),
            ("We are friends", "They are friends"),
            ("We were at home", "They were at home"),
            ("We have a plan", "They have a plan"),
            ("We can help you", "They can help you"),
            ("I eat breakfast early", "She eats breakfast early"),
            ("I read the newspaper", "He reads the newspaper"),
            ("I write in my diary", "She writes in her diary"),
            ("We study together", "They study together"),
            ("We work hard every day", "They work hard every day"),
            ("I run five miles daily", "She runs five miles daily"),
            ("I sing in the choir", "He sings in the choir"),
            ("We cook dinner together", "They cook dinner together"),
            ("We travel every summer", "They travel every summer"),
            ("I paint watercolors", "She paints watercolors"),
            ("I drive a blue car", "He drives a blue car"),
            ("We dance at weddings", "They dance at weddings"),
            ("We visit our grandparents", "They visit their grandparents"),
            ("I enjoy reading mystery novels", "She enjoys reading mystery novels"),
            ("I play the guitar", "He plays the guitar"),
            ("We support local businesses", "They support local businesses"),
            ("We celebrate holidays together", "They celebrate holidays together"),
            ("I bake chocolate cookies", "She bakes chocolate cookies"),
            ("I swim in the ocean", "He swims in the ocean"),
            ("We hike on weekends", "They hike on weekends"),
            ("We volunteer at the shelter", "They volunteer at the shelter"),
            ("I teach mathematics", "She teaches mathematics"),
            ("I grow tomatoes in my garden", "He grows tomatoes in his garden"),
            ("We walk in the park", "They walk in the park"),
            ("We watch movies on Friday", "They watch movies on Friday"),
        ],
    },
    'definiteness': {
        'type': 'SYN',
        'pairs': [
            ("A cat sleeps on the mat", "The cat sleeps on the mat"),
            ("A dog barks in the yard", "The dog barks in the yard"),
            ("An apple falls from the tree", "The apple falls from the tree"),
            ("A student reads a book", "The student reads the book"),
            ("A bird sings in the morning", "The bird sings in the morning"),
            ("A car drives down the street", "The car drives down the street"),
            ("A flower blooms in spring", "The flower blooms in spring"),
            ("A child plays in the garden", "The child plays in the garden"),
            ("A star shines at night", "The star shines at night"),
            ("A house stands on the hill", "The house stands on the hill"),
            ("A river flows through the valley", "The river flows through the valley"),
            ("A teacher explains the lesson", "The teacher explains the lesson"),
            ("A book lies on the table", "The book lies on the table"),
            ("A cloud drifts across the sky", "The cloud drifts across the sky"),
            ("A door opens slowly", "The door opens slowly"),
            ("A man walks along the road", "The man walks along the road"),
            ("A lamp glows in the window", "The lamp glows in the window"),
            ("A fish swims in the pond", "The fish swims in the pond"),
            ("A tree grows near the fence", "The tree grows near the fence"),
            ("A cat chases the mouse", "The cat chases the mouse"),
            ("A plane flies overhead", "The plane flies overhead"),
            ("A bell rings in the tower", "The bell rings in the tower"),
            ("A boat sails on the lake", "The boat sails on the lake"),
            ("A train arrives at the station", "The train arrives at the station"),
            ("A dog guards the house", "The dog guards the house"),
            ("A girl reads quietly", "The girl reads quietly"),
            ("A cake sits on the plate", "The cake sits on the plate"),
            ("A song plays on the radio", "The song plays on the radio"),
            ("A letter arrives in the mail", "The letter arrives in the mail"),
            ("A key opens the lock", "The key opens the lock"),
            ("A boy runs on the field", "The boy runs on the field"),
            ("A chair sits in the corner", "The chair sits in the corner"),
            ("A candle burns on the table", "The candle burns on the table"),
            ("A horse gallops across the meadow", "The horse gallops across the meadow"),
            ("A kite flies in the wind", "The kite flies in the wind"),
            ("A robot cleans the floor", "The robot cleans the floor"),
            ("A picture hangs on the wall", "The picture hangs on the wall"),
            ("A window faces the garden", "The window faces the garden"),
            ("A bridge crosses the river", "The bridge crosses the river"),
            ("A path leads to the forest", "The path leads to the forest"),
        ],
    },
    'info_structure': {
        'type': 'SYN',
        'pairs': [
            ("John broke the window", "It was John who broke the window"),
            ("Mary found the key", "It was Mary who found the key"),
            ("Tom won the prize", "It was Tom who won the prize"),
            ("Alice wrote the letter", "It was Alice who wrote the letter"),
            ("Bob fixed the computer", "It was Bob who fixed the computer"),
            ("Sarah cooked the dinner", "It was Sarah who cooked the dinner"),
            ("David painted the house", "It was David who painted the house"),
            ("Emma solved the puzzle", "It was Emma who solved the puzzle"),
            ("James drove the bus", "It was James who drove the bus"),
            ("Linda sang the song", "It was Linda who sang the song"),
            ("Peter built the cabinet", "It was Peter who built the cabinet"),
            ("Nancy designed the logo", "It was Nancy who designed the logo"),
            ("Steve caught the fish", "It was Steve who caught the fish"),
            ("Rachel baked the cake", "It was Rachel who baked the cake"),
            ("Mike repaired the roof", "It was Mike who repaired the roof"),
            ("Julia taught the class", "It was Julia who taught the class"),
            ("Chris planted the tree", "It was Chris who planted the tree"),
            ("Anna organized the event", "It was Anna who organized the event"),
            ("Mark discovered the truth", "It was Mark who discovered the truth"),
            ("Lisa wrote the report", "It was Lisa who wrote the report"),
            ("Paul delivered the package", "It was Paul who delivered the package"),
            ("Kate translated the document", "It was Kate who translated the document"),
            ("Ryan composed the music", "It was Ryan who composed the music"),
            ("Amy directed the film", "It was Amy who directed the film"),
            ("Greg invented the device", "It was Greg who invented the device"),
            ("Sue managed the project", "It was Sue who managed the project"),
            ("Tim coached the team", "It was Tim who coached the team"),
            ("Judy decorated the room", "It was Judy who decorated the room"),
            ("Phil photographed the wedding", "It was Phil who photographed the wedding"),
            ("Helen edited the video", "It was Helen who edited the video"),
            ("Joe programmed the app", "It was Joe who programmed the app"),
            ("Cathy modeled the dress", "It was Cathy who modeled the dress"),
            ("Dan analyzed the data", "It was Dan who analyzed the data"),
            ("Grace led the expedition", "It was Grace who led the expedition"),
            ("Sam carved the statue", "It was Sam who carved the statue"),
            ("Rosa performed the surgery", "It was Rosa who performed the surgery"),
            ("Erik navigated the ship", "It was Erik who navigated the ship"),
            ("Yuki arranged the flowers", "It was Yuki who arranged the flowers"),
            ("Carlos brewed the coffee", "It was Carlos who brewed the coffee"),
            ("Marie painted the portrait", "It was Marie who painted the portrait"),
        ],
    },
    'sentiment': {
        'type': 'SEM',
        'pairs': [
            ("The movie was wonderful and exciting", "The movie was terrible and boring"),
            ("She had a great day at work", "She had a awful day at work"),
            ("The food was delicious and fresh", "The food was disgusting and stale"),
            ("He is a kind and generous person", "He is a cruel and selfish person"),
            ("The weather is beautiful today", "The weather is horrible today"),
            ("The book was fascinating and engaging", "The book was dull and tedious"),
            ("She gave a brilliant performance", "She gave a terrible performance"),
            ("The garden looks lovely in spring", "The garden looks ugly in spring"),
            ("He received a warm welcome", "He received a cold welcome"),
            ("The concert was amazing last night", "The concert was dreadful last night"),
            ("She looks stunning in that dress", "She looks awful in that dress"),
            ("The project was a huge success", "The project was a massive failure"),
            ("He told a hilarious joke", "He told a boring joke"),
            ("The vacation was perfect", "The vacation was disastrous"),
            ("She speaks with gentle words", "She speaks with harsh words"),
            ("The sunrise was breathtaking", "The smog was suffocating"),
            ("He has a bright future ahead", "He has a bleak future ahead"),
            ("The cake was sweet and tasty", "The cake was bitter and tasteless"),
            ("She showed great compassion", "She showed great cruelty"),
            ("The team won a glorious victory", "The team suffered a humiliating defeat"),
            ("He felt joyful and content", "He felt miserable and frustrated"),
            ("The music was soothing and calm", "The noise was jarring and chaotic"),
            ("She has a cheerful personality", "She has a gloomy personality"),
            ("The house is cozy and warm", "The house is cold and damp"),
            ("He made a wise decision", "He made a foolish decision"),
            ("The painting is magnificent", "The painting is hideous"),
            ("She gave a generous donation", "She made a greedy demand"),
            ("The park is peaceful and serene", "The park is noisy and chaotic"),
            ("He has a confident attitude", "He has a fearful attitude"),
            ("The flowers smell wonderful", "The garbage smells terrible"),
            ("She wrote an inspiring speech", "She wrote a depressing speech"),
            ("The room is clean and tidy", "The room is filthy and messy"),
            ("He is honest and trustworthy", "He is deceitful and untrustworthy"),
            ("The meal was satisfying and filling", "The meal was unsatisfying and meager"),
            ("She has a positive outlook", "She has a negative outlook"),
            ("The story has a happy ending", "The story has a tragic ending"),
            ("He is friendly and outgoing", "He is hostile and withdrawn"),
            ("The gift was thoughtful and meaningful", "The gift was thoughtless and meaningless"),
            ("She felt grateful and blessed", "She felt resentful and cursed"),
            ("The journey was pleasant and smooth", "The journey was unpleasant and rough"),
        ],
    },
    'semantic_topic': {
        'type': 'SEM',
        'pairs': [
            ("The doctor treated the patient carefully", "The mechanic fixed the engine carefully"),
            ("The teacher explained the lesson clearly", "The chef prepared the dish clearly"),
            ("The pilot flew the plane safely", "The driver steered the car safely"),
            ("The lawyer argued the case persuasively", "The scientist presented the data persuasively"),
            ("The musician played the symphony beautifully", "The athlete performed the routine beautifully"),
            ("The architect designed the building creatively", "The programmer coded the software creatively"),
            ("The nurse cared for the sick gently", "The gardener tended the plants gently"),
            ("The judge decided the verdict fairly", "The referee called the game fairly"),
            ("The author wrote the novel brilliantly", "The painter created the artwork brilliantly"),
            ("The engineer built the bridge solidly", "The farmer grew the crops solidly"),
            ("The detective solved the mystery cleverly", "The student solved the equation cleverly"),
            ("The baker made the bread perfectly", "The carpenter made the furniture perfectly"),
            ("The singer performed the song gracefully", "The dancer performed the routine gracefully"),
            ("The fisherman caught the fish skillfully", "The hunter tracked the deer skillfully"),
            ("The librarian organized the books neatly", "The accountant organized the records neatly"),
            ("The therapist helped the client patiently", "The coach trained the athlete patiently"),
            ("The electrician wired the house properly", "The plumber piped the bathroom properly"),
            ("The photographer captured the moment artistically", "The journalist reported the event artistically"),
            ("The chef seasoned the soup perfectly", "The pharmacist mixed the medicine perfectly"),
            ("The tailor sewed the dress elegantly", "The sculptor shaped the clay elegantly"),
            ("The scientist discovered the element accidentally", "The explorer found the cave accidentally"),
            ("The conductor led the orchestra masterfully", "The general led the army masterfully"),
            ("The dentist cleaned the teeth thoroughly", "The cleaner washed the floor thoroughly"),
            ("The translator converted the text accurately", "The banker calculated the interest accurately"),
            ("The firefighter rescued the cat bravely", "The lifeguard saved the swimmer bravely"),
            ("The astronaut explored the space boldly", "The diver explored the ocean boldly"),
            ("The philosopher thought about existence deeply", "The poet thought about beauty deeply"),
            ("The banker managed the portfolio wisely", "The farmer managed the land wisely"),
            ("The surgeon operated on the heart precisely", "The jeweler crafted the ring precisely"),
            ("The journalist covered the story thoroughly", "The historian documented the era thoroughly"),
            ("The designer created the logo innovatively", "The inventor created the device innovatively"),
            ("The teacher graded the papers carefully", "The inspector examined the building carefully"),
            ("The coach motivated the team effectively", "The manager motivated the staff effectively"),
            ("The researcher analyzed the data rigorously", "The critic analyzed the film rigorously"),
            ("The waiter served the meal attentively", "The nurse served the medicine attentively"),
            ("The builder constructed the wall sturdily", "The weaver constructed the basket sturdily"),
            ("The priest blessed the congregation warmly", "The mayor addressed the city warmly"),
            ("The athlete trained the muscles intensely", "The scholar trained the mind intensely"),
            ("The florist arranged the roses beautifully", "The decorator arranged the room beautifully"),
            ("The guide showed the tourists around helpfully", "The consultant advised the clients helpfully"),
        ],
    },
    'voice': {
        'type': 'SEM',
        'pairs': [
            ("The cat chased the mouse", "The mouse was chased by the cat"),
            ("The dog bit the man", "The man was bitten by the dog"),
            ("The wind destroyed the house", "The house was destroyed by the wind"),
            ("She wrote the letter", "The letter was written by her"),
            ("They built the bridge", "The bridge was built by them"),
            ("He fixed the car", "The car was fixed by him"),
            ("The teacher praised the student", "The student was praised by the teacher"),
            ("The company launched the product", "The product was launched by the company"),
            ("She painted the portrait", "The portrait was painted by her"),
            ("They discovered the island", "The island was discovered by them"),
            ("He composed the symphony", "The symphony was composed by him"),
            ("The chef prepared the meal", "The meal was prepared by the chef"),
            ("She directed the movie", "The movie was directed by her"),
            ("They won the championship", "The championship was won by them"),
            ("He designed the building", "The building was designed by him"),
            ("The police arrested the thief", "The thief was arrested by the police"),
            ("She wrote the report", "The report was written by her"),
            ("They organized the festival", "The festival was organized by them"),
            ("He invented the machine", "The machine was invented by him"),
            ("The author published the book", "The book was published by the author"),
            ("She cleaned the room", "The room was cleaned by her"),
            ("They delivered the package", "The package was delivered by them"),
            ("He repaired the roof", "The roof was repaired by him"),
            ("The artist created the sculpture", "The sculpture was created by the artist"),
            ("She baked the cake", "The cake was baked by her"),
            ("They planted the garden", "The garden was planted by them"),
            ("He recorded the song", "The song was recorded by him"),
            ("The scientist conducted the experiment", "The experiment was conducted by the scientist"),
            ("She translated the document", "The document was translated by her"),
            ("They manufactured the car", "The car was manufactured by them"),
            ("He programmed the software", "The software was programmed by him"),
            ("The judge delivered the verdict", "The verdict was delivered by the judge"),
            ("She taught the class", "The class was taught by her"),
            ("They managed the project", "The project was managed by them"),
            ("He analyzed the data", "The data was analyzed by him"),
            ("The committee approved the plan", "The plan was approved by the committee"),
            ("She photographed the wedding", "The wedding was photographed by her"),
            ("They decorated the hall", "The hall was decorated by them"),
            ("He edited the video", "The video was edited by him"),
            ("The team completed the mission", "The mission was completed by the team"),
        ],
    },
    'formality': {
        'type': 'SEM',
        'pairs': [
            ("Hey, what's up?", "Greetings, how are you doing?"),
            ("Wanna grab some food?", "Would you like to have some dinner?"),
            ("Gimme that thing", "Could you please hand me that item?"),
            ("It's super cool!", "It is quite impressive!"),
            ("I'm gonna head out", "I shall take my leave"),
            ("That's awesome, dude!", "That is excellent, sir!"),
            ("No way!", "I find that difficult to believe!"),
            ("What's the deal with this?", "What is the situation regarding this matter?"),
            ("Let's bounce", "Let us depart"),
            ("Gotcha, makes sense", "I understand, that is logical"),
            ("Cool beans!", "That is satisfactory!"),
            ("So what?", "What is the relevance?"),
            ("I dunno", "I am uncertain"),
            ("See ya later!", "I look forward to our next meeting!"),
            ("That's dope!", "That is remarkable!"),
            ("My bad!", "I apologize for my error!"),
            ("Chill out!", "Please calm down!"),
            ("You rock!", "You are exceptional!"),
            ("Hang on a sec", "Please wait a moment"),
            ("That's legit!", "That is legitimate!"),
            ("I'm down for that", "I am amenable to that proposal"),
            ("Tell me about it!", "Please elaborate on that matter!"),
            ("Hit me up!", "Please contact me!"),
            ("It's a bummer", "It is unfortunate"),
            ("Keep it real!", "Maintain authenticity!"),
            ("I'm beat", "I am exhausted"),
            ("Right on!", "I concur!"),
            ("What a drag!", "How tedious!"),
            ("Cut it out!", "Please desist!"),
            ("Way to go!", "Congratulations on your achievement!"),
            ("Take it easy!", "Please relax!"),
            ("It's a no-brainer!", "It is self-evident!"),
            ("That's wild!", "That is extraordinary!"),
            ("Hold your horses!", "Please be patient!"),
            ("Piece of cake!", "That was effortless!"),
            ("Break a leg!", "I wish you the best of luck!"),
            ("Under the weather", "Feeling unwell"),
            ("On the same page", "In agreement"),
            ("Give it a shot", "Attempt the endeavor"),
            ("By the way", "Incidentally"),
        ],
    },
}

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=50)
    args = parser.parse_args()
    
    model_name = args.model
    n_pairs = min(args.n_pairs, 40)
    
    cfg = MODEL_CONFIGS[model_name]
    
    outdir = f'results/causal_fiber/{model_name}_ccxvii'
    os.makedirs(outdir, exist_ok=True)
    log = get_log_fn(outdir)
    
    log(f"Phase CCXVII: Per-feature Head贡献矩阵 + 全层PCA扫描")
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
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    d_model = model.config.hidden_size
    
    # 采样层: 每隔4层采样
    layer_indices = list(range(0, n_layers, 4))
    if (n_layers - 1) not in layer_indices:
        layer_indices.append(n_layers - 1)
    log(f"n_layers={n_layers}, n_heads={n_heads}, head_dim={head_dim}, d_model={d_model}")
    log(f"Sampled layers: {layer_indices}")
    
    feature_names = list(FEATURE_PAIRS.keys())
    
    # ============================================================
    # S1: 全层差分向量收集
    # ============================================================
    log(f"\n{'='*60}")
    log("S1: 全层差分向量收集 (逐对Hook方式, 避免OOM)")
    log(f"{'='*60}")
    
    all_deltas = {feat: {l: [] for l in layer_indices} for feat in feature_names}
    # Head-level差分(只收集3关键层)
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
    
    # 一次性注册所有hooks (高效方式)
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
    
    # 移除hooks
    for h in handles:
        h.remove()
    
    # ============================================================
    # S2: 全层PCA扫描
    # ============================================================
    log(f"\n{'='*60}")
    log("S2: 全层PCA扫描")
    log(f"{'='*60}")
    
    from sklearn.decomposition import TruncatedSVD
    
    layer_pca = {}
    
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
            'syn_mean_norm': float(np.mean(syn_norms)) if syn_norms else 0,
            'sem_mean_norm': float(np.mean(sem_norms)) if sem_norms else 0,
            'n_syn': len(syn_deltas),
            'n_sem': len(sem_deltas),
            'feat_pc1': feat_pc1_centroids,
        }
        
        log(f"  L{l}: PC1={var_top5[0]:.4f}, cum5={cumvar_top5[-1]:.4f}, "
            f"cos={centroid_cos:.4f}, norm(S/E)={np.mean(syn_norms):.0f}/{np.mean(sem_norms):.0f}")
    
    # 打印全层演变
    log("\n--- Layer-by-Layer Evolution ---")
    pc1_evolution = [layer_pca[l]['pc1'] for l in layer_indices if l in layer_pca]
    cos_evolution = [layer_pca[l]['centroid_cos'] for l in layer_indices if l in layer_pca]
    norm_syn_evolution = [layer_pca[l]['syn_mean_norm'] for l in layer_indices if l in layer_pca]
    norm_sem_evolution = [layer_pca[l]['sem_mean_norm'] for l in layer_indices if l in layer_pca]
    
    # 找信息瓶颈 (PC1峰值)
    if pc1_evolution:
        max_pc1_idx = np.argmax(pc1_evolution)
        bottleneck_layer = layer_indices[max_pc1_idx]
        log(f"  Information bottleneck: L{bottleneck_layer} (PC1={pc1_evolution[max_pc1_idx]:.4f})")
    
    # 找最小centroid_cos (最正交的层)
    if cos_evolution:
        min_cos_idx = np.argmin(cos_evolution)
        most_ortho_layer = layer_indices[min_cos_idx]
        log(f"  Most orthogonal layer: L{most_ortho_layer} (cos={cos_evolution[min_cos_idx]:.4f})")
    
    # ============================================================
    # S3: Per-feature Head贡献矩阵
    # ============================================================
    log(f"\n{'='*60}")
    log("S3: Per-feature Head贡献矩阵")
    log(f"{'='*60}")
    
    head_feature_matrix = {}  # {layer: matrix(n_heads, n_features)}
    
    for l in key_layers:
        matrix = np.zeros((n_heads, len(feature_names)))
        
        for f_idx, feat in enumerate(feature_names):
            for h in range(n_heads):
                hds = head_deltas[feat][l][h]
                if hds:
                    matrix[h, f_idx] = np.mean([np.linalg.norm(d) for d in hds])
        
        head_feature_matrix[l] = matrix
        
        # 打印每个Head的top-3 features
        log(f"  L{l}: Head-Feature contribution matrix ({n_heads}x{len(feature_names)})")
        
        # 打印每个Feature的top-3 Heads
        log(f"  Per-feature top Heads:")
        for f_idx, feat in enumerate(feature_names):
            col = matrix[:, f_idx]
            top3 = np.argsort(-col)[:3]
            log(f"    {feat}: Top Heads={top3.tolist()}, norms={[f'{col[h]:.1f}' for h in top3]}")
        
        # 打印每个Head的top-3 Features
        log(f"  Per-head top Features:")
        for h in range(min(10, n_heads)):  # 只打印前10个head
            row = matrix[h]
            top3 = np.argsort(-row)[:3]
            log(f"    H{h}: Top Features={[feature_names[i] for i in top3]}, norms={[f'{row[i]:.1f}' for i in top3]}")
    
    # ============================================================
    # S4: Head聚类分析
    # ============================================================
    log(f"\n{'='*60}")
    log("S4: Head聚类分析")
    log(f"{'='*60}")
    
    for l in key_layers:
        matrix = head_feature_matrix[l]
        
        # 归一化: 每行归一化为概率分布
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix_norm = matrix / row_sums
        
        # 计算Head间的距离
        if n_heads > 2:
            dist_matrix = pdist(matrix_norm, metric='correlation')
            Z = linkage(dist_matrix, method='ward')
            
            # 尝试2-5个聚类
            for n_clusters in [2, 3, 4]:
                clusters = fcluster(Z, n_clusters, criterion='maxclust')
                cluster_sizes = [np.sum(clusters == c) for c in range(1, n_clusters + 1)]
                
                # 检查聚类是否按语法/语义分
                # 计算每个聚类的语法/语义偏好
                syn_cols = [i for i, f in enumerate(feature_names) if FEATURE_PAIRS[f]['type'] == 'SYN']
                sem_cols = [i for i, f in enumerate(feature_names) if FEATURE_PAIRS[f]['type'] == 'SEM']
                
                cluster_syn_prefs = []
                for c in range(1, n_clusters + 1):
                    mask = clusters == c
                    cluster_matrix = matrix_norm[mask]
                    syn_pref = cluster_matrix[:, syn_cols].mean() / (cluster_matrix[:, sem_cols].mean() + 1e-8)
                    cluster_syn_prefs.append(syn_pref)
                
                # 判断是否有语法/语义专化
                max_ratio = max(cluster_syn_prefs) / (min(cluster_syn_prefs) + 1e-8)
                
                log(f"  L{l} clusters={n_clusters}: sizes={cluster_sizes}, "
                    f"syn_prefs={[f'{r:.2f}' for r in cluster_syn_prefs]}, max_ratio={max_ratio:.2f}")
            
            # 找最优聚类(2个聚类+最大ratio)
            clusters_2 = fcluster(Z, 2, criterion='maxclust')
            mask1 = clusters_2 == 1
            mask2 = clusters_2 == 2
            c1_syn = matrix_norm[mask1][:, syn_cols].mean() / (matrix_norm[mask1][:, sem_cols].mean() + 1e-8)
            c2_syn = matrix_norm[mask2][:, syn_cols].mean() / (matrix_norm[mask2][:, sem_cols].mean() + 1e-8)
            
            if c1_syn > c2_syn:
                syn_cluster = 1
                sem_cluster = 2
            else:
                syn_cluster = 2
                sem_cluster = 1
            
            syn_heads = np.where(clusters_2 == syn_cluster)[0].tolist()
            sem_heads = np.where(clusters_2 == sem_cluster)[0].tolist()
            
            log(f"  L{l} Best 2-cluster: SYN heads={syn_heads}, SEM heads={sem_heads}")
            
            # 检查聚类质量
            # 用silhouette-like度量: 类内vs类间距离
            if len(syn_heads) > 0 and len(sem_heads) > 0:
                intra_syn = np.mean(pdist(matrix_norm[mask1], 'correlation')) if len(syn_heads) > 1 else 0
                intra_sem = np.mean(pdist(matrix_norm[mask2], 'correlation')) if len(sem_heads) > 1 else 0
                inter = np.mean(pdist(np.vstack([matrix_norm[mask1], matrix_norm[mask2]]), 'correlation'))
                
                log(f"  L{l} Cluster quality: intra_syn={intra_syn:.3f}, intra_sem={intra_sem:.3f}, inter={inter:.3f}")
    
    # ============================================================
    # S5: 因果原子词典 + 统计总结
    # ============================================================
    log(f"\n{'='*60}")
    log("S5: 因果原子词典 + 统计总结")
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
        growth_data[feat] = growth
    
    syn_growths = [growth_data[f] for f in feature_names if FEATURE_PAIRS[f]['type'] == 'SYN']
    sem_growths = [growth_data[f] for f in feature_names if FEATURE_PAIRS[f]['type'] == 'SEM']
    
    if syn_growths and sem_growths:
        stat, p_growth = stats.mannwhitneyu(syn_growths, sem_growths, alternative='greater')
        log(f"  Growth: syn_mean={np.mean(syn_growths):.2f}x vs sem_mean={np.mean(sem_growths):.2f}x, p={p_growth:.4f}")
    
    # Per-feature separability ranking (用最后一层)
    last_l = n_layers - 1
    if last_l in layer_pca:
        feat_pc1 = layer_pca[last_l]['feat_pc1']
        sep_ranking = sorted(feat_pc1.items(), key=lambda x: abs(x[1]), reverse=True)
        log(f"\n  Causal Atom Dictionary (L{last_l}, ranked by |PC1 centroid|):")
        for feat, val in sep_ranking:
            log(f"    {feat} [{FEATURE_PAIRS[feat]['type']}]: PC1 centroid={val:.3f}, growth={growth_data.get(feat, 0):.1f}x")
    
    # ============================================================
    # 保存完整结果
    # ============================================================
    log(f"\n{'='*60}")
    log(f"FINAL: {model_name}")
    log(f"{'='*60}")
    
    # Delta Norms
    log("\n--- Delta Norms (sampled layers) ---")
    for feat in feature_names:
        norms = []
        for l in [0, n_layers//2, n_layers-1]:
            if l in layer_pca:
                if FEATURE_PAIRS[feat]['type'] == 'SYN':
                    norms.append(f"{layer_pca[l]['syn_mean_norm']:.0f}")
                else:
                    norms.append(f"{layer_pca[l]['sem_mean_norm']:.0f}")
            else:
                norms.append("N/A")
        log(f"  {feat} [{FEATURE_PAIRS[feat]['type']}]: {', '.join(norms)}")
    
    # PCA Evolution
    log("\n--- PCA Evolution ---")
    for l in layer_indices:
        if l in layer_pca:
            p = layer_pca[l]
            log(f"  L{l}: PC1={p['pc1']:.4f}, cum5={p['cum5']:.4f}, cos={p['centroid_cos']:.4f}")
    
    # Growth
    log("\n--- Fiber Growth ---")
    for feat in feature_names:
        log(f"  {feat} [{FEATURE_PAIRS[feat]['type']}]: growth={growth_data[feat]:.2f}x")
    
    # Head Feature Matrix (last layer)
    log(f"\n--- Head Feature Matrix (L{n_layers-1}) ---")
    if (n_layers - 1) in head_feature_matrix:
        mat = head_feature_matrix[n_layers - 1]
        # 打印top-5 heads per feature
        for f_idx, feat in enumerate(feature_names):
            col = mat[:, f_idx]
            top5 = np.argsort(-col)[:5]
            log(f"  {feat}: Top5 Heads={top5.tolist()}, weights={[f'{col[h]:.1f}' for h in top5]}")
    
    # 保存JSON
    save_data = {
        'model': cfg['name'],
        'model_id': model_name,
        'phase': 'CCXVII',
        'n_pairs': n_pairs,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'd_model': d_model,
        'layer_indices': layer_indices,
        'key_layers': key_layers,
        'feature_names': feature_names,
        'layer_pca': {str(k): v for k, v in layer_pca.items()},
        'growth': growth_data,
        'syn_growth_mean': float(np.mean(syn_growths)) if syn_growths else 0,
        'sem_growth_mean': float(np.mean(sem_growths)) if sem_growths else 0,
        'growth_p': float(p_growth) if syn_growths and sem_growths else 1.0,
    }
    
    with open(f'{outdir}/full_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    # 保存Head-Feature矩阵
    for l in key_layers:
        if l in head_feature_matrix:
            np.save(f'{outdir}/head_feature_matrix_L{l}.npy', head_feature_matrix[l])
    
    log(f"\nDONE! Saved to {outdir}")


def get_log_fn(outdir):
    def log(msg):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(f'{outdir}/run.log', 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    return log


if __name__ == '__main__':
    main()
