"""
Phase CCXII: 6语法+6语义 因果对偶6vs6验证 (修正版)
核心改进:
  1. 修正person→语法组 (I→She涉及动词变形am→is)
  2. 新增2种纯语义特征: definiteness(定指/不定指), info_structure(焦点/预设)
  3. 正确分组: 6语法+6语义, 统计检验6vs6
  4. n=180对, 6层采样
  5. 增加Permutation test (更稳健)
  
语法特征 (涉及词形变化/morphological marking):
  tense, polarity, number, negation, question, person

语义特征 (不涉及词形变化, 只改变指称/视角/信息结构):
  sentiment, semantic_topic, voice, formality, definiteness, info_structure
"""
import os, sys, gc, time, json, argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'w', buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

PATHS = {
    "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
    "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

# ===== 语法特征 (词形变化) =====
TENSE = [
    ("The cat sat quietly on the mat", "The cat sits quietly on the mat"),
    ("She walked to the store yesterday", "She walks to the store today"),
    ("He played guitar every evening", "He plays guitar every evening"),
    ("They worked on the project", "They work on the project"),
    ("The dog ran across the field", "The dog runs across the field"),
    ("She wrote a long letter", "She writes a long letter"),
    ("He drove the car fast", "He drives the car fast"),
    ("They built a new house", "They build a new house"),
    ("The bird flew over the lake", "The bird flies over the lake"),
    ("She cooked dinner for us", "She cooks dinner for us"),
    ("He read many books last year", "He reads many books this year"),
    ("They sang beautiful songs", "They sing beautiful songs"),
    ("The train arrived late", "The train arrives late"),
    ("She taught the children math", "She teaches the children math"),
    ("He caught the ball easily", "He catches the ball easily"),
    ("They grew vegetables in spring", "They grow vegetables in spring"),
    ("The water froze overnight", "The water freezes overnight"),
    ("She drew a colorful picture", "She draws a colorful picture"),
    ("He held the baby gently", "He holds the baby gently"),
    ("They knew the answer quickly", "They know the answer quickly"),
    ("The sun rose early this morning", "The sun rises early every morning"),
    ("She chose the blue dress", "She chooses the blue dress"),
    ("He broke the window accidentally", "He breaks the window accidentally"),
    ("They spoke about the problem", "They speak about the problem"),
    ("The wind blew strongly today", "The wind blows strongly today"),
    ("I thought about the plan carefully", "I think about the plan carefully"),
    ("We found the solution quickly", "We find the solution quickly"),
    ("She kept the secret safe", "She keeps the secret safe"),
    ("He brought the documents yesterday", "He brings the documents today"),
    ("They fought for their rights", "They fight for their rights"),
    ("She spent the whole day reading", "She spends the whole day reading"),
    ("He lost his keys this morning", "He loses his keys often"),
    ("The children slept peacefully", "The children sleep peacefully"),
    ("She felt the cold wind blowing", "She feels the cold wind blowing"),
    ("He understood the instructions", "He understands the instructions"),
    ("They won the championship game", "They win the championship game"),
    ("The plant grew very quickly", "The plant grows very quickly"),
    ("She sent the package early", "She sends the package early"),
    ("He wore his favorite shirt", "He wears his favorite shirt"),
    ("They drank the cold water", "They drink the cold water"),
    ("The bell rang three times", "The bell rings three times"),
    ("She shut the door quietly", "She shuts the door quietly"),
    ("He fed the hungry animals", "He feeds the hungry animals"),
    ("They met at the restaurant", "They meet at the restaurant"),
    ("The snow fell all night long", "The snow falls all night long"),
    ("She bent the metal pipe", "She bends the metal pipe"),
    ("He lit the candle carefully", "He lights the candle carefully"),
    ("They led the team forward", "They lead the team forward"),
    ("The boat sank near the shore", "The boat sinks near the shore"),
    ("She hung the picture straight", "She hangs the picture straight"),
]

POLARITY = [
    ("The cat is happy today", "The cat is not happy today"),
    ("She likes the new movie", "She does not like the new movie"),
    ("He can solve the problem", "He cannot solve the problem"),
    ("They will attend the meeting", "They will not attend the meeting"),
    ("The dog is friendly to strangers", "The dog is not friendly to strangers"),
    ("She has finished the work", "She has not finished the work"),
    ("He was available yesterday", "He was not available yesterday"),
    ("They are coming to the party", "They are not coming to the party"),
    ("The food was delicious", "The food was not delicious"),
    ("She does know the answer", "She does not know the answer"),
    ("He should go to school", "He should not go to school"),
    ("They would help with chores", "They would not help with chores"),
    ("The test was easy for me", "The test was not easy for me"),
    ("She could swim very well", "She could not swim very well"),
    ("He must leave right now", "He must not leave right now"),
    ("They had enough money saved", "They had not enough money saved"),
    ("The door was open this morning", "The door was not open this morning"),
    ("She did enjoy the concert", "She did not enjoy the concert"),
    ("He has been working hard", "He has not been working hard"),
    ("They were ready on time", "They were not ready on time"),
    ("The plan was successful", "The plan was not successful"),
    ("She will accept the offer", "She will not accept the offer"),
    ("He can drive the truck", "He cannot drive the truck"),
    ("They did win the game", "They did not win the game"),
    ("The idea was brilliant", "The idea was not brilliant"),
    ("I believe this is correct", "I do not believe this is correct"),
    ("We trust the new system", "We do not trust the new system"),
    ("She understands the material", "She does not understand the material"),
    ("He remembers the details", "He does not remember the details"),
    ("They appreciate the effort", "They do not appreciate the effort"),
    ("The machine is working well", "The machine is not working well"),
    ("She enjoys cooking meals", "She does not enjoy cooking meals"),
    ("He owns a small boat", "He does not own a small boat"),
    ("They play tennis regularly", "They do not play tennis regularly"),
    ("We support the proposal", "We do not support the proposal"),
    ("The method works perfectly", "The method does not work perfectly"),
    ("She cares about the issue", "She does not care about the issue"),
    ("He agrees with the decision", "He does not agree with the decision"),
    ("They deserve the recognition", "They do not deserve the recognition"),
    ("The book contains useful info", "The book does not contain useful info"),
    ("She speaks French fluently", "She does not speak French fluently"),
    ("He knows the right answer", "He does not know the right answer"),
    ("They want to join the club", "They do not want to join the club"),
    ("I like the new design", "I do not like the new design"),
    ("We need more time to finish", "We do not need more time to finish"),
    ("The project seems feasible", "The project does not seem feasible"),
    ("She follows the rules carefully", "She does not follow the rules carefully"),
    ("He takes the bus every day", "He does not take the bus every day"),
    ("They watch the news regularly", "They do not watch the news regularly"),
    ("The food tastes good today", "The food does not taste good today"),
]

NUMBER = [
    ("The cat is sleeping now", "The cats are sleeping now"),
    ("A book was on the table", "The books were on the table"),
    ("The child plays outside", "The children play outside"),
    ("A dog runs in the park", "The dogs run in the park"),
    ("The woman walks to work", "The women walk to work"),
    ("A man drives the bus", "The men drive the bus"),
    ("The bird flies high above", "The birds fly high above"),
    ("A student reads the text", "The students read the text"),
    ("The teacher explains clearly", "The teachers explain clearly"),
    ("A flower grows in spring", "The flowers grow in spring"),
    ("The tree stands very tall", "The trees stand very tall"),
    ("A car drives down the road", "The cars drive down the road"),
    ("The house looks beautiful", "The houses look beautiful"),
    ("A river flows through town", "The rivers flow through town"),
    ("The mountain rises above clouds", "The mountains rise above clouds"),
    ("A star shines in the sky", "The stars shine in the sky"),
    ("The country borders two seas", "The countries border two seas"),
    ("A city grows very fast", "The cities grow very fast"),
    ("The church stands on the hill", "The churches stand on the hill"),
    ("A bridge crosses the river", "The bridges cross the river"),
    ("The knife cuts the bread", "The knives cut the bread"),
    ("A wolf howls at night", "The wolves howl at night"),
    ("The leaf falls from the tree", "The leaves fall from the tree"),
    ("A shelf holds many items", "The shelves hold many items"),
    ("The wolf hunts in the pack", "The wolves hunt in the pack"),
    ("A calf drinks the milk", "The calves drink the milk"),
    ("The half was given to each", "The halves were given to each"),
    ("A loaf sits on the counter", "The loaves sit on the counter"),
    ("The thief stole the jewel", "The thieves stole the jewel"),
    ("A wife prepares the meal", "The wives prepare the meal"),
    ("The life was saved by doctors", "The lives were saved by doctors"),
    ("A mouse ran under the couch", "The mice ran under the couch"),
    ("The goose swam in the pond", "The geese swam in the pond"),
    ("A foot stepped on the grass", "The feet stepped on the grass"),
    ("The tooth was cleaned carefully", "The teeth were cleaned carefully"),
    ("A person entered the room", "The people entered the room"),
    ("The boy plays the game", "The boys play the game"),
    ("A girl sings the song", "The girls sing the song"),
    ("The baby cries loudly", "The babies cry loudly"),
    ("A lady walks gracefully", "The ladies walk gracefully"),
    ("The hero saves the day", "The heroes save the day"),
    ("The potato grows underground", "The potatoes grow underground"),
    ("A tomato ripens on the vine", "The tomatoes ripen on the vine"),
    ("The echo bounced off the wall", "The echoes bounced off the wall"),
    ("A volcano erupted violently", "The volcanoes erupted violently"),
    ("The studio produced the film", "The studios produced the film"),
    ("A piano plays the melody", "The pianos play the melody"),
    ("The radio broadcast the news", "The radios broadcast the news"),
    ("A video was shared online", "The videos were shared online"),
    ("The roof covers the building", "The roofs cover the buildings"),
]

NEGATION = [
    ("She is going to the store", "She is not going to the store"),
    ("He was working late today", "He was not working late today"),
    ("They have seen the movie", "They have not seen the movie"),
    ("The cat will eat the food", "The cat will not eat the food"),
    ("I could hear the music", "I could not hear the music"),
    ("She did finish the report", "She did not finish the report"),
    ("He would agree to help", "He would not agree to help"),
    ("We should try the method", "We should not try the method"),
    ("The door was locked tight", "The door was not locked tight"),
    ("She had written the essay", "She had not written the essay"),
    ("He can speak the language", "He cannot speak the language"),
    ("They were watching the show", "They were not watching the show"),
    ("I am enjoying the book", "I am not enjoying the book"),
    ("The kids are playing outside", "The kids are not playing outside"),
    ("She has visited Paris", "She has not visited Paris"),
    ("He does understand the rules", "He does not understand the rules"),
    ("They might come tomorrow", "They might not come tomorrow"),
    ("We need to leave early", "We do not need to leave early"),
    ("The system is working well", "The system is not working well"),
    ("She knows the correct answer", "She does not know the correct answer"),
    ("He believes in the cause", "He does not believe in the cause"),
    ("They want to join the club", "They do not want to join the club"),
    ("I like the new design", "I do not like the new design"),
    ("The plan makes sense now", "The plan does not make sense now"),
    ("She enjoys cooking meals", "She does not enjoy cooking meals"),
    ("He owns a small boat", "He does not own a small boat"),
    ("They play tennis regularly", "They do not play tennis regularly"),
    ("We support the proposal", "We do not support the proposal"),
    ("The method works perfectly", "The method does not work perfectly"),
    ("She remembers the details", "She does not remember the details"),
    ("The engine is running smoothly", "The engine is not running smoothly"),
    ("He follows the recipe exactly", "He does not follow the recipe exactly"),
    ("They attend every meeting", "They do not attend every meeting"),
    ("I understand the question", "I do not understand the question"),
    ("We recommend this product", "We do not recommend this product"),
    ("The device connects easily", "The device does not connect easily"),
    ("She practices the piano daily", "She does not practice the piano daily"),
    ("He watches the news every night", "He does not watch the news every night"),
    ("They recycle their waste", "They do not recycle their waste"),
    ("The software updates automatically", "The software does not update automatically"),
    ("I recognize that person", "I do not recognize that person"),
    ("We celebrate the holiday", "We do not celebrate the holiday"),
    ("The paint covers the wall", "The paint does not cover the wall"),
    ("She sings in the choir", "She does not sing in the choir"),
    ("He drives carefully always", "He does not drive carefully always"),
    ("They sell organic produce", "They do not sell organic produce"),
    ("I drink coffee every morning", "I do not drink coffee every morning"),
    ("We open the shop early", "We do not open the shop early"),
    ("The water flows downhill", "The water does not flow downhill"),
    ("She reads before bedtime", "She does not read before bedtime"),
]

QUESTION = [
    ("The cat is sleeping now", "Is the cat sleeping now"),
    ("She walked to the store", "Did she walk to the store"),
    ("He can solve the problem", "Can he solve the problem"),
    ("They will attend the meeting", "Will they attend the meeting"),
    ("The dog is friendly to strangers", "Is the dog friendly to strangers"),
    ("She has finished the work", "Has she finished the work"),
    ("He was available yesterday", "Was he available yesterday"),
    ("They are coming to the party", "Are they coming to the party"),
    ("The food was delicious", "Was the food delicious"),
    ("She should go to school", "Should she go to school"),
    ("He could swim very well", "Could he swim very well"),
    ("They would help with chores", "Would they help with chores"),
    ("The test was easy for me", "Was the test easy for me"),
    ("He must leave right now", "Must he leave right now"),
    ("The door was open this morning", "Was the door open this morning"),
    ("He has been working hard", "Has he been working hard"),
    ("They were ready on time", "Were they ready on time"),
    ("The plan was successful", "Was the plan successful"),
    ("She will accept the offer", "Will she accept the offer"),
    ("He can drive the truck", "Can he drive the truck"),
    ("The children are playing outside", "Are the children playing outside"),
    ("She knows the correct answer", "Does she know the correct answer"),
    ("They enjoy the new movie", "Do they enjoy the new movie"),
    ("He likes the restaurant", "Does he like the restaurant"),
    ("The system works efficiently", "Does the system work efficiently"),
    ("We need more time to finish", "Do we need more time to finish"),
    ("She wants to learn the skill", "Does she want to learn the skill"),
    ("They believe in the mission", "Do they believe in the mission"),
    ("The project requires funding", "Does the project require funding"),
    ("He plays guitar every day", "Does he play guitar every day"),
    ("The manager approved the request", "Did the manager approve the request"),
    ("She speaks three languages", "Does she speak three languages"),
    ("They completed the assignment", "Did they complete the assignment"),
    ("He understands the problem", "Does he understand the problem"),
    ("The movie starts at eight", "Does the movie start at eight"),
    ("We should call the doctor", "Should we call the doctor"),
    ("She has read the report", "Has she read the report"),
    ("They found the missing keys", "Did they find the missing keys"),
    ("He wrote the final chapter", "Did he write the final chapter"),
    ("The machine needs repairs", "Does the machine need repairs"),
    ("She teaches the advanced class", "Does she teach the advanced class"),
    ("They bought the new equipment", "Did they buy the new equipment"),
    ("He heard the announcement", "Did he hear the announcement"),
    ("The bridge crosses the river", "Does the bridge cross the river"),
    ("We can finish by Friday", "Can we finish by Friday"),
    ("She organized the event", "Did she organize the event"),
    ("They support the initiative", "Do they support the initiative"),
    ("He solved the complex puzzle", "Did he solve the complex puzzle"),
    ("The train arrives on time", "Does the train arrive on time"),
    ("She recommended the book", "Did she recommend the book"),
]

PERSON = [
    ("I went to the store yesterday", "She went to the store yesterday"),
    ("I love reading science fiction", "He loves reading science fiction"),
    ("I have finished the project", "She has finished the project"),
    ("I am working on the design", "He is working on the design"),
    ("I need some help with this", "She needs some help with this"),
    ("I enjoyed the concert last night", "He enjoyed the concert last night"),
    ("I believe we should proceed", "She believes we should proceed"),
    ("I prefer tea over coffee", "He prefers tea over coffee"),
    ("I have been studying all day", "She has been studying all day"),
    ("I think the plan will work", "He thinks the plan will work"),
    ("I was at the office today", "She was at the office today"),
    ("I had seen the movie before", "He had seen the movie before"),
    ("I am going to call them", "She is going to call them"),
    ("I have two cats at home", "He has two cats at home"),
    ("I was running late this morning", "She was running late this morning"),
    ("I did the homework already", "He did the homework already"),
    ("I am feeling much better now", "She is feeling much better now"),
    ("I have been here before", "He has been here before"),
    ("I was hoping for good weather", "She was hoping for good weather"),
    ("I had finished the report", "He had finished the report"),
    ("I am looking for a solution", "She is looking for a solution"),
    ("I was surprised by the news", "He was surprised by the news"),
    ("I have read that book twice", "She has read that book twice"),
    ("I did enjoy the presentation", "He did enjoy the presentation"),
    ("I am leaving tomorrow morning", "She is leaving tomorrow morning"),
    ("I was working on the budget", "He was working on the budget"),
    ("I have visited that museum", "She has visited that museum"),
    ("I did remember the appointment", "He did remember the appointment"),
    ("I am planning the trip now", "She is planning the trip now"),
    ("I was watching the sunset", "He was watching the sunset"),
    ("I have completed the training", "She has completed the training"),
    ("I did understand the lecture", "He did understand the lecture"),
    ("I am trying the new method", "She is trying the new method"),
    ("I was thinking about you", "He was thinking about you"),
    ("I have seen the results", "She has seen the results"),
    ("I did finish the painting", "He did finish the painting"),
    ("I am having a great time", "She is having a great time"),
    ("I was cooking the dinner", "He was cooking the dinner"),
    ("I have written the essay", "She has written the essay"),
    ("I did pass the examination", "He did pass the examination"),
    ("I am living in the city", "She is living in the city"),
    ("I was driving the truck", "He was driving the truck"),
    ("I have solved the puzzle", "She has solved the puzzle"),
    ("I did learn the language", "He did learn the language"),
    ("I am teaching the class", "She is teaching the class"),
    ("I was fixing the computer", "He was fixing the computer"),
    ("I have found the error", "She has found the error"),
    ("I did enjoy the vacation", "He did enjoy the vacation"),
    ("I am reading the article", "She is reading the article"),
    ("I was walking the dog", "He was walking the dog"),
]

# ===== 语义特征 (不涉及词形变化) =====

SENTIMENT = [
    ("I love this amazing product", "I hate this terrible product"),
    ("The experience was wonderful", "The experience was awful"),
    ("She felt joyful and grateful", "She felt angry and resentful"),
    ("This is the best day ever", "This is the worst day ever"),
    ("The gift was thoughtful and kind", "The insult was cruel and mean"),
    ("We had a fantastic celebration", "We had a dreadful argument"),
    ("The performance was outstanding", "The performance was pathetic"),
    ("He spoke with warmth and care", "He spoke with coldness and spite"),
    ("The results exceeded expectations", "The results fell below expectations"),
    ("She smiled with genuine happiness", "She frowned with deep sorrow"),
    ("The music was uplifting and inspiring", "The noise was annoying and disturbing"),
    ("His advice was helpful and wise", "His advice was harmful and foolish"),
    ("The meal was a delightful treat", "The meal was a terrible disappointment"),
    ("They showed great courage and bravery", "They showed great fear and cowardice"),
    ("The story had a happy ending", "The story had a tragic ending"),
    ("I admire her dedication and effort", "I despise her laziness and neglect"),
    ("The garden was a paradise of beauty", "The dump was a wasteland of ugliness"),
    ("She embraced him with love", "She rejected him with hatred"),
    ("The news brought hope and relief", "The news brought despair and panic"),
    ("His actions showed generosity", "His actions showed greed"),
    ("The atmosphere was friendly and warm", "The atmosphere was hostile and cold"),
    ("She praised his excellent work", "She criticized his poor work"),
    ("The day was filled with laughter", "The day was filled with tears"),
    ("We celebrated our great victory", "We mourned our bitter defeat"),
    ("The future looks bright and promising", "The future looks bleak and hopeless"),
    ("He welcomed the guests warmly", "He rejected the guests coldly"),
    ("The team showed great teamwork", "The team showed great conflict"),
    ("She appreciated the beautiful gesture", "She resented the ugly gesture"),
    ("The weather was sunny and pleasant", "The weather was stormy and miserable"),
    ("He found peace in the garden", "He found chaos in the garden"),
    ("The book was fascinating and engaging", "The book was boring and tedious"),
    ("She expressed genuine gratitude", "She expressed deep resentment"),
    ("The movie was thrilling and exciting", "The movie was dull and uninspiring"),
    ("He delivered a brilliant performance", "He delivered a mediocre performance"),
    ("The community was united and strong", "The community was divided and weak"),
    ("She felt confident and empowered", "She felt insecure and powerless"),
    ("The garden bloomed with vibrant colors", "The garden withered in gray decay"),
    ("He offered sincere compliments", "He made harsh insults"),
    ("The child was joyful and playful", "The child was miserable and withdrawn"),
    ("She spoke with gentle kindness", "She spoke with bitter cruelty"),
    ("The gift brought great pleasure", "The injury brought great pain"),
    ("He showed deep compassion", "He showed cold indifference"),
    ("The evening was romantic and magical", "The evening was awkward and dreadful"),
    ("She radiated positive energy", "She radiated negative energy"),
    ("The speech was inspiring and moving", "The speech was demoralizing and offensive"),
    ("He felt proud of his achievement", "He felt ashamed of his failure"),
    ("The festival was lively and colorful", "The funeral was somber and gray"),
    ("She received warm congratulations", "She received harsh criticism"),
    ("The project was a great success", "The project was a total failure"),
    ("He experienced profound joy", "He experienced profound grief"),
]

SEMANTIC_TOPIC = [
    ("The doctor examined the patient carefully", "The chef prepared the meal carefully"),
    ("Scientists discovered a new particle", "Artists created a new painting"),
    ("The engine roared to life", "The orchestra played to life"),
    ("She solved the complex equation", "She wrote the complex poem"),
    ("The rocket launched into orbit", "The ship sailed into harbor"),
    ("He programmed the computer algorithm", "He composed the musical symphony"),
    ("The experiment yielded interesting data", "The novel yielded interesting insights"),
    ("The telescope observed distant galaxies", "The microscope observed tiny organisms"),
    ("She calculated the trajectory precisely", "She choreographed the dance precisely"),
    ("The reactor generated enormous power", "The storm generated enormous waves"),
    ("The mathematician proved the theorem", "The philosopher proved the argument"),
    ("The satellite orbited the planet", "The moon orbited the earth"),
    ("He analyzed the chemical compound", "He analyzed the literary work"),
    ("The bridge spanned the wide river", "The rainbow spanned the wide sky"),
    ("The laser cut through the metal", "The scissors cut through the fabric"),
    ("She researched the historical period", "She explored the geographical region"),
    ("The microscope revealed cell structures", "The telescope revealed star patterns"),
    ("The algorithm sorted the database", "The librarian sorted the collection"),
    ("The vaccine prevented the disease", "The umbrella prevented the soaking"),
    ("The code compiled without errors", "The speech delivered without pauses"),
    ("The surgeon performed the operation", "The conductor performed the symphony"),
    ("He measured the voltage precisely", "He measured the rhythm precisely"),
    ("The factory produced steel beams", "The bakery produced fresh bread"),
    ("She calibrated the microscope lens", "She tuned the piano keys"),
    ("The electron orbited the nucleus", "The dancer orbited the stage"),
    ("He synthesized the new compound", "He choreographed the new routine"),
    ("The database stored the records", "The library stored the manuscripts"),
    ("She optimized the neural network", "She perfected the painting technique"),
    ("The valve controlled the pressure", "The conductor controlled the tempo"),
    ("He diagnosed the mechanical fault", "He interpreted the literary symbol"),
    ("The circuit carried the current", "The melody carried the emotion"),
    ("She engineered the support structure", "She designed the artistic composition"),
    ("The catalyst accelerated the reaction", "The inspiration accelerated the creation"),
    ("He decoded the genetic sequence", "He decoded the poetic meaning"),
    ("The turbine generated electricity", "The orchestra generated emotion"),
    ("She verified the experimental results", "She validated the artistic vision"),
    ("The sensor detected the radiation", "The critic detected the symbolism"),
    ("He formulated the scientific theory", "He crafted the literary narrative"),
    ("The processor computed the solution", "The author composed the story"),
    ("She modeled the physical system", "She sketched the portrait"),
    ("The robot assembled the components", "The artist assembled the collage"),
    ("He translated the programming code", "He translated the foreign text"),
    ("The thermometer measured the heat", "The scale measured the harmony"),
    ("She debugged the software error", "She revised the draft essay"),
    ("The antenna received the signal", "The ear received the music"),
    ("He simulated the fluid dynamics", "He imagined the fictional world"),
    ("The experiment tested the hypothesis", "The audition tested the performance"),
    ("She published the research paper", "She exhibited the art installation"),
    ("The reactor sustained the chain reaction", "The choir sustained the harmony"),
    ("He specialized in quantum physics", "He specialized in classical music"),
]

VOICE = [
    ("The chef cooked the meal perfectly", "The meal was cooked perfectly by the chef"),
    ("She wrote the report yesterday", "The report was written yesterday by her"),
    ("They built the bridge last year", "The bridge was built last year by them"),
    ("He painted the entire house", "The entire house was painted by him"),
    ("The company launched the product", "The product was launched by the company"),
    ("She delivered the presentation", "The presentation was delivered by her"),
    ("They discovered the ancient ruins", "The ancient ruins were discovered by them"),
    ("He fixed the broken engine", "The broken engine was fixed by him"),
    ("The team won the championship", "The championship was won by the team"),
    ("She composed the beautiful melody", "The beautiful melody was composed by her"),
    ("They repaired the old roof", "The old roof was repaired by them"),
    ("He designed the new building", "The new building was designed by him"),
    ("The students completed the assignment", "The assignment was completed by the students"),
    ("She organized the conference", "The conference was organized by her"),
    ("They cleaned the entire house", "The entire house was cleaned by them"),
    ("He managed the project efficiently", "The project was managed efficiently by him"),
    ("The police arrested the suspect", "The suspect was arrested by the police"),
    ("She translated the document", "The document was translated by her"),
    ("They decorated the room beautifully", "The room was decorated beautifully by them"),
    ("He solved the complex puzzle", "The complex puzzle was solved by him"),
    ("The teacher explained the concept", "The concept was explained by the teacher"),
    ("She recorded the important data", "The important data was recorded by her"),
    ("They developed the software", "The software was developed by them"),
    ("He directed the new movie", "The new movie was directed by him"),
    ("The scientist discovered the element", "The element was discovered by the scientist"),
    ("She published the research findings", "The research findings were published by her"),
    ("They manufactured the product locally", "The product was manufactured locally by them"),
    ("He programmed the robot arm", "The robot arm was programmed by him"),
    ("The editor revised the manuscript", "The manuscript was revised by the editor"),
    ("She performed the experiment", "The experiment was performed by her"),
    ("The government approved the plan", "The plan was approved by the government"),
    ("He analyzed the sample carefully", "The sample was analyzed carefully by him"),
    ("They implemented the new system", "The new system was implemented by them"),
    ("She created the original design", "The original design was created by her"),
    ("The committee selected the winner", "The winner was selected by the committee"),
    ("He evaluated the proposal thoroughly", "The proposal was evaluated thoroughly by him"),
    ("They reviewed the application", "The application was reviewed by them"),
    ("She prepared the detailed report", "The detailed report was prepared by her"),
    ("The doctor treated the patient", "The patient was treated by the doctor"),
    ("He drove the delivery truck", "The delivery truck was driven by him"),
    ("The artist painted the portrait", "The portrait was painted by the artist"),
    ("She taught the beginner class", "The beginner class was taught by her"),
    ("They distributed the supplies", "The supplies were distributed by them"),
    ("He tested the new device", "The new device was tested by him"),
    ("The author wrote the novel", "The novel was written by the author"),
    ("She managed the busy restaurant", "The busy restaurant was managed by her"),
    ("They transported the heavy cargo", "The heavy cargo was transported by them"),
    ("He calibrated the instrument", "The instrument was calibrated by him"),
    ("The chef prepared the special dish", "The special dish was prepared by the chef"),
    ("She photographed the beautiful landscape", "The beautiful landscape was photographed by her"),
]

FORMALITY = [
    ("I would like to request your assistance", "I wanna ask for your help"),
    ("It is imperative that we proceed cautiously", "We gotta be careful going forward"),
    ("The individual demonstrated exceptional proficiency", "The guy was really good at it"),
    ("We regret to inform you of the cancellation", "Sorry but we have to cancel"),
    ("Please do not hesitate to contact us", "Just give us a call anytime"),
    ("The aforementioned proposal requires consideration", "That idea needs some thought"),
    ("It is advisable to consult a professional", "You should probably ask an expert"),
    ("Furthermore, the data indicates a correlation", "Also, the numbers seem connected"),
    ("In accordance with the established protocol", "Following the usual way of doing things"),
    ("The results were satisfactory and encouraging", "The results turned out pretty good"),
    ("I am writing to inquire about the position", "I am asking about the job"),
    ("Please accept our sincerest apologies", "We are really sorry about this"),
    ("The committee has reached a consensus", "The group agreed on something"),
    ("We sincerely appreciate your cooperation", "Thanks a lot for helping out"),
    ("The documentation must be submitted promptly", "The papers gotta be sent in fast"),
    ("It would be greatly appreciated if you could", "It would be awesome if you could"),
    ("The implementation was executed flawlessly", "They did a really great job"),
    ("We would like to extend our gratitude", "We wanna say thanks a lot"),
    ("The procedure has been finalized", "The process is all set now"),
    ("Please find enclosed the required documentation", "Here are the papers you need"),
    ("I respectfully disagree with that assessment", "I do not think that is right"),
    ("The endeavor proved to be successful", "The whole thing worked out well"),
    ("We are delighted to announce the partnership", "We are happy to say we are partnering"),
    ("The matter requires immediate attention", "This needs to be dealt with right away"),
    ("Kindly refrain from making excessive noise", "Please do not make too much noise"),
    ("The expenditure exceeded the allocated budget", "We spent more than we planned"),
    ("He was terminated due to insufficient performance", "He got fired for not doing well enough"),
    ("The accommodation met our expectations", "The place was just what we expected"),
    ("We encountered unforeseen complications", "We ran into some unexpected problems"),
    ("The presentation was informative and engaging", "The talk was really interesting"),
    ("I am unable to attend the scheduled meeting", "I cannot make it to the meeting"),
    ("The product is of superior quality", "The product is really well made"),
    ("We anticipate significant improvements", "We expect things to get a lot better"),
    ("The investigation revealed discrepancies", "The check found some things that did not add up"),
    ("Please ensure timely completion of the task", "Make sure you finish on time"),
    ("The organization is committed to excellence", "The company really wants to do great work"),
    ("We acknowledge your valid concerns", "We hear what you are saying"),
    ("The assessment indicated considerable progress", "The review showed good improvement"),
    ("I wish to express my sincere gratitude", "I really want to say thank you"),
    ("The circumstances necessitate a revision", "The situation means we need to change things"),
    ("He demonstrated remarkable competence", "He showed he really knew his stuff"),
    ("The facility is currently unavailable", "The place is not open right now"),
    ("We recommend exercising caution", "We think you should be careful"),
    ("The modifications yielded positive outcomes", "The changes made things better"),
    ("It is essential to maintain accuracy", "You really need to get it right"),
    ("The undertaking was completed ahead of schedule", "We finished earlier than planned"),
    ("Please provide the requested information", "Please give us the info we asked for"),
    ("The organization extended an invitation", "The group sent out an invite"),
    ("We sincerely regret any inconvenience caused", "We are really sorry for the trouble"),
    ("The discussion yielded productive insights", "The talk gave us some good ideas"),
]

# 新增: 定指/不定指 (纯语义, 无词形变化)
DEFINITENESS = [
    ("A cat sat on the mat quietly", "The cat sat on the mat quietly"),
    ("A student solved the problem", "The student solved the problem"),
    ("A doctor examined the patient", "The doctor examined the patient"),
    ("A book was on the shelf", "The book was on the shelf"),
    ("A car drove down the street", "The car drove down the street"),
    ("A teacher explained the lesson", "The teacher explained the lesson"),
    ("A scientist made the discovery", "The scientist made the discovery"),
    ("A musician played the symphony", "The musician played the symphony"),
    ("A child found the treasure", "The child found the treasure"),
    ("A woman entered the building", "The woman entered the building"),
    ("A dog chased the ball", "The dog chased the ball"),
    ("A bird sang the melody", "The bird sang the melody"),
    ("A farmer grew the crops", "The farmer grew the crops"),
    ("A pilot flew the airplane", "The pilot flew the airplane"),
    ("A chef prepared the dinner", "The chef prepared the dinner"),
    ("A nurse cared for the patient", "The nurse cared for the patient"),
    ("A painter created the artwork", "The painter created the artwork"),
    ("A writer published the novel", "The writer published the novel"),
    ("A driver delivered the package", "The driver delivered the package"),
    ("A singer performed the song", "The singer performed the song"),
    ("A builder constructed the house", "The builder constructed the house"),
    ("A fisherman caught the fish", "The fisherman caught the fish"),
    ("A baker made the bread", "The baker made the bread"),
    ("A soldier defended the fort", "The soldier defended the fort"),
    ("A guard watched the gate", "The guard watched the gate"),
    ("A student wrote the essay", "The student wrote the essay"),
    ("A lawyer argued the case", "The lawyer argued the case"),
    ("A manager approved the request", "The manager approved the request"),
    ("A player scored the goal", "The player scored the goal"),
    ("A worker fixed the machine", "The worker fixed the machine"),
    ("A citizen reported the crime", "The citizen reported the crime"),
    ("A visitor admired the garden", "The visitor admired the garden"),
    ("A guest enjoyed the meal", "The guest enjoyed the meal"),
    ("A tourist visited the museum", "The tourist visited the museum"),
    ("A patient followed the advice", "The patient followed the advice"),
    ("A customer bought the product", "The customer bought the product"),
    ("A neighbor heard the noise", "The neighbor heard the noise"),
    ("A friend sent the message", "The friend sent the message"),
    ("A colleague shared the idea", "The colleague shared the idea"),
    ("A volunteer helped the team", "The volunteer helped the team"),
    ("A passenger boarded the train", "The passenger boarded the train"),
    ("A reader finished the chapter", "The reader finished the chapter"),
    ("A viewer watched the program", "The viewer watched the program"),
    ("A listener heard the broadcast", "The listener heard the broadcast"),
    ("A participant joined the study", "The participant joined the study"),
    ("A member attended the meeting", "The member attended the meeting"),
    ("A candidate passed the test", "The candidate passed the test"),
    ("A competitor won the race", "The competitor won the race"),
    ("A spectator cheered the team", "The spectator cheered the team"),
    ("A witness described the event", "The witness described the event"),
]

# 新增: 信息结构-焦点/预设 (纯语义, 无词形变化)
# cleft句 vs 非cleft句, 改变信息焦点但不改变词形
INFO_STRUCTURE = [
    ("Mary broke the window yesterday", "It was Mary who broke the window yesterday"),
    ("John stole the painting last night", "It was John who stole the painting last night"),
    ("The storm destroyed the house", "It was the storm that destroyed the house"),
    ("She found the solution this morning", "It was she who found the solution this morning"),
    ("They canceled the event on Friday", "It was they who canceled the event on Friday"),
    ("The dog bit the neighbor last week", "It was the dog that bit the neighbor last week"),
    ("He broke the record at the meet", "It was he who broke the record at the meet"),
    ("The rain ruined the picnic today", "It was the rain that ruined the picnic today"),
    ("She won the award this year", "It was she who won the award this year"),
    ("They built the bridge last decade", "It was they who built the bridge last decade"),
    ("The fire damaged the building yesterday", "It was the fire that damaged the building yesterday"),
    ("He discovered the error last month", "It was he who discovered the error last month"),
    ("The virus caused the illness recently", "It was the virus that caused the illness recently"),
    ("She wrote the report on Monday", "It was she who wrote the report on Monday"),
    ("They approved the budget this morning", "It was they who approved the budget this morning"),
    ("The earthquake cracked the foundation", "It was the earthquake that cracked the foundation"),
    ("He fixed the problem yesterday", "It was he who fixed the problem yesterday"),
    ("The teacher explained the concept well", "It was the teacher who explained the concept well"),
    ("She organized the conference last spring", "It was she who organized the conference last spring"),
    ("They published the findings this week", "It was they who published the findings this week"),
    ("The wind knocked down the tree", "It was the wind that knocked down the tree"),
    ("He delivered the speech at noon", "It was he who delivered the speech at noon"),
    ("The committee made the decision", "It was the committee that made the decision"),
    ("She completed the project early", "It was she who completed the project early"),
    ("They designed the system last year", "It was they who designed the system last year"),
    ("The flood damaged the crops badly", "It was the flood that damaged the crops badly"),
    ("He solved the mystery last night", "It was he who solved the mystery last night"),
    ("The manager approved the proposal", "It was the manager who approved the proposal"),
    ("She painted the mural last summer", "It was she who painted the mural last summer"),
    ("They launched the product in March", "It was they who launched the product in March"),
    ("The heat melted the ice quickly", "It was the heat that melted the ice quickly"),
    ("He composed the music last winter", "It was he who composed the music last winter"),
    ("The team won the championship game", "It was the team that won the championship game"),
    ("She directed the film this season", "It was she who directed the film this season"),
    ("They renovated the kitchen recently", "It was they who renovated the kitchen recently"),
    ("The frost killed the plants overnight", "It was the frost that killed the plants overnight"),
    ("He programmed the app last month", "It was he who programmed the app last month"),
    ("The storm flooded the basement", "It was the storm that flooded the basement"),
    ("She edited the manuscript carefully", "It was she who edited the manuscript carefully"),
    ("They hosted the event on Saturday", "It was they who hosted the event on Saturday"),
    ("The noise woke the baby up", "It was the noise that woke the baby up"),
    ("He led the expedition last autumn", "It was he who led the expedition last autumn"),
    ("The cat knocked the vase over", "It was the cat that knocked the vase over"),
    ("She recorded the song yesterday", "It was she who recorded the song yesterday"),
    ("They cleaned the beach this morning", "It was they who cleaned the beach this morning"),
    ("The light attracted the insects", "It was the light that attracted the insects"),
    ("He repaired the clock last Tuesday", "It was he who repaired the clock last Tuesday"),
    ("The snow covered the road completely", "It was the snow that covered the road completely"),
    ("She trained the dog patiently", "It was she who trained the dog patiently"),
    ("They decorated the hall beautifully", "It was they who decorated the hall beautifully"),
]

PREFIXES = ["Actually, ", "In fact, ", "Indeed, ", "Clearly, ", "Surely, ",
            "Perhaps, ", "Maybe, ", "Certainly, ", "Obviously, ", "Naturally, "]
SUFFIXES = [" right now", " today", " this time", " as expected", " for sure"]

def gen_pairs(templates, n, seed=42):
    np.random.seed(seed)
    pairs = list(templates)
    while len(pairs) < n:
        idx = np.random.randint(len(templates))
        a, b = templates[idx]
        aug = np.random.randint(4)
        if aug == 0:
            p = PREFIXES[np.random.randint(len(PREFIXES))]
            pairs.append((p+a, p+b))
        elif aug == 1:
            s = SUFFIXES[np.random.randint(len(SUFFIXES))]
            pairs.append((a+s, b+s))
        elif aug == 2:
            p = PREFIXES[np.random.randint(len(PREFIXES))]
            s = SUFFIXES[np.random.randint(len(SUFFIXES))]
            pairs.append((p+a+s, p+b+s))
        else:
            pairs.append((a, b))
    return pairs[:n]


def fit_algebraic_model(layers, l2_values):
    x = np.array(layers, dtype=float)
    y = np.array(l2_values, dtype=float)
    results = {}
    for name, fit_fn in [
        ("linear", lambda: _polyfit(x, y, 1)),
        ("exponential", lambda: _exp_fit(x, y)),
        ("power_law", lambda: _power_fit(x, y)),
        ("logarithmic", lambda: _log_fit(x, y)),
    ]:
        try:
            results[name] = fit_fn()
        except:
            results[name] = {"r2": 0}
    return results

def _polyfit(x, y, deg):
    coeffs = np.polyfit(x, y, deg)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"a": float(coeffs[0]), "b": float(coeffs[1]), "r2": float(r2)}

def _exp_fit(x, y):
    y_pos = np.maximum(y, 1e-6)
    coeffs = np.polyfit(x, np.log(y_pos), 1)
    y_pred = np.exp(coeffs[1]) * np.exp(coeffs[0] * x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"a": float(np.exp(coeffs[1])), "b": float(coeffs[0]), "r2": float(r2)}

def _power_fit(x, y):
    x_pos = np.maximum(x, 1.0)
    y_pos = np.maximum(y, 1e-6)
    log_x, log_y = np.log(x_pos), np.log(y_pos)
    valid = np.isfinite(log_x) & np.isfinite(log_y)
    if valid.sum() < 2: return {"r2": 0}
    coeffs = np.polyfit(log_x[valid], log_y[valid], 1)
    y_pred = np.exp(coeffs[1]) * x_pos ** coeffs[0]
    ss_res = np.sum((y[valid] - y_pred[valid])**2)
    ss_tot = np.sum((y[valid] - np.mean(y[valid]))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"a": float(np.exp(coeffs[1])), "b": float(coeffs[0]), "r2": float(r2)}

def _log_fit(x, y):
    log_x = np.log(x + 1)
    valid = np.isfinite(log_x) & np.isfinite(y)
    if valid.sum() < 2: return {"r2": 0}
    coeffs = np.polyfit(log_x[valid], y[valid], 1)
    y_pred = coeffs[0] * log_x + coeffs[1]
    ss_res = np.sum((y[valid] - y_pred[valid])**2)
    ss_tot = np.sum((y[valid] - np.mean(y[valid]))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"a": float(coeffs[0]), "b": float(coeffs[1]), "r2": float(r2)}


def do_patching(model, tokenizer, device, layer_idx, a_text, b_text, component=None):
    try:
        src_ids = tokenizer(a_text, return_tensors='pt')['input_ids'].to(device)
        clean_ids = tokenizer(b_text, return_tensors='pt')['input_ids'].to(device)
        layer = model.model.layers[layer_idx]
        target = layer if component is None else (layer.self_attn if component == 'attn' else layer.mlp)

        src_val = {}
        def cap(mod, inp, out):
            if isinstance(out, tuple):
                src_val['h'] = out[0][0, -1, :].detach().clone()
            else:
                src_val['h'] = out[0, -1, :].detach().clone()

        h = target.register_forward_hook(cap)
        with torch.no_grad(): _ = model(src_ids)
        h.remove()
        if 'h' not in src_val: return None
        sv = src_val['h']

        with torch.no_grad():
            clean_logits = model(clean_ids).logits[0, -1].detach().cpu().float()

        def phook(mod, inp, out, _sv=sv):
            if isinstance(out, tuple):
                out[0][0, -1, :] = _sv.to(out[0].device)
            else:
                out[0, -1, :] = _sv.to(out.device)

        h = target.register_forward_hook(phook)
        with torch.no_grad():
            patched_logits = model(clean_ids).logits[0, -1].detach().cpu().float()
        h.remove()

        diff = patched_logits - clean_logits
        return torch.norm(diff).item()
    except:
        return None


def permutation_test(syn_vals, sem_vals, n_perm=10000):
    """Permutation test: more robust than Mann-Whitney for small samples"""
    combined = np.array(syn_vals + sem_vals)
    n_syn = len(syn_vals)
    observed_diff = np.mean(syn_vals) - np.mean(sem_vals)
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(combined)
        perm_diff = np.mean(perm[:n_syn]) - np.mean(perm[n_syn:])
        if perm_diff >= observed_diff:
            count += 1
    return count / n_perm


def save_incremental(out_dir, all_results):
    with open(out_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['deepseek7b', 'qwen3', 'glm4'])
    parser.add_argument('--n_pairs', type=int, default=180)
    args = parser.parse_args()

    out_dir = Path(f"results/causal_fiber/{args.model}_ccxii")
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(out_dir / 'run.log')

    path = PATHS[args.model]
    print(f"\n{'='*60}")
    print(f"Phase CCXII: {args.model} (n_pairs={args.n_pairs}, 6 syn + 6 sem)")
    print(f"Path: {path}")
    print(f"{'='*60}")

    print(f"[{time.strftime('%H:%M:%S')}] Loading model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model in ["glm4", "deepseek7b"]:
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
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    print(f"[{time.strftime('%H:%M:%S')}] Loaded in {time.time()-t0:.1f}s: n_layers={n_layers}, device={device}", flush=True)

    N = args.n_pairs
    resid_layers = [int(i * (n_layers - 1) / 5) for i in range(6)]
    print(f"Resid layers (6): {resid_layers}", flush=True)

    # 6种语法特征 (person修正到语法组!)
    syntactic_features = {
        'tense': gen_pairs(TENSE, N),
        'polarity': gen_pairs(POLARITY, N),
        'number': gen_pairs(NUMBER, N),
        'negation': gen_pairs(NEGATION, N),
        'question': gen_pairs(QUESTION, N),
        'person': gen_pairs(PERSON, N),  # I→She: am→is, have→has
    }

    # 6种语义特征 (新增definiteness和info_structure)
    semantic_features = {
        'sentiment': gen_pairs(SENTIMENT, N),
        'semantic_topic': gen_pairs(SEMANTIC_TOPIC, N),
        'voice': gen_pairs(VOICE, N),
        'formality': gen_pairs(FORMALITY, N),
        'definiteness': gen_pairs(DEFINITENESS, N),    # a→the: 纯指称变化
        'info_structure': gen_pairs(INFO_STRUCTURE, N), # 焦点/预设: 纯信息结构
    }

    all_features = {**syntactic_features, **semantic_features}
    all_results = {}

    # ===== S1: Residual全层扫描 =====
    print(f"\n{'='*60}\nS1: Residual全层扫描 (12 features x 6 layers x {N} pairs)\n{'='*60}", flush=True)
    resid_results = {}

    for layer_idx in resid_layers:
        layer_name = f'L{layer_idx}'
        resid_results[layer_name] = {}
        t_layer = time.time()

        for feat_name, pairs in all_features.items():
            n_test = min(len(pairs), N)
            l2s = []
            for i in range(n_test):
                l2 = do_patching(model, tokenizer, device, layer_idx, pairs[i][0], pairs[i][1])
                if l2 is not None:
                    l2s.append(l2)

            if l2s:
                resid_results[layer_name][feat_name] = {
                    'mean_l2': float(np.mean(l2s)),
                    'median_l2': float(np.median(l2s)),
                    'std_l2': float(np.std(l2s)),
                    'n': len(l2s),
                }

        elapsed = time.time() - t_layer
        parts = []
        for feat_name in list(all_features.keys()):
            if feat_name in resid_results[layer_name]:
                d = resid_results[layer_name][feat_name]
                parts.append(f"{feat_name}={d['median_l2']:.0f}(n={d['n']})")
        print(f"  {layer_name} [{elapsed:.0f}s]: {', '.join(parts)}", flush=True)

        all_results['resid'] = resid_results
        save_incremental(out_dir, all_results)

    # ===== S2: Attn vs MLP =====
    print(f"\n{'='*60}\nS2: Attn vs MLP\n{'='*60}", flush=True)
    comp_layers = [0, n_layers - 1]
    comp_results = {}
    # 测试关键特征: 2语法+4语义
    comp_features = ['tense', 'person', 'sentiment', 'voice', 'formality', 'definiteness']
    for component in ['attn', 'mlp']:
        comp_results[component] = {}
        for layer_idx in comp_layers:
            ln = f'L{layer_idx}'
            comp_results[component][ln] = {}
            for feat_name in comp_features:
                pairs = all_features[feat_name]
                n_test = min(len(pairs), 80)
                l2s = []
                for i in range(n_test):
                    l2 = do_patching(model, tokenizer, device, layer_idx, pairs[i][0], pairs[i][1], component=component)
                    if l2 is not None:
                        l2s.append(l2)
                if l2s:
                    comp_results[component][ln][feat_name] = {
                        'median_l2': float(np.median(l2s)), 'n': len(l2s)
                    }
            all_results['components'] = comp_results
            save_incremental(out_dir, all_results)

    contrib = {}
    for layer_idx in comp_layers:
        ln = f'L{layer_idx}'
        contrib[ln] = {}
        for feat in comp_features:
            a = comp_results['attn'].get(ln, {}).get(feat, {}).get('median_l2', 0)
            m = comp_results['mlp'].get(ln, {}).get(feat, {}).get('median_l2', 0)
            total = a + m if (a + m) > 0 else 1
            contrib[ln][feat] = {'attn_pct': round(a/total*100,1), 'mlp_pct': round(m/total*100,1)}
            print(f"  {ln} {feat}: attn={a:.1f}({a/total*100:.0f}%), mlp={m:.1f}({m/total*100:.0f}%)")
    all_results['contribution'] = contrib
    save_incremental(out_dir, all_results)

    # ===== S3: 代数拟合 + 对偶验证 =====
    print(f"\n{'='*60}\nS3: 代数拟合 + 对偶验证\n{'='*60}", flush=True)
    algebraic_fits = {}
    duality_metrics = {}

    for feat_name in list(all_features.keys()):
        layers_x, l2_medians = [], []
        for layer_idx in resid_layers:
            key = f'L{layer_idx}'
            if key in resid_results and feat_name in resid_results[key]:
                layers_x.append(float(layer_idx))
                l2_medians.append(resid_results[key][feat_name]['median_l2'])

        if len(layers_x) >= 3:
            fits = fit_algebraic_model(layers_x, l2_medians)
            algebraic_fits[feat_name] = fits
            best = max(fits.keys(), key=lambda k: fits[k].get('r2', 0))
            best_r2 = fits[best].get('r2', 0)

            l0_val = l2_medians[0]
            last_val = l2_medians[-1]
            growth_ratio = last_val / l0_val if l0_val > 0 else 0
            slope = fits.get('linear', {}).get('a', 0)
            intercept = fits.get('linear', {}).get('b', 0)
            relative_growth = slope / intercept if intercept > 0 else 0

            duality_metrics[feat_name] = {
                'L0_median': float(l0_val),
                'L_final_median': float(last_val),
                'growth_ratio': float(growth_ratio),
                'best_fit': best,
                'best_r2': float(best_r2),
                'linear_slope': float(slope),
                'linear_intercept': float(intercept),
                'relative_growth': float(relative_growth),
            }

            category = "SYNTACTIC" if growth_ratio > 1.5 else "SEMANTIC"

            print(f"  {feat_name}: {best} R2={best_r2:.3f}, L0={l0_val:.0f}, "
                  f"L_final={last_val:.0f}, growth={growth_ratio:.2f}x, "
                  f"relative_growth={relative_growth:.4f} => {category}")

    all_results['algebraic_fits'] = algebraic_fits
    all_results['duality_metrics'] = duality_metrics
    save_incremental(out_dir, all_results)

    # ===== S4: 对偶假设统计检验 (6 vs 6) =====
    print(f"\n{'='*60}\nS4: 对偶假设统计检验 (6 syn vs 6 sem)\n{'='*60}", flush=True)

    # 正确分组!
    syntactic_names = ['tense', 'polarity', 'number', 'negation', 'question', 'person']
    semantic_names = ['sentiment', 'semantic_topic', 'voice', 'formality', 'definiteness', 'info_structure']

    syn_growth = [duality_metrics[f]['growth_ratio'] for f in syntactic_names if f in duality_metrics]
    sem_growth = [duality_metrics[f]['growth_ratio'] for f in semantic_names if f in duality_metrics]
    syn_l0 = [duality_metrics[f]['L0_median'] for f in syntactic_names if f in duality_metrics]
    sem_l0 = [duality_metrics[f]['L0_median'] for f in semantic_names if f in duality_metrics]
    syn_rel = [duality_metrics[f]['relative_growth'] for f in syntactic_names if f in duality_metrics]
    sem_rel = [duality_metrics[f]['relative_growth'] for f in semantic_names if f in duality_metrics]

    print(f"  Syntactic growth_ratios: {[f'{x:.2f}' for x in syn_growth]}")
    print(f"  Semantic growth_ratios: {[f'{x:.2f}' for x in sem_growth]}")
    print(f"  Syntactic L0: {[f'{x:.0f}' for x in syn_l0]}")
    print(f"  Semantic L0: {[f'{x:.0f}' for x in sem_l0]}")
    print(f"  Syntactic relative_growth: {[f'{x:.4f}' for x in syn_rel]}")
    print(f"  Semantic relative_growth: {[f'{x:.4f}' for x in sem_rel]}")

    duality_test = {}
    if syn_growth and sem_growth:
        from scipy import stats as scipy_stats
        try:
            u_growth, p_growth = scipy_stats.mannwhitneyu(syn_growth, sem_growth, alternative='greater')
            print(f"  Growth ratio (Mann-Whitney): U={u_growth:.1f}, p={p_growth:.4f}")
            duality_test['mann_whitney_growth_p'] = float(p_growth)
        except:
            p_growth = 1.0
            print(f"  Mann-Whitney growth test failed")

        try:
            u_l0, p_l0 = scipy_stats.mannwhitneyu(sem_l0, syn_l0, alternative='greater')
            print(f"  L0 value (Mann-Whitney): U={u_l0:.1f}, p={p_l0:.4f}")
            duality_test['mann_whitney_l0_p'] = float(p_l0)
        except:
            p_l0 = 1.0
            print(f"  Mann-Whitney L0 test failed")

        # Permutation test (更稳健)
        try:
            p_perm_growth = permutation_test(syn_growth, sem_growth, n_perm=10000)
            print(f"  Growth ratio (Permutation, 10k): p={p_perm_growth:.4f}")
            duality_test['permutation_growth_p'] = float(p_perm_growth)
        except:
            print(f"  Permutation test failed")

        try:
            p_perm_l0 = permutation_test(sem_l0, syn_l0, n_perm=10000)
            print(f"  L0 value (Permutation, 10k): p={p_perm_l0:.4f}")
            duality_test['permutation_l0_p'] = float(p_perm_l0)
        except:
            print(f"  Permutation L0 test failed")

        # Effect size
        syn_g_arr = np.array(syn_growth)
        sem_g_arr = np.array(sem_growth)
        pooled_std = np.sqrt((np.var(syn_g_arr) + np.var(sem_g_arr)) / 2) if len(syn_g_arr) > 1 and len(sem_g_arr) > 1 else 1
        effect_size = (np.mean(syn_g_arr) - np.mean(sem_g_arr)) / pooled_std if pooled_std > 0 else 0

        duality_test.update({
            'syntactic_growth_mean': float(np.mean(syn_growth)),
            'semantic_growth_mean': float(np.mean(sem_growth)),
            'syntactic_l0_mean': float(np.mean(syn_l0)),
            'semantic_l0_mean': float(np.mean(sem_l0)),
            'syntactic_rel_mean': float(np.mean(syn_rel)),
            'semantic_rel_mean': float(np.mean(sem_rel)),
            'effect_size': float(effect_size),
        })
        all_results['duality_test'] = duality_test
        save_incremental(out_dir, all_results)

    # ===== FINAL =====
    print(f"\n{'='*60}\nFINAL: {args.model}\n{'='*60}", flush=True)
    print(f"\nResidual median_l2:")
    for ln in sorted(resid_results.keys()):
        parts = []
        for feat in list(all_features.keys()):
            if feat in resid_results[ln]:
                d = resid_results[ln][feat]
                parts.append(f"{feat}={d['median_l2']:.0f}")
        print(f"  {ln}: {', '.join(parts)}")

    print(f"\nDuality Classification (6 syn vs 6 sem):")
    for feat in syntactic_names + semantic_names:
        if feat in duality_metrics:
            dm = duality_metrics[feat]
            preset = "SYN" if feat in syntactic_names else "SEM"
            cat = "SYNTACTIC" if dm['growth_ratio'] > 1.5 else "SEMANTIC"
            match = "OK" if (preset == "SYN" and cat == "SYNTACTIC") or (preset == "SEM" and cat == "SEMANTIC") else "MISMATCH!"
            print(f"  {feat} [{preset}]: growth={dm['growth_ratio']:.2f}x, L0={dm['L0_median']:.0f}, "
                  f"best={dm['best_fit']}(R2={dm['best_r2']:.3f}) => {cat} {match}")

    print(f"\nDONE! Saved to {out_dir}")


if __name__ == '__main__':
    main()
