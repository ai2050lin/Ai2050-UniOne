"""
Phase CCII: 严格SO(n)验证 + Attention Head因果定位 + 大样本因果追踪
=====================================================================

Phase CCI关键硬伤:
  1. H2的"正交偏差≈0"是三角恒等式: perp_norm = sin(θ) 恒成立
  2. 需要用3+方向的内积保持性来严格验证SO(n)
  3. DS7B L27的Attn因果汇聚cos=+0.989, 但不知道哪个head负责

核心测试:
  R1: 严格SO(n)验证 — 3个因果方向在层间是否保持内积?
      方法: 用5个特征的因果方向, 计算5×5内积矩阵在层间是否保持
      如果内积矩阵保持 → 正交旋转(SO(n)) → 因果纤维丛有联络结构
      如果内积矩阵不保持 → 非正交变换 → 需要更一般的几何框架

  R2: Attention Head因果定位 — DS7B L27哪个head负责因果汇聚?
      方法: 提取每个head的输出, 计算每个head中极性/时态的因果对齐
      如果某个head的cos特别高 → 该head是因果汇聚的核心

  R3: 大样本因果SAE预研 — 在因果空间(差分方向)训练简单SAE
      用PCA近似: 因果差分向量集合的PCA → 发现因果空间的主成分

运行:
  逐模型运行, 测试完一个后再测试另外一个, 避免GPU内存溢出
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json, gc, time
import numpy as np
import torch
from pathlib import Path


# ============================================================
# 数据集 (复用)
# ============================================================

def generate_polarity_pairs():
    """200对极性对"""
    templates = [
        ("The cat is here", "The cat is not here"), ("The dog is happy", "The dog is not happy"),
        ("The house is big", "The house is not big"), ("The phone is working", "The phone is not working"),
        ("The bridge is safe", "The bridge is not safe"), ("The star is visible", "The star is not visible"),
        ("The door is open", "The door is not open"), ("The lake is deep", "The lake is not deep"),
        ("The road is clear", "The road is not clear"), ("The wall is strong", "The wall is not strong"),
        ("The bird is flying", "The bird is not flying"), ("The fish is swimming", "The fish is not swimming"),
        ("The car is fast", "The car is not fast"), ("The tree is tall", "The tree is not tall"),
        ("The river is wide", "The river is not wide"), ("The book is interesting", "The book is not interesting"),
        ("The food is fresh", "The food is not fresh"), ("The light is bright", "The light is not bright"),
        ("The music is loud", "The music is not loud"), ("The weather is cold", "The weather is not cold"),
        ("The door was closed", "The door was not closed"), ("The child was playing", "The child was not playing"),
        ("The man was running", "The man was not running"), ("The woman was singing", "The woman was not singing"),
        ("The dog was barking", "The dog was not barking"), ("The sun was shining", "The sun was not shining"),
        ("The wind was blowing", "The wind was not blowing"), ("The rain was falling", "The rain was not falling"),
        ("The snow was melting", "The snow is not melting"), ("The fire was burning", "The fire was not burning"),
        ("I like the car", "I do not like the car"), ("She knows the answer", "She does not know the answer"),
        ("The river flows north", "The river does not flow north"), ("I understand the plan", "I do not understand the plan"),
        ("She likes the movie", "She does not like the movie"), ("The key works well", "The key does not work well"),
        ("The food tastes good", "The food does not taste good"), ("He drives the truck", "He does not drive the truck"),
        ("The system runs smoothly", "The system does not run smoothly"), ("The machine operates well", "The machine does not operate well"),
        ("He can swim", "He cannot swim"), ("The bird will come", "The bird will not come"),
        ("She can help", "She cannot help"), ("They will agree", "They will not agree"),
        ("He can solve it", "He cannot solve it"), ("The team will win", "The team will not win"),
        ("She can finish it", "She cannot finish it"), ("They will succeed", "They will not succeed"),
        ("The water is clean", "The water is not clean"), ("The machine works", "The machine does not work"),
        ("He can see", "He cannot see"), ("The plan was approved", "The plan was not approved"),
        ("She will come", "She will not come"), ("The road is open", "The road is not open"),
        ("The system is stable", "The system is not stable"), ("I trust the result", "I do not trust the result"),
        ("The door is locked", "The door is not locked"), ("The glass is empty", "The glass is not empty"),
        ("The room is dark", "The room is not dark"), ("The sky is blue", "The sky is not blue"),
        ("The grass is green", "The grass is not green"), ("The coffee is hot", "The coffee is not hot"),
        ("The ice is thick", "The ice is not thick"), ("The path is narrow", "The path is not narrow"),
        ("The cake is sweet", "The cake is not sweet"), ("The movie is long", "The movie is not long"),
        ("The test is hard", "The test is not hard"), ("The game is fair", "The game is not fair"),
        ("The price is low", "The price is not low"), ("The quality is high", "The quality is not high"),
        ("The speed is slow", "The speed is not slow"), ("The sound is clear", "The sound is not clear"),
        ("The color is bright", "The color is not bright"), ("The shape is round", "The shape is not round"),
        ("The size is small", "The size is not small"), ("The weight is heavy", "The weight is not heavy"),
        ("The distance is short", "The distance is not short"), ("The temperature is warm", "The temperature is not warm"),
        ("The apple is ripe", "The apple is not ripe"), ("The ocean is calm", "The ocean is not calm"),
        ("The mountain is high", "The mountain is not high"), ("The forest is dense", "The forest is not dense"),
        ("The desert is dry", "The desert is not dry"), ("The city is noisy", "The city is not noisy"),
        ("The village is quiet", "The village is not quiet"), ("This is correct", "This is not correct"),
        ("That is true", "That is not true"), ("It is real", "It is not real"),
        ("The answer is right", "The answer is not right"), ("The method is effective", "The method is not effective"),
        ("The solution is simple", "The solution is not simple"), ("The problem is easy", "The problem is not easy"),
        ("The task is difficult", "The task is not difficult"), ("The process is fast", "The process is not fast"),
        ("The result is accurate", "The result is not accurate"), ("The building is tall", "The building is not tall"),
        ("The park is beautiful", "The park is not beautiful"), ("The market is busy", "The market is not busy"),
        ("The engine is powerful", "The engine is not powerful"), ("The battery is full", "The battery is not full"),
        ("The signal is strong", "The signal is not strong"), ("The connection is stable", "The connection is not stable"),
        ("The network is secure", "The network is not secure"), ("The system is reliable", "The system is not reliable"),
        ("The screen is bright", "The screen is not bright"), ("He is running fast", "He is not running fast"),
        ("She is reading quietly", "She is not reading quietly"), ("They are sleeping peacefully", "They are not sleeping peacefully"),
        ("We are learning quickly", "We are not learning quickly"), ("The children are playing", "The children are not playing"),
        ("The birds are singing", "The birds are not singing"), ("The students are studying", "The students are not studying"),
        ("The workers are building", "The workers are not building"), ("The doctors are helping", "The doctors are not helping"),
        ("The boy is tall", "The boy is not tall"), ("The girl is short", "The girl is not short"),
        ("The man is strong", "The man is not strong"), ("The woman is kind", "The woman is not kind"),
        ("The cat is black", "The cat is not black"), ("The dog is brown", "The dog is not brown"),
        ("The flower is red", "The flower is not red"), ("The leaf is green", "The leaf is not green"),
        ("The stone is hard", "The stone is not hard"), ("The feather is soft", "The feather is not soft"),
        ("The metal is shiny", "The metal is not shiny"), ("The wood is rough", "The wood is not rough"),
        ("The glass is smooth", "The glass is not smooth"), ("The river is clean", "The river is not clean"),
        ("The mountain is steep", "The mountain is not steep"), ("The valley is deep", "The valley is not deep"),
        ("The island is small", "The island is not small"), ("The coast is rocky", "The coast is not rocky"),
        ("The plain is flat", "The plain is not flat"), ("The hill is green", "The hill is not green"),
        ("The cave is dark", "The cave is not dark"), ("The shore is sandy", "The shore is not sandy"),
        ("The cliff is high", "The cliff is not high"), ("The stream is narrow", "The stream is not narrow"),
        ("The pond is still", "The pond is not still"), ("The field is wide", "The field is not wide"),
        ("The garden is beautiful", "The garden is not beautiful"), ("The bridge is long", "The bridge is not long"),
        ("The tower is tall", "The tower is not tall"), ("The castle is old", "The castle is not old"),
        ("The church is quiet", "The church is not quiet"), ("The school is large", "The school is not large"),
        ("The hospital is clean", "The hospital is not clean"), ("The office is busy", "The office is not busy"),
        ("The shop is open", "The shop is not open"), ("The bank is safe", "The bank is not safe"),
        ("The hotel is comfortable", "The hotel is not comfortable"), ("The pool is deep", "The pool is not deep"),
        ("The track is fast", "The track is not fast"), ("The court is fair", "The court is not fair"),
        ("The stage is bright", "The stage is not bright"), ("The hall is wide", "The hall is not wide"),
        ("The room is warm", "The room is not warm"), ("The desk is clean", "The desk is not clean"),
        ("The chair is soft", "The chair is not soft"), ("The bed is comfortable", "The bed is not comfortable"),
        ("The table is round", "The table is not round"), ("The window is clear", "The window is not clear"),
        ("The floor is smooth", "The floor is not smooth"), ("The wall is thick", "The wall is not thick"),
        ("The roof is strong", "The roof is not strong"), ("The door is wide", "The door is not wide"),
        ("The stairs are steep", "The stairs are not steep"), ("The path is straight", "The path is not straight"),
        ("The road is long", "The road is not long"), ("The street is wide", "The street is not wide"),
        ("The lane is narrow", "The lane is not narrow"), ("The cloud disappeared", "The cloud did not disappear"),
        ("He arrived early", "He did not arrive early"), ("She passed the test", "She did not pass the test"),
        ("They finished the work", "They did not finish the work"), ("The student graduated", "The student did not graduate"),
        ("He answered correctly", "He did not answer correctly"), ("She returned home", "She did not return home"),
        ("The flower has bloomed", "The flower has not bloomed"), ("He has finished", "He has not finished"),
        ("She has arrived", "She has not arrived"), ("The tree has grown", "The tree has not grown"),
        ("The project has started", "The project has not started"), ("The river is calm", "The river is not calm"),
        ("The wind is gentle", "The wind is not gentle"), ("The rain is heavy", "The rain is not heavy"),
        ("The snow is light", "The snow is not light"), ("The sun is bright", "The sun is not bright"),
        ("The moon is full", "The moon is not full"), ("The cloud is thick", "The cloud is not thick"),
        ("The fog is dense", "The fog is not dense"), ("The storm is fierce", "The storm is not fierce"),
        ("The air is fresh", "The air is not fresh"), ("The soil is rich", "The soil is not rich"),
        ("The rock is solid", "The rock is not solid"), ("The sand is soft", "The sand is not soft"),
        ("The mud is wet", "The mud is not wet"), ("The dust is dry", "The dust is not dry"),
        ("The smoke is thick", "The smoke is not thick"), ("The fire is hot", "The fire is not hot"),
        ("The steam is hot", "The steam is not hot"), ("The ice is cold", "The ice is not cold"),
        ("The water is clear", "The water is not clear"), ("The oil is thick", "The oil is not thick"),
        ("The gas is light", "The gas is not light"), ("The liquid is cold", "The liquid is not cold"),
        ("The solid is hard", "The solid is not hard"), ("The crystal is clear", "The crystal is not clear"),
        ("The diamond is hard", "The diamond is not hard"), ("The pearl is round", "The pearl is not round"),
        ("The ruby is red", "The ruby is not red"), ("The emerald is green", "The emerald is not green"),
        ("The sapphire is blue", "The sapphire is not blue"), ("The gold is shiny", "The gold is not shiny"),
        ("The silver is bright", "The silver is not bright"), ("The copper is warm", "The copper is not warm"),
        ("The iron is strong", "The iron is not strong"), ("The steel is tough", "The steel is not tough"),
        ("The aluminum is light", "The aluminum is not light"), ("The lead is heavy", "The lead is not heavy"),
        ("The zinc is dull", "The zinc is not dull"), ("The tin is soft", "The tin is not soft"),
        ("The brass is yellow", "The brass is not yellow"), ("The bronze is brown", "The bronze is not brown"),
        ("The marble is smooth", "The marble is not smooth"), ("The granite is hard", "The granite is not hard"),
        ("The limestone is white", "The limestone is not white"), ("The slate is dark", "The slate is not dark"),
        ("The chalk is soft", "The chalk is not soft"), ("The clay is sticky", "The clay is not sticky"),
        ("The ash is light", "The ash is not light"), ("The charcoal is black", "The charcoal is not black"),
    ]
    return templates[:200]


def generate_tense_pairs():
    """200对时态对"""
    verbs = [
        ("runs", "ran"), ("walks", "walked"), ("plays", "played"),
        ("sings", "sang"), ("eats", "ate"), ("drinks", "drank"),
        ("writes", "wrote"), ("reads", "read"), ("speaks", "spoke"),
        ("drives", "drove"), ("swims", "swam"), ("flies", "flew"),
        ("grows", "grew"), ("knows", "knew"), ("thinks", "thought"),
        ("brings", "brought"), ("builds", "built"), ("catches", "caught"),
        ("teaches", "taught"), ("feels", "felt"), ("finds", "found"),
        ("gives", "gave"), ("holds", "held"), ("keeps", "kept"),
        ("leaves", "left"), ("loses", "lost"), ("meets", "met"),
        ("pays", "paid"), ("sells", "sold"), ("sends", "sent"),
        ("shuts", "shut"), ("sits", "sat"), ("sleeps", "slept"),
        ("spends", "spent"), ("stands", "stood"), ("takes", "took"),
        ("tells", "told"), ("understands", "understood"), ("wears", "wore"),
        ("wins", "won"), ("begins", "began"), ("breaks", "broke"),
        ("chooses", "chose"), ("draws", "drew"), ("falls", "fell"),
        ("forgets", "forgot"), ("gets", "got"), ("hangs", "hung"),
        ("hides", "hid"), ("hurts", "hurt"),
    ]
    subjects = [
        "The cat", "The dog", "He", "She", "The bird", "The man",
        "The woman", "The child", "The teacher", "The student",
        "The river", "The wind", "The fire", "The rain", "The sun",
        "I", "We", "They", "The team", "The group",
        "The boy", "The girl", "The king", "The queen", "The doctor",
        "The farmer", "The driver", "The artist", "The writer", "The singer",
        "The player", "The runner", "The worker", "The leader", "The speaker",
        "The nurse", "The chef", "The guard", "The pilot", "The judge",
    ]
    pairs = []
    for i, subj in enumerate(subjects):
        for pres, past in verbs[i % len(verbs):i % len(verbs) + 5]:
            pairs.append((f"{subj} {pres} every day", f"{subj} {past} yesterday"))
            if len(pairs) >= 200: return pairs[:200]
    return pairs[:200]


def generate_semantic_pairs():
    """200对语义对(具体vs抽象)"""
    pairs = [
        ("The rock is heavy","The idea is complex"),("The water is cold","The theory is abstract"),
        ("The fire is hot","The concept is vague"),("The tree is tall","The principle is clear"),
        ("The stone is hard","The notion is simple"),("The river flows","The argument flows"),
        ("The bird flies","The rumor spreads"),("The sun shines","The truth emerges"),
        ("The rain falls","The doubt grows"),("The wind blows","The debate continues"),
        ("The cat sleeps","The mind rests"),("The dog runs","The thought races"),
        ("The fish swims","The logic flows"),("The flower grows","The knowledge expands"),
        ("The cloud moves","The trend shifts"),("The mountain stands","The belief persists"),
        ("The ocean waves","The emotion surges"),("The snow melts","The tension dissolves"),
        ("The ice forms","The habit develops"),("The star glows","The hope shines"),
        ("The door opens","The opportunity arises"),("The path bends","The reasoning curves"),
        ("The bridge connects","The argument links"),("The wall blocks","The barrier prevents"),
        ("The light fades","The memory dims"),("The sound echoes","The influence resonates"),
        ("The color brightens","The mood improves"),("The shape changes","The perspective shifts"),
        ("The size increases","The importance grows"),("The weight decreases","The burden lessens"),
        ("The table is wooden","The statement is logical"),("The chair is metal","The claim is valid"),
        ("The book is paper","The argument is solid"),("The glass is clear","The explanation is transparent"),
        ("The coin is gold","The evidence is compelling"),("The rope is thick","The connection is strong"),
        ("The needle is sharp","The analysis is precise"),("The blanket is warm","The conclusion is comforting"),
        ("The mirror is flat","The comparison is fair"),("He held the cup","He held the opinion"),
        ("She touched the wall","She touched the issue"),("They built the house","They built the relationship"),
        ("We crossed the river","We crossed the boundary"),("He broke the stick","He broke the rule"),
        ("She opened the box","She opened the discussion"),("They closed the gate","They closed the case"),
        ("We found the key","We found the solution"),("He lost the map","He lost the direction"),
        ("She kept the coin","She kept the secret"),("The apple is red","The theory is sound"),
        ("The grass is green","The logic is valid"),("The sky is blue","The concept is clear"),
        ("The blood is red","The passion is intense"),("The snow is white","The truth is pure"),
        ("The night is black","The mystery is deep"),("The gold is yellow","The value is high"),
        ("The ocean is blue","The freedom is vast"),("The fire is orange","The anger is hot"),
        ("The rock is solid","The argument is firm"),("The water is liquid","The situation is fluid"),
        ("The ice is frozen","The relationship is cold"),("The cloud is soft","The idea is gentle"),
        ("The steel is hard","The evidence is strong"),("The feather is light","The comment is subtle"),
        ("The mountain is massive","The problem is enormous"),("The desert is vast","The implication is broad"),
        ("The diamond is precious","The insight is valuable"),("The bread is fresh","The approach is novel"),
        ("The paint is wet","The plan is flexible"),("The metal is cold","The response is neutral"),
        ("The wood is warm","The tone is friendly"),("The brick is heavy","The consequence is significant"),
        ("The paper is thin","The excuse is weak"),("The rope is strong","The bond is durable"),
        ("The glass is fragile","The agreement is delicate"),("The rubber is flexible","The policy is adaptable"),
        ("The shadow is temporary","The effect is brief"),("The current is flowing","The trend is ongoing"),
        ("The wave is rising","The movement is growing"),("The flame is dancing","The debate is active"),
        ("The storm is approaching","The crisis is looming"),("The tide is turning","The attitude is shifting"),
        ("The dawn is breaking","The understanding is emerging"),("The seed is growing","The idea is developing"),
        ("The fruit is ripening","The project is maturing"),("The leaf is falling","The influence is declining"),
        ("The root is deepening","The foundation is strengthening"),("The branch is extending","The scope is expanding"),
        ("The garden is blooming","The community is thriving"),("The field is producing","The effort is yielding"),
        ("The winter is ending","The difficulty is passing"),("The spring is returning","The hope is reviving"),
        ("The summer is peaking","The activity is intensifying"),("The autumn is arriving","The transition is happening"),
        ("The rain is stopping","The conflict is ceasing"),("The wind is calming","The tension is easing"),
        ("The sun is setting","The phase is concluding"),("The moon is rising","The alternative is appearing"),
        ("The cloud is clearing","The confusion is resolving"),("The fog is lifting","The mystery is clarifying"),
        ("The storm is passing","The crisis is resolving"),("The flood is receding","The pressure is decreasing"),
        ("The lightning is striking","The breakthrough is happening"),("The rainbow is appearing","The solution is emerging"),
        ("The horizon is expanding","The perspective is widening"),("The landscape is changing","The situation is evolving"),
        ("The pattern is emerging","The structure is appearing"),("The structure is forming","The organization is developing"),
        ("The order is establishing","The system is stabilizing"),("The chaos is increasing","The disorder is growing"),
        ("The information is accumulating","The knowledge is building"),("The noise is decreasing","The clarity is improving"),
        ("The signal is strengthening","The evidence is mounting"),("The meaning is deepening","The significance is growing"),
    ]
    return pairs[:200]


def generate_sentiment_pairs():
    """200对情感对"""
    pairs = [
        ("The movie was wonderful","The movie was terrible"),("The food was delicious","The food was disgusting"),
        ("The weather was beautiful","The weather was awful"),("The music was amazing","The music was horrible"),
        ("The book was fascinating","The book was boring"),("The game was exciting","The game was dull"),
        ("The trip was enjoyable","The trip was miserable"),("The party was fantastic","The party was dreadful"),
        ("The show was brilliant","The show was mediocre"),("The concert was spectacular","The concert was disappointing"),
        ("The hotel was comfortable","The hotel was uncomfortable"),("The service was excellent","The service was poor"),
        ("The product was outstanding","The product was inferior"),("The performance was superb","The performance was subpar"),
        ("The result was impressive","The result was unimpressive"),("The experience was pleasant","The experience was unpleasant"),
        ("The atmosphere was welcoming","The atmosphere was hostile"),("The staff was friendly","The staff was rude"),
        ("The quality was superior","The quality was inferior"),("The design was elegant","The design was clumsy"),
        ("The solution was clever","The solution was foolish"),("The idea was innovative","The idea was outdated"),
        ("The approach was effective","The approach was ineffective"),("The method was efficient","The method was wasteful"),
        ("The strategy was successful","The strategy was failing"),("The outcome was positive","The outcome was negative"),
        ("The feedback was encouraging","The feedback was discouraging"),("The progress was remarkable","The progress was negligible"),
        ("The improvement was significant","The improvement was minimal"),("The achievement was extraordinary","The achievement was ordinary"),
        ("The discovery was groundbreaking","The discovery was trivial"),("The invention was revolutionary","The invention was conventional"),
        ("The contribution was valuable","The contribution was worthless"),("The effort was commendable","The effort was lackluster"),
        ("The dedication was admirable","The dedication was questionable"),("The commitment was unwavering","The commitment was halfhearted"),
        ("The determination was fierce","The determination was weak"),("The courage was heroic","The courage was cowardly"),
        ("The kindness was generous","The kindness was selfish"),("The honesty was refreshing","The honesty was deceptive"),
        ("The patience was admirable","The patience was irritable"),("The wisdom was profound","The wisdom was shallow"),
        ("The humor was delightful","The humor was offensive"),("The creativity was inspiring","The creativity was uninspired"),
        ("The intelligence was remarkable","The intelligence was unremarkable"),("The beauty was stunning","The beauty was plain"),
        ("The grace was elegant","The grace was awkward"),("The charm was captivating","The charm was repulsive"),
        ("The warmth was comforting","The warmth was chilling"),("The joy was contagious","The joy was forced"),
        ("The hope was uplifting","The hope was diminishing"),("The love was deep","The love was shallow"),
        ("The peace was serene","The peace was turbulent"),("The freedom was liberating","The freedom was restricting"),
        ("The justice was fair","The justice was biased"),("The truth was clear","The truth was obscured"),
        ("The trust was solid","The trust was broken"),("The respect was mutual","The respect was one-sided"),
        ("The cooperation was seamless","The cooperation was chaotic"),("The harmony was perfect","The harmony was discordant"),
        ("The unity was strong","The unity was fragmented"),("The balance was stable","The balance was precarious"),
        ("The order was maintained","The order was disrupted"),("The structure was solid","The structure was crumbling"),
        ("The foundation was firm","The foundation was shaky"),("The system was reliable","The system was unreliable"),
        ("The process was smooth","The process was rough"),("The journey was pleasant","The journey was arduous"),
        ("The path was clear","The path was blocked"),("The road was easy","The road was difficult"),
        ("The answer was simple","The answer was complex"),("The question was easy","The question was hard"),
        ("The task was manageable","The task was overwhelming"),("The challenge was stimulating","The challenge was exhausting"),
        ("The situation was favorable","The situation was unfavorable"),("The condition was optimal","The condition was suboptimal"),
        ("The opportunity was golden","The opportunity was missed"),("The moment was perfect","The moment was wrong"),
        ("The choice was wise","The choice was foolish"),("The decision was correct","The decision was mistaken"),
        ("The action was appropriate","The action was inappropriate"),("The attitude was positive","The attitude was negative"),
        ("The outlook was optimistic","The outlook was pessimistic"),("The vision was clear","The vision was blurry"),
        ("The goal was achievable","The goal was unrealistic"),("The plan was feasible","The plan was impractical"),
        ("The dream was attainable","The dream was impossible"),("The ambition was noble","The ambition was greedy"),
        ("The standard was high","The standard was low"),("The limit was pushed","The limit was accepted"),
        ("The boundary was expanded","The boundary was restricted"),("The impact was profound","The impact was superficial"),
        ("The effect was lasting","The effect was temporary"),("The influence was strong","The influence was weak"),
        ("The power was great","The power was small"),("The strength was formidable","The strength was feeble"),
        ("The energy was vibrant","The energy was depleted"),("The spirit was indomitable","The spirit was broken"),
        ("The will was unbreakable","The will was fragile"),("The faith was unwavering","The faith was shaky"),
        ("The confidence was high","The confidence was low"),("The clarity was crystal","The clarity was murky"),
        ("The precision was exact","The precision was approximate"),("The accuracy was perfect","The accuracy was flawed"),
        ("The care was thorough","The care was careless"),("The focus was sharp","The focus was blurred"),
        ("The understanding was deep","The understanding was surface"),("The insight was penetrating","The insight was shallow"),
        ("The analysis was rigorous","The analysis was sloppy"),("The judgment was sound","The judgment was flawed"),
        ("The reasoning was logical","The reasoning was illogical"),("The argument was compelling","The argument was weak"),
        ("The evidence was convincing","The evidence was dubious"),("The explanation was lucid","The explanation was obscure"),
        ("The description was vivid","The description was vague"),("The performance was stellar","The performance was mediocre"),
        ("The execution was flawless","The execution was flawed"),("The quality was premium","The quality was cheap"),
        ("The value was exceptional","The value was ordinary"),("The price was reasonable","The price was exorbitant"),
        ("The investment was profitable","The investment was losing"),("The benefit was significant","The benefit was negligible"),
        ("The advantage was clear","The advantage was marginal"),("The profit was handsome","The profit was thin"),
        ("The resource was plentiful","The resource was scarce"),("The supply was abundant","The supply was limited"),
        ("The demand was strong","The demand was weak"),("The market was booming","The market was crashing"),
        ("The growth was robust","The growth was stagnant"),("The development was rapid","The development was slow"),
        ("The innovation was breakthrough","The innovation was incremental"),("The change was transformative","The change was superficial"),
        ("The transformation was complete","The transformation was partial"),("The evolution was remarkable","The evolution was gradual"),
        ("The progress was steady","The progress was halting"),("The advancement was significant","The advancement was minor"),
    ]
    return pairs[:200]


def generate_number_pairs():
    """150对数量对(单数vs复数)"""
    pairs = []
    nouns_sg = ["cat","dog","bird","tree","flower","car","house","book","river","mountain",
                "child","woman","man","student","teacher","doctor","artist","farmer","driver","player",
                "city","village","forest","desert","island","lake","ocean","bridge","tower","castle",
                "apple","orange","banana","grape","cherry","peach","lemon","melon","plum","pear",
                "table","chair","desk","lamp","mirror","window","door","floor","roof","wall",
                "knife","fork","spoon","plate","cup","bowl","glass","bottle","pot","pan"]
    adjectives = ["is tall","is small","is heavy","is light","is fast","is slow","is strong",
                  "is weak","is old","is new","is big","is little","is clean","is dirty",
                  "is hard","is soft","is hot","is cold","is bright","is dark"]
    verbs_pres = ["runs","walks","jumps","swims","flies","climbs","sings","dances","plays","sleeps"]
    
    idx = 0
    for noun in nouns_sg:
        for adj in adjectives[idx % len(adjectives):idx % len(adjectives) + 2]:
            sg = f"The {noun} {adj}"
            pl_noun = noun + "s" if not noun.endswith("s") else noun + "es"
            pl_adj = adj.replace("is ", "are ")
            pl = f"The {pl_noun} {pl_adj}"
            pairs.append((sg, pl))
            idx += 1
            if len(pairs) >= 150: return pairs[:150]
    
    for noun in nouns_sg[:30]:
        for verb in verbs_pres[idx % len(verbs_pres):idx % len(verbs_pres) + 2]:
            pl_noun = noun + "s" if not noun.endswith("s") else noun + "es"
            verb_no_s = verb[:-1] if verb.endswith("s") else verb
            sg2 = f"The {pl_noun} {verb} well"
            pl2 = f"The {pl_noun} {verb_no_s} well"
            pairs.append((sg2, pl2))
            idx += 1
            if len(pairs) >= 150: return pairs[:150]
    
    return pairs[:150]


# ============================================================
# 模型加载
# ============================================================

def load_model_fast(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    PATHS = {
        "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    path = PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if model_name in ["glm4", "deepseek7b"]:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb_config, device_map="auto",
                                                      trust_remote_code=True, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cpu',
                                                      trust_remote_code=True, local_files_only=True)
        model = model.to('cuda')
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


# ============================================================
# R1: 严格SO(n)验证 — 内积矩阵保持性
# ============================================================

def test_r1(model_name, model, tokenizer, device, d_model, n_layers):
    """
    R1: 严格验证因果纤维丛的层间变换是否为SO(n)旋转
    
    方法: 提取5个特征在多个层的因果方向, 计算5×5内积矩阵
    如果内积矩阵在层间保持 → 正交旋转(SO(n))
    如果内积矩阵不保持 → 非正交变换
    
    关键: 之前H2的"正交偏差≈0"是三角恒等式(对2个方向恒成立)
    现在用5个方向的完整内积矩阵来严格验证
    """
    print("\n" + "=" * 70)
    print("R1: 严格SO(n)验证 — 5×5内积矩阵保持性 (200对/特征)")
    print("=" * 70)
    
    feature_generators = {
        'polarity': generate_polarity_pairs,
        'tense': generate_tense_pairs,
        'semantic': generate_semantic_pairs,
        'sentiment': generate_sentiment_pairs,
        'number': generate_number_pairs,
    }
    
    fnames = list(feature_generators.keys())
    n_features = len(fnames)
    
    # 采样层: 首层, 中间层, 末层 + 额外几个
    sample_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]))
    if n_layers > 10:
        # 加入更多层以获得连续性
        extra = list(range(0, n_layers, max(1, n_layers//8)))
        sample_layers = sorted(set(sample_layers + extra))
    
    print(f"  采样层: {sample_layers}")
    print(f"  特征: {fnames}")
    
    # 对每个层, 提取5个特征的因果方向
    layer_gram_matrices = {}  # {layer_idx: 5x5 内积矩阵}
    layer_diff_vectors = {}   # {layer_idx: {feature: diff_n}}
    
    for li in sample_layers:
        # 对每个特征, 提取在该层的因果方向
        diff_vectors = {}
        for fname in fnames:
            pairs = feature_generators[fname]()[:200]
            texts_a = [a for a, b in pairs]
            texts_b = [b for a, b in pairs]
            all_texts = texts_a + texts_b
            n_half = len(texts_a)
            
            # 用hook提取该层输出
            captured = []
            layer = model.model.layers[li]
            
            def make_hook(storage):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple): h = output[0].detach().cpu().float()
                    else: h = output.detach().cpu().float()
                    storage.append(h[0])
                return hook_fn
            
            h = layer.register_forward_hook(make_hook(captured))
            
            reprs_a = []
            reprs_b = []
            for i, text in enumerate(all_texts):
                captured.clear()
                toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    try: _ = model(**toks)
                    except: continue
                if captured:
                    h_vec = captured[0].numpy()[-1]  # last token
                    if i < n_half:
                        reprs_a.append(h_vec)
                    else:
                        reprs_b.append(h_vec)
            
            h.remove()
            
            # 差分方向
            if len(reprs_a) > 0 and len(reprs_b) > 0:
                diff = np.mean(reprs_b, axis=0) - np.mean(reprs_a, axis=0)
                norm = np.linalg.norm(diff)
                if norm > 1e-10:
                    diff_n = diff / norm
                else:
                    diff_n = np.zeros(d_model)
            else:
                diff_n = np.zeros(d_model)
                norm = 0.0
            
            diff_vectors[fname] = {'diff_n': diff_n, 'norm': float(norm),
                                   'n_a': len(reprs_a), 'n_b': len(reprs_b)}
            print(f"  L{li} {fname:>10}: n={len(reprs_a)}/{len(reprs_b)}, norm={norm:.2f}")
        
        layer_diff_vectors[li] = diff_vectors
        
        # 计算5×5内积矩阵 (Gram matrix)
        gram = np.zeros((n_features, n_features))
        for i, f1 in enumerate(fnames):
            for j, f2 in enumerate(fnames):
                gram[i, j] = float(np.dot(diff_vectors[f1]['diff_n'], diff_vectors[f2]['diff_n']))
        layer_gram_matrices[li] = gram
    
    # 分析: 相邻层的Gram矩阵是否保持?
    results = {'sample_layers': sample_layers, 'n_pairs': 200, 'fnames': fnames}
    
    print(f"\n  {'Layer':>6} {'Gram行列式':>12} {'Gram trace':>12} {'与L0 Gram差':>14} {'与L0 Frobenius':>16}")
    print("  " + "-" * 65)
    
    gram_L0 = layer_gram_matrices[sample_layers[0]]
    det_L0 = np.linalg.det(gram_L0)
    trace_L0 = np.trace(gram_L0)
    
    gram_drifts = []
    
    for li in sample_layers:
        gram = layer_gram_matrices[li]
        det = np.linalg.det(gram)
        trace = np.trace(gram)
        
        # 与L0的Gram矩阵差异
        gram_diff = gram - gram_L0
        frob_diff = np.linalg.norm(gram_diff, 'fro')
        
        # 归一化的差异 (相对于L0的Frobenius范数)
        frob_L0 = np.linalg.norm(gram_L0, 'fro')
        rel_diff = frob_diff / max(frob_L0, 1e-10)
        
        print(f"  L{li:>4} {det:>12.4f} {trace:>12.4f} {frob_diff:>14.4f} {rel_diff:>15.2%}")
        
        gram_drifts.append({
            'layer': li, 'det': float(det), 'trace': float(trace),
            'frob_diff_L0': float(frob_diff), 'rel_diff_L0': float(rel_diff),
        })
    
    results['gram_drifts'] = gram_drifts
    results['gram_L0'] = gram_L0.tolist()
    
    # 关键分析: Gram矩阵是否保持?
    # 如果层间变换是正交旋转(R^T R = I), 则 Gram' = Gram (完全保持)
    # 如果Gram变化 → 非正交变换
    
    # 相邻层的Gram差异
    adj_gram_diffs = []
    for i in range(len(sample_layers) - 1):
        li1, li2 = sample_layers[i], sample_layers[i+1]
        gram1, gram2 = layer_gram_matrices[li1], layer_gram_matrices[li2]
        diff = np.linalg.norm(gram2 - gram1, 'fro')
        base = np.linalg.norm(gram1, 'fro')
        rel = diff / max(base, 1e-10)
        adj_gram_diffs.append({
            'from': li1, 'to': li2, 
            'frob_diff': float(diff), 'rel_diff': float(rel),
        })
    
    results['adj_gram_diffs'] = adj_gram_diffs
    
    mean_rel = np.mean([d['rel_diff'] for d in adj_gram_diffs])
    max_rel = np.max([d['rel_diff'] for d in adj_gram_diffs])
    
    print(f"\n  相邻层Gram差异: mean={mean_rel:.2%}, max={max_rel:.2%}")
    
    if mean_rel < 0.05:
        print(f"  → ★★★ Gram矩阵近似保持! 层间变换近似正交旋转! ★★★")
    elif mean_rel < 0.15:
        print(f"  → Gram矩阵部分保持, 有非正交分量")
    else:
        print(f"  → Gram矩阵显著变化, 层间变换非正交")
    
    # 额外分析: Gram矩阵的特征值分布变化
    print(f"\n  Gram矩阵特征值分布:")
    for li in sample_layers[:5] + [sample_layers[-1]]:
        if li in layer_gram_matrices:
            eigs = np.sort(np.linalg.eigvalsh(layer_gram_matrices[li]))[::-1]
            print(f"  L{li:>4}: {[f'{e:.3f}' for e in eigs]}")
    
    # 保存所有Gram矩阵
    results['all_gram_matrices'] = {str(li): layer_gram_matrices[li].tolist() for li in sample_layers}
    
    return results


# ============================================================
# R2: Attention Head因果定位
# ============================================================

def test_r2(model_name, model, tokenizer, device, d_model, n_layers):
    """
    R2: 定位哪个Attention Head负责因果汇聚
    
    方法: 提取DS7B L27每个head的attn_output
    计算每个head中极性/时态的因果对齐cos
    如果某个head的cos特别高 → 该head是因果汇聚的核心
    
    对于非4bit模型, 可以直接提取W_o权重来分离head
    对于4bit模型, 用hook在attn内部捕获每个head的输出
    """
    print("\n" + "=" * 70)
    print("R2: Attention Head因果定位 (200对/特征)")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()[:200]
    tense_pairs = generate_tense_pairs()[:200]
    
    # 选择关键层
    if model_name == 'deepseek7b':
        target_layers = [n_layers - 1, n_layers - 2, n_layers - 3]  # L27, L26, L25
    elif model_name == 'glm4':
        target_layers = [0, n_layers - 1]  # L0(显著), L39
    else:
        target_layers = [n_layers - 1]
    
    # 获取n_heads
    layer0 = model.model.layers[0]
    n_heads = layer0.self_attn.config.num_attention_heads
    d_head = d_model // n_heads if hasattr(layer0.self_attn.config, 'head_dim') is False else layer0.self_attn.config.head_dim
    
    # 对于Qwen2架构, 检查head_dim
    if hasattr(layer0.self_attn.config, 'head_dim') and layer0.self_attn.config.head_dim is not None:
        d_head = layer0.self_attn.config.head_dim
    else:
        d_head = d_model // n_heads
    
    print(f"  n_heads={n_heads}, d_head={d_head}")
    print(f"  target_layers={target_layers}")
    
    results = {'n_heads': n_heads, 'd_head': d_head, 'target_layers': target_layers}
    
    for li in target_layers:
        print(f"\n  --- L{li} ---")
        layer = model.model.layers[li]
        
        # 方法: 用hook捕获self_attn输出, 然后用W_o分离head
        # 但4bit模型的W_o形状异常, 所以用另一种方法:
        # 在attn的output_proj之前注册hook, 获取concat_heads
        # 或者更简单: 直接在attn_output上分析, 因为attn_output = W_o @ concat_heads
        # 如果W_o是方阵, concat_heads = W_o^{-1} @ attn_output → 分离head
        # 但W_o可能不是方阵(GQA), 所以用另一种方法
        
        # 更好的方法: 直接分析W_o的列空间
        # 每个head的输出是W_o的d_head列的线性组合
        # 我们可以计算attn_output在每个head对应的W_o子空间中的投影
        
        # 但4bit模型的W_o异常 → 用最直接的方法:
        # 注册hook到self_attn, 获取output
        # 然后对output做SVD, 看head级贡献
        
        # 最实际的方法: 逐head patching
        # 但这太慢了, 200对 × n_heads × n_layers
        
        # 简化方法: 用W_o的SVD分解来估计每个head的贡献
        # W_o shape: [d_model, d_model] (标准) 或 [d_model, n_heads*d_head] (GQA)
        
        # 尝试获取W_o
        W_o_raw = layer.self_attn.o_proj.weight
        
        # 尝试dequantize (4bit)
        try:
            W_o = W_o_raw.detach().cpu().float().numpy()
            print(f"  W_o shape: {W_o.shape}")
            valid_shape = len(W_o.shape) == 2 and W_o.shape[0] == d_model
        except:
            W_o = None
            valid_shape = False
            print(f"  W_o dequantize failed")
        
        if valid_shape and W_o.shape[1] == n_heads * d_head:
            # 标准情况: W_o [d_model, n_heads*d_head]
            # 分离每个head对应的W_o列
            print(f"  → W_o形状正常, 分离{n_heads}个head的子空间")
            
            head_subspaces = []
            for h in range(n_heads):
                start = h * d_head
                end = (h + 1) * d_head
                W_o_h = W_o[:, start:end]  # [d_model, d_head]
                head_subspaces.append(W_o_h)
            
        elif valid_shape and W_o.shape[1] == d_model:
            # 方阵: W_o [d_model, d_model]
            # 假设标准head布局
            print(f"  → W_o为方阵, 分离{n_heads}个head的子空间")
            
            head_subspaces = []
            for h in range(n_heads):
                start = h * d_head
                end = (h + 1) * d_head
                W_o_h = W_o[:, start:end]  # [d_model, d_head]
                head_subspaces.append(W_o_h)
        else:
            # 4bit异常: 用attn_output的整体分析代替
            print(f"  → W_o形状异常, 用attn_output整体分析代替head分离")
            head_subspaces = None
        
        # 提取极性和时态在该层的attn输出
        def get_attn_diffs(pairs, label):
            texts_a = [a for a, b in pairs]
            texts_b = [b for a, b in pairs]
            all_texts = texts_a + texts_b
            n_half = len(texts_a)
            
            attn_outs = []
            
            def make_hook(storage):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple): storage.append(output[0].detach().cpu().float().numpy())
                    else: storage.append(output.detach().cpu().float().numpy())
                return hook_fn
            
            h_attn = layer.self_attn.register_forward_hook(make_hook(attn_outs))
            
            reprs_a = []
            reprs_b = []
            for i, text in enumerate(all_texts):
                attn_outs.clear()
                toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    try: _ = model(**toks)
                    except: continue
                if attn_outs:
                    out = attn_outs[0][0, -1, :]  # [d_model]
                    if i < n_half:
                        reprs_a.append(out)
                    else:
                        reprs_b.append(out)
            
            h_attn.remove()
            
            if len(reprs_a) > 0 and len(reprs_b) > 0:
                diff = np.mean(reprs_b, axis=0) - np.mean(reprs_a, axis=0)
                norm = np.linalg.norm(diff)
                diff_n = diff / max(norm, 1e-10)
            else:
                diff = np.zeros(d_model)
                diff_n = np.zeros(d_model)
                norm = 0.0
            
            return diff, diff_n, norm, len(reprs_a), len(reprs_b)
        
        print(f"  提取极性attn输出...")
        pol_diff, pol_diff_n, pol_norm, n_a, n_b = get_attn_diffs(pol_pairs, "polarity")
        print(f"  极性: n={n_a}/{n_b}, norm={pol_norm:.2f}")
        
        print(f"  提取时态attn输出...")
        tense_diff, tense_diff_n, tense_norm, n_a, n_b = get_attn_diffs(tense_pairs, "tense")
        print(f"  时态: n={n_a}/{n_b}, norm={tense_norm:.2f}")
        
        # 整体attn因果cos
        overall_cos = float(np.dot(pol_diff_n, tense_diff_n))
        print(f"\n  整体attn因果cos: {overall_cos:+.4f}")
        
        if head_subspaces is not None:
            # 分离每个head的贡献
            print(f"\n  Head级分析:")
            head_results = []
            
            for h_idx in range(n_heads):
                W_o_h = head_subspaces[h_idx]  # [d_model, d_head]
                
                # 极性差分在head h子空间中的投影
                pol_proj = W_o_h @ (W_o_h.T @ pol_diff)  # [d_model]
                pol_proj_norm = np.linalg.norm(pol_proj)
                
                # 时态差分在head h子空间中的投影
                tense_proj = W_o_h @ (W_o_h.T @ tense_diff)  # [d_model]
                tense_proj_norm = np.linalg.norm(tense_proj)
                
                # 投影向量的cos
                if pol_proj_norm > 1e-10 and tense_proj_norm > 1e-10:
                    head_cos = float(np.dot(pol_proj / pol_proj_norm, tense_proj / tense_proj_norm))
                else:
                    head_cos = 0.0
                
                # 该head的贡献比例
                pol_energy = pol_proj_norm ** 2
                tense_energy = tense_proj_norm ** 2
                
                head_results.append({
                    'head': h_idx,
                    'head_cos': float(head_cos),
                    'pol_proj_norm': float(pol_proj_norm),
                    'tense_proj_norm': float(tense_proj_norm),
                    'pol_energy': float(pol_energy),
                    'tense_energy': float(tense_energy),
                })
            
            # 按能量排序
            head_results.sort(key=lambda x: x['pol_energy'] + x['tense_energy'], reverse=True)
            
            print(f"  {'Head':>6} {'cos':>8} {'pol_proj':>10} {'tense_proj':>11} {'total_energy':>13}")
            print("  " + "-" * 55)
            
            for hr in head_results[:15]:  # 只显示top-15
                total_e = hr['pol_energy'] + hr['tense_energy']
                mark = "★★★" if abs(hr['head_cos']) > 0.5 else ("★★" if abs(hr['head_cos']) > 0.3 else "")
                print(f"  h{hr['head']:>4} {hr['head_cos']:>+8.4f} {hr['pol_proj_norm']:>10.2f} {hr['tense_proj_norm']:>11.2f} {total_e:>13.2f} {mark}")
            
            # 找出cos最高的head
            top_cos_heads = sorted(head_results, key=lambda x: abs(x['head_cos']), reverse=True)[:5]
            print(f"\n  Top-5 因果对齐heads:")
            for hr in top_cos_heads:
                print(f"    h{hr['head']}: cos={hr['head_cos']:+.4f}, pol_proj={hr['pol_proj_norm']:.2f}, tense_proj={hr['tense_proj_norm']:.2f}")
            
            results[f'L{li}'] = {
                'overall_cos': float(overall_cos),
                'head_results': head_results,
                'top_cos_heads': top_cos_heads,
                'n_heads': n_heads,
            }
        else:
            # W_o异常, 只保存整体结果
            results[f'L{li}'] = {
                'overall_cos': float(overall_cos),
                'head_analysis': 'skipped_W_o_anomaly',
            }
    
    return results


# ============================================================
# R3: 因果空间PCA — 大样本
# ============================================================

def test_r3(model_name, model, tokenizer, device, d_model, n_layers):
    """
    R3: 在因果空间(差分方向)做PCA, 发现因果空间的主成分
    
    方法: 收集大量差分向量(5个特征×200对), 在最后一层提取
    对差分向量集合做PCA, 分析主成分的结构
    """
    print("\n" + "=" * 70)
    print("R3: 因果空间PCA — 大样本 (200对/特征)")
    print("=" * 70)
    
    feature_generators = {
        'polarity': generate_polarity_pairs,
        'tense': generate_tense_pairs,
        'semantic': generate_semantic_pairs,
        'sentiment': generate_sentiment_pairs,
        'number': generate_number_pairs,
    }
    
    fnames = list(feature_generators.keys())
    last_layer = n_layers - 1
    
    # 收集大量差分向量 (每个特征200对, 取subsample得到更多向量)
    all_diffs = []
    diff_labels = []
    
    layer = model.model.layers[last_layer]
    
    def make_hook(storage):
        def hook_fn(module, input, output):
            if isinstance(output, tuple): h = output[0].detach().cpu().float()
            else: h = output.detach().cpu().float()
            storage.append(h[0])
        return hook_fn
    
    for fname in fnames:
        pairs = feature_generators[fname]()[:200]
        texts_a = [a for a, b in pairs]
        texts_b = [b for a, b in pairs]
        
        # 为了得到更多差分向量, 分组计算(每20对一组)
        group_size = 20
        n_groups = len(texts_a) // group_size
        
        for g in range(n_groups):
            ga = texts_a[g*group_size:(g+1)*group_size]
            gb = texts_b[g*group_size:(g+1)*group_size]
            all_texts = ga + gb
            n_half = len(ga)
            
            captured = []
            h = layer.register_forward_hook(make_hook(captured))
            
            reprs_a = []
            reprs_b = []
            for i, text in enumerate(all_texts):
                captured.clear()
                toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    try: _ = model(**toks)
                    except: continue
                if captured:
                    h_vec = captured[0].numpy()[-1]
                    if i < n_half:
                        reprs_a.append(h_vec)
                    else:
                        reprs_b.append(h_vec)
            
            h.remove()
            
            if len(reprs_a) > 0 and len(reprs_b) > 0:
                diff = np.mean(reprs_b, axis=0) - np.mean(reprs_a, axis=0)
                all_diffs.append(diff)
                diff_labels.append(fname)
        
        print(f"  {fname:>10}: {n_groups} groups collected")
    
    # 转为矩阵
    diff_matrix = np.array(all_diffs)  # [n_diffs, d_model]
    n_diffs = len(all_diffs)
    print(f"\n  总差分向量数: {n_diffs}")
    
    # PCA
    from sklearn.decomposition import PCA
    
    # 中心化
    diff_mean = diff_matrix.mean(axis=0)
    diff_centered = diff_matrix - diff_mean
    
    n_components = min(50, n_diffs - 1, d_model)
    pca = PCA(n_components=n_components)
    pca.fit(diff_centered)
    
    print(f"\n  PCA结果 (n_components={n_components}):")
    print(f"  前10主成分方差解释率: {[f'{v:.4f}' for v in pca.explained_variance_ratio_[:10]]}")
    print(f"  前10主成分累计: {np.cumsum(pca.explained_variance_ratio_[:10])}")
    
    # 分析: 每个特征在主成分空间中的分布
    pc_coords = pca.transform(diff_centered)  # [n_diffs, n_components]
    
    print(f"\n  各特征在前5个PC上的均值:")
    print(f"  {'Feature':>10}", end="")
    for pc in range(5): print(f"  {'PC'+str(pc):>8}", end="")
    print()
    
    feature_pc_means = {}
    for fname in fnames:
        mask = [i for i, l in enumerate(diff_labels) if l == fname]
        if mask:
            means = pc_coords[mask, :5].mean(axis=0)
            feature_pc_means[fname] = means.tolist()
            print(f"  {fname:>10}", end="")
            for m in means: print(f"  {m:>8.3f}", end="")
            print()
    
    # 分析: 不同特征是否占据不同的PC子空间?
    # 如果是的 → 因果空间有稀疏结构 → SAE可能有意义
    
    # 计算每个特征在PC1-5上的"主导度"
    print(f"\n  每个PC上哪个特征主导:")
    for pc in range(5):
        # 该PC上各特征的投影绝对均值
        feature_proj = {}
        for fname in fnames:
            mask = [i for i, l in enumerate(diff_labels) if l == fname]
            if mask:
                feature_proj[fname] = abs(pc_coords[mask, pc].mean())
        dominant = max(feature_proj, key=feature_proj.get) if feature_proj else "N/A"
        print(f"  PC{pc}: {dominant} (proj={feature_proj.get(dominant, 0):.3f})")
    
    results = {
        'n_diffs': n_diffs,
        'n_components': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'feature_pc_means': feature_pc_means,
        'diff_labels': diff_labels,
    }
    
    # 保存PCA components
    results['pca_components_shape'] = list(pca.components_.shape)
    
    # 前20个PC的方差解释率
    n_top = min(20, len(pca.explained_variance_ratio_))
    top_var = pca.explained_variance_ratio_[:n_top]
    print(f"\n  前{n_top}个PC方差解释率: sum={sum(top_var):.4f}")
    print(f"  → 因果空间的有效维度: ~{np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1} (90%方差)")
    
    # 如果有效维度小 → 因果空间是低秩的 → SAE可行
    effective_dim_90 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1)
    effective_dim_95 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1)
    effective_dim_99 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.99) + 1)
    
    print(f"  → 90%/95%/99%方差需要: {effective_dim_90}/{effective_dim_95}/{effective_dim_99} 维")
    
    results['effective_dim_90'] = effective_dim_90
    results['effective_dim_95'] = effective_dim_95
    results['effective_dim_99'] = effective_dim_99
    
    return results


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    model, tokenizer, device = load_model_fast(model_name)
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    
    r1 = test_r1(model_name, model, tokenizer, device, d_model, n_layers)
    r2 = test_r2(model_name, model, tokenizer, device, d_model, n_layers)
    r3 = test_r3(model_name, model, tokenizer, device, d_model, n_layers)
    
    result_dir = Path(f'results/causal_fiber/{model_name}_rigorous')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: convert_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    for name, data in [('r1', r1), ('r2', r2), ('r3', r3)]:
        with open(result_dir / f'{name}_results.json', 'w') as f:
            json.dump(convert_keys(data), f, indent=2, default=str)
        print(f'{name} saved!')
    
    del model
    import torch
    torch.cuda.empty_cache()
    gc.collect()
