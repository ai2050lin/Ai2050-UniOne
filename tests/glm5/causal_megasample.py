"""
Phase CCIII: 大样本因果追踪 + Head Hook定位 + 归一化因果PCA
================================================================

Phase CCII关键硬伤:
  1. R3的PCA只有38个差分向量(5特征×10组/20对一组) → 有效维度估计不可靠
  2. R2对4bit模型失败(DS7B/GLM4的W_o异常无法分离head)
  3. 因果空间"坍缩到1维"可能只是大方差特征主导 → 需要归一化

核心改进:
  S1: 大样本因果PCA — 分组5对/组 → 每特征40组×5=200个差分向量
      加上每个对单独差分 → 更多向量
  S2: Head Hook定位 — 直接在attn内部hook每个head的输出, 绕过W_o
  S3: 归一化因果PCA — 对每个差分向量先归一化再PCA, 排除范数差异影响
  S4: 交叉验证 — 用bootstrap重采样估计有效维度的置信区间

运行:
  逐模型运行, 测试完一个后再测试另外一个, 避免GPU内存溢出
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json, gc, time
import numpy as np
import torch
from pathlib import Path
from itertools import combinations


# ============================================================
# 数据集 (复用causal_rigorous_so_n.py的数据生成)
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
# S1: 大样本因果PCA — 5对/组 + 逐对差分
# ============================================================

def test_s1(model_name, model, tokenizer, device, d_model, n_layers):
    """
    S1: 大样本因果PCA
    
    改进:
    1. 小分组(5对/组) → 每特征40个差分向量 → 总共约200个
    2. 逐对差分 → 每特征200个差分向量 → 总共约950个!
    3. Bootstrap交叉验证有效维度的置信区间
    """
    print("\n" + "=" * 70)
    print("S1: 大样本因果PCA (5对/组 + 逐对差分 + Bootstrap)")
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
    
    layer = model.model.layers[last_layer]
    
    # ===== Phase A: 逐对差分(最大样本) =====
    print("\n  === Phase A: 逐对差分(最大样本) ===")
    
    all_pair_diffs = []      # 逐对差分向量(归一化前)
    all_pair_diffs_n = []    # 逐对差分向量(归一化后)
    all_pair_labels = []
    
    for fname in fnames:
        pairs = feature_generators[fname]()
        texts_a = [a for a, b in pairs]
        texts_b = [b for a, b in pairs]
        
        captured = []
        def make_hook(storage):
            def hook_fn(module, input, output):
                if isinstance(output, tuple): h = output[0].detach().cpu().float()
                else: h = output.detach().cpu().float()
                storage.append(h[0])
            return hook_fn
        
        h = layer.register_forward_hook(make_hook(captured))
        
        reprs_a = []
        reprs_b = []
        
        for i, text in enumerate(texts_a):
            captured.clear()
            toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
            with torch.no_grad():
                try: _ = model(**toks)
                except: continue
            if captured:
                reprs_a.append(captured[0].numpy()[-1])
        
        for i, text in enumerate(texts_b):
            captured.clear()
            toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
            with torch.no_grad():
                try: _ = model(**toks)
                except: continue
            if captured:
                reprs_b.append(captured[0].numpy()[-1])
        
        h.remove()
        
        # 逐对差分(配对)
        n_pairs = min(len(reprs_a), len(reprs_b))
        for i in range(n_pairs):
            diff = reprs_b[i] - reprs_a[i]
            norm = np.linalg.norm(diff)
            all_pair_diffs.append(diff)
            if norm > 1e-10:
                all_pair_diffs_n.append(diff / norm)
            else:
                all_pair_diffs_n.append(np.zeros(d_model))
            all_pair_labels.append(fname)
        
        print(f"  {fname:>10}: {n_pairs} pair diffs collected")
    
    n_total = len(all_pair_diffs)
    print(f"\n  总逐对差分向量数: {n_total}")
    
    # ===== PCA on 原始差分 =====
    from sklearn.decomposition import PCA
    
    diff_matrix = np.array(all_pair_diffs)  # [n_total, d_model]
    diff_mean = diff_matrix.mean(axis=0)
    diff_centered = diff_matrix - diff_mean
    
    n_components = min(50, n_total - 1, d_model)
    pca_raw = PCA(n_components=n_components)
    pca_raw.fit(diff_centered)
    
    eff_dim_90_raw = int(np.searchsorted(np.cumsum(pca_raw.explained_variance_ratio_), 0.9) + 1)
    eff_dim_95_raw = int(np.searchsorted(np.cumsum(pca_raw.explained_variance_ratio_), 0.95) + 1)
    eff_dim_99_raw = int(np.searchsorted(np.cumsum(pca_raw.explained_variance_ratio_), 0.99) + 1)
    
    print(f"\n  原始差分PCA:")
    print(f"  前10方差解释率: {[f'{v:.4f}' for v in pca_raw.explained_variance_ratio_[:10]]}")
    print(f"  有效维度(90%/95%/99%): {eff_dim_90_raw}/{eff_dim_95_raw}/{eff_dim_99_raw}")
    
    # ===== PCA on 归一化差分 =====
    print("\n  === Phase B: 归一化差分PCA ===")
    
    diff_n_matrix = np.array(all_pair_diffs_n)
    diff_n_mean = diff_n_matrix.mean(axis=0)
    diff_n_centered = diff_n_matrix - diff_n_mean
    
    pca_norm = PCA(n_components=n_components)
    pca_norm.fit(diff_n_centered)
    
    eff_dim_90_norm = int(np.searchsorted(np.cumsum(pca_norm.explained_variance_ratio_), 0.9) + 1)
    eff_dim_95_norm = int(np.searchsorted(np.cumsum(pca_norm.explained_variance_ratio_), 0.95) + 1)
    eff_dim_99_norm = int(np.searchsorted(np.cumsum(pca_norm.explained_variance_ratio_), 0.99) + 1)
    
    print(f"  归一化差分PCA:")
    print(f"  前10方差解释率: {[f'{v:.4f}' for v in pca_norm.explained_variance_ratio_[:10]]}")
    print(f"  有效维度(90%/95%/99%): {eff_dim_90_norm}/{eff_dim_95_norm}/{eff_dim_99_norm}")
    
    # ===== Bootstrap交叉验证 =====
    print("\n  === Phase C: Bootstrap有效维度估计 ===")
    
    n_bootstrap = 50
    bootstrap_dims_90 = []
    bootstrap_dims_95 = []
    
    rng = np.random.RandomState(42)
    for b in range(n_bootstrap):
        # 重采样(有放回)
        idx = rng.choice(n_total, size=n_total, replace=True)
        boot_data = diff_centered[idx]
        
        pca_boot = PCA(n_components=min(30, n_total - 1))
        pca_boot.fit(boot_data)
        
        d90 = int(np.searchsorted(np.cumsum(pca_boot.explained_variance_ratio_), 0.9) + 1)
        d95 = int(np.searchsorted(np.cumsum(pca_boot.explained_variance_ratio_), 0.95) + 1)
        bootstrap_dims_90.append(d90)
        bootstrap_dims_95.append(d95)
    
    print(f"  Bootstrap有效维度(90%): mean={np.mean(bootstrap_dims_90):.1f}, "
          f"std={np.std(bootstrap_dims_90):.1f}, "
          f"CI=[{np.percentile(bootstrap_dims_90, 2.5):.0f}, {np.percentile(bootstrap_dims_90, 97.5):.0f}]")
    print(f"  Bootstrap有效维度(95%): mean={np.mean(bootstrap_dims_95):.1f}, "
          f"std={np.std(bootstrap_dims_95):.1f}, "
          f"CI=[{np.percentile(bootstrap_dims_95, 2.5):.0f}, {np.percentile(bootstrap_dims_95, 97.5):.0f}]")
    
    # ===== 每个PC被哪个特征主导 =====
    pc_coords_raw = pca_raw.transform(diff_centered)
    
    print(f"\n  每个PC上哪个特征主导:")
    pc_dominance = []
    for pc in range(min(10, n_components)):
        feature_proj = {}
        for fname in fnames:
            mask = [i for i, l in enumerate(all_pair_labels) if l == fname]
            if mask:
                feature_proj[fname] = abs(pc_coords_raw[mask, pc].mean())
        dominant = max(feature_proj, key=feature_proj.get) if feature_proj else "N/A"
        score = feature_proj.get(dominant, 0)
        pc_dominance.append({'pc': pc, 'dominant': dominant, 'score': float(score)})
        print(f"  PC{pc}: {dominant} (proj={score:.3f})")
    
    # ===== 特征间的PC子空间分离度 =====
    print(f"\n  特征间的PC子空间分离度:")
    for f1 in fnames:
        for f2 in fnames:
            if f1 >= f2: continue
            mask1 = [i for i, l in enumerate(all_pair_labels) if l == f1]
            mask2 = [i for i, l in enumerate(all_pair_labels) if l == f2]
            if mask1 and mask2:
                # 在前5个PC上的余弦距离
                c1 = pc_coords_raw[mask1, :5].mean(axis=0)
                c2 = pc_coords_raw[mask2, :5].mean(axis=0)
                n1 = np.linalg.norm(c1)
                n2 = np.linalg.norm(c2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos = float(np.dot(c1, c2) / (n1 * n2))
                else:
                    cos = 0.0
                print(f"  {f1:>10} vs {f2:>10}: PC-cos={cos:+.4f}")
    
    results = {
        'n_total_diffs': n_total,
        'raw_pca': {
            'eff_dim_90': eff_dim_90_raw,
            'eff_dim_95': eff_dim_95_raw,
            'eff_dim_99': eff_dim_99_raw,
            'top10_var': pca_raw.explained_variance_ratio_[:10].tolist(),
            'cumulative_var_top20': np.cumsum(pca_raw.explained_variance_ratio_[:20]).tolist(),
        },
        'norm_pca': {
            'eff_dim_90': eff_dim_90_norm,
            'eff_dim_95': eff_dim_95_norm,
            'eff_dim_99': eff_dim_99_norm,
            'top10_var': pca_norm.explained_variance_ratio_[:10].tolist(),
        },
        'bootstrap': {
            'dim_90_mean': float(np.mean(bootstrap_dims_90)),
            'dim_90_std': float(np.std(bootstrap_dims_90)),
            'dim_90_ci': [float(np.percentile(bootstrap_dims_90, 2.5)), float(np.percentile(bootstrap_dims_90, 97.5))],
            'dim_95_mean': float(np.mean(bootstrap_dims_95)),
            'dim_95_std': float(np.std(bootstrap_dims_95)),
            'dim_95_ci': [float(np.percentile(bootstrap_dims_95, 2.5)), float(np.percentile(bootstrap_dims_95, 97.5))],
        },
        'pc_dominance': pc_dominance,
        'diff_labels': all_pair_labels,
    }
    
    return results


# ============================================================
# S2: Head Hook定位 — 直接hook每个head的输出
# ============================================================

def test_s2(model_name, model, tokenizer, device, d_model, n_layers):
    """
    S2: 用hook直接捕获每个head的attn输出, 绕过W_o分解问题
    
    方法:
    1. 在self_attn.o_proj之前注册hook
    2. 获取concat_heads (shape: [n_heads * d_head])
    3. 分离每个head的输出: head_h = concat_heads[h*d_head:(h+1)*d_head]
    4. 对每个head, 计算极性和时态差分在head输出中的cos对齐
    """
    print("\n" + "=" * 70)
    print("S2: Head Hook因果定位 (直接hook每个head)")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()[:200]
    tense_pairs = generate_tense_pairs()[:200]
    
    # 获取架构信息
    layer0 = model.model.layers[0]
    n_heads = layer0.self_attn.config.num_attention_heads
    if hasattr(layer0.self_attn.config, 'head_dim') and layer0.self_attn.config.head_dim is not None:
        d_head = layer0.self_attn.config.head_dim
    else:
        d_head = d_model // n_heads
    
    # KV heads (GQA)
    n_kv_heads = n_heads
    if hasattr(layer0.self_attn.config, 'num_key_value_heads'):
        n_kv_heads = layer0.self_attn.config.num_key_value_heads
    
    print(f"  n_heads={n_heads}, d_head={d_head}, n_kv_heads={n_kv_heads}")
    
    # 选择关键层
    if model_name == 'deepseek7b':
        target_layers = [n_layers - 1, n_layers - 2, n_layers - 3, n_layers // 2]
    elif model_name == 'glm4':
        target_layers = [0, n_layers - 1, n_layers - 2, n_layers // 2]
    else:
        target_layers = [n_layers - 1, n_layers - 2, n_layers // 2]
    
    print(f"  target_layers={target_layers}")
    
    results = {'n_heads': n_heads, 'd_head': d_head, 'n_kv_heads': n_kv_heads, 
               'target_layers': target_layers}
    
    for li in target_layers:
        print(f"\n  --- L{li} ---")
        layer = model.model.layers[li]
        
        # Hook策略: 在o_proj的input处捕获concat_heads
        # o_proj接收 [batch, seq, n_heads*d_head] 的输入
        concat_captured = []
        
        def make_concat_hook(storage):
            def hook_fn(module, input, kwargs):
                # input[0] = concat_heads [batch, seq, n_heads*d_head]
                if isinstance(input, tuple) and len(input) > 0:
                    h = input[0].detach().cpu().float()
                    storage.append(h[0])  # [seq, n_heads*d_head]
                return None  # 不修改输出
            return hook_fn
        
        h_concat = layer.self_attn.o_proj.register_forward_hook(
            make_concat_hook(concat_captured), with_kwargs=True
        )
        
        # 提取每个head对极性/时态的差分
        def get_head_diffs(pairs, label):
            texts_a = [a for a, b in pairs]
            texts_b = [b for a, b in pairs]
            all_texts = texts_a + texts_b
            n_half = len(texts_a)
            
            head_reprs_a = {h: [] for h in range(n_heads)}
            head_reprs_b = {h: [] for h in range(n_heads)}
            
            for i, text in enumerate(all_texts):
                concat_captured.clear()
                toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    try: _ = model(**toks)
                    except: continue
                
                if concat_captured:
                    concat = concat_captured[0].numpy()  # [seq, n_heads*d_head]
                    last_token = concat[-1]  # [n_heads*d_head]
                    
                    # 分离每个head
                    for h in range(n_heads):
                        start = h * d_head
                        end = (h + 1) * d_head
                        head_vec = last_token[start:end]  # [d_head]
                        if i < n_half:
                            head_reprs_a[h].append(head_vec)
                        else:
                            head_reprs_b[h].append(head_vec)
            
            # 每个head的差分方向
            head_diffs = {}
            for h in range(n_heads):
                if len(head_reprs_a[h]) > 0 and len(head_reprs_b[h]) > 0:
                    diff = np.mean(head_reprs_b[h], axis=0) - np.mean(head_reprs_a[h], axis=0)
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        head_diffs[h] = {'diff': diff, 'diff_n': diff / norm, 'norm': float(norm),
                                         'n_a': len(head_reprs_a[h]), 'n_b': len(head_reprs_b[h])}
                    else:
                        head_diffs[h] = {'diff': diff, 'diff_n': np.zeros(d_head), 'norm': 0.0,
                                         'n_a': len(head_reprs_a[h]), 'n_b': len(head_reprs_b[h])}
            
            return head_diffs
        
        print(f"  提取极性head输出...")
        pol_head_diffs = get_head_diffs(pol_pairs, "polarity")
        print(f"  提取时态head输出...")
        tense_head_diffs = get_head_diffs(tense_pairs, "tense")
        
        h_concat.remove()
        
        # 计算每个head的因果对齐cos
        head_results = []
        for h in range(n_heads):
            if h in pol_head_diffs and h in tense_head_diffs:
                pol_n = pol_head_diffs[h]['diff_n']
                tense_n = tense_head_diffs[h]['diff_n']
                pol_norm = pol_head_diffs[h]['norm']
                tense_norm = tense_head_diffs[h]['norm']
                
                head_cos = float(np.dot(pol_n, tense_n))
                
                head_results.append({
                    'head': h,
                    'head_cos': head_cos,
                    'pol_norm': pol_norm,
                    'tense_norm': tense_norm,
                    'pol_n': len(pol_head_diffs[h].get('n_a', 0)),
                    'tense_n': len(tense_head_diffs[h].get('n_a', 0)),
                })
        
        # 按cos绝对值排序
        head_results.sort(key=lambda x: abs(x['head_cos']), reverse=True)
        
        print(f"\n  {'Head':>6} {'cos':>8} {'pol_norm':>10} {'tense_norm':>11}")
        print("  " + "-" * 45)
        for hr in head_results[:15]:
            mark = "★★★" if abs(hr['head_cos']) > 0.5 else ("★★" if abs(hr['head_cos']) > 0.3 else "")
            print(f"  h{hr['head']:>4} {hr['head_cos']:>+8.4f} {hr['pol_norm']:>10.2f} {hr['tense_norm']:>11.2f} {mark}")
        
        # 统计
        high_cos = [hr for hr in head_results if abs(hr['head_cos']) > 0.5]
        med_cos = [hr for hr in head_results if 0.3 < abs(hr['head_cos']) <= 0.5]
        low_cos = [hr for hr in head_results if abs(hr['head_cos']) <= 0.3]
        
        print(f"\n  统计: ★★★(>0.5)={len(high_cos)}个, ★★(0.3-0.5)={len(med_cos)}个, ≤0.3={len(low_cos)}个")
        
        # 整体cos (所有head的差分加起来)
        total_pol = np.zeros(d_model)
        total_tense = np.zeros(d_model)
        
        # 需要重新计算: 用attn_output的整体差分
        # 简化: 用head差分的加权平均
        for h in range(n_heads):
            if h in pol_head_diffs:
                # 将head_diff投影回d_model空间(近似)
                total_pol += pol_head_diffs[h]['norm']
                total_tense += tense_head_diffs[h]['norm']
        
        results[f'L{li}'] = {
            'head_results': head_results[:32],
            'n_high_cos': len(high_cos),
            'n_med_cos': len(med_cos),
            'n_low_cos': len(low_cos),
        }
    
    return results


# ============================================================
# S3: 归一化Gram矩阵分析
# ============================================================

def test_s3(model_name, model, tokenizer, device, d_model, n_layers):
    """
    S3: 归一化Gram矩阵分析 — 排除范数差异对Gram的影响
    
    方法: 对每个特征的差分向量先归一化, 再计算Gram矩阵
    如果归一化后Gram仍然变化 → 因果方向的内积关系确实在变化
    如果归一化后Gram接近保持 → 之前的Gram变化主要来自范数差异
    """
    print("\n" + "=" * 70)
    print("S3: 归一化Gram矩阵分析 (200对/特征)")
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
    
    # 采样层
    sample_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]))
    if n_layers > 10:
        extra = list(range(0, n_layers, max(1, n_layers//8)))
        sample_layers = sorted(set(sample_layers + extra))
    
    print(f"  采样层: {sample_layers}")
    
    layer_gram_norm = {}
    layer_diff_norms = {}
    
    for li in sample_layers:
        diff_vectors = {}
        diff_norms = {}
        
        for fname in fnames:
            pairs = feature_generators[fname]()[:200]
            texts_a = [a for a, b in pairs]
            texts_b = [b for a, b in pairs]
            all_texts = texts_a + texts_b
            n_half = len(texts_a)
            
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
                    h_vec = captured[0].numpy()[-1]
                    if i < n_half:
                        reprs_a.append(h_vec)
                    else:
                        reprs_b.append(h_vec)
            
            h.remove()
            
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
            
            diff_vectors[fname] = diff_n
            diff_norms[fname] = float(norm)
            print(f"  L{li} {fname:>10}: norm={norm:.2f}")
        
        layer_diff_norms[li] = diff_norms
        
        # 归一化Gram矩阵 (diff_n都是单位向量, 对角线=1)
        gram = np.zeros((n_features, n_features))
        for i, f1 in enumerate(fnames):
            for j, f2 in enumerate(fnames):
                gram[i, j] = float(np.dot(diff_vectors[f1], diff_vectors[f2]))
        layer_gram_norm[li] = gram
    
    # 分析归一化Gram变化
    print(f"\n  {'Layer':>6} {'Gram det':>10} {'trace':>8} {'与L0差':>10} {'rel_diff':>10}")
    print("  " + "-" * 50)
    
    gram_L0 = layer_gram_norm[sample_layers[0]]
    norm_drifts = []
    
    for li in sample_layers:
        gram = layer_gram_norm[li]
        det = np.linalg.det(gram)
        trace = np.trace(gram)
        diff = np.linalg.norm(gram - gram_L0, 'fro')
        base = np.linalg.norm(gram_L0, 'fro')
        rel = diff / max(base, 1e-10)
        
        print(f"  L{li:>4} {det:>10.4f} {trace:>8.4f} {diff:>10.4f} {rel:>9.2%}")
        norm_drifts.append({'layer': li, 'det': float(det), 'trace': float(trace),
                            'frob_diff_L0': float(diff), 'rel_diff_L0': float(rel)})
    
    # 特征值分析
    print(f"\n  归一化Gram特征值分布:")
    for li in [sample_layers[0], sample_layers[len(sample_layers)//2], sample_layers[-1]]:
        if li in layer_gram_norm:
            eigs = np.sort(np.linalg.eigvalsh(layer_gram_norm[li]))[::-1]
            print(f"  L{li:>4}: {[f'{e:.3f}' for e in eigs]}")
    
    # 相邻层Gram差异
    adj_diffs = []
    for i in range(len(sample_layers) - 1):
        li1, li2 = sample_layers[i], sample_layers[i+1]
        g1, g2 = layer_gram_norm[li1], layer_gram_norm[li2]
        diff = np.linalg.norm(g2 - g1, 'fro')
        base = np.linalg.norm(g1, 'fro')
        rel = diff / max(base, 1e-10)
        adj_diffs.append({'from': li1, 'to': li2, 'rel_diff': float(rel)})
    
    mean_rel = np.mean([d['rel_diff'] for d in adj_diffs])
    max_rel = np.max([d['rel_diff'] for d in adj_diffs])
    print(f"\n  相邻层归一化Gram差异: mean={mean_rel:.2%}, max={max_rel:.2%}")
    
    # 对比: L26→L27是否出现突变(DS7B)?
    last_adj = adj_diffs[-1]
    print(f"  最后一跳: L{last_adj['from']}→L{last_adj['to']}, rel_diff={last_adj['rel_diff']:.2%}")
    
    results = {
        'sample_layers': sample_layers,
        'norm_drifts': norm_drifts,
        'adj_gram_diffs': adj_diffs,
        'mean_rel_diff': float(mean_rel),
        'max_rel_diff': float(max_rel),
        'diff_norms': {str(li): layer_diff_norms[li] for li in sample_layers},
    }
    
    # 保存所有Gram矩阵
    results['all_gram_norm'] = {str(li): layer_gram_norm[li].tolist() for li in sample_layers}
    
    return results


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test", default="all", choices=["s1", "s2", "s3", "all"])
    args = parser.parse_args()
    
    model_name = args.model
    model, tokenizer, device = load_model_fast(model_name)
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    
    print(f"\n{'='*70}")
    print(f"Model: {model_name}, d_model={d_model}, n_layers={n_layers}")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    result_dir = Path(f'results/causal_fiber/{model_name}_megasample')
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
    
    if args.test in ['s1', 'all']:
        t0 = time.time()
        s1 = test_s1(model_name, model, tokenizer, device, d_model, n_layers)
        with open(result_dir / 's1_results.json', 'w') as f:
            json.dump(convert_keys(s1), f, indent=2, default=str)
        print(f"\nS1 saved! ({time.time()-t0:.0f}s)")
    
    if args.test in ['s2', 'all']:
        t0 = time.time()
        s2 = test_s2(model_name, model, tokenizer, device, d_model, n_layers)
        with open(result_dir / 's2_results.json', 'w') as f:
            json.dump(convert_keys(s2), f, indent=2, default=str)
        print(f"\nS2 saved! ({time.time()-t0:.0f}s)")
    
    if args.test in ['s3', 'all']:
        t0 = time.time()
        s3 = test_s3(model_name, model, tokenizer, device, d_model, n_layers)
        with open(result_dir / 's3_results.json', 'w') as f:
            json.dump(convert_keys(s3), f, indent=2, default=str)
        print(f"\nS3 saved! ({time.time()-t0:.0f}s)")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("\nDone!")
