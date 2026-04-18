"""
Phase CCI: Attention因果汇聚直接验证 + SO(n)旋转 + 非功能词特征
================================================================

Phase CC核心发现:
  1. DS7B L27 Attention主导因果汇聚(94.7%, cos=+0.927)
  2. DS7B L27因果对齐显著(p=0.042), GLM4 L0极显著(p=0.000)
  3. 但M3用的是切向/法向分解近似, 需要直接hook验证

核心测试:
  H1: 直接hook捕获Attn/MLP输出 — 验证Attn是否直接产生因果汇聚
      大样本: 150对/特征, 捕获L_last的attn_out和mlp_out
  H2: 连续旋转验证 — 相邻层因果方向是否构成SO(n)旋转
      分析: 旋转矩阵是否正交? 旋转角是否连续变化?
  H3: 非功能词特征的因果对齐 — 语义/情感是否也有因果汇聚
      排除功能词共享假说

运行:
  python -c "import sys; sys.path.insert(0,'tests/glm5'); ..."
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
    """200对语义对(具体vs抽象) — 不涉及功能词!"""
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
    """200对情感对 — 不涉及功能词!"""
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
    """150对数量对(单数vs复数) — 不涉及not/do等否定/时态功能词!"""
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
        # 单数: The cat is tall / 复数: The cats are tall
        for adj in adjectives[idx % len(adjectives):idx % len(adjectives) + 2]:
            sg = f"The {noun} {adj}"
            pl_noun = noun + "s" if not noun.endswith("s") else noun + "es"
            pl_adj = adj.replace("is ", "are ")
            pl = f"The {pl_noun} {pl_adj}"
            pairs.append((sg, pl))
            idx += 1
            if len(pairs) >= 150: return pairs[:150]
    
    # 额外: 动词单复数
    for noun in nouns_sg[:30]:
        for verb in verbs_pres[idx % len(verbs_pres):idx % len(verbs_pres) + 2]:
            sg = f"The {noun} {verb} well"
            pl_noun = noun + "s" if not noun.endswith("s") else noun + "es"
            # 单数动词没有s, 复数动词有s (反过来!)
            verb_no_s = verb[:-1] if verb.endswith("s") else verb
            sg2 = f"The {pl_noun} {verb} well"  # 复数主语+动词s(错误)
            pl2 = f"The {pl_noun} {verb_no_s} well"  # 复数主语+动词无s(正确)
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
# H1: 直接hook验证Attention因果汇聚
# ============================================================

def test_h1(model_name, model, tokenizer, device, d_model, n_layers):
    """H1: 用hook直接捕获Attn/MLP输出, 验证Attention产生因果汇聚"""
    print("\n" + "=" * 70)
    print("H1: 直接hook验证Attention因果汇聚 (150对/特征)")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()[:150]
    tense_pairs = generate_tense_pairs()[:150]
    
    last_layer = n_layers - 1
    
    # 提取极性对在最后一层的attn_out和mlp_out
    def get_attn_mlp_diffs(pairs, label_name):
        texts_a = [a for a, b in pairs]
        texts_b = [b for a, b in pairs]
        all_texts = texts_a + texts_b
        n_half = len(texts_a)
        
        attn_outs = []
        mlp_outs = []
        resid_outs = []
        
        layer = model.model.layers[last_layer]
        
        def make_attn_hook(storage):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    storage.append(output[0].detach().cpu().float().numpy())
                else:
                    storage.append(output.detach().cpu().float().numpy())
            return hook_fn
        
        def make_mlp_hook(storage):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    storage.append(output[0].detach().cpu().float().numpy())
                else:
                    storage.append(output.detach().cpu().float().numpy())
            return hook_fn
        
        def make_resid_hook(storage):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    storage.append(output[0].detach().cpu().float().numpy())
                else:
                    storage.append(output.detach().cpu().float().numpy())
            return hook_fn
        
        h_attn = layer.self_attn.register_forward_hook(make_attn_hook(attn_outs))
        h_mlp = layer.mlp.register_forward_hook(make_mlp_hook(mlp_outs))
        h_resid = layer.register_forward_hook(make_resid_hook(resid_outs))
        
        reprs_a = {'attn': [], 'mlp': [], 'resid': []}
        reprs_b = {'attn': [], 'mlp': [], 'resid': []}
        
        for i, text in enumerate(all_texts):
            attn_outs.clear()
            mlp_outs.clear()
            resid_outs.clear()
            
            toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
            with torch.no_grad():
                try: _ = model(**toks)
                except: continue
            
            if attn_outs and mlp_outs and resid_outs:
                # 取last token
                a_out = attn_outs[0][0, -1, :]  # [d_model]
                m_out = mlp_outs[0][0, -1, :]
                r_out = resid_outs[0][0, -1, :]
                
                if i < n_half:
                    reprs_a['attn'].append(a_out)
                    reprs_a['mlp'].append(m_out)
                    reprs_a['resid'].append(r_out)
                else:
                    reprs_b['attn'].append(a_out)
                    reprs_b['mlp'].append(m_out)
                    reprs_b['resid'].append(r_out)
        
        h_attn.remove()
        h_mlp.remove()
        h_resid.remove()
        
        # 计算差分方向
        results = {}
        for key in ['attn', 'mlp', 'resid']:
            arr_a = np.array(reprs_a[key])
            arr_b = np.array(reprs_b[key])
            if len(arr_a) > 0 and len(arr_b) > 0:
                diff = arr_b.mean(axis=0) - arr_a.mean(axis=0)
                norm = np.linalg.norm(diff)
                if norm > 1e-10:
                    diff_n = diff / norm
                else:
                    diff_n = np.zeros(d_model)
                results[key] = {'diff': diff, 'diff_n': diff_n, 'norm': float(norm),
                               'n_a': len(arr_a), 'n_b': len(arr_b)}
            else:
                results[key] = {'diff': np.zeros(d_model), 'diff_n': np.zeros(d_model), 'norm': 0.0,
                               'n_a': 0, 'n_b': 0}
        
        return results
    
    print(f"  提取极性对 L{last_layer} 的Attn/MLP/Resid输出...")
    pol_diffs = get_attn_mlp_diffs(pol_pairs, "polarity")
    
    print(f"  提取时态对 L{last_layer} 的Attn/MLP/Resid输出...")
    tense_diffs = get_attn_mlp_diffs(tense_pairs, "tense")
    
    # 计算交叉余弦
    results = {'last_layer': last_layer, 'n_pairs': 150}
    
    print(f"\n  {'来源':>10} {'pol_norm':>10} {'tense_norm':>11} {'cross_cos':>10} {'对齐?':>8}")
    print("  " + "-" * 55)
    
    for key in ['attn', 'mlp', 'resid']:
        pol_n = pol_diffs[key]['diff_n']
        tense_n = tense_diffs[key]['diff_n']
        cos = float(np.dot(pol_n, tense_n))
        pol_norm = pol_diffs[key]['norm']
        tense_norm = tense_diffs[key]['norm']
        
        align = "★★★" if abs(cos) > 0.5 else ("★★" if abs(cos) > 0.3 else ("★" if abs(cos) > 0.1 else ""))
        
        print(f"  {key:>10} {pol_norm:>10.2f} {tense_norm:>11.2f} {cos:>+10.4f} {align:>8}")
        
        results[key] = {
            'pol_norm': float(pol_norm),
            'tense_norm': float(tense_norm),
            'cross_cos': cos,
            'pol_n_a': pol_diffs[key]['n_a'],
            'pol_n_b': pol_diffs[key]['n_b'],
        }
    
    # 关键分析: Attn的因果对齐是否强于MLP?
    attn_cos = abs(results['attn']['cross_cos'])
    mlp_cos = abs(results['mlp']['cross_cos'])
    resid_cos = abs(results['resid']['cross_cos'])
    
    print(f"\n  |cos|对比: Attn={attn_cos:.4f}, MLP={mlp_cos:.4f}, Resid={resid_cos:.4f}")
    
    if attn_cos > mlp_cos * 1.5 and attn_cos > 0.3:
        print(f"  → ★★★ Attention直接产生因果汇聚! ★★★")
    elif mlp_cos > attn_cos * 1.5 and mlp_cos > 0.3:
        print(f"  → ★★★ MLP直接产生因果汇聚! ★★★")
    elif attn_cos > 0.3 and mlp_cos > 0.3:
        print(f"  → Attn和MLP共同产生因果汇聚")
    else:
        print(f"  → 因果汇聚在单个子模块中不明显")
    
    # 验证: resid_diff ≈ attn_diff + mlp_diff?
    resid_diff = pol_diffs['resid']['diff']
    attn_diff = pol_diffs['attn']['diff']
    mlp_diff = pol_diffs['mlp']['diff']
    combined = attn_diff + mlp_diff
    combined_norm = np.linalg.norm(combined)
    resid_norm = np.linalg.norm(resid_diff)
    if combined_norm > 1e-10 and resid_norm > 1e-10:
        reconstruction_cos = float(np.dot(combined / combined_norm, resid_diff / resid_norm))
        print(f"\n  残差流重构: attn+mlp与resid的cos={reconstruction_cos:.4f}")
        results['reconstruction_cos'] = reconstruction_cos
    
    return results


# ============================================================
# H2: 连续旋转验证 — SO(n)旋转结构
# ============================================================

def test_h2(model_name, model, tokenizer, device, d_model, n_layers):
    """H2: 相邻层因果方向是否构成连续旋转(SO(n))?"""
    print("\n" + "=" * 70)
    print("H2: 连续旋转验证 — SO(n)旋转结构 (150对)")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()[:150]
    
    # 采样足够密的层
    n_sample = min(15, n_layers)
    sample_layers = sorted(set([0] + list(range(0, n_layers, max(1, n_layers // n_sample))) + [n_layers - 1]))
    if len(sample_layers) > 20:
        step = len(sample_layers) // 15
        sample_layers = sorted(set(sample_layers[::step] + [0, n_layers - 1]))
    
    print(f"  采样层: {sample_layers}")
    
    # 提取极性表示
    pol_aff = [a for a, b in pol_pairs]
    pol_neg = [b for a, b in pol_pairs]
    all_texts = pol_aff + pol_neg
    n_half = len(pol_aff)
    
    # 用hook提取残差流
    captured = {}
    hooks = []
    layers = model.model.layers
    for li in sample_layers: captured[li] = []
    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple): h = output[0].detach().cpu().float()
            else: h = output.detach().cpu().float()
            captured[li].append(h[0])
        return hook_fn
    for li in sample_layers: hooks.append(layers[li].register_forward_hook(make_hook(li)))
    
    pol_dirs = {}
    for text in all_texts:
        for li in sample_layers: captured[li] = []
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            try: _ = model(**toks)
            except: continue
        for li in sample_layers:
            if captured[li]:
                h = captured[li][0].numpy()
                if li not in pol_dirs: pol_dirs[li] = {'a': [], 'b': []}
                idx_curr = len([t for t in all_texts[:all_texts.index(text)]])
                if idx_curr < n_half:
                    pol_dirs[li]['a'].append(h[-1])
                else:
                    pol_dirs[li]['b'].append(h[-1])
    
    for h in hooks: h.remove()
    
    # 计算每层的因果方向
    dir_list = []
    for li in sample_layers:
        if li in pol_dirs and len(pol_dirs[li]['a']) > 0 and len(pol_dirs[li]['b']) > 0:
            diff = np.mean(pol_dirs[li]['b'], axis=0) - np.mean(pol_dirs[li]['a'], axis=0)
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                dir_list.append((li, diff / norm, norm))
            else:
                dir_list.append((li, np.zeros(d_model), 0.0))
        else:
            dir_list.append((li, np.zeros(d_model), 0.0))
    
    # 分析相邻层旋转
    results = {'sample_layers': sample_layers, 'n_pairs': 150, 'adj_rotations': [], 'rot_matrices': []}
    
    print(f"\n  {'Layer':>6} {'→Next':>6} {'cos':>8} {'angle':>8} {'正交偏差':>10}")
    print("  " + "-" * 45)
    
    for i in range(len(dir_list) - 1):
        li1, d1, n1 = dir_list[i]
        li2, d2, n2 = dir_list[i + 1]
        
        cos_val = float(np.dot(d1, d2))
        angle = float(np.degrees(np.arccos(np.clip(cos_val, -1, 1))))
        
        # 检查旋转是否正交: d2在d1的补空间中的分量
        # 如果旋转正交: d2 = cos(θ)*d1 + sin(θ)*d1_perp
        # 正交偏差 = ||d2 - cos(θ)*d1|| - sin(θ) (应该≈0)
        proj = cos_val * d1
        perp = d2 - proj
        perp_norm = np.linalg.norm(perp)
        sin_angle = np.sin(np.radians(angle))
        ortho_deviation = abs(perp_norm - sin_angle)
        
        print(f"  L{li1:>4} →L{li2:>4} {cos_val:>+8.4f} {angle:>7.1f}° {ortho_deviation:>10.6f}")
        
        results['adj_rotations'].append({
            'from_layer': li1, 'to_layer': li2,
            'cos': cos_val, 'angle': angle,
            'ortho_deviation': float(ortho_deviation),
            'perp_norm': float(perp_norm),
            'sin_angle': float(sin_angle),
        })
    
    # 总结旋转模式
    angles = [r['angle'] for r in results['adj_rotations']]
    ortho_devs = [r['ortho_deviation'] for r in results['adj_rotations']]
    
    print(f"\n  旋转角: mean={np.mean(angles):.1f}°, std={np.std(angles):.1f}°, range=[{np.min(angles):.1f}°, {np.max(angles):.1f}°]")
    print(f"  正交偏差: mean={np.mean(ortho_devs):.6f}, max={np.max(ortho_devs):.6f}")
    
    # 如果正交偏差小 → SO(n)旋转; 如果大 → 非正交变换
    if np.mean(ortho_devs) < 0.01:
        print(f"  → ★★★ 旋转近似正交! 因果纤维丛有SO(n)结构! ★★★")
    elif np.mean(ortho_devs) < 0.05:
        print(f"  → 旋转近似正交, 但有非正交分量")
    else:
        print(f"  → 旋转非正交, 不是简单SO(n)旋转")
    
    # 检查旋转角是否连续变化(联络是否光滑)
    angle_changes = [abs(angles[i+1] - angles[i]) for i in range(len(angles) - 1)]
    if angle_changes:
        print(f"  旋转角变化: mean={np.mean(angle_changes):.1f}°, max={np.max(angle_changes):.1f}°")
        if np.max(angle_changes) > 30:
            print(f"  → 旋转角有突变(>{np.max(angle_changes):.0f}°) → 联络不光滑")
        else:
            print(f"  → 旋转角连续变化 → 联络可能光滑")
    
    results['angle_mean'] = float(np.mean(angles))
    results['angle_std'] = float(np.std(angles))
    results['ortho_dev_mean'] = float(np.mean(ortho_devs))
    results['ortho_dev_max'] = float(np.max(ortho_devs))
    results['angle_change_max'] = float(np.max(angle_changes)) if angle_changes else 0.0
    
    return results


# ============================================================
# H3: 非功能词特征的因果对齐
# ============================================================

def test_h3(model_name, model, tokenizer, device, d_model, n_layers):
    """H3: 非功能词特征(语义/情感/数量)的因果对齐"""
    print("\n" + "=" * 70)
    print("H3: 非功能词特征因果对齐 (150对/特征)")
    print("=" * 70)
    
    # 3种特征: 语义(具体vs抽象), 情感(正面vs负面), 数量(单数vs复数)
    # + 极性(有功能词)和时态(有功能词)作为对照
    feature_generators = {
        'polarity': generate_polarity_pairs,    # 有功能词(not/do)
        'tense': generate_tense_pairs,          # 有功能词(-ed/was)
        'semantic': generate_semantic_pairs,     # 无功能词!
        'sentiment': generate_sentiment_pairs,   # 无功能词(wonderful/terrible)
        'number': generate_number_pairs,         # 无功能词(s/plural)
    }
    
    last_layer = n_layers - 1
    
    # 提取所有特征的表示
    feature_reprs = {}
    for fname, gen_fn in feature_generators.items():
        pairs = gen_fn()[:150]
        texts_a = [a for a, b in pairs]
        texts_b = [b for a, b in pairs]
        all_texts = texts_a + texts_b
        n_half = len(texts_a)
        
        # 用hook提取残差流
        captured = []
        hooks = []
        layers = model.model.layers
        
        def make_hook(storage):
            def hook_fn(module, input, output):
                if isinstance(output, tuple): h = output[0].detach().cpu().float()
                else: h = output.detach().cpu().float()
                storage.append(h[0])
            return hook_fn
        
        h = layers[last_layer].register_forward_hook(make_hook(captured))
        
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
        
        # 差分方向
        if len(reprs_a) > 0 and len(reprs_b) > 0:
            diff = np.mean(reprs_b, axis=0) - np.mean(reprs_a, axis=0)
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                diff_n = diff / norm
            else:
                diff_n = np.zeros(d_model)
        else:
            diff = np.zeros(d_model)
            diff_n = np.zeros(d_model)
            norm = 0.0
        
        feature_reprs[fname] = {'diff_n': diff_n, 'norm': float(norm),
                                 'n_a': len(reprs_a), 'n_b': len(reprs_b)}
        
        print(f"  {fname:>10}: n={len(reprs_a)}/{len(reprs_b)}, norm={norm:.2f}")
    
    # 计算5×5因果余弦矩阵
    fnames = list(feature_generators.keys())
    n_features = len(fnames)
    causal_matrix = np.zeros((n_features, n_features))
    
    print(f"\n  L{last_layer} 因果余弦矩阵:")
    print(f"       ", end="")
    for fn in fnames: print(f"{fn[:7]:>8}", end="")
    print()
    
    for i, f1 in enumerate(fnames):
        print(f"  {f1[:7]:>7}", end="")
        for j, f2 in enumerate(fnames):
            cos_val = float(np.dot(feature_reprs[f1]['diff_n'], feature_reprs[f2]['diff_n']))
            causal_matrix[i, j] = cos_val
            print(f"{cos_val:+7.3f}", end="")
        print()
    
    # 分析: 功能词特征vs非功能词特征的因果对齐
    func_word_features = ['polarity', 'tense']  # 有功能词
    non_func_features = ['semantic', 'sentiment', 'number']  # 无功能词
    
    # 功能词之间的因果对齐
    func_func_cos = []
    for i, f1 in enumerate(func_word_features):
        for j, f2 in enumerate(func_word_features):
            if i < j:
                func_func_cos.append(abs(causal_matrix[fnames.index(f1), fnames.index(f2)]))
    
    # 非功能词之间的因果对齐
    nonfunc_nonfunc_cos = []
    for i, f1 in enumerate(non_func_features):
        for j, f2 in enumerate(non_func_features):
            if i < j:
                nonfunc_nonfunc_cos.append(abs(causal_matrix[fnames.index(f1), fnames.index(f2)]))
    
    # 功能词与非功能词之间的因果对齐
    func_nonfunc_cos = []
    for f1 in func_word_features:
        for f2 in non_func_features:
            func_nonfunc_cos.append(abs(causal_matrix[fnames.index(f1), fnames.index(f2)]))
    
    print(f"\n  因果对齐分析:")
    print(f"  功能词∩功能词 |cos|平均: {np.mean(func_func_cos):.4f}")
    print(f"  非功能词∩非功能词 |cos|平均: {np.mean(nonfunc_nonfunc_cos):.4f}")
    print(f"  功能词∩非功能词 |cos|平均: {np.mean(func_nonfunc_cos):.4f}")
    
    if np.mean(func_func_cos) > np.mean(nonfunc_nonfunc_cos) * 2:
        print(f"  → 功能词特征的因果对齐显著强于非功能词 → 可能是功能词共享效应")
    elif np.mean(nonfunc_nonfunc_cos) > 0.1:
        print(f"  → 非功能词特征也有因果对齐 → 因果汇聚不仅仅是功能词共享!")
    else:
        print(f"  → 非功能词特征的因果对齐弱 → 因果汇聚可能主要是功能词共享")
    
    results = {
        'last_layer': last_layer,
        'causal_matrix': causal_matrix.tolist(),
        'feature_names': fnames,
        'func_func_mean': float(np.mean(func_func_cos)),
        'nonfunc_nonfunc_mean': float(np.mean(nonfunc_nonfunc_cos)),
        'func_nonfunc_mean': float(np.mean(func_nonfunc_cos)),
        'feature_norms': {f: feature_reprs[f]['norm'] for f in fnames},
    }
    
    return results
