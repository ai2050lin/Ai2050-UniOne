"""
Phase CXCIX: 因果验证实验 — 大样本验证因果空间结构
====================================================

Phase CXCVIII关键问题:
  1. DS7B L27因果对齐p=0.180不显著 → 需要200+置换
  2. 5×5因果矩阵10x差异是否随机也能产生?
  3. 最后一层72°旋转的机制(MLP vs Attn)?

核心测试:
  V1: 大样本置换(200次) — 验证因果对齐统计显著性
  V2: 随机标签5×5因果矩阵 — 随机标签的因果offdiag是否也有10x差异
  V3: 最后一层机制 — MLP vs Attention对72°旋转的贡献

运行:
  python -c "import sys; sys.path.insert(0,'tests/glm5'); ..."
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json, gc, time
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# 数据集 (与causal_fiber_tracing.py相同)
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


def generate_grammar_pairs():
    """200对语法对(主被动)"""
    pairs = []
    subjects = ["The cat","The dog","He","She","The child","The teacher","The student",
        "The doctor","The farmer","The driver","The artist","The writer","The singer",
        "The player","The worker","The leader","The boy","The girl","The king","The queen",
        "The judge","The pilot","The nurse","The chef","The guard","The manager",
        "The clerk","The baker","The tailor","The painter","The builder","The hunter",
        "The fisher","The miner","The weaver","The potter","The bird","The fish",
        "The horse","The cow","The sheep","The lion","The tiger","The bear",
        "The wolf","The fox","The rabbit","The deer","The eagle","The snake"]
    verbs_past = ["chased","caught","hit","pushed","pulled","lifted","dropped","moved",
        "fixed","broke","opened","closed","found","lost","kept","gave","took",
        "brought","sent","threw","built","destroyed","created","designed","painted",
        "wrote","read","sang","played","cooked","cleaned","washed","dried","folded",
        "ironed","planted","picked","cut","watered","grew","drove","rode","flew",
        "sailed","walked","kicked","held","touched","saw"]
    objects = ["the ball","the car","the house","the door","the window","the book",
        "the letter","the package","the message","the gift","the food","the water",
        "the medicine","the tool","the key","the map","the picture","the song",
        "the story","the poem","the tree","the flower","the fruit","the seed",
        "the leaf","the road","the bridge","the path","the fence","the wall"]
    idx = 0
    while idx < 200:
        s = subjects[idx % len(subjects)]
        v = verbs_past[idx % len(verbs_past)]
        o = objects[idx % len(objects)]
        pairs.append((f"{s} {v} {o}", f"{o.capitalize()} was {v} by {s.lower()}"))
        idx += 1
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


# ============================================================
# 模型加载 & 残差流提取
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


def extract_residual_stream(model, tokenizer, device, texts, layers_to_probe, pool_method='last'):
    captured = {}
    hooks = []
    layers = model.model.layers
    for li in layers_to_probe: captured[li] = []
    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple): h = output[0].detach().cpu().float()
            else: h = output.detach().cpu().float()
            captured[li].append(h[0])
        return hook_fn
    for li in layers_to_probe: hooks.append(layers[li].register_forward_hook(make_hook(li)))
    results = {li: [] for li in layers_to_probe}
    for text in texts:
        for li in layers_to_probe: captured[li] = []
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            try: _ = model(**toks)
            except:
                for li in layers_to_probe: results[li].append(np.zeros(model.config.hidden_size))
                continue
        for li in layers_to_probe:
            if captured[li]:
                h = captured[li][0].numpy()
                results[li].append(h[-1] if pool_method == 'last' else h.mean(axis=0))
            else:
                results[li].append(np.zeros(model.config.hidden_size))
    for h in hooks: h.remove()
    for li in layers_to_probe: results[li] = np.array(results[li])
    return results


# ============================================================
# V1: 大样本置换(200次) — 验证因果对齐统计显著性
# ============================================================

def test_v1(model_name, model, tokenizer, device, d_model, n_layers):
    """V1: 200次置换测试因果对齐 — 重点DS7B L27"""
    print("\n" + "=" * 70)
    print("V1: 大样本置换(200次) — 因果对齐统计显著性")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()[:100]
    tense_pairs = generate_tense_pairs()[:100]
    
    key_layers = [0, n_layers // 2, n_layers - 1]
    key_layers = sorted(set(key_layers))
    
    n_permutations = 200
    
    # 提取表示
    pol_aff = [a for a, b in pol_pairs]
    pol_neg = [b for a, b in pol_pairs]
    pol_texts = pol_aff + pol_neg
    n_half = len(pol_aff)
    
    pol_repr = extract_residual_stream(model, tokenizer, device, pol_texts, key_layers, 'last')
    
    tense_pres = [a for a, b in tense_pairs]
    tense_past = [b for a, b in tense_pairs]
    tense_texts = tense_pres + tense_past
    n_half_t = len(tense_pres)
    
    tense_repr = extract_residual_stream(model, tokenizer, device, tense_texts, key_layers, 'last')
    
    results = {'key_layers': key_layers, 'n_permutations': n_permutations,
               'real_cross_cos': {}, 'permutation_stats': {}}
    
    for li in key_layers:
        # 真实因果方向
        X_pol = pol_repr[li]
        mean_aff = X_pol[:n_half].mean(axis=0)
        mean_neg = X_pol[n_half:].mean(axis=0)
        pol_dir = mean_neg - mean_aff
        pol_norm = np.linalg.norm(pol_dir)
        if pol_norm > 1e-10: pol_dir = pol_dir / pol_norm
        
        X_tense = tense_repr[li]
        mean_pres = X_tense[:n_half_t].mean(axis=0)
        mean_past = X_tense[n_half_t:].mean(axis=0)
        tense_dir = mean_past - mean_pres
        tense_norm = np.linalg.norm(tense_dir)
        if tense_norm > 1e-10: tense_dir = tense_dir / tense_norm
        
        real_cos = float(np.dot(pol_dir, tense_dir))
        results['real_cross_cos'][li] = real_cos
        
        # 200次置换
        perm_cos_values = []
        for perm_idx in range(n_permutations):
            np.random.seed(perm_idx * 1000 + li)
            pol_labels_perm = np.random.permutation(np.array([0]*n_half + [1]*n_half))
            tense_labels_perm = np.random.permutation(np.array([0]*n_half_t + [1]*n_half_t))
            
            p0 = X_pol[pol_labels_perm == 0].mean(axis=0)
            p1 = X_pol[pol_labels_perm == 1].mean(axis=0)
            rp = p1 - p0
            rn = np.linalg.norm(rp)
            if rn > 1e-10: rp = rp / rn
            
            t0 = X_tense[tense_labels_perm == 0].mean(axis=0)
            t1 = X_tense[tense_labels_perm == 1].mean(axis=0)
            rt = t1 - t0
            tn = np.linalg.norm(rt)
            if tn > 1e-10: rt = rt / tn
            
            perm_cos_values.append(float(np.dot(rp, rt)))
        
        perm_cos_values = np.array(perm_cos_values)
        
        # 计算p值 (双侧)
        p_two_sided = float(np.mean(np.abs(perm_cos_values) >= abs(real_cos)))
        # 也计算单侧 (正值)
        p_one_sided = float(np.mean(perm_cos_values >= real_cos)) if real_cos > 0 else float(np.mean(perm_cos_values <= real_cos))
        
        results['permutation_stats'][li] = {
            'mean': float(np.mean(perm_cos_values)),
            'std': float(np.std(perm_cos_values)),
            'max': float(np.max(perm_cos_values)),
            'min': float(np.min(perm_cos_values)),
            'max_abs': float(np.max(np.abs(perm_cos_values))),
            'percentile_95': float(np.percentile(np.abs(perm_cos_values), 95)),
            'percentile_99': float(np.percentile(np.abs(perm_cos_values), 99)),
            'p_two_sided': p_two_sided,
            'p_one_sided': p_one_sided,
            'z_score': float((real_cos - np.mean(perm_cos_values)) / max(np.std(perm_cos_values), 1e-6)),
        }
        
        stats = results['permutation_stats'][li]
        sig = " ★★★" if p_two_sided < 0.01 else (" ★★" if p_two_sided < 0.05 else (" ★" if p_two_sided < 0.10 else ""))
        
        print(f"  L{li}: real={real_cos:+.4f}")
        print(f"    perm: mean={stats['mean']:+.4f}, std={stats['std']:.4f}")
        print(f"    perm: max={stats['max']:+.4f}, min={stats['min']:+.4f}, max_abs={stats['max_abs']:.4f}")
        print(f"    |cos|95th={stats['percentile_95']:.4f}, 99th={stats['percentile_99']:.4f}")
        print(f"    p(two)={p_two_sided:.4f}, p(one)={p_one_sided:.4f}, z={stats['z_score']:.2f}{sig}")
        
        if p_two_sided < 0.05:
            print(f"    → 因果对齐是语言特有的! (p<0.05)")
        elif abs(real_cos) > stats['percentile_99']:
            print(f"    → 因果对齐可能语言特有(超出99th percentile)")
        elif abs(real_cos) > stats['percentile_95']:
            print(f"    → 因果对齐可能语言特有(超出95th percentile)")
        else:
            print(f"    → 因果对齐是高维一般性质")
    
    return results


# ============================================================
# V2: 随机标签5×5因果矩阵 — 随机也能10x差异?
# ============================================================

def test_v2(model_name, model, tokenizer, device, d_model, n_layers):
    """V2: 随机标签的5×5因果余弦矩阵 — 关键验证!"""
    print("\n" + "=" * 70)
    print("V2: 随机标签5×5因果矩阵 — 随机也能10x差异?")
    print("=" * 70)
    
    feature_names = ['polarity', 'tense', 'semantic', 'grammar', 'sentiment']
    data_generators = {
        'polarity': generate_polarity_pairs, 'tense': generate_tense_pairs,
        'semantic': generate_semantic_pairs, 'grammar': generate_grammar_pairs,
        'sentiment': generate_sentiment_pairs,
    }
    
    feature_pairs = {}
    for fname in feature_names:
        feature_pairs[fname] = data_generators[fname]()[:100]
    
    # 只测最后一层(最关键)
    last_layer = n_layers - 1
    
    print(f"  目标层: L{last_layer}")
    
    # 提取所有特征表示
    feature_reprs = {}
    for fname in feature_names:
        pairs = feature_pairs[fname]
        texts_a = [a for a, b in pairs]
        texts_b = [b for a, b in pairs]
        all_texts = texts_a + texts_b
        n_half = len(texts_a)
        reprs = extract_residual_stream(model, tokenizer, device, all_texts, [last_layer], 'last')
        feature_reprs[fname] = {'reprs': reprs[last_layer], 'n_half': n_half}
    
    # 真实5×5因果矩阵
    real_causal_matrix = np.zeros((5, 5))
    for i, f1 in enumerate(feature_names):
        for j, f2 in enumerate(feature_names):
            X1 = feature_reprs[f1]['reprs']
            n1 = feature_reprs[f1]['n_half']
            d1 = X1[n1:].mean(axis=0) - X1[:n1].mean(axis=0)
            n1n = np.linalg.norm(d1)
            if n1n > 1e-10: d1 = d1 / n1n
            
            X2 = feature_reprs[f2]['reprs']
            n2 = feature_reprs[f2]['n_half']
            d2 = X2[n2:].mean(axis=0) - X2[:n2].mean(axis=0)
            n2n = np.linalg.norm(d2)
            if n2n > 1e-10: d2 = d2 / n2n
            
            real_causal_matrix[i, j] = float(np.dot(d1, d2))
    
    real_offdiag = real_causal_matrix[np.triu_indices(5, k=1)]
    real_offdiag_abs_mean = float(np.mean(np.abs(real_offdiag)))
    
    print(f"\n  真实因果矩阵 L{last_layer}:")
    print(f"       ", end="")
    for fn in feature_names: print(f"{fn[:7]:>8}", end="")
    print()
    for i, f1 in enumerate(feature_names):
        print(f"  {f1[:7]:>7}", end="")
        for j in range(5):
            print(f"{real_causal_matrix[i,j]:+7.3f}", end="")
        print()
    print(f"  真实offdiag |cos|平均: {real_offdiag_abs_mean:.4f}")
    
    # 随机标签5×5矩阵 (50次置换)
    n_perm = 50
    rand_offdiag_abs_means = []
    
    print(f"\n  随机置换 ({n_perm} 次)...")
    for perm_idx in range(n_perm):
        np.random.seed(perm_idx * 777 + 42)
        rand_matrix = np.zeros((5, 5))
        
        for i, f1 in enumerate(feature_names):
            X1 = feature_reprs[f1]['reprs']
            n1 = feature_reprs[f1]['n_half']
            perm1 = np.random.permutation(np.array([0]*n1 + [1]*n1))
            p0 = X1[perm1 == 0].mean(axis=0)
            p1 = X1[perm1 == 1].mean(axis=0)
            d1 = p1 - p0
            d1n = np.linalg.norm(d1)
            if d1n > 1e-10: d1 = d1 / d1n
            
            for j, f2 in enumerate(feature_names):
                if i == j:
                    rand_matrix[i, j] = 1.0
                    continue
                X2 = feature_reprs[f2]['reprs']
                n2 = feature_reprs[f2]['n_half']
                perm2 = np.random.permutation(np.array([0]*n2 + [1]*n2))
                q0 = X2[perm2 == 0].mean(axis=0)
                q1 = X2[perm2 == 1].mean(axis=0)
                d2 = q1 - q0
                d2n = np.linalg.norm(d2)
                if d2n > 1e-10: d2 = d2 / d2n
                
                rand_matrix[i, j] = float(np.dot(d1, d2))
        
        rand_offdiag = rand_matrix[np.triu_indices(5, k=1)]
        rand_offdiag_abs_means.append(float(np.mean(np.abs(rand_offdiag))))
    
    rand_offdiag_abs_means = np.array(rand_offdiag_abs_means)
    
    p_value = float(np.mean(rand_offdiag_abs_means >= real_offdiag_abs_mean))
    
    print(f"\n  随机offdiag |cos|平均: {np.mean(rand_offdiag_abs_means):.4f} ± {np.std(rand_offdiag_abs_means):.4f}")
    print(f"  随机max: {np.max(rand_offdiag_abs_means):.4f}, 95th: {np.percentile(rand_offdiag_abs_means, 95):.4f}")
    print(f"  真实/随机比率: {real_offdiag_abs_mean / max(np.mean(rand_offdiag_abs_means), 1e-6):.2f}x")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  → 因果矩阵offdiag显著高于随机! 因果汇聚是语言特有的! ★★★")
    elif real_offdiag_abs_mean > np.percentile(rand_offdiag_abs_means, 95):
        print(f"  → 因果矩阵offdiag超出95th percentile, 可能语言特有 ★")
    else:
        print(f"  → 因果矩阵offdiag与随机无显著差异, 因果汇聚是高维一般性质")
    
    results = {
        'last_layer': last_layer,
        'n_perm': n_perm,
        'real_causal_matrix': real_causal_matrix.tolist(),
        'real_offdiag_abs_mean': real_offdiag_abs_mean,
        'rand_offdiag_abs_means': rand_offdiag_abs_means.tolist(),
        'rand_mean': float(np.mean(rand_offdiag_abs_means)),
        'rand_std': float(np.std(rand_offdiag_abs_means)),
        'rand_max': float(np.max(rand_offdiag_abs_means)),
        'p_value': p_value,
        'ratio': float(real_offdiag_abs_mean / max(np.mean(rand_offdiag_abs_means), 1e-6)),
    }
    
    return results


# ============================================================
# V3: 最后一层机制 — MLP vs Attention
# ============================================================

def test_v3(model_name, model, tokenizer, device, d_model, n_layers):
    """V3: 最后一层旋转的MLP vs Attention贡献"""
    print("\n" + "=" * 70)
    print("V3: 最后一层MLP vs Attention贡献")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()[:80]
    
    last_layer = n_layers - 1
    prev_layer = max(0, n_layers - 2)
    
    print(f"  比较 L{prev_layer} vs L{last_layer}")
    
    # 提取L_prev和L_last的残差流
    pol_aff = [a for a, b in pol_pairs]
    pol_neg = [b for a, b in pol_pairs]
    all_texts = pol_aff + pol_neg
    n_half = len(pol_aff)
    
    reprs = extract_residual_stream(model, tokenizer, device, all_texts, [prev_layer, last_layer], 'last')
    
    X_prev = reprs[prev_layer]
    X_last = reprs[last_layer]
    
    # 差分方向
    prev_diff = X_prev[n_half:].mean(axis=0) - X_prev[:n_half].mean(axis=0)
    last_diff = X_last[n_half:].mean(axis=0) - X_last[:n_half].mean(axis=0)
    
    prev_norm = np.linalg.norm(prev_diff)
    last_norm = np.linalg.norm(last_diff)
    if prev_norm > 1e-10: prev_diff_n = prev_diff / prev_norm
    else: prev_diff_n = np.zeros(d_model)
    if last_norm > 1e-10: last_diff_n = last_diff / last_norm
    else: last_diff_n = np.zeros(d_model)
    
    cos_rotation = float(np.dot(prev_diff_n, last_diff_n))
    
    # 差分向量变化 = last_diff - prev_diff
    # 理论上: h_last = h_prev + attn_out + mlp_out
    # 所以: last_diff - prev_diff = delta_attn + delta_mlp (近似)
    delta = last_diff - prev_diff
    delta_norm = np.linalg.norm(delta)
    
    # 用每个样本的attn和mlp输出来估计贡献
    # 简化: 用残差流差分的投影来估计
    # 投影到prev_diff方向(切向)和正交方向(法向)
    proj_tangential = np.dot(delta, prev_diff_n) * prev_diff_n
    proj_normal = delta - proj_tangential
    
    tang_norm = np.linalg.norm(proj_tangential)
    norm_norm = np.linalg.norm(proj_normal)
    
    print(f"  旋转角: {np.degrees(np.arccos(np.clip(cos_rotation, -1, 1))):.1f}°")
    print(f"  差分范数: prev={prev_norm:.4f}, last={last_norm:.4f}")
    print(f"  变化向量范数: {delta_norm:.4f}")
    print(f"  切向分量(沿prev方向): {tang_norm:.4f} ({tang_norm/max(delta_norm,1e-6)*100:.1f}%)")
    print(f"  法向分量(垂直prev): {norm_norm:.4f} ({norm_norm/max(delta_norm,1e-6)*100:.1f}%)")
    
    # MLP vs Attention: 用权重矩阵估计
    # 从L_prev到L_last: h_last = h_prev + attn(h_prev) + mlp(h_prev + attn(h_prev))
    # 差分变化: delta ≈ W_attn @ prev_diff + W_mlp @ (prev_diff + attn_diff)
    # 简化: 只用W_down (MLP输出投影) 和 W_o (Attn输出投影) 来估计
    
    try:
        layer = model.model.layers[last_layer]
        
        # Attention: W_o投影
        W_o = layer.self_attn.o_proj.weight.detach().cpu().float().numpy()  # [d_model, d_model]
        
        # MLP: W_down投影
        if hasattr(layer.mlp, 'down_proj'):
            W_down = layer.mlp.down_proj.weight.detach().cpu().float().numpy()  # [d_model, intermediate]
        elif hasattr(layer.mlp, 'dense_4h_to_h'):
            W_down = layer.mlp.dense_4h_to_h.weight.detach().cpu().float().numpy()
        else:
            W_down = None
        
        # 估计attn对旋转的贡献: W_o @ prev_diff
        attn_contrib = W_o @ prev_diff
        attn_cos = float(np.dot(attn_contrib / max(np.linalg.norm(attn_contrib), 1e-10), last_diff_n))
        
        # 估计MLP对旋转的贡献: W_down @ (gate(W_gate @ prev_diff) * (W_up @ prev_diff))
        # 简化: 用W_down的行空间投影
        if W_down is not None:
            # W_down @ activation ≈ 旋转后的向量
            # 用奇异值分解取前100个分量
            from scipy.sparse.linalg import svds
            k = min(100, min(W_down.shape) - 2)
            U_wd, s_wd, Vt_wd = svds(W_down.astype(np.float32), k=k)
            
            # prev_diff在W_down行空间中的投影
            proj_mlp = Vt_wd.T @ (Vt_wd @ prev_diff)
            mlp_cos = float(np.dot(proj_mlp / max(np.linalg.norm(proj_mlp), 1e-10), last_diff_n))
        else:
            mlp_cos = 0
        
        print(f"\n  Attn贡献: W_o@prev_diff与last_diff的cos={attn_cos:+.4f}")
        print(f"  MLP贡献: proj_MLP(prev_diff)与last_diff的cos={mlp_cos:+.4f}")
        
        if abs(attn_cos) > abs(mlp_cos) * 1.5:
            print(f"  → Attention主导最后一层旋转")
        elif abs(mlp_cos) > abs(attn_cos) * 1.5:
            print(f"  → MLP主导最后一层旋转")
        else:
            print(f"  → Attention和MLP共同贡献")
        
        results = {
            'prev_layer': prev_layer, 'last_layer': last_layer,
            'cos_rotation': cos_rotation,
            'rotation_angle': float(np.degrees(np.arccos(np.clip(cos_rotation, -1, 1)))),
            'prev_norm': float(prev_norm), 'last_norm': float(last_norm),
            'delta_norm': float(delta_norm),
            'tang_norm': float(tang_norm), 'norm_norm': float(norm_norm),
            'attn_cos': attn_cos, 'mlp_cos': mlp_cos,
        }
    except Exception as e:
        print(f"  权重分析失败: {e}")
        results = {
            'prev_layer': prev_layer, 'last_layer': last_layer,
            'cos_rotation': cos_rotation,
            'rotation_angle': float(np.degrees(np.arccos(np.clip(cos_rotation, -1, 1)))),
            'error': str(e),
        }
    
    return results


# ============================================================
# 主入口
# ============================================================

def run_all(model_name):
    print(f"\n{'='*70}")
    print(f"Phase CXCIX: Causal Validation - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model_fast(model_name)
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    print(f"  n_layers={n_layers}, d_model={d_model}")
    
    result_dir = Path(f"results/causal_validation/{model_name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    def convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: convert_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert_keys(i) for i in obj]
        return obj
    
    v1 = test_v1(model_name, model, tokenizer, device, d_model, n_layers)
    with open(result_dir / "v1_results.json", 'w') as f:
        json.dump(convert_keys(v1), f, indent=2, default=str)
    print("V1 saved!")
    
    v2 = test_v2(model_name, model, tokenizer, device, d_model, n_layers)
    with open(result_dir / "v2_results.json", 'w') as f:
        json.dump(convert_keys(v2), f, indent=2, default=str)
    print("V2 saved!")
    
    v3 = test_v3(model_name, model, tokenizer, device, d_model, n_layers)
    with open(result_dir / "v3_results.json", 'w') as f:
        json.dump(convert_keys(v3), f, indent=2, default=str)
    print("V3 saved!")
    
    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.1f}s")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return {'v1': v1, 'v2': v2, 'v3': v3}
