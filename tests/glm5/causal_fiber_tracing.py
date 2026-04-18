"""
Phase CXCVIII: 因果纤维追踪 — 因果空间的几何结构
====================================================

Phase CXCVII核心发现:
  DS7B L27: 因果方向cos=+0.430, 但解码方向cos=-0.014
  → 因果空间≠解码空间! 因果空间可能有语言特有结构!

核心测试:
  F1: 逐层因果方向旋转追踪 — 因果方向如何逐层演化?
      - 极性差分方向在每层的旋转角
      - 时态差分方向在每层的旋转角
      - 是否正交旋转(SO(n))? 还是非正交?
  
  F2: 随机特征因果方向对齐 — 因果对齐是否语言特有?
      - 随机标签的因果方向是否也在深层对齐?
      - 与DS7B的+0.430对比
  
  F3: 多特征因果余弦矩阵 — 5种语言特征的因果空间结构
      - 极性/时态/语义/语法/情感
      - 因果余弦矩阵 vs 解码余弦矩阵

大样本: 200对/特征, 全层追踪

运行:
  python -c "import sys; sys.path.insert(0,'tests/glm5'); from causal_fiber_tracing import *; ..."
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import gc
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# 大样本数据集 — 200对/特征
# ============================================================

def generate_polarity_pairs():
    """200对极性对"""
    templates = [
        ("The cat is here", "The cat is not here"),
        ("The dog is happy", "The dog is not happy"),
        ("The house is big", "The house is not big"),
        ("The phone is working", "The phone is not working"),
        ("The bridge is safe", "The bridge is not safe"),
        ("The star is visible", "The star is not visible"),
        ("The door is open", "The door is not open"),
        ("The lake is deep", "The lake is not deep"),
        ("The road is clear", "The road is not clear"),
        ("The wall is strong", "The wall is not strong"),
        ("The bird is flying", "The bird is not flying"),
        ("The fish is swimming", "The fish is not swimming"),
        ("The car is fast", "The car is not fast"),
        ("The tree is tall", "The tree is not tall"),
        ("The river is wide", "The river is not wide"),
        ("The book is interesting", "The book is not interesting"),
        ("The food is fresh", "The food is not fresh"),
        ("The light is bright", "The light is not bright"),
        ("The music is loud", "The music is not loud"),
        ("The weather is cold", "The weather is not cold"),
        ("The door was closed", "The door was not closed"),
        ("The child was playing", "The child was not playing"),
        ("The man was running", "The man was not running"),
        ("The woman was singing", "The woman was not singing"),
        ("The dog was barking", "The dog was not barking"),
        ("The sun was shining", "The sun was not shining"),
        ("The wind was blowing", "The wind was not blowing"),
        ("The rain was falling", "The rain was not falling"),
        ("The snow was melting", "The snow was not melting"),
        ("The fire was burning", "The fire was not burning"),
        ("I like the car", "I do not like the car"),
        ("She knows the answer", "She does not know the answer"),
        ("The river flows north", "The river does not flow north"),
        ("I understand the plan", "I do not understand the plan"),
        ("She likes the movie", "She does not like the movie"),
        ("The key works well", "The key does not work well"),
        ("The food tastes good", "The food does not taste good"),
        ("He drives the truck", "He does not drive the truck"),
        ("The system runs smoothly", "The system does not run smoothly"),
        ("The machine operates well", "The machine does not operate well"),
        ("He can swim", "He cannot swim"),
        ("The bird will come", "The bird will not come"),
        ("She can help", "She cannot help"),
        ("They will agree", "They will not agree"),
        ("He can solve it", "He cannot solve it"),
        ("The team will win", "The team will not win"),
        ("She can finish it", "She cannot finish it"),
        ("They will succeed", "They will not succeed"),
        ("The water is clean", "The water is not clean"),
        ("The machine works", "The machine does not work"),
        ("He can see", "He cannot see"),
        ("The plan was approved", "The plan was not approved"),
        ("She will come", "She will not come"),
        ("The road is open", "The road is not open"),
        ("The system is stable", "The system is not stable"),
        ("I trust the result", "I do not trust the result"),
        ("The door is locked", "The door is not locked"),
        ("The glass is empty", "The glass is not empty"),
        ("The room is dark", "The room is not dark"),
        ("The sky is blue", "The sky is not blue"),
        ("The grass is green", "The grass is not green"),
        ("The coffee is hot", "The coffee is not hot"),
        ("The ice is thick", "The ice is not thick"),
        ("The path is narrow", "The path is not narrow"),
        ("The cake is sweet", "The cake is not sweet"),
        ("The movie is long", "The movie is not long"),
        ("The test is hard", "The test is not hard"),
        ("The game is fair", "The game is not fair"),
        ("The price is low", "The price is not low"),
        ("The quality is high", "The quality is not high"),
        ("The speed is slow", "The speed is not slow"),
        ("The sound is clear", "The sound is not clear"),
        ("The color is bright", "The color is not bright"),
        ("The shape is round", "The shape is not round"),
        ("The size is small", "The size is not small"),
        ("The weight is heavy", "The weight is not heavy"),
        ("The distance is short", "The distance is not short"),
        ("The temperature is warm", "The temperature is not warm"),
        ("The apple is ripe", "The apple is not ripe"),
        ("The ocean is calm", "The ocean is not calm"),
        ("The mountain is high", "The mountain is not high"),
        ("The forest is dense", "The forest is not dense"),
        ("The desert is dry", "The desert is not dry"),
        ("The city is noisy", "The city is not noisy"),
        ("The village is quiet", "The village is not quiet"),
        ("This is correct", "This is not correct"),
        ("That is true", "That is not true"),
        ("It is real", "It is not real"),
        ("The answer is right", "The answer is not right"),
        ("The method is effective", "The method is not effective"),
        ("The solution is simple", "The solution is not simple"),
        ("The problem is easy", "The problem is not easy"),
        ("The task is difficult", "The task is not difficult"),
        ("The process is fast", "The process is not fast"),
        ("The result is accurate", "The result is not accurate"),
        ("The building is tall", "The building is not tall"),
        ("The park is beautiful", "The park is not beautiful"),
        ("The market is busy", "The market is not busy"),
        ("The engine is powerful", "The engine is not powerful"),
        ("The battery is full", "The battery is not full"),
        ("The signal is strong", "The signal is not strong"),
        ("The connection is stable", "The connection is not stable"),
        ("The network is secure", "The network is not secure"),
        ("The system is reliable", "The system is not reliable"),
        ("The screen is bright", "The screen is not bright"),
        ("The cloud disappeared", "The cloud did not disappear"),
        ("He arrived early", "He did not arrive early"),
        ("She passed the test", "She did not pass the test"),
        ("They finished the work", "They did not finish the work"),
        ("The student graduated", "The student did not graduate"),
        ("He answered correctly", "He did not answer correctly"),
        ("She returned home", "She did not return home"),
        ("The flower has bloomed", "The flower has not bloomed"),
        ("He has finished", "He has not finished"),
        ("She has arrived", "She has not arrived"),
        ("The tree has grown", "The tree has not grown"),
        ("The project has started", "The project has not started"),
        ("The boy is tall", "The boy is not tall"),
        ("The girl is short", "The girl is not short"),
        ("The man is strong", "The man is not strong"),
        ("The woman is kind", "The woman is not kind"),
        ("The cat is black", "The cat is not black"),
        ("The dog is brown", "The dog is not brown"),
        ("The flower is red", "The flower is not red"),
        ("The leaf is green", "The leaf is not green"),
        ("The stone is hard", "The stone is not hard"),
        ("The feather is soft", "The feather is not soft"),
        ("The metal is shiny", "The metal is not shiny"),
        ("The wood is rough", "The wood is not rough"),
        ("The glass is smooth", "The glass is not smooth"),
        ("He is running fast", "He is not running fast"),
        ("She is reading quietly", "She is not reading quietly"),
        ("They are sleeping peacefully", "They are not sleeping peacefully"),
        ("We are learning quickly", "We are not learning quickly"),
        ("The children are playing", "The children are not playing"),
        ("The birds are singing", "The birds are not singing"),
        ("The students are studying", "The students are not studying"),
        ("The workers are building", "The workers are not building"),
        ("The doctors are helping", "The doctors are not helping"),
        ("The scientists are researching", "The scientists are not researching"),
        ("He was running fast", "He was not running fast"),
        ("She was reading quietly", "She was not reading quietly"),
        ("They were sleeping peacefully", "They were not sleeping peacefully"),
        ("We were learning quickly", "We were not learning quickly"),
        ("The river is clean", "The river is not clean"),
        ("The mountain is steep", "The mountain is not steep"),
        ("The valley is deep", "The valley is not deep"),
        ("The island is small", "The island is not small"),
        ("The coast is rocky", "The coast is not rocky"),
        ("The plain is flat", "The plain is not flat"),
        ("The hill is green", "The hill is not green"),
        ("The cave is dark", "The cave is not dark"),
        ("The shore is sandy", "The shore is not sandy"),
        ("The cliff is high", "The cliff is not high"),
        ("The stream is narrow", "The stream is not narrow"),
        ("The pond is still", "The pond is not still"),
        ("The field is wide", "The field is not wide"),
        ("The garden is beautiful", "The garden is not beautiful"),
        ("The bridge is long", "The bridge is not long"),
        ("The tower is tall", "The tower is not tall"),
        ("The castle is old", "The castle is not old"),
        ("The church is quiet", "The church is not quiet"),
        ("The school is large", "The school is not large"),
        ("The hospital is clean", "The hospital is not clean"),
        ("The office is busy", "The office is not busy"),
        ("The shop is open", "The shop is not open"),
        ("The bank is safe", "The bank is not safe"),
        ("The hotel is comfortable", "The hotel is not comfortable"),
        ("The pool is deep", "The pool is not deep"),
        ("The track is fast", "The track is not fast"),
        ("The court is fair", "The court is not fair"),
        ("The stage is bright", "The stage is not bright"),
        ("The hall is wide", "The hall is not wide"),
        ("The room is warm", "The room is not warm"),
        ("The desk is clean", "The desk is not clean"),
        ("The chair is soft", "The chair is not soft"),
        ("The bed is comfortable", "The bed is not comfortable"),
        ("The table is round", "The table is not round"),
        ("The window is clear", "The window is not clear"),
        ("The floor is smooth", "The floor is not smooth"),
        ("The wall is thick", "The wall is not thick"),
        ("The roof is strong", "The roof is not strong"),
        ("The door is wide", "The door is not wide"),
        ("The stairs are steep", "The stairs are not steep"),
        ("The path is straight", "The path is not straight"),
        ("The road is long", "The road is not long"),
        ("The street is wide", "The street is not wide"),
        ("The lane is narrow", "The lane is not narrow"),
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
            if len(pairs) >= 200:
                return pairs[:200]
    return pairs[:200]


def generate_semantic_pairs():
    """200对语义对(具体vs抽象)"""
    pairs = [
        ("The rock is heavy", "The idea is complex"),
        ("The water is cold", "The theory is abstract"),
        ("The fire is hot", "The concept is vague"),
        ("The tree is tall", "The principle is clear"),
        ("The stone is hard", "The notion is simple"),
        ("The river flows", "The argument flows"),
        ("The bird flies", "The rumor spreads"),
        ("The sun shines", "The truth emerges"),
        ("The rain falls", "The doubt grows"),
        ("The wind blows", "The debate continues"),
        ("The cat sleeps", "The mind rests"),
        ("The dog runs", "The thought races"),
        ("The fish swims", "The logic flows"),
        ("The flower grows", "The knowledge expands"),
        ("The cloud moves", "The trend shifts"),
        ("The mountain stands", "The belief persists"),
        ("The ocean waves", "The emotion surges"),
        ("The snow melts", "The tension dissolves"),
        ("The ice forms", "The habit develops"),
        ("The star glows", "The hope shines"),
        ("The door opens", "The opportunity arises"),
        ("The path bends", "The reasoning curves"),
        ("The bridge connects", "The argument links"),
        ("The wall blocks", "The barrier prevents"),
        ("The light fades", "The memory dims"),
        ("The sound echoes", "The influence resonates"),
        ("The color brightens", "The mood improves"),
        ("The shape changes", "The perspective shifts"),
        ("The size increases", "The importance grows"),
        ("The weight decreases", "The burden lessens"),
        ("The table is wooden", "The statement is logical"),
        ("The chair is metal", "The claim is valid"),
        ("The book is paper", "The argument is solid"),
        ("The shirt is cotton", "The reasoning is sound"),
        ("The glass is clear", "The explanation is transparent"),
        ("The coin is gold", "The evidence is compelling"),
        ("The rope is thick", "The connection is strong"),
        ("The needle is sharp", "The analysis is precise"),
        ("The blanket is warm", "The conclusion is comforting"),
        ("The mirror is flat", "The comparison is fair"),
        ("He held the cup", "He held the opinion"),
        ("She touched the wall", "She touched the issue"),
        ("They built the house", "They built the relationship"),
        ("We crossed the river", "We crossed the boundary"),
        ("He broke the stick", "He broke the rule"),
        ("She opened the box", "She opened the discussion"),
        ("They closed the gate", "They closed the case"),
        ("We found the key", "We found the solution"),
        ("He lost the map", "He lost the direction"),
        ("She kept the coin", "She kept the secret"),
        ("The apple is red", "The theory is sound"),
        ("The grass is green", "The logic is valid"),
        ("The sky is blue", "The concept is clear"),
        ("The blood is red", "The passion is intense"),
        ("The snow is white", "The truth is pure"),
        ("The night is black", "The mystery is deep"),
        ("The gold is yellow", "The value is high"),
        ("The ocean is blue", "The freedom is vast"),
        ("The fire is orange", "The anger is hot"),
        ("The leaf is green", "The growth is natural"),
        ("The rock is solid", "The argument is firm"),
        ("The water is liquid", "The situation is fluid"),
        ("The ice is frozen", "The relationship is cold"),
        ("The cloud is soft", "The idea is gentle"),
        ("The steel is hard", "The evidence is strong"),
        ("The feather is light", "The comment is subtle"),
        ("The mountain is massive", "The problem is enormous"),
        ("The stream is narrow", "The distinction is fine"),
        ("The desert is vast", "The implication is broad"),
        ("The diamond is precious", "The insight is valuable"),
        ("The bread is fresh", "The approach is novel"),
        ("The paint is wet", "The plan is flexible"),
        ("The metal is cold", "The response is neutral"),
        ("The wood is warm", "The tone is friendly"),
        ("The brick is heavy", "The consequence is significant"),
        ("The paper is thin", "The excuse is weak"),
        ("The rope is strong", "The bond is durable"),
        ("The chain is long", "The process is extended"),
        ("The spring is tight", "The deadline is close"),
        ("The glass is fragile", "The agreement is delicate"),
        ("The rubber is flexible", "The policy is adaptable"),
        ("The stone is permanent", "The change is lasting"),
        ("The shadow is temporary", "The effect is brief"),
        ("The echo is repeating", "The pattern is recurring"),
        ("The current is flowing", "The trend is ongoing"),
        ("The wave is rising", "The movement is growing"),
        ("The flame is dancing", "The debate is active"),
        ("The storm is approaching", "The crisis is looming"),
        ("The tide is turning", "The attitude is shifting"),
        ("The dawn is breaking", "The understanding is emerging"),
        ("The dusk is falling", "The era is ending"),
        ("The seed is growing", "The idea is developing"),
        ("The bud is opening", "The opportunity is presenting"),
        ("The fruit is ripening", "The project is maturing"),
        ("The leaf is falling", "The influence is declining"),
        ("The root is deepening", "The foundation is strengthening"),
        ("The branch is extending", "The scope is expanding"),
        ("The trunk is thickening", "The core is solidifying"),
        ("The bark is roughening", "The exterior is hardening"),
        ("The sap is flowing", "The energy is circulating"),
        ("The flower is wilting", "The enthusiasm is fading"),
        ("The garden is blooming", "The community is thriving"),
        ("The field is producing", "The effort is yielding"),
        ("The harvest is coming", "The result is approaching"),
        ("The winter is ending", "The difficulty is passing"),
        ("The spring is returning", "The hope is reviving"),
        ("The summer is peaking", "The activity is intensifying"),
        ("The autumn is arriving", "The transition is happening"),
        ("The rain is stopping", "The conflict is ceasing"),
        ("The wind is calming", "The tension is easing"),
        ("The sun is setting", "The phase is concluding"),
        ("The moon is rising", "The alternative is appearing"),
        ("The star is fading", "The influence is diminishing"),
        ("The cloud is clearing", "The confusion is resolving"),
        ("The fog is lifting", "The mystery is clarifying"),
        ("The mist is dissipating", "The uncertainty is vanishing"),
        ("The storm is passing", "The crisis is resolving"),
        ("The flood is receding", "The pressure is decreasing"),
        ("The earthquake is subsiding", "The disruption is stabilizing"),
        ("The volcano is erupting", "The conflict is exploding"),
        ("The tsunami is approaching", "The impact is imminent"),
        ("The hurricane is forming", "The problem is intensifying"),
        ("The tornado is spinning", "The controversy is swirling"),
        ("The lightning is striking", "The breakthrough is happening"),
        ("The thunder is roaring", "The response is thunderous"),
        ("The rainbow is appearing", "The solution is emerging"),
        ("The sunrise is glowing", "The beginning is promising"),
        ("The sunset is fading", "The ending is peaceful"),
        ("The horizon is expanding", "The perspective is widening"),
        ("The landscape is changing", "The situation is evolving"),
        ("The terrain is shifting", "The balance is moving"),
        ("The ecosystem is adapting", "The system is adjusting"),
        ("The habitat is transforming", "The environment is changing"),
        ("The climate is warming", "The atmosphere is heating"),
        ("The weather is improving", "The condition is bettering"),
        ("The temperature is rising", "The intensity is increasing"),
        ("The humidity is falling", "The moisture is decreasing"),
        ("The pressure is building", "The stress is accumulating"),
        ("The altitude is climbing", "The level is rising"),
        ("The depth is increasing", "The complexity is deepening"),
        ("The width is expanding", "The scope is broadening"),
        ("The height is growing", "The ambition is escalating"),
        ("The length is extending", "The duration is prolonging"),
        ("The volume is increasing", "The amount is growing"),
        ("The density is thickening", "The concentration is intensifying"),
        ("The weight is lightening", "The burden is easing"),
        ("The speed is accelerating", "The pace is quickening"),
        ("The direction is changing", "The course is altering"),
        ("The position is shifting", "The stance is moving"),
        ("The angle is adjusting", "The approach is modifying"),
        ("The distance is closing", "The gap is narrowing"),
        ("The gap is widening", "The disparity is growing"),
        ("The difference is decreasing", "The distinction is blurring"),
        ("The similarity is increasing", "The resemblance is strengthening"),
        ("The contrast is sharpening", "The distinction is clarifying"),
        ("The comparison is revealing", "The analysis is exposing"),
        ("The pattern is emerging", "The structure is appearing"),
        ("The structure is forming", "The organization is developing"),
        ("The order is establishing", "The system is stabilizing"),
        ("The chaos is increasing", "The disorder is growing"),
        ("The entropy is rising", "The randomness is increasing"),
        ("The information is accumulating", "The knowledge is building"),
        ("The noise is decreasing", "The clarity is improving"),
        ("The signal is strengthening", "The evidence is mounting"),
        ("The meaning is deepening", "The significance is growing"),
    ]
    return pairs[:200]


def generate_grammar_pairs():
    """200对语法对(主被动语态)"""
    pairs = []
    subjects_a = ["The cat", "The dog", "The man", "The woman", "The child",
                  "The teacher", "The student", "The doctor", "The farmer", "The driver",
                  "The artist", "The writer", "The singer", "The player", "The worker",
                  "The leader", "The speaker", "The nurse", "The chef", "The guard",
                  "The boy", "The girl", "The king", "The queen", "The judge",
                  "The pilot", "The sailor", "The soldier", "The officer", "The manager",
                  "The clerk", "The baker", "The tailor", "The painter", "The builder",
                  "The hunter", "The fisher", "The miner", "The weaver", "The potter",
                  "The bird", "The fish", "The horse", "The cow", "The sheep",
                  "The lion", "The tiger", "The bear", "The wolf", "The fox"]
    verbs_past = ["chased", "caught", "hit", "pushed", "pulled",
                  "lifted", "dropped", "moved", "fixed", "broke",
                  "opened", "closed", "found", "lost", "kept",
                  "gave", "took", "brought", "sent", "threw",
                  "built", "destroyed", "created", "designed", "painted",
                  "wrote", "read", "sang", "played", "cooked",
                  "cleaned", "washed", "dried", "folded", "ironed",
                  "planted", "picked", "cut", "watered", "grew",
                  "drove", "rode", "flew", "sailed", "walked"]
    objects = ["the ball", "the car", "the house", "the door", "the window",
               "the book", "the letter", "the package", "the message", "the gift",
               "the food", "the water", "the medicine", "the tool", "the key",
               "the map", "the picture", "the song", "the story", "the poem",
               "the tree", "the flower", "the fruit", "the seed", "the leaf",
               "the road", "the bridge", "the path", "the fence", "the wall"]
    
    for i, subj in enumerate(subjects_a):
        verb = verbs_past[i % len(verbs_past)]
        obj = objects[i % len(objects)]
        active = f"{subj} {verb} {obj}"
        passive = f"{obj.capitalize()} was {verb} by {subj.lower()}"
        pairs.append((active, passive))
        if len(pairs) >= 200:
            break
    
    # 补充更多
    more_verbs = ["kicked", "held", "touched", "saw", "heard",
                  "felt", "smelled", "tasted", "loved", "hated",
                  "liked", "wanted", "needed", "helped", "saved",
                  "protected", "watched", "followed", "guided", "led",
                  "taught", "told", "asked", "answered", "called"]
    more_objects = ["the child", "the animal", "the machine", "the device", "the instrument",
                    "the weapon", "the treasure", "the document", "the report", "the plan",
                    "the project", "the system", "the method", "the technique", "the skill",
                    "the lesson", "the problem", "the question", "the answer", "the solution"]
    
    idx = len(pairs)
    while idx < 200:
        subj = subjects_a[idx % len(subjects_a)]
        verb = more_verbs[idx % len(more_verbs)]
        obj = more_objects[idx % len(more_objects)]
        active = f"{subj} {verb} {obj}"
        passive = f"{obj.capitalize()} was {verb} by {subj.lower()}"
        pairs.append((active, passive))
        idx += 1
    
    return pairs[:200]


def generate_sentiment_pairs():
    """200对情感对(正面vs负面)"""
    pairs = [
        ("The movie was wonderful", "The movie was terrible"),
        ("The food was delicious", "The food was disgusting"),
        ("The weather was beautiful", "The weather was awful"),
        ("The music was amazing", "The music was horrible"),
        ("The book was fascinating", "The book was boring"),
        ("The game was exciting", "The game was dull"),
        ("The trip was enjoyable", "The trip was miserable"),
        ("The party was fantastic", "The party was dreadful"),
        ("The show was brilliant", "The show was mediocre"),
        ("The concert was spectacular", "The concert was disappointing"),
        ("The hotel was comfortable", "The hotel was uncomfortable"),
        ("The service was excellent", "The service was poor"),
        ("The product was outstanding", "The product was inferior"),
        ("The performance was superb", "The performance was subpar"),
        ("The result was impressive", "The result was unimpressive"),
        ("The experience was pleasant", "The experience was unpleasant"),
        ("The atmosphere was welcoming", "The atmosphere was hostile"),
        ("The staff was friendly", "The staff was rude"),
        ("The quality was superior", "The quality was inferior"),
        ("The design was elegant", "The design was clumsy"),
        ("The solution was clever", "The solution was foolish"),
        ("The idea was innovative", "The idea was outdated"),
        ("The approach was effective", "The approach was ineffective"),
        ("The method was efficient", "The method was wasteful"),
        ("The strategy was successful", "The strategy was failing"),
        ("The outcome was positive", "The outcome was negative"),
        ("The feedback was encouraging", "The feedback was discouraging"),
        ("The progress was remarkable", "The progress was negligible"),
        ("The improvement was significant", "The improvement was minimal"),
        ("The achievement was extraordinary", "The achievement was ordinary"),
        ("The discovery was groundbreaking", "The discovery was trivial"),
        ("The invention was revolutionary", "The invention was conventional"),
        ("The contribution was valuable", "The contribution was worthless"),
        ("The effort was commendable", "The effort was lackluster"),
        ("The dedication was admirable", "The dedication was questionable"),
        ("The commitment was unwavering", "The commitment was halfhearted"),
        ("The determination was fierce", "The determination was weak"),
        ("The courage was heroic", "The courage was cowardly"),
        ("The kindness was generous", "The kindness was selfish"),
        ("The honesty was refreshing", "The honesty was deceptive"),
        ("The patience was admirable", "The patience was irritable"),
        ("The wisdom was profound", "The wisdom was shallow"),
        ("The humor was delightful", "The humor was offensive"),
        ("The creativity was inspiring", "The creativity was uninspired"),
        ("The intelligence was remarkable", "The intelligence was unremarkable"),
        ("The beauty was stunning", "The beauty was plain"),
        ("The grace was elegant", "The grace was awkward"),
        ("The charm was captivating", "The charm was repulsive"),
        ("The warmth was comforting", "The warmth was chilling"),
        ("The joy was contagious", "The joy was forced"),
        ("The hope was uplifting", "The hope was diminishing"),
        ("The love was deep", "The love was shallow"),
        ("The peace was serene", "The peace was turbulent"),
        ("The freedom was liberating", "The freedom was restricting"),
        ("The justice was fair", "The justice was biased"),
        ("The truth was clear", "The truth was obscured"),
        ("The trust was solid", "The trust was broken"),
        ("The respect was mutual", "The respect was one-sided"),
        ("The cooperation was seamless", "The cooperation was chaotic"),
        ("The harmony was perfect", "The harmony was discordant"),
        ("The unity was strong", "The unity was fragmented"),
        ("The balance was stable", "The balance was precarious"),
        ("The order was maintained", "The order was disrupted"),
        ("The structure was solid", "The structure was crumbling"),
        ("The foundation was firm", "The foundation was shaky"),
        ("The system was reliable", "The system was unreliable"),
        ("The process was smooth", "The process was rough"),
        ("The journey was pleasant", "The journey was arduous"),
        ("The path was clear", "The path was blocked"),
        ("The road was easy", "The road was difficult"),
        ("The way was straightforward", "The way was complicated"),
        ("The answer was simple", "The answer was complex"),
        ("The question was easy", "The question was hard"),
        ("The task was manageable", "The task was overwhelming"),
        ("The challenge was stimulating", "The challenge was exhausting"),
        ("The problem was solvable", "The problem was unsolvable"),
        ("The situation was favorable", "The situation was unfavorable"),
        ("The condition was optimal", "The condition was suboptimal"),
        ("The circumstance was advantageous", "The circumstance was disadvantageous"),
        ("The opportunity was golden", "The opportunity was missed"),
        ("The moment was perfect", "The moment was wrong"),
        ("The timing was ideal", "The timing was poor"),
        ("The choice was wise", "The choice was foolish"),
        ("The decision was correct", "The decision was mistaken"),
        ("The action was appropriate", "The action was inappropriate"),
        ("The reaction was proportional", "The reaction was extreme"),
        ("The response was measured", "The response was excessive"),
        ("The behavior was professional", "The behavior was amateurish"),
        ("The attitude was positive", "The attitude was negative"),
        ("The outlook was optimistic", "The outlook was pessimistic"),
        ("The perspective was broad", "The perspective was narrow"),
        ("The vision was clear", "The vision was blurry"),
        ("The goal was achievable", "The goal was unrealistic"),
        ("The plan was feasible", "The plan was impractical"),
        ("The dream was attainable", "The dream was impossible"),
        ("The ambition was noble", "The ambition was greedy"),
        ("The desire was reasonable", "The desire was excessive"),
        ("The expectation was realistic", "The expectation was unrealistic"),
        ("The standard was high", "The standard was low"),
        ("The bar was raised", "The bar was lowered"),
        ("The limit was pushed", "The limit was accepted"),
        ("The boundary was expanded", "The boundary was restricted"),
        ("The horizon was broadened", "The horizon was narrowed"),
        ("The scope was widened", "The scope was limited"),
        ("The reach was extended", "The reach was reduced"),
        ("The impact was profound", "The impact was superficial"),
        ("The effect was lasting", "The effect was temporary"),
        ("The influence was strong", "The influence was weak"),
        ("The power was great", "The power was small"),
        ("The strength was formidable", "The strength was feeble"),
        ("The force was mighty", "The force was feeble"),
        ("The energy was vibrant", "The energy was depleted"),
        ("The vitality was abundant", "The vitality was scarce"),
        ("The spirit was indomitable", "The spirit was broken"),
        ("The will was unbreakable", "The will was fragile"),
        ("The resolve was firm", "The resolve was wavering"),
        ("The faith was unwavering", "The faith was shaky"),
        ("The belief was strong", "The belief was weak"),
        ("The conviction was deep", "The conviction was shallow"),
        ("The confidence was high", "The confidence was low"),
        ("The assurance was certain", "The assurance was doubtful"),
        ("The certainty was absolute", "The certainty was relative"),
        ("The clarity was crystal", "The clarity was murky"),
        ("The precision was exact", "The precision was approximate"),
        ("The accuracy was perfect", "The accuracy was flawed"),
        ("The detail was meticulous", "The detail was sloppy"),
        ("The care was thorough", "The care was careless"),
        ("The attention was focused", "The attention was scattered"),
        ("The focus was sharp", "The focus was blurred"),
        ("The concentration was intense", "The concentration was weak"),
        ("The awareness was heightened", "The awareness was diminished"),
        ("The understanding was deep", "The understanding was surface"),
        ("The insight was penetrating", "The insight was shallow"),
        ("The perception was acute", "The perception was dull"),
        ("The observation was keen", "The observation was poor"),
        ("The analysis was rigorous", "The analysis was sloppy"),
        ("The evaluation was fair", "The evaluation was biased"),
        ("The assessment was objective", "The assessment was subjective"),
        ("The judgment was sound", "The judgment was flawed"),
        ("The reasoning was logical", "The reasoning was illogical"),
        ("The argument was compelling", "The argument was weak"),
        ("The evidence was convincing", "The evidence was dubious"),
        ("The proof was conclusive", "The proof was inconclusive"),
        ("The demonstration was clear", "The demonstration was confusing"),
        ("The explanation was lucid", "The explanation was obscure"),
        ("The description was vivid", "The description was vague"),
        ("The presentation was polished", "The presentation was rough"),
        ("The performance was stellar", "The performance was mediocre"),
        ("The execution was flawless", "The execution was flawed"),
        ("The delivery was smooth", "The delivery was bumpy"),
        ("The production was professional", "The production was amateurish"),
        ("The quality was premium", "The quality was cheap"),
        ("The value was exceptional", "The value was ordinary"),
        ("The price was reasonable", "The price was exorbitant"),
        ("The cost was affordable", "The cost was prohibitive"),
        ("The investment was profitable", "The investment was losing"),
        ("The return was generous", "The return was meager"),
        ("The reward was substantial", "The reward was minimal"),
        ("The benefit was significant", "The benefit was negligible"),
        ("The advantage was clear", "The advantage was marginal"),
        ("The gain was considerable", "The gain was minimal"),
        ("The profit was handsome", "The profit was thin"),
        ("The income was ample", "The income was scarce"),
        ("The wealth was abundant", "The wealth was depleted"),
        ("The resource was plentiful", "The resource was scarce"),
        ("The supply was abundant", "The supply was limited"),
        ("The demand was strong", "The demand was weak"),
        ("The market was booming", "The market was crashing"),
        ("The economy was thriving", "The economy was struggling"),
        ("The growth was robust", "The growth was stagnant"),
        ("The development was rapid", "The development was slow"),
        ("The progress was steady", "The progress was halting"),
        ("The advancement was significant", "The advancement was minor"),
        ("The innovation was breakthrough", "The innovation was incremental"),
        ("The change was transformative", "The change was superficial"),
        ("The transformation was complete", "The transformation was partial"),
        ("The evolution was remarkable", "The evolution was gradual"),
        ("The revolution was profound", "The revolution was superficial"),
    ]
    return pairs[:200]


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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if model_name in ["glm4", "deepseek7b"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map='cpu',
            trust_remote_code=True, local_files_only=True,
        )
        model = model.to('cuda')
    
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


# ============================================================
# 核心工具: 提取残差流
# ============================================================

def extract_residual_stream(model, tokenizer, device, texts, layers_to_probe, pool_method='last'):
    """批量提取残差流表示"""
    captured = {}
    hooks = []
    layers = model.model.layers
    
    for li in layers_to_probe:
        captured[li] = []
    
    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0].detach().cpu().float()
            else:
                h = output.detach().cpu().float()
            captured[li].append(h[0])
        return hook_fn
    
    for li in layers_to_probe:
        hooks.append(layers[li].register_forward_hook(make_hook(li)))
    
    results = {li: [] for li in layers_to_probe}
    
    for text in texts:
        for li in layers_to_probe:
            captured[li] = []
        
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        
        with torch.no_grad():
            try:
                _ = model(**toks)
            except Exception as e:
                for li in layers_to_probe:
                    results[li].append(np.zeros(model.config.hidden_size))
                continue
        
        for li in layers_to_probe:
            if captured[li]:
                h = captured[li][0].numpy()
                if pool_method == 'last':
                    results[li].append(h[-1])
                elif pool_method == 'mean':
                    results[li].append(h.mean(axis=0))
                else:
                    results[li].append(h[-1])
            else:
                results[li].append(np.zeros(model.config.hidden_size))
    
    for h in hooks:
        h.remove()
    
    for li in layers_to_probe:
        results[li] = np.array(results[li])
    
    return results


# ============================================================
# F1: 逐层因果方向旋转追踪
# ============================================================

def test_f1(model_name, model, tokenizer, device, d_model, n_layers):
    """
    F1: 逐层追踪因果方向的旋转
    
    在每一层计算极性/时态的差分方向, 
    然后计算相邻层的旋转角(余弦距离)
    
    关键问题:
    - 因果方向是正交旋转(SO(n))? 还是非正交?
    - 旋转角是否有"汇聚点"(DS7B L27)?
    - 极性和时态的旋转是否不同?
    """
    
    print("\n" + "=" * 70)
    print("F1: 逐层因果方向旋转追踪")
    print("=" * 70)
    
    # 大样本: 200对
    pol_pairs = generate_polarity_pairs()[:150]
    tense_pairs = generate_tense_pairs()[:150]
    
    # 所有层 (太慢就每隔一层)
    if n_layers > 20:
        all_layers = list(range(0, n_layers, 2))
        if n_layers - 1 not in all_layers:
            all_layers.append(n_layers - 1)
    else:
        all_layers = list(range(n_layers))
    all_layers = sorted(set(all_layers))
    
    print(f"  层数: {len(all_layers)}, 样本: 150对/特征")
    
    # 提取极性在所有层的表示
    pol_aff = [a for a, b in pol_pairs]
    pol_neg = [b for a, b in pol_pairs]
    pol_texts = pol_aff + pol_neg
    n_half = len(pol_aff)
    
    print(f"  提取极性表示 ({len(pol_texts)} 文本, {len(all_layers)} 层)...")
    pol_repr = extract_residual_stream(model, tokenizer, device, pol_texts, all_layers, 'last')
    
    # 提取时态在所有层的表示
    tense_pres = [a for a, b in tense_pairs]
    tense_past = [b for a, b in tense_pairs]
    tense_texts = tense_pres + tense_past
    n_half_t = len(tense_pres)
    
    print(f"  提取时态表示 ({len(tense_texts)} 文本, {len(all_layers)} 层)...")
    tense_repr = extract_residual_stream(model, tokenizer, device, tense_texts, all_layers, 'last')
    
    # 计算每层的差分方向
    pol_dirs = {}  # {layer: normalized_direction}
    tense_dirs = {}
    pol_norms = {}
    tense_norms = {}
    
    for li in all_layers:
        X_pol = pol_repr[li]
        mean_aff = X_pol[:n_half].mean(axis=0)
        mean_neg = X_pol[n_half:].mean(axis=0)
        d = mean_neg - mean_aff
        norm = np.linalg.norm(d)
        pol_norms[li] = float(norm)
        if norm > 1e-10:
            pol_dirs[li] = d / norm
        else:
            pol_dirs[li] = np.zeros(d_model)
        
        X_tense = tense_repr[li]
        mean_pres = X_tense[:n_half_t].mean(axis=0)
        mean_past = X_tense[n_half_t:].mean(axis=0)
        d2 = mean_past - mean_pres
        norm2 = np.linalg.norm(d2)
        tense_norms[li] = float(norm2)
        if norm2 > 1e-10:
            tense_dirs[li] = d2 / norm2
        else:
            tense_dirs[li] = np.zeros(d_model)
    
    # 计算相邻层旋转角
    pol_rotation = {}
    tense_rotation = {}
    cross_cos = {}  # 极性vs时态的跨特征余弦
    
    for i in range(len(all_layers) - 1):
        li = all_layers[i]
        li_next = all_layers[i + 1]
        
        # 极性相邻层旋转
        pol_cos = float(np.dot(pol_dirs[li], pol_dirs[li_next]))
        pol_rotation[(li, li_next)] = pol_cos
        
        # 时态相邻层旋转
        tense_cos = float(np.dot(tense_dirs[li], tense_dirs[li_next]))
        tense_rotation[(li, li_next)] = tense_cos
        
        # 跨特征余弦(极性vs时态在同一层)
        cross_cos[li] = float(np.dot(pol_dirs[li], tense_dirs[li]))
    
    # 最后一层的跨特征余弦
    cross_cos[all_layers[-1]] = float(np.dot(pol_dirs[all_layers[-1]], tense_dirs[all_layers[-1]]))
    
    # 计算从第一层到每层的累积旋转
    pol_cumulative = {}
    tense_cumulative = {}
    for li in all_layers:
        pol_cumulative[li] = float(np.dot(pol_dirs[all_layers[0]], pol_dirs[li]))
        tense_cumulative[li] = float(np.dot(tense_dirs[all_layers[0]], tense_dirs[li]))
    
    # === 输出 ===
    print("\n  === 极性差分方向逐层旋转 ===")
    for i in range(len(all_layers) - 1):
        li = all_layers[i]
        li_next = all_layers[i + 1]
        cos_val = pol_rotation[(li, li_next)]
        angle = np.degrees(np.arccos(np.clip(cos_val, -1, 1)))
        print(f"    L{li}→L{li_next}: cos={cos_val:+.4f}, angle={angle:.1f}°")
    
    print("\n  === 时态差分方向逐层旋转 ===")
    for i in range(len(all_layers) - 1):
        li = all_layers[i]
        li_next = all_layers[i + 1]
        cos_val = tense_rotation[(li, li_next)]
        angle = np.degrees(np.arccos(np.clip(cos_val, -1, 1)))
        print(f"    L{li}→L{li_next}: cos={cos_val:+.4f}, angle={angle:.1f}°")
    
    print("\n  === 极性vs时态因果余弦(关键!) ===")
    for li in all_layers:
        cc = cross_cos[li]
        marker = " ★★★" if abs(cc) > 0.3 else (" ★" if abs(cc) > 0.1 else "")
        print(f"    L{li}: cos(pol,tense)={cc:+.4f}{marker}")
    
    print("\n  === 累积旋转(从L0) ===")
    for li in all_layers:
        pc = pol_cumulative[li]
        tc = tense_cumulative[li]
        print(f"    L{li}: pol_cos(L0)={pc:+.4f}, tense_cos(L0)={tc:+.4f}")
    
    # === 检测汇聚点 ===
    print("\n  === 汇聚点检测 ===")
    max_cross = max(cross_cos.values(), key=abs)
    max_layer = [k for k, v in cross_cos.items() if v == max_cross][0]
    print(f"    最大因果对齐: L{max_layer}, cos={max_cross:+.4f}")
    
    # 相邻层旋转角是否正交(接近90°)?
    pol_angles = [np.degrees(np.arccos(np.clip(v, -1, 1))) for v in pol_rotation.values()]
    tense_angles = [np.degrees(np.arccos(np.clip(v, -1, 1))) for v in tense_rotation.values()]
    print(f"    极性平均旋转角: {np.mean(pol_angles):.1f}° ± {np.std(pol_angles):.1f}°")
    print(f"    时态平均旋转角: {np.mean(tense_angles):.1f}° ± {np.std(tense_angles):.1f}°")
    
    # SO(n)检测: 如果旋转角接近0°, 则相邻层差分方向几乎不变
    # 如果旋转角接近90°, 则相邻层差分方向完全旋转
    if np.mean(pol_angles) < 15:
        print("    → 极性因果方向几乎不变(SO(n)中缓慢旋转)")
    elif np.mean(pol_angles) > 75:
        print("    → 极性因果方向大幅旋转(非连续流)")
    else:
        print("    → 极性因果方向中等旋转(连续流)")
    
    results = {
        'all_layers': all_layers,
        'n_pairs': len(pol_pairs),
        'pol_rotation': {f"{k[0]}_{k[1]}": v for k, v in pol_rotation.items()},
        'tense_rotation': {f"{k[0]}_{k[1]}": v for k, v in tense_rotation.items()},
        'cross_cos': {str(k): v for k, v in cross_cos.items()},
        'pol_cumulative': {str(k): v for k, v in pol_cumulative.items()},
        'tense_cumulative': {str(k): v for k, v in tense_cumulative.items()},
        'pol_norms': {str(k): v for k, v in pol_norms.items()},
        'tense_norms': {str(k): v for k, v in tense_norms.items()},
        'max_cross_layer': max_layer,
        'max_cross_cos': max_cross,
        'pol_avg_rotation': float(np.mean(pol_angles)),
        'tense_avg_rotation': float(np.mean(tense_angles)),
    }
    
    return results


# ============================================================
# F2: 随机特征因果方向对齐 — 因果对齐是否语言特有?
# ============================================================

def test_f2(model_name, model, tokenizer, device, d_model, n_layers):
    """
    F2: 随机标签的因果方向是否也在深层对齐?
    
    Phase CXCVII发现DS7B L27因果cos=+0.430
    问题: 随机标签的因果方向是否也能达到这么高?
    
    方法: 
    - 随机打乱极性对标签(50%正→负, 50%负→正)
    - 计算随机标签的极性差分方向和时态差分方向
    - 在深层计算两个随机因果方向的余弦
    - 与真实的+0.430对比
    """
    
    print("\n" + "=" * 70)
    print("F2: 随机特征因果方向对齐 — 因果对齐是否语言特有?")
    print("=" * 70)
    
    pol_pairs = generate_polarity_pairs()[:100]
    tense_pairs = generate_tense_pairs()[:100]
    
    # 关键层: 首层+中间+深层+最后
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    key_layers = sorted(set(key_layers))
    
    n_permutations = 50  # 50次随机置换
    
    results = {
        'key_layers': key_layers,
        'n_permutations': n_permutations,
        'real_cross_cos': {},
        'random_cross_cos': {},
        'random_stats': {},
    }
    
    # 先计算真实的因果方向余弦
    print(f"  计算真实因果方向余弦...")
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
    
    for li in key_layers:
        X_pol = pol_repr[li]
        mean_aff = X_pol[:n_half].mean(axis=0)
        mean_neg = X_pol[n_half:].mean(axis=0)
        pol_dir = mean_neg - mean_aff
        pol_norm = np.linalg.norm(pol_dir)
        if pol_norm > 1e-10:
            pol_dir = pol_dir / pol_norm
        
        X_tense = tense_repr[li]
        mean_pres = X_tense[:n_half_t].mean(axis=0)
        mean_past = X_tense[n_half_t:].mean(axis=0)
        tense_dir = mean_past - mean_pres
        tense_norm = np.linalg.norm(tense_dir)
        if tense_norm > 1e-10:
            tense_dir = tense_dir / tense_norm
        
        real_cos = float(np.dot(pol_dir, tense_dir))
        results['real_cross_cos'][li] = real_cos
        print(f"    L{li}: real causal cos={real_cos:+.4f}")
    
    # 随机置换测试
    print(f"\n  随机置换测试 ({n_permutations} 次)...")
    
    for li in key_layers:
        random_cos_values = []
        
        for perm_idx in range(n_permutations):
            # 随机打乱极性标签
            np.random.seed(perm_idx * 1000 + li)
            pol_labels_perm = np.random.permutation(np.array([0] * n_half + [1] * n_half))
            
            # 随机打乱时态标签
            tense_labels_perm = np.random.permutation(np.array([0] * n_half_t + [1] * n_half_t))
            
            X_pol = pol_repr[li]
            X_tense = tense_repr[li]
            
            # 按随机标签分组
            pol_group0 = X_pol[pol_labels_perm == 0].mean(axis=0)
            pol_group1 = X_pol[pol_labels_perm == 1].mean(axis=0)
            rand_pol_dir = pol_group1 - pol_group0
            rand_pol_norm = np.linalg.norm(rand_pol_dir)
            if rand_pol_norm > 1e-10:
                rand_pol_dir = rand_pol_dir / rand_pol_norm
            
            tense_group0 = X_tense[tense_labels_perm == 0].mean(axis=0)
            tense_group1 = X_tense[tense_labels_perm == 1].mean(axis=0)
            rand_tense_dir = tense_group1 - tense_group0
            rand_tense_norm = np.linalg.norm(rand_tense_dir)
            if rand_tense_norm > 1e-10:
                rand_tense_dir = rand_tense_dir / rand_tense_norm
            
            rand_cos = float(np.dot(rand_pol_dir, rand_tense_dir))
            random_cos_values.append(rand_cos)
        
        random_cos_values = np.array(random_cos_values)
        
        results['random_cross_cos'][li] = random_cos_values.tolist()
        results['random_stats'][li] = {
            'mean': float(np.mean(random_cos_values)),
            'std': float(np.std(random_cos_values)),
            'max_abs': float(np.max(np.abs(random_cos_values))),
            'p_value': float(np.mean(np.abs(random_cos_values) >= abs(results['real_cross_cos'][li]))),
        }
        
        real_val = results['real_cross_cos'][li]
        rand_mean = float(np.mean(random_cos_values))
        rand_std = float(np.std(random_cos_values))
        rand_max = float(np.max(np.abs(random_cos_values)))
        p_val = results['random_stats'][li]['p_value']
        
        significant = " ★★★ 语言特有!" if p_val < 0.05 else (" ★ 可能" if p_val < 0.15 else "")
        
        print(f"    L{li}: real={real_val:+.4f}, rand_mean={rand_mean:+.4f}±{rand_std:.4f}, "
              f"rand_max_abs={rand_max:.4f}, p={p_val:.3f}{significant}")
    
    # === 汇总 ===
    print("\n  === F2 汇总: 因果对齐是否语言特有? ===")
    for li in key_layers:
        real = results['real_cross_cos'][li]
        rand_mean = results['random_stats'][li]['mean']
        rand_max = results['random_stats'][li]['max_abs']
        p_val = results['random_stats'][li]['p_value']
        
        if p_val < 0.05:
            verdict = "因果对齐是语言特有!"
        elif abs(real) > rand_max:
            verdict = "因果对齐可能语言特有(超出随机范围)"
        else:
            verdict = "因果对齐是高维一般性质"
        
        print(f"    L{li}: real={real:+.4f}, rand_max_abs={rand_max:.4f}, p={p_val:.3f} → {verdict}")
    
    return results


# ============================================================
# F3: 多特征因果余弦矩阵
# ============================================================

def test_f3(model_name, model, tokenizer, device, d_model, n_layers):
    """
    F3: 5种语言特征的因果余弦矩阵 vs 解码余弦矩阵
    
    特征: 极性/时态/语义/语法/情感
    层: 首层+中间+深层+最后
    
    输出:
    - 因果余弦矩阵 (5×5 per layer)
    - 解码余弦矩阵 (5×5 per layer)
    - 两者对比
    """
    
    print("\n" + "=" * 70)
    print("F3: 多特征因果余弦矩阵 — 5种语言特征")
    print("=" * 70)
    
    feature_names = ['polarity', 'tense', 'semantic', 'grammar', 'sentiment']
    
    # 生成数据 (100对/特征, 节省内存)
    data_generators = {
        'polarity': generate_polarity_pairs,
        'tense': generate_tense_pairs,
        'semantic': generate_semantic_pairs,
        'grammar': generate_grammar_pairs,
        'sentiment': generate_sentiment_pairs,
    }
    
    feature_pairs = {}
    for fname in feature_names:
        feature_pairs[fname] = data_generators[fname]()[:100]
    
    key_layers = [0, n_layers // 2, n_layers - 1]
    key_layers = sorted(set(key_layers))
    
    print(f"  特征数: {len(feature_names)}, 层数: {len(key_layers)}, 样本: 100对/特征")
    
    # 提取所有特征在关键层的表示
    feature_reprs = {}  # {feature_name: {layer: X}}
    
    for fname in feature_names:
        pairs = feature_pairs[fname]
        texts_a = [a for a, b in pairs]
        texts_b = [b for a, b in pairs]
        all_texts = texts_a + texts_b
        n_half = len(texts_a)
        
        print(f"  提取 {fname} 表示...")
        reprs = extract_residual_stream(model, tokenizer, device, all_texts, key_layers, 'last')
        feature_reprs[fname] = {'reprs': reprs, 'n_half': n_half}
    
    # 计算因果余弦矩阵和解码余弦矩阵
    results = {
        'feature_names': feature_names,
        'key_layers': key_layers,
        'causal_cos_matrix': {},
        'decode_cos_matrix': {},
    }
    
    for li in key_layers:
        print(f"\n  --- L{li} ---")
        
        # 计算每个特征的差分方向和probe方向
        causal_dirs = {}
        decode_dirs = {}
        
        for fname in feature_names:
            reprs = feature_reprs[fname]['reprs']
            n_half = feature_reprs[fname]['n_half']
            X = reprs[li]
            labels = np.array([0] * n_half + [1] * n_half)
            
            # 差分方向(因果方向)
            mean_a = X[:n_half].mean(axis=0)
            mean_b = X[n_half:].mean(axis=0)
            diff_dir = mean_b - mean_a
            diff_norm = np.linalg.norm(diff_dir)
            if diff_norm > 1e-10:
                causal_dirs[fname] = diff_dir / diff_norm
            else:
                causal_dirs[fname] = np.zeros(d_model)
            
            # Probe方向(解码方向)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
            clf.fit(X_scaled, labels)
            w = clf.coef_[0]
            w_norm = np.linalg.norm(w)
            if w_norm > 1e-10:
                decode_dirs[fname] = w / w_norm
            else:
                decode_dirs[fname] = np.zeros(d_model)
        
        # 因果余弦矩阵
        causal_matrix = np.zeros((len(feature_names), len(feature_names)))
        for i, f1 in enumerate(feature_names):
            for j, f2 in enumerate(feature_names):
                causal_matrix[i, j] = float(np.dot(causal_dirs[f1], causal_dirs[f2]))
        
        # 解码余弦矩阵
        decode_matrix = np.zeros((len(feature_names), len(feature_names)))
        for i, f1 in enumerate(feature_names):
            for j, f2 in enumerate(feature_names):
                decode_matrix[i, j] = float(np.dot(decode_dirs[f1], decode_dirs[f2]))
        
        results['causal_cos_matrix'][li] = causal_matrix.tolist()
        results['decode_cos_matrix'][li] = decode_matrix.tolist()
        
        # 输出因果余弦矩阵
        print(f"    因果余弦矩阵:")
        print(f"         ", end="")
        for fname in feature_names:
            print(f"{fname[:7]:>8}", end="")
        print()
        for i, f1 in enumerate(feature_names):
            print(f"    {f1[:7]:>7}", end="")
            for j in range(len(feature_names)):
                val = causal_matrix[i, j]
                marker = " ★" if abs(val) > 0.3 and i != j else ""
                print(f"{val:+7.3f}{marker}", end="")
            print()
        
        # 输出解码余弦矩阵
        print(f"    解码余弦矩阵:")
        print(f"         ", end="")
        for fname in feature_names:
            print(f"{fname[:7]:>8}", end="")
        print()
        for i, f1 in enumerate(feature_names):
            print(f"    {f1[:7]:>7}", end="")
            for j in range(len(feature_names)):
                val = decode_matrix[i, j]
                print(f"{val:+7.3f}", end="")
            print()
        
        # 对比: 因果vs解码的非对角线元素
        causal_offdiag = causal_matrix[np.triu_indices(len(feature_names), k=1)]
        decode_offdiag = decode_matrix[np.triu_indices(len(feature_names), k=1)]
        
        causal_abs_mean = float(np.mean(np.abs(causal_offdiag)))
        decode_abs_mean = float(np.mean(np.abs(decode_offdiag)))
        
        print(f"    非对角线|cos|平均: causal={causal_abs_mean:.4f}, decode={decode_abs_mean:.4f}")
        
        if causal_abs_mean > decode_abs_mean * 1.5:
            print(f"    → 因果空间特征更相关! (因果汇聚)")
        elif decode_abs_mean > causal_abs_mean * 1.5:
            print(f"    → 解码空间特征更相关!")
        else:
            print(f"    → 因果和解码空间特征相关性相似")
        
        # 找因果空间中最大的跨特征对齐
        max_idx = np.unravel_index(np.argmax(np.abs(causal_matrix - np.eye(len(feature_names)))), causal_matrix.shape)
        max_val = causal_matrix[max_idx]
        max_pair = (feature_names[max_idx[0]], feature_names[max_idx[1]])
        print(f"    最大因果对齐: {max_pair[0]}∩{max_pair[1]} = {max_val:+.4f}")
    
    return results


# ============================================================
# 主入口
# ============================================================

def run_all_tests(model_name):
    print(f"\n{'='*70}")
    print(f"Phase CXCVIII: Causal Fiber Tracing - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model_fast(model_name)
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    print(f"  n_layers={n_layers}, d_model={d_model}")
    
    result_dir = Path(f"results/causal_fiber/{model_name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    def convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: convert_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys(i) for i in obj]
        return obj
    
    # F1: 逐层因果方向旋转追踪
    print("\n\n>>> F1: 逐层因果方向旋转追踪 <<<")
    f1_results = test_f1(model_name, model, tokenizer, device, d_model, n_layers)
    with open(result_dir / "f1_results.json", 'w') as f:
        json.dump(convert_keys(f1_results), f, indent=2, default=str)
    print("F1 results saved!")
    
    # F2: 随机特征因果方向对齐
    print("\n\n>>> F2: 随机特征因果方向对齐 <<<")
    f2_results = test_f2(model_name, model, tokenizer, device, d_model, n_layers)
    with open(result_dir / "f2_results.json", 'w') as f:
        json.dump(convert_keys(f2_results), f, indent=2, default=str)
    print("F2 results saved!")
    
    # F3: 多特征因果余弦矩阵
    print("\n\n>>> F3: 多特征因果余弦矩阵 <<<")
    f3_results = test_f3(model_name, model, tokenizer, device, d_model, n_layers)
    with open(result_dir / "f3_results.json", 'w') as f:
        json.dump(convert_keys(f3_results), f, indent=2, default=str)
    print("F3 results saved!")
    
    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.1f}s")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return {'f1': f1_results, 'f2': f2_results, 'f3': f3_results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    run_all_tests(args.model)
