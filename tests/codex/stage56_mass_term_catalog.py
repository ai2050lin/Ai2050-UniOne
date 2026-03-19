from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TermItem:
    term: str
    category: str


def default_term_catalog() -> List[TermItem]:
    rows = [
        ("apple", "fruit"), ("banana", "fruit"), ("orange", "fruit"), ("grape", "fruit"), ("pear", "fruit"),
        ("peach", "fruit"), ("mango", "fruit"), ("lemon", "fruit"), ("strawberry", "fruit"), ("watermelon", "fruit"),
        ("pineapple", "fruit"), ("cherry", "fruit"), ("plum", "fruit"), ("kiwi", "fruit"), ("coconut", "fruit"),
        ("rabbit", "animal"), ("cat", "animal"), ("dog", "animal"), ("horse", "animal"), ("tiger", "animal"),
        ("lion", "animal"), ("bird", "animal"), ("fish", "animal"), ("elephant", "animal"), ("monkey", "animal"),
        ("wolf", "animal"), ("bear", "animal"), ("deer", "animal"), ("goat", "animal"), ("zebra", "animal"),
        ("sun", "celestial"), ("moon", "celestial"), ("star", "celestial"), ("planet", "celestial"), ("comet", "celestial"),
        ("galaxy", "celestial"), ("asteroid", "celestial"), ("meteor", "celestial"), ("satellite", "celestial"), ("nebula", "celestial"),
        ("cloud", "weather"), ("rain", "weather"), ("snow", "weather"), ("wind", "weather"), ("storm", "weather"),
        ("thunder", "weather"), ("lightning", "weather"), ("fog", "weather"), ("humidity", "weather"), ("temperature", "weather"),
        ("car", "vehicle"), ("bus", "vehicle"), ("train", "vehicle"), ("bicycle", "vehicle"), ("airplane", "vehicle"),
        ("ship", "vehicle"), ("truck", "vehicle"), ("motorcycle", "vehicle"), ("subway", "vehicle"), ("boat", "vehicle"),
        ("chair", "object"), ("table", "object"), ("bed", "object"), ("lamp", "object"), ("door", "object"),
        ("window", "object"), ("bottle", "object"), ("cup", "object"), ("spoon", "object"), ("knife", "object"),
        ("clock", "object"), ("mirror", "object"), ("phone", "object"), ("computer", "object"), ("keyboard", "object"),
        ("bread", "food"), ("rice", "food"), ("meat", "food"), ("soup", "food"), ("pizza", "food"),
        ("cake", "food"), ("coffee", "food"), ("tea", "food"), ("milk", "food"), ("cheese", "food"),
        ("noodle", "food"), ("egg", "food"), ("salad", "food"), ("butter", "food"), ("chocolate", "food"),
        ("tree", "nature"), ("flower", "nature"), ("grass", "nature"), ("forest", "nature"), ("river", "nature"),
        ("mountain", "nature"), ("ocean", "nature"), ("desert", "nature"), ("leaf", "nature"), ("seed", "nature"),
        ("child", "human"), ("teacher", "human"), ("doctor", "human"), ("student", "human"), ("parent", "human"),
        ("friend", "human"), ("king", "human"), ("queen", "human"), ("artist", "human"), ("worker", "human"),
        ("lawyer", "human"), ("pilot", "human"), ("engineer", "human"), ("farmer", "human"), ("nurse", "human"),
        ("algorithm", "tech"), ("data", "tech"), ("number", "tech"), ("equation", "tech"), ("database", "tech"),
        ("network", "tech"), ("software", "tech"), ("hardware", "tech"), ("robot", "tech"), ("chip", "tech"),
        ("love", "abstract"), ("hate", "abstract"), ("justice", "abstract"), ("peace", "abstract"), ("war", "abstract"),
        ("music", "abstract"), ("art", "abstract"), ("history", "abstract"), ("future", "abstract"), ("memory", "abstract"),
    ]
    return [TermItem(term=term, category=category) for term, category in rows]


def load_terms(path: str | None, max_terms: int | None) -> List[TermItem]:
    if not path:
        rows = default_term_catalog()
        return rows[:max_terms] if max_terms else rows

    out: List[TermItem] = []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"terms file not found: {path}")

    for line in file_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "," in text:
            term, category = [item.strip() for item in text.split(",", 1)]
            out.append(TermItem(term=term, category=category or "uncategorized"))
        else:
            out.append(TermItem(term=text, category="uncategorized"))
    if max_terms:
        out = out[:max_terms]
    return out


def indefinite_article(term: str) -> str:
    head = (term or "").strip().lower()
    if not head:
        return "a"
    return "an" if head[0] in {"a", "e", "i", "o", "u"} else "a"


def base_term_prompts(term: str, category: str) -> List[str]:
    category_key = str(category or "").strip().lower()
    if category_key == "action":
        return [
            f"People often {term}.",
            f"To {term} is to",
            f"They decided to {term} because",
            f"We can {term} when",
            f"The act to {term} often",
        ]
    if category_key == "abstract":
        return [
            f"People discuss {term}.",
            f"The idea of {term} is",
            f"Many debates involve {term} because",
            f"{term.capitalize()} matters when",
            f"In philosophy, {term} often",
        ]
    if category_key == "weather":
        return [
            f"The weather pattern {term} often",
            f"When people talk about {term}, they usually mean",
            f"{term.capitalize()} changes when",
            f"In the atmosphere, {term} can",
            f"The effect of {term} is often",
        ]
    if category_key == "human":
        article = indefinite_article(term)
        return [
            f"This is {article} {term}.",
            f"A person such as {article} {term} often",
            f"When people describe {article} {term}, they mention",
            f"The social role of {article} {term} is",
            f"In a community, {article} {term} can",
        ]
    if category_key == "tech":
        return [
            f"The technical concept {term} often",
            f"In computing, {term} is",
            f"When engineers discuss {term}, they mention",
            f"The system role of {term} is",
            f"A precise use of {term} is",
        ]
    article = indefinite_article(term)
    return [
        f"This is {article} {term}.",
        f"I saw {article} {term}.",
        f"People discuss {term}.",
        f"The {term} is often",
        f"A {term} can be",
    ]


def term_prompts(term: str, category: str) -> List[str]:
    return base_term_prompts(term, category)


def pool_term_prompts(term: str, category: str, pool: str) -> List[str]:
    base = base_term_prompts(term, category)
    if pool == "survey":
        return base[:3]
    if pool == "deep":
        return base + [
            f"The concept {term} is related to",
            f"When experts discuss {term}, they often mention",
            f"The role of {term} in a larger system is",
            f"Compared with similar concepts, {term} is",
        ]
    if pool == "closure":
        return base + [
            f"A precise definition of {term} is",
            f"The family of {term} can be described as",
            f"The key attribute of {term} is",
            f"The relation between {term} and nearby concepts is",
            f"In a reasoning chain, {term} usually leads to",
            f"The stage-conditioned continuation of {term} is",
            f"An incorrect family assignment for {term} would be",
            f"Under a protocol change, {term} should still preserve",
            f"The minimal explanation for {term} is",
        ]
    raise ValueError(f"unknown pool: {pool}")
