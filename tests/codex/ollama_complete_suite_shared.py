#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import subprocess
from typing import Iterable


ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SPINNER_RE = re.compile(r"^[\s⠁-⣿]+$")


def clean_output(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "\n")
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if SPINNER_RE.fullmatch(line):
            continue
        lower = line.lower()
        if "thinking..." in lower or "...done thinking" in lower:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def run_ollama_prompt(model_name: str, prompt: str, timeout: int = 700) -> str:
    completed = subprocess.run(
        ["ollama", "run", model_name, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    combined = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    cleaned = clean_output(combined)
    if completed.returncode != 0 and not cleaned:
        raise RuntimeError(f"模型调用失败: {completed.returncode}\n{combined[:500]}")
    return cleaned


def english_ratio(text: str) -> float:
    if not text:
        return 0.0
    english_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return english_count / max(len(text), 1)


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    chinese_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return chinese_count / max(len(text), 1)


def extract_token(text: str, candidates: Iterable[str]) -> str:
    lower = text.lower()
    candidate_list = [candidate.lower() for candidate in candidates]
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line in candidate_list:
            return line
        for candidate in candidate_list:
            if re.search(rf"\b{re.escape(candidate)}\b", line):
                return candidate
    for candidate in candidate_list:
        if re.search(rf"\b{re.escape(candidate)}\b", lower):
            return candidate
    return "unknown"
