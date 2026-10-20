"""Load SQuAD v2 dev and build chunked corpora with answer-span relevance labels."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    para_id: int
    start: int  # char offset inside the paragraph
    end: int
    text: str


@dataclass(frozen=True)
class Question:
    qid: str
    para_id: int
    text: str
    answers: tuple  # tuple of (start, end, text) character spans in the paragraph


def load_squad(path):
    data = json.loads(Path(path).read_text())
    paragraphs, questions = [], []
    for article in data["data"]:
        for para in article["paragraphs"]:
            pid = len(paragraphs)
            paragraphs.append(para["context"])
            for qa in para["qas"]:
                if qa["is_impossible"]:
                    continue
                spans = tuple(
                    (a["answer_start"], a["answer_start"] + len(a["text"]), a["text"])
                    for a in qa["answers"]
                )
                questions.append(Question(qa["id"], pid, qa["question"], spans))
    return paragraphs, questions


def chunk_paragraphs(paragraphs, words_per_chunk, stride=None):
    """Split each paragraph into word windows. words_per_chunk=None keeps whole paragraphs."""
    chunks = []
    for pid, text in enumerate(paragraphs):
        if words_per_chunk is None:
            chunks.append(Chunk(len(chunks), pid, 0, len(text), text))
            continue
        step = stride or words_per_chunk
        words = list(WORD.finditer(text))
        i = 0
        while i < len(words):
            j = min(i + words_per_chunk, len(words))
            s, e = words[i].start(), words[j - 1].end()
            chunks.append(Chunk(len(chunks), pid, s, e, text[s:e]))
            if j == len(words):
                break
            i += step
    return chunks


def relevant_chunks(question, chunks_by_para):
    """Chunk ids that fully contain at least one gold answer span of the question."""
    rel = set()
    for ch in chunks_by_para[question.para_id]:
        for s, e, _ in question.answers:
            if ch.start <= s and e <= ch.end:
                rel.add(ch.chunk_id)
                break
    return rel


def index_by_para(chunks):
    out = {}
    for ch in chunks:
        out.setdefault(ch.para_id, []).append(ch)
    return out


def sample_questions(questions, n, seed=0):
    rng = random.Random(seed)
    qs = list(questions)
    rng.shuffle(qs)
    return qs[:n]
