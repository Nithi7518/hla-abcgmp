"""Sourced knowledge base + TF-IDF retrieval (Phase D).

Every passage carries its source URL(s). The assistant only gives advice that comes from
a retrieved passage; if nothing relevant is found it says so instead of improvising.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = "data/knowledge/knowledge_base.json"

# Question words that map straight to a passage topic (checked before TF-IDF, which is
# too strict for short natural questions like "how does it spread?").
INTENT_WORDS = {
    "spread": ("spread", "spreads", "spreading", "transmit", "transmitted", "contagious", "infect", "infects",
               "infection", "cause", "causes", "caused", "why", "carried", "rain", "wet", "humid", "weather",
               "reason", "reasons", "origin", "come", "comes"),
    "symptoms": ("symptom", "symptoms", "look", "looks", "sign", "signs", "identify", "recognise", "recognize",
                 "appearance", "confirm", "confused", "lesion", "lesions", "spots"),
    "management": ("treat", "treatment", "treating", "manage", "management", "control", "controlling", "prevent",
                   "prevention", "cure", "cures", "curing", "remedy", "remedies", "solution", "medicine",
                   "medication", "kill", "spray", "spraying", "fungicide", "pesticide", "bactericide", "copper",
                   "remove", "prune", "pruning", "avoid", "protect", "save"),
}
INTENT_PHRASES = {
    "management": ("what should i do", "what can i do", "what do i do", "how do i deal", "how to deal",
                   "get rid", "save the tree", "save my tree", "save the plant", "save my plant", "how to stop",
                   "how do i stop", "how can i stop", "stop it"),
}


def intent_topics(question: str) -> tuple[str, ...]:
    """Which knowledge-base topics (spread / symptoms / management) the question is clearly about."""
    q = question.lower()
    words = {w.strip("?.,!") for w in q.split()}
    return tuple(t for t, kws in INTENT_WORDS.items()
                 if words & set(kws) or any(ph in q for ph in INTENT_PHRASES.get(t, ())))


@dataclass
class Passage:
    id: str
    labels: list[str]
    topic: str
    text: str
    sources: list[dict]


class KnowledgeBase:
    def __init__(self, path: str | Path = KB_PATH):
        raw = json.loads(Path(path).read_text())
        self.note = raw.get("_note", "")
        self.passages = [Passage(**p) for p in raw["passages"]]
        self._by_id = {p.id: p for p in self.passages}
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._mat = self._vec.fit_transform([f"{p.topic} {p.text}" for p in self.passages])

    def get(self, passage_id: str) -> Passage:
        return self._by_id[passage_id]

    def covers(self, label: str) -> bool:
        return any(label in p.labels for p in self.passages)

    def for_label(self, label: str, topics: tuple[str, ...] | None = None) -> list[Passage]:
        return [p for p in self.passages if label in p.labels and (topics is None or p.topic in topics)]

    def search(self, query: str, label: str | None = None, k: int = 2, min_score: float = 0.08) -> list[Passage]:
        """Top-k passages for `query` (restricted to `label` when given). Empty list when
        nothing is similar enough -- the caller must then decline to answer."""
        if label is not None:
            topics = intent_topics(query)
            if topics:
                hits = self.for_label(label, topics)
                if hits:
                    return hits[:k]
        scores = cosine_similarity(self._vec.transform([query]), self._mat)[0]
        ranked = sorted(range(len(self.passages)), key=lambda i: -scores[i])
        out = []
        for i in ranked:
            p = self.passages[i]
            if scores[i] < min_score:
                break
            if label is not None and label not in p.labels:
                continue
            out.append(p)
            if len(out) == k:
                break
        return out


def citations(passages: list[Passage]) -> list[dict]:
    seen, out = set(), []
    for p in passages:
        for s in p.sources:
            if s["source_url"] not in seen:
                seen.add(s["source_url"])
                out.append(s)
    return out
