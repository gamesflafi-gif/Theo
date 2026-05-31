"""Leichtgewichtiger Retriever über die Wissensbasis (BM25, ohne Abhängigkeiten).

Findet zu einer Frage die am besten passenden Abschnitte. Bewusst dependency-frei,
damit Theo ohne Installation schwerer ML-Pakete sofort läuft. BM25 bewertet
seltene (informative) Treffer hoch und normalisiert die Abschnittslänge robust.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from theo.knowledge import Section, load_sections

# Sehr häufige deutsche/englische Füllwörter, die keine Relevanz tragen.
_STOPWORDS = {
    "der", "die", "das", "und", "oder", "ein", "eine", "einen", "einem", "einer",
    "ist", "sind", "war", "waren", "wird", "werden", "wie", "was", "wer", "wo",
    "warum", "wann", "welche", "welcher", "welches", "im", "in", "an", "am", "auf",
    "für", "mit", "von", "vom", "zu", "zum", "zur", "den", "dem", "des", "es",
    "man", "sich", "auch", "nicht", "noch", "nur", "bei", "aus", "als", "wenn",
    "dass", "kann", "muss", "soll", "the", "a", "an", "of", "to", "in", "is",
    "are", "what", "how", "who", "why", "when", "which", "and", "or", "do", "does",
}

_TOKEN_RE = re.compile(r"[a-zA-ZäöüÄÖÜß0-9]+")

# BM25-Parameter (Standardwerte).
_K1 = 1.5
_B = 0.75
# Wie stark der Abschnittstitel gegenüber dem Fließtext gewichtet wird.
_TITLE_BOOST = 3


def tokenize(text: str) -> list[str]:
    """Zerlegt Text in normalisierte Tokens ohne Stoppwörter."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


@dataclass
class ScoredSection:
    section: Section
    score: float


class Retriever:
    """BM25-Index über die Abschnitte der Wissensbasis."""

    def __init__(self, sections: tuple[Section, ...] | None = None) -> None:
        self.sections: tuple[Section, ...] = sections or load_sections()
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._build()

    def _build(self) -> None:
        n = len(self.sections)
        df: Counter[str] = Counter()
        for sec in self.sections:
            # Titel-Tokens stärker gewichten – sie beschreiben den Abschnitt.
            tokens = tokenize(sec.title) * _TITLE_BOOST + tokenize(sec.body)
            tf = Counter(tokens)
            self._tf.append(tf)
            self._len.append(len(tokens))
            for term in tf:
                df[term] += 1
        self._avgdl = (sum(self._len) / n) if n else 0.0
        for term, freq in df.items():
            # BM25-IDF (mit +1, damit nie negativ).
            self._idf[term] = math.log(1 + (n - freq + 0.5) / (freq + 0.5))

    def _score(self, q_terms: list[str], doc_idx: int) -> float:
        tf = self._tf[doc_idx]
        dl = self._len[doc_idx]
        denom_len = _K1 * (1 - _B + _B * (dl / self._avgdl if self._avgdl else 1))
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = self._idf.get(term, 0.0)
            score += idf * (f * (_K1 + 1)) / (f + denom_len)
        return score

    def search(self, query: str, top_k: int = 4) -> list[ScoredSection]:
        """Liefert die `top_k` relevantesten Abschnitte zur Frage."""
        q_terms = tokenize(query)
        if not q_terms:
            return []
        scored: list[ScoredSection] = []
        for i, sec in enumerate(self.sections):
            s = self._score(q_terms, i)
            if s > 0:
                scored.append(ScoredSection(section=sec, score=s))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


__all__ = ["Retriever", "ScoredSection", "tokenize"]
