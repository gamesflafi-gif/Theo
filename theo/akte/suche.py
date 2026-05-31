"""Lokaler Suchindex auf Basis von BM25.

BM25 ist das bewährte Standard-Verfahren der Informationssuche (es steckt z. B.
in Elasticsearch). Es bewertet, wie gut ein Abschnitt zu einer Suchanfrage passt:
- Ein Wort aus der Frage zählt mehr, wenn es im Abschnitt oft vorkommt …
- … aber weniger, wenn es ohnehin in fast jedem Abschnitt steht (z. B. "und").
- Kurze Abschnitte mit Treffern werden bevorzugt vor langen.

Alles läuft offline und ist nachvollziehbar – keine Blackbox, kein Internet.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .dokumente import Abschnitt

# Sehr häufige deutsche Wörter, die für die Suche kaum Bedeutung tragen.
STOPPWOERTER = {
    "der", "die", "das", "und", "oder", "ist", "sind", "war", "waren", "ein",
    "eine", "einen", "einem", "einer", "eines", "im", "in", "an", "auf", "für",
    "mit", "von", "zu", "zur", "zum", "den", "dem", "des", "es", "sich", "auch",
    "wird", "werden", "wie", "als", "am", "nicht", "nur", "aber", "dass", "daß",
    "so", "man", "bei", "aus", "nach", "über", "wenn", "wir", "ich", "sie", "er",
}

# Zerlegt Text in Wörter: alles klein, nur Buchstaben/Ziffern (inkl. Umlaute).
_WORT = re.compile(r"[a-zA-ZäöüÄÖÜß0-9]+")


def tokenisiere(text: str) -> list[str]:
    """Text -> Liste relevanter Wörter (klein, ohne Stoppwörter)."""
    return [
        w for w in (m.group().lower() for m in _WORT.finditer(text))
        if w not in STOPPWOERTER and len(w) > 1
    ]


@dataclass
class Treffer:
    quelle: str
    nummer: int
    text: str
    punkte: float


class BM25Index:
    """Ein durchsuchbarer BM25-Index über Dokument-Abschnitte."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1                      # Sättigung: wie stark zählt Häufigkeit
        self.b = b                        # wie stark zählt die Abschnittslänge
        self.abschnitte: list[Abschnitt] = []
        self.tokens: list[list[str]] = []     # Wörter je Abschnitt
        self.df: Counter[str] = Counter()     # in wie vielen Abschnitten kommt Wort vor
        self.avg_len: float = 0.0

    def bauen(self, abschnitte: list[Abschnitt]) -> None:
        """Baut den Index aus einer Liste von Abschnitten."""
        self.abschnitte = abschnitte
        self.tokens = [tokenisiere(a.text) for a in abschnitte]
        self.df = Counter()
        for toks in self.tokens:
            for wort in set(toks):
                self.df[wort] += 1
        gesamt = sum(len(t) for t in self.tokens)
        self.avg_len = gesamt / len(self.tokens) if self.tokens else 0.0

    def _idf(self, df: int) -> float:
        """Seltene Wörter sind aussagekräftiger -> höheres Gewicht."""
        n = len(self.abschnitte)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _passende_vokabeln(self, wort: str) -> list[str]:
        """Findet Vokabeln, die zu einem Suchwort passen – auch als Wortteil.

        Das fängt deutsche Zusammensetzungen ab: Die Suche nach "urlaub" findet
        auch "erholungsurlaub" und "urlaubstage", und umgekehrt.
        """
        if wort in self.df:
            treffer = [wort]
        else:
            treffer = []
        for vokabel in self.df:
            if vokabel == wort:
                continue
            if (len(wort) >= 4 and wort in vokabel) or (len(vokabel) >= 4 and vokabel in wort):
                treffer.append(vokabel)
        return treffer

    def suche(self, frage: str, k: int = 5) -> list[Treffer]:
        """Gibt die ``k`` am besten passenden Abschnitte zurück."""
        frage_tokens = set(tokenisiere(frage))
        # Für jedes Suchwort: passende Vokabeln + ihr Gewicht (idf) vorab bestimmen.
        suchplan: list[tuple[set[str], float]] = []
        for wort in frage_tokens:
            vokabeln = self._passende_vokabeln(wort)
            if not vokabeln:
                continue
            df = min(self.df[v] for v in vokabeln)  # seltenste Treffervokabel
            suchplan.append((set(vokabeln), self._idf(df)))

        treffer: list[Treffer] = []
        for idx, toks in enumerate(self.tokens):
            if not toks:
                continue
            freq = Counter(toks)
            laenge = len(toks)
            punkte = 0.0
            for vokabeln, idf in suchplan:
                tf = sum(freq[v] for v in vokabeln if v in freq)
                if tf == 0:
                    continue
                norm = tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * laenge / (self.avg_len or 1))
                )
                punkte += idf * norm
            if punkte > 0:
                a = self.abschnitte[idx]
                treffer.append(Treffer(a.quelle, a.nummer, a.text, punkte))
        treffer.sort(key=lambda t: t.punkte, reverse=True)
        return treffer[:k]

    # ---- Speichern / Laden (damit man nicht jedes Mal neu indizieren muss) ----

    def speichern(self, pfad: str | Path) -> None:
        daten = {
            "k1": self.k1,
            "b": self.b,
            "abschnitte": [a.__dict__ for a in self.abschnitte],
        }
        Path(pfad).write_text(json.dumps(daten, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def laden(cls, pfad: str | Path) -> "BM25Index":
        daten = json.loads(Path(pfad).read_text(encoding="utf-8"))
        index = cls(k1=daten["k1"], b=daten["b"])
        abschnitte = [Abschnitt(**a) for a in daten["abschnitte"]]
        index.bauen(abschnitte)
        return index
