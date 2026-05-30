"""Trainingskorpus laden und säubern.

Wir trainieren ausschließlich auf **gemeinfreien** Texten (Urheber über 70 Jahre
verstorben) – hier "Die Wahlverwandtschaften" von Johann Wolfgang von Goethe
(† 1832). Das ist in Deutschland urheberrechtsfrei und enthält keine
personenbezogenen Daten. Der Text stammt vom GITenberg-Mirror (Project Gutenberg
auf GitHub).

Dieses Modul lädt den Rohtext bei Bedarf herunter und entfernt den
Gutenberg-Vor- und Nachspann, sodass nur der reine Romantext übrig bleibt.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

# Rohtext-Quelle (gemeinfrei, GitHub-Mirror von Project Gutenberg)
QUELLE_URL = (
    "https://raw.githubusercontent.com/GITenberg/"
    "Die-Wahlverwandtschaften_2403/master/2403-8.txt"
)

DATEN_DIR = Path(__file__).resolve().parent.parent / "daten"
ROH_DATEI = DATEN_DIR / "raw_2403.txt"
KORPUS_DATEI = DATEN_DIR / "korpus.txt"


def _herunterladen() -> str:
    """Lädt den Rohtext (einmalig) und legt ihn lokal ab."""
    DATEN_DIR.mkdir(exist_ok=True)
    if ROH_DATEI.exists():
        return ROH_DATEI.read_text(encoding="utf-8")
    req = urllib.request.Request(QUELLE_URL, headers={"User-Agent": "Mozilla/5.0"})
    rohbytes = urllib.request.urlopen(req, timeout=60).read()
    text = rohbytes.decode("latin-1")  # altes Gutenberg-Format ist Latin-1
    ROH_DATEI.write_text(text, encoding="utf-8")
    return text


def _saeubern(roh: str) -> str:
    """Entfernt Gutenberg-Vorspann/Nachspann und glättet Leerraum."""
    # Vorspann endet mit dem "*END*"-Hinweis des Gutenberg "small print".
    start = roh.rfind("*END*")
    if start != -1:
        roh = roh[start + len("*END*") :]
    # Nachspann beginnt mit der deutschen Gutenberg-Schlussformel.
    ende = roh.find("Ende dieses Projekt Gutenberg")
    if ende != -1:
        roh = roh[:ende]
    # Mehr als zwei Leerzeilen zu genau zwei zusammenfassen.
    roh = re.sub(r"\n{3,}", "\n\n", roh)
    return roh.strip() + "\n"


def korpus_laden() -> str:
    """Gibt den gesäuberten Trainingstext zurück (lädt/säubert bei Bedarf)."""
    if KORPUS_DATEI.exists():
        return KORPUS_DATEI.read_text(encoding="utf-8")
    sauber = _saeubern(_herunterladen())
    DATEN_DIR.mkdir(exist_ok=True)
    KORPUS_DATEI.write_text(sauber, encoding="utf-8")
    return sauber


if __name__ == "__main__":
    text = korpus_laden()
    print(f"Korpus bereit: {len(text):,} Zeichen")
    print("--- Anfang ---")
    print(text[:400])
