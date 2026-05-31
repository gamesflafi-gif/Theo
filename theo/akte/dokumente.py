"""Dokumente einlesen und in durchsuchbare Häppchen ("Chunks") schneiden.

Warum Häppchen? Ein 80-seitiger Vertrag als ein einziger Block wäre als
Suchergebnis nutzlos. Wir zerlegen ihn in kleine, zitierbare Abschnitte, damit
Theo später sagen kann: "Die Antwort steht in *vertrag.txt*, Abschnitt 12."

Alles passiert lokal – die Dateien verlassen den Rechner nie.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Welche Dateitypen wir direkt lesen können (reiner Text).
TEXT_ENDUNGEN = {".txt", ".md", ".text"}


@dataclass
class Abschnitt:
    """Ein durchsuchbares Häppchen aus einem Dokument."""

    quelle: str   # Dateiname/Pfad
    nummer: int   # wievielter Abschnitt im Dokument
    text: str     # der eigentliche Inhalt


def _lies_text(pfad: Path) -> str | None:
    """Liest eine Datei als Text. Gibt None zurück, wenn nicht lesbar."""
    if pfad.suffix.lower() in TEXT_ENDUNGEN:
        for enc in ("utf-8", "latin-1"):
            try:
                return pfad.read_text(encoding=enc)
            except (UnicodeDecodeError, OSError):
                continue
    elif pfad.suffix.lower() == ".pdf":
        return _lies_pdf(pfad)
    return None


def _lies_pdf(pfad: Path) -> str | None:
    """Liest ein PDF, falls die Bibliothek ``pypdf`` installiert ist."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"  (PDF übersprungen, 'pip install pypdf' fehlt): {pfad.name}")
        return None
    try:
        reader = PdfReader(str(pfad))
        return "\n".join(seite.extract_text() or "" for seite in reader.pages)
    except Exception as e:  # robust bleiben, ein kaputtes PDF darf nicht stoppen
        print(f"  (PDF nicht lesbar: {pfad.name}: {e})")
        return None


def zerteile(text: str, max_zeichen: int = 450) -> list[str]:
    """Schneidet Text an Absatzgrenzen in Häppchen von höchstens ``max_zeichen``.

    Wir schneiden bevorzugt an leeren Zeilen (Absätzen), damit Sätze nicht
    mitten durchtrennt werden.
    """
    absaetze = [a.strip() for a in text.split("\n\n") if a.strip()]
    haeppchen: list[str] = []
    aktuell = ""
    for absatz in absaetze:
        if len(aktuell) + len(absatz) + 2 <= max_zeichen:
            aktuell = f"{aktuell}\n\n{absatz}".strip()
        else:
            if aktuell:
                haeppchen.append(aktuell)
            # Sehr lange Einzelabsätze hart nachschneiden.
            while len(absatz) > max_zeichen:
                haeppchen.append(absatz[:max_zeichen])
                absatz = absatz[max_zeichen:]
            aktuell = absatz
    if aktuell:
        haeppchen.append(aktuell)
    return haeppchen


def lade_abschnitte(ordner: str | Path, max_zeichen: int = 450) -> list[Abschnitt]:
    """Liest alle unterstützten Dateien in ``ordner`` (rekursiv) als Abschnitte."""
    ordner = Path(ordner)
    abschnitte: list[Abschnitt] = []
    dateien = sorted(p for p in ordner.rglob("*") if p.is_file())
    for pfad in dateien:
        text = _lies_text(pfad)
        if not text or not text.strip():
            continue
        rel = str(pfad.relative_to(ordner))
        for i, haeppchen in enumerate(zerteile(text, max_zeichen)):
            abschnitte.append(Abschnitt(quelle=rel, nummer=i, text=haeppchen))
    return abschnitte
