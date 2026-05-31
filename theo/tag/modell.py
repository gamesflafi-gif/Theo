"""Datentypen und lokale Speicherung für Theo Tag.

Alles wird in einer einzigen JSON-Datei abgelegt (standardmäßig
``~/.theo/tag.json``). Kein Server, keine Cloud – deine Tagesdaten gehören dir.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

STANDARD_PFAD = Path.home() / ".theo" / "tag.json"

# Wochentage 0=Montag ... 6=Sonntag
WOCHENTAGE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


@dataclass
class Termin:
    """Ein fester Termin zu einer bestimmten Uhrzeit (ggf. wiederkehrend)."""

    titel: str
    uhrzeit: str                 # "HH:MM"
    dauer_min: int = 30
    wochentage: list[int] = field(default_factory=list)  # leer = jeden Tag
    erinnerung_min: int = 10     # so viele Minuten vorher erinnern

    def gilt_am(self, wochentag: int) -> bool:
        return not self.wochentage or wochentag in self.wochentage


@dataclass
class Aufgabe:
    """Eine zu erledigende Aufgabe (ohne feste Uhrzeit)."""

    titel: str
    prioritaet: int = 2          # 1 = hoch, 2 = mittel, 3 = niedrig
    dauer_min: int = 30
    erledigt: bool = False
    faellig: str | None = None   # optionales Datum "YYYY-MM-DD"


@dataclass
class Aktivitaet:
    """Eine protokollierte Aktivität – die Datenbasis für Gewohnheiten."""

    name: str
    zeitstempel: str             # ISO-Format "YYYY-MM-DDTHH:MM"

    @property
    def zeit(self) -> datetime:
        return datetime.fromisoformat(self.zeitstempel)


@dataclass
class TagDaten:
    """Der gesamte Datenbestand."""

    termine: list[Termin] = field(default_factory=list)
    aufgaben: list[Aufgabe] = field(default_factory=list)
    verlauf: list[Aktivitaet] = field(default_factory=list)


def laden(pfad: str | Path = STANDARD_PFAD) -> TagDaten:
    """Lädt die Daten (oder gibt einen leeren Bestand zurück)."""
    pfad = Path(pfad)
    if not pfad.exists():
        return TagDaten()
    roh = json.loads(pfad.read_text(encoding="utf-8"))
    return TagDaten(
        termine=[Termin(**t) for t in roh.get("termine", [])],
        aufgaben=[Aufgabe(**a) for a in roh.get("aufgaben", [])],
        verlauf=[Aktivitaet(**v) for v in roh.get("verlauf", [])],
    )


def speichern(daten: TagDaten, pfad: str | Path = STANDARD_PFAD) -> None:
    """Speichert die Daten lokal als JSON."""
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    roh = {
        "termine": [asdict(t) for t in daten.termine],
        "aufgaben": [asdict(a) for a in daten.aufgaben],
        "verlauf": [asdict(v) for v in daten.verlauf],
    }
    pfad.write_text(json.dumps(roh, ensure_ascii=False, indent=2), encoding="utf-8")
