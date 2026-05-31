"""Aus Terminen, Gewohnheiten und Aufgaben den Tagesplan bauen.

Vorgehen:
1. Feste Termine des Tages übernehmen.
2. Erkannte Gewohnheiten des Wochentags als Vorschläge ergänzen.
3. Offene Aufgaben in die freien Lücken zwischen den festen Zeiten einplanen
   (wichtigste zuerst).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from .gewohnheiten import erkenne_gewohnheiten
from .modell import TagDaten

# In diesem Zeitfenster plant Theo Aufgaben in freie Lücken ein.
TAG_START = 8 * 60       # 08:00
TAG_ENDE = 20 * 60       # 20:00


@dataclass
class Planeintrag:
    minute: int           # Startzeit als Minute des Tages (zum Sortieren)
    titel: str
    dauer_min: int
    art: str              # "Termin" | "Gewohnheit" | "Aufgabe"
    erinnerung_min: int = 10   # so viele Minuten vorher erinnern

    @property
    def uhrzeit(self) -> str:
        return f"{self.minute // 60:02d}:{self.minute % 60:02d}"


def _minute(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def tagesplan(
    daten: TagDaten,
    tag: date | None = None,
    gewohnheit_min_konfidenz: float = 0.4,
) -> list[Planeintrag]:
    """Erstellt den Tagesplan für ``tag`` (Standard: heute)."""
    tag = tag or datetime.now().date()
    wochentag = tag.weekday()
    eintraege: list[Planeintrag] = []
    belegte_titel: set[str] = set()

    # 1) Feste Termine
    for t in daten.termine:
        if t.gilt_am(wochentag):
            eintraege.append(
                Planeintrag(_minute(t.uhrzeit), t.titel, t.dauer_min, "Termin", t.erinnerung_min)
            )
            belegte_titel.add(t.titel.lower())

    # 2) Gewohnheiten als Vorschläge (nur sichere, keine Dubletten zu Terminen)
    for g in erkenne_gewohnheiten(daten.verlauf):
        if g.wochentag != wochentag or g.konfidenz < gewohnheit_min_konfidenz:
            continue
        if g.name.lower() in belegte_titel:
            continue
        eintraege.append(Planeintrag(_minute(g.uhrzeit), g.name.capitalize(), 30, "Gewohnheit"))
        belegte_titel.add(g.name.lower())

    eintraege.sort(key=lambda e: e.minute)

    # 3) Offene Aufgaben in die Lücken einplanen (wichtigste zuerst)
    offene = sorted(
        (a for a in daten.aufgaben if not a.erledigt),
        key=lambda a: (a.prioritaet, a.faellig or "9999"),
    )
    eintraege = _aufgaben_einplanen(eintraege, offene)
    eintraege.sort(key=lambda e: e.minute)
    return eintraege


def _aufgaben_einplanen(fest: list[Planeintrag], aufgaben) -> list[Planeintrag]:
    """Legt Aufgaben in die freien Zeitfenster zwischen festen Einträgen."""
    fest = sorted(fest, key=lambda e: e.minute)
    ergebnis = list(fest)
    cursor = TAG_START
    fest_index = 0
    for aufgabe in aufgaben:
        platziert = False
        # vorhandene feste Einträge der Reihe nach abklappern und Lücken nutzen
        while not platziert:
            naechster_start = (
                fest[fest_index].minute if fest_index < len(fest) else TAG_ENDE
            )
            if cursor + aufgabe.dauer_min <= naechster_start:
                ergebnis.append(Planeintrag(cursor, aufgabe.titel, aufgabe.dauer_min, "Aufgabe"))
                cursor += aufgabe.dauer_min
                platziert = True
            elif fest_index < len(fest):
                # Lücke zu klein: hinter den nächsten festen Eintrag springen
                cursor = max(cursor, fest[fest_index].minute + fest[fest_index].dauer_min)
                fest_index += 1
            else:
                break  # kein Platz mehr im Tagesfenster
    return ergebnis


def plan_text(eintraege: list[Planeintrag], tag: date | None = None) -> str:
    """Hübsche Textdarstellung des Tagesplans."""
    tag = tag or datetime.now().date()
    symbol = {"Termin": "📌", "Gewohnheit": "🔁", "Aufgabe": "✅"}
    zeilen = [f"Theos Plan für {tag.strftime('%A, %d.%m.%Y')}", "=" * 40]
    if not eintraege:
        zeilen.append("(noch nichts geplant – füge Termine/Aufgaben hinzu)")
    for e in eintraege:
        zeilen.append(f"{e.uhrzeit}  {symbol.get(e.art, '•')} {e.titel}  ({e.dauer_min} Min)")
    return "\n".join(zeilen)
