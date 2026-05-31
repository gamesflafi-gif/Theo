"""Kommandozeile für Theo Tag – deinen lokalen Tagesplaner.

Beispiele:
    # Feste Termine anlegen (z. B. werktags 7:00 Sport)
    python -m theo.tag.cli termin "Sport" 07:00 --dauer 45 --tage Mo,Mi,Fr

    # Aufgaben anlegen (ohne feste Zeit, werden in Lücken geplant)
    python -m theo.tag.cli aufgabe "Steuererklärung" --prio 1 --dauer 60

    # Aktivitäten protokollieren (Datenbasis für Gewohnheiten)
    python -m theo.tag.cli log "Kaffee" --zeit 2026-05-31T07:05

    # Tagesplan anzeigen / Gewohnheiten sehen / Erinnerungen prüfen
    python -m theo.tag.cli plan
    python -m theo.tag.cli gewohnheiten
    python -m theo.tag.cli erinnere
    python -m theo.tag.cli dienst        # läuft und erinnert automatisch
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

from . import modell
from .erinnerung import melde_faellige, waehle_melder
from .gewohnheiten import erkenne_gewohnheiten
from .modell import WOCHENTAGE, Aktivitaet, Aufgabe, Termin
from .planer import plan_text, tagesplan


def _tage_parsen(text: str | None) -> list[int]:
    """'Mo,Mi,Fr' -> [0, 2, 4]. Leer/None -> jeden Tag ([])."""
    if not text:
        return []
    namen = {w.lower(): i for i, w in enumerate(WOCHENTAGE)}
    return [namen[t.strip().lower()] for t in text.split(",") if t.strip().lower() in namen]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Theo Tag – lokaler Tagesplaner")
    p.add_argument("--datei", default=str(modell.STANDARD_PFAD), help="Speicherort")
    sub = p.add_subparsers(dest="befehl", required=True)

    pt = sub.add_parser("termin", help="festen Termin anlegen")
    pt.add_argument("titel")
    pt.add_argument("uhrzeit", help="HH:MM")
    pt.add_argument("--dauer", type=int, default=30)
    pt.add_argument("--tage", default="", help="z. B. Mo,Di,Mi (leer = täglich)")
    pt.add_argument("--erinnerung", type=int, default=10, help="Minuten vorher")

    pa = sub.add_parser("aufgabe", help="Aufgabe anlegen")
    pa.add_argument("titel")
    pa.add_argument("--prio", type=int, default=2, help="1=hoch,2=mittel,3=niedrig")
    pa.add_argument("--dauer", type=int, default=30)
    pa.add_argument("--faellig", default=None, help="YYYY-MM-DD")

    pl = sub.add_parser("log", help="Aktivität protokollieren")
    pl.add_argument("name")
    pl.add_argument("--zeit", default=None, help="ISO YYYY-MM-DDTHH:MM (Standard: jetzt)")

    sub.add_parser("plan", help="Tagesplan anzeigen")
    sub.add_parser("gewohnheiten", help="erkannte Gewohnheiten anzeigen")
    sub.add_parser("erinnere", help="jetzt fällige Erinnerungen senden")

    pd = sub.add_parser("dienst", help="laufend erinnern (jede Minute prüfen)")
    pd.add_argument("--intervall", type=int, default=60, help="Sekunden")

    args = p.parse_args(argv)
    daten = modell.laden(args.datei)

    if args.befehl == "termin":
        daten.termine.append(Termin(
            args.titel, args.uhrzeit, args.dauer, _tage_parsen(args.tage), args.erinnerung
        ))
        modell.speichern(daten, args.datei)
        print(f"Termin angelegt: {args.uhrzeit} {args.titel}")

    elif args.befehl == "aufgabe":
        daten.aufgaben.append(Aufgabe(args.titel, args.prio, args.dauer, False, args.faellig))
        modell.speichern(daten, args.datei)
        print(f"Aufgabe angelegt: {args.titel} (Priorität {args.prio})")

    elif args.befehl == "log":
        zeit = args.zeit or datetime.now().strftime("%Y-%m-%dT%H:%M")
        daten.verlauf.append(Aktivitaet(args.name, zeit))
        modell.speichern(daten, args.datei)
        print(f"Protokolliert: {args.name} um {zeit}")

    elif args.befehl == "plan":
        print(plan_text(tagesplan(daten)))

    elif args.befehl == "gewohnheiten":
        gew = erkenne_gewohnheiten(daten.verlauf)
        if not gew:
            print("Noch keine Gewohnheiten erkannt – protokolliere mehr mit 'log'.")
        else:
            print("Theo hat diese Gewohnheiten erkannt:")
            for g in gew:
                print(f"  {g}")

    elif args.befehl == "erinnere":
        n = melde_faellige(tagesplan(daten))
        if n == 0:
            print("Gerade ist keine Erinnerung fällig.")

    elif args.befehl == "dienst":
        melder = waehle_melder()
        print(f"Theo-Dienst läuft (Melder: {type(melder).__name__}). Strg+C zum Beenden.")
        try:
            while True:
                melde_faellige(tagesplan(daten), melder)
                time.sleep(args.intervall)
        except KeyboardInterrupt:
            print("\nTheo-Dienst beendet.")


if __name__ == "__main__":
    main()
