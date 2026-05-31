"""Kommandozeile für den Doku-Assistenten "Theo Akte".

Beispiele:
    # 1) Einen Ordner mit Dokumenten indizieren (lokal, offline)
    python -m theo.akte.cli index beispiele/akten

    # 2) Eine Frage stellen – Theo zeigt die passendsten Stellen mit Quelle
    python -m theo.akte.cli frage "Wie lang ist die Kündigungsfrist?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dokumente import lade_abschnitte
from .suche import BM25Index

INDEX_DATEI = ".theo_index.json"


def _index_bauen(ordner: str, ziel: str) -> None:
    print(f"Lese Dokumente aus: {ordner}  (alles bleibt lokal)")
    abschnitte = lade_abschnitte(ordner)
    if not abschnitte:
        print("Keine lesbaren Dokumente gefunden (.txt, .md, .pdf).")
        return
    index = BM25Index()
    index.bauen(abschnitte)
    index.speichern(ziel)
    quellen = len({a.quelle for a in abschnitte})
    print(f"Fertig: {quellen} Dokument(e), {len(abschnitte)} Abschnitte -> {ziel}")


def _fragen(frage: str, index_pfad: str, k: int) -> None:
    if not Path(index_pfad).exists():
        print(f"Kein Index gefunden ({index_pfad}). Erst 'index <ordner>' ausführen.")
        return
    index = BM25Index.laden(index_pfad)
    treffer = index.suche(frage, k=k)
    if not treffer:
        print("Keine passende Stelle gefunden.")
        return
    print(f"\nFrage: {frage}\n" + "=" * 60)
    for rang, t in enumerate(treffer, 1):
        auszug = t.text.replace("\n", " ").strip()
        if len(auszug) > 400:
            auszug = auszug[:400] + " …"
        print(f"\n[{rang}] {t.quelle} (Abschnitt {t.nummer}, Relevanz {t.punkte:.1f})")
        print(f"    {auszug}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Theo Akte – lokaler Doku-Assistent")
    sub = p.add_subparsers(dest="befehl", required=True)

    pi = sub.add_parser("index", help="einen Ordner indizieren")
    pi.add_argument("ordner")
    pi.add_argument("--ziel", default=INDEX_DATEI)

    pf = sub.add_parser("frage", help="eine Frage an die Dokumente stellen")
    pf.add_argument("frage")
    pf.add_argument("--index", default=INDEX_DATEI)
    pf.add_argument("-k", type=int, default=5, help="Anzahl der Stellen")

    args = p.parse_args(argv)
    if args.befehl == "index":
        _index_bauen(args.ordner, args.ziel)
    elif args.befehl == "frage":
        _fragen(args.frage, args.index, args.k)


if __name__ == "__main__":
    main()
