# Theo 🟢 — KI, die lokal & datenschutzsicher arbeitet

Theo besteht aus drei Teilen:

1. **🗓️ Theo Tag** *(persönlicher Assistent)* — plant deinen Tag, **erkennt deine
   Gewohnheiten** aus deinem Verlauf und **erinnert dich** rechtzeitig (Konsole,
   Desktop oder Push aufs Handy via Telegram). Alle Daten bleiben **lokal**.
2. **🔒 Theo Akte** *(Doku-Assistent)* — durchsucht vertrauliche Dokumente
   blitzschnell und findet passende Stellen — **komplett offline**. Für alle, wo
   Cloud-KI aus Datenschutzgründen **verboten** ist: Kanzleien, Praxen, Behörden.
3. **🧠 Theo Lab** *(das Fundament)* — ein selbst gebauter Mini-GPT, an dem wir die
   KI-Technik von Grund auf verstehen.

> Roter Faden: **Deine Daten bleiben bei dir.** Egal ob Tagesablauf oder Akten –
> nichts wandert in die Cloud. Das ist Theos Versprechen und sein Marktvorteil.

> **Warum das ein Markt ist:** Anwälte, Ärzte und Ämter in Deutschland *dürfen*
> ihre sensiblen Daten nicht in die Cloud geben (DSGVO). Eine KI, die **lokal**
> läuft, ist für sie oft wichtiger als perfekte Qualität — und genau das können
> die großen Cloud-Anbieter nicht bieten. Das ist unsere Lücke.

---

## 🗓️ Theo Tag — dein persönlicher Tagesplaner

Theo plant deinen Tag, lernt deine Gewohnheiten und erinnert dich – alles lokal.

```bash
# Feste Termine (z. B. werktags 9:30 Standup, 5 Min vorher erinnern)
python -m theo.tag.cli termin "Standup" 09:30 --tage Mo,Di,Mi,Do,Fr --erinnerung 5

# Aufgaben ohne feste Zeit (werden automatisch in freie Lücken geplant)
python -m theo.tag.cli aufgabe "Steuererklärung" --prio 1 --dauer 60

# Was du tust protokollieren -> daraus lernt Theo deine Gewohnheiten
python -m theo.tag.cli log "Sport"

python -m theo.tag.cli gewohnheiten   # erkannte Muster anzeigen
python -m theo.tag.cli plan           # heutigen Tagesplan anzeigen
python -m theo.tag.cli dienst         # laufend automatisch erinnern
```

Beispiel-Tagesplan, den Theo selbst zusammenstellt:

```
Theos Plan für Monday, 01.06.2026
========================================
07:05  🔁 Sport  (30 Min)            <- als Gewohnheit erkannt
07:40  🔁 Kaffee  (30 Min)           <- als Gewohnheit erkannt
08:10  ✅ Steuererklärung  (60 Min)  <- Aufgabe, in Lücke geplant
09:30  📌 Standup-Meeting  (15 Min)  <- fester Termin
```

**Gewohnheits-Erkennung:** Theo gruppiert deinen Verlauf nach Tätigkeit und
Wochentag. Wiederholt sich etwas an genügend Tagen zur ähnlichen Zeit, wird es
als Gewohnheit mit Konfidenz erkannt – nachvollziehbar, keine Blackbox.

**Push aufs Handy:** über einen kostenlosen Telegram-Bot (Token als
`THEO_TG_TOKEN`/`THEO_TG_CHAT` setzen) – sonst Desktop-Benachrichtigung oder Konsole.

| Datei | Inhalt |
|---|---|
| `theo/tag/modell.py` | Datentypen (Termin, Aufgabe, Aktivität) + lokale Speicherung |
| `theo/tag/gewohnheiten.py` | Gewohnheiten aus dem Verlauf erkennen |
| `theo/tag/planer.py` | Tagesplan bauen (Termine + Gewohnheiten + Aufgaben) |
| `theo/tag/erinnerung.py` | fällige Erinnerungen + Push-Kanäle |
| `theo/tag/cli.py` | Kommandozeile |

---

## 🔒 Theo Akte — der lokale Doku-Assistent

Findet in Sekunden die richtige Stelle in hunderten Seiten Akten. Mit
Quellenangabe. Ohne Internet.

```bash
# 1) Einen Ordner mit Dokumenten indizieren (.txt, .md, .pdf)
python -m theo.akte.cli index /pfad/zu/meinen/dokumenten

# 2) In normaler Sprache fragen
python -m theo.akte.cli frage "Wie lang ist die Kündigungsfrist?"
```

Beispiel-Ausgabe (mit den mitgelieferten Beispiel-Dokumenten in `beispiele/akten`):

```
Frage: Wie lang ist die Kündigungsfrist für den Mieter?
============================================================
[1] mietvertrag.txt (Abschnitt 1, Relevanz 2.7)
    § 5 Kündigung – Die Kündigungsfrist für den Mieter beträgt drei Monate
    zum Monatsende. Die Kündigung bedarf der Schriftform. ...
```

**Technik:** lokaler BM25-Suchindex (derselbe Algorithmus wie in großen
Suchmaschinen), mit Sonderbehandlung für deutsche zusammengesetzte Wörter
(„Urlaub" findet auch „Erholungsurlaub"). Alles nachvollziehbar, keine Blackbox,
keine Internetverbindung nötig.

| Datei | Inhalt |
|---|---|
| `theo/akte/dokumente.py` | Dateien einlesen & in zitierbare Abschnitte schneiden |
| `theo/akte/suche.py` | lokaler BM25-Suchindex |
| `theo/akte/cli.py` | Kommandozeile (`index`, `frage`) |

### Fahrplan zum Produkt

1. **Lokale Suche (erledigt):** findet die relevanten Stellen offline. ✅
2. **Semantische Suche:** lokales Embedding-Modell → findet auch sinngleiche
   Formulierungen (Synonyme), nicht nur Stichwörter.
3. **Lokale Antworten:** ein offline-LLM (z. B. via `llama.cpp`/Ollama) formuliert
   aus den Fundstellen eine fertige Antwort — weiterhin 100 % lokal.
4. **Oberfläche:** einfache Desktop-/Web-App zum Reinziehen von Dokumenten.
5. **Erstkunde:** eine Kanzlei/Praxis als Pilot — echtes Feedback, erster Umsatz.

> **Ehrlich:** „Garantierte Millionen" kann niemand versprechen. Aber dieser Weg —
> ein klarer Bedarf, ein zahlungsbereiter Kunde, Datenschutz als Trumpf — ist real.

---

## 🧠 Theo Lab — unser eigenes Sprachmodell von Grund auf

Ein kleiner GPT-Transformer (dieselbe Grundarchitektur wie ChatGPT), den wir
selbst gebaut und trainiert haben — als Lernfundament, voll auf Deutsch
kommentiert. Trainiert auf **gemeinfreiem** Text (Goethe, *Die
Wahlverwandtschaften* † 1832 → urheberrechtsfrei, keine personenbezogenen Daten).

```bash
python -m theo.cli train --max-iters 5000        # trainieren (CPU genügt)
python -m theo.cli schreibe --start "Eduard "    # Theo schreiben lassen
```

Echtes Beispiel (Start: „Charlotte ", val-loss 1.32):

```
Charlotte und daß dir Zeit des Gesellschaft hatten will auf einen dir einsten
Vorhältnisse zu nehmen. Als wünschen sollte sich Ottilien ihre Zug, ...
```

Noch wackelige Grammatik, aber echte, selbst gelernte deutsche Wörter inkl. der
korrekten Romanfiguren. Mehr in [`beispiele/erste_texte.md`](beispiele/erste_texte.md).

| Datei | Inhalt |
|---|---|
| `theo/model.py` | Transformer-Architektur (Self-Attention, Blöcke) |
| `theo/data.py` / `theo/corpus.py` | Daten vorbereiten & gemeinfreien Text laden |
| `theo/train.py` / `theo/generate.py` | trainieren & Text erzeugen |

---

## Installation & Tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## ⚖️ Legal & sauber

- Nur **gemeinfreie** bzw. **eigene/lokale** Daten — keine personenbezogenen
  Daten ohne Grundlage, keine urheberrechtlich geschützten Texte.
- Theo Akte verarbeitet alles **lokal** → DSGVO-freundlich „by design".

## Lizenz

Code: MIT (siehe `LICENSE`). Trainingstext: gemeinfrei.
