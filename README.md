# Theo 🏈

**Theo** ist eine KI rund um American Football. Das Ziel: eine Anwendung, die

1. **alle Football-Fragen beantwortet** – Regeln, Positionen/Rollen, Spielzüge,
   Strafen, Begriffe – aus einer kuratierten Wissensbasis, optional verfeinert
   über die Claude API (RAG), und
2. **Spiel- und Trainingsvideos analysiert** – von Basis-Bewegungsanalyse bis hin
   zu Spieler-Tracking und Spielzug-Erkennung.

Das Projekt wird **stufenweise** aufgebaut und ist nach jeder Stufe lauffähig.

## Status

| Stufe | Inhalt | Status |
|-------|--------|--------|
| 1 | Wissensbasis + Q&A-Engine (lokal + optional Claude) + CLI + Video-Basisanalyse | ✅ fertig |
| 2 | Computer Vision: Spieler-/Ball-Erkennung (HOG/YOLO), Tracking, Formations- & Spielzug-Schätzung | ✅ fertig |
| 3 | Web-/Upload-Oberfläche (Fragen stellen + Videos hochladen) | ✅ fertig |
| 4 | **Installierbare App (PWA)** – auf Handy & Desktop installierbar, offline-fähige Shell | ✅ fertig |
| 5 | **Spielzug-Simulator** – Offense-Play vs. Defense-Play, animierter Verlauf + Ausgangsverteilung | ✅ fertig |
| 6 | Lernende Komponente (trainiertes Modell), Live-Daten | 🔜 geplant |

## Installation

```bash
# Basis (nur Q&A, keine schweren Abhängigkeiten):
pip install -e .

# Mit Claude-Backend für bessere Antworten (RAG):
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=sk-...

# Mit Video-Analyse (OpenCV, HOG-Detektor):
pip install -e ".[video]"

# Mit genauerem YOLO-Detektor (Spieler + Ball, zieht PyTorch):
pip install -e ".[video-yolo]"

# Mit Weboberfläche (Fragen + Video-Upload):
pip install -e ".[web]"

# Alles inkl. Tests:
pip install -e ".[all]"
```

## Nutzung

```bash
# Eine Frage stellen
theo ask "Was macht der Quarterback?"
theo ask "Wie viele Punkte gibt ein Field Goal?"

# Interaktiver Chat
theo chat

# Inhalte der Wissensbasis auflisten
theo topics

# Video analysieren – Basisanalyse (Metadaten + Bewegungs-/Snap-Erkennung)
theo analyze spiel.mp4

# Video analysieren – volle CV-Pipeline (Spieler, Tracking, Formation, Spielzug)
theo analyze spiel.mp4 --detect --detector hog     # oder: --detector yolo

# Weboberfläche starten (Fragen stellen + Videos hochladen)
theo serve                      # http://127.0.0.1:8000
```

### Weboberfläche

`theo serve` startet eine Single-Page-Oberfläche mit zwei Funktionen:
**Fragen stellen** (nutzt dieselbe Q&A-Engine) und **Video hochladen** zur
Analyse (max. 30 s; das Video wird nach der Analyse serverseitig gelöscht).
API-Endpunkte: `POST /api/ask`, `POST /api/analyze`, `GET /api/health`.

### Als App installieren (PWA)

Theo ist eine **Progressive Web App** und lässt sich als eigenständige App
installieren – auf Handy wie Desktop, ohne App Store:

1. `theo serve` starten und die Adresse im Browser öffnen.
2. **Desktop (Chrome/Edge):** Install-Symbol in der Adressleiste oder den Button
   „📲 Als App installieren" nutzen.
3. **Handy (Android/iOS):** Browser-Menü → „Zum Startbildschirm hinzufügen".

Die App startet dann im eigenen Fenster (Standalone), hat ein eigenes Icon und
eine offline-fähige Shell (Service Worker). Fragen-/Video-Funktionen brauchen
weiterhin den laufenden Server.

> Für die Installation auf anderen Geräten muss der Server erreichbar sein
> (`theo serve --host 0.0.0.0`) – idealerweise über HTTPS, da PWAs außerhalb von
> `localhost` einen sicheren Kontext verlangen.

### Spielzug-Simulator

Im Bereich **„Spielzug-Simulator"** der Web-App lässt sich ein Offense-Play gegen
ein Defense-Play antreten:
- Offense- und Defense-Spielzug aus der Bibliothek wählen (z. B. *Four Verticals*
  vs. *Cover 2*) oder über „Routen anpassen" einen **eigenen Spielzug** bauen.
- **▶ Simulieren** zeigt den Verlauf als Animation auf dem Feld (Offense blau,
  Defense rot, Ball) samt Ausgang (komplett/inkomplett/Sack/INT/Lauf, Yards).
- **📊 100× simulieren** rechnet eine **Ausgangsverteilung** (Ø Yards, beste/
  schlechteste, Quoten je Ausgang).

Das Modell ist bewusst vereinfacht (Routen, Mann-/Zone-Deckung, Pass-Rush,
Separation am Catch), bildet Football-Tendenzen aber plausibel ab – z. B. schlägt
*Smash* eine *Cover 2*, während ein *Blitz* tiefe Pässe zu Sacks zwingt.

API: `GET /api/plays`, `POST /api/simulate`, `POST /api/simulate/batch`.

Ohne `ANTHROPIC_API_KEY` antwortet Theo **extraktiv** direkt aus der Wissensbasis.
Mit Key und installiertem `anthropic`-Paket formuliert Claude die Antwort als RAG
über die gefundenen Abschnitte (Modus `auto`).

### Als Bibliothek

```python
from theo import QAEngine

engine = QAEngine()                 # mode="auto"
antwort = engine.ask("Was ist ein Blitz?")
print(antwort.text)
print(antwort.source_titles)
```

## Projektstruktur

```
src/theo/
  knowledge/        Wissensbasis (Markdown) + Loader/Abschnitts-Parser
    data/*.md       Football-Wissen (Grundlagen, Scoring, Positionen, Plays, Regeln, Glossar)
  qa/
    retriever.py    TF-IDF-Retrieval über die Abschnitte (dependency-frei)
    llm.py          optionales Claude-Backend (RAG)
    engine.py       Q&A-Engine (verbindet Retrieval + LLM/Extraktion)
  video/
    analyzer.py     Basisanalyse (Metadaten + Bewegungs-/Snap-Erkennung)
    detection.py    Detektoren (HOG-Baseline, optional YOLO)
    tracking.py     CentroidTracker (IDs über Frames)
    formations.py   Formations- & Spielzug-Heuristik (mit Konfidenz)
    pipeline.py     volle Pipeline: Detektion + Tracking + Analyse
  simulation/
    model.py        Datenmodell (Frames, Ergebnis, Verteilung)
    plays.py        Routen, Formationen, Spielzug-Bibliothek
    engine.py       Simulations-Engine + Monte-Carlo
  web/
    app.py          FastAPI-Backend + Single-Page-Oberfläche (inkl. Simulator)
  cli.py            Kommandozeile (ask/chat/analyze/topics/serve)
tests/              pytest-Tests
```

## Wissensbasis erweitern

Lege einfach neue oder erweiterte `.md`-Dateien unter
`src/theo/knowledge/data/` ab. Überschriften (`#`, `##`, `###`) definieren die
durchsuchbaren Abschnitte – kein Code-Eingriff nötig.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```
