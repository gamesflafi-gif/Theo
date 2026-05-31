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
| 4 | Lernende Komponente (Spielzüge/Strategie), Live-Daten | 🔜 geplant |

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
  web/
    app.py          FastAPI-Backend + Single-Page-Oberfläche
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
