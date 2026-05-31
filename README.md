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
| 2 | Computer Vision: Spieler-/Ball-Erkennung, Tracking, Formations- & Spielzug-Erkennung | 🔜 geplant |
| 3 | Lernende Komponente (Spielzüge/Strategie), Web-/Upload-Oberfläche | 🔜 geplant |

## Installation

```bash
# Basis (nur Q&A, keine schweren Abhängigkeiten):
pip install -e .

# Mit Claude-Backend für bessere Antworten (RAG):
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=sk-...

# Mit Video-Analyse (OpenCV):
pip install -e ".[video]"

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

# Video analysieren (benötigt theo[video])
theo analyze spiel.mp4 --show-roadmap
```

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
    analyzer.py     Video-Pipeline (Metadaten + Bewegungsanalyse; CV in Stufe 2)
  cli.py            Kommandozeile (ask/chat/analyze/topics)
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
