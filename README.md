# Theo — unser eigenes Sprachmodell von Grund auf 🧠

**Theo ist ein kleiner GPT-Transformer — dieselbe Grundarchitektur wie ChatGPT —
den wir selbst gebaut und selbst trainiert haben.** Du gibst einen Satzanfang ein,
und Theo schreibt weiter.

Kein fertiges Modell, keine API: Jede Zeile dieses neuronalen Netzes ist hier
nachvollziehbar und auf Deutsch kommentiert. Das ist der echte Kern moderner KI —
zum Anfassen und Verstehen.

## Wie es funktioniert (in einem Satz)

Theo lernt nur **eine** Aufgabe: „Welches Zeichen kommt als Nächstes?" Wenn ein
Netz das richtig gut kann, entsteht daraus die Fähigkeit, ganze Texte zu schreiben.

```
Eingabe:  "Eduard saß im "
Theo:     "Eduard saß im Garten und betrachtete die Bäume, ..."
```

## ⚖️ Legal & sauber

- **Trainingsdaten:** ausschließlich **gemeinfreier** Text — hier *Die
  Wahlverwandtschaften* von J. W. von Goethe († 1832). In Deutschland
  urheberrechtsfrei, keine personenbezogenen Daten → DSGVO- und
  urheberrechtssicher.
- Quelle: [GITenberg](https://github.com/GITenberg/Die-Wahlverwandtschaften_2403)
  (Project-Gutenberg-Mirror auf GitHub).

## Schnellstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Theo trainieren (CPU genügt; ~15-20 Min für einen ersten Eindruck)
python -m theo.cli train --max-iters 5000

# 2) Theo schreiben lassen
python -m theo.cli schreibe --start "Eduard " --laenge 400
```

Beim ersten Lauf lädt Theo den gemeinfreien Text automatisch herunter (oder nutzt
die mitgelieferte Datei `daten/korpus.txt`).

### Stellschrauben

| Schalter | Bedeutung |
|---|---|
| `--max-iters` | Anzahl der Lernschritte (mehr = besser, aber langsamer) |
| `--n-layer` / `--n-head` / `--n-embd` | Größe des Modells |
| `--temperatur` | beim Schreiben: <1 = braver, >1 = kreativer |
| `--top-k` | beim Schreiben: nur aus den k besten Zeichen wählen |

## Aufbau des Projekts

| Datei | Inhalt |
|---|---|
| `theo/model.py` | die Transformer-Architektur (Embeddings, Self-Attention, Blöcke) |
| `theo/data.py` | Text ⇄ Zahlen, Trainings-Häppchen |
| `theo/corpus.py` | gemeinfreien Text laden & säubern |
| `theo/train.py` | die Trainingsschleife |
| `theo/generate.py` | mit dem fertigen Modell Text schreiben |
| `theo/cli.py` | Kommandozeile |

## Tests

```bash
python -m pytest -q
```

## Fahrplan — wohin Theo wächst

1. **Schritt 1 (erledigt):** echter, lauffähiger GPT, der deutschen Text lernt. ✅
2. **Schritt 2:** mehr & vielfältigere gemeinfreie Texte → reichere Sprache.
3. **Schritt 3:** von Zeichen- auf Wort-/Subwort-Ebene (BPE) für besseres Deutsch.
4. **Schritt 4:** größeres Modell + (geliehene) GPU → spürbarer Qualitätssprung.
5. **Schritt 5:** auf eine echte, nützliche Aufgabe spezialisieren.

> **Ehrlich:** Ohne große Rechenpower bleibt Theo klein und schlägt ChatGPT nicht.
> Aber er ist **echt**, er **läuft**, und wir **verstehen jedes Teil**. Das ist das
> Fundament, auf dem sich Großes bauen lässt — Schritt für Schritt.

## Lizenz

Code: MIT (siehe `LICENSE`). Trainingstext: gemeinfrei.
