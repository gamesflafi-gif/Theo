# Theo Radar 📡 — Signalverarbeitung & Zielverfolgung

**Theo Radar ist das „Gehirn" eines zivilen Radars: vollständige
Signalverarbeitung von der Rohmessung bis zur verfolgten Ziel-Spur — in reinem
Python, nachvollziehbar und getestet.**

Simuliert wird ein **FMCW-Radar** (wie in Auto-, Drohnen- und Verkehrssensorik).
Die verwendeten Algorithmen sind **identisch mit denen auf echter
Radar-Hardware** — es fehlt nur die HF-Elektronik/Antenne (in Deutschland
frequenz-reguliert über die Bundesnetzagentur).

> **Anwendung & Recht:** ausgelegt auf **zivile, sicherheitsorientierte** Zwecke
> (Drohnen-Erkennung, Luftraum-/Verkehrsüberwachung, Kollisionswarnung). Keine
> militärischen Waffen-/Feuerleitsysteme.

## Was Theo Radar kann

| Schritt | Verfahren |
|---|---|
| 📥 Signal erzeugen | realistisches FMCW-Echo mehrerer Ziele inkl. Rauschen (SNR) |
| 🗺️ Range-Doppler-Karte | 2D-FFT → Entfernung & Geschwindigkeit sichtbar machen |
| 🎯 Zielerkennung | **CA-CFAR** (Constant False Alarm Rate) + Non-Maximum-Suppression |
| 🛰️ Verfolgung | **Kalman-Filter** + Nächste-Nachbar-Zuordnung (Multi-Target) |
| 📊 Visualisierung | Range-Doppler-Karte & Track-Verlauf als PNG |

## Schnellstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Komplette Demo: 3 bewegte Ziele erkennen & verfolgen, Bilder speichern
python -m theo.radar.cli demo --frames 25 --snr 18 --plots ausgabe
```

Verifizierte Beispiel-Ausgabe (77-GHz-Parameter, gut getrennte Ziele):

```
Theo Radar – Demo
Auflösung: 0.75 m / 0.30 m/s | Reichweite: 96 m, ±19 m/s
25 Messungen, SNR 18 dB

Letzte Messung: 3 Detektionen, 3 bestätigte Tracks:
  Track 2:  29.5 m |  +6.1 m/s | 25 Treffer
  Track 3:  50.2 m |  +0.0 m/s | 25 Treffer
  Track 1:  73.1 m |  -4.9 m/s | 25 Treffer
```

Mit `--plots ausgabe` entstehen zwei Bilder: die **Range-Doppler-Karte** (Ziele
als helle Punkte, erkannte Ziele mit rotem X) und der **Track-Verlauf** (jede
Ziel-Spur über die Zeit).

## Technische Eckdaten (Standard-Konfiguration)

- Trägerfrequenz 77 GHz, Bandbreite 200 MHz → **Entfernungsauflösung 0,75 m**
- 128 Chirps → **Geschwindigkeitsauflösung 0,30 m/s**
- nutzbare Reichweite ~96 m, Geschwindigkeit ±19 m/s
- alle Werte in `theo/radar/szenario.py` einstellbar

## Aufbau

| Datei | Inhalt |
|---|---|
| `theo/radar/szenario.py` | FMCW-Radar-Parameter & Ziele (mit Auflösungs-Formeln) |
| `theo/radar/signal.py` | FMCW-Empfangssignal erzeugen (Datenwürfel + Rauschen) |
| `theo/radar/verarbeitung.py` | Range-Doppler-FFT + CA-CFAR-Erkennung |
| `theo/radar/tracking.py` | Kalman-Filter-Multi-Target-Tracking |
| `theo/radar/plot.py` | Visualisierung (PNG) |
| `theo/radar/cli.py` | Demo-Kommandozeile |

## Tests

```bash
python -m pytest -q
```

Die Tests prüfen mit harten Zusicherungen, dass Entfernung & Geschwindigkeit
korrekt gemessen werden und der Tracker bewegten Zielen stabil folgt (5 Tests).

## Fahrplan

1. **Erkennung + Tracking (erledigt):** FMCW-Pipeline mit CFAR & Kalman. ✅
2. **Winkelschätzung:** mehrere Antennen → Azimut/Elevation, damit aus
   Entfernung+Geschwindigkeit echte 3D-Positionen werden.
3. **Klassifikation:** Ziel-Typ (Drohne / Vogel / Fahrzeug) aus dem Doppler-Profil.
4. **Echte Signale:** Empfang über einen günstigen SDR (nur Empfang ist
   genehmigungsfrei) statt Simulation.

## Lizenz

MIT (siehe `LICENSE`).
