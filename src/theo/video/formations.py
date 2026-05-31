"""Heuristische Formations- & Spielzug-Schätzung aus Detektionen/Tracks.

Wichtig: Dies sind **Heuristiken** mit ausgewiesener Konfidenz, keine trainierte
Klassifikation. Sie gehen von einer Seitenlinien-Perspektive (Broadcast) aus, bei
der das Feld horizontal verläuft und die Line of Scrimmage annähernd vertikal
liegt. Eine echte, robuste Erkennung braucht Feld-Kalibrierung und ein gelerntes
Modell (Roadmap Stufe 2+).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from theo.video.detection import Detection
from theo.video.tracking import Track


@dataclass
class FormationSnapshot:
    player_count: int
    los_x: float                 # geschätzte Line of Scrimmage (x-Pixel)
    players_left: int            # Spieler links der LOS
    players_right: int           # Spieler rechts der LOS
    lateral_spread: float        # vertikale Streuung (Breite der Aufstellung)
    descriptor: str              # grobe verbale Einordnung
    confidence: float            # 0..1


@dataclass
class PlayEstimate:
    play_type: str               # "lauf" | "pass" | "unklar"
    confidence: float            # 0..1
    reasoning: str


def estimate_formation(
    detections: list[Detection], frame_shape: tuple[int, int]
) -> FormationSnapshot:
    """Schätzt eine Pre-Snap-Formation aus den Personendetektionen eines Frames."""
    persons = [d for d in detections if d.label == "person"]
    h, w = frame_shape[:2]
    if not persons:
        return FormationSnapshot(0, w / 2, 0, 0, 0.0, "keine Spieler erkannt", 0.0)

    xs = [d.centroid[0] for d in persons]
    ys = [d.centroid[1] for d in persons]
    los_x = statistics.median(xs)
    left = sum(1 for x in xs if x < los_x)
    right = len(xs) - left
    spread = statistics.pstdev(ys) if len(ys) > 1 else 0.0
    spread_norm = spread / h if h else 0.0

    if spread_norm > 0.28:
        descriptor = "breite/gespreizte Aufstellung (viele Receiver außen)"
    elif spread_norm < 0.12:
        descriptor = "kompakte Aufstellung (lauflastig / enge Formation)"
    else:
        descriptor = "ausgewogene Aufstellung"

    # Konfidenz: am höchsten, wenn ~22 Spieler erkannt wurden (11 vs 11).
    n = len(persons)
    confidence = max(0.0, 1.0 - abs(n - 22) / 22.0) * 0.7  # gedeckelt: nur Heuristik
    return FormationSnapshot(
        player_count=n,
        los_x=los_x,
        players_left=left,
        players_right=right,
        lateral_spread=spread,
        descriptor=descriptor,
        confidence=round(confidence, 2),
    )


def estimate_play(tracks: list[Track], *, los_x: float | None = None) -> PlayEstimate:
    """Schätzt grob Lauf vs. Pass aus der Bewegung der Tracks nach dem Snap.

    Idee: Bei einem Pass entfernen sich einzelne Receiver weit (große, gerichtete
    Bewegung), während andere im Pocket bleiben → hohe Streuung der
    Bewegungsdistanzen. Bei einem Lauf bewegt sich die Masse kompakter vorwärts.
    Bewusst niedrige Konfidenz – nur ein Indikator.
    """
    moving = [t for t in tracks if len(t.history) >= 3]
    if len(moving) < 3:
        return PlayEstimate("unklar", 0.0, "zu wenige verfolgte Spieler")

    distances = [t.total_distance for t in moving]
    mean_d = statistics.mean(distances)
    max_d = max(distances)
    spread = statistics.pstdev(distances) if len(distances) > 1 else 0.0

    # Verhältnis Maximal- zu Durchschnittsbewegung als grober Indikator.
    ratio = max_d / mean_d if mean_d > 0 else 1.0
    if ratio > 2.2 and spread > mean_d * 0.6:
        return PlayEstimate(
            "pass", 0.4,
            "einzelne Spieler legen deutlich mehr Strecke zurück (Receiver-Routen),"
            " andere bleiben zurück (Pocket) – Indiz für ein Passspiel.",
        )
    if mean_d > 0 and ratio < 1.8:
        return PlayEstimate(
            "lauf", 0.35,
            "die Spieler bewegen sich relativ gleichmäßig/kompakt vorwärts –"
            " Indiz für ein Laufspiel.",
        )
    return PlayEstimate(
        "unklar", 0.2,
        "Bewegungsmuster nicht eindeutig genug für Lauf/Pass.",
    )


__all__ = ["FormationSnapshot", "PlayEstimate", "estimate_formation", "estimate_play"]
