"""Ziele über mehrere Messungen verfolgen (Multi-Target-Tracking).

Eine einzelne Messung sagt nur "da ist ein Ziel". Erst wenn man Ziele über die
Zeit *verfolgt*, entsteht eine Spur (Track) – die Grundlage für Vorhersagen und
Warnungen.

Wir nutzen das Standard-Werkzeug echter Tracking-Systeme: das **Kalman-Filter**.
Es kombiniert die Vorhersage (das Ziel bewegt sich gleichförmig weiter) mit der
neuen Messung und glättet so das Rauschen heraus. Mehrere Ziele werden über eine
einfache Nächste-Nachbar-Zuordnung getrennt gehalten.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .verarbeitung import Detektion


@dataclass
class Track:
    id: int
    x: np.ndarray              # Zustand [entfernung, geschwindigkeit]
    P: np.ndarray              # Unsicherheit (Kovarianz)
    treffer: int = 1
    verfehlt: int = 0
    bestaetigt: bool = False
    verlauf: list[tuple[float, float]] = field(default_factory=list)

    @property
    def entfernung(self) -> float:
        return float(self.x[0])

    @property
    def geschwindigkeit(self) -> float:
        return float(self.x[1])


class MultiTracker:
    """Verfolgt mehrere Ziele mit je einem Kalman-Filter."""

    def __init__(self, dt: float = 0.1, mess_rausch_r: float = 0.5,
                 mess_rausch_v: float = 0.3, prozess_rausch: float = 1.0,
                 gate: float = 4.0, bestaetigen_ab: int = 3, loeschen_ab: int = 3):
        self.dt = dt
        self.gate = gate                # Zuordnungs-Schwelle (norm. Abstand)
        self.bestaetigen_ab = bestaetigen_ab
        self.loeschen_ab = loeschen_ab
        self._naechste_id = 1
        self.tracks: list[Track] = []

        # Bewegungsmodell (konstante Geschwindigkeit)
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.eye(2)               # wir messen Entfernung UND Geschwindigkeit
        self.R = np.diag([mess_rausch_r ** 2, mess_rausch_v ** 2])
        q = prozess_rausch
        self.Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])

    def _vorhersage(self, t: Track) -> None:
        t.x = self.F @ t.x
        t.P = self.F @ t.P @ self.F.T + self.Q

    def _abstand(self, t: Track, z: np.ndarray) -> float:
        """Mahalanobis-Abstand zwischen Vorhersage und Messung."""
        S = self.H @ t.P @ self.H.T + self.R
        d = z - self.H @ t.x
        return float(np.sqrt(d @ np.linalg.inv(S) @ d))

    def _aktualisiere(self, t: Track, z: np.ndarray) -> None:
        S = self.H @ t.P @ self.H.T + self.R
        K = t.P @ self.H.T @ np.linalg.inv(S)        # Kalman-Verstärkung
        t.x = t.x + K @ (z - self.H @ t.x)
        t.P = (np.eye(2) - K @ self.H) @ t.P
        t.treffer += 1
        t.verfehlt = 0
        if t.treffer >= self.bestaetigen_ab:
            t.bestaetigt = True

    def schritt(self, detektionen: list[Detektion]) -> list[Track]:
        """Verarbeitet eine Messung und gibt die aktiven Tracks zurück."""
        for t in self.tracks:
            self._vorhersage(t)

        messungen = [np.array([d.entfernung, d.geschwindigkeit]) for d in detektionen]
        zugeordnet: set[int] = set()

        # Greedy-Zuordnung: alle (Track, Messung)-Paare nach Abstand sortieren
        paare = []
        for ti, t in enumerate(self.tracks):
            for mi, z in enumerate(messungen):
                dist = self._abstand(t, z)
                if dist <= self.gate:
                    paare.append((dist, ti, mi))
        paare.sort()
        track_belegt: set[int] = set()
        for dist, ti, mi in paare:
            if ti in track_belegt or mi in zugeordnet:
                continue
            self._aktualisiere(self.tracks[ti], messungen[mi])
            track_belegt.add(ti)
            zugeordnet.add(mi)

        # Nicht zugeordnete Tracks: als "verfehlt" markieren
        for ti, t in enumerate(self.tracks):
            if ti not in track_belegt:
                t.verfehlt += 1

        # Neue Tracks aus nicht zugeordneten Messungen
        for mi, z in enumerate(messungen):
            if mi not in zugeordnet:
                self.tracks.append(Track(
                    id=self._naechste_id,
                    x=z.copy(),
                    P=np.diag([1.0, 1.0]),
                ))
                self._naechste_id += 1

        # Verlauf fortschreiben und veraltete Tracks löschen
        for t in self.tracks:
            t.verlauf.append((t.entfernung, t.geschwindigkeit))
        self.tracks = [t for t in self.tracks if t.verfehlt < self.loeschen_ab]
        return [t for t in self.tracks if t.bestaetigt]
