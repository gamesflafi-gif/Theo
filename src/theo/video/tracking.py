"""Einfaches Multi-Object-Tracking per Centroid-Zuordnung.

Weist Detektionen über Frames hinweg stabile IDs zu (greedy nach Distanz). Rein
in Python/NumPy – ohne Modell, voll testbar. Für robustes Tracking kann später
ein dedizierter Tracker (z. B. ByteTrack über YOLO) eingehängt werden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from theo.video.detection import Detection


@dataclass
class Track:
    """Ein über mehrere Frames verfolgtes Objekt."""

    id: int
    label: str
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    history: list[tuple[float, float]] = field(default_factory=list)
    frames_seen: int = 1
    frames_missing: int = 0

    @property
    def total_distance(self) -> float:
        """Zurückgelegte Strecke des Centroids über die History (in Pixeln)."""
        dist = 0.0
        for (x1, y1), (x2, y2) in zip(self.history, self.history[1:]):
            dist += math.hypot(x2 - x1, y2 - y1)
        return dist


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class CentroidTracker:
    """Greedy-Centroid-Tracker mit Distanzschwelle und Verschwinde-Toleranz."""

    def __init__(self, *, max_distance: float = 80.0, max_missing: int = 5):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self._next_id = 0
        self.tracks: dict[int, Track] = {}

    def _new_track(self, det: Detection) -> None:
        c = det.centroid
        self.tracks[self._next_id] = Track(
            id=self._next_id,
            label=det.label,
            centroid=c,
            bbox=(det.x, det.y, det.w, det.h),
            history=[c],
        )
        self._next_id += 1

    def update(self, detections: list[Detection]) -> dict[int, Track]:
        """Aktualisiert die Tracks mit den Detektionen eines Frames."""
        if not self.tracks:
            for det in detections:
                self._new_track(det)
            return self.tracks

        track_ids = list(self.tracks.keys())
        unmatched_dets = set(range(len(detections)))

        # Alle (Track, Detektion)-Paare nach Distanz sortieren, greedy zuordnen.
        pairs = []
        for tid in track_ids:
            tc = self.tracks[tid].centroid
            for di, det in enumerate(detections):
                d = _distance(tc, det.centroid)
                if d <= self.max_distance:
                    pairs.append((d, tid, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks: set[int] = set()
        for _, tid, di in pairs:
            if tid in matched_tracks or di not in unmatched_dets:
                continue
            det = detections[di]
            t = self.tracks[tid]
            t.centroid = det.centroid
            t.bbox = (det.x, det.y, det.w, det.h)
            t.history.append(det.centroid)
            t.frames_seen += 1
            t.frames_missing = 0
            matched_tracks.add(tid)
            unmatched_dets.discard(di)

        # Nicht zugeordnete Tracks altern lassen / entfernen.
        for tid in track_ids:
            if tid not in matched_tracks:
                t = self.tracks[tid]
                t.frames_missing += 1
                if t.frames_missing > self.max_missing:
                    del self.tracks[tid]

        # Übrige Detektionen werden zu neuen Tracks.
        for di in unmatched_dets:
            self._new_track(detections[di])

        return self.tracks


__all__ = ["Track", "CentroidTracker"]
