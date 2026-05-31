"""Video-Analyse-Pipeline.

Stufe 1 liefert bereits eine echte, dependency-leichte Basisanalyse mit OpenCV:
Metadaten und einen Bewegungs-Zeitstrahl (Frame-Differenzen), aus dem sich
aktive Phasen (vermutliche Spielzüge/Snaps) ableiten lassen.

Geplant für Stufe 2 (`planned_features`): Spieler-/Ball-Erkennung (Detektion),
Tracking, Formations- und Spielzug-Klassifikation, Yard-Linien-Kalibrierung.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _require_cv2():
    try:
        import cv2  # type: ignore
        import numpy  # noqa: F401  # type: ignore
    except ImportError as exc:  # pragma: no cover - abhängig von Installation
        raise RuntimeError(
            "Video-Analyse benötigt OpenCV. Installiere es mit "
            "`pip install theo[video]`."
        ) from exc
    return cv2


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass
class MotionSegment:
    """Ein zusammenhängender Zeitabschnitt erhöhter Bewegung."""

    start_s: float
    end_s: float
    peak_intensity: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class AnalysisResult:
    metadata: VideoMetadata
    motion_timeline: list[tuple[float, float]] = field(default_factory=list)
    active_segments: list[MotionSegment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    planned_features: list[str] = field(default_factory=list)

    def summary(self) -> str:
        m = self.metadata
        lines = [
            f"Video: {m.path}",
            f"Auflösung: {m.width}x{m.height} @ {m.fps:.1f} fps",
            f"Dauer: {m.duration_seconds:.1f} s ({m.frame_count} Frames)",
            f"Erkannte aktive Phasen: {len(self.active_segments)}",
        ]
        for i, seg in enumerate(self.active_segments, 1):
            lines.append(
                f"  Phase {i}: {seg.start_s:.1f}s–{seg.end_s:.1f}s "
                f"(Dauer {seg.duration_s:.1f}s, Intensität {seg.peak_intensity:.2f})"
            )
        if self.notes:
            lines.append("Hinweise:")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


class VideoAnalyzer:
    """Analysiert ein Video und liefert ein :class:`AnalysisResult`."""

    #: Roadmap der noch folgenden CV-Analysen (Stufe 2).
    PLANNED_FEATURES = [
        "Spieler- und Ball-Erkennung (Objektdetektion)",
        "Multi-Object-Tracking der Spieler über die Frames",
        "Formations-Erkennung vor dem Snap (Pre-Snap)",
        "Spielzug-Klassifikation (Lauf/Pass/Konzept)",
        "Yard-Linien-/Feld-Kalibrierung für Raumgewinn-Messung",
        "Automatische Highlight- und Statistik-Erstellung",
    ]

    def __init__(self, *, sample_fps: float = 4.0, motion_threshold: float = 6.0):
        # Wie viele Frames pro Sekunde für die Bewegungsanalyse gesampelt werden.
        self.sample_fps = sample_fps
        # Schwelle (mittlere Pixeldifferenz), ab der ein Frame als "aktiv" gilt.
        self.motion_threshold = motion_threshold

    def analyze(self, video_path: str | Path) -> AnalysisResult:
        cv2 = _require_cv2()
        path = str(video_path)
        if not Path(path).exists():
            raise FileNotFoundError(f"Video nicht gefunden: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Video konnte nicht geöffnet werden: {path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            meta = VideoMetadata(path, width, height, fps, frame_count)

            timeline, segments, notes = self._motion_analysis(cv2, cap, meta)
        finally:
            cap.release()

        return AnalysisResult(
            metadata=meta,
            motion_timeline=timeline,
            active_segments=segments,
            notes=notes,
            planned_features=list(self.PLANNED_FEATURES),
        )

    def _motion_analysis(self, cv2, cap, meta: VideoMetadata):
        notes: list[str] = []
        if meta.fps <= 0:
            notes.append("Keine FPS-Information – Bewegungsanalyse übersprungen.")
            return [], [], notes

        stride = max(1, int(round(meta.fps / self.sample_fps)))
        timeline: list[tuple[float, float]] = []
        prev_small = None
        idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                small = cv2.cvtColor(
                    cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY
                )
                if prev_small is not None:
                    diff = cv2.absdiff(small, prev_small)
                    intensity = float(diff.mean())
                    timeline.append((idx / meta.fps, intensity))
                prev_small = small
            idx += 1

        segments = self._segments_from_timeline(timeline)
        if not timeline:
            notes.append("Zu wenige Frames für eine Bewegungsanalyse.")
        return timeline, segments, notes

    def _segments_from_timeline(
        self, timeline: list[tuple[float, float]]
    ) -> list[MotionSegment]:
        segments: list[MotionSegment] = []
        start: float | None = None
        peak = 0.0
        last_t = 0.0
        for t, intensity in timeline:
            if intensity >= self.motion_threshold:
                if start is None:
                    start = t
                    peak = intensity
                else:
                    peak = max(peak, intensity)
                last_t = t
            else:
                if start is not None:
                    segments.append(MotionSegment(start, last_t, peak))
                    start = None
                    peak = 0.0
        if start is not None:
            segments.append(MotionSegment(start, last_t, peak))
        return segments


def analyze_video(video_path: str | Path, **kwargs) -> AnalysisResult:
    """Komfortfunktion: analysiert ein Video mit Standard-Einstellungen."""
    return VideoAnalyzer(**kwargs).analyze(video_path)


__all__ = [
    "VideoAnalyzer",
    "AnalysisResult",
    "VideoMetadata",
    "MotionSegment",
    "analyze_video",
]
