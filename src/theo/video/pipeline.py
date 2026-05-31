"""High-Level-Video-Pipeline: Detektion + Tracking + Analyse in einem Durchlauf.

Kombiniert die Bausteine zu einer auswertbaren Analyse eines Football-Clips:
Spielererkennung pro Frame, Tracking über Frames, Bewegungs-/Snap-Erkennung,
Pre-Snap-Formation und eine grobe Lauf/Pass-Schätzung.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from theo.video.analyzer import MotionSegment, VideoMetadata
from theo.video.detection import Detection, Detector, get_detector
from theo.video.formations import (
    FormationSnapshot,
    PlayEstimate,
    estimate_formation,
    estimate_play,
)
from theo.video.tracking import CentroidTracker, Track


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Die Video-Pipeline benötigt OpenCV. Installiere es mit "
            "`pip install theo[video]`."
        ) from exc
    return cv2


@dataclass
class PipelineResult:
    metadata: VideoMetadata
    detector_name: str
    sampled_frames: int
    avg_players: float
    max_players: int
    active_segments: list[MotionSegment] = field(default_factory=list)
    formation: FormationSnapshot | None = None
    play: PlayEstimate | None = None
    tracks: list[Track] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        m = self.metadata
        lines = [
            f"Video: {m.path}",
            f"Auflösung: {m.width}x{m.height} @ {m.fps:.1f} fps, "
            f"Dauer {m.duration_seconds:.1f}s",
            f"Detektor: {self.detector_name} | analysierte Frames: "
            f"{self.sampled_frames}",
            f"Spieler erkannt: Ø {self.avg_players:.1f}, max {self.max_players}",
            f"Verfolgte Objekte (Tracks): {len(self.tracks)}",
            f"Aktive Phasen (Bewegung): {len(self.active_segments)}",
        ]
        if self.formation:
            f = self.formation
            lines += [
                "Pre-Snap-Formation (Heuristik):",
                f"  {f.descriptor}",
                f"  Spieler: {f.player_count} (links {f.players_left} / "
                f"rechts {f.players_right}), Konfidenz {f.confidence:.0%}",
            ]
        if self.play:
            p = self.play
            lines += [
                f"Spielzug-Schätzung (Heuristik): {p.play_type} "
                f"(Konfidenz {p.confidence:.0%})",
                f"  {p.reasoning}",
            ]
        if self.notes:
            lines.append("Hinweise:")
            lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


class VideoPipeline:
    """Führt die komplette Detektions- und Analyse-Pipeline auf einem Clip aus."""

    def __init__(
        self,
        detector: Detector | str = "hog",
        *,
        sample_fps: float = 2.0,
        motion_threshold: float = 6.0,
        tracker: CentroidTracker | None = None,
    ):
        self.detector = (
            detector if isinstance(detector, Detector) else get_detector(detector)
        )
        self.sample_fps = sample_fps
        self.motion_threshold = motion_threshold
        self.tracker = tracker or CentroidTracker()

    def process(
        self, video_path: str | Path, *, max_seconds: float | None = 30.0
    ) -> PipelineResult:
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
            return self._run(cv2, cap, meta, max_seconds)
        finally:
            cap.release()

    def _run(self, cv2, cap, meta: VideoMetadata, max_seconds) -> PipelineResult:
        notes: list[str] = []
        if meta.fps <= 0:
            notes.append("Keine FPS-Information – Analyse eingeschränkt.")
            fps = 30.0
        else:
            fps = meta.fps

        stride = max(1, int(round(fps / self.sample_fps)))
        max_frames = int(max_seconds * fps) if max_seconds else None

        player_counts: list[int] = []
        timeline: list[tuple[float, float]] = []
        detections_by_time: list[tuple[float, list[Detection]]] = []
        prev_small = None
        idx = 0
        sampled = 0

        while True:
            if max_frames is not None and idx >= max_frames:
                break
            # grab() schiebt nur vor (kein Dekodieren); retrieve() dekodiert
            # ausschließlich die tatsächlich gesampelten Frames -> schneller.
            if not cap.grab():
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = idx / fps
                # Bewegungsanalyse (Frame-Differenz auf Mini-Graustufenbild).
                small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
                if prev_small is not None:
                    intensity = float(cv2.absdiff(small, prev_small).mean())
                    timeline.append((t, intensity))
                prev_small = small

                # Detektion + Tracking.
                dets = self.detector.detect(frame)
                persons = [d for d in dets if d.label == "person"]
                player_counts.append(len(persons))
                detections_by_time.append((t, dets))
                self.tracker.update(dets)
                sampled += 1
            idx += 1

        active = self._segments(timeline)
        avg_players = sum(player_counts) / len(player_counts) if player_counts else 0.0
        max_players = max(player_counts) if player_counts else 0

        formation = self._formation_near_snap(
            detections_by_time, active, (meta.height, meta.width)
        )
        play = estimate_play(list(self.tracker.tracks.values()))

        if sampled == 0:
            notes.append("Keine Frames analysiert (leeres oder unlesbares Video).")
        if avg_players < 2 and sampled > 0:
            notes.append(
                "Sehr wenige Spieler erkannt – evtl. Kameraperspektive/Auflösung "
                "ungünstig oder den YOLO-Detektor (theo[video-yolo]) nutzen."
            )

        return PipelineResult(
            metadata=meta,
            detector_name=self.detector.name,
            sampled_frames=sampled,
            avg_players=avg_players,
            max_players=max_players,
            active_segments=active,
            formation=formation,
            play=play,
            tracks=list(self.tracker.tracks.values()),
            notes=notes,
        )

    def _formation_near_snap(
        self, detections_by_time, active, frame_shape
    ) -> FormationSnapshot | None:
        if not detections_by_time:
            return None
        # Zeitpunkt: kurz vor Beginn der ersten aktiven Phase (Snap-Kandidat),
        # sonst der Frame mit den meisten erkannten Spielern.
        target_t = active[0].start_s if active else None
        if target_t is not None:
            best = min(detections_by_time, key=lambda it: abs(it[0] - target_t))
        else:
            best = max(detections_by_time, key=lambda it: len(it[1]))
        return estimate_formation(best[1], frame_shape)

    def _segments(self, timeline) -> list[MotionSegment]:
        segments: list[MotionSegment] = []
        start = None
        peak = 0.0
        last_t = 0.0
        for t, intensity in timeline:
            if intensity >= self.motion_threshold:
                if start is None:
                    start, peak = t, intensity
                else:
                    peak = max(peak, intensity)
                last_t = t
            elif start is not None:
                segments.append(MotionSegment(start, last_t, peak))
                start, peak = None, 0.0
        if start is not None:
            segments.append(MotionSegment(start, last_t, peak))
        return segments


def process_video(
    video_path: str | Path, detector: str = "hog", **kwargs
) -> PipelineResult:
    """Komfortfunktion: komplette Pipeline mit Standard-Einstellungen."""
    return VideoPipeline(detector=detector).process(video_path, **kwargs)


__all__ = ["VideoPipeline", "PipelineResult", "process_video"]
