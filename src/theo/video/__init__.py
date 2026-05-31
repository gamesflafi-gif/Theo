"""Video-Analyse für Spiel- und Trainingsaufnahmen.

Zwei Ebenen:
- :func:`analyze_video` / :class:`VideoAnalyzer` – schnelle Basisanalyse
  (Metadaten + Bewegungs-/Snap-Erkennung), nur OpenCV nötig.
- :func:`process_video` / :class:`VideoPipeline` – volle CV-Pipeline mit
  Spielererkennung, Tracking, Formations- und Spielzug-Schätzung. Mit dem
  HOG-Baseline-Detektor sofort nutzbar, optional mit YOLO (``theo[video-yolo]``).
"""

from theo.video.analyzer import (
    AnalysisResult,
    MotionSegment,
    VideoAnalyzer,
    VideoMetadata,
    analyze_video,
)
from theo.video.detection import (
    Detection,
    Detector,
    HOGPeopleDetector,
    YOLODetector,
    get_detector,
)
from theo.video.formations import (
    FormationSnapshot,
    PlayEstimate,
    estimate_formation,
    estimate_play,
)
from theo.video.annotate import (
    draw_detections,
    encode_jpeg,
    encode_jpeg_data_url,
)
from theo.video.pipeline import (
    Keyframe,
    PipelineResult,
    VideoPipeline,
    process_video,
    render_annotated_video,
)
from theo.video.tracking import CentroidTracker, Track

__all__ = [
    # Basisanalyse
    "VideoAnalyzer", "AnalysisResult", "VideoMetadata", "MotionSegment",
    "analyze_video",
    # Detektion
    "Detection", "Detector", "HOGPeopleDetector", "YOLODetector", "get_detector",
    # Tracking
    "CentroidTracker", "Track",
    # Formation / Spielzug
    "FormationSnapshot", "PlayEstimate", "estimate_formation", "estimate_play",
    # Annotation
    "draw_detections", "encode_jpeg", "encode_jpeg_data_url",
    # Pipeline
    "VideoPipeline", "PipelineResult", "Keyframe", "process_video",
    "render_annotated_video",
]
