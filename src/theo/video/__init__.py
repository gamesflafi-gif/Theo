"""Video-Analyse für Spiel- und Trainingsaufnahmen.

Dieses Modul definiert die Pipeline-Schnittstelle. Die rechenintensive Computer
Vision (Spielererkennung, Tracking, Formations-/Spielzug-Erkennung) kommt in
Stufe 2 und ist über das Extra `theo[video]` (OpenCV) optional.
"""

from theo.video.analyzer import (
    AnalysisResult,
    VideoAnalyzer,
    VideoMetadata,
    analyze_video,
)

__all__ = [
    "VideoAnalyzer",
    "AnalysisResult",
    "VideoMetadata",
    "analyze_video",
]
