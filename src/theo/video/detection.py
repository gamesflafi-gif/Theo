"""Objekt-Detektion für Football-Videos (Spieler & Ball).

Pluggbare Detektoren:
- :class:`HOGPeopleDetector` nutzt den in OpenCV eingebauten HOG-Personendetektor
  und braucht **keine** zusätzlichen Abhängigkeiten (Baseline).
- :class:`YOLODetector` nutzt optional Ultralytics YOLO (deutlich genauer,
  erkennt auch den Ball) – installierbar über das Extra ``theo[video-yolo]``.

Beide liefern eine einheitliche Liste von :class:`Detection`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Detection:
    """Eine einzelne Detektion in einem Frame."""

    x: int
    y: int
    w: int
    h: int
    confidence: float
    label: str = "person"

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> int:
        return self.w * self.h

    def iou(self, other: "Detection") -> float:
        """Intersection-over-Union mit einer anderen Box."""
        ax2, ay2 = self.x + self.w, self.y + self.h
        bx2, by2 = other.x + other.w, other.y + other.h
        ix1, iy1 = max(self.x, other.x), max(self.y, other.y)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class Detector(ABC):
    """Schnittstelle für Frame-Detektoren."""

    name: str = "detector"

    @abstractmethod
    def detect(self, frame) -> list[Detection]:
        """Liefert die Detektionen für ein einzelnes BGR-Frame (numpy-Array)."""
        raise NotImplementedError


class HOGPeopleDetector(Detector):
    """Baseline-Personendetektor (OpenCV HOG + SVM), ohne Extra-Abhängigkeiten.

    Schnell aufzusetzen, aber bei vielen sich überlappenden Spielern und
    Weitwinkelaufnahmen begrenzt – gut als Standard und für Tests/Demos.
    """

    name = "hog"

    def __init__(self, *, work_width: int = 640, hit_threshold: float = 0.0):
        import cv2  # lokal: nur wenn wirklich genutzt

        self._cv2 = cv2
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.work_width = work_width
        self.hit_threshold = hit_threshold

    def detect(self, frame) -> list[Detection]:
        cv2 = self._cv2
        h, w = frame.shape[:2]
        scale = self.work_width / w if w > self.work_width else 1.0
        if scale != 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame
        rects, weights = self._hog.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.05,
            hitThreshold=self.hit_threshold,
        )
        detections: list[Detection] = []
        for (rx, ry, rw, rh), score in zip(rects, weights):
            inv = 1.0 / scale
            detections.append(
                Detection(
                    x=int(rx * inv), y=int(ry * inv),
                    w=int(rw * inv), h=int(rh * inv),
                    confidence=float(score), label="person",
                )
            )
        return detections


class YOLODetector(Detector):
    """Optionaler YOLO-Detektor (Ultralytics) – Spieler **und** Ball.

    Benötigt ``pip install theo[video-yolo]``. Lädt das Modell beim ersten
    Gebrauch (lazy). Erkennt die COCO-Klassen ``person`` und ``sports ball``.
    """

    name = "yolo"
    _PERSON_CLASS = 0
    _BALL_CLASS = 32  # "sports ball" in COCO

    def __init__(self, model: str = "yolov8n.pt", *, conf: float = 0.25):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - abhängig von Installation
            raise RuntimeError(
                "YOLO-Detektor benötigt Ultralytics. Installiere es mit "
                "`pip install theo[video-yolo]`."
            ) from exc
        self._model = YOLO(model)
        self.conf = conf

    def detect(self, frame) -> list[Detection]:
        results = self._model.predict(frame, conf=self.conf, verbose=False)
        detections: list[Detection] = []
        for res in results:
            for box in res.boxes:
                cls = int(box.cls[0])
                if cls == self._PERSON_CLASS:
                    label = "person"
                elif cls == self._BALL_CLASS:
                    label = "ball"
                else:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        x=int(x1), y=int(y1),
                        w=int(x2 - x1), h=int(y2 - y1),
                        confidence=float(box.conf[0]), label=label,
                    )
                )
        return detections


def get_detector(name: str = "hog", **kwargs) -> Detector:
    """Factory: erzeugt einen Detektor anhand seines Namens ('hog' oder 'yolo')."""
    name = name.lower()
    if name == "hog":
        return HOGPeopleDetector(**kwargs)
    if name == "yolo":
        return YOLODetector(**kwargs)
    raise ValueError(f"Unbekannter Detektor: {name!r} (erlaubt: 'hog', 'yolo')")


__all__ = [
    "Detection",
    "Detector",
    "HOGPeopleDetector",
    "YOLODetector",
    "get_detector",
]
