"""Visualisierung der Analyse: Detektionen & Infos ins Frame zeichnen.

Reine Zeichenfunktionen auf BGR-Frames (numpy), damit man die Erkennung *sieht*:
Spieler-/Ball-Boxen, Line of Scrimmage und eine Kopfzeile mit Formationsinfo.
"""

from __future__ import annotations

import base64

from theo.video.detection import Detection
from theo.video.formations import FormationSnapshot

# Farben (BGR).
_COLOR_PERSON = (90, 220, 90)
_COLOR_BALL = (40, 140, 240)
_COLOR_LOS = (60, 60, 230)
_COLOR_TEXT = (255, 255, 255)
_COLOR_BANNER = (28, 22, 18)


def draw_detections(
    frame,
    detections: list[Detection],
    *,
    los_x: float | None = None,
    formation: FormationSnapshot | None = None,
    title: str | None = None,
):
    """Zeichnet Detektionen, LOS und eine Info-Kopfzeile auf eine Frame-Kopie."""
    import cv2

    img = frame.copy()
    h, w = img.shape[:2]
    thick = max(1, w // 320)

    # Line of Scrimmage (vertikale Linie).
    if los_x is not None:
        x = int(los_x)
        cv2.line(img, (x, 0), (x, h), _COLOR_LOS, thick)

    # Detektions-Boxen.
    for det in detections:
        color = _COLOR_BALL if det.label == "ball" else _COLOR_PERSON
        cv2.rectangle(img, (det.x, det.y), (det.x + det.w, det.y + det.h),
                      color, thick)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(img, label, (det.x, max(12, det.y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick, color, max(1, thick // 2),
                    cv2.LINE_AA)

    # Kopfzeile mit Titel / Formationsinfo.
    lines = []
    if title:
        lines.append(title)
    if formation:
        lines.append(
            f"Spieler: {formation.player_count} "
            f"(L{formation.players_left}/R{formation.players_right}) | "
            f"{formation.descriptor}"
        )
    if lines:
        banner_h = 6 + 20 * len(lines)
        cv2.rectangle(img, (0, 0), (w, banner_h), _COLOR_BANNER, -1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (8, 18 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_TEXT, 1, cv2.LINE_AA)
    return img


def encode_jpeg(frame, *, max_width: int = 720, quality: int = 80) -> bytes:
    """Skaliert ein Frame herunter und kodiert es als JPEG-Bytes."""
    import cv2

    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG-Kodierung fehlgeschlagen.")
    return buf.tobytes()


def encode_jpeg_data_url(frame, **kwargs) -> str:
    """Frame als `data:`-URL (base64-JPEG) – direkt im Browser anzeigbar."""
    b64 = base64.b64encode(encode_jpeg(frame, **kwargs)).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


__all__ = ["draw_detections", "encode_jpeg", "encode_jpeg_data_url"]
