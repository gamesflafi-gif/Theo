import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from theo.video.annotate import (  # noqa: E402
    draw_detections,
    encode_jpeg,
    encode_jpeg_data_url,
)
from theo.video.detection import Detection  # noqa: E402
from theo.video.formations import FormationSnapshot  # noqa: E402


def _frame():
    return np.zeros((180, 320, 3), dtype=np.uint8)


def test_draw_detections_changes_pixels():
    frame = _frame()
    dets = [Detection(50, 60, 30, 60, 0.9, "person")]
    out = draw_detections(frame, dets, los_x=160)
    assert out.shape == frame.shape
    # Original bleibt unangetastet (Kopie), Ausgabe enthält Zeichnung.
    assert not np.array_equal(out, frame)
    assert frame.sum() == 0


def test_draw_with_formation_banner():
    f = FormationSnapshot(22, 160, 11, 11, 40.0, "ausgewogene Aufstellung", 0.7)
    out = draw_detections(_frame(), [], formation=f, title="Snap @ 1.0s")
    assert out[:30].sum() > 0  # Banner oben gezeichnet


def test_encode_jpeg_magic_bytes():
    data = encode_jpeg(_frame())
    assert data[:2] == b"\xff\xd8"  # JPEG SOI-Marker


def test_encode_jpeg_data_url():
    url = encode_jpeg_data_url(_frame())
    assert url.startswith("data:image/jpeg;base64,")
