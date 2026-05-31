import pytest

from theo.video.detection import Detection, Detector
from theo.video.pipeline import VideoPipeline

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


class FakeDetector(Detector):
    """Liefert zwei sich bewegende Spieler – deterministisch, ohne Modell."""

    name = "fake"

    def __init__(self):
        self._t = 0

    def detect(self, frame):
        self._t += 1
        offset = self._t * 3
        return [
            Detection(100 + offset, 300, 20, 40, 1.0, "person"),
            Detection(500 - offset, 300, 20, 40, 1.0, "person"),
        ]


def _make_video(path, frames=30, size=(320, 240), fps=10):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        return False
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # Ein bewegtes Rechteck erzeugt Bewegung für die Motion-Analyse.
        x = (i * 7) % (size[0] - 30)
        cv2.rectangle(frame, (x, 100), (x + 30, 160), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return True


def test_pipeline_runs_end_to_end(tmp_path):
    video = tmp_path / "clip.avi"
    if not _make_video(video):
        pytest.skip("VideoWriter nicht verfügbar (kein Codec).")

    pipeline = VideoPipeline(detector=FakeDetector(), sample_fps=5.0)
    result = pipeline.process(video, max_seconds=None)

    assert result.sampled_frames > 0
    assert result.detector_name == "fake"
    assert result.avg_players == pytest.approx(2.0)
    assert result.formation is not None
    assert result.formation.player_count == 2
    assert result.play is not None
    # Summary darf nicht crashen und enthält Kerninfos.
    text = result.summary()
    assert "Detektor: fake" in text


def test_pipeline_missing_file():
    with pytest.raises(FileNotFoundError):
        VideoPipeline(detector=FakeDetector()).process("nicht_da.avi")
