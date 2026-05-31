import pytest

from theo.video.analyzer import (
    MotionSegment,
    VideoAnalyzer,
    VideoMetadata,
)


def test_metadata_duration():
    m = VideoMetadata("x.mp4", 1920, 1080, 30.0, 900)
    assert m.duration_seconds == pytest.approx(30.0)


def test_metadata_duration_zero_fps():
    m = VideoMetadata("x.mp4", 0, 0, 0.0, 0)
    assert m.duration_seconds == 0.0


def test_segments_from_timeline():
    analyzer = VideoAnalyzer(motion_threshold=5.0)
    timeline = [
        (0.0, 1.0), (0.25, 2.0),      # ruhig
        (0.5, 8.0), (0.75, 9.0),      # aktiv
        (1.0, 1.0),                   # ruhig
        (1.25, 7.0),                  # aktiv (einzeln)
    ]
    segs = analyzer._segments_from_timeline(timeline)
    assert len(segs) == 2
    assert all(isinstance(s, MotionSegment) for s in segs)
    assert segs[0].peak_intensity == pytest.approx(9.0)


def test_missing_file_raises(tmp_path):
    pytest.importorskip("cv2")
    with pytest.raises(FileNotFoundError):
        VideoAnalyzer().analyze(tmp_path / "fehlt.mp4")
