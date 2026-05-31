from theo.video.detection import Detection
from theo.video.tracking import CentroidTracker


def d(x, y, label="person"):
    return Detection(x=x, y=y, w=20, h=40, confidence=1.0, label=label)


def test_assigns_and_keeps_ids():
    tr = CentroidTracker(max_distance=30)
    tr.update([d(100, 100), d(200, 200)])
    assert len(tr.tracks) == 2
    ids = set(tr.tracks)
    # Leichte Bewegung – IDs müssen erhalten bleiben.
    tr.update([d(105, 102), d(195, 205)])
    assert set(tr.tracks) == ids
    for t in tr.tracks.values():
        assert t.frames_seen == 2


def test_new_object_gets_new_id():
    tr = CentroidTracker(max_distance=30)
    tr.update([d(100, 100)])
    tr.update([d(100, 100), d(400, 400)])
    assert len(tr.tracks) == 2


def test_disappearing_track_is_dropped():
    tr = CentroidTracker(max_distance=30, max_missing=2)
    tr.update([d(100, 100)])
    # Objekt taucht nicht mehr auf -> nach max_missing Frames entfernt.
    for _ in range(3):
        tr.update([])
    assert len(tr.tracks) == 0


def test_history_tracks_movement():
    tr = CentroidTracker(max_distance=100)
    tr.update([d(0, 0)])
    tr.update([d(30, 40)])  # 50px Strecke
    (track,) = tr.tracks.values()
    assert track.total_distance == 50.0
