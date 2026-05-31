from theo.video.detection import Detection
from theo.video.formations import estimate_formation, estimate_play
from theo.video.tracking import Track


def person(x, y):
    return Detection(x=x, y=y, w=20, h=40, confidence=1.0, label="person")


def test_formation_empty():
    snap = estimate_formation([], (720, 1280))
    assert snap.player_count == 0
    assert snap.confidence == 0.0


def test_formation_counts_left_right():
    # 4 links der Mitte, 4 rechts.
    dets = [person(x, 360) for x in (100, 150, 200, 250, 800, 850, 900, 950)]
    snap = estimate_formation(dets, (720, 1280))
    assert snap.player_count == 8
    assert snap.players_left == 4
    assert snap.players_right == 4


def test_formation_spread_descriptor():
    # Breit gestreute y-Positionen -> "gespreizt".
    wide = [person(640, y) for y in range(0, 720, 60)]
    snap = estimate_formation(wide, (720, 1280))
    assert "gespreizt" in snap.descriptor or "breite" in snap.descriptor


def _track(tid, points):
    t = Track(id=tid, label="person", centroid=points[-1],
              bbox=(0, 0, 10, 10), history=list(points))
    return t


def test_play_too_few_tracks():
    est = estimate_play([_track(0, [(0, 0), (1, 1), (2, 2)])])
    assert est.play_type == "unklar"


def test_play_pass_like():
    # Ein Track bewegt sich weit, andere kaum -> Pass-Indiz.
    tracks = [
        _track(0, [(0, 0), (50, 0), (120, 0), (220, 0)]),  # weit
        _track(1, [(0, 0), (2, 0), (3, 0), (4, 0)]),
        _track(2, [(0, 0), (1, 0), (2, 0), (3, 0)]),
        _track(3, [(0, 0), (2, 1), (3, 1), (4, 1)]),
    ]
    est = estimate_play(tracks)
    assert est.play_type in {"pass", "unklar"}
    assert 0.0 <= est.confidence <= 1.0
