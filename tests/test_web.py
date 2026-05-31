import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from theo.web.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Theo" in r.text
    assert "Video analysieren" in r.text


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_returns_answer(client):
    r = client.post("/api/ask", data={"question": "Was ist ein Touchdown?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert "touchdown" in body["answer"].lower()
    assert "backend" in body


def test_ask_empty_rejected(client):
    r = client.post("/api/ask", data={"question": "   "})
    assert r.status_code == 400


def test_analyze_rejects_bad_format(client):
    files = {"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    r = client.post("/api/analyze", files=files, data={"detect": "true"})
    assert r.status_code == 400


def test_manifest_served(client):
    r = client.get("/manifest.webmanifest")
    assert r.status_code == 200
    assert "manifest" in r.headers["content-type"]
    body = r.json()
    assert body["name"].startswith("Theo")
    assert body["display"] == "standalone"
    assert len(body["icons"]) >= 2


def test_service_worker_served(client):
    r = client.get("/sw.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]
    assert "addEventListener" in r.text


def test_icons_served(client):
    for size in (192, 512):
        r = client.get(f"/static/icons/icon-{size}.png")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"


def test_index_links_pwa(client):
    html = client.get("/").text
    assert 'rel="manifest"' in html
    assert "/sw.js" in html


def test_plays_listed(client):
    d = client.get("/api/plays").json()
    assert any(p["id"] == "four_verticals" for p in d["offense"])
    assert any(p["id"] == "cover2_zone" for p in d["defense"])
    assert "slant" in d["routes"]
    assert {s["id"] for s in d["slots"]} >= {"WR_L", "RB"}


def test_simulate_endpoint(client):
    r = client.post("/api/simulate",
                    json={"offense_id": "slant_flat", "defense_id": "cover2_zone",
                          "seed": 1})
    assert r.status_code == 200
    d = r.json()
    assert d["outcome"] in {"complete", "incomplete", "sack", "interception", "run"}
    assert d["frames"] and len(d["ball"]) == len(d["frames"])


def test_simulate_custom_routes(client):
    r = client.post("/api/simulate", json={
        "routes": {"WR_L": "post", "WR_R": "go", "SLOT": "slant",
                   "TE": "out", "RB": "flat"},
        "defense_id": "cover3_zone", "seed": 2})
    assert r.status_code == 200
    assert "summary" in r.json()


def test_simulate_batch_endpoint(client):
    r = client.post("/api/simulate/batch",
                    json={"offense_id": "four_verticals",
                          "defense_id": "cover2_zone", "n": 50})
    assert r.status_code == 200
    d = r.json()
    assert d["n"] == 50
    assert sum(d["outcomes"].values()) == 50


def test_simulate_bad_defense(client):
    r = client.post("/api/simulate",
                    json={"offense_id": "slant_flat", "defense_id": "nope"})
    assert r.status_code == 400


def test_upload_too_large_rejected(client, monkeypatch):
    # Upload-Limit künstlich klein setzen und ein größeres "Video" senden.
    import importlib

    appmod = importlib.import_module("theo.web.app")
    monkeypatch.setattr(appmod, "_MAX_UPLOAD_MB", 0.001)  # ~1 KB
    big = io.BytesIO(b"x" * 50_000)
    files = {"file": ("clip.mp4", big, "video/mp4")}
    r = client.post("/api/analyze", files=files, data={"detect": "false"})
    assert r.status_code == 413
