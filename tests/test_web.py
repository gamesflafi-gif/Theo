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
