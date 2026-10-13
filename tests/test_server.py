import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server


@pytest.fixture(scope="module")
def client():
    return TestClient(server.app)


@pytest.fixture(scope="module")
def sample_session(client):
    r = client.post("/api/diagnose", data={"sample": "mango__bacterial_canker|0"})
    assert r.status_code == 200
    return r.json()


def test_index_and_static_assets_served(client):
    assert client.get("/").status_code == 200
    assert "Plant Disease Assistant" in client.get("/").text
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_info_lists_questions_and_samples(client):
    info = client.get("/api/info").json()
    assert {q["id"] for q in info["questions"]} == {"crop", "lesion", "weather"}
    assert len(info["samples"]) == 23 * 5
    assert info["samples"][0]["name"].endswith("sample 1")


def test_diagnose_sample_returns_ranked_result_with_images(sample_session):
    d = sample_session
    assert d["tier"] in {"confident", "uncertain", "refuse"}
    assert d["ranking"][0]["label"] == "mango__bacterial_canker"
    assert d["sample"]["true_label"] == "mango / bacterial canker" and d["sample"]["matches"] is True
    assert d["original"].startswith("data:image/jpeg;base64,")
    assert d["overlay"] is None or d["overlay"].startswith("data:image/png;base64,")


def test_upload_and_bad_inputs(client):
    buf = io.BytesIO()
    Image.new("RGB", (120, 120), (40, 120, 60)).save(buf, format="PNG")
    ok = client.post("/api/diagnose", files={"file": ("leaf.png", buf.getvalue(), "image/png")})
    assert ok.status_code == 200 and ok.json()["session_id"]
    bad = client.post("/api/diagnose", files={"file": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 400
    assert client.post("/api/diagnose", data={}).status_code == 400
    assert client.post("/api/diagnose", data={"sample": "../../etc/passwd|0"}).status_code == 400


def test_answers_advice_and_ask_flow(client, sample_session):
    sid = sample_session["session_id"]
    a = client.post("/api/answers", json={"session_id": sid, "answers": {"lesion": "angular_black", "weather": "wet"}})
    assert a.status_code == 200 and a.json()["evidence_log"]
    adv = client.post("/api/advice", json={"session_id": sid}).json()
    assert adv["sources"] and "extension" in adv["text"].lower()
    q = client.post("/api/ask", json={"session_id": sid, "question": "how does it spread?"}).json()
    assert q["sources"]
    off = client.post("/api/ask", json={"session_id": sid, "question": "what is the price of tomatoes?"}).json()
    assert off["sources"] == [] and "sourced answer" in off["text"]


def test_unknown_session_and_unknown_question(client, sample_session):
    assert client.post("/api/advice", json={"session_id": "nope"}).status_code == 404
    r = client.post("/api/answers", json={"session_id": sample_session["session_id"], "answers": {"favourite": "x"}})
    assert r.status_code == 400
    assert client.post("/api/ask", json={"session_id": sample_session["session_id"], "question": "  "}).status_code == 400
