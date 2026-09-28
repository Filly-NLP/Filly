from __future__ import annotations

import asyncio
import inspect

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.database import Base, get_db
from app.main import app
import app.main as main_module
from app.api import endpoints


@pytest.fixture
def client(monkeypatch):
    """Run the real application lifecycle against an isolated in-memory DB."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    test_sessions = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = test_sessions()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(main_module, "init_db", lambda: Base.metadata.create_all(engine))
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def test_startup_fails_closed_when_normalizer_resources_cannot_load(monkeypatch):
    from app.services import normalizer as normalizer_module

    monkeypatch.setattr(main_module, "init_db", lambda: None)

    def fail_to_load_normalizer(**_kwargs):
        raise FileNotFoundError("normalizer resources missing")

    monkeypatch.setattr(normalizer_module, "get_normalizer", fail_to_load_normalizer)

    async def start_application():
        async with main_module.lifespan(app):
            pytest.fail("startup continued without required normalizer resources")

    with pytest.raises(FileNotFoundError, match="normalizer resources missing"):
        asyncio.run(start_application())


def test_startup_fails_closed_when_gec_checkpoint_cannot_load(monkeypatch):
    from app.services import gec as gec_module

    monkeypatch.setattr(main_module, "init_db", lambda: None)

    def fail_to_load_checkpoint(*, device=None):
        raise FileNotFoundError("GEC checkpoint missing")

    monkeypatch.setattr(gec_module, "get_gec_service", fail_to_load_checkpoint)

    async def start_application():
        async with main_module.lifespan(app):
            pytest.fail("startup continued without the GEC checkpoint")

    with pytest.raises(FileNotFoundError, match="GEC checkpoint missing"):
        asyncio.run(start_application())


def test_health_routes_report_ready_services(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["ready"] is True
    assert response.json()["gec_iterations"] == 5
    assert response.json()["device"] == "cpu" or response.json()["device"].startswith("cuda")
    assert client.get("/api/health").json() == {"status": "ok", "service": "FILLY"}


def test_normalize_route_runs_normalizer_independently(client):
    response = client.post("/api/v1/normalize", json={"text": "aq ay"})

    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == "aq ay"
    assert data["normalized_text"] == "ako ay"
    assert data["changes"][0]["start"] == 0
    assert data["changes"][0]["word"] == "aq"


def test_gec_route_runs_five_dependent_passes_without_normalizing(client):
    source = "kumain ako"
    response = client.post("/api/v1/gec", json={"text": source})

    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == source
    assert data["gec"]["iterations"] == 5
    assert len(data["gec"]["passes"]) == 5
    assert data["gec"]["passes"][0]["input_text"] == source
    assert all(item["model_invoked"] for item in data["gec"]["passes"])
    for previous, current in zip(data["gec"]["passes"], data["gec"]["passes"][1:]):
        assert current["input_text"] == previous["output_text"]
    assert data["corrected_text"] == data["gec"]["passes"][-1]["output_text"]


def test_analyze_runs_normalization_before_gec_and_preserves_safe_suggestions(client):
    source = "aq ay"
    response = client.post("/api/v1/analyze", json={"text": source})

    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == source
    assert data["normalized_text"] == "ako ay"
    assert data["gec"]["passes"][0]["input_text"] == data["normalized_text"]
    assert data["gec"]["iterations"] == 5
    for previous, current in zip(data["gec"]["passes"], data["gec"]["passes"][1:]):
        assert current["input_text"] == previous["output_text"]
    assert data["corrected_text"] == data["gec"]["passes"][-1]["output_text"]
    suggestions = data["suggestions"]
    assert any(
        item["start"] == 0
        and item["end"] == 2
        and item["original"] == "aq"
        and item["source"] in {"normalization", "combined"}
        for item in suggestions
    )
    assert any(item["source"] in {"gec", "combined"} for item in suggestions)
    assert all(item["source"] in {"normalization", "gec", "combined"} for item in suggestions)
    ordered_suggestions = sorted(suggestions, key=lambda item: (item["start"], item["end"]))
    for previous, current in zip(ordered_suggestions, ordered_suggestions[1:]):
        assert current["start"] >= previous["end"]

    accepted_all = source
    for item in reversed(ordered_suggestions):
        assert accepted_all[item["start"] : item["end"]] == item["original"]
        accepted_all = (
            accepted_all[: item["start"]]
            + item["replacement"]
            + accepted_all[item["end"] :]
        )
    assert accepted_all == data["corrected_text"]


def test_empty_and_unicode_inputs_are_preserved(client):
    empty = client.post("/api/v1/analyze", json={"text": ""})
    unicode = client.post("/api/v1/normalize", json={"text": "🙂 AQ, po!"})

    assert empty.status_code == 200
    assert empty.json()["original_text"] == ""
    assert empty.json()["normalized_text"] == ""
    assert empty.json()["corrected_text"] == ""
    assert empty.json()["gec"]["iterations"] == 0
    assert unicode.status_code == 200
    assert unicode.json()["normalized_text"] == "🙂 AKO, po!"
    assert unicode.json()["changes"][0]["start"] == 2


def test_invalid_and_excessive_inputs_are_rejected(client):
    assert client.post("/api/v1/analyze", json={}).status_code == 422
    assert client.post("/api/v1/gec", json={"text": 42}).status_code == 422
    assert client.post("/api/v1/normalize", json={"text": "x" * 5001}).status_code == 422


def test_gec_token_limit_returns_422_and_normalization_remains_independent(client):
    text = "kumain " * 520
    assert len(text) < 5000

    analyze = client.post("/api/v1/analyze", json={"text": text})
    gec = client.post("/api/v1/gec", json={"text": text})
    normalize = client.post("/api/v1/normalize", json={"text": text})

    assert analyze.status_code == 422
    assert "token limit" in analyze.json()["detail"]
    assert gec.status_code == 422
    assert "token limit" in gec.json()["detail"]
    assert normalize.status_code == 200
    assert normalize.json()["normalized_text"] == text


def test_model_backed_document_handlers_are_sync_threadpool_routes():
    assert not inspect.iscoroutinefunction(endpoints.create_document)
    assert not inspect.iscoroutinefunction(endpoints.update_document)


def test_legacy_analysis_route_remains_available(client):
    response = client.post("/api/analyze", json={"text": "aq nman kc"})

    assert response.status_code == 200
    data = response.json()
    assert {item["word"] for item in data["normalizations"]} == {"aq", "nman", "kc"}
    assert "grammar_corrections" in data


def test_document_routes_remain_available(client):
    response = client.post("/api/document", json={"title": "Test Title", "content": ""})
    assert response.status_code == 200
    document = response.json()
    assert document["title"] == "Test Title"
    assert document["content"] == ""

    fetched = client.get(f"/api/document/{document['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["title"] == "Test Title"

    updated = client.put(f"/api/document/{document['id']}", json={"content": ""})
    assert updated.status_code == 200
    assert updated.json()["content"] == ""
