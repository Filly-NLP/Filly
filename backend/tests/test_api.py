from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "service": "FILLY"}

def test_analyze_endpoint():
    res = client.post("/api/analyze", json={"text": "aq nman kc"})
    assert res.status_code == 200
    data = res.json()
    assert "normalizations" in data
    assert len(data["normalizations"]) == 3
    # Check Normalizations
    words = [n["word"] for n in data["normalizations"]]
    assert "aq" in words
    assert "nman" in words
    assert "kc" in words

def test_document_crud():
    # Create Document
    doc_payload = {"title": "Test Title", "content": "aq nman kc"}
    res = client.post("/api/document", json=doc_payload)
    assert res.status_code == 200
    doc_data = res.json()
    assert doc_data["title"] == "Test Title"
    assert doc_data["content"] == "aq nman kc"
    doc_id = doc_data["id"]
    
    # Retrieve Document
    res = client.get(f"/api/document/{doc_id}")
    assert res.status_code == 200
    assert res.json()["title"] == "Test Title"
    
    # List Documents
    res = client.get("/api/documents")
    assert res.status_code == 200
    docs = res.json()
    assert len(docs) > 0
    assert any(d["id"] == doc_id for d in docs)
    
    # Update Document
    res = client.put(f"/api/document/{doc_id}", json={"content": "bata din"})
    assert res.status_code == 200
    assert res.json()["content"] == "bata din"
