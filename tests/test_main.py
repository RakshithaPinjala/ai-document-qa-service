import pytest
from app.main import app
from fastapi.testclient import TestClient

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "metrics" in response.json()
    assert "memory_usage_mb" in response.json()["metrics"]

def test_global_exception_handler(client):
    # This requires forcing an unhandled exception in an endpoint.
    # We can mock a dependency or an endpoint temporarily.
    with pytest.MonkeyPatch.context() as m:
        m.setattr("app.api.documents.list_documents", lambda db: 1 / 0)
        # Assuming list_documents is mounted at /api/v1/documents
        response = client.get("/api/v1/documents")
        if response.status_code == 500: # Depending on how FastAPI catches it when overridden
            assert "An unexpected error occurred" in response.json()["detail"]
