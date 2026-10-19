"""Integration test for complete FastAPI application factory and lifespan."""

from fastapi.testclient import TestClient

from server.main import create_app


def test_app_lifespan_and_routes():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
