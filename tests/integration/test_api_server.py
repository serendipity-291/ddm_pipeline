from fastapi.testclient import TestClient
import pytest

import api_server


@pytest.mark.integration
def test_health_endpoint():
    with TestClient(api_server.app) as client:
        res = client.get("/health")
    assert res.status_code == 200
    payload = res.json()
    assert "status" in payload
    assert "model_ready" in payload


@pytest.mark.integration
def test_inject_fault_validation():
    with TestClient(api_server.app) as client:
        res = client.post("/sim/inject?fault_type=IR")
    assert res.status_code == 400
    assert "Invalid fault_type" in res.json()["detail"]
