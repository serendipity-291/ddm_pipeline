import os

import pytest
import requests


@pytest.mark.e2e
def test_api_health_smoke():
    base_url = os.environ.get("E2E_BASE_URL")
    if not base_url:
        pytest.skip("E2E_BASE_URL is not set; skipping e2e smoke test.")

    response = requests.get(f"{base_url}/api/health", timeout=10)
    assert response.status_code == 200
    assert response.json().get("status") == "healthy"
