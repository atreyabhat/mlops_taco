"""
A simple smoke test for the FastAPI API.
"""

from fastapi.testclient import TestClient
from src.api.main import app  # Import your FastAPI app

client = TestClient(app)


def test_health_check():
    """
    Tests that the /health endpoint is working.
    """
    response = client.get("/health")

    # Check that the API returns a 200 OK status
    assert response.status_code == 200

    # Check that the model is (correctly) reported as not loaded
    assert response.json() == {"status": "error", "model_loaded": False}
