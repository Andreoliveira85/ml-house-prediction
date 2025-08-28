import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Make 'src' importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

# Import FastAPI app
from api.main import app


class DummyModel:
    def predict(self, df):
        # Return a constant for determinism in tests
        return np.array([123456.78] * len(df))


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    """
    Ensure the API has a model during tests without needing a real artifact on disk.
    The tutorial uses a global 'model' loaded on startup; we set it here.
    """
    import api.main as main_mod
    main_mod.model = DummyModel()


client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "House Price Prediction API" in r.json().get("message", "")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") in {"ok", "healthy", "ready", "UP", "OK"}


def test_predict_valid():
    payload = {
        "MedInc": 5.0,
        "HouseAge": 10.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.2,
        "Population": 3000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "predicted_price" in body
    assert body["features_used"]["MedInc"] == payload["MedInc"]


def test_predict_invalid_schema():
    # Missing required fields should trigger validation error
    r = client.post("/predict", json={"MedInc": 5.0})
    assert r.status_code in (400, 422)