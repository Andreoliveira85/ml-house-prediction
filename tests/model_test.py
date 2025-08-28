import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make 'src' importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

# Import the predictor from your project
from models.model import HousePricePredictor  # src/models/model.py


# --- FIX: completely neutralize MLflow during unit tests ---
@pytest.fixture(autouse=True)
def silence_mlflow(monkeypatch):
    """
    Neutralize MLflow so unit tests don't require a tracking server or artifact store.
    We no-op start_run, log_param/metric, log_model, and log_artifacts.
    """
    try:
        import mlflow
        import mlflow.sklearn  # ensure submodule is importable
        from mlflow.tracking import fluent as _fluent

        class NoopRun:
            def __enter__(self): return self
            def __exit__(self, *args): pass

        # No-op the common MLflow calls
        monkeypatch.setattr(mlflow, "start_run", lambda *a, **k: NoopRun(), raising=False)
        monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None, raising=False)
        monkeypatch.setattr(mlflow, "log_metric", lambda *a, **k: None, raising=False)

        # Critically: prevent artifact logging that triggers repository resolution
        monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *a, **k: None, raising=False)
        monkeypatch.setattr(_fluent, "log_artifacts", lambda *a, **k: None, raising=False)
    except Exception:
        # If mlflow isn't installed or structure differs, just ignore.
        pass


@pytest.fixture
def tiny_data():
    # 8 features expected by the tutorial’s API schema (California housing-like)
    X = pd.DataFrame({
        "MedInc":    [5.0, 3.2, 8.1, 1.5, 4.4, 2.3, 6.7, 3.9],
        "HouseAge":  [10, 15, 5, 30, 25, 18, 12, 20],
        "AveRooms":  [6.0, 5.1, 7.2, 4.0, 5.6, 5.0, 6.5, 5.8],
        "AveBedrms": [1.1, 1.0, 1.2, 0.9, 1.0, 1.1, 1.2, 1.0],
        "Population":[3000, 1500, 4000, 800, 2200, 1800, 3600, 2000],
        "AveOccup":  [3.0, 2.5, 3.1, 2.0, 2.8, 2.9, 3.0, 2.7],
        "Latitude":  [34.0, 36.1, 37.3, 33.9, 35.2, 34.8, 36.5, 35.9],
        "Longitude": [-118.0, -121.5, -122.3, -117.9, -120.1, -118.7, -121.9, -120.6],
    })
    y = pd.Series(
        [250000, 180000, 400000, 120000, 220000, 200000, 350000, 210000],
        name="MedianHouseValue"
    )
    return X, y


def _has_any_key(d: dict, candidates: set) -> bool:
    return any(k in d for k in candidates)


def test_pipeline_creation():
    model = HousePricePredictor(n_estimators=5, random_state=0)
    model.create_pipeline()
    assert model.pipeline is not None


def test_train_and_predict_shape(tiny_data):
    X, y = tiny_data
    model = HousePricePredictor(n_estimators=5, random_state=0)
    metrics = model.train(X, y)
    assert model.is_trained is True

    # Accept either plain names or train_/val_ prefixed names
    assert _has_any_key(metrics, {"rmse", "train_rmse", "val_rmse"})
    assert _has_any_key(metrics, {"mae", "train_mae", "val_mae"})
    assert _has_any_key(metrics, {"r2", "train_r2", "val_r2", "r2_score"})

    preds = model.predict(X.head(3))
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3,)


def test_save_then_load(tmp_path, tiny_data):
    X, y = tiny_data
    model = HousePricePredictor(n_estimators=5, random_state=0)
    model.train(X, y)

    out = tmp_path / "house_price_model.joblib"
    model.save_model(str(out))
    assert out.exists()

    # New instance should be able to load and predict
    model2 = HousePricePredictor()
    model2.load_model(str(out))
    p2 = model2.predict(X.head(2))
    assert p2.shape == (2,)


def test_save_raises_when_not_trained(tmp_path):
    model = HousePricePredictor()
    with pytest.raises(ValueError):
        model.save_model(str(tmp_path / "x.joblib"))