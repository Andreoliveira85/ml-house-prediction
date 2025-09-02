"""
FastAPI app for house price prediction.

Robust model loading rules (in order):
1) MODEL_PATH env var, if it exists and the file is present
2) Container default: /app/models/house_price_model.joblib
3) Project-relative default: <repo_root>/models/house_price_model.joblib
4) Auto-discover the first *.joblib under a nearby models/ folder

Health endpoint returns {"status":"healthy"} when the model is loaded,
so simple grep-based checks pass in CI/Cloud Run scripts.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import os
import glob
import joblib
import logging
from typing import List, Optional

log = logging.getLogger("uvicorn.error")

# ----------------------------
# Pydantic schema
# ----------------------------
class HouseInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# ----------------------------
# Global state
# ----------------------------
model = None
loaded_model_path: Optional[Path] = None  # for diagnostics

# ----------------------------
# Model path helpers
# ----------------------------
def _candidate_paths() -> List[Path]:
    """Return candidate paths to try, in priority order."""
    # 1) Explicit env
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        yield Path(env_path)

    # 2) Container default
    yield Path("/app/models/house_price_model.joblib")

    # 3) Project-relative default
    #    repo_root ≈ two levels up from this file: src/api/main.py -> repo root
    here = Path(__file__).resolve()
    repo_root = here.parents[2] if len(here.parents) >= 3 else here.parent
    yield repo_root / "models" / "house_price_model.joblib"


def _discover_joblib() -> Optional[Path]:
    """Try to find any *.joblib near the source tree."""
    here = Path(__file__).resolve()

    # Prefer a nearby 'models' directory in current/ancestor paths
    for ancestor in [here.parent, *here.parents]:
        models_dir = ancestor / "models"
        if models_dir.is_dir():
            hits = sorted(models_dir.rglob("*.joblib"))
            if hits:
                return hits[0]

    # Last resort: search the repo root (can be slower on huge repos)
    repo_root = here.parents[2] if len(here.parents) >= 3 else here.parent
    hits = sorted(repo_root.rglob("*.joblib"))
    return hits[0] if hits else None


def _list_dir_safe(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except Exception as e:
        log.warning("Could not list %s: %s", path, e)
        return []

# ----------------------------
# FastAPI app with lifespan
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, loaded_model_path

    # Breadcrumbs for Cloud Run logs
    log.info("CWD=%s", os.getcwd())
    log.info("Listing /app: %s", _list_dir_safe("/app"))
    log.info("Listing /app/models: %s", _list_dir_safe("/app/models"))

    # Also log a recursive peek to help spot the model file
    try:
        listing = glob.glob("/app/models/**/*", recursive=True)
        log.info("Recursive list of /app/models (first 50): %s", listing[:50])
    except Exception as e:
        log.warning("Could not recursively list /app/models: %s", e)

    # Try candidates in order
    candidate = None
    tried = []
    for p in _candidate_paths():
        tried.append(str(p))
        if p.exists():
            candidate = p
            log.info("Using model candidate: %s", candidate)
            break

    # Fallback discovery
    if not candidate:
        fallback = _discover_joblib()
        if fallback:
            log.warning("MODEL_PATH candidates missing; discovered model at: %s", fallback)
            candidate = fallback
        else:
            log.error(
                "No model file found. Checked candidates: %s; "
                "no *.joblib discovered under nearby models/ directories.",
                tried,
            )

    # Load if we have a path
    if candidate:
        try:
            model = joblib.load(candidate)
            loaded_model_path = candidate
            log.info("Model loaded OK from %s", candidate)
        except Exception as e:
            log.exception("Failed to load model from %s: %s", candidate, e)

    # Hand control back to FastAPI
    yield

    # (Optional) cleanup on shutdown


app = FastAPI(lifespan=lifespan)

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    ok = model is not None
    # Return "healthy"/"unhealthy" for simple grep-based checks
    return {
        "status": "healthy" if ok else "unhealthy",
        "model_loaded": ok,
        "model_path_env": os.getenv("MODEL_PATH"),
        "loaded_model_path": str(loaded_model_path) if loaded_model_path else None,
    }


@app.post("/predict")
def predict(payload: HouseInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = [[
        payload.MedInc, payload.HouseAge, payload.AveRooms, payload.AveBedrms,
        payload.Population, payload.AveOccup, payload.Latitude, payload.Longitude
    ]]
    try:
        y = model.predict(X)
    except Exception as e:
        # Surface model/runtime issues clearly
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    return {"predicted_price": float(y[0])}

# ----------------------------
# Local dev entrypoint
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)