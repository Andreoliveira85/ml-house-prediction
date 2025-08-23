from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import List
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using ML",
    version="1.0.0"
)

# Global variable to store the model
model = None

class HouseFeatures(BaseModel):
    MedInc: float  # Median income
    HouseAge: float  # House age
    AveRooms: float  # Average rooms
    AveBedrms: float  # Average bedrooms
    Population: float  # Population
    AveOccup: float  # Average occupancy
    Latitude: float  # Latitude
    Longitude: float  # Longitude

class PredictionResponse(BaseModel):
    predicted_price: float
    features_used: dict

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model
    
    # Try multiple possible paths
    possible_paths = [
        "models/house_price_model.joblib",  # From project root
        "../models/house_price_model.joblib",  # From src directory
        "../../models/house_price_model.joblib",  # From src/api directory
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "house_price_model.joblib"),
        os.path.join(os.getcwd(), "models", "house_price_model.joblib")
    ]
    
    model_loaded = False
    for model_path in possible_paths:
        try:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                print(f"Model loaded successfully from {model_path}")
                model_loaded = True
                break
        except Exception as e:
            continue
    
    if not model_loaded:
        print("Error: Could not find model file. Available paths checked:")
        for path in possible_paths:
            print(f"  - {path} (exists: {os.path.exists(path)})")
        print("\nMake sure to run the training script first:")
        print("  cd src && python models/train.py")
        model = None

@app.get("/")
async def root():
    return {"message": "House Price Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """Predict house price based on features"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        return PredictionResponse(
            predicted_price=round(prediction, 2),
            features_used=feature_dict
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(features_list: List[HouseFeatures]):
    """Predict prices for multiple houses"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        features_data = [f.dict() for f in features_list]
        df = pd.DataFrame(features_data)
        
        # Make predictions
        predictions = model.predict(df)
        
        results = []
        for i, prediction in enumerate(predictions):
            results.append({
                "house_id": i,
                "predicted_price": round(prediction, 2),
                "features": features_data[i]
            })
        
        return {"predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)