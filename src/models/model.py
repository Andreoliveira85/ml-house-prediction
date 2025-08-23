import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import Dict, Any
import mlflow
import mlflow.sklearn

class HousePricePredictor:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.pipeline = None
        self.is_trained = False
        
    def create_pipeline(self):
        """Create ML pipeline with preprocessing and model"""
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Train the model"""
        if self.pipeline is None:
            self.create_pipeline()
            
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("random_state", self.random_state)
            
            # Train model
            self.pipeline.fit(X_train, y_train)
            self.is_trained = True
            
            # Make predictions on training set
            y_pred = self.pipeline.predict(X_train)
            
            # Calculate metrics
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
                'train_mae': mean_absolute_error(y_train, y_pred),
                'train_r2': r2_score(y_train, y_pred)
            }
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(self.pipeline, "model")
            
            print(f"Training completed. RMSE: {metrics['train_rmse']:.2f}")
            return metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model on test set"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
            
        y_pred = self.pipeline.predict(X_test)
        
        metrics = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred)
        }
        
        # Log test metrics to MLflow
        with mlflow.start_run():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        return self.pipeline.predict(X)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        self.pipeline = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")