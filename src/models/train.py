import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import mlflow
from data.data_loader import DataLoader
from models.model import HousePricePredictor

def main():
    # Set MLflow to use local file storage instead of remote server
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name("house-price-prediction")
        if experiment is None:
            mlflow.create_experiment("house-price-prediction")
        mlflow.set_experiment("house-price-prediction")
    except Exception as e:
        print(f"Using default experiment due to: {e}")
        # Just use default experiment if there are issues
    
    # Load data
    print("Loading data...")
    loader = DataLoader()
    loader.load_california_housing()
    X_train, X_test, y_train, y_test = loader.get_train_test_split()
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Initialize and train model
    print("Training model...")
    model = HousePricePredictor(n_estimators=100)
    train_metrics = model.train(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    test_metrics = model.evaluate(X_test, y_test)
    
    print("Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save_model("models/house_price_model.joblib")
    
    print("Training completed successfully!")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print("You can view experiments by running: mlflow ui")

if __name__ == "__main__":
    main()