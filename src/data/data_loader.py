import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from typing import Tuple
import os

class DataLoader:
    def __init__(self):
        self.data = None
        self.target = None
        
    def load_california_housing(self) -> pd.DataFrame:
        """Load California housing dataset"""
        housing = fetch_california_housing()
        
        # Create DataFrame
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['target'] = housing.target
        
        # Convert target to actual house prices (multiply by 100k)
        df['target'] = df['target'] * 100000
        
        self.data = df
        return df
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_california_housing() first.")
            
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def save_data(self, filepath: str):
        """Save data to CSV"""
        if self.data is None:
            raise ValueError("Data not loaded.")
        self.data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
