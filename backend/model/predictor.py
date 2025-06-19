import numpy as np
import pandas as pd
from typing import Union, List
import joblib
import os

class MLPredictor:
    def __init__(self, model_path: str = None):
        """
        Initialize the ML predictor.
        
        Args:
            model_path (str, optional): Path to the saved model file. If None, a new model will be created.
        """
        self.model = None
        self.model_path = model_path or 'model/saved_model.pkl'
        
        # Load model if it exists
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.initialize_model()
    
    def load_model(self):
        """Load the saved model."""
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def initialize_model(self):
        """
        Initialize and fit a dummy model if no trained model is found.
        Replace this with your actual model training and fitting logic.
        """
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier()
        # --- DUMMY FIT: Replace this with your real model training ---
        # Fit on random data so the model is always fitted
        X_dummy = np.random.rand(10, 3)
        y_dummy = np.random.randint(0, 2, 10)
        self.model.fit(X_dummy, y_dummy)
        # Save the dummy model so it doesn't have to be refit every time
        self.save_model()
        print("[WARNING] Using a dummy-fitted RandomForestClassifier. Replace with your real model.")
    
    def save_model(self):
        """Save the current model."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")
    
    def predict(self, data: pd.DataFrame) -> Union[np.ndarray, List]:
        """
        Make predictions on the input data.
        
        Args:
            data (pd.DataFrame): Processed input data
            
        Returns:
            Union[np.ndarray, List]: Model predictions
        """
        if self.model is None:
            raise ValueError("Model not initialized")
            
        try:
            predictions = self.model.predict(data)
            return predictions
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities if the model supports it.
        
        Args:
            data (pd.DataFrame): Processed input data
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("Model does not support probability predictions")
            
        try:
            probabilities = self.model.predict_proba(data)
            return probabilities
        except Exception as e:
            raise Exception(f"Error getting prediction probabilities: {str(e)}") 