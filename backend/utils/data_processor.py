import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os

class DataProcessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None
        self.required_columns = []  # Add your required columns here
        
    def validate_data(self) -> bool:
        """Validate the CSV data structure and content."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Do NOT block on missing values for preview
        # Optionally, you can log or count missing values here if you want
        
        return True
        
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the data for model input."""
        # Add your preprocessing steps here
        # For example:
        # - Handle categorical variables
        # - Scale numerical features
        # - Remove outliers
        # - Feature engineering
        
        processed_data = self.data.copy()
        
        # Example preprocessing steps:
        # 1. Convert categorical variables to numerical
        # 2. Scale numerical features
        # 3. Handle missing values
        
        return processed_data
        
    def process(self, skip_load=False) -> pd.DataFrame:
        """Main processing pipeline. If skip_load is True, use self.data as is."""
        try:
            # Load data if not already set
            if not skip_load or self.data is None:
                ext = os.path.splitext(self.filepath)[1].lower()
                if ext == '.csv':
                    self.data = pd.read_csv(self.filepath)
                elif ext in ['.xlsx', '.xls']:
                    self.data = pd.read_excel(self.filepath)
                else:
                    raise Exception(f"Unsupported file extension: {ext}")
            # Validate data
            self.validate_data()
            # Preprocess data
            processed_data = self.preprocess_data()
            return processed_data
        except Exception as e:
            raise Exception(f"Error processing data: {str(e)}")
            
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the processed data."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        return {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "summary": self.data.describe().to_dict(),
            "missing_values": self.data.isnull().sum().to_dict()
        }

    @staticmethod
    def extract_project_ids(filepath: str) -> List:
        """Parse the uploaded file, read the column 'projectid' (case-insensitive), and return unique project IDs."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            raise Exception(f"Unsupported file extension: {ext}")
        # Find the projectid column (case-insensitive)
        projectid_col = None
        for col in df.columns:
            if col.lower() in ['projectid', 'project_id']:
                projectid_col = col
                break
        if projectid_col is None:
            raise Exception("No 'projectid' or 'project_id' column found in file.")
        return list(df[projectid_col].dropna().unique()) 