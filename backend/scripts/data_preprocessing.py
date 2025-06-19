# preprocessing.py
import pandas as pd
import numpy as np
import re
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# === Initial Information ===
# =============================================
# ORDINAL ENCODING (for features with natural order)
# =============================================
ordinal_features = ['decision_time_frame', 'age', 'car_type', 'room_size_wanted',
                    'purchase_budget', 'residences_count', 'would_recommend', 'family_monthly_income',
                    'individual_monthly_income_baht', 'Project Type'
                    ]  # Add your ordinal feature names here
ordinal_categories = [['Missing', 'ภายใน 1 เดือน', '1 - 3 เดือน', '4 - 6 เดือน',
                '7 - 12 เดือน', 'มากกว่า 1 ปี'],                                # Decision Time order (use en dash)
                ['Missing', 'ต่ำกว่า 25 ปี', '25-35 ปี', '36-45 ปี', '45 ปีขึ้นไป'],   # Age order (use en dash)
                ['Missing', 'ไม่มีรถ', 'มอเตอร์ไซค์', 'มอเตอร์ไซค์บิ๊กไบค์',
                'ECO Car (Vios, Yaris, Jazz, City,  Mazda2 ...)',
                'รถกระบะ 2/4 ประตู',
                'รถเก๋ง Size M (Civic, Altis ...)','รถยนต์อเนกประสงค์ (SUV, MPV)',
                'รถเก๋ง Size L (Accord, Camry ...)',
                'รถเก๋ง Luxury (Benz ,BMW ,Lexus ,Volvo ,Audi ,Mini Cooper)'],  # CarType order (use en dash)
                ['Missing', 'น้อยกว่า 23  ตร.ม.', '23-25 ตร.ม.', '26-29 ตร.ม.','30-34 ตร.ม.', '35-39 ตร.ม.',
                '40-50 ตร.ม.', '51-60 ตร.ม.', '61-80 ตร.ม.', '81-100 ตร.ม.',
                '101 -140 ตร.ม.', '191-200 ตร.ม.', '200 - 250 ตร.ม.', ''], # Room Size Wanted
                ['Missing', "≤ 1.0M", "1.01 - 1.5M", "1.51 - 2.0M", "2.01 - 2.5M",
                "2.51 - 3.0M", "3.01 - 3.5M", "3.51 - 4.0M", "4.01 - 4.5M",
                "4.51 - 5.0M", "5.01 - 6.0M", "6.01 - 7.0M", "7.01 - 8.0M", "8.01 - 9.0M",
                "9.01 - 10.0M", "10.01 - 11.0M", "11.01 - 12.0M", "12.01 - 13.0M",
                "13.01 - 14.0M", "14.01 - 15.0M", "15.01 - 16.0M","16.01 - 17.0M",], # purchase_budget order
                ['Missing', 'หลังที่ 1', 'หลังที่ 2', 'มากกว่า 2 หลัง'], # Residences count
                ['Missing', 'ไม่บอกต่อ','บอกต่อ'], # Recommend or not
                ['Missing', '≤ 20,000', '20,001 - 35,000', '35,001 - 50,000',
                '50,001 - 65,000', '65,001 - 80,000', '80,001 - 100,000', '100,001 - 120,000',
                '120,001 - 140,000', '140,001 - 160,000', '160,001 - 180,000',
                '180,001 - 200,000', '200,001 - 300,000', '300,001 - 400,000', '≥ 400,001'], # Family Monthly Income
                ['Missing', '≤ 15,000','15,001 - 20,000', '20,001 - 30,000', '30,001 - 40,000',
                    '40,001 - 50,000', '50,001 - 65,000', '65,001 - 80,000', '80,001 - 100,000', 
                    '100,001 - 120,000', '120,001 - 150,000', '150,001 - 200,000', '200,001 - 300,000',
                    '300,001 - 400,000', '≥ 400,001'], # Individual Monthly Income
                    ['Campus Condo', 'LOW RISE', 'HIGH RISE'] # Project Type
] # Add your ordered categories lists here (one per feature)
# =============================================
# NOMINAL ENCODING (for features without order)
# =============================================
nominal_features = ['gender', 'occupation', 'marital_status', 'information_source',
                'purchasing_reason', 'decide_purchase_reason', 'not_book_reason',
                'other_projects_before_deicde', 'condo_payment', 'day_off_activity',
                'most_interested_activites_participation', 'saw_sign', 'exercise_preference',
                'condo_living_style', 'car_brand', 'purchase_intent', 'travel_route_today', 
                'Project Brand', 'Location']

# === Utility functions ===

def fix_year(dt_str):
    if pd.isna(dt_str):
        return pd.NaT
    try:
        parts = str(dt_str).split('-')
        fixed_year = int(parts[0]) - 543
        return pd.to_datetime(f"{fixed_year}-{'-'.join(parts[1:])}")
    except:
        return pd.NaT

def add_seasonal_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    df = df.copy()
    df['hour'] = df[date_column].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df.drop(columns=['hour'], inplace=True)
    
    df['day'] = df[date_column].dt.day
    df['month'] = df[date_column].dt.month
    df['quarter'] = df[date_column].dt.quarter
    df['year'] = df[date_column].dt.year
    df['week'] = df[date_column].dt.isocalendar().week
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['is_weekend'] = (df[date_column].dt.dayofweek >= 5).astype(int)
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['season'] = df[date_column].dt.month % 12 // 3 + 1
    
    return df

def group_rare_categories_by_threshold(df: pd.DataFrame, columns: List[str], threshold=0.01) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        value_counts = df[col].value_counts(normalize=True)
        rare_vals = value_counts[value_counts < threshold].index
        df[col] = df[col].apply(lambda x: 'Other' if x in rare_vals else x)
    return df

# === Financial processing ===

def create_bin_assignment_functions():
    # Define bins...
    # (Use exactly what you already have, truncated here for brevity)
     # Standardized bin ranges
    budget_bins = [
        (0, 1.01, "≤ 1.0M"),
        (1.01, 1.51, "1.01 - 1.5M"),
        (1.51, 2.01, "1.51 - 2.0M"),
        (2.01, 2.51, "2.01 - 2.5M"),
        (2.51, 3.01, "2.51 - 3.0M"),
        (3.01, 3.51, "3.01 - 3.5M"),
        (3.51, 4.01, "3.51 - 4.0M"),
        (4.01, 4.51, "4.01 - 4.5M"),
        (4.51, 5.01, "4.51 - 5.0M"),
        (5.01, 6.01, "5.01 - 6.0M"),
        (6.01, 7.01, "6.01 - 7.0M"),
        (7.01, 8.01, "7.01 - 8.0M"),
        (8.01, 9.01, "8.01 - 9.0M"),
        (9.01, 10.01, "9.01 - 10.0M"),
        (10.01, 11.01, "10.01 - 11.0M"),
        (11.01, 12.01, "11.01 - 12.0M"),
        (12.01, 13.01, "12.01 - 13.0M"),
        (13.01, 14.01, "13.01 - 14.0M"),
        (14.01, 15.01, "14.01 - 15.0M"),
        (15.01, 16.01, "15.01 - 16.0M"),
        (16.01, 17.01, "16.01 - 17.0M"),
        (17.01, 20.01, "17.01 - 20.0M"),
        (20.01, 25.01, "20.01 - 25.0M"),
        (25.01, float("inf"), "≥ 25.01M")
    ]

    income_bins = [
        (0, 20001, '≤ 20,000'),
        (20001, 35001, '20,001 - 35,000'),
        (35001, 50001, '35,001 - 50,000'),
        (50001, 65001, '50,001 - 65,000'),
        (65001, 80001, '65,001 - 80,000'),
        (80001, 100001, '80,001 - 100,000'),
        (100001, 120001, '100,001 - 120,000'),
        (120001, 140001, '120,001 - 140,000'),
        (140001, 160001, '140,001 - 160,000'),
        (160001, 180001, '160,001 - 180,000'),
        (180001, 200001, '180,001 - 200,000'),
        (200001, 300001, '200,001 - 300,000'),
        (300001, 400001, '300,001 - 400,000'),
        (400001, float('inf'), '≥ 400,001'),
    ]

    individual_income_bins = [
        (0, 15001, '≤ 15,000'),
        (15001, 20001, '15,001 - 20,000'),
        (20001, 30001, '20,001 - 30,000'),
        (30001, 40001, '30,001 - 40,000'),
        (40001, 50001, '40,001 - 50,000'),
        (50001, 65001, '50,001 - 65,000'),
        (65001, 80001, '65,001 - 80,000'),
        (80001, 100001, '80,001 - 100,000'),
        (100001, 120001, '100,001 - 120,000'),
        (120001, 150001, '120,001 - 150,000'),
        (150001, 200001, '150,001 - 200,000'),
        (200001, 300001, '200,001 - 300,000'),
        (300001, 400001, '300,001 - 400,000'),
        (400001, float('inf'), '≥ 400,001'),
    ]

    
    # Return processors
    def parse_value(val, is_income=False):
        """Parse string values to numeric midpoint"""
        if pd.isna(val):
            return np.nan
        
        val = str(val).replace(',', '').replace('บาท', '').replace('ล้าน', '').strip()
        
        # Handle different string patterns
        if 'ไม่เกิน' in val or 'น้อยกว่า' in val:
            nums = re.findall(r'\d+\.\d+|\d+', val)
            return float(nums[0]) - 0.01 if nums else np.nan
        elif 'มากกว่า' in val or 'ขึ้นไป' in val:
            nums = re.findall(r'\d+\.\d+|\d+', val)
            return float(nums[0]) + 0.01 if nums else np.nan
        
        nums = re.findall(r'\d+\.\d+|\d+', val)
        if len(nums) == 2:
            return (float(nums[0]) + float(nums[1])) / 2
        elif len(nums) == 1:
            return float(nums[0])
        return np.nan

    def assign_bin(mid, bins):
        if np.isnan(mid):
            return "Missing"
        for low, high, label in bins:
            if low <= mid < high:
                return label
        return "Out of Range"

    return {
        'process_budget': lambda x: assign_bin(parse_value(x), budget_bins),
        'process_family_income': lambda x: assign_bin(parse_value(x, True), income_bins),
        'process_individual_income': lambda x: assign_bin(parse_value(x, True), individual_income_bins)
    }

def clean_financial_columns(df: pd.DataFrame) -> pd.DataFrame:
    processors = create_bin_assignment_functions()
    df = df.copy()
    
    if 'purchase_budget' in df.columns:
        df['purchase_budget'] = df['purchase_budget'].apply(processors['process_budget'])
    if 'family_monthly_income' in df.columns:
        df['family_monthly_income'] = df['family_monthly_income'].apply(processors['process_family_income'])
    if 'individual_monthly_income_baht' in df.columns:
        df['individual_monthly_income_baht'] = df['individual_monthly_income_baht'].apply(processors['process_individual_income'])

    return df

def get_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

# === Main pipeline function ===

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Fix column names
    df.rename(columns={
        'Occcupation': 'Occupation',
        'Project ID': 'projectid'
    }, inplace=True)

    # Fix Buddhist year dates
    df['questiondate'] = df['questiondate'].apply(fix_year)
    df['bookingdate'] = df['bookingdate'].apply(fix_year)
    
    # Create target if available
    df['has_booked'] = df['bookingdate'].notnull().astype(int)
    
    # Condo only
    df = df[df['Type'] == 'คอนโดมิเนียม'].copy()

    # Drop unused cols
    df.drop(columns=[
        'home_purchase_budget','land_house_size_wanted','functions_wanted',
        'moving_in_count','preferred_discount_categories_AssetWise_Clubs',
        'decision_influencer','current_residence_type','desired_living_area',
        'monthly_family_income_baht','individual_monthly_income_fill',
        'preferred_house_style','preferred_house_features','Type'
    ], inplace=True, errors='ignore')
    
    # Financial cleanup
    df = clean_financial_columns(df)

    # Fill missing

    for col in ordinal_features:
        if col in df.columns:
            df[col] = df[col].fillna("Missing")
    for col in nominal_features:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # Group rare categories
    high_cardinality_cols = [
        'information_source', 'purchasing_reason', 'decide_purchase_reason',
        'not_book_reason', 'other_projects_before_deicde',
        'day_off_activity', 'saw_sign', 'car_brand',
        'travel_route_today', 'Location'
    ]
    df = group_rare_categories_by_threshold(df, high_cardinality_cols, threshold=0.01)

    # Date-based features
    df = df.dropna(subset=['questiondate'])
    df = add_seasonal_features(df, 'questiondate')
    
    # Sort for downstream time series (optional)
    df = df.sort_values(by='questiondate')
    
    return df
