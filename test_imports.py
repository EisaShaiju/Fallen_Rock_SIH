"""
Test script to verify all required packages can be imported
"""

def test_imports():
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn imported successfully")
    except ImportError as e:
        print(f"✗ seaborn import failed: {e}")
        return False
    
    try:
        from scipy import stats
        print("✓ scipy imported successfully")
    except ImportError as e:
        print(f"✗ scipy import failed: {e}")
        return False
    
    try:
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.feature_selection import SelectKBest, f_regression
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import xgboost as xgb
        print(f"✓ xgboost imported successfully (version: {xgb.__version__})")
    except ImportError as e:
        print(f"✗ xgboost import failed: {e}")
        return False
    
    try:
        import joblib
        print("✓ joblib imported successfully")
    except ImportError as e:
        print(f"✗ joblib import failed: {e}")
        return False
    
    print("\n" + "="*50)
    print("ALL IMPORTS SUCCESSFUL!")
    print("="*50)
    return True

def test_module_imports():
    print("\nTesting custom module imports...")
    
    try:
        from data_cleaning import RockfallDataCleaner
        print("✓ data_cleaning module imported successfully")
    except ImportError as e:
        print(f"✗ data_cleaning import failed: {e}")
        return False
    
    try:
        from feature_engineering import RockfallFeatureEngineer
        print("✓ feature_engineering module imported successfully")
    except ImportError as e:
        print(f"✗ feature_engineering import failed: {e}")
        return False
    
    try:
        from xgboost_hazard_model import XGBoostHazardPredictor
        print("✓ xgboost_hazard_model module imported successfully")
    except ImportError as e:
        print(f"✗ xgboost_hazard_model import failed: {e}")
        return False
    
    print("\n" + "="*50)
    print("ALL CUSTOM MODULE IMPORTS SUCCESSFUL!")
    print("="*50)
    return True

if __name__ == "__main__":
    print("PACKAGE IMPORT TEST")
    print("="*50)
    
    basic_imports_ok = test_imports()
    module_imports_ok = test_module_imports()
    
    if basic_imports_ok and module_imports_ok:
        print("\n🎉 ALL TESTS PASSED! Pipeline is ready to run.")
    else:
        print("\n❌ Some imports failed. Please install missing packages.")