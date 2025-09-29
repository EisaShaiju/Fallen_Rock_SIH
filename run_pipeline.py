"""
Main Pipeline Script for Rockfall Hazard Prediction
===================================================

This script runs the complete pipeline:
1. Data cleaning
2. Feature engineering
3. XGBoost model training and evaluation

Usage:
    python run_pipeline.py
"""

import os
import sys
from data_cleaning import RockfallDataCleaner
from feature_engineering import RockfallFeatureEngineer
from xgboost_hazard_model import XGBoostHazardPredictor

def run_complete_pipeline():
    """
    Run the complete rockfall hazard prediction pipeline
    """
    print("ROCKFALL HAZARD PREDICTION PIPELINE")
    print("="*60)
    
    # File paths
    raw_data_file = "Rock_fall_dataset - advanced_rockfall_dataset.csv"
    cleaned_data_file = "cleaned_rockfall_dataset.csv"
    engineered_data_file = "engineered_rockfall_dataset.csv"
    
    # Check if raw data exists
    if not os.path.exists(raw_data_file):
        print(f"Error: Raw data file not found: {raw_data_file}")
        return
    
    print("\n" + "="*60)
    print("STEP 1: DATA CLEANING")
    print("="*60)
    
    # Initialize data cleaner and clean data
    cleaner = RockfallDataCleaner()
    cleaned_df = cleaner.clean_dataset(
        raw_data_file,
        missing_method='drop',
        outlier_method='cap',
        optimize_memory=True
    )
    
    if cleaned_df is None:
        print("Error: Data cleaning failed!")
        return
    
    # Save cleaned data
    cleaned_df.to_csv(cleaned_data_file, index=False)
    cleaner.save_cleaning_report()
    
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    # Initialize feature engineer and create features
    fe = RockfallFeatureEngineer()
    engineered_df = fe.engineer_features(cleaned_data_file, engineered_data_file)
    
    if engineered_df is None:
        print("Error: Feature engineering failed!")
        return
    
    # Get feature engineering summary
    summary = fe.get_feature_summary(engineered_df)
    print(f"\nFeature Engineering Summary:")
    print(f"Total features: {summary['total_features']}")
    print(f"Original features: {summary['original_features']}")
    print(f"Engineered features: {summary['engineered_features']}")
    
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    # Initialize XGBoost predictor
    predictor = XGBoostHazardPredictor(random_state=42)
    
    # Prepare features and target
    X, y = predictor.prepare_features(engineered_df)
    
    # Feature selection - using fewer features to prevent overfitting
    X_selected = predictor.select_features(X, y, selection_method='correlation', k=15)
    
    # Split data with stratified sampling for balanced risk levels
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_data(
        X_selected, y, 
        test_size=0.2, 
        val_size=0.2, 
        stratified=True  # Enable stratified sampling
    )
    
    # Hyperparameter tuning (using random search for faster execution)
    print("\nTuning hyperparameters...")
    best_params = predictor.tune_hyperparameters(
        X_train, y_train, X_val, y_val, 
        tuning_method='random'
    )
    
    # Train model
    print("\nTraining XGBoost model...")
    model = predictor.train_model(X_train, y_train, X_val, y_val, best_params)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = predictor.evaluate_model(X_test, y_test, X_train, y_train)
    
    # Cross-validation to check for overfitting
    print("\nChecking model robustness with cross-validation...")
    cv_scores = predictor.validate_model_robustness(X_selected, y, cv_folds=5)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_importance = predictor.analyze_feature_importance(X_selected.columns)
    
    # Plot results
    print("\nGenerating visualizations...")
    predictor.plot_training_history()
    predictor.plot_predictions(X_test, y_test)
    
    # Save model and artifacts
    print("\nSaving model and artifacts...")
    predictor.save_model()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Print final summary
    print(f"\nFinal Results:")
    print(f"Test R² Score: {metrics['test']['r2']:.4f}")
    print(f"Test RMSE: {metrics['test']['rmse']:.6f}")
    print(f"Test MAE: {metrics['test']['mae']:.6f}")
    print(f"Test MAPE: {metrics['test']['mape']:.2f}%")
    print(f"Cross-Validation RMSE: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Model reliability assessment
    if cv_scores.std() < 0.01 and metrics['test']['r2'] < 0.95:
        print("✅ Model appears robust and not overfitted")
    elif cv_scores.std() > 0.01:
        print("⚠️  High variance in CV - possible overfitting")
    elif metrics['test']['r2'] > 0.97:
        print("⚠️  Very high accuracy - possible overfitting")
    
    print(f"\nFiles Generated:")
    print(f"- {cleaned_data_file}")
    print(f"- {engineered_data_file}")
    print(f"- xgboost_hazard_model.pkl")
    print(f"- feature_scaler.pkl")
    print(f"- feature_importance.csv")
    print(f"- selected_features.txt")
    print(f"- data_cleaning_report.txt")

if __name__ == "__main__":
    try:
        run_complete_pipeline()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\nError occurred during pipeline execution: {e}")
        import traceback
        traceback.print_exc()