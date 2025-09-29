"""
XGBoost Hazard Score Prediction Model
=====================================

This module implements a comprehensive XGBoost model for predicting rockfall hazard scores.
The hazard score is calculated as risk_probability_normalized × risk_severity_normalized.

Features:
- Feature selection and preprocessing
- Hyperparameter tuning with GridSearchCV
- Model training and evaluation
- Feature importance analysis
- Model persistence (save/load)
- Prediction on new data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# XGBoost and ML libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor

# Feature engineering
from feature_engineering import RockfallFeatureEngineer

class XGBoostHazardPredictor:
    """
    XGBoost model for predicting rockfall hazard scores
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the XGBoost predictor
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.feature_importance_df = None
        self.training_history = {}
        
    def load_and_prepare_data(self, data_path):
        """
        Load and prepare data using feature engineering
        
        Args:
            data_path (str): Path to the CSV dataset
            
        Returns:
            pandas.DataFrame: Engineered dataset
        """
        print("Loading and preparing data...")
        
        # Initialize feature engineer
        fe = RockfallFeatureEngineer()
        
        # Apply feature engineering
        engineered_df = fe.engineer_features(data_path)
        
        if engineered_df is None:
            raise ValueError("Failed to load and engineer features")
        
        print(f"Data loaded successfully. Shape: {engineered_df.shape}")
        return engineered_df
    
    def select_features(self, X, y, selection_method='correlation', k=20):
        """
        Select the most important features for modeling
        
        Args:
            X (pandas.DataFrame): Feature matrix
            y (pandas.Series): Target variable
            selection_method (str): Feature selection method ('correlation', 'univariate', 'rfe')
            k (int): Number of features to select
            
        Returns:
            pandas.DataFrame: Selected features
        """
        print(f"Selecting top {k} features using {selection_method} method...")
        
        if selection_method == 'correlation':
            # Select features based on correlation with target
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            self.selected_features = correlations.head(k).index.tolist()
            
        elif selection_method == 'univariate':
            # Select features using univariate statistical tests
            selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            self.selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selector = selector
            
        elif selection_method == 'rfe':
            # Recursive Feature Elimination with Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            selector = RFE(estimator=rf, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            self.selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selector = selector
        
        print(f"Selected features: {self.selected_features}")
        return X[self.selected_features]
    
    def prepare_features(self, df, target_column='hazard_score'):
        """
        Prepare features for modeling by removing non-predictive columns
        
        Args:
            df (pandas.DataFrame): Engineered dataset
            target_column (str): Name of target column
            
        Returns:
            tuple: (X, y) - features and target
        """
        print("Preparing features for modeling...")
        
        # Remove target-related columns and categorical columns for now
        columns_to_remove = [
            target_column,
            'risk_prob_7d',  # Original risk probability (we use normalized version)
            'risk_severity',  # Original risk severity (we use normalized version)
            'risk_probability_normalized',  # Component of target
            'risk_severity_normalized',     # Component of target
            'hazard_level',  # Categorical derived from target
        ]
        
        # Remove categorical columns (we'll encode them later if needed)
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        columns_to_remove.extend(categorical_columns)
        
        # Keep only columns that exist in the dataframe
        columns_to_remove = [col for col in columns_to_remove if col in df.columns]
        
        # Prepare feature matrix and target
        X = df.drop(columns=columns_to_remove)
        y = df[target_column]
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        print(f"Available features: {list(X.columns)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, stratified=True):
        """
        Split data into train, validation, and test sets with optional stratification
        
        Args:
            X (pandas.DataFrame): Feature matrix
            y (pandas.Series): Target variable
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set
            stratified (bool): Whether to use stratified sampling for balanced risk levels
            
        Returns:
            tuple: Split datasets
        """
        if stratified:
            return self.stratified_split_regression(X, y, test_size, val_size, n_bins=5)
        else:
            # Original random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=None
            )
            
            # Second split: separate train and validation from remaining data
            val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
            )
            
            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Validation set: {X_val.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
    
    def stratified_split_regression(self, X, y, test_size=0.2, val_size=0.2, n_bins=5):
        """
        Perform stratified sampling for regression targets by binning continuous values
        
        Args:
            X: Features
            y: Target variable (hazard_score)
            test_size: Proportion for test set
            val_size: Proportion for validation set  
            n_bins: Number of risk level bins
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Using stratified sampling for balanced risk level distribution...")
        
        # Create risk level bins
        y_binned = pd.cut(y, bins=n_bins, labels=['Very Low', 'Low', 'Medium', 'High', 'Critical'])
        
        print("\nOriginal Risk Level Distribution:")
        print(y_binned.value_counts().sort_index())
        print(f"Percentage distribution:")
        print((y_binned.value_counts(normalize=True) * 100).round(2).sort_index())
        
        # First split: separate test set with stratification
        X_temp, X_test, y_temp, y_test, y_temp_binned, y_test_binned = train_test_split(
            X, y, y_binned, 
            test_size=test_size, 
            stratify=y_binned, 
            random_state=self.random_state
        )
        
        # Second split: separate train and validation with stratification
        relative_val_size = val_size / (1 - test_size)  # Adjust validation size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=relative_val_size,
            stratify=y_temp_binned,
            random_state=self.random_state
        )
        
        # Print distribution verification
        print(f"\nDataset splits:")
        print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")  
        print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Verify stratification worked
        train_bins = pd.cut(y_train, bins=n_bins, labels=['Very Low', 'Low', 'Medium', 'High', 'Critical'])
        val_bins = pd.cut(y_val, bins=n_bins, labels=['Very Low', 'Low', 'Medium', 'High', 'Critical'])
        test_bins = pd.cut(y_test, bins=n_bins, labels=['Very Low', 'Low', 'Medium', 'High', 'Critical'])
        
        print(f"\nRisk level distribution across splits (%):")
        distribution_df = pd.DataFrame({
            'Train': train_bins.value_counts(normalize=True) * 100,
            'Validation': val_bins.value_counts(normalize=True) * 100,
            'Test': test_bins.value_counts(normalize=True) * 100
        }).round(2)
        print(distribution_df)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def validate_model_robustness(self, X, y, cv_folds=5):
        """
        Perform k-fold cross-validation to detect overfitting
        
        Args:
            X: Feature matrix
            y: Target variable  
            cv_folds: Number of cross-validation folds
            
        Returns:
            Array of RMSE scores from cross-validation
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        print(f"Performing {cv_folds}-fold cross-validation to check model robustness...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, 
                                   scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        print(f"\nCross-Validation Results ({cv_folds}-fold):")
        print(f"  Mean RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
        print(f"  Individual folds: {cv_rmse.round(4)}")
        
        # Check for overfitting indicators
        if cv_rmse.std() > 0.01:
            print("⚠️  WARNING: High variance across folds - possible overfitting!")
        else:
            print("✅ Good: Low variance across folds - model appears robust")
            
        if cv_rmse.mean() < 0.02:
            print("⚠️  WARNING: Extremely low error - possible overfitting!")
        
        return cv_rmse
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, tuning_method='grid'):
        """
        Tune XGBoost hyperparameters
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            tuning_method (str): 'grid' or 'random' search
            
        Returns:
            dict: Best hyperparameters
        """
        print(f"Tuning hyperparameters using {tuning_method} search...")
        
        # Define conservative parameter grid to prevent overfitting
        if tuning_method == 'grid':
            param_grid = {
                'n_estimators': [30, 50, 100],         # Fewer trees
                'max_depth': [3, 4, 5],                # Shallower trees
                'learning_rate': [0.05, 0.1, 0.15],    # Slower learning
                'subsample': [0.7, 0.8, 0.9],          # Sample less data
                'colsample_bytree': [0.7, 0.8, 0.9],   # Sample fewer features
                'reg_alpha': [0.5, 1.0, 2.0],          # L1 regularization
                'reg_lambda': [1.0, 2.0, 3.0]          # L2 regularization
            }
            
            search = GridSearchCV(
                estimator=xgb.XGBRegressor(
                    random_state=self.random_state, 
                    objective='reg:squarederror'
                ),
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=3,
                n_jobs=-1,
                verbose=1
            )
        else:
            param_distributions = {
                'n_estimators': [30, 50, 100, 150],        # Fewer trees
                'max_depth': [3, 4, 5, 6],                 # Reasonable depth
                'learning_rate': [0.05, 0.1, 0.15, 0.2],  # Conservative learning rates
                'subsample': [0.6, 0.7, 0.8, 0.9],        # More subsampling
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9], # More feature subsampling
                'reg_alpha': [0.5, 1.0, 2.0, 3.0],        # Stronger L1 regularization
                'reg_lambda': [1.0, 2.0, 3.0, 4.0],       # Stronger L2 regularization
                'min_child_weight': [3, 5, 7]             # More samples required per leaf
            }
            
            search = RandomizedSearchCV(
                estimator=xgb.XGBRegressor(
                    random_state=self.random_state, 
                    objective='reg:squarederror'
                ),
                param_distributions=param_distributions,
                n_iter=30,  # Reduced iterations for faster tuning
                scoring='neg_mean_squared_error',
                cv=3,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
        
        # Fit the search
        search.fit(X_train, y_train)
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score: {-search.best_score_:.6f}")
        
        return search.best_params_
    
    def train_model(self, X_train, y_train, X_val, y_val, best_params=None):
        """
        Train the XGBoost model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (unused in this simplified version)
            best_params (dict): Hyperparameters to use
            
        Returns:
            xgb.XGBRegressor: Trained model
        """
        print("Training XGBoost model...")
        
        # Conservative default parameters to prevent overfitting
        if best_params is None:
            best_params = {
                'n_estimators': 50,            # Fewer trees
                'max_depth': 4,                # Shallower trees
                'learning_rate': 0.1,          # Moderate learning rate
                'subsample': 0.8,              # Use 80% of data per tree
                'colsample_bytree': 0.8,       # Use 80% of features per tree
                'reg_alpha': 1.0,              # L1 regularization
                'reg_lambda': 2.0,             # L2 regularization
                'min_child_weight': 5,         # More samples required per leaf
                'random_state': self.random_state,
                'objective': 'reg:squarederror'
            }
        else:
            # Clean up parameters - remove any problematic ones
            clean_params = best_params.copy()
            clean_params.pop('early_stopping_rounds', None)
            clean_params.pop('eval_metric', None)
            best_params = clean_params
        
        # Initialize and train model
        self.model = xgb.XGBRegressor(**best_params)
        
        # Simple training without early stopping
        self.model.fit(X_train, y_train)
        
        # Store basic training information
        self.training_history = {
            'completed': True,
            'n_estimators': self.model.n_estimators
        }
        
        print("Model training completed!")
        return self.model
    
    def evaluate_model(self, X_test, y_test, X_train=None, y_train=None):
        """
        Evaluate the trained model
        
        Args:
            X_test, y_test: Test data
            X_train, y_train: Training data (optional)
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics for test set
        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test),
            'mape': mean_absolute_percentage_error(y_test, y_pred_test) * 100
        }
        
        # Calculate metrics for training set if provided
        if X_train is not None and y_train is not None:
            y_pred_train = self.model.predict(X_train)
            train_metrics = {
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train),
                'mape': mean_absolute_percentage_error(y_train, y_pred_train) * 100
            }
        else:
            train_metrics = None
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        if train_metrics:
            print("Training Set Performance:")
            print(f"  RMSE: {train_metrics['rmse']:.6f}")
            print(f"  MAE:  {train_metrics['mae']:.6f}")
            print(f"  R²:   {train_metrics['r2']:.6f}")
            print(f"  MAPE: {train_metrics['mape']:.2f}%")
            print()
        
        print("Test Set Performance:")
        print(f"  RMSE: {test_metrics['rmse']:.6f}")
        print(f"  MAE:  {test_metrics['mae']:.6f}")
        print(f"  R²:   {test_metrics['r2']:.6f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        
        return {'test': test_metrics, 'train': train_metrics}
    
    def analyze_feature_importance(self, feature_names):
        """
        Analyze and visualize feature importance
        
        Args:
            feature_names (list): Names of features used in training
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print("Analyzing feature importance...")
        
        # Get feature importance scores
        importance_scores = self.model.feature_importances_
        
        # Create feature importance dataframe
        self.feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Display top features
        print("\nTop 15 Most Important Features:")
        print("-" * 40)
        for i, (_, row) in enumerate(self.feature_importance_df.head(15).iterrows()):
            print(f"{i+1:2d}. {row['feature']:30s}: {row['importance']:.4f}")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot top 20 features
        top_features = self.feature_importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importance (XGBoost)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return self.feature_importance_df
    
    def plot_training_history(self):
        """
        Plot training and validation loss curves
        """
        if not self.training_history or 'completed' not in self.training_history:
            print("No training history available")
            return
        
        # Check if we have detailed training history
        if 'train_rmse' in self.training_history and 'val_rmse' in self.training_history:
            plt.figure(figsize=(10, 6))
            
            epochs = range(1, len(self.training_history['train_rmse']) + 1)
            plt.plot(epochs, self.training_history['train_rmse'], 'b-', label='Training RMSE')
            plt.plot(epochs, self.training_history['val_rmse'], 'r-', label='Validation RMSE')
            
            plt.title('Model Training History')
            plt.xlabel('Epochs')
            plt.ylabel('RMSE')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("Training completed successfully but detailed history not available")
            print(f"Model trained with {self.training_history.get('n_estimators', 'unknown')} estimators")
    
    def plot_predictions(self, X_test, y_test):
        """
        Plot predicted vs actual values
        
        Args:
            X_test, y_test: Test data
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        y_pred = self.model.predict(X_test)
        
        plt.figure(figsize=(12, 5))
        
        # Scatter plot of predictions vs actual
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.6, s=30)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Hazard Score')
        plt.ylabel('Predicted Hazard Score')
        plt.title('Predicted vs Actual Hazard Scores')
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, s=30)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Hazard Score')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path='xgboost_hazard_model.pkl', scaler_path='feature_scaler.pkl'):
        """
        Save the trained model and scaler
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str): Path to save the scaler
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save scaler if it was fitted
        if hasattr(self.scaler, 'scale_'):
            joblib.dump(self.scaler, scaler_path)
        
        # Save feature importance if available
        if self.feature_importance_df is not None:
            self.feature_importance_df.to_csv('feature_importance.csv', index=False)
        
        # Save selected features
        if self.selected_features:
            with open('selected_features.txt', 'w') as f:
                f.write('\n'.join(self.selected_features))
        
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print("Feature importance saved to: feature_importance.csv")
        print("Selected features saved to: selected_features.txt")
    
    def load_model(self, model_path='xgboost_hazard_model.pkl', scaler_path='feature_scaler.pkl'):
        """
        Load a trained model and scaler
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str): Path to the saved scaler
        """
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from: {model_path}")
            
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"Scaler loaded from: {scaler_path}")
            except FileNotFoundError:
                print(f"Scaler file not found: {scaler_path}")
            
            # Load selected features if available
            try:
                with open('selected_features.txt', 'r') as f:
                    self.selected_features = f.read().strip().split('\n')
                print("Selected features loaded")
            except FileNotFoundError:
                print("Selected features file not found")
                
        except FileNotFoundError:
            print(f"Model file not found: {model_path}")
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X (pandas.DataFrame): Feature matrix
            
        Returns:
            numpy.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded first")
        
        # Select features if feature selection was used
        if self.selected_features:
            X = X[self.selected_features]
        
        # Scale features if scaler was fitted
        if hasattr(self.scaler, 'scale_'):
            X = self.scaler.transform(X)
        
        return self.model.predict(X)

def main():
    """
    Main function to train and evaluate the XGBoost hazard prediction model
    """
    print("Starting XGBoost Hazard Score Prediction Pipeline")
    print("=" * 60)
    
    # Initialize predictor
    predictor = XGBoostHazardPredictor(random_state=42)
    
    # Load and prepare data
    data_path = "Rock_fall_dataset - advanced_rockfall_dataset.csv"
    df = predictor.load_and_prepare_data(data_path)
    
    # Prepare features and target
    X, y = predictor.prepare_features(df)
    
    # Feature selection
    X_selected = predictor.select_features(X, y, selection_method='correlation', k=25)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_data(X_selected, y)
    
    # Hyperparameter tuning
    best_params = predictor.tune_hyperparameters(X_train, y_train, X_val, y_val, tuning_method='random')
    
    # Train model
    model = predictor.train_model(X_train, y_train, X_val, y_val, best_params)
    
    # Evaluate model
    metrics = predictor.evaluate_model(X_test, y_test, X_train, y_train)
    
    # Analyze feature importance
    feature_importance = predictor.analyze_feature_importance(X_selected.columns)
    
    # Plot training history and predictions
    predictor.plot_training_history()
    predictor.plot_predictions(X_test, y_test)
    
    # Save model
    predictor.save_model()
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("Model and artifacts saved.")

if __name__ == "__main__":
    main()