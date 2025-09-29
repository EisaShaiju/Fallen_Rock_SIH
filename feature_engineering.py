"""
Feature Engineering for Rockfall Hazard Prediction
This module handles feature engineering including normalization of risk probability and severity
to create a hazard score for machine learning prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RockfallFeatureEngineer:
    """
    Feature engineering class for rockfall hazard prediction
    """
    
    def __init__(self):
        self.risk_prob_scaler = MinMaxScaler()
        self.risk_severity_scaler = MinMaxScaler()
        self.feature_scalers = {}
        
    def load_data(self, file_path):
        """
        Load the rockfall dataset
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def create_normalized_risk_features(self, df):
        """
        Create normalized risk probability and severity features
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with normalized features
        """
        df_copy = df.copy()
        
        # Normalize risk_prob_7d (0-1 scale)
        df_copy['risk_probability_normalized'] = self.risk_prob_scaler.fit_transform(
            df_copy[['risk_prob_7d']]
        ).flatten()
        
        # Normalize risk_severity (0-1 scale)
        df_copy['risk_severity_normalized'] = self.risk_severity_scaler.fit_transform(
            df_copy[['risk_severity']]
        ).flatten()
        
        print("Normalized risk features created:")
        print(f"Risk Probability - Original range: [{df['risk_prob_7d'].min():.4f}, {df['risk_prob_7d'].max():.4f}]")
        print(f"Risk Probability - Normalized range: [{df_copy['risk_probability_normalized'].min():.4f}, {df_copy['risk_probability_normalized'].max():.4f}]")
        print(f"Risk Severity - Original range: [{df['risk_severity'].min():.4f}, {df['risk_severity'].max():.4f}]")
        print(f"Risk Severity - Normalized range: [{df_copy['risk_severity_normalized'].min():.4f}, {df_copy['risk_severity_normalized'].max():.4f}]")
        
        return df_copy
    
    def create_hazard_score(self, df):
        """
        Create hazard score as the product of normalized risk probability and severity
        
        Args:
            df (pandas.DataFrame): Dataframe with normalized features
            
        Returns:
            pandas.DataFrame: Dataframe with hazard score
        """
        df_copy = df.copy()
        
        # Create hazard score
        df_copy['hazard_score'] = (
            df_copy['risk_probability_normalized'] * 
            df_copy['risk_severity_normalized']
        )
        
        print(f"\nHazard Score created:")
        print(f"Hazard Score range: [{df_copy['hazard_score'].min():.4f}, {df_copy['hazard_score'].max():.4f}]")
        print(f"Hazard Score mean: {df_copy['hazard_score'].mean():.4f}")
        print(f"Hazard Score std: {df_copy['hazard_score'].std():.4f}")
        
        return df_copy
    
    def create_interaction_features(self, df):
        """
        Create interaction features that might be important for hazard prediction
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with interaction features
        """
        df_copy = df.copy()
        
        # Slope-related interactions
        df_copy['slope_angle_roughness'] = df_copy['slope_angle'] * df_copy['slope_roughness']
        df_copy['slope_angle_curvature'] = df_copy['slope_angle'] * df_copy['curvature']
        
        # Weather-related interactions
        df_copy['rain_temp_interaction'] = df_copy['rain_7d_mm'] * df_copy['temp_mean_7d_c']
        df_copy['freeze_thaw_rain'] = df_copy['freeze_thaw_7d'] * df_copy['rain_7d_mm']
        
        # Energy-related features
        df_copy['energy_distance_ratio'] = df_copy['kinetic_energy'] / (df_copy['runout_distance'] + 1)
        df_copy['height_energy_interaction'] = df_copy['seeder_height'] * df_copy['kinetic_energy']
        
        # Displacement and pressure interactions
        df_copy['disp_pressure_interaction'] = df_copy['disp_rate_mm_day'] * df_copy['pore_pressure_kpa']
        df_copy['strain_vibration_interaction'] = df_copy['strain_rate_micro'] * df_copy['vibration_rms_24h']
        
        print(f"Created {8} interaction features")
        
        return df_copy
    
    def create_categorical_features(self, df):
        """
        Create categorical features based on continuous variables
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with categorical features
        """
        df_copy = df.copy()
        
        # Slope angle categories
        df_copy['slope_category'] = pd.cut(df_copy['slope_angle'], 
                                         bins=[0, 30, 45, 60, 90], 
                                         labels=['gentle', 'moderate', 'steep', 'very_steep'],
                                         include_lowest=True)
        
        # Rain intensity categories  
        df_copy['rain_intensity'] = pd.cut(df_copy['rain_7d_mm'], 
                                         bins=[0, 10, 25, 50, float('inf')], 
                                         labels=['low', 'moderate', 'high', 'extreme'],
                                         include_lowest=True)
        
        # Temperature categories
        df_copy['temp_category'] = pd.cut(df_copy['temp_mean_7d_c'], 
                                        bins=[float('-inf'), 0, 10, 20, float('inf')], 
                                        labels=['freezing', 'cold', 'mild', 'warm'],
                                        include_lowest=True)
        
        # Hazard score categories for analysis
        df_copy['hazard_level'] = pd.cut(df_copy['hazard_score'], 
                                       bins=[0, 0.1, 0.3, 0.6, 1.0], 
                                       labels=['low', 'moderate', 'high', 'critical'],
                                       include_lowest=True)
        
        # Convert categorical columns to strings immediately to avoid CSV issues
        categorical_cols = ['slope_category', 'rain_intensity', 'temp_category', 'hazard_level']
        for col in categorical_cols:
            df_copy[col] = df_copy[col].astype(str)
        
        print("Created categorical features: slope_category, rain_intensity, temp_category, hazard_level")
        
        return df_copy
    
    def engineer_features(self, file_path, save_path=None):
        """
        Complete feature engineering pipeline
        
        Args:
            file_path (str): Path to input CSV file
            save_path (str): Path to save engineered features (optional)
            
        Returns:
            pandas.DataFrame: Fully engineered dataset
        """
        print("Starting feature engineering pipeline...")
        
        try:
            # Load data
            df = self.load_data(file_path)
            if df is None:
                return None
            
            # Create normalized risk features
            df = self.create_normalized_risk_features(df)
            
            # Create hazard score
            df = self.create_hazard_score(df)
            
            # Create interaction features
            df = self.create_interaction_features(df)
            
            # Create categorical features
            df = self.create_categorical_features(df)
            
            print(f"\nFeature engineering completed. Final dataset shape: {df.shape}")
            
            # Save if path provided
            if save_path:
                try:
                    # Ensure all data is in a CSV-compatible format
                    df_to_save = df.copy()
                    
                    # Handle any remaining categorical columns
                    categorical_columns = df_to_save.select_dtypes(include=['category']).columns
                    for col in categorical_columns:
                        df_to_save[col] = df_to_save[col].astype(str)
                    
                    # Replace any infinite values with NaN, then handle them
                    df_to_save = df_to_save.replace([np.inf, -np.inf], np.nan)
                    
                    # Check for any problematic columns
                    for col in df_to_save.columns:
                        if df_to_save[col].dtype == 'object':
                            # Ensure string columns don't have problematic values
                            df_to_save[col] = df_to_save[col].astype(str)
                    
                    df_to_save.to_csv(save_path, index=False)
                    print(f"Engineered features saved to: {save_path}")
                    
                except Exception as save_error:
                    print(f"Warning: Could not save to {save_path}: {save_error}")
                    print("Continuing without saving...")
            
            return df
            
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_feature_summary(self, df):
        """
        Get summary of engineered features-
        
        Args:
            df (pandas.DataFrame): Engineered dataframe
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_features': len(df.columns),
            'original_features': len([col for col in df.columns if col in [
                'slope_angle', 'slope_roughness', 'seeder_height', 'aspect_sin', 'aspect_cos',
                'curvature', 'local_relief', 'roughness_m', 'roughness_l', 'kinetic_energy',
                'impact_position', 'runout_distance', 'rain_1d_mm', 'rain_3d_mm', 'rain_7d_mm',
                'rain_30d_mm', 'api_7d', 'api_30d', 'temp_mean_7d_c', 'temp_min_7d_c',
                'temp_max_7d_c', 'freeze_thaw_7d', 'vibration_events_7d', 'vibration_rms_24h',
                'disp_rate_mm_day', 'disp_accel_mm_day2', 'pore_pressure_kpa', 'pore_trend_kpa_day',
                'strain_rate_micro', 'risk_prob_7d', 'risk_severity'
            ]]),
            'engineered_features': len([col for col in df.columns if col not in [
                'slope_angle', 'slope_roughness', 'seeder_height', 'aspect_sin', 'aspect_cos',
                'curvature', 'local_relief', 'roughness_m', 'roughness_l', 'kinetic_energy',
                'impact_position', 'runout_distance', 'rain_1d_mm', 'rain_3d_mm', 'rain_7d_mm',
                'rain_30d_mm', 'api_7d', 'api_30d', 'temp_mean_7d_c', 'temp_min_7d_c',
                'temp_max_7d_c', 'freeze_thaw_7d', 'vibration_events_7d', 'vibration_rms_24h',
                'disp_rate_mm_day', 'disp_accel_mm_day2', 'pore_pressure_kpa', 'pore_trend_kpa_day',
                'strain_rate_micro', 'risk_prob_7d', 'risk_severity'
            ]]),
            'hazard_score_stats': {
                'min': df['hazard_score'].min(),
                'max': df['hazard_score'].max(),
                'mean': df['hazard_score'].mean(),
                'std': df['hazard_score'].std()
            }
        }
        
        return summary

def main():
    """
    Main function to demonstrate feature engineering
    """
    # Initialize feature engineer
    fe = RockfallFeatureEngineer()
    
    # File paths
    input_file = "Rock_fall_dataset - advanced_rockfall_dataset.csv"
    output_file = "engineered_rockfall_dataset.csv"
    
    # Engineer features
    engineered_df = fe.engineer_features(input_file, output_file)
    
    if engineered_df is not None:
        # Get summary
        summary = fe.get_feature_summary(engineered_df)
        print("\n" + "="*50)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*50)
        print(f"Total features: {summary['total_features']}")
        print(f"Original features: {summary['original_features']}")
        print(f"Engineered features: {summary['engineered_features']}")
        print(f"\nHazard Score Statistics:")
        print(f"  Min: {summary['hazard_score_stats']['min']:.4f}")
        print(f"  Max: {summary['hazard_score_stats']['max']:.4f}")
        print(f"  Mean: {summary['hazard_score_stats']['mean']:.4f}")
        print(f"  Std: {summary['hazard_score_stats']['std']:.4f}")
        
        # Display sample of engineered features
        print(f"\nSample of engineered dataset:")
        print(engineered_df[['risk_probability_normalized', 'risk_severity_normalized', 
                           'hazard_score', 'hazard_level']].head())

if __name__ == "__main__":
    main()