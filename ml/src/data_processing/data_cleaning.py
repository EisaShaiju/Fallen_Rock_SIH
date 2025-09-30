"""
Data Cleaning Module for Rockfall Hazard Prediction
===================================================

This module provides comprehensive data cleaning functionality for the rockfall dataset.
It handles missing values, outliers, duplicates, and data quality issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RockfallDataCleaner:
    """
    Data cleaning class for rockfall hazard prediction dataset
    """
    
    def __init__(self):
        self.cleaning_report = {}
        self.outlier_bounds = {}
        
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
            self.cleaning_report['original_shape'] = df.shape
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def check_missing_values(self, df):
        """
        Check and analyze missing values
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Missing value summary
        """
        print("MISSING VALUES ANALYSIS")
        print("="*50)
        
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        missing_info = pd.DataFrame({
            'Column': missing_values.index,
            'Missing_Count': missing_values.values,
            'Missing_Percentage': missing_percentage.values
        })
        
        # Filter out columns with no missing values
        missing_info = missing_info[missing_info['Missing_Count'] > 0]
        
        if len(missing_info) > 0:
            print("Columns with missing values:")
            print(missing_info.to_string(index=False))
            self.cleaning_report['missing_values'] = missing_info
        else:
            print("✓ No missing values found in the dataset!")
            self.cleaning_report['missing_values'] = "None"
        
        total_missing = df.isnull().sum().sum()
        complete_rows_pct = (len(df) - df.isnull().any(axis=1).sum()) / len(df) * 100
        
        print(f"\nTotal missing values: {total_missing}")
        print(f"Percentage of complete rows: {complete_rows_pct:.2f}%")
        
        self.cleaning_report['total_missing'] = total_missing
        self.cleaning_report['complete_rows_percentage'] = complete_rows_pct
        
        return missing_info
    
    def handle_missing_values(self, df, method='drop', fill_value=None):
        """
        Handle missing values in the dataset
        
        Args:
            df (pandas.DataFrame): Input dataframe
            method (str): Method to handle missing values ('drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_value')
            fill_value: Value to fill if method is 'fill_value'
            
        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        df_cleaned = df.copy()
        original_shape = df_cleaned.shape
        
        if method == 'drop':
            df_cleaned = df_cleaned.dropna()
            print(f"Dropped rows with missing values. New shape: {df_cleaned.shape}")
            
        elif method == 'fill_mean':
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())
            print("Filled missing values with column means")
            
        elif method == 'fill_median':
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].median())
            print("Filled missing values with column medians")
            
        elif method == 'fill_mode':
            for column in df_cleaned.columns:
                mode_value = df_cleaned[column].mode()
                if len(mode_value) > 0:
                    df_cleaned[column] = df_cleaned[column].fillna(mode_value[0])
            print("Filled missing values with column modes")
            
        elif method == 'fill_value':
            df_cleaned = df_cleaned.fillna(fill_value)
            print(f"Filled missing values with: {fill_value}")
        
        self.cleaning_report['missing_handling_method'] = method
        self.cleaning_report['shape_after_missing_handling'] = df_cleaned.shape
        
        return df_cleaned
    
    def check_duplicates(self, df):
        """
        Check for and handle duplicate rows
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe without duplicates
        """
        print("DUPLICATE ANALYSIS")
        print("="*50)
        
        duplicates = df.duplicated().sum()
        duplicate_percentage = duplicates / len(df) * 100
        
        print(f"Number of duplicate rows: {duplicates}")
        print(f"Percentage of duplicates: {duplicate_percentage:.2f}%")
        
        self.cleaning_report['duplicates_found'] = duplicates
        self.cleaning_report['duplicate_percentage'] = duplicate_percentage
        
        if duplicates > 0:
            print("\nRemoving duplicate rows...")
            df_cleaned = df.drop_duplicates()
            print(f"Rows after removing duplicates: {len(df_cleaned)}")
            print(f"Removed {len(df) - len(df_cleaned)} duplicate rows")
            self.cleaning_report['duplicates_removed'] = len(df) - len(df_cleaned)
            return df_cleaned
        else:
            print("✓ No duplicate rows found!")
            self.cleaning_report['duplicates_removed'] = 0
            return df
    
    def detect_outliers_iqr(self, df, columns=None):
        """
        Detect outliers using IQR method
        
        Args:
            df (pandas.DataFrame): Input dataframe
            columns (list): Specific columns to check (if None, check all numeric columns)
            
        Returns:
            dict: Outlier information for each column
        """
        print("OUTLIER DETECTION (IQR METHOD)")
        print("="*50)
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outlier_info = {}
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            
            outlier_info[col] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': outliers.index.tolist()
            }
            
            self.outlier_bounds[col] = (lower_bound, upper_bound)
        
        # Create summary dataframe
        outlier_summary = []
        for col, info in outlier_info.items():
            if info['count'] > 0:
                outlier_summary.append({
                    'Column': col,
                    'Outlier_Count': info['count'],
                    'Outlier_Percentage': round(info['percentage'], 2),
                    'Lower_Bound': round(info['lower_bound'], 4),
                    'Upper_Bound': round(info['upper_bound'], 4)
                })
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            outlier_df = outlier_df.sort_values('Outlier_Percentage', ascending=False)
            print("Columns with outliers (sorted by percentage):")
            print(outlier_df.to_string(index=False))
        else:
            print("✓ No outliers detected using IQR method!")
        
        self.cleaning_report['outlier_analysis'] = outlier_info
        return outlier_info
    
    def handle_outliers(self, df, method='cap', columns=None):
        """
        Handle outliers in the dataset
        
        Args:
            df (pandas.DataFrame): Input dataframe
            method (str): Method to handle outliers ('remove', 'cap', 'transform')
            columns (list): Specific columns to process
            
        Returns:
            pandas.DataFrame: Dataframe with outliers handled
        """
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df_cleaned.select_dtypes(include=[np.number]).columns
        
        outliers_handled = 0
        
        for col in columns:
            if col in self.outlier_bounds:
                lower_bound, upper_bound = self.outlier_bounds[col]
                
                if method == 'remove':
                    # Remove outlier rows
                    before_count = len(df_cleaned)
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                    outliers_handled += before_count - len(df_cleaned)
                    
                elif method == 'cap':
                    # Cap outliers to bounds
                    outliers_before = len(df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)])
                    df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
                    outliers_handled += outliers_before
                    
                elif method == 'transform':
                    # Log transform (for positive values only)
                    if df_cleaned[col].min() > 0:
                        df_cleaned[f'{col}_log'] = np.log1p(df_cleaned[col])
        
        print(f"Outliers handled using '{method}' method: {outliers_handled}")
        self.cleaning_report['outliers_handling_method'] = method
        self.cleaning_report['outliers_handled'] = outliers_handled
        
        return df_cleaned
    
    def check_data_types(self, df):
        """
        Check and optimize data types
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with optimized data types
        """
        print("DATA TYPE OPTIMIZATION")
        print("="*50)
        
        df_optimized = df.copy()
        memory_before = df_optimized.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize numeric columns
        for col in df_optimized.select_dtypes(include=[np.number]).columns:
            # Check if column can be converted to a smaller integer type
            if df_optimized[col].dtype in ['int64', 'int32']:
                if df_optimized[col].min() >= 0:
                    if df_optimized[col].max() < 255:
                        df_optimized[col] = df_optimized[col].astype('uint8')
                    elif df_optimized[col].max() < 65535:
                        df_optimized[col] = df_optimized[col].astype('uint16')
                    elif df_optimized[col].max() < 4294967295:
                        df_optimized[col] = df_optimized[col].astype('uint32')
                else:
                    if df_optimized[col].min() >= -128 and df_optimized[col].max() < 127:
                        df_optimized[col] = df_optimized[col].astype('int8')
                    elif df_optimized[col].min() >= -32768 and df_optimized[col].max() < 32767:
                        df_optimized[col] = df_optimized[col].astype('int16')
                    elif df_optimized[col].min() >= -2147483648 and df_optimized[col].max() < 2147483647:
                        df_optimized[col] = df_optimized[col].astype('int32')
            
            # Convert float64 to float32 if precision allows
            elif df_optimized[col].dtype == 'float64':
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        memory_after = df_optimized.memory_usage(deep=True).sum() / 1024**2
        memory_reduction = ((memory_before - memory_after) / memory_before) * 100
        
        print(f"Memory usage before: {memory_before:.2f} MB")
        print(f"Memory usage after: {memory_after:.2f} MB")
        print(f"Memory reduction: {memory_reduction:.1f}%")
        
        self.cleaning_report['memory_optimization'] = {
            'before_mb': memory_before,
            'after_mb': memory_after,
            'reduction_percentage': memory_reduction
        }
        
        return df_optimized
    
    def validate_data_quality(self, df):
        """
        Perform comprehensive data quality validation
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            dict: Data quality report
        """
        print("DATA QUALITY VALIDATION")
        print("="*50)
        
        quality_report = {}
        
        # Check for infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        quality_report['infinite_values'] = inf_count
        
        # Check for extremely large values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        extreme_values = {}
        for col in numeric_cols:
            extreme_threshold = df[col].quantile(0.99) * 10  # 10x the 99th percentile
            extreme_count = (df[col].abs() > extreme_threshold).sum()
            if extreme_count > 0:
                extreme_values[col] = extreme_count
        
        quality_report['extreme_values'] = extreme_values
        
        # Check for constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        quality_report['constant_columns'] = constant_cols
        
        # Check value ranges
        range_check = {}
        for col in numeric_cols:
            range_check[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'range': df[col].max() - df[col].min(),
                'std': df[col].std()
            }
        
        quality_report['value_ranges'] = range_check
        
        # Print summary
        print(f"Infinite values found: {inf_count}")
        print(f"Columns with extreme values: {len(extreme_values)}")
        print(f"Constant columns: {len(constant_cols)}")
        
        if extreme_values:
            print("\nColumns with extreme values:")
            for col, count in extreme_values.items():
                print(f"  {col}: {count} extreme values")
        
        if constant_cols:
            print(f"\nConstant columns to consider removing: {constant_cols}")
        
        return quality_report
    
    def clean_dataset(self, file_path, missing_method='drop', outlier_method='cap', optimize_memory=True):
        """
        Complete data cleaning pipeline
        
        Args:
            file_path (str): Path to input CSV file
            missing_method (str): Method to handle missing values
            outlier_method (str): Method to handle outliers
            optimize_memory (bool): Whether to optimize memory usage
            
        Returns:
            pandas.DataFrame: Cleaned dataset
        """
        print("STARTING COMPREHENSIVE DATA CLEANING PIPELINE")
        print("="*60)
        
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        # Check missing values
        self.check_missing_values(df)
        
        # Handle missing values
        df = self.handle_missing_values(df, method=missing_method)
        
        # Check and remove duplicates
        df = self.check_duplicates(df)
        
        # Detect outliers
        self.detect_outliers_iqr(df)
        
        # Handle outliers
        df = self.handle_outliers(df, method=outlier_method)
        
        # Optimize data types
        if optimize_memory:
            df = self.check_data_types(df)
        
        # Validate data quality
        quality_report = self.validate_data_quality(df)
        self.cleaning_report['data_quality'] = quality_report
        
        print(f"\n" + "="*60)
        print("DATA CLEANING COMPLETED")
        print("="*60)
        print(f"Final dataset shape: {df.shape}")
        print(f"Original shape: {self.cleaning_report['original_shape']}")
        
        rows_removed = self.cleaning_report['original_shape'][0] - df.shape[0]
        print(f"Rows removed: {rows_removed}")
        
        return df
    
    def save_cleaning_report(self, file_path='data_cleaning_report.txt'):
        """
        Save the cleaning report to a file
        
        Args:
            file_path (str): Path to save the report
        """
        with open(file_path, 'w') as f:
            f.write("DATA CLEANING REPORT\n")
            f.write("="*50 + "\n\n")
            
            for key, value in self.cleaning_report.items():
                f.write(f"{key.upper().replace('_', ' ')}: {value}\n")
        
        print(f"Cleaning report saved to: {file_path}")

def main():
    """
    Main function to demonstrate data cleaning
    """
    # Initialize data cleaner
    cleaner = RockfallDataCleaner()
    
    # File paths
    input_file = "Rock_fall_dataset - advanced_rockfall_dataset.csv"
    output_file = "cleaned_rockfall_dataset.csv"
    
    # Clean dataset
    cleaned_df = cleaner.clean_dataset(
        input_file, 
        missing_method='drop',
        outlier_method='cap',
        optimize_memory=True
    )
    
    if cleaned_df is not None:
        # Save cleaned dataset
        cleaned_df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to: {output_file}")
        
        # Save cleaning report
        cleaner.save_cleaning_report()

if __name__ == "__main__":
    main()