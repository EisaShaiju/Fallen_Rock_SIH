"""
Test Client for Rockfall Hazard Prediction API
==============================================

This script demonstrates how to use the FastAPI backend for making predictions.

Usage:
    python test_api.py
"""

import requests
import json
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_model_info():
    """Test the model info endpoint"""
    print("\n📊 Testing model info...")
    
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        if response.status_code == 200:
            print("✅ Model info retrieved!")
            info = response.json()
            print(f"Model type: {info['model_type']}")
            print(f"Features used: {len(info['features_used'])}")
            print(f"Model loaded: {info['model_loaded']}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {e}")

def test_single_prediction():
    """Test single prediction"""
    print("\n🎯 Testing single prediction...")
    
    # Example rockfall scenario data
    test_data = {
        "slope_angle": 45.5,
        "slope_roughness": 8.2,
        "seeder_height": 35.0,
        "aspect_sin": 0.8,
        "aspect_cos": 0.6,
        "curvature": -0.02,
        "local_relief": 25.5,
        "roughness_m": 5.5,
        "roughness_l": 7.8,
        "kinetic_energy": 250.0,
        "impact_position": 30.5,
        "runout_distance": 45.0,
        "rain_1d_mm": 5.2,
        "rain_3d_mm": 12.8,
        "rain_7d_mm": 18.5,
        "rain_30d_mm": 45.2,
        "api_7d": 15.8,
        "api_30d": 28.5,
        "temp_mean_7d_c": 8.5,
        "temp_min_7d_c": 2.1,
        "temp_max_7d_c": 15.2,
        "freeze_thaw_7d": 2.0,
        "vibration_events_7d": 5,
        "vibration_rms_24h": 0.025,
        "disp_rate_mm_day": 0.8,
        "disp_accel_mm_day2": 0.05,
        "pore_pressure_kpa": 48.5,
        "pore_trend_kpa_day": 0.5,
        "strain_rate_micro": 0.25
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"Hazard Score: {result['hazard_score']:.4f}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Timestamp: {result['timestamp']}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")

def test_batch_prediction():
    """Test batch prediction"""
    print("\n📦 Testing batch prediction...")
    
    # Multiple scenarios
    batch_data = {
        "features": [
            {
                "slope_angle": 45.5, "slope_roughness": 8.2, "seeder_height": 35.0,
                "aspect_sin": 0.8, "aspect_cos": 0.6, "curvature": -0.02,
                "local_relief": 25.5, "roughness_m": 5.5, "roughness_l": 7.8,
                "kinetic_energy": 250.0, "impact_position": 30.5, "runout_distance": 45.0,
                "rain_1d_mm": 5.2, "rain_3d_mm": 12.8, "rain_7d_mm": 18.5,
                "rain_30d_mm": 45.2, "api_7d": 15.8, "api_30d": 28.5,
                "temp_mean_7d_c": 8.5, "temp_min_7d_c": 2.1, "temp_max_7d_c": 15.2,
                "freeze_thaw_7d": 2.0, "vibration_events_7d": 5, "vibration_rms_24h": 0.025,
                "disp_rate_mm_day": 0.8, "disp_accel_mm_day2": 0.05,
                "pore_pressure_kpa": 48.5, "pore_trend_kpa_day": 0.5, "strain_rate_micro": 0.25
            },
            {
                "slope_angle": 35.0, "slope_roughness": 6.1, "seeder_height": 25.0,
                "aspect_sin": 0.5, "aspect_cos": 0.8, "curvature": 0.01,
                "local_relief": 15.2, "roughness_m": 4.2, "roughness_l": 5.8,
                "kinetic_energy": 150.0, "impact_position": 20.5, "runout_distance": 30.0,
                "rain_1d_mm": 2.1, "rain_3d_mm": 8.5, "rain_7d_mm": 12.2,
                "rain_30d_mm": 28.5, "api_7d": 10.2, "api_30d": 18.8,
                "temp_mean_7d_c": 12.5, "temp_min_7d_c": 6.1, "temp_max_7d_c": 18.2,
                "freeze_thaw_7d": 1.0, "vibration_events_7d": 2, "vibration_rms_24h": 0.015,
                "disp_rate_mm_day": 0.3, "disp_accel_mm_day2": 0.02,
                "pore_pressure_kpa": 35.2, "pore_trend_kpa_day": 0.2, "strain_rate_micro": 0.15
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict_batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful!")
            print(f"Total predictions: {result['total_predictions']}")
            print(f"Processing time: {result['processing_time_ms']:.2f} ms")
            
            for i, pred in enumerate(result['predictions']):
                print(f"  Scenario {i+1}: {pred['hazard_score']:.4f} ({pred['risk_level']})")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")

def main():
    """Run all tests"""
    print("🚀 Testing Rockfall Hazard Prediction API")
    print("=" * 50)
    
    # Test all endpoints
    test_health_check()
    test_model_info()
    test_single_prediction()
    test_batch_prediction()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")

if __name__ == "__main__":
    main()