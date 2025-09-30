"""
Quick API Test Script
====================

Test the Rockfall Hazard Prediction API with sample data.

Usage:
    python test_api_simple.py
"""

import requests
import json

def test_api():
    """Test the API with a simple prediction"""
    
    # API URL
    url = "http://localhost:8000/predict"
    
    # Sample data
    sample_data = {
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
        # Test health check first
        health_response = requests.get("http://localhost:8000/health")
        if health_response.status_code == 200:
            print("✅ API is healthy!")
        else:
            print("❌ API health check failed")
            return
        
        # Make prediction
        response = requests.post(url, json=sample_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"📊 Hazard Score: {result['hazard_score']:.4f}")
            print(f"🎯 Risk Level: {result['risk_level']}")
            print(f"🔍 Confidence: {result['confidence']}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API")
        print("💡 Make sure to start the server first: python app.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Rockfall Hazard Prediction API")
    print("=" * 40)
    test_api()