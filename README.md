# Rockfall Hazard Prediction API

A FastAPI-based web service for predicting rockfall hazard scores using machine learning. This project provides real-time risk assessment for geological monitoring and infrastructure safety.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Start the API Server

```bash
python app.py
```

### 3. Access the API

- **API Base URL**: `http://localhost:8000`
- **Interactive Documentation**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`

## 📊 Model Overview

Our XGBoost regression model predicts rockfall hazard scores by combining:

- **Target Variable**: `hazard_score = risk_probability_normalized × risk_severity_normalized`
- **Input Features**: 27 geological, environmental, and monitoring parameters
- **Selected Features**: Top 15 most predictive features using correlation-based selection
- **Model Architecture**: XGBoost Regressor with stratified sampling and cross-validation
- **Performance**: ~87% accuracy (R²) with realistic overfitting prevention
- **Output Range**: 0.0 (minimal risk) to 1.0 (critical risk)

## 🎯 Risk Assessment Categories

| Hazard Score | Risk Level | Action Required              |
| ------------ | ---------- | ---------------------------- |
| 0.0 - 0.2    | Very Low   | ✅ Normal operations         |
| 0.2 - 0.4    | Low        | ⚠️ Continue monitoring       |
| 0.4 - 0.6    | Medium     | 🔶 Enhanced monitoring       |
| 0.6 - 0.8    | High       | 🔥 Prepare countermeasures   |
| 0.8 - 1.0    | Critical   | 🚨 Immediate action required |

## 📁 Project Structure

```
sih_ml_part/
├── app.py                              # 🚀 FastAPI web application
├── test_api.py                         # 🧪 API testing script
├── quick_batch_test.py                 # ⚡ Quick batch testing
├── batch_prediction_input.json         # 📄 Example batch input
├── API_DOCUMENTATION.md               # 📖 Detailed API docs
├── requirements_api.txt               # 📦 API dependencies
├── xgboost_hazard_model.pkl           # 🤖 Trained model
├── feature_scaler.pkl                 # 📐 Feature preprocessing
├── selected_features.txt              # 📋 Selected features list
└── README.md                          # 📚 This file
```

## 🔌 API Endpoints

### 1. Health Check

```bash
GET /health
```

Check if the API and model are running properly.

### 2. Model Information

```bash
GET /model_info
```

Get details about the loaded model and features.

### 3. Single Prediction

```bash
POST /predict
```

Predict hazard score for one location.

### 4. Batch Prediction

```bash
POST /predict_batch
```

Predict hazard scores for multiple locations simultaneously.

## 💻 Usage Examples

### Single Prediction with curl

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

**Response:**

```json
{
  "hazard_score": 0.7342,
  "risk_level": "High",
  "confidence": "High",
  "timestamp": "2025-09-29T10:30:15"
}
```

### Batch Prediction with Python

```python
import requests
import json

# Load example batch input
with open('batch_prediction_input.json', 'r') as f:
    batch_data = json.load(f)

# Send batch request
response = requests.post(
    "http://localhost:8000/predict_batch",
    json=batch_data
)

result = response.json()
print(f"Processed {result['total_predictions']} locations")
for i, pred in enumerate(result['predictions']):
    print(f"Location {i+1}: {pred['hazard_score']:.3f} ({pred['risk_level']})")
```

### Interactive Testing

1. Start the API: `python app.py`
2. Open browser: `http://localhost:8000/docs`
3. Click on any endpoint to test it interactively
4. Use "Try it out" feature with sample data

## 🧪 Testing the API

### Quick Health Check

```bash
python -c "import requests; print(requests.get('http://localhost:8000/health').json())"
```

### Run All Tests

```bash
python test_api.py
```

### Quick Batch Test

```bash
python quick_batch_test.py
```

## 📝 Required Input Parameters

All prediction endpoints require these 27 parameters:

### Terrain Features

- `slope_angle`: Slope angle in degrees (0-90)
- `slope_roughness`: Surface roughness measure
- `seeder_height`: Initial release height in meters
- `aspect_sin`, `aspect_cos`: Slope aspect (direction)
- `curvature`: Surface curvature
- `local_relief`: Local relief in meters
- `roughness_m`, `roughness_l`: Medium and large-scale roughness

### Physical Features

- `kinetic_energy`: Kinetic energy in Joules
- `impact_position`: Impact position coordinate
- `runout_distance`: Runout distance in meters

### Weather Features

- `rain_1d_mm`, `rain_3d_mm`, `rain_7d_mm`, `rain_30d_mm`: Rainfall amounts
- `api_7d`, `api_30d`: Antecedent Precipitation Index
- `temp_mean_7d_c`, `temp_min_7d_c`, `temp_max_7d_c`: Temperature data
- `freeze_thaw_7d`: Number of freeze-thaw cycles

### Monitoring Features

- `vibration_events_7d`: 7-day vibration events count
- `vibration_rms_24h`: 24-hour RMS vibration level
- `disp_rate_mm_day`: Displacement rate in mm/day
- `disp_accel_mm_day2`: Displacement acceleration in mm/day²
- `pore_pressure_kpa`: Pore water pressure in kPa
- `pore_trend_kpa_day`: Pore pressure trend in kPa/day
- `strain_rate_micro`: Strain rate in microstrain

## 🤖 Model Technical Details

### Training Pipeline

1. **Data Processing**: 60,000+ rockfall scenarios from advanced geological dataset
2. **Feature Engineering**: Created normalized risk features and interaction terms
3. **Feature Selection**: Correlation-based selection of top 15 features
4. **Data Splitting**: Stratified sampling ensuring balanced risk levels across train/test sets
5. **Model Training**: XGBoost regression with conservative hyperparameters to prevent overfitting
6. **Validation**: 5-fold cross-validation with realistic performance targets

### Selected Features (Top 15)

The model uses these most predictive features:

- `runout_distance`, `kinetic_energy`, `height_energy_interaction`
- `seeder_height`, `slope_angle`, `api_7d`, `api_30d`
- `rain_7d_mm`, `impact_position`, `slope_angle_roughness`
- `rain_30d_mm`, `vibration_events_7d`, `rain_3d_mm`
- `energy_distance_ratio`, `rain_temp_interaction`

### Model Performance

- **Accuracy**: ~87% (R² score) - realistic for geological prediction
- **Cross-validation**: Consistent performance across different data splits
- **Confidence**: High reliability for normal conditions, medium for extreme values
- **Processing Speed**: <50ms per prediction, ~200ms for batch of 100

## 🚀 Production Deployment

### Local Development

```bash
# Start development server
python app.py

# Server runs on http://localhost:8000
# Auto-reloads on code changes
```

### Production Deployment

```bash
# Using Gunicorn (recommended)
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker
docker build -t rockfall-api .
docker run -p 8000:8000 rockfall-api
```

## 🔧 Troubleshooting

### Model Not Loading

- Ensure `xgboost_hazard_model.pkl`, `feature_scaler.pkl`, and `selected_features.txt` exist
- Check file paths are correct
- Verify XGBoost version compatibility

### API Connection Issues

- Confirm server is running: `curl http://localhost:8000/health`
- Check port availability: `netstat -an | grep 8000`
- Verify firewall settings

### Prediction Errors

- Validate all 27 input parameters are provided
- Check parameter value ranges (e.g., slope_angle: 0-90°)
- Ensure numeric values (not strings) for all parameters

## 📚 Additional Resources

- **Detailed API Documentation**: `API_DOCUMENTATION.md`
- **Example Requests**: `batch_prediction_input.json`
- **Model Training Code**: `run_pipeline.py` (for retraining)
- **Data Processing**: `feature_engineering.py`, `data_cleaning.py`

## 🔮 Future Enhancements

- **Real-time Monitoring**: Integration with sensor networks
- **Geographic Mapping**: Spatial risk visualization
- **Mobile App**: Field data collection interface
- **Alert System**: Automated warning notifications
- **Multi-model Ensemble**: Combining multiple prediction models

## 📄 License & Usage

This project is developed for geological hazard assessment and infrastructure safety. Suitable for:

- ✅ Research and educational purposes
- ✅ Infrastructure monitoring systems
- ✅ Risk assessment applications
- ✅ Emergency response planning

**Note**: Always combine predictions with expert geological assessment for critical decisions.
