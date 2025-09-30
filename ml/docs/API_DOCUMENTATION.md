# Rockfall Hazard Prediction API Documentation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Train Your Model (if not done already)

```bash
python run_pipeline.py
```

This creates: `xgboost_hazard_model.pkl`, `feature_scaler.pkl`, `selected_features.txt`

### 3. Start the API Server

```bash
python app.py
```

Or using uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
python test_api.py
```

## 📡 API Endpoints

### Base URL: `http://localhost:8000`

### 1. Health Check

```
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-09-29T10:30:00"
}
```

### 2. Model Information

```
GET /model_info
```

**Response:**

```json
{
    "model_type": "XGBoost Regressor",
    "features_used": ["runout_distance", "kinetic_energy", ...],
    "model_loaded": true,
    "load_time": "2025-09-29T10:29:45",
    "version": "1.0.0"
}
```

### 3. Single Prediction

```
POST /predict
```

**Request Body:**

```json
{
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

### 4. Batch Prediction

```
POST /predict_batch
```

**Request Body:**

```json
{
  "features": [
    {
      "slope_angle": 45.5,
      "slope_roughness": 8.2
      // ... all other features
    },
    {
      "slope_angle": 35.0,
      "slope_roughness": 6.1
      // ... all other features
    }
  ]
}
```

**Response:**

```json
{
  "predictions": [
    {
      "hazard_score": 0.7342,
      "risk_level": "High",
      "confidence": "High",
      "timestamp": "2025-09-29T10:30:15"
    },
    {
      "hazard_score": 0.4521,
      "risk_level": "Medium",
      "confidence": "High",
      "timestamp": "2025-09-29T10:30:15"
    }
  ],
  "total_predictions": 2,
  "processing_time_ms": 15.42
}
```

## 🎯 Risk Level Categories

| Hazard Score | Risk Level | Description   |
| ------------ | ---------- | ------------- |
| 0.0 - 0.2    | Very Low   | Minimal risk  |
| 0.2 - 0.4    | Low        | Low risk      |
| 0.4 - 0.6    | Medium     | Moderate risk |
| 0.6 - 0.8    | High       | High risk     |
| 0.8 - 1.0    | Critical   | Extreme risk  |

## 🔧 Feature Descriptions

### Terrain Features

- `slope_angle`: Slope angle in degrees (0-90)
- `slope_roughness`: Surface roughness measure
- `seeder_height`: Initial release height in meters
- `aspect_sin`, `aspect_cos`: Slope aspect (direction)
- `curvature`: Surface curvature
- `local_relief`: Local relief in meters

### Physical Features

- `kinetic_energy`: Kinetic energy in Joules
- `impact_position`: Impact position coordinate
- `runout_distance`: Runout distance in meters
- `roughness_m`, `roughness_l`: Medium and large-scale roughness

### Weather Features

- `rain_1d_mm`, `rain_3d_mm`, `rain_7d_mm`, `rain_30d_mm`: Rainfall amounts
- `api_7d`, `api_30d`: Antecedent Precipitation Index
- `temp_mean_7d_c`, `temp_min_7d_c`, `temp_max_7d_c`: Temperature data
- `freeze_thaw_7d`: Freeze-thaw cycles

### Monitoring Features

- `vibration_events_7d`: 7-day vibration events count
- `vibration_rms_24h`: 24-hour RMS vibration
- `disp_rate_mm_day`: Displacement rate in mm/day
- `disp_accel_mm_day2`: Displacement acceleration
- `pore_pressure_kpa`: Pore water pressure in kPa
- `pore_trend_kpa_day`: Pore pressure trend
- `strain_rate_micro`: Strain rate in microstrain

## 🐛 Error Handling

### Common Error Responses

**400 Bad Request:**

```json
{
  "detail": "Validation error message"
}
```

**500 Internal Server Error:**

```json
{
  "detail": "Prediction failed: error details"
}
```

## 📊 Interactive API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔐 Security Notes

- This is a local development server
- For production deployment, add authentication
- Configure proper CORS settings
- Use HTTPS in production
