# SIH ML Project

A comprehensive Smart India Hackathon project with machine learning and web development components.

## 🤖 ML Component - Rockfall Hazard Prediction API

The ML component provides a FastAPI-based web service for predicting rockfall hazard scores using machine learning.

### 🚀 Quick Start

#### 1. Navigate to ML Directory

```bash
cd ml
```

#### 2. Install Dependencies

```bash
pip install -r requirements_api.txt
```

#### 3. Start the API

```bash
python app.py
```

The API will start running on `http://localhost:8000`

### 📡 API Usage

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Single Prediction

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

#### Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

### 📊 API Response

The API returns predictions in this format:

```json
{
  "hazard_score": 0.7342,
  "risk_level": "High",
  "confidence": "High",
  "timestamp": "2025-09-30T10:30:15"
}
```

### 🎯 Risk Levels

| Hazard Score | Risk Level | Action Required              |
| ------------ | ---------- | ---------------------------- |
| 0.0 - 0.2    | Very Low   | ✅ Normal operations         |
| 0.2 - 0.4    | Low        | ⚠️ Continue monitoring       |
| 0.4 - 0.6    | Medium     | 🔶 Enhanced monitoring       |
| 0.6 - 0.8    | High       | 🔥 Prepare countermeasures   |
| 0.8 - 1.0    | Critical   | 🚨 Immediate action required |

### 🔧 Troubleshooting

#### API Won't Start

- Ensure you're in the `ml` directory: `cd ml`
- Install dependencies: `pip install -r requirements_api.txt`
- Check if port 8000 is available

#### Model Loading Issues

- Verify model files exist in `ml/models/` directory
- Check file permissions

#### Connection Issues

- Confirm API is running: `curl http://localhost:8000/health`
- Check firewall settings

---
