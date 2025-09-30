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

## 🌐 Web Development Component

_[Web development documentation will be added here]_

---

## 📋 Project Structure

```
sih_ml_part/
├── ml/                    # Machine Learning API
│   ├── app.py            # FastAPI application
│   ├── models/           # Trained ML models
│   ├── src/              # Source code
│   └── data/             # Datasets
└── [webdev components to be added]
```

## � Getting Started

1. **For ML API**: Navigate to `ml/` directory and follow ML component instructions above
2. **For Web Development**: _[Instructions will be added when webdev component is ready]_

## 📄 License

This project is developed for Smart India Hackathon.

#### Health Check- **Input Features**: 27 geological, environmental, and monitoring parameters

````bash- **Selected Features**: Top 15 most predictive features using correlation-based selection

curl http://localhost:8000/health- **Model Architecture**: XGBoost Regressor with stratified sampling and cross-validation

```- **Performance**: ~87% accuracy (R²) with realistic overfitting prevention

- **Output Range**: 0.0 (minimal risk) to 1.0 (critical risk)

#### Single Prediction

```bash## 🎯 Risk Assessment Categories

curl -X POST "http://localhost:8000/predict" \

     -H "Content-Type: application/json" \| Hazard Score | Risk Level | Action Required              |

     -d '{| ------------ | ---------- | ---------------------------- |

       "slope_angle": 45.5,| 0.0 - 0.2    | Very Low   | ✅ Normal operations         |

       "slope_roughness": 8.2,| 0.2 - 0.4    | Low        | ⚠️ Continue monitoring       |

       "seeder_height": 35.0,| 0.4 - 0.6    | Medium     | 🔶 Enhanced monitoring       |

       "aspect_sin": 0.8,| 0.6 - 0.8    | High       | 🔥 Prepare countermeasures   |

       "aspect_cos": 0.6,| 0.8 - 1.0    | Critical   | 🚨 Immediate action required |

       "curvature": -0.02,

       "local_relief": 25.5,## 📁 Project Structure

       "roughness_m": 5.5,

       "roughness_l": 7.8,```

       "kinetic_energy": 250.0,sih_ml_part/

       "impact_position": 30.5,├── app.py                              # 🚀 FastAPI web application

       "runout_distance": 45.0,├── test_api.py                         # 🧪 API testing script

       "rain_1d_mm": 5.2,├── quick_batch_test.py                 # ⚡ Quick batch testing

       "rain_3d_mm": 12.8,├── batch_prediction_input.json         # 📄 Example batch input

       "rain_7d_mm": 18.5,├── API_DOCUMENTATION.md               # 📖 Detailed API docs

       "rain_30d_mm": 45.2,├── requirements_api.txt               # 📦 API dependencies

       "api_7d": 15.8,├── xgboost_hazard_model.pkl           # 🤖 Trained model

       "api_30d": 28.5,├── feature_scaler.pkl                 # 📐 Feature preprocessing

       "temp_mean_7d_c": 8.5,├── selected_features.txt              # 📋 Selected features list

       "temp_min_7d_c": 2.1,└── README.md                          # 📚 This file

       "temp_max_7d_c": 15.2,```

       "freeze_thaw_7d": 2.0,

       "vibration_events_7d": 5,## 🔌 API Endpoints

       "vibration_rms_24h": 0.025,

       "disp_rate_mm_day": 0.8,### 1. Health Check

       "disp_accel_mm_day2": 0.05,

       "pore_pressure_kpa": 48.5,```bash

       "pore_trend_kpa_day": 0.5,GET /health

       "strain_rate_micro": 0.25```

     }'

```Check if the API and model are running properly.



#### Interactive Documentation### 2. Model Information

Visit `http://localhost:8000/docs` for interactive API documentation.

```bash

### 📊 API ResponseGET /model_info

````

The API returns predictions in this format:

````jsonGet details about the loaded model and features.

{

  "hazard_score": 0.7342,### 3. Single Prediction

  "risk_level": "High",

  "confidence": "High",```bash

  "timestamp": "2025-09-30T10:30:15"POST /predict

}```

````

Predict hazard score for one location.

### 🎯 Risk Levels

### 4. Batch Prediction

| Hazard Score | Risk Level | Action Required |

|--------------|------------|-----------------|```bash

| 0.0 - 0.2 | Very Low | ✅ Normal operations |POST /predict_batch

| 0.2 - 0.4 | Low | ⚠️ Continue monitoring |```

| 0.4 - 0.6 | Medium | 🔶 Enhanced monitoring |

| 0.6 - 0.8 | High | 🔥 Prepare countermeasures |Predict hazard scores for multiple locations simultaneously.

| 0.8 - 1.0 | Critical | 🚨 Immediate action required |
