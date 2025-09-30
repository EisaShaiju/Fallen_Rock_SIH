"""
FastAPI Backend for Rockfall Hazard Prediction
==============================================

This FastAPI application provides a REST API for predicting rockfall hazard scores
using the trained XGBoost model.

Usage:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict - Single prediction
    POST /predict_batch - Batch predictions
    GET /health - Health check
    GET /model_info - Model information
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Rockfall Hazard Prediction API",
    description="API for predicting rockfall hazard scores using XGBoost model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
model_data = {
    "model": None,
    "scaler": None,
    "selected_features": None,
    "model_loaded": False,
    "load_time": None
}

# Input data models
class RockfallFeatures(BaseModel):
    """Single rockfall feature set for prediction"""
    
    # Original features (these would be measured/observed)
    slope_angle: float = Field(..., ge=0, le=90, description="Slope angle in degrees")
    slope_roughness: float = Field(..., ge=0, description="Surface roughness measure")
    seeder_height: float = Field(..., ge=0, description="Initial release height in meters")
    aspect_sin: float = Field(..., ge=-1, le=1, description="Sine of slope aspect")
    aspect_cos: float = Field(..., ge=-1, le=1, description="Cosine of slope aspect")
    curvature: float = Field(..., description="Surface curvature")
    local_relief: float = Field(..., ge=0, description="Local relief in meters")
    roughness_m: float = Field(..., ge=0, description="Medium-scale roughness")
    roughness_l: float = Field(..., ge=0, description="Large-scale roughness")
    kinetic_energy: float = Field(..., ge=0, description="Kinetic energy in Joules")
    impact_position: float = Field(..., ge=0, description="Impact position coordinate")
    runout_distance: float = Field(..., ge=0, description="Runout distance in meters")
    
    # Weather features
    rain_1d_mm: float = Field(..., ge=0, description="1-day rainfall in mm")
    rain_3d_mm: float = Field(..., ge=0, description="3-day rainfall in mm")
    rain_7d_mm: float = Field(..., ge=0, description="7-day rainfall in mm")
    rain_30d_mm: float = Field(..., ge=0, description="30-day rainfall in mm")
    api_7d: float = Field(..., ge=0, description="7-day Antecedent Precipitation Index")
    api_30d: float = Field(..., ge=0, description="30-day Antecedent Precipitation Index")
    temp_mean_7d_c: float = Field(..., description="7-day mean temperature in Celsius")
    temp_min_7d_c: float = Field(..., description="7-day minimum temperature in Celsius")
    temp_max_7d_c: float = Field(..., description="7-day maximum temperature in Celsius")
    freeze_thaw_7d: float = Field(..., ge=0, description="7-day freeze-thaw cycles")
    
    # Monitoring features
    vibration_events_7d: int = Field(..., ge=0, description="7-day vibration events count")
    vibration_rms_24h: float = Field(..., ge=0, description="24-hour RMS vibration")
    disp_rate_mm_day: float = Field(..., description="Displacement rate in mm/day")
    disp_accel_mm_day2: float = Field(..., description="Displacement acceleration in mm/day²")
    pore_pressure_kpa: float = Field(..., ge=0, description="Pore water pressure in kPa")
    pore_trend_kpa_day: float = Field(..., description="Pore pressure trend in kPa/day")
    strain_rate_micro: float = Field(..., description="Strain rate in microstrain")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    features: List[RockfallFeatures]
    
class PredictionResponse(BaseModel):
    """Single prediction response"""
    hazard_score: float = Field(..., description="Predicted hazard score (0-1)")
    risk_level: str = Field(..., description="Risk level category")
    confidence: Optional[str] = Field(None, description="Prediction confidence level")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_predictions: int
    processing_time_ms: float

class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str
    features_used: List[str]
    model_loaded: bool
    load_time: Optional[str]
    version: str

# Utility functions
def load_model():
    """Load the trained model and preprocessing artifacts"""
    global model_data
    
    try:
        # Model file paths
        model_path = "models/xgboost_hazard_model.pkl"
        scaler_path = "models/feature_scaler.pkl"
        features_path = "models/selected_features.txt"
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        logger.info("Loading XGBoost model...")
        model_data["model"] = joblib.load(model_path)
        
        # Load scaler if available
        if os.path.exists(scaler_path):
            logger.info("Loading feature scaler...")
            model_data["scaler"] = joblib.load(scaler_path)
        
        # Load selected features
        if os.path.exists(features_path):
            logger.info("Loading selected features...")
            with open(features_path, 'r') as f:
                model_data["selected_features"] = [line.strip() for line in f.readlines()]
        
        model_data["model_loaded"] = True
        model_data["load_time"] = datetime.now().isoformat()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Selected features: {model_data['selected_features']}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

def preprocess_features(features: RockfallFeatures) -> pd.DataFrame:
    """Convert input features to DataFrame and engineer features"""
    
    # Convert to dictionary
    feature_dict = features.dict()
    
    # Create DataFrame
    df = pd.DataFrame([feature_dict])
    
    # Feature engineering (basic interactions matching your pipeline)
    try:
        # Height-energy interaction
        df['height_energy_interaction'] = df['seeder_height'] * df['kinetic_energy']
        
        # Slope-angle roughness
        df['slope_angle_roughness'] = df['slope_angle'] * df['slope_roughness']
        
        # Energy distance ratio
        df['energy_distance_ratio'] = df['kinetic_energy'] / (df['runout_distance'] + 1e-6)
        
        # Rain-temperature interaction
        df['rain_temp_interaction'] = df['rain_7d_mm'] * df['temp_mean_7d_c']
        
        # Add any other engineered features that your model expects
        # (You may need to add more based on your feature engineering pipeline)
        
    except Exception as e:
        logger.warning(f"Error in feature engineering: {e}")
    
    return df

def get_risk_level(hazard_score: float) -> str:
    """Convert hazard score to risk level category"""
    if hazard_score < 0.2:
        return "Very Low"
    elif hazard_score < 0.4:
        return "Low"
    elif hazard_score < 0.6:
        return "Medium"
    elif hazard_score < 0.8:
        return "High"
    else:
        return "Critical"

def make_prediction(features: RockfallFeatures) -> Dict[str, Any]:
    """Make a single prediction"""
    
    if not model_data["model_loaded"]:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Preprocess features
        df = preprocess_features(features)
        
        # Select only the features used in training
        if model_data["selected_features"]:
            # Ensure all required features are present
            missing_features = set(model_data["selected_features"]) - set(df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features with default values (you may want to handle this differently)
                for feature in missing_features:
                    df[feature] = 0.0
            
            df = df[model_data["selected_features"]]
        
        # Scale features if scaler is available
        if model_data["scaler"] is not None:
            df_scaled = model_data["scaler"].transform(df)
            df = pd.DataFrame(df_scaled, columns=df.columns)
        
        # Make prediction
        hazard_score = float(model_data["model"].predict(df)[0])
        
        # Ensure score is within valid range
        hazard_score = max(0.0, min(1.0, hazard_score))
        
        # Get risk level
        risk_level = get_risk_level(hazard_score)
        
        return {
            "hazard_score": hazard_score,
            "risk_level": risk_level,
            "confidence": "High" if 0.1 < hazard_score < 0.9 else "Medium",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Rockfall Hazard Prediction API...")
    try:
        load_model()
        logger.info("API ready!")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Rockfall Hazard Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_data["model_loaded"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_data["model_loaded"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model_info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    return ModelInfo(
        model_type="XGBoost Regressor",
        features_used=model_data.get("selected_features", []),
        model_loaded=model_data["model_loaded"],
        load_time=model_data.get("load_time"),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_hazard(features: RockfallFeatures):
    """Make a single hazard prediction"""
    
    start_time = datetime.now()
    
    try:
        result = make_prediction(features)
        
        return PredictionResponse(
            hazard_score=result["hazard_score"],
            risk_level=result["risk_level"],
            confidence=result["confidence"],
            timestamp=result["timestamp"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions"""
    
    start_time = datetime.now()
    
    try:
        predictions = []
        
        for features in request.features:
            result = make_prediction(features)
            predictions.append(PredictionResponse(
                hazard_score=result["hazard_score"],
                risk_level=result["risk_level"],
                confidence=result["confidence"],
                timestamp=result["timestamp"]
            ))
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )