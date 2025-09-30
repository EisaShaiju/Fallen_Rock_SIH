from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import random

# In a real app, these would be in a shared module
from db.base import SessionLocal
from db.models import Sensor, Zone # We will need to add a SensorReading model later

# This is a placeholder for the real database model for sensor readings.
# For now, we'll use an in-memory dictionary to simulate it.
FAKE_SENSOR_READINGS_DB = {}


def get_zone_and_required_sensors(db: Session, sensor_id: str):
    """
    Given a sensor_id, finds its zone and all other sensors required for that zone.
    """
    sensor = db.query(Sensor).filter(Sensor.id == sensor_id).first()
    if not sensor:
        return None, []
    
    zone = sensor.zone
    required_sensors = db.query(Sensor).filter(Sensor.zone_id == zone.id).all()
    return zone, required_sensors

def check_data_completeness(required_sensors: list) -> bool:
    """
    Checks if we have recent readings for all required sensors in a zone.
    """
    freshness_threshold = datetime.utcnow() - timedelta(minutes=5)
    
    for sensor in required_sensors:
        reading = FAKE_SENSOR_READINGS_DB.get(sensor.id)
        if not reading or reading['timestamp'] < freshness_threshold:
            print(f"   - ⏳ Data incomplete. Waiting for fresh reading from sensor: {sensor.id} ({sensor.sensor_type})")
            return False
            
    print("   - ✅ Data is complete for this zone.")
    return True

def calculate_risk_stub(sensor_data: dict) -> float:
    """
    The placeholder "Model Stub".
    Calculates a risk score based on simple rules.
    This is the component your AI team will eventually replace.
    """
    # Example: if strain is high, risk increases significantly.
    strain = sensor_data.get("Strain Gauge", 0)
    
    base_risk = 0.1
    strain_factor = strain * 5 # Amplifying the effect of strain
    
    risk_score = base_risk + strain_factor + (random.random() * 0.1) # Add some noise
    
    return min(max(risk_score, 0.0), 1.0) # Clamp between 0.0 and 1.0


def process_sensor_update(sensor_id: str, value: float):
    """
    This is the main function for this service.
    """
    print(f"--- Starting Prediction Process for Sensor {sensor_id} ---")
    db = SessionLocal()
    try:
        # Step 1: Log the new reading (using our fake in-memory DB for now)
        FAKE_SENSOR_READINGS_DB[sensor_id] = {'value': value, 'timestamp': datetime.utcnow()}

        # Step 2: Find out which zone this sensor belongs to and what other sensors are needed
        zone, required_sensors = get_zone_and_required_sensors(db, sensor_id)
        if not zone:
            print(f"   - ⚠️  Sensor '{sensor_id}' not found in configuration database.")
            return

        print(f"   - Sensor belongs to Zone: '{zone.name}'")

        # Step 3: Check if we have all the necessary data for a prediction
        if not check_data_completeness(required_sensors):
            return

        # Step 4: If data is complete, gather it all into one place
        latest_data = {}
        for sensor in required_sensors:
            latest_data[sensor.sensor_type] = FAKE_SENSOR_READINGS_DB[sensor.id]['value']
        
        print(f"   - Assembled Data for Model: {latest_data}")

        # Step 5: Calculate the risk score using the model stub
        risk_score = calculate_risk_stub(latest_data)
        print(f"   - 📈 Calculated Risk Score for Zone '{zone.name}': {risk_score:.3f}")

        # --- TRIGGER FOR NEXT STEPS ---
        # Here you would call the WebSocket Hub and the Alerting Service
        # broadcast_to_frontend(zone.id, risk_score)
        # check_and_send_alerts(zone.id, risk_score)

    finally:
        db.close()
