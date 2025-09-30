import paho.mqtt.client as mqtt
import os
import json
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Note: In a real production app, you would import your DB models
# from db.models import SensorReading, Sensor
# For this standalone script, we will define a simple logging function.

# =================================================================================
# --- 1. CONFIGURATION ---
# =================================================================================
load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

# This is the "wildcard" topic the service will subscribe to.
# The '+' is a single-level wildcard. '#' is a multi-level wildcard.
# This topic will match messages from ANY mine, ANY zone, and ANY sensor.
# e.g., "mines/mine-123/zones/zone-456/sensors/sensor-789/data"
MQTT_SUBSCRIBE_TOPIC = "mines/+/zones/+/sensors/+/data"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/minedb")

# =================================================================================
# --- 2. DATABASE CONNECTION (Simplified) ---
# =================================================================================
# In a full app, this would use the SessionLocal from your db/base.py
try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    print("✅ Database connection successful.")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    engine = None

def log_sensor_reading(sensor_id, value):
    """
    Logs the sensor reading to the database.
    In a real application, this would interact with SQLAlchemy models.
    """
    if not engine:
        print("Database not connected. Skipping log.")
        return

    # This is a simplified example of how you would insert data.
    # A real implementation would first validate the sensor_id exists in the 'sensors' table.
    db = SessionLocal()
    try:
        # A more robust solution would be:
        # reading = SensorReading(sensor_id=sensor_id, value=value)
        # db.add(reading)
        # db.commit()
        print(f"DB LOG: Sensor '{sensor_id}' reading: {value}")
        
        # --- TRIGGER FOR NEXT STEP ---
        # After logging, this is where you would trigger the Prediction Generation Service.
        # For example, by adding a job to a task queue (like Celery or RQ).
        # trigger_prediction_for_sensor(sensor_id)

    except Exception as e:
        print(f"Error logging to DB: {e}")
        db.rollback()
    finally:
        db.close()


# =================================================================================
# --- 3. MQTT CALLBACK FUNCTIONS ---
# =================================================================================

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT Broker!")
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe(MQTT_SUBSCRIBE_TOPIC)
        print(f"👂 Subscribed to topic: {MQTT_SUBSCRIBE_TOPIC}")
    else:
        print(f"❌ Failed to connect, return code {rc}\n")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(f"📩 Message received on topic: {msg.topic}")
    
    try:
        # 1. Decode the payload from bytes to a string
        payload_str = msg.payload.decode("utf-8")
        
        # 2. Parse the JSON string into a Python dictionary
        data = json.loads(payload_str)
        
        # 3. Extract the sensor ID from the topic string itself. This is more secure
        #    than trusting the ID inside the payload.
        #    Topic format: mines/{mine_id}/zones/{zone_id}/sensors/{sensor_id}/data
        topic_parts = msg.topic.split('/')
        sensor_id = topic_parts[5]
        
        value = data.get("value")

        if sensor_id and isinstance(value, (int, float)):
            print(f"   - Parsed Data: Sensor ID = {sensor_id}, Value = {value}")
            # 4. Log the validated data to the database
            log_sensor_reading(sensor_id, value)
        else:
            print(f"   - ⚠️  Invalid data format in payload: {payload_str}")

    except json.JSONDecodeError:
        print(f"   - ⚠️  Could not decode JSON from payload: {msg.payload.decode('utf-8')}")
    except Exception as e:
        print(f"   - 💥 An error occurred: {e}")


# =================================================================================
# --- 4. MAIN EXECUTION BLOCK ---
# =================================================================================

if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print("🚀 Starting Ingestion Service...")
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            # Blocking call that processes network traffic, dispatches callbacks and
            # handles reconnecting.
            client.loop_forever()
        except ConnectionRefusedError:
            print("Connection refused. Is the MQTT broker running?")
        except Exception as e:
            print(f"An error occurred: {e}. Retrying in 5 seconds...")
        time.sleep(5)

    
