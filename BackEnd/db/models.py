from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base
import uuid

# A helper function to generate unique string IDs for our records.
def generate_uuid():
    return str(uuid.uuid4())

class Mine(Base):
    __tablename__ = "mines"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, index=True, nullable=False)
    location = Column(String)
    contact_phone_number = Column(String) # For sending SMS alerts
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # This creates a relationship so you can easily access all zones belonging to a mine.
    # e.g., my_mine.zones
    zones = relationship("Zone", back_populates="mine", cascade="all, delete-orphan")

class Zone(Base):
    __tablename__ = "zones"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, index=True, nullable=False)
    mine_id = Column(String, ForeignKey("mines.id"))
    
    # This links a zone back to its parent mine.
    mine = relationship("Mine", back_populates="zones")
    # This links a zone to all its child sensors.
    sensors = relationship("Sensor", back_populates="zone", cascade="all, delete-orphan")

class Sensor(Base):
    __tablename__ = "sensors"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    custom_name = Column(String)
    sensor_type = Column(String, index=True, nullable=False) # e.g., "Strain Gauge", "Piezometer"
    model = Column(String)
    zone_id = Column(String, ForeignKey("zones.id"))
    
    # The unique MQTT topic that this specific sensor should publish its data to.
    mqtt_topic = Column(String, unique=True)
    
    # This links a sensor back to its parent zone.
    zone = relationship("Zone", back_populates="sensors")

