from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List
import uuid

from db import models
from db.base import get_db
from . import schemas
from core.websocket_manager import manager

router = APIRouter()

# ===============================================================
# CREATE Endpoints (Phase 1 - Configuration)
# ===============================================================

@router.post("/mines", response_model=schemas.Mine, status_code=status.HTTP_201_CREATED)
def create_mine(mine: schemas.MineCreate, db: Session = Depends(get_db)):
    db_mine = models.Mine(id=str(uuid.uuid4()), name=mine.name, location=mine.location, contact_phone_number=mine.contact_phone_number)
    db.add(db_mine)
    db.commit()
    db.refresh(db_mine)
    return db_mine

@router.post("/mines/{mine_id}/zones", response_model=schemas.Zone, status_code=status.HTTP_201_CREATED)
def create_zone_for_mine(mine_id: str, zone: schemas.ZoneCreate, db: Session = Depends(get_db)):
    db_mine = db.query(models.Mine).filter(models.Mine.id == mine_id).first()
    if not db_mine:
        raise HTTPException(status_code=404, detail="Mine not found")
    
    db_zone = models.Zone(id=str(uuid.uuid4()), **zone.dict(), mine_id=mine_id)
    db.add(db_zone)
    db.commit()
    db.refresh(db_zone)
    return db_zone

@router.post("/zones/{zone_id}/sensors", response_model=schemas.Sensor, status_code=status.HTTP_201_CREATED)
def create_sensor_for_zone(zone_id: str, sensor: schemas.SensorCreate, db: Session = Depends(get_db)):
    db_zone = db.query(models.Zone).filter(models.Zone.id == zone_id).first()
    if not db_zone:
        raise HTTPException(status_code=404, detail="Zone not found")

    sensor_id = f"{sensor.sensor_type[:3].lower()}-{db_zone.mine_id[:4]}-{zone_id[:4]}-{str(uuid.uuid4())[:4]}"
    
    mqtt_topic = f"mines/{db_zone.mine_id}/zones/{zone_id}/sensors/{sensor_id}/data"

    db_sensor = models.Sensor(
        id=sensor_id,
        **sensor.dict(),
        zone_id=zone_id,
        mqtt_topic=mqtt_topic
    )
    db.add(db_sensor)
    db.commit()
    db.refresh(db_sensor)
    return db_sensor

# ===============================================================
# READ Endpoints (For Frontend Consumption)
# ===============================================================

@router.get("/mines", response_model=List[schemas.Mine])
def get_all_mines(db: Session = Depends(get_db)):
    """
    Retrieve a list of all configured mines.
    """
    return db.query(models.Mine).all()

@router.get("/mines/{mine_id}", response_model=schemas.Mine)
def get_mine_details(mine_id: str, db: Session = Depends(get_db)):
    """
    Retrieve all details for a specific mine, including its zones and sensors.
    """
    db_mine = db.query(models.Mine).filter(models.Mine.id == mine_id).first()
    if not db_mine:
        raise HTTPException(status_code=404, detail="Mine not found")
    return db_mine

@router.get("/zones/{zone_id}", response_model=schemas.Zone)
def get_zone_details(zone_id: str, db: Session = Depends(get_db)):
    """
    Retrieve all details for a specific zone, including its configured sensors.
    """
    db_zone = db.query(models.Zone).filter(models.Zone.id == zone_id).first()
    if not db_zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return db_zone

# ===============================================================
# WebSocket Endpoint (For Real-Time Communication)
# ===============================================================

@router.websocket("/ws/{mine_id}")
async def websocket_endpoint(websocket: WebSocket, mine_id: str):
    """
    Handles WebSocket connections for a specific mine for real-time updates.
    """
    await manager.connect(websocket, mine_id)
    print(f"Dashboard connected for mine: {mine_id}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, mine_id)
        print(f"Dashboard disconnected for mine: {mine_id}")

