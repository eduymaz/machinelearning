from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from datetime import datetime
from uuid import uuid4, UUID

app = FastAPI(
    title="Vehicle License Plate API",
    description="API for managing vehicle license plates",
    version="1.0.0"
)

# Pydantic model for vehicle data
class VehicleCreate(BaseModel):
    plate: str = Field(..., example="34ABC123", min_length=5, max_length=10)
    brand: str = Field(..., example="Toyota")
    model: str = Field(..., example="Corolla")
    year: int = Field(..., example=2020, gt=1900, lt=2100)
    color: str = Field(..., example="Red")
    
class Vehicle(VehicleCreate):
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True
        
class VehicleUpdate(BaseModel):
    plate: Optional[str] = Field(None, example="34XYZ789", min_length=5, max_length=10)
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = Field(None, gt=1900, lt=2100)
    color: Optional[str] = None

# In-memory database
vehicles_db = []

# GET endpoint to retrieve all vehicles
@app.get("/vehicles", response_model=List[Vehicle], tags=["Vehicles"])
async def get_all_vehicles():
    return vehicles_db

# GET endpoint to retrieve a specific vehicle by ID
@app.get("/vehicles/{vehicle_id}", response_model=Vehicle, tags=["Vehicles"])
async def get_vehicle(vehicle_id: UUID):
    for vehicle in vehicles_db:
        if vehicle["id"] == vehicle_id:
            return vehicle
    raise HTTPException(status_code=404, detail="Vehicle not found")

# POST endpoint to create a new vehicle
@app.post("/vehicles", response_model=Vehicle, status_code=status.HTTP_201_CREATED, tags=["Vehicles"])
async def create_vehicle(vehicle: VehicleCreate):
    # Check if plate already exists
    for v in vehicles_db:
        if v["plate"] == vehicle.plate:
            raise HTTPException(status_code=400, detail="License plate already registered")
    
    new_vehicle = {
        "id": uuid4(),
        "created_at": datetime.now(),
        "updated_at": None,
        **vehicle.dict()
    }
    vehicles_db.append(new_vehicle)
    return new_vehicle

# PUT endpoint to update a vehicle
@app.put("/vehicles/{vehicle_id}", response_model=Vehicle, tags=["Vehicles"])
async def update_vehicle(vehicle_id: UUID, vehicle_update: VehicleUpdate):
    for idx, vehicle in enumerate(vehicles_db):
        if vehicle["id"] == vehicle_id:
            # Check if new plate already exists (if updating plate)
            if vehicle_update.plate and vehicle_update.plate != vehicle["plate"]:
                for v in vehicles_db:
                    if v["plate"] == vehicle_update.plate:
                        raise HTTPException(status_code=400, detail="License plate already registered")
            
            # Update only provided fields
            update_data = vehicle_update.dict(exclude_unset=True)
            updated_vehicle = {**vehicle, **update_data, "updated_at": datetime.now()}
            vehicles_db[idx] = updated_vehicle
            return updated_vehicle
            
    raise HTTPException(status_code=404, detail="Vehicle not found")

# DELETE endpoint to remove a vehicle
@app.delete("/vehicles/{vehicle_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Vehicles"])
async def delete_vehicle(vehicle_id: UUID):
    for idx, vehicle in enumerate(vehicles_db):
        if vehicle["id"] == vehicle_id:
            vehicles_db.pop(idx)
            return
    raise HTTPException(status_code=404, detail="Vehicle not found")

# GET endpoint to find vehicles by license plate
@app.get("/find-plates", response_model=List[Vehicle], tags=["Search"])
async def find_vehicles_by_plate(plate: str):
    matching_vehicles = [v for v in vehicles_db if plate.lower() in v["plate"].lower()]
    return matching_vehicles

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to the Vehicle License Plate API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
