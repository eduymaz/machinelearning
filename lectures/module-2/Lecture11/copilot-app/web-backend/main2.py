from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from jose import JWTError, jwt
from passlib.context import CryptContext

# Security settings for JWT
SECRET_KEY = "a8f7d9e2c5b3k4j6h8g7f9e2d5c3b6a9s8d7f6g5h4j3k2l1"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password handling
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(
    title="Vehicle License Plate API",
    description="API for managing vehicle license plates",
    version="1.0.0"
)

# User models
class UserBase(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: UUID
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

# Token models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Vehicle models
class VehicleBase(BaseModel):
    plate: str = Field(..., example="34ABC123", min_length=5, max_length=10)
    brand: str = Field(..., example="Toyota")
    model: str = Field(..., example="Corolla")

class VehicleCreate(VehicleBase):
    year: int = Field(..., example=2020, gt=1900, lt=2100)
    color: str = Field(..., example="Red")

class Vehicle(VehicleCreate):
    id: UUID
    owner: str  # Username of the owner
    created_at: datetime

# In-memory databases
users_db = {}
vehicles_db = []

# Helper functions for authentication
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Initialize some test users
def init_users():
    # Create admin user
    admin_id = uuid4()
    users_db["admin"] = {
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "Administrator",
        "id": admin_id,
        "hashed_password": get_password_hash("adminpass"),
        "disabled": False
    }
    
    # Create regular user
    user_id = uuid4()
    users_db["user"] = {
        "username": "user",
        "email": "user@example.com",
        "full_name": "Regular User",
        "id": user_id,
        "hashed_password": get_password_hash("userpass"),
        "disabled": False
    }

# Authentication endpoints
@app.post("/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", status_code=status.HTTP_201_CREATED, tags=["Authentication"])
async def register_user(user: UserCreate):
    if user.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    user_id = uuid4()
    hashed_password = get_password_hash(user.password)
    user_dict = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "id": user_id,
        "hashed_password": hashed_password,
        "disabled": False
    }
    users_db[user.username] = user_dict
    
    return {"username": user.username, "message": "User registered successfully"}

@app.get("/users/me", response_model=User, tags=["Users"])
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# Vehicle endpoints
@app.post("/vehicles", response_model=Vehicle, status_code=status.HTTP_201_CREATED, tags=["Vehicles"])
async def create_vehicle(vehicle: VehicleCreate, current_user: User = Depends(get_current_active_user)):
    # Check if plate already exists
    for v in vehicles_db:
        if v["plate"] == vehicle.plate:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="License plate already registered"
            )
    
    vehicle_id = uuid4()
    new_vehicle = {
        "id": vehicle_id,
        "owner": current_user.username,
        "created_at": datetime.now(),
        **vehicle.dict()
    }
    vehicles_db.append(new_vehicle)
    return new_vehicle

@app.get("/vehicles", response_model=List[Vehicle], tags=["Vehicles"])
async def get_vehicles(current_user: User = Depends(get_current_active_user)):
    if current_user.username == "admin":
        return vehicles_db
    return [v for v in vehicles_db if v["owner"] == current_user.username]

@app.get("/vehicles/{vehicle_id}", response_model=Vehicle, tags=["Vehicles"])
async def get_vehicle(vehicle_id: UUID, current_user: User = Depends(get_current_active_user)):
    for vehicle in vehicles_db:
        if vehicle["id"] == vehicle_id:
            # Check if user is admin or vehicle owner
            if current_user.username == "admin" or vehicle["owner"] == current_user.username:
                return vehicle
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this vehicle"
            )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Vehicle not found"
    )

@app.get("/", tags=["Root"])
async def root():
    return {"message": "GYK Backend API'ye hoş geldiniz!"}

# Initialize users at startup
@app.on_event("startup")
async def startup_event():
    init_users()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

