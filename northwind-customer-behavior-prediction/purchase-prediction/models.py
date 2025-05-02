from pydantic import BaseModel
from typing import Dict, List

class CategorySpending(BaseModel):
    """Category spending data for prediction"""
    Beverages: float = 0.0
    Condiments: float = 0.0
    Confections: float = 0.0
    Dairy_Products: float = 0.0
    Grains_Cereals: float = 0.0
    Meat_Poultry: float = 0.0
    Produce: float = 0.0
    Seafood: float = 0.0

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    customer_id: str
    prediction: float
    prediction_label: str 