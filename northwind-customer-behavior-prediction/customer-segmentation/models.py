from pydantic import BaseModel
from typing import List, Optional

class CustomerSummary(BaseModel):
    customer_id: str
    total_orders: int
    orders_less_than_180: int
    percentage_less_than_180: float
    target: int
    risk_group: str

class CustomerResponse(BaseModel):
    customer_id: str
    risk_score: float
    risk_group: str
    recommendations: List[str]

class AnalysisResponse(BaseModel):
    total_customers: int
    risky_customers: int
    risk_distribution: dict
    top_risky_customers: List[CustomerSummary] 