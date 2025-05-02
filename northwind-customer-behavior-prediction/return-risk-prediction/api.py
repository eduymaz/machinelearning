from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import torch
import shap
import json

from data_loading import load_and_preprocess_data
from model import prepare_data, train_model, predict, ReturnRiskModel

app = FastAPI(
    title="Sipariş İade Riski Tahmin API",
    description="Müşterilerin siparişlerinin iade edilme riskini tahmin eden API",
    version="1.0.0"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler
model = None
scaler = None
explainer = None

class OrderInput(BaseModel):
    unit_price: float
    quantity: int
    discount: float

class PredictionResponse(BaseModel):
    return_risk: float
    risk_level: str
    explanation: dict

class ModelInfo(BaseModel):
    model_status: str
    feature_importance: dict

@app.on_event("startup")
async def startup_event():
    """API başlatıldığında modeli yükle ve eğit"""
    global model, scaler, explainer
    
    # Veriyi yükle ve hazırla
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Modeli eğit
    input_size = X_train.shape[1]
    model = train_model(X_train, y_train, input_size)
    
    # SHAP explainer oluştur
    background = torch.FloatTensor(X_train[:100])
    explainer = shap.DeepExplainer(model, background)

@app.get("/")
async def root():
    """API'nin çalıştığını kontrol et"""
    return {"message": "Sipariş İade Riski Tahmin API'si çalışıyor"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_return_risk(order: OrderInput):
    """Sipariş için iade riski tahmini yap"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model henüz yüklenmedi")
    
    # Giriş verisini hazırla
    spending = order.unit_price * order.quantity * (1 - order.discount)
    input_data = np.array([[order.unit_price, order.quantity, order.discount, spending]])
    input_data = scaler.transform(input_data)
    
    # Tahmin yap
    risk_score = float(predict(model, input_data)[0][0])
    
    # Risk seviyesini belirle
    if risk_score < 0.3:
        risk_level = "Düşük"
    elif risk_score < 0.7:
        risk_level = "Orta"
    else:
        risk_level = "Yüksek"
    
    # SHAP değerlerini hesapla
    shap_values = explainer.shap_values(torch.FloatTensor(input_data))
    feature_names = ['unit_price', 'quantity', 'discount', 'spending']
    explanation = {
        feature: float(value) 
        for feature, value in zip(feature_names, shap_values[0])
    }
    
    return PredictionResponse(
        return_risk=risk_score,
        risk_level=risk_level,
        explanation=explanation
    )

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Model bilgilerini getir"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model henüz yüklenmedi")
    
    # Feature importance hesapla
    feature_names = ['unit_price', 'quantity', 'discount', 'spending']
    background = torch.FloatTensor(np.zeros((1, len(feature_names))))
    shap_values = explainer.shap_values(background)
    
    feature_importance = {
        feature: float(abs(value))
        for feature, value in zip(feature_names, shap_values[0])
    }
    
    return ModelInfo(
        model_status="Aktif",
        feature_importance=feature_importance
    ) 