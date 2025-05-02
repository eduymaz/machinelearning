from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import CustomerSummary, CustomerResponse, AnalysisResponse
from data_loading import load_data
from data_preprocessing import preprocess_data
from feature_engineering import create_summary_features
import pandas as pd
from typing import List

app = FastAPI(
    title="Müşteri Risk Analizi API",
    description="Müşterilerin sipariş davranışlarına göre risk analizi yapan API",
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
df = None
df_unique = None
summary = None

@app.on_event("startup")
async def startup_event():
    """API başlatıldığında veriyi yükle"""
    global df, df_unique, summary
    df = load_data()
    df_unique = preprocess_data(df)
    summary = create_summary_features(df_unique)

@app.get("/")
async def root():
    """API'nin çalıştığını kontrol et"""
    return {"message": "Müşteri Risk Analizi API'si çalışıyor"}

@app.get("/customers/{customer_id}", response_model=CustomerResponse)
async def get_customer_risk(customer_id: str):
    """Belirli bir müşterinin risk analizini getir"""
    if customer_id not in summary['customer_id'].values:
        raise HTTPException(status_code=404, detail="Müşteri bulunamadı")
    
    customer_data = summary[summary['customer_id'] == customer_id].iloc[0]
    
    # Öneriler oluştur
    recommendations = []
    if customer_data['risk_group'] in ['mid-high', 'high']:
        recommendations.append("Müşteri ile düzenli iletişim kurulmalı")
        recommendations.append("Özel kampanyalar sunulmalı")
    if customer_data['percentage_less_than_180'] > 80:
        recommendations.append("Müşteri sadakat programına dahil edilmeli")
    
    return CustomerResponse(
        customer_id=customer_id,
        risk_score=customer_data['percentage_less_than_180'],
        risk_group=customer_data['risk_group'],
        recommendations=recommendations
    )

@app.get("/analysis", response_model=AnalysisResponse)
async def get_analysis():
    """Genel analiz sonuçlarını getir"""
    risk_distribution = summary['risk_group'].value_counts().to_dict()
    top_risky = summary[summary['target'] == 1].sort_values(
        'percentage_less_than_180', 
        ascending=False
    ).head(5)
    
    return AnalysisResponse(
        total_customers=len(summary),
        risky_customers=summary['target'].sum(),
        risk_distribution=risk_distribution,
        top_risky_customers=[
            CustomerSummary(**row) for row in top_risky.to_dict('records')
        ]
    )

@app.get("/customers", response_model=List[CustomerSummary])
async def get_all_customers():
    """Tüm müşterilerin özet bilgilerini getir"""
    return [CustomerSummary(**row) for row in summary.to_dict('records')] 