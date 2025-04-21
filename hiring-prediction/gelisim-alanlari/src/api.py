from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(
    title="İşe Alım Tahmin API",
    description="Adayların işe alım durumunu tahmin eden API",
    version="1.0.0"
)

class Candidate(BaseModel):
    tecrube_yili: float
    teknik_puan: float

class PredictionResponse(BaseModel):
    tahmin: int
    aciklama: str

# Model ve scaler'ı yükle
try:
    model = joblib.load('../data/processed/model.joblib')
    scaler = joblib.load('../data/processed/scaler.joblib')
except FileNotFoundError:
    raise RuntimeError("Model dosyaları bulunamadı. Lütfen önce model eğitimini çalıştırın.")

@app.get("/")
async def root():
    return {"message": "İşe Alım Tahmin API'sine Hoş Geldiniz"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(candidate: Candidate):
    """
    Adayın işe alım durumunu tahmin eder.
    
    - **tecrube_yili**: Adayın tecrübe yılı (0-10 arası)
    - **teknik_puan**: Adayın teknik sınav puanı (0-100 arası)
    """
    try:
        # Veriyi hazırla
        data = np.array([[candidate.tecrube_yili, candidate.teknik_puan]])
        
        # Veriyi ölçeklendir
        scaled_data = scaler.transform(data)
        
        # Tahmin yap
        prediction = model.predict(scaled_data)[0]
        
        # Sonucu hazırla
        if prediction == 0:
            message = "Tebrikler! Başvurunuz başarılı olmuştur."
        else:
            message = "Üzgünüz, başvurunuz başarısız olmuştur."
        
        return PredictionResponse(tahmin=int(prediction), aciklama=message)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(candidates: List[Candidate]):
    
    try:
        # Verileri hazırla
        data = np.array([[c.tecrube_yili, c.teknik_puan] for c in candidates])
        
        # Verileri ölçeklendir
        scaled_data = scaler.transform(data)
        
        # Tahminleri yap
        predictions = model.predict(scaled_data)
        
        # Sonuçları hazırla
        results = []
        for i, pred in enumerate(predictions):
            if pred == 0:
                message = "Tebrikler! Başvurunuz başarılı olmuştur."
            else:
                message = "Üzgünüz, başvurunuz bu sefer başarısız olmuştur."
            
            results.append({
                "aday_id": i + 1,
                "tahmin": int(pred),
                "aciklama": message
            })
        
        return {"sonuclar": results}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003) 