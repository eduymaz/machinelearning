import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

from models import CategorySpending, PredictionResponse

app = FastAPI(
    title="Product Purchase Prediction API",
    description="API for predicting new product purchases based on customer spending patterns",
    version="1.0.0"
)

# Global variables for model and scaler
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    """Load the trained model and scaler on startup"""
    global model, scaler
    try:
        model = tf.keras.models.load_model('product_purchase_model.h5')
        scaler = joblib.load('scaler.joblib')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Product Purchase Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer_id: str, spending: CategorySpending):
    """Predict new product purchase probability for a customer"""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert spending data to DataFrame
        spending_dict = spending.dict()
        spending_df = pd.DataFrame([spending_dict])
        
        # Scale the features
        spending_scaled = scaler.transform(spending_df)
        
        # Make prediction
        prediction = model.predict(spending_scaled)[0][0]
        
        # Create response
        response = PredictionResponse(
            customer_id=customer_id,
            prediction=float(prediction),
            prediction_label="Will Purchase" if prediction > 0.5 else "Will Not Purchase"
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "Neural Network",
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "layers": len(model.layers)
    } 