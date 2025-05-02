import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

def load_and_preprocess_data():
    """Load and preprocess the data"""
    # Load data
    data = pd.read_csv("question3.csv")
    
    # Calculate spending
    data['spend'] = data['unit_price'] * data['quantity'] * (1 - data['discount'])
    
    # Create category spending matrix
    category_spend = data.groupby(['customer_id', 'category_name'])['spend'].sum().unstack(fill_value=0)
    
    return category_spend

def prepare_target_variable(data):
    """Prepare target variable for new product purchase prediction"""
    data['new_product_purchase'] = 0  # Initialize with 0
    data.loc[data['product_id'] == 'new_product_id', 'new_product_purchase'] = 1
    new_product_purchase = data.groupby('customer_id')['new_product_purchase'].max()
    return new_product_purchase

def build_model(input_dim):
    """Build and compile the neural network model"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    """Train the model with validation data"""
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Build and train model
    model = build_model(X_train.shape[1])
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_scaled, y_val),
        verbose=1
    )
    
    return model, scaler, history

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model performance on test data"""
    X_test_scaled = scaler.transform(X_test)
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
    return test_loss, test_accuracy

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    category_spend = load_and_preprocess_data()
    
    # Prepare target variable
    print("Preparing target variable...")
    data = pd.read_csv("question3.csv")
    new_product_purchase = prepare_target_variable(data)
    
    # Split data
    print("Splitting data into train/validation/test sets...")
    X = category_spend
    y = new_product_purchase
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Second split: separate validation set from training set
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    # Train model
    print("Training model...")
    model, scaler, history = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = evaluate_model(model, X_test, y_test, scaler)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save model and scaler
    print("Saving model and scaler...")
    model.save('product_purchase_model.h5')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model saved as 'product_purchase_model.h5'")
    print("Scaler saved as 'scaler.joblib'")

if __name__ == "__main__":
    main() 