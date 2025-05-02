import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if requirements.txt exists and install requirements"""
    if not os.path.exists('requirements.txt'):
        print("Error: requirements.txt not found!")
        return False
    
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        print("Error: Failed to install requirements!")
        return False

def train_model():
    """Run the model training script"""
    print("\nTraining the model...")
    try:
        subprocess.check_call([sys.executable, "automate_ml.py"])
        return True
    except subprocess.CalledProcessError:
        print("Error: Model training failed!")
        return False

def check_model_files():
    """Check if model files exist"""
    required_files = ['product_purchase_model.h5', 'scaler.joblib']
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: {file} not found!")
            return False
    return True

def start_api():
    """Start the FastAPI server"""
    print("\nStarting API server...")
    try:
        # Start the API server in a new process
        api_process = subprocess.Popen([sys.executable, "run_api.py"])
        
        # Wait for the server to start
        time.sleep(3)
        
        # Open the API documentation in the default browser
        webbrowser.open('http://localhost:9393/docs')
        
        print("\nAPI is running!")
        print("API Documentation: http://localhost:9393/docs")
        print("API Root: http://localhost:9393")
        print("Model Info: http://localhost:9393/model-info")
        print("\nPress Ctrl+C to stop the server...")
        
        # Keep the script running
        api_process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping API server...")
        api_process.terminate()
    except Exception as e:
        print(f"Error: Failed to start API server: {str(e)}")
        return False

def main():
    print("Starting setup and run process...")
    
    # Step 1: Install requirements
    if not check_requirements():
        return
    
    # Step 2: Train model
    if not train_model():
        return
    
    # Step 3: Check if model files exist
    if not check_model_files():
        return
    
    # Step 4: Start API
    start_api()

if __name__ == "__main__":
    main() 