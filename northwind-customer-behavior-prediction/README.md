# Machine Learning Project: Customer Behavior Analysis and Prediction

This repository contains three distinct machine learning projects focused on analyzing customer behavior and making predictions using different approaches. Each project is designed to solve specific business problems using advanced machine learning techniques.

## Project Structure

```
northwind-customer-behavior-prediction/
├── customer-segmentation/          # Customer Segmentation Analysis (Port: 8880)
│   ├── data_loading.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── analysis.py
│   ├── main.py
│   ├── models.py
│   ├── api.py
│   ├── run_api.py
│   ├── setup_and_run.py
│   └── requirements.txt
│
├── return-risk-prediction/          # Order Return Risk Prediction (Port: 8787)
│   ├── data_loading.py
│   ├── model.py
│   ├── api.py
│   ├── run_api.py
│   ├── setup_and_run.py
│   └── requirements.txt
│
└── purchase-prediction/          # New Product Purchase Prediction (Port: 9393)
    ├── automate_ml.py
    ├── models.py
    ├── api.py
    ├── run_api.py
    ├── setup_and_run.py
    └── requirements.txt
```

## Project Descriptions

### 1. Customer Segmentation Analysis
This project focuses on customer segmentation using various clustering techniques. It analyzes customer purchasing patterns and behaviors to identify distinct customer groups.

**Key Features:**
- Data preprocessing and feature engineering
- Multiple clustering algorithms implementation
- Interactive visualization of customer segments
- RESTful API for segment prediction (Port: 8880)
- Automated setup and deployment

### 2. Order Return Risk Prediction
A deep learning-based system that predicts the likelihood of order returns based on customer and order characteristics.

**Key Features:**
- Neural network model for risk prediction
- Feature importance analysis
- Real-time prediction API (Port: 8787)
- Model performance monitoring
- Automated deployment pipeline

### 3. New Product Purchase Prediction
This project predicts customer likelihood to purchase new products based on their historical spending patterns across different product categories.

**Key Features:**
- Category-based spending analysis
- Deep learning model for purchase prediction
- Real-time prediction API (Port: 9393)
- Automated model training and deployment
- Interactive API documentation

## Technical Details

### Data Sources
- Customer transaction data
- Product category information
- Order history
- Customer demographics

### Technologies Used
- Python 3.8+
- TensorFlow/Keras
- FastAPI
- Pandas
- Scikit-learn
- NumPy
- Joblib

### API Endpoints
Each project exposes RESTful APIs with the following common endpoints:
- `/docs` - Interactive API documentation
- `/predict` - Prediction endpoint
- `/model-info` - Model information
- `/health` - Health check

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/eduymaz/northwind-customer-behavior-prediction.git
cd northwind-customer-behavior-prediction
```

2. Choose the project you want to run and navigate to its directory:
```bash
cd customer-segmentation  # or return-risk-prediction or purchase-prediction
```

3. Run the setup script:
```bash
python setup_and_run.py
```

This will:
- Install required dependencies
- Train the model
- Start the API server
- Open the API documentation in your browser

## API Usage

Each project's API can be accessed at its specific port:

### Customer Segmentation API (Port: 8880)
```python
import requests

url = "http://localhost:8880/predict"
data = {
    # Project-specific input data
}
response = requests.post(url, json=data)
print(response.json())
```

### Return Risk Prediction API (Port: 8787)
```python
import requests

url = "http://localhost:8787/predict"
data = {
    # Project-specific input data
}
response = requests.post(url, json=data)
print(response.json())
```

### Purchase Prediction API (Port: 9393)
```python
import requests

url = "http://localhost:9393/predict"
data = {
    # Project-specific input data
}
response = requests.post(url, json=data)
print(response.json())
```

## Project-Specific Requirements

Each project has its own `requirements.txt` file with specific dependencies. The setup script will automatically install the required packages. 