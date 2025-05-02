import requests

# API endpoint - port 9393 olarak güncellendi
url = "http://localhost:9393/predict"

# Örnek müşteri harcama verileri
data = {
    "customer_id": "CUST001",
    "spending": {
        "Beverages": 100.0,
        "Condiments": 50.0,
        "Confections": 75.0,
        "Dairy_Products": 60.0,
        "Grains_Cereals": 40.0,
        "Meat_Poultry": 80.0,
        "Produce": 30.0,
        "Seafood": 45.0
    }
}

# İstek gönder
response = requests.post(url, json=data)
print(response.json())