# Makine Öğrenmesi Projesi: Müşteri Davranış Analizi ve Tahmini

Bu depo, müşteri davranışlarını analiz eden ve farklı yaklaşımlar kullanarak tahminler yapan üç ayrı makine öğrenmesi projesini içermektedir. Her proje, gelişmiş makine öğrenmesi teknikleri kullanarak belirli iş problemlerini çözmek üzere tasarlanmıştır.

## Proje Yapısı

```
northwind-customer-behavior-prediction/
├── customer-segmentation/          # Müşteri Segmentasyonu Analizi (Port: 8880)
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
├── return-risk-prediction/          # Sipariş İade Riski Tahmini (Port: 8787)
│   ├── data_loading.py
│   ├── model.py
│   ├── api.py
│   ├── run_api.py
│   ├── setup_and_run.py
│   └── requirements.txt
│
└── purchase-prediction/          # Yeni Ürün Satın Alma Tahmini (Port: 9393)
    ├── automate_ml.py
    ├── models.py
    ├── api.py
    ├── run_api.py
    ├── setup_and_run.py
    └── requirements.txt
```

## Proje Açıklamaları

### 1. Müşteri Segmentasyonu Analizi
Bu proje, çeşitli kümeleme teknikleri kullanarak müşteri segmentasyonuna odaklanmaktadır. Müşteri satın alma kalıplarını ve davranışlarını analiz ederek farklı müşteri gruplarını belirler.

**Temel Özellikler:**
- Veri ön işleme ve özellik mühendisliği
- Çoklu kümeleme algoritmaları uygulaması
- Müşteri segmentlerinin interaktif görselleştirmesi
- Segment tahmini için RESTful API (Port: 8880)
- Otomatik kurulum ve dağıtım

### 2. Sipariş İade Riski Tahmini
Müşteri ve sipariş özelliklerine dayalı olarak sipariş iade olasılığını tahmin eden derin öğrenme tabanlı bir sistem.

**Temel Özellikler:**
- Risk tahmini için sinir ağı modeli
- Özellik önem analizi
- Gerçek zamanlı tahmin API'si (Port: 8787)
- Model performans izleme
- Otomatik dağıtım pipeline'ı

### 3. Yeni Ürün Satın Alma Tahmini
Bu proje, müşterilerin farklı ürün kategorilerindeki geçmiş harcama kalıplarına dayanarak yeni ürün satın alma olasılığını tahmin eder.

**Temel Özellikler:**
- Kategori bazlı harcama analizi
- Satın alma tahmini için derin öğrenme modeli
- Gerçek zamanlı tahmin API'si (Port: 9393)
- Otomatik model eğitimi ve dağıtımı
- İnteraktif API dokümantasyonu

## Teknik Detaylar

### Veri Kaynakları
- Müşteri işlem verileri
- Ürün kategori bilgileri
- Sipariş geçmişi
- Müşteri demografik bilgileri

### Kullanılan Teknolojiler
- Python 3.8+
- TensorFlow/Keras
- FastAPI
- Pandas
- Scikit-learn
- NumPy
- Joblib

### API Endpoint'leri
Her proje aşağıdaki ortak endpoint'leri sunan RESTful API'ler içerir:
- `/docs` - İnteraktif API dokümantasyonu
- `/predict` - Tahmin endpoint'i
- `/model-info` - Model bilgisi
- `/health` - Sağlık kontrolü

## Kurulum ve Yükleme

1. Depoyu klonlayın:
```bash
git clone https://github.com/eduymaz/northwind-customer-behavior-prediction.git
cd northwind-customer-behavior-prediction
```

2. Çalıştırmak istediğiniz projeyi seçin ve dizinine gidin:
```bash
cd customer-segmentation  
```

3. Kurulum scriptini çalıştırın:
```bash
python setup_and_run.py
```

Bu işlem:
- Gerekli bağımlılıkları yükleyecek
- Modeli eğitecek
- API sunucusunu başlatacak
- API dokümantasyonunu tarayıcınızda açacak

## API Kullanımı

Her projenin API'sine kendi port numarası üzerinden erişilebilir:

### Müşteri Segmentasyonu API'si (Port: 8880)
```python
import requests

url = "http://localhost:8880/predict"
data = {
    # Projeye özel giriş verileri
}
response = requests.post(url, json=data)
print(response.json())
```

### İade Riski Tahmin API'si (Port: 8787)
```python
import requests

url = "http://localhost:8787/predict"
data = {
    # Projeye özel giriş verileri
}
response = requests.post(url, json=data)
print(response.json())
```

### Satın Alma Tahmin API'si (Port: 9393)
```python
import requests

url = "http://localhost:9393/predict"
data = {
    # Projeye özel giriş verileri
}
response = requests.post(url, json=data)
print(response.json())
```

## Proje-Spesifik Gereksinimler

Her projenin kendi `requirements.txt` dosyası ve özel bağımlılıkları vardır. Kurulum scripti gerekli paketleri otomatik olarak yükleyecektir.


