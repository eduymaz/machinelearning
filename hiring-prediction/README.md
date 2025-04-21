# 🎯 İşe Alımda Aday Seçimi: SVM ile Başvuru Değerlendirme

## 📋 Proje Özeti
Bu proje, yazılım geliştirici pozisyonu için başvuran adayların tecrübe yılı ve teknik sınav puanına göre işe alınıp alınmamasını tahmin eden bir makine öğrenimi modeli geliştirmeyi amaçlamaktadır.

## 🎯 Hedefler
- SVM (Support Vector Machine) kullanarak aday değerlendirme modeli oluşturma
- Veri üretimi ve ön işleme
- Model eğitimi ve değerlendirme
- Tahmin servisi oluşturma

## 📊 Veri Yapısı
- **tecrube_yili**: Adayın toplam yazılım deneyimi (0-10 yıl)
- **teknik_puan**: Teknik sınav puanı (0-100)
- **etiket**: 
  - 1: İşe alınmadı
  - 0: İşe alındı

## 🎯 Etiketleme Kriteri
- Tecrübesi 2 yıldan az VE sınav puanı 60'tan düşük olanlar işe alınmıyor.

## 📋 Görevler

### 1. Veri Üretimi
- Faker kütüphanesi ile 200 başvuru verisi üretimi
- Tecrübe ve teknik puan dağılımlarının oluşturulması
- Etiketleme kriterine göre veri etiketleme

### 2. Veri Ön İşleme
- Veri setinin eğitim ve test olarak ayrılması
- StandardScaler ile veri ölçeklendirme
- Veri görselleştirme

### 3. Model Geliştirme
- SVC(kernel='linear') modelinin eğitimi
- Karar sınırının görselleştirilmesi
- Model performans metriklerinin hesaplanması

### 4. Tahmin Sistemi
- Kullanıcıdan tecrübe ve teknik puan alarak tahmin yapma
- Tahmin sonuçlarının görselleştirilmesi

### 5. Model Değerlendirme
- accuracy_score hesaplama
- confusion_matrix oluşturma
- classification_report ile detaylı analiz

## 🚀 Gelişim Alanları

### 1. Model İyileştirme
- Farklı kernel'lerin denenmesi
- Parametre optimizasyonu (C, gamma)
- Cross-validation

### 2. API Geliştirme
- FastAPI ile tahmin servisi oluşturma
- Swagger UI entegrasyonu
- API dokümantasyonu

## 📁 Proje Yapısı
```
hiring-prediction/
├── data/                      # Veri klasörü
│   ├── raw/                   # Ham veriler
│   │   └── candidates.csv     # Aday verileri
│   └── processed/            # İşlenmiş veriler
│       ├── model.joblib      # Eğitilmiş model
│       ├── scaler.joblib     # Ölçeklendirici
│       ├── confusion_matrix.png  # Karışıklık matrisi
│       └── decision_boundary.png # Karar sınırı görseli
│
├── src/                       # Kaynak kodlar
│   ├── data_generation.py    # Veri oluşturma
│   ├── model_training.py     # Model eğitimi
│   └── prediction.py         # Tahmin yapma
│
├── gelisim-alanlari/
│   ├── data/                      # Veri klasörü
│   │   ├── raw/                   # Ham veriler
│   │   │   └── candidates.csv     # Aday verileri
│   │   └── processed/            # İşlenmiş veriler
│   │       ├── model.joblib      # Eğitilmiş model
│   │       └── scaler.joblib     # Ölçeklendirici
    ├── src/                       # Kaynak kodlar
│   │    ├── kernel_denemeleri.py    # Veri oluşturma
│   │    ├── parametre_tuning.py     # Model eğitimi/modelde güncelleme
│   │    └── api.py         # Swager bağlanma api
│               
│
└── notebooks/                 # Jupyter notebook'lar
    ├── hiring_prediction.ipynb  # Ana analiz ve görselleştirme
    ├── ARGE.ipynb  # Arge çalışmalarının yer aldığı notebook 
    └── candidates.csv     # Aday verileri
```

## 🛠️ Teknolojiler
- Python 3.12+
- Scikit-learn
- Faker
- FastAPI
- Matplotlib
- Pandas
- NumPy

## 🚀 Kurulum
```bash
# Gerekli paketlerin yüklenmesi
pip install -r requirements.txt

# Veri üretimi
python src/data_generation.py

# Model eğitimi
python src/model_training.py

# API başlatma
python src/api.py
```

## 📊 Performans Metrikleri
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🔍 Sonraki Adımlar
- Daha fazla özellik eklenmesi
- Model optimizasyonu
  - Web arayüzü geliştirilmesi (...)
  - Gerçek zamanlı tahmin sistemi (...)

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/eduymaz">eduymaz</a></sub>
</div> 
