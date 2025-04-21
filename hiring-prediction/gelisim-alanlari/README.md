# 🚀 Gelişim Alanları

Bu klasör, projenin gelişim alanlarını içeren çalışmaları barındırır.

## 📁 Klasör Yapısı

```
gelisim-alanlari/
├── src/
│   ├── kernel_denemeleri.py    # Farklı kernel'ların test edilmesi
│   ├── api.py                  # FastAPI servisi
│   └── parametre_tuning.py     # Model parametrelerinin optimizasyonu
└── notebooks/                  # Analiz notebook'ları
```

## 📋 Gelişim Alanları

### 1. Kernel Denemeleri (`kernel_denemeleri.py`)
- Farklı SVM kernel'larının test edilmesi
- Her kernel için performans metriklerinin hesaplanması
- Karar sınırlarının görselleştirilmesi
- Sonuçların CSV dosyasına kaydedilmesi

### 2. FastAPI Servisi (`api.py`)
- RESTful API oluşturulması
- Tekil ve toplu tahmin endpoint'leri
- Hata yönetimi ve doğrulama
- Swagger UI desteği

### 3. Parametre Tuning (`parametre_tuning.py`)
- Grid Search ile parametre optimizasyonu
- C ve gamma parametrelerinin test edilmesi
- Sonuçların görselleştirilmesi
- En iyi modelin kaydedilmesi

## 🚀 Kullanım

1. Kernel denemeleri için:
```bash
python src/kernel_denemeleri.py
```

2. API'yi başlatmak için:
```bash
python src/api.py
```

3. Parametre tuning için:
```bash
python src/parametre_tuning.py
```

## 📊 Çıktılar

Her script, sonuçları `data/processed/` klasörüne kaydeder:
- Kernel denemeleri: `kernel_results.csv` ve karar sınırı görselleri
- Parametre tuning: `parameter_tuning_results.csv`, `parameter_heatmap.png` ve `best_model.joblib`

## 🔍 Swagger UI Kullanım Kılavuzu

API'yi test etmek için Swagger UI'ı kullanabilirsiniz. API başlatıldıktan sonra http://localhost:8003/docs adresine giderek aşağıdaki adımları izleyebilirsiniz:

### 1. Ana Sayfa (GET /)
- "Try it out" butonuna tıklayın
- "Execute" butonuna tıklayın
- Beklenen yanıt:
```json
{
  "message": "İşe Alım Tahmin API'sine Hoş Geldiniz"
}
```

### 2. Tekil Tahmin (POST /predict)
- "Try it out" butonuna tıklayın
- Request body kısmına örnek veri girin:
```json
{
  "tecrube_yili": 3.5,
  "teknik_puan": 75.0
}
```
- "Execute" butonuna tıklayın
- Beklenen yanıt:
```json
{
  "tahmin": 0,
  "aciklama": "Tebrikler! Başvurunuz başarılı olmuştur."
}
```

### 3. Toplu Tahmin (POST /batch_predict)
- "Try it out" butonuna tıklayın
- Request body kısmına örnek veri girin:
```json
[
  {
    "tecrube_yili": 1.5,
    "teknik_puan": 55.0
  },
  {
    "tecrube_yili": 4.0,
    "teknik_puan": 85.0
  }
]
```
- "Execute" butonuna tıklayın
- Beklenen yanıt:
```json
{
  "sonuclar": [
    {
      "aday_id": 1,
      "tahmin": 1,
      "aciklama": "Üzgünüz, başvurunuz başarısız olmuştur."
    },
    {
      "aday_id": 2,
      "tahmin": 0,
      "aciklama": "Tebrikler! Başvurunuz başarılı olmuştur."
    }
  ]
}
```

### ⚠️ Önemli Notlar
- `tecrube_yili` değeri 0-10 arasında olmalıdır
- `teknik_puan` değeri 0-100 arasında olmalıdır
- Geçersiz değerler için API hata mesajı döndürecektir
- Toplu tahmin için birden fazla aday bilgisi gönderilebilir

### 🔧 API Özellikleri
- Otomatik veri doğrulama
- Hata yönetimi
- Swagger UI dokümantasyonu
- JSON formatında giriş/çıkış
- Batch işlem desteği 