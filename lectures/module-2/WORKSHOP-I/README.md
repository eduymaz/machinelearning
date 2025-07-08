# Northwind Satış Dinamikleri: Makine Öğrenmesi Projesi


## 🧾 Proje Özeti
Bu projede, Northwind veritabanındaki sipariş verileri kullanılarak müşteri satın alma davranışı analiz edilmiştir. Amaç, bir müşterinin önümüzdeki 60 gün içinde tekrar sipariş verip vermeyeceğini tahmin eden öngörücü bir model geliştirmektir.

## 📁 Klasör ve Dosya Yapısı
```
├── workshop1-pair3.ipynb           # Ana analiz ve modelleme notebook'u
├── data_output.csv         # SQL'den çekilen ham veri
├── features_output.csv     # Özellik mühendisliği sonrası veri
├── features_final.csv      # Nihai modelleme veri seti
├── interactions_output.csv # Etkileşimli değişkenler
├── final_pipeline_model.pkl# Eğitilmiş pipeline model dosyası
├── raw_data.csv            # Ham veri
├── sql_query.txt           # Kullanılan SQL sorgusu
├── workshop1-pair3.pdf       # (Oluşacak) PDF rapor
```

## 🚀 Kurulum ve Gereksinimler
- Python 3.8+
- Jupyter Notebook
- Pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, sqlalchemy, psycopg2
- PostgreSQL (Northwind veritabanı yüklü)
- Pandoc (PDF için)
- LaTeX (PDF için, ör: MacTeX)

Kurulum için:
```sh
pip install pandas numpy scikit-learn matplotlib seaborn joblib sqlalchemy psycopg2-binary
```

## 🔎 Analiz ve Modelleme Adımları
1. **Veri Çekimi:**
   - PostgreSQL Northwind veritabanından SQLAlchemy ile veri çekildi.
   - `data_output.csv` olarak kaydedildi.
2. **Özellik Mühendisliği:**
   - Zaman bazlı göstergeler, sipariş sıklığı, harcama yoğunluğu, kargo gecikmesi ve etkileşim değişkenleri oluşturuldu.
   - Sonuçlar `features_output.csv` ve `features_final.csv` dosyalarına kaydedildi.
3. **Modelleme:**
   - LassoCV ile özellik seçimi yapıldı.
   - Pipeline içinde StandardScaler, PCA ve LogisticRegression kullanıldı.
   - Nested cross-validation ile model seçimi ve hiperparametre optimizasyonu gerçekleştirildi.
   - En iyi model `final_pipeline_model.pkl` olarak kaydedildi.
4. **Raporlama ve Görselleştirme:**
   - Tüm tablo ve grafik çıktıları `.ipynb` uzantılı dosya içerisinde verilmiştir.
   

## 📊 Çıktıların Kaydedilmesi
- DataFrame ve tablo görselleri: `img/tablo/`
- Grafik görselleri: `img/grafik/`
- Tüm görseller notebook kodlarında otomatik olarak kaydedilmektedir.

## 📄 PDF Raporu Oluşturma
Notebook'u PDF olarak dışa aktarmak için:

1. Pandoc ve LaTeX kurulu olmalı.
2. Terminalde aşağıdaki komutu çalıştırın:

```sh
jupyter nbconvert --to pdf --output calisma-pair3.pdf Sample1.ipynb
```

> **Not:** Eğer LaTeX eksikse, macOS için `brew install --cask mactex` ile kurabilirsiniz.


