# CNN - Convolutional 

## Kullanım Alanları

### Bilgisayarlı Görü Uygulamaları
- **Görüntü Sınıflandırma**: Nesneleri kategorilere ayırma
- **Nesne Tespiti**: Görüntüdeki nesnelerin yerini belirleme
- **Yüz Tanıma**: Fotoğraflardan kişi kimlik doğrulama
- **OCR (Optical Character Recognition)**: Metin tanıma

### Tıbbi Görüntü Analizi
- **Radyoloji**: MR, CT, X-Ray görüntü analizi
- **Patoloji**: Mikroskobik doku analizi
- **Dermatoloji**: Cilt lezyonu ve kanser tespiti
- **Oftalmoloji**: Retina hastalıkları tespiti

### Endüstriyel Uygulamalar
- **Kalite Kontrol**: Üretim hattında hata tespiti
- **Robotik Görme**: Otonom sistemler
- **Tarım**: Mahsul analizi ve hastalık tespiti
- **Güvenlik**: Yüz tanıma ve davranış analizi

### Eğlence ve Sosyal Medya
- **Filtreler**: Instagram, Snapchat filtreleri
- **Stil Transferi**: Artistik görüntü dönüşümü
- **Deepfake**: Video ve görüntü manipülasyonu
- **Oyun Geliştirme**: Gerçek zamanlı görüntü işleme Network

## Kavramlar ve Tanımlar

**Convolutional Neural Network (CNN)**, görsel verileri işlemek ve anlamak için özel olarak tasarlanmış derin öğrenme mimarisidir. CNN'ler, geleneksel sinir ağlarından farklı olarak, görüntülerdeki uzamsal ve hiyerarşik özellikleri etkili bir şekilde öğrenebilir.

### Geleneksel Sinir Ağları vs CNN

| Özellik | Geleneksel Sinir Ağları | CNN |
|---------|-------------------------|-----|
| Veri İşleme | Piksel piksel işleme | Bölgesel özellik çıkarma |
| Parametre Sayısı | Çok yüksek | Paylaşımlı ağırlıklar ile düşük |
| Uzamsal İlişki | Göz ardı edilir | Korunur ve kullanılır |
| Performans | Görüntülerde düşük | Görüntülerde yüksek |

### CNN'in Avantajları

1. **Translation Invariance**: Nesnelerin konumundan bağımsız tanıma
2. **Parameter Sharing**: Aynı filtrelerin tüm görüntüde kullanılması
3. **Hierarchical Feature Learning**: Basit kenarlardan karmaşık şekillere
4. **Spatial Locality**: Komşu pikseller arasındaki ilişkileri anlama

## Kullanım Alanları

- Fotoğraftan yüz tanıma
- MR/BT gibi tıbbi görüntülerde hastalık tespiti
- Trafikte otonom araçlar
- Instagram filtreleri (Etkileşimli olanlar)

## CNN Katmanları

### 1. Convolution Katmanı (Evrişim Katmanı)
- **Amaç**: Görüntüden özellik çıkarma
- **Çalışma Prensibi**: Filtreler (kernels) görüntü üzerinde kaydırılarak özellik haritaları oluşturulur
- **Matematik**: Evrişim işlemi ile yerel özellikler tespit edilir
- **Parametreler**: Filter boyutu (3x3, 5x5), stride, padding

### 2. Aktivasyon Fonksiyonu (ReLU)
- **Amaç**: Doğrusal olmayan özellikler ekleme
- **ReLU**: f(x) = max(0, x) - Negatif değerleri sıfırlama
- **Fayda**: Gradyan kaybını önleme, hesaplama hızını artırma

### 3. Pooling Katmanı (Havuzlama)
- **Amaç**: Boyut azaltma ve genelleştirme
- **Max Pooling**: Bölgedeki maksimum değeri seçme
- **Average Pooling**: Bölgenin ortalamasını alma
- **Fayda**: Hesaplama yükünü azaltma, aşırı öğrenmeyi önleme

### 4. Normalization Katmanları
- **Batch Normalization**: Mini-batch düzeyinde normalleştirme
- **Layer Normalization**: Katman düzeyinde normalleştirme
- **Fayda**: Eğitim hızını artırma, gradyan kararlılığı

### 5. Flatten Katmanı (Düzleştirme)
- **Amaç**: Çok boyutlu özellik haritalarını tek boyutlu vektöre dönüştürme
- **Matematik**: 3D tensor → 1D vector
- **Gereklilik**: Tam bağlı katmanlara geçiş için zorunlu

### 6. Fully Connected Layer (Tam Bağlı Katman)
- **Amaç**: Özellik kombinasyonları ile sınıflandırma
- **Yapı**: Her nöron bir önceki katmanın tüm nöronlarına bağlı
- **Çıktı**: Sınıf olasılıkları (Softmax ile)

### 7. Dropout Katmanı
- **Amaç**: Aşırı öğrenmeyi (overfitting) önleme
- **Çalışma**: Rastgele nöronları devre dışı bırakma
- **Oran**: Genellikle 0.2-0.5 arası

## CNN Mimarileri

### 1. LeNet-5 (1998)
- **Özellik**: İlk başarılı CNN mimarisi
- **Kullanım**: El yazısı rakam tanıma
- **Katmanlar**: 2 Conv + 2 Pooling + 3 FC

### 2. AlexNet (2012)
- **Breakthrough**: ImageNet yarışmasını kazanan ilk CNN
- **Yenilikler**: ReLU, Dropout, Data Augmentation
- **Derinlik**: 8 katman

### 3. VGGNet (2014)
- **Özellik**: Küçük (3x3) filtreler kullanımı
- **Derinlik**: 16-19 katman
- **Avantaj**: Daha derin ağlar ile daha iyi performans

### 4. ResNet (2015)
- **Yenilik**: Residual connections (skip connections)
- **Çözüm**: Vanishing gradient problemini çözme
- **Derinlik**: 50, 101, 152 katman

### 5. EfficientNet (2019)
- **Yaklaşım**: Compound scaling (width, depth, resolution)
- **Avantaj**: Daha az parametre ile daha iyi performans

## Veri Ön İşleme

### 1. Normalizasyon
```python
# Piksel değerlerini 0-1 arası normalize etme
normalized_image = image / 255.0

# StandardScaler ile normalizasyon
mean = [0.485, 0.456, 0.406]  # ImageNet ortalaması
std = [0.229, 0.224, 0.225]   # ImageNet standart sapması
```

### 2. Boyut Ayarlama
- **Input Format**: (batch_size, height, width, channels) - TensorFlow
- **Input Format**: (batch_size, channels, height, width) - PyTorch
- **Tipik Boyutlar**: 224x224, 299x299, 512x512

### 3. Data Augmentation
- **Rotation**: Görüntüyü döndürme
- **Flipping**: Yatay/dikey çevirme
- **Scaling**: Büyütme/küçültme
- **Cropping**: Rastgele kırpma
- **Color Jittering**: Renk değişimleri

## Model Optimizasyonu

### 1. Loss Functions
- **Categorical Crossentropy**: Çok sınıflı sınıflandırma
- **Binary Crossentropy**: İkili sınıflandırma
- **Sparse Categorical Crossentropy**: Tek hot encoding olmadan

### 2. Optimizers
- **Adam**: Adaptif öğrenme oranı
- **SGD**: Stochastic Gradient Descent
- **RMSprop**: Gradyan karelerinin hareketli ortalaması

### 3. Learning Rate Scheduling
- **Step Decay**: Belirli epoch'larda azaltma
- **Exponential Decay**: Üstel azalma
- **Cosine Annealing**: Kosinüs fonksiyonu ile azalma

## Transfer Learning

### Kavram
Önceden eğitilmiş modellerin farklı görevlerde kullanılması

### Yaklaşımlar
1. **Feature Extraction**: Sadece son katmanları eğitme
2. **Fine-tuning**: Tüm ağı düşük öğrenme oranı ile eğitme
3. **Progressive Fine-tuning**: Katman katman açarak eğitme

### Popüler Pretrained Modeller
- **ImageNet**: 1000 sınıf, 14M görüntü
- **COCO**: Nesne tespiti için
- **Places365**: Sahne tanıma için

---

## Özet

CNN'ler görsel veri işlemede devrim yaratmış derin öğrenme mimarisidir:

1. **Parça parça özellik bul** (Convolution)
2. **Gereksizleri at, önemli bilgiyi topla** (Pooling)
3. **Her şeyi tek boyutlu listele** (Flatten)
4. **Sonuca var** (Fully Connected + Softmax)

CNN'lerin başarısı, görüntülerdeki uzamsal ilişkileri koruyması ve hiyerarşik özellik öğrenmesinden kaynaklanmaktadır.


