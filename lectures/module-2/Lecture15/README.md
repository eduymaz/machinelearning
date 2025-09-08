# Image Segmentation (Görüntü Segmentasyonu)

## Görüntü Segmentasyonu Nedir?
Görüntü segmentasyonu, bir görüntüyü anlamsal olarak benzer bölgelere ayırma işlemidir. Bu işlem, görüntüdeki nesneleri birbirinden ayırmak ve her nesneyi ayrı bir bölge olarak tanımlamak için kullanılır.

## Segmentasyon Türleri

### 1. Semantic Segmentation (Anlamsal Segmentasyon)
Görüntüdeki her pikseli belirli bir sınıfa atama işlemi. Aynı sınıftaki nesneler aynı etiketle işaretlenir.

### 2. Instance Segmentation (Örnek Segmentasyonu)
Aynı sınıftaki farklı nesneleri birbirinden ayırt ederek her nesneyi ayrı bir örnek olarak segmentleme işlemi.

### 3. Interactive Segmentation (Etkileşimli Segmentasyon)
Kullanıcının müdahalesi ile gerçekleştirilen segmentasyon. GrabCut bu kategoride yer alır.

## Segmentasyon Algoritmaları

### 1. GrabCut Algoritması
- **Tanım**: Kullanıcının minimal müdahalesi ile ön plan ve arka plan segmentasyonu yapan algoritma
- **Çalışma Prensibi**: Gaussian Mixture Model (GMM) kullanarak piksel dağılımlarını modellemektedir
- **Kullanım Alanları**: Nesne çıkarma, arka plan değiştirme, fotoğraf düzenleme
- **Avantajları**: Az kullanıcı müdahalesi, yüksek kaliteli sonuçlar
- **Dezavantajları**: Karmaşık arka planlar için zorlanabilir

### 2. Watershed Algoritması
- **Tanım**: Görüntüyü topografik bir harita gibi düşünerek su havzalarına ayırma algoritması
- **Kullanım**: Birbirine yakın nesneleri ayırmak için kullanılır

### 3. K-Means Clustering
- **Tanım**: Pikselleri renk benzerliklerine göre K adet kümeye ayırma algoritması
- **Kullanım**: Renk tabanlı segmentasyon için kullanılır

### 4. Region Growing (Bölge Büyütme)
- **Tanım**: Tohum noktalarından başlayarak benzer pikselleri birleştirerek bölgeleri büyütme algoritması
- **Kullanım**: Homojen bölgelerin segmentasyonu için kullanılır

### 5. Thresholding Based Segmentation
- **Tanım**: Eşik değerleri kullanarak görüntüyü ikili bölgelere ayırma
- **Çeşitleri**: Otsu, Adaptive Thresholding

## Görüntü Ön İşleme Adımları

### 1. Gürültü Azaltma (Noise Reduction)
- **Amaç**: Segmentasyon kalitesini artırmak için görüntüdeki gürültüyü temizleme
- **Yöntemler**: Gaussian blur, Median filter, Bilateral filter
- **Önem**: Gürültülü görüntüler yanlış segmentasyon sonuçlarına yol açabilir

### 2. Kontrast İyileştirme (Contrast Enhancement)
- **Amaç**: Nesneler arasındaki farkları belirginleştirme
- **Yöntemler**: Histogram eşitleme, CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Fayda**: Segmentasyon algoritmalarının daha iyi çalışmasını sağlar

### 3. Renk Uzayı Dönüşümü (Color Space Conversion)
- **Amaç**: Segmentasyon için en uygun renk uzayını seçme
- **Yaygın Dönüşümler**: RGB → HSV, RGB → LAB, RGB → Grayscale
- **HSV Avantajı**: Aydınlatma değişikliklerine daha az duyarlı

### 4. Morfolojik İşlemler (Morphological Operations)
- **Erosion**: Nesneleri küçültme, gürültü temizleme
- **Dilation**: Nesneleri büyütme, boşlukları doldurma
- **Opening**: Erosion + Dilation (gürültü temizleme)
- **Closing**: Dilation + Erosion (boşluk doldurma)

### 5. Kenar Yumuşatma (Edge Smoothing)
- **Amaç**: Keskin kenarları yumuşatarak daha doğal segmentasyon
- **Yöntemler**: Gaussian smoothing, Bilateral filtering
- **Dikkat**: Aşırı yumuşatma detay kaybına yol açabilir

## Segmentasyon Kalitesi Değerlendirme

### 1. IoU (Intersection over Union)
- **Tanım**: Kesişim alanının birleşim alanına oranı
- **Formül**: IoU = (A ∩ B) / (A ∪ B)
- **Değer Aralığı**: 0-1 arası (1 mükemmel)

### 2. Dice Coefficient
- **Tanım**: İki bölgenin benzerlik ölçüsü
- **Formül**: Dice = 2 * |A ∩ B| / (|A| + |B|)
- **Kullanım**: Tıbbi görüntü segmentasyonunda yaygın

### 3. Pixel Accuracy
- **Tanım**: Doğru sınıflandırılan piksel oranı
- **Basitlik**: Hesaplaması kolay ama sınıf dengesizliğinde yanıltıcı

## Uygulama Alanları

### Tıbbi Görüntü Analizi
- Organ segmentasyonu (beyin, kalp, akciğer)
- Tümör tespiti ve ölçümü
- Kan damarı analizi
- Kemik yapısı inceleme

### Otonom Araçlar
- Yol tespiti ve takibi
- Araç ve yaya tespiti
- Trafik işareti tanıma
- Engel algılama

### Endüstriyel Uygulamalar
- Kalite kontrol ve hata tespiti
- Robotik görme sistemleri
- Malzeme sınıflandırma
- Montaj doğrulama

### Tarım ve Çevre
- Bitki hastalık tespiti
- Mahsul verimlilik analizi
- Toprak analizi
- Orman izleme