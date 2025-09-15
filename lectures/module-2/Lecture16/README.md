# Lecture 16: OCR (Optical Character Recognition) Teknolojileri

## İçindekiler
1. [OCR Nedir?](#ocr-nedir)
2. [Açık Kaynaklı OCR Kütüphaneleri](#açık-kaynaklı-ocr-kütüphaneleri)
3. [Bulut Tabanlı OCR Servisleri](#bulut-tabanlı-ocr-servisleri)
4. [Karşılaştırma ve Seçim Kriterleri](#karşılaştırma-ve-seçim-kriterleri)
5. [Pratik Uygulamalar](#pratik-uygulamalar)

## OCR Nedir?

Optical Character Recognition (Optik Karakter Tanıma), görüntülerdeki metinleri dijital metin formatına dönüştüren bir teknolojidir. OCR, fotoğraflar, taranmış belgeler, PDF'ler ve diğer görsel materyallerdeki yazılı içeriği düzenlenebilir metne çevirir.

### OCR Kullanım Alanları:
- **Belge Dijitalleştirme**: Kağıt belgelerin dijital ortama aktarılması
- **Fatura İşleme**: Otomatik fatura okuma ve veri çıkarma
- **Kimlik Doğrulama**: Pasaport, ehliyet gibi kimlik belgelerinden bilgi çıkarma
- **Plaka Tanıma**: Araç plaklarının otomatik okunması
- **Metin Çevirisi**: Görüntülerdeki metinlerin çevrilmesi
- **Erişilebilirlik**: Görme engelliler için metin okuma

## Açık Kaynaklı OCR Kütüphaneleri

### 1. Tesseract OCR
**En popüler ve güçlü açık kaynaklı OCR motoru**

#### Özellikler:
- Google tarafından geliştirilmiş ve sürdürülmekte
- 100+ dil desteği
- LSTM tabanlı neural network motoru
- Yüksek doğruluk oranı
- Çok çeşitli görüntü formatı desteği

#### Kurulum:
```bash
# macOS
brew install tesseract

# Python paketi
pip install pytesseract pillow
```

#### Avantajlar:
- Ücretsiz ve açık kaynak
- Geniş dil desteği
- Aktif geliştirme ve topluluk
- Özelleştirilebilir

#### Dezavantajlar:
- Karmaşık düzen için ek işlem gereksinimi
- Elle yazılmış metinlerde düşük performans

**Dokümantasyon**: https://tesseract-ocr.github.io/
**İndirme**: https://sourceforge.net/projects/tesseract-ocr-alt/files/

### 2. EasyOCR
**Kullanım kolaylığına odaklanan modern OCR kütüphanesi**

#### Özellikler:
- 80+ dil desteği
- GPU desteği (CUDA)
- Deep learning tabanlı
- Kolay kurulum ve kullanım
- Çoklu dil algılama

#### Kurulum:
```bash
pip install easyocr
```

#### Avantajlar:
- Çok kolay kullanım
- GPU desteği ile hızlı işlem
- İyi doğruluk oranı
- Çoklu dil desteği

#### Dezavantajlar:
- Daha büyük model boyutu
- GPU olmadan yavaş

### 3. PaddleOCR
**Baidu tarafından geliştirilen yüksek performanslı OCR sistemi**

#### Özellikler:
- Çin, İngilizce ve diğer dillerde yüksek performans
- Metin algılama + tanıma
- Mobil deployment desteği
- Hafif modeller

#### Kurulum:
```bash
pip install paddlepaddle paddleocr
```

#### Avantajlar:
- Çok hızlı inference
- Mobil cihazlarda çalışabilir
- İyi Çince desteği
- Kompakt modeller

#### Dezavantajlar:
- Daha az dil desteği
- Dokümantasyon çoğunlukla Çince

## Bulut Tabanlı OCR Servisleri

### 1. AWS Textract
**Amazon'un gelişmiş belge analiz servisi**

#### Özellikler:
- Metin, tablo ve form algılama
- Anahtar-değer çifti çıkarma
- Çok sayfalı belge desteği
- Yüksek doğruluk oranı

#### Fiyatlandırma:
- İlk 1000 sayfa/ay ücretsiz
- Sonrası sayfa başına ücretlendirme

### 2. Google Cloud Vision API
**Google'ın güçlü görüntü analiz servisi**

#### Özellikler:
- OCR ve belge metni algılama
- Elle yazılmış metin tanıma
- 50+ dil desteği
- JSON çıktı formatı

#### Fiyatlandırma:
- İlk 1000 istek/ay ücretsiz
- Sonrası istek başına ücret

### 3. Azure Computer Vision
**Microsoft'un bulut tabanlı görüntü analiz hizmeti**

#### Özellikler:
- Read API ile gelişmiş OCR
- Elle yazılmış ve basılı metin
- Çok sayfalı PDF desteği
- Real-time işlem

#### Fiyatlandırma:
- Ücretsiz tier mevcut
- İstek sayısına göre ücretlendirme

## Karşılaştırma ve Seçim Kriterleri

| Kriter | Tesseract | EasyOCR | PaddleOCR | AWS Textract | Google Vision |
|--------|-----------|---------|-----------|--------------|---------------|
| **Maliyet** | Ücretsiz | Ücretsiz | Ücretsiz | Ücretli | Ücretli |
| **Dil Desteği** | 100+ | 80+ | 15+ | 15+ | 50+ |
| **Doğruluk** | Yüksek | Yüksek | Çok Yüksek | Çok Yüksek | Çok Yüksek |
| **Hız** | Orta | Hızlı (GPU) | Çok Hızlı | Hızlı | Hızlı |
| **Kurulum** | Orta | Kolay | Kolay | API | API |
| **Offline Çalışma** | ✓ | ✓ | ✓ | ✗ | ✗ |
| **Form/Tablo** | ✗ | ✗ | ✗ | ✓ | ✓ |

### Seçim Rehberi:

**Tesseract kullanın:**
- Maliyet önemli
- Çok dilli proje
- Offline çalışma gerekli
- Özelleştirme ihtiyacı var

**EasyOCR kullanın:**
- Kolay kurulum istiyorsanız
- GPU desteği var
- Hızlı sonuç gerekli

**PaddleOCR kullanın:**
- En hızlı çözüm istiyorsanız
- Çince metin var
- Mobil deployment planlıyorsanız

**Bulut servisleri kullanın:**
- En yüksek doğruluk gerekli
- Form/tablo işleme var
- Altyapı yönetimi istemiyorsanız
- Elle yazılmış metin çok

## Pratik Uygulamalar

### Görüntü Ön İşleme Tips:
1. **Kontrast artırma** - Metin ve arka plan arasındaki farkı belirginleştirin
2. **Gürültü azaltma** - Medyan filtresi kullanın
3. **Döndürme düzeltme** - Metni yatay hale getirin
4. **Çözünürlük artırma** - Küçük metinler için upscaling yapın
5. **İkili görüntü** - Siyah-beyaz dönüşüm yapın

### En İyi Pratikler:
- Yüksek kaliteli görüntü kullanın (300+ DPI)
- Tek tip fontlar tercih edin
- İyi aydınlatma sağlayın
- Çarpık açıları düzeltin
- Gereksiz gürültüyü temizleyin

### Performans İyileştirme:
- Görüntü boyutunu optimize edin
- ROI (Region of Interest) kullanın
- Batch processing yapın
- Cache mekanizması ekleyin
- Paralel işlem kullanın

---

## Örnek Kodlar

Detaylı Python örnekleri için `main.py` dosyasını inceleyin.

Denoise Document: https://docs.opencv.org/3.4/d1/d79/group__photo__denoise.html 


