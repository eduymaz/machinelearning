"""
Lecture 16: OCR (Optical Character Recognition) Örnekleri
=========================================================

Bu dosya, farklı OCR kütüphaneleri ve bulut servislerinin 
kullanımına dair pratik örnekler içermektedir.

Gereksinimler:
- pip install pytesseract pillow opencv-python easyocr paddleocr
- pip install boto3 google-cloud-vision azure-cognitiveservices-vision-computervision
- tesseract kurulumu (brew install tesseract - macOS)

"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import easyocr
import os
import base64
from typing import List, Dict, Any
import json

# ============================================================================
# YARDIMCI FONKSİYONLAR - Görüntü Ön İşleme
# ============================================================================

def preprocess_image(image_path: str, method='auto') -> np.ndarray:
    """
    Görüntüyü OCR için optimize eder
    
    Args:
        image_path: Görüntü dosya yolu
        method: 'auto', 'grayscale', 'binary', 'denoise'
    
    Returns:
        İşlenmiş görüntü array'i
    """
    # Görüntüyü yükle
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")
    
    if method == 'auto' or method == 'grayscale':
        # Gri tonlamaya çevir
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == 'auto' or method == 'binary':
        # İkili (binary) görüntüye çevir
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if method == 'denoise':
        # Gürültü azaltma
        img = cv2.medianBlur(img, 3)
    
    return img

def enhance_image_quality(image_path: str, scale_factor: int = 2) -> Image.Image:
    """
    Görüntü kalitesini artırır
    
    Args:
        image_path: Görüntü dosya yolu
        scale_factor: Büyütme faktörü
    
    Returns:
        Geliştirilmiş PIL Image
    """
    img = Image.open(image_path)
    
    # Boyutu artır
    width, height = img.size
    img = img.resize((width * scale_factor, height * scale_factor), Image.LANCZOS)
    
    # Kontrastı artır
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # Keskinliği artır
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    
    return img

def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Eğik görüntüyü düzeltir
    
    Args:
        image: Giriş görüntüsü
    
    Returns:
        Düzeltilmiş görüntü
    """
    # Kenarları bul
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Hough Line Transform ile çizgileri bul
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        # En yaygın açıyı hesapla
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)
        
        # Median açıyı al
        median_angle = np.median(angles)
        
        # Görüntüyü döndür
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    return image

# ============================================================================
# TESSERACT OCR İMPLEMENTASYONU
# ============================================================================

class TesseractOCR:
    """Tesseract OCR wrapper class"""
    
    def __init__(self, lang='eng+tur'):
        """
        Args:
            lang: Dil kodu (ör: 'eng', 'tur', 'eng+tur')
        """
        self.lang = lang
        # Tesseract kurulu mu kontrol et
        try:
            pytesseract.get_tesseract_version()
        except:
            raise RuntimeError("Tesseract kurulu değil. Kurulum: brew install tesseract")
    
    def extract_text(self, image_path: str, preprocess: bool = True) -> str:
        """
        Görüntüden metin çıkarır
        
        Args:
            image_path: Görüntü dosya yolu
            preprocess: Ön işlem yapılacak mı
        
        Returns:
            Çıkarılan metin
        """
        try:
            if preprocess:
                img = preprocess_image(image_path)
            else:
                img = cv2.imread(image_path)
            
            # Tesseract konfigürasyonu
            config = f'--oem 3 --psm 6 -l {self.lang}'
            
            # Metin çıkar
            text = pytesseract.image_to_string(img, config=config)
            return text.strip()
        
        except Exception as e:
            return f"Hata: {str(e)}"
    
    def extract_with_boxes(self, image_path: str) -> List[Dict]:
        """
        Metin ve bounding box'larını çıkarır
        
        Args:
            image_path: Görüntü dosya yolu
        
        Returns:
            [{text, x, y, w, h, confidence}, ...]
        """
        try:
            img = preprocess_image(image_path)
            
            # Detaylı çıktı al
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 30:  # Güven eşiği
                    text = data['text'][i].strip()
                    if text:
                        results.append({
                            'text': text,
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'w': data['width'][i],
                            'h': data['height'][i],
                            'confidence': data['conf'][i]
                        })
            
            return results
        
        except Exception as e:
            print(f"Hata: {str(e)}")
            return []

# ============================================================================
# EASYOCR İMPLEMENTASYONU
# ============================================================================

class EasyOCRProcessor:
    """EasyOCR wrapper class"""
    
    def __init__(self, languages=['en', 'tr'], gpu=False):
        """
        Args:
            languages: Desteklenen diller
            gpu: GPU kullanımı
        """
        try:
            import easyocr
            self.reader = easyocr.Reader(languages, gpu=gpu)
        except ImportError:
            raise ImportError("EasyOCR kurulu değil. Kurulum: pip install easyocr")
    
    def extract_text(self, image_path: str, detail=0) -> str:
        """
        Görüntüden metin çıkarır
        
        Args:
            image_path: Görüntü dosya yolu
            detail: 0=sadece metin, 1=koordinatlar da
        
        Returns:
            Çıkarılan metin
        """
        try:
            results = self.reader.readtext(image_path, detail=detail)
            
            if detail == 0:
                return ' '.join(results)
            else:
                texts = [result[1] for result in results if result[2] > 0.5]
                return ' '.join(texts)
        
        except Exception as e:
            return f"Hata: {str(e)}"
    
    def extract_with_details(self, image_path: str) -> List[Dict]:
        """
        Detaylı metin çıkarımı
        
        Args:
            image_path: Görüntü dosya yolu
        
        Returns:
            [{text, bbox, confidence}, ...]
        """
        try:
            results = self.reader.readtext(image_path)
            
            processed_results = []
            for bbox, text, confidence in results:
                if confidence > 0.5:  # Güven eşiği
                    processed_results.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence
                    })
            
            return processed_results
        
        except Exception as e:
            print(f"Hata: {str(e)}")
            return []

# ============================================================================
# PADDLEOCR İMPLEMENTASYONU
# ============================================================================

class PaddleOCRProcessor:
    """PaddleOCR wrapper class"""
    
    def __init__(self, lang='en', use_gpu=False):
        """
        Args:
            lang: Dil kodu ('en', 'ch', 'fr', 'german', vb.)
            use_gpu: GPU kullanımı
        """
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)
        except ImportError:
            raise ImportError("PaddleOCR kurulu değil. Kurulum: pip install paddlepaddle paddleocr")
    
    def extract_text(self, image_path: str) -> str:
        """
        Görüntüden metin çıkarır
        
        Args:
            image_path: Görüntü dosya yolu
        
        Returns:
            Çıkarılan metin
        """
        try:
            results = self.ocr.ocr(image_path, cls=True)
            
            texts = []
            if results[0]:
                for line in results[0]:
                    if line[1][1] > 0.5:  # Güven eşiği
                        texts.append(line[1][0])
            
            return ' '.join(texts)
        
        except Exception as e:
            return f"Hata: {str(e)}"

# ============================================================================
# BULUT SERVİSLERİ
# ============================================================================

class AWSTextractOCR:
    """AWS Textract wrapper class"""
    
    def __init__(self, region_name='us-east-1'):
        """
        AWS Textract kullanımı için AWS credentials gereklidir
        """
        try:
            import boto3
            self.textract = boto3.client('textract', region_name=region_name)
        except ImportError:
            raise ImportError("boto3 kurulu değil. Kurulum: pip install boto3")
    
    def extract_text_from_local_image(self, image_path: str) -> str:
        """
        Yerel görüntüden metin çıkarır
        
        Args:
            image_path: Görüntü dosya yolu
        
        Returns:
            Çıkarılan metin
        """
        try:
            with open(image_path, 'rb') as image:
                img_bytes = image.read()
            
            response = self.textract.detect_document_text(
                Document={'Bytes': img_bytes}
            )
            
            # Metni birleştir
            text_blocks = []
            for block in response['Blocks']:
                if block['BlockType'] == 'LINE':
                    text_blocks.append(block['Text'])
            
            return '\n'.join(text_blocks)
        
        except Exception as e:
            return f"AWS Textract Hatası: {str(e)}"

class GoogleVisionOCR:
    """Google Cloud Vision OCR wrapper class"""
    
    def __init__(self, credentials_path: str = None):
        """
        Google Cloud Vision kullanımı için service account key gereklidir
        """
        try:
            from google.cloud import vision
            if credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            self.client = vision.ImageAnnotatorClient()
        except ImportError:
            raise ImportError("google-cloud-vision kurulu değil. Kurulum: pip install google-cloud-vision")
    
    def extract_text(self, image_path: str) -> str:
        """
        Görüntüden metin çıkarır
        
        Args:
            image_path: Görüntü dosya yolu
        
        Returns:
            Çıkarılan metin
        """
        try:
            from google.cloud import vision
            
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = self.client.text_detection(image=image)
            
            if response.text_annotations:
                return response.text_annotations[0].description
            else:
                return "Metin bulunamadı"
        
        except Exception as e:
            return f"Google Vision Hatası: {str(e)}"

# ============================================================================
# KARŞILAŞTIRMA VE TEST FONKSİYONLARI
# ============================================================================

def compare_ocr_methods(image_path: str) -> Dict[str, Any]:
    """
    Farklı OCR yöntemlerini karşılaştırır
    
    Args:
        image_path: Test görüntüsü yolu
    
    Returns:
        Karşılaştırma sonuçları
    """
    results = {}
    
    print(f"🔍 {image_path} görüntüsü test ediliyor...")
    
    # Tesseract
    try:
        tesseract = TesseractOCR()
        import time
        start_time = time.time()
        tesseract_text = tesseract.extract_text(image_path)
        tesseract_time = time.time() - start_time
        
        results['tesseract'] = {
            'text': tesseract_text,
            'time': tesseract_time,
            'char_count': len(tesseract_text.strip())
        }
    except Exception as e:
        results['tesseract'] = {'error': str(e)}
    
    # EasyOCR
    try:
        easyocr_proc = EasyOCRProcessor()
        start_time = time.time()
        easyocr_text = easyocr_proc.extract_text(image_path)
        easyocr_time = time.time() - start_time
        
        results['easyocr'] = {
            'text': easyocr_text,
            'time': easyocr_time,
            'char_count': len(easyocr_text.strip())
        }
    except Exception as e:
        results['easyocr'] = {'error': str(e)}
    
    # PaddleOCR
    try:
        paddle = PaddleOCRProcessor()
        start_time = time.time()
        paddle_text = paddle.extract_text(image_path)
        paddle_time = time.time() - start_time
        
        results['paddleocr'] = {
            'text': paddle_text,
            'time': paddle_time,
            'char_count': len(paddle_text.strip())
        }
    except Exception as e:
        results['paddleocr'] = {'error': str(e)}
    
    return results

def print_comparison_results(results: Dict[str, Any]) -> None:
    """Karşılaştırma sonuçlarını yazdırır"""
    
    print("\n" + "="*80)
    print("🎯 OCR KARŞILAŞTIRMA SONUÇLARI")
    print("="*80)
    
    for method, result in results.items():
        print(f"\n📋 {method.upper()}:")
        print("-" * 50)
        
        if 'error' in result:
            print(f"❌ Hata: {result['error']}")
        else:
            print(f"⏱️  Süre: {result['time']:.2f} saniye")
            print(f"📝 Karakter sayısı: {result['char_count']}")
            print(f"📄 Çıkarılan metin:")
            print(f"   {result['text'][:200]}...")  # İlk 200 karakter

def create_sample_image_for_testing():
    """Test için örnek görüntü oluşturur"""
    
    from PIL import Image, ImageDraw, ImageFont
    
    # Beyaz arka plan
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Metin ekle
    try:
        # Sistem fontunu kullan
        font = ImageFont.load_default()
    except:
        font = None
    
    test_text = """
    OCR Test Görüntüsü
    
    Bu bir test metnidir.
    Türkçe karakterler: ğ, ü, ş, ı, ö, ç
    Sayılar: 0123456789
    
    Telefon: +90 555 123 45 67
    Email: test@example.com
    Tarih: 15.09.2025
    """
    
    draw.multiline_text((50, 50), test_text, fill='black', font=font, spacing=10)
    
    # Kaydet
    sample_path = 'sample_ocr_image.png'
    img.save(sample_path)
    print(f"✅ Örnek görüntü oluşturuldu: {sample_path}")
    return sample_path

# ============================================================================
# MAIN FUNCTION - DEMO
# ============================================================================

def main():
    """Ana demo fonksiyonu"""
    
    print("🚀 OCR Teknolojileri Demo'su Başlıyor...")
    print("="*60)
    
    # Örnek görüntü oluştur
    sample_image = create_sample_image_for_testing()
    
    print("\n1️⃣  Temel Tesseract OCR Testi:")
    print("-" * 40)
    tesseract = TesseractOCR()
    text = tesseract.extract_text(sample_image)
    print(f"Çıkarılan metin: {text}")
    
    print("\n2️⃣  Görüntü Ön İşleme Örnekleri:")
    print("-" * 40)
    
    # Farklı ön işleme yöntemleri
    methods = ['grayscale', 'binary', 'denoise']
    for method in methods:
        try:
            processed = preprocess_image(sample_image, method)
            processed_path = f'processed_{method}.png'
            cv2.imwrite(processed_path, processed)
            print(f"✅ {method} işlemi tamamlandı: {processed_path}")
        except Exception as e:
            print(f"❌ {method} işleminde hata: {e}")
    
    print("\n3️⃣  OCR Yöntemleri Karşılaştırması:")
    print("-" * 40)
    
    # Tüm yöntemleri karşılaştır
    results = compare_ocr_methods(sample_image)
    print_comparison_results(results)
    
    print("\n4️⃣  Detaylı Metin Çıkarımı (Bounding Boxes):")
    print("-" * 40)
    
    try:
        boxes = tesseract.extract_with_boxes(sample_image)
        print(f"Toplam {len(boxes)} metin bloğu bulundu:")
        for i, box in enumerate(boxes[:3]):  # İlk 3'ü göster
            print(f"  {i+1}. '{box['text']}' - Güven: {box['confidence']}%")
    except Exception as e:
        print(f"❌ Detaylı çıkarımda hata: {e}")
    
    print("\n5️⃣  EasyOCR Detaylı Test:")
    print("-" * 40)
    
    try:
        easyocr_proc = EasyOCRProcessor()
        details = easyocr_proc.extract_with_details(sample_image)
        print(f"EasyOCR ile {len(details)} metin bloğu bulundu:")
        for detail in details[:3]:
            print(f"  '{detail['text']}' - Güven: {detail['confidence']:.2f}")
    except Exception as e:
        print(f"❌ EasyOCR testinde hata: {e}")
    
    # Temizlik
    cleanup_files = [sample_image, 'processed_grayscale.png', 'processed_binary.png', 'processed_denoise.png']
    for file in cleanup_files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass
    
    print("\n🎉 Demo tamamlandı!")
    print("\nİpuçları:")
    print("- Gerçek projelerinizde yüksek kaliteli görüntüler kullanın")
    print("- Ön işleme adımlarını görüntünüze göre optimize edin")
    print("- Bulut servisleri için API anahtarları ayarlayın")
    print("- Performans için GPU desteğini aktifleştirin")

# ============================================================================
# ÖRNEK KULLANIM FUNKSİYONLARI
# ============================================================================

def example_batch_processing(image_folder: str) -> None:
    """
    Toplu görüntü işleme örneği
    
    Args:
        image_folder: Görüntülerin bulunduğu klasör
    """
    print(f"📁 {image_folder} klasöründeki görüntüler işleniyor...")
    
    tesseract = TesseractOCR()
    
    # Desteklenen formatlar
    supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    
    results = []
    for filename in os.listdir(image_folder):
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            image_path = os.path.join(image_folder, filename)
            text = tesseract.extract_text(image_path)
            results.append({
                'filename': filename,
                'text': text,
                'char_count': len(text.strip())
            })
            print(f"✅ {filename} işlendi - {len(text.strip())} karakter")
    
    # Sonuçları JSON'a kaydet
    with open('batch_ocr_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"📊 {len(results)} dosya işlendi. Sonuçlar: batch_ocr_results.json")

def example_real_time_ocr():
    """
    Gerçek zamanlı OCR örneği (webcam)
    """
    print("📹 Gerçek zamanlı OCR başlatılıyor... (ESC ile çıkış)")
    
    tesseract = TesseractOCR()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Her 30 frame'de bir OCR yap (performans için)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 30 == 0:
            # Frame'i geçici dosya olarak kaydet
            cv2.imwrite('temp_frame.jpg', frame)
            
            # OCR yap
            text = tesseract.extract_text('temp_frame.jpg', preprocess=True)
            
            # Metin varsa göster
            if text.strip():
                print(f"📝 Algılanan: {text.strip()}")
        
        # Frame'i göster
        cv2.imshow('Real-time OCR (ESC to quit)', frame)
        
        # ESC tuşuyla çık
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Temizlik
    if os.path.exists('temp_frame.jpg'):
        os.remove('temp_frame.jpg')
    
    print("📹 Gerçek zamanlı OCR sonlandırıldı.")

if __name__ == "__main__":
    # Ana demo'yu çalıştır
    main()
    
    # Kullanıcıdan ilave test isteyip istemediğini sor
    print("\n" + "="*60)
    print("İlave testler:")
    print("1. Toplu işleme testi için bir klasör yolu girin")
    print("2. Gerçek zamanlı OCR için 'webcam' yazın")
    print("3. Çıkış için Enter'a basın")
    
    user_input = input("\nSeçiminiz: ").strip()
    
    if user_input.lower() == 'webcam':
        example_real_time_ocr()
    elif user_input and os.path.isdir(user_input):
        example_batch_processing(user_input)
    else:
        print("👋 Görüşmek üzere!")