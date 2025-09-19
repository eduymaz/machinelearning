"""
YOLO8 Turkish License Plate Detection - Training Script
Bu script YOLO8 modelini Türkçe plaka tespiti için eğitir.
"""

import os
from ultralytics import YOLO
import torch

def check_environment():

    print("=== Ortam Kontrolü ===")
    print(f"PyTorch versiyonu: {torch.__version__}")
    print(f"CUDA kullanılabilir mi: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA cihaz sayısı: {torch.cuda.device_count()}")
        print(f"Mevcut CUDA cihazı: {torch.cuda.get_device_name()}")
    print("=" * 30)

def prepare_directories():
    
    directories = ['runs', 'datasets', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Klasör hazırlandı: {directory}")

def train_model():
   
    print("=== Model Eğitimi Başlıyor ===")
    
    # YOLOv8n (nano) modelini yükle - daha hızlı eğitim için
    model = YOLO("yolov8n.pt")
    
    # Model mimarisini göster
    print(f"Model mimarisi: {model.model}")
    
    # Eğitim parametreleri
    results = model.train(
        data="data.yaml",           # Veri konfigürasyon dosyası
        epochs=100,                 # Eğitim epoch sayısı
        imgsz=640,                  # Görüntü boyutu
        batch=16,                   # Batch boyutu
        name="turkish_license_plate", # Experiment adı
        save=True,                  # Modeli kaydet
        save_period=10,             # Her 10 epoch'ta kaydet
        cache=True,                 # Veriyi cache'le (hızlandırır)
        device='auto',              # GPU varsa kullan
        workers=4,                  # Veri yükleme işçi sayısı
        project='runs/detect',      # Sonuçların kaydedileceği klasör
        exist_ok=True,              # Varolan klasörü kullan
        pretrained=True,            # Ön eğitilmiş ağırlıkları kullan
        optimizer='SGD',            # Optimizer: SGD, Adam, AdamW
        verbose=True,               # Detaylı çıktı
        seed=42,                    # Reproducible sonuçlar için
        deterministic=True,         # Deterministic training
        single_cls=False,           # Çok sınıflı classification
        rect=False,                 # Rectangular training
        cos_lr=False,               # Cosine learning rate scheduler
        close_mosaic=10,            # Mosaic augmentation'ı son 10 epoch'ta kapat
        resume=False,               # Eğitimi devam ettir
        amp=True,                   # Automatic Mixed Precision (hızlandırır)
        fraction=1.0,               # Dataset'in kullanılacak kısmı
        profile=False,              # Profiling yap
        # Augmentation parametreleri
        hsv_h=0.015,               # Hue augmentation
        hsv_s=0.7,                 # Saturation augmentation  
        hsv_v=0.4,                 # Value augmentation
        degrees=0.0,               # Rotation augmentation
        translate=0.1,             # Translation augmentation
        scale=0.5,                 # Scale augmentation
        shear=0.0,                 # Shear augmentation
        perspective=0.0,           # Perspective augmentation
        flipud=0.0,                # Vertical flip probability
        fliplr=0.5,                # Horizontal flip probability
        mosaic=1.0,                # Mosaic augmentation probability
        mixup=0.0,                 # Mixup augmentation probability
        copy_paste=0.0,            # Copy paste augmentation probability
    )
    
    print("=== Eğitim Tamamlandı ===")
    print(f"En iyi model: {model.trainer.best}")
    print(f"Son model: {model.trainer.last}")
    
    return results

def validate_model():
    
    print("=== Model Doğrulaması ===")
    
    # En iyi modeli yükle
    best_model = YOLO('runs/detect/turkish_license_plate/weights/best.pt')
    
    # Validation sonuçları
    results = best_model.val(data='data.yaml')
    
    print(f"mAP50: {results.box.map50}")
    print(f"mAP50-95: {results.box.map}")
    print("=" * 30)

def export_model():
    
    print("=== Model Export ===")
    
    model = YOLO('runs/detect/turkish_license_plate/weights/best.pt')
    
    # Farklı formatlarda export
    formats = ['onnx', 'torchscript', 'tflite']
    
    for format_type in formats:
        try:
            exported = model.export(format=format_type)
            print(f"Model {format_type} formatında export edildi: {exported}")
        except Exception as e:
            print(f"{format_type} export hatası: {e}")

def main():
    
    print("🚗 YOLO8 Türkçe Plaka Tespit Sistemi Eğitimi")
    print("=" * 50)
    
    try:
        # Ortamı kontrol et
        check_environment()
        
        # Klasörleri hazırla
        prepare_directories()
        
        # data.yaml dosyasının varlığını kontrol et
        if not os.path.exists('data.yaml'):
            print("⚠️  UYARI: data.yaml dosyası bulunamadı!")
            print("Lütfen önce data.yaml dosyasını oluşturun.")
            return
        
        # Modeli eğit
        results = train_model()
        
        # Modeli doğrula
        validate_model()
        
        # Modeli export et
        export_model()
        
        print("✅ Tüm işlemler başarıyla tamamlandı!")
        print("Eğitilen model: runs/detect/turkish_license_plate/weights/best.pt")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()