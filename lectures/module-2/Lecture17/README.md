# YOLO8 ile Türkçe Plaka Tespiti - Lecture 17

## 📚 Ders İçeriği
Bu ders notunda YOLO8 (You Only Look Once version 8) modelini kullanarak Türkçe plaka tespiti sisteminin geliştirilmesi ele alınmaktadır.

---

## 🧠 1. YOLO8 Model Nedir?

### Teorik Temeller
**YOLO (You Only Look Once)**, nesne tespit (object detection) alanında devrim yaratan bir deep learning mimarisidir. YOLO8, Ultralytics tarafından geliştirilen en güncel ve gelişmiş versiyondur.

### Temel Özellikler
- **Single-Shot Detection**: Görüntüyü tek geçişte işler
- **Real-time Performance**: Gerçek zamanlı performans sunar
- **Multi-task Learning**: Detection, segmentation, classification destekler
- **Anchor-free**: Anchor box'lara ihtiyaç duymaz
- **Advanced Architecture**: CSPDarknet53 backbone + PANet neck

### Mimari Yapı
```
Input (640x640) → Backbone (Feature Extraction) → Neck (Feature Fusion) → Head (Prediction)
```

### YOLO8 Model Varyantları
| Model | Boyut | Parameters | mAP | Speed |
|-------|--------|------------|-----|-------|
| YOLOv8n | Nano | 3.2M | 37.3 | 0.99ms |
| YOLOv8s | Small | 11.2M | 44.9 | 1.20ms |
| YOLOv8m | Medium | 25.9M | 50.2 | 1.83ms |
| YOLOv8l | Large | 43.7M | 52.9 | 2.39ms |
| YOLOv8x | Extra Large | 68.2M | 53.9 | 3.53ms |

---

## 🎯 2. Bounding Box Nedir?

### Tanım ve Amaç
**Bounding Box (Sınırlayıcı Kutu)**, nesne tespitinde tespit edilen nesnenin etrafını çevreleyen dikdörtgen kutudur. Nesnenin konumsal bilgilerini içerir.

### Koordinat Sistemleri

#### 1. XYXY Format (Absolute Coordinates)
```python
# (x1, y1) = Sol üst köşe
# (x2, y2) = Sağ alt köşe
bbox = [x1, y1, x2, y2]  # Örnek: [100, 50, 300, 200]
```

#### 2. XYWH Format (Center-based)
```python
# (x, y) = Merkez koordinatları
# w, h = Genişlik ve yükseklik
bbox = [x_center, y_center, width, height]
```

#### 3. YOLO Format (Normalized)
```python
# Tüm değerler 0-1 arasında normalize edilir
# x_center / image_width
# y_center / image_height  
# width / image_width
# height / image_height
bbox = [0.5, 0.3, 0.4, 0.2]
```

### Bounding Box Kalitesi Metrikleri

#### IoU (Intersection over Union)
```python
def calculate_iou(box1, box2):
    """
    IoU = Area of Intersection / Area of Union
    """
    # Intersection area hesaplama
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area hesaplama
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
```

---

## 📊 3. Turkish License Plate Dataset

### Dataset Özellikleri
**Kaynak**: [Kaggle - Turkish License Plate Dataset](https://www.kaggle.com/datasets/smaildurcan/turkish-license-plate-dataset?resource=download-directory)

### Dataset Yapısı
```
turkish-license-plate/
├── train/
│   ├── images/           # Eğitim görüntüleri
│   └── labels/           # YOLO format annotations
├── val/
│   ├── images/           # Doğrulama görüntüleri  
│   └── labels/           # YOLO format annotations
└── test/
    ├── images/           # Test görüntüleri
    └── labels/           # YOLO format annotations
```

### Annotation Format Örneği
```
# Dosya: image001.txt
# Format: class_id center_x center_y width height (normalized)
0 0.512 0.375 0.186 0.094
0 0.823 0.621 0.154 0.078
```

### Veri Ön İşleme Gereksinimleri
- **Görüntü formatları**: JPG, PNG, BMP
- **Çözünürlük**: Minimum 416x416, önerilen 640x640
- **Normalizasyon**: 0-1 arası pixel değerleri
- **Augmentation**: Rotation, scaling, brightness, contrast

---

## 🔧 4. Model Eğitimi ve Implementasyon

### Kurulum ve Bağımlılıklar
```bash
# Ultralytics YOLO8 kurulumu
pip install ultralytics

# Ek bağımlılıklar
pip install opencv-python torch torchvision matplotlib pillow
```

### Temel Kullanım
```python
from ultralytics import YOLO
import cv2
import numpy as np

# Model yükleme
model = YOLO("yolov8n.pt")  # Pre-trained model

# Tahmin
results = model("image.jpg")

# Eğitim
model.train(data="data.yaml", epochs=100, imgsz=640)
```

### Eğitim Parametreleri Açıklaması

#### Temel Parametreler
```python
def train():
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data="data.yaml",           # Dataset konfigürasyonu
        epochs=100,                 # Eğitim epoch sayısı
        imgsz=640,                  # Görüntü boyutu (640x640)
        batch=16,                   # Batch boyutu
        lr0=0.01,                   # İlk learning rate
        lrf=0.01,                   # Final learning rate
        momentum=0.937,             # SGD momentum
        weight_decay=0.0005,        # L2 regularization
        warmup_epochs=3,            # Warm-up epoch sayısı
        warmup_momentum=0.8,        # Warm-up momentum
        warmup_bias_lr=0.1,         # Warm-up bias learning rate
        box=7.5,                    # Box loss weight
        cls=0.5,                    # Classification loss weight
        dfl=1.5,                    # Distribution focal loss weight
        pose=12.0,                  # Pose loss weight
        kobj=2.0,                   # Keypoint objectness loss weight
        label_smoothing=0.0,        # Label smoothing
        nbs=64,                     # Nominal batch size
        hsv_h=0.015,               # Hue augmentation
        hsv_s=0.7,                 # Saturation augmentation
        hsv_v=0.4,                 # Value augmentation
        degrees=0.0,               # Rotation degrees
        translate=0.1,             # Translation fraction
        scale=0.5,                 # Scaling fraction
        shear=0.0,                 # Shear degrees
        perspective=0.0,           # Perspective coefficient
        flipud=0.0,                # Vertical flip probability
        fliplr=0.5,                # Horizontal flip probability
        mosaic=1.0,                # Mosaic probability
        mixup=0.0,                 # Mixup probability
        copy_paste=0.0             # Copy-paste probability
    )
    
    return results
```

---

## 📈 5. Model Değerlendirme Metrikleri

### Temel Metrikler

#### Precision (Kesinlik)
```python
# Precision = TP / (TP + FP)
# Tespit edilen nesnelerin ne kadarı doğru?
```

#### Recall (Duyarlılık)
```python  
# Recall = TP / (TP + FN)
# Gerçek nesnelerin ne kadarı tespit edildi?
```

#### mAP (mean Average Precision)
```python
# mAP@0.5: IoU threshold 0.5'te ortalama precision
# mAP@0.5:0.95: IoU threshold 0.5-0.95 arasında ortalama
```

#### F1-Score
```python
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
# Precision ve Recall'un harmonik ortalaması
```

### Confusion Matrix Analizi
```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(results):
    """Model performansını görselleştir"""
    cm = results.confusion_matrix.matrix
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - License Plate Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
```

---

## 🚀 6. Production Deployment

### Model Optimizasyonu

#### TensorRT Optimizasyonu (NVIDIA GPU)
```python
# TensorRT export
model.export(format='engine', half=True, dynamic=True, workspace=4)
```

#### ONNX Export (Cross-platform)
```python
# ONNX export
model.export(format='onnx', dynamic=True, simplify=True)
```

#### OpenVINO (Intel CPU optimizasyonu)
```python
# OpenVINO export  
model.export(format='openvino', half=True)
```

### Inference Optimizasyonu
```python
class OptimizedPredictor:
    def __init__(self, model_path, device='cuda'):
        self.model = YOLO(model_path)
        self.model.to(device)
        
    @torch.no_grad()
    def predict_batch(self, images, conf_threshold=0.25):
        """Batch inference için optimize edilmiş tahmin"""
        results = self.model(images, 
                           conf=conf_threshold,
                           iou=0.45,
                           half=True,  # FP16 precision
                           verbose=False)
        return results
```

---

## 💡 7. İleri Seviye Teknikler

### Data Augmentation Stratejileri
```python
# Albumentations ile gelişmiş augmentation
import albumentations as A

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(p=0.3),
    A.GaussNoise(p=0.3),
    A.RandomFog(p=0.2),
    A.RandomSunFlare(p=0.1),
], bbox_params=A.BboxParams(format='yolo'))
```

### Transfer Learning Stratejileri
```python
# Kademeli fine-tuning
def progressive_training():
    # 1. Freeze backbone, train head
    model.freeze([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    model.train(epochs=20, lr0=0.001)
    
    # 2. Unfreeze son katmanlar
    model.freeze([0, 1, 2, 3, 4, 5])
    model.train(epochs=30, lr0=0.0001)
    
    # 3. Full fine-tuning
    model.freeze([])
    model.train(epochs=50, lr0=0.00001)
```

### Ensemble Methods
```python
def ensemble_predict(models, image):
    """Çoklu model ensemble tahmin"""
    predictions = []
    
    for model in models:
        result = model(image, conf=0.1)
        predictions.append(result[0].boxes)
    
    # Weighted Box Fusion (WBF) uygula
    final_boxes = weighted_boxes_fusion(predictions)
    return final_boxes
```

---

## 🔍 8. Hata Ayıklama ve Troubleshooting

### Yaygın Problemler ve Çözümleri

#### 1. Düşük mAP Performansı
```python
# Olası çözümler:
# - Learning rate ayarlama
# - Daha fazla epoch
# - Data augmentation artırma  
# - Model boyutunu büyütme
# - Class imbalance kontrolü
```

#### 2. Overfitting
```python
# Çözüm stratejileri:
# - Early stopping
# - Dropout ekleme
# - Data augmentation
# - Regularization artırma
# - Validation split kontrolü
```

#### 3. Slow Training
```python
# Hızlandırma teknikleri:
# - Mixed precision training
# - Gradient accumulation
# - Multi-GPU training
# - Efficient data loading
# - Cache optimization
```

---

## 📚 9. Kaynaklar ve İleri Okuma

### Resmi Dokümantasyon
- [Ultralytics YOLO8 Docs](https://docs.ultralytics.com/)
- [PyTorch Object Detection](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)


### Benchmark Datasets
- COCO Dataset
- Open Images Dataset  
- Turkish License Plate Dataset

---

## 🎯 10. Proje Ödevleri

### Temel Seviye
1. YOLO8n modelini Turkish License Plate dataset ile eğitin
2. Farklı confidence threshold değerlerinin etkisini analiz edin
3. Test setinde mAP değerini hesaplayın

### İleri Seviye  
1. Custom augmentation pipeline geliştirin
2. Multi-scale training implementasyonu yapın
3. Real-time webcam application geliştirin
4. Model compression teknikleri uygulayın

### Expert Seviye
1. Custom loss function tasarlayın
2. Attention mechanism ekleyin  
3. Federated learning ile distributed training
4. Edge deployment için model optimize edin
