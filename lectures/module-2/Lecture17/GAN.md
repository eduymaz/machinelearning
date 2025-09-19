## 🎨 GAN ile Veri Artırma ve Synthetic Data Generation

### GAN (Generative Adversarial Networks) Nedir?

**GAN**, Ian Goodfellow tarafından 2014'te önerilen, iki sinir ağının (Generator ve Discriminator) birbirleriyle yarıştığı generative model mimarisidir. YOLO8 plaka tespiti bağlamında, GAN'lar veri artırma ve sentetik veri üretimi için kullanılır.

### GAN'ların Plaka Tespitindeki Rolü

#### 1. **Data Augmentation**: Mevcut plaka verilerini çeşitlendirme
#### 2. **Synthetic Data Generation**: Yeni plaka görüntüleri üretme
#### 3. **Domain Adaptation**: Farklı koşullar için veri üretme
#### 4. **Adversarial Training**: Robust model eğitimi

---

### 🏗️ GAN Mimarileri ve Plaka Tespiti Uygulamaları

#### 1. **StyleGAN ile Plaka Varyasyonu**
```python
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np

class LicensePlateStyleGAN:
    """
    StyleGAN tabanlı plaka görüntü üreticisi
    Farklı stil ve koşullarda plaka görüntüleri üretir
    """
    
    def __init__(self, model_path="stylegan2-license-plate.pkl"):
        self.generator = self.load_pretrained_stylegan(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pretrained_stylegan(self, model_path):
        """Pre-trained StyleGAN modelini yükle"""
        # StyleGAN2 implementasyonu
        try:
            import pickle
            with open(model_path, 'rb') as f:
                generator = pickle.load(f)['G_ema']
            return generator
        except:
            print("StyleGAN modeli bulunamadı, basit GAN kullanılacak")
            return self.create_simple_generator()
    
    def create_simple_generator(self):
        """Basit generator ağı oluştur"""
        class SimpleGenerator(nn.Module):
            def __init__(self, latent_dim=100, img_channels=3, img_size=128):
                super().__init__()
                self.img_size = img_size
                
                self.model = nn.Sequential(
                    # Latent vector'dan feature map'e
                    nn.Linear(latent_dim, 256 * 8 * 8),
                    nn.BatchNorm1d(256 * 8 * 8),
                    nn.ReLU(True),
                    
                    # Reshape ve ConvTranspose katmanları
                    nn.Unflatten(1, (256, 8, 8)),
                    
                    # 8x8 -> 16x16
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    
                    # 16x16 -> 32x32
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    
                    # 32x32 -> 64x64
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    
                    # 64x64 -> 128x128
                    nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
                    nn.Tanh()
                )
            
            def forward(self, z):
                return self.model(z)
        
        return SimpleGenerator()
    
    def generate_plates(self, num_plates=100, style_mixing=True):
        """
        Çeşitli stilerde plaka görüntüleri üret
        
        Args:
            num_plates: Üretilecek plaka sayısı
            style_mixing: Stil karıştırması kullan
        """
        generated_plates = []
        
        with torch.no_grad():
            for i in range(num_plates):
                # Latent kod üret
                latent_code = torch.randn(1, 100).to(self.device)
                
                if style_mixing and hasattr(self.generator, 'synthesis'):
                    # StyleGAN style mixing
                    fake_plate = self.generator.synthesis(latent_code, noise_mode='const')
                else:
                    # Basit generation
                    fake_plate = self.generator(latent_code)
                
                # Tensor'ı görüntüye çevir
                plate_img = self.tensor_to_image(fake_plate)
                generated_plates.append(plate_img)
        
        return generated_plates
    
    def tensor_to_image(self, tensor):
        """PyTorch tensor'ını OpenCV görüntüsüne çevir"""
        # [-1, 1] aralığından [0, 255] aralığına
        img = (tensor.squeeze().cpu().numpy() + 1) * 127.5
        img = img.astype(np.uint8)
        
        # CHW -> HWC formatına çevir
        if len(img.shape) == 3:
            img = np.transpose(img, (1, 2, 0))
        
        # RGB -> BGR (OpenCV formatı)
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img

# Kullanım örneği
style_gan = LicensePlateStyleGAN()
synthetic_plates = style_gan.generate_plates(num_plates=500)
```

#### 2. **CycleGAN ile Domain Transfer**
```python
class PlateConditionTransfer:
    """
    CycleGAN ile plaka görüntülerini farklı koşullara transfer et
    Gündüz -> Gece, Temiz -> Kirli, Normal -> Yağmurlu vb.
    """
    
    def __init__(self):
        self.day_to_night = self.load_cyclegan_model('day2night.pth')
        self.clean_to_dirty = self.load_cyclegan_model('clean2dirty.pth')
        self.normal_to_rainy = self.load_cyclegan_model('normal2rainy.pth')
    
    def load_cyclegan_model(self, model_path):
        """CycleGAN modelini yükle"""
        class ResNetGenerator(nn.Module):
            def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
                super().__init__()
                
                # Encoder
                model = [
                    nn.Conv2d(input_nc, ngf, 7, 1, 3),
                    nn.InstanceNorm2d(ngf),
                    nn.ReLU(True)
                ]
                
                # Downsampling
                for i in range(2):
                    mult = 2**i
                    model += [
                        nn.Conv2d(ngf * mult, ngf * mult * 2, 3, 2, 1),
                        nn.InstanceNorm2d(ngf * mult * 2),
                        nn.ReLU(True)
                    ]
                
                # ResNet blocks
                mult = 2**2
                for i in range(n_blocks):
                    model += [ResNetBlock(ngf * mult)]
                
                # Upsampling
                for i in range(2):
                    mult = 2**(2-i)
                    model += [
                        nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, 2, 1, 1),
                        nn.InstanceNorm2d(ngf * mult // 2),
                        nn.ReLU(True)
                    ]
                
                # Output layer
                model += [
                    nn.Conv2d(ngf, output_nc, 7, 1, 3),
                    nn.Tanh()
                ]
                
                self.model = nn.Sequential(*model)
            
            def forward(self, x):
                return self.model(x)
        
        generator = ResNetGenerator()
        try:
            generator.load_state_dict(torch.load(model_path))
        except:
            print(f"Model {model_path} bulunamadı, rastgele ağırlıklar kullanılıyor")
        
        return generator
    
    def transfer_conditions(self, plate_images, target_condition='night'):
        """
        Plaka görüntülerini farklı koşullara transfer et
        
        Args:
            plate_images: Kaynak plaka görüntüleri
            target_condition: 'night', 'dirty', 'rainy', 'foggy'
        """
        transferred_images = []
        
        # Model seçimi
        if target_condition == 'night':
            model = self.day_to_night
        elif target_condition == 'dirty':
            model = self.clean_to_dirty
        elif target_condition == 'rainy':
            model = self.normal_to_rainy
        else:
            raise ValueError(f"Desteklenmeyen koşul: {target_condition}")
        
        model.eval()
        with torch.no_grad():
            for plate_img in plate_images:
                # Görüntüyü tensor'a çevir
                tensor_img = self.image_to_tensor(plate_img)
                
                # Domain transfer uygula
                transferred_tensor = model(tensor_img.unsqueeze(0))
                
                # Tensor'ı görüntüye çevir
                transferred_img = self.tensor_to_image(transferred_tensor)
                transferred_images.append(transferred_img)
        
        return transferred_images
    
    def image_to_tensor(self, image):
        """OpenCV görüntüsünü PyTorch tensor'ına çevir"""
        # BGR -> RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize [0, 255] -> [-1, 1]
        image = (image / 127.5) - 1.0
        
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        
        return torch.FloatTensor(image)

# Kullanım örneği
condition_transfer = PlateConditionTransfer()
night_plates = condition_transfer.transfer_conditions(day_plates, 'night')
dirty_plates = condition_transfer.transfer_conditions(clean_plates, 'dirty')
```

#### 3. **Conditional GAN ile Kontrollü Üretim**
```python
class ConditionalPlateGAN:
    """
    Conditional GAN ile belirli özelliklerde plaka üretimi
    Plaka formatı, renk, arka plan vb. kontrol edilebilir
    """
    
    def __init__(self):
        self.generator = self.build_conditional_generator()
        self.condition_encoder = self.build_condition_encoder()
        
    def build_conditional_generator(self):
        """Conditional generator ağını oluştur"""
        class ConditionalGenerator(nn.Module):
            def __init__(self, latent_dim=100, condition_dim=50, img_channels=3):
                super().__init__()
                
                # Condition ve noise'i birleştir
                input_dim = latent_dim + condition_dim
                
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, 256 * 8 * 8),
                    nn.BatchNorm1d(256 * 8 * 8),
                    nn.ReLU(True)
                )
                
                self.conv_blocks = nn.Sequential(
                    nn.Unflatten(1, (256, 8, 8)),
                    
                    # 8x8 -> 16x16
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    
                    # 16x16 -> 32x32
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    
                    # 32x32 -> 64x64
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    
                    # 64x64 -> 128x128
                    nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
                    nn.Tanh()
                )
            
            def forward(self, noise, condition):
                # Noise ve condition'ı birleştir
                combined_input = torch.cat([noise, condition], dim=1)
                
                # FC layer'dan geçir
                x = self.fc(combined_input)
                
                # Conv blocks'dan geçir
                x = self.conv_blocks(x)
                
                return x
        
        return ConditionalGenerator()
    
    def build_condition_encoder(self):
        """Condition encoder ağını oluştur"""
        class ConditionEncoder(nn.Module):
            def __init__(self, vocab_size=1000, embed_dim=50):
                super().__init__()
                
                # Text/Label embedding
                self.text_embedding = nn.Embedding(vocab_size, embed_dim)
                
                # Numeric features için MLP
                self.numeric_mlp = nn.Sequential(
                    nn.Linear(10, 32),  # 10 numeric feature
                    nn.ReLU(),
                    nn.Linear(32, embed_dim//2)
                )
                
                # Final projection
                self.projection = nn.Linear(embed_dim + embed_dim//2, embed_dim)
            
            def forward(self, text_labels, numeric_features):
                # Text embeddings
                text_emb = self.text_embedding(text_labels).mean(dim=1)
                
                # Numeric features
                numeric_emb = self.numeric_mlp(numeric_features)
                
                # Birleştir ve project et
                combined = torch.cat([text_emb, numeric_emb], dim=1)
                condition_vector = self.projection(combined)
                
                return condition_vector
        
        return ConditionEncoder()
    
    def generate_controlled_plates(self, conditions_list):
        """
        Belirli koşullarda plaka üret
        
        Args:
            conditions_list: Liste of dicts containing:
                - plate_format: 'old', 'new', 'diplomatic'
                - background: 'street', 'parking', 'highway'
                - lighting: 'day', 'night', 'sunset'
                - weather: 'clear', 'rainy', 'foggy'
                - angle: 'front', 'side', 'angled'
        """
        generated_plates = []
        
        self.generator.eval()
        with torch.no_grad():
            for condition_dict in conditions_list:
                # Condition'ı encode et
                condition_vector = self.encode_condition(condition_dict)
                
                # Noise üret
                noise = torch.randn(1, 100)
                
                # Plaka üret
                fake_plate = self.generator(noise, condition_vector)
                
                # Tensor'ı görüntüye çevir
                plate_img = self.tensor_to_image(fake_plate)
                generated_plates.append(plate_img)
        
        return generated_plates
    
    def encode_condition(self, condition_dict):
        """Condition dict'ini vector'a çevir"""
        # Condition mapping
        condition_vocab = {
            'plate_format': {'old': 1, 'new': 2, 'diplomatic': 3},
            'background': {'street': 1, 'parking': 2, 'highway': 3},
            'lighting': {'day': 1, 'night': 2, 'sunset': 3},
            'weather': {'clear': 1, 'rainy': 2, 'foggy': 3},
            'angle': {'front': 1, 'side': 2, 'angled': 3}
        }
        
        # Text labels
        text_labels = []
        for key, value in condition_dict.items():
            if key in condition_vocab and value in condition_vocab[key]:
                text_labels.append(condition_vocab[key][value])
            else:
                text_labels.append(0)  # Unknown token
        
        text_tensor = torch.tensor([text_labels])
        
        # Numeric features (örnek: brightness, contrast, saturation, vb.)
        numeric_features = torch.randn(1, 10)  # Placeholder
        
        # Condition encode et
        condition_vector = self.condition_encoder(text_tensor, numeric_features)
        
        return condition_vector

# Kullanım örneği
cond_gan = ConditionalPlateGAN()

# Farklı koşullarda plaka üret
conditions = [
    {'plate_format': 'new', 'background': 'street', 'lighting': 'day', 'weather': 'clear', 'angle': 'front'},
    {'plate_format': 'old', 'background': 'highway', 'lighting': 'night', 'weather': 'rainy', 'angle': 'side'},
    {'plate_format': 'diplomatic', 'background': 'parking', 'lighting': 'sunset', 'weather': 'foggy', 'angle': 'angled'}
]

controlled_plates = cond_gan.generate_controlled_plates(conditions)
```

---

### 🎯 GAN ile YOLO8 Eğitimini Geliştirme

#### 1. **Adversarial Data Augmentation**
```python
class AdversarialDataAugmentation:
    """
    GAN ile üretilen adversarial örneklerle YOLO8 robustluğunu artır
    """
    
    def __init__(self, yolo_model, gan_generator):
        self.yolo_model = yolo_model
        self.gan_generator = gan_generator
        
    def generate_hard_negatives(self, real_plates, num_negatives=100):
        """
        YOLO8'in yanlış tespit ettiği hard negative örnekler üret
        """
        hard_negatives = []
        
        for i in range(num_negatives):
            # GAN ile sahte plaka üret
            fake_plate = self.gan_generator.generate_single_plate()
            
            # YOLO8 ile tespit yap
            detections = self.yolo_model(fake_plate, verbose=False)
            
            # Yanlış pozitif kontrolü
            if len(detections[0].boxes) > 0:  # YOLO bir şey tespit etti
                confidence = detections[0].boxes.conf.max()
                
                # Düşük confidence'li tespitler hard negative olarak kullan
                if confidence < 0.7:
                    hard_negatives.append(fake_plate)
        
        return hard_negatives
    
    def adversarial_training_loop(self, train_data, epochs=100):
        """
        Adversarial training döngüsü
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Normal eğitim
            normal_loss = self.yolo_model.train(
                data=train_data, 
                epochs=1, 
                verbose=False
            )
            
            # Her 5 epoch'ta adversarial samples ekle
            if epoch % 5 == 0:
                # Hard negative örnekler üret
                hard_negatives = self.generate_hard_negatives(
                    train_data, 
                    num_negatives=50
                )
                
                # Hard negatives ile eğitim datasını genişlet
                augmented_data = self.merge_data(train_data, hard_negatives)
                
                # Adversarial eğitim
                adv_loss = self.yolo_model.train(
                    data=augmented_data,
                    epochs=1,
                    lr0=0.001,  # Düşük learning rate
                    verbose=False
                )
                
                print(f"   Normal Loss: {normal_loss}, Adversarial Loss: {adv_loss}")

# Kullanım
adv_augmentation = AdversarialDataAugmentation(yolo_model, style_gan.generator)
adv_augmentation.adversarial_training_loop("data.yaml", epochs=100)
```

#### 2. **Synthetic Dataset Creation Pipeline**
```python
class SyntheticPlateDatasetCreator:
    """
    GAN'lar kullanarak kapsamlı synthetic plaka dataseti oluştur
    """
    
    def __init__(self):
        self.style_gan = LicensePlateStyleGAN()
        self.condition_gan = ConditionalPlateGAN()
        self.cycle_gan = PlateConditionTransfer()
        
    def create_balanced_dataset(self, target_size=10000):
        """
        Balanced synthetic dataset oluştur
        """
        dataset = {
            'images': [],
            'annotations': [],
            'metadata': []
        }
        
        # Farklı kategorilerde veri üret
        categories = {
            'standard_plates': 0.4,      # %40 standard plakalar
            'old_format_plates': 0.2,    # %20 eski format
            'diplomatic_plates': 0.1,    # %10 diplomatik
            'challenging_conditions': 0.3 # %30 zor koşullar
        }
        
        for category, ratio in categories.items():
            num_samples = int(target_size * ratio)
            print(f"Generating {num_samples} {category} samples...")
            
            if category == 'standard_plates':
                samples = self.generate_standard_plates(num_samples)
            elif category == 'old_format_plates':
                samples = self.generate_old_format_plates(num_samples)
            elif category == 'diplomatic_plates':
                samples = self.generate_diplomatic_plates(num_samples)
            else:  # challenging_conditions
                samples = self.generate_challenging_plates(num_samples)
            
            dataset['images'].extend(samples['images'])
            dataset['annotations'].extend(samples['annotations'])
            dataset['metadata'].extend(samples['metadata'])
        
        return dataset
    
    def generate_standard_plates(self, num_samples):
        """Standard Türk plakalarını üret"""
        conditions = []
        
        for i in range(num_samples):
            condition = {
                'plate_format': 'new',
                'background': np.random.choice(['street', 'parking', 'highway']),
                'lighting': np.random.choice(['day', 'sunset'], p=[0.8, 0.2]),
                'weather': np.random.choice(['clear', 'cloudy'], p=[0.9, 0.1]),
                'angle': np.random.choice(['front', 'angled'], p=[0.7, 0.3])
            }
            conditions.append(condition)
        
        # Conditional GAN ile üret
        images = self.condition_gan.generate_controlled_plates(conditions)
        
        # Annotations oluştur (synthetic için perfect bounding boxes)
        annotations = []
        for img in images:
            h, w = img.shape[:2]
            # Plaka genelde görüntünün %60-80'ini kaplar
            bbox_w = w * np.random.uniform(0.6, 0.8)
            bbox_h = h * np.random.uniform(0.15, 0.25)
            center_x = w * np.random.uniform(0.3, 0.7)
            center_y = h * np.random.uniform(0.4, 0.6)
            
            # YOLO format
            annotation = [
                0,  # class_id (license_plate)
                center_x / w,  # normalized center_x
                center_y / h,  # normalized center_y
                bbox_w / w,    # normalized width
                bbox_h / h     # normalized height
            ]
            annotations.append(annotation)
        
        # Metadata
        metadata = [{'category': 'standard', 'synthetic': True} for _ in range(num_samples)]
        
        return {
            'images': images,
            'annotations': annotations,
            'metadata': metadata
        }
    
    def generate_challenging_plates(self, num_samples):
        """Zor koşullarda plaka görüntüleri üret"""
        # Önce normal plakalar üret
        base_plates = self.style_gan.generate_plates(num_samples)
        
        # Challenging conditions uygula
        challenging_images = []
        for plate in base_plates:
            # Rastgele challenging condition seç
            condition = np.random.choice(['night', 'dirty', 'rainy', 'foggy'])
            
            if condition in ['night', 'dirty', 'rainy']:
                transformed = self.cycle_gan.transfer_conditions([plate], condition)[0]
            else:  # foggy
                transformed = self.add_fog_effect(plate)
            
            challenging_images.append(transformed)
        
        # Annotations (challenging conditions için daha az kesin)
        annotations = []
        for img in challenging_images:
            h, w = img.shape[:2]
            # Challenging conditions'da bounding box daha belirsiz
            bbox_w = w * np.random.uniform(0.5, 0.7)
            bbox_h = h * np.random.uniform(0.12, 0.22)
            center_x = w * np.random.uniform(0.25, 0.75)
            center_y = h * np.random.uniform(0.35, 0.65)
            
            annotation = [0, center_x/w, center_y/h, bbox_w/w, bbox_h/h]
            annotations.append(annotation)
        
        metadata = [{'category': 'challenging', 'synthetic': True} for _ in range(num_samples)]
        
        return {
            'images': challenging_images,
            'annotations': annotations,
            'metadata': metadata
        }
    
    def add_fog_effect(self, image):
        """Görüntüye sis efekti ekle"""
        fog_intensity = np.random.uniform(0.3, 0.7)
        
        # Gaussian noise ile sis simülasyonu
        fog = np.random.normal(0, 30, image.shape).astype(np.uint8)
        fog = cv2.GaussianBlur(fog, (15, 15), 0)
        
        # Sis efektini uygula
        foggy_image = cv2.addWeighted(image, 1-fog_intensity, fog, fog_intensity, 0)
        
        return np.clip(foggy_image, 0, 255).astype(np.uint8)
    
    def save_synthetic_dataset(self, dataset, output_dir="synthetic_dataset"):
        """Synthetic dataset'i YOLO formatında kaydet"""
        import os
        
        # Klasör yapısını oluştur
        os.makedirs(f"{output_dir}/train/images", exist_ok=True)
        os.makedirs(f"{output_dir}/train/labels", exist_ok=True)
        os.makedirs(f"{output_dir}/val/images", exist_ok=True)
        os.makedirs(f"{output_dir}/val/labels", exist_ok=True)
        
        # Train/Val split (80/20)
        total_samples = len(dataset['images'])
        train_split = int(total_samples * 0.8)
        
        for i, (image, annotation) in enumerate(zip(dataset['images'], dataset['annotations'])):
            # Split belirleme
            split = 'train' if i < train_split else 'val'
            
            # Görüntüyü kaydet
            img_filename = f"synthetic_{i:06d}.jpg"
            img_path = f"{output_dir}/{split}/images/{img_filename}"
            cv2.imwrite(img_path, image)
            
            # Annotation'ı kaydet
            label_filename = f"synthetic_{i:06d}.txt"
            label_path = f"{output_dir}/{split}/labels/{label_filename}"
            
            with open(label_path, 'w') as f:
                ann_str = ' '.join(map(str, annotation))
                f.write(ann_str + '\n')
        
        # data.yaml oluştur
        yaml_content = f"""
path: {output_dir}
train: train/images
val: val/images

nc: 1
names:
  0: license_plate

# Synthetic dataset generated with GANs
# Total samples: {total_samples}
# Train samples: {train_split}
# Val samples: {total_samples - train_split}
"""
        
        with open(f"{output_dir}/data.yaml", 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ Synthetic dataset kaydedildi: {output_dir}")
        print(f"   📊 Toplam örnek: {total_samples}")
        print(f"   🏋️ Eğitim: {train_split}")
        print(f"   🔬 Doğrulama: {total_samples - train_split}")

# Kullanım örneği
synthetic_creator = SyntheticPlateDatasetCreator()
synthetic_dataset = synthetic_creator.create_balanced_dataset(target_size=5000)
synthetic_creator.save_synthetic_dataset(synthetic_dataset, "gan_synthetic_plates")
```

---

### 📊 GAN Kalitesi Değerlendirme

#### **FID (Fréchet Inception Distance) Hesaplama**
```python
def calculate_fid_score(real_images, generated_images):
    """
    GAN kalitesini FID ile değerlendir
    Düşük FID = Daha iyi GAN kalitesi
    """
    from scipy.linalg import sqrtm
    from sklearn.metrics import mean_squared_error
    
    # Feature extraction (Inception v3 kullan)
    inception_model = load_inception_v3()
    
    # Real ve fake images için features çıkar
    real_features = extract_inception_features(real_images, inception_model)
    fake_features = extract_inception_features(generated_images, inception_model)
    
    # Gaussian distribution parametrelerini hesapla
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # FID hesapla
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    
    return fid
```

---

*Bu GAN bölümü, YOLO8 plaka tespiti projesinde synthetic data generation, data augmentation ve adversarial training konularını kapsamlı şekilde ele alır. GAN'ların computer vision projelerindeki pratik uygulamalarını gösterir.*
