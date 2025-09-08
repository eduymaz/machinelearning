try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import the main CNN class if TensorFlow is available
if TENSORFLOW_AVAILABLE:
    from cnn_demo import CNNDemo

class SimpleCNNDemo:
    """
    TensorFlow olmadan CNN kavramlarını gösteren basit demo
    """
    
    def __init__(self):
        self.model = None
        
    def create_sample_data(self, num_samples=1000, img_size=64):
        """
        Örnek veri seti oluşturma
        3 farklı şekil sınıfı: Daire, Kare, Üçgen
        """
        print("Örnek veri seti oluşturuluyor...")
        
        X = []
        y = []
        
        for i in range(num_samples):
            # Boş görüntü oluştur
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
            # Rastgele şekil seç (0: Daire, 1: Kare, 2: Üçgen)
            shape_type = np.random.randint(0, 3)
            
            # Rastgele renk
            color = (np.random.randint(100, 255),
                    np.random.randint(100, 255),
                    np.random.randint(100, 255))
            
            if shape_type == 0:  # Daire
                center = (img_size//2, img_size//2)
                radius = np.random.randint(15, 25)
                cv2.circle(img, center, radius, color, -1)
                
            elif shape_type == 1:  # Kare
                size = np.random.randint(20, 30)
                x1 = (img_size - size) // 2
                y1 = (img_size - size) // 2
                cv2.rectangle(img, (x1, y1), (x1 + size, y1 + size), color, -1)
                
            else:  # Üçgen
                center_x, center_y = img_size//2, img_size//2
                size = np.random.randint(15, 25)
                pts = np.array([
                    [center_x, center_y - size],
                    [center_x - size, center_y + size],
                    [center_x + size, center_y + size]
                ], np.int32)
                cv2.fillPoly(img, [pts], color)
            
            X.append(img)
            y.append(shape_type)
        
        return np.array(X), np.array(y)
    
    def extract_features(self, images):
        """
        Basit özellik çıkarma (CNN benzeri)
        """
        print("Özellik çıkarma işlemi...")
        features = []
        
        for img in images:
            # Gri tonlamaya çevir
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Basit özellikler çıkar
            feature_vector = []
            
            # 1. Ortalama parlaklık
            feature_vector.append(np.mean(gray))
            
            # 2. Standart sapma
            feature_vector.append(np.std(gray))
            
            # 3. Kenar tespiti (Canny)
            edges = cv2.Canny(gray, 50, 150)
            feature_vector.append(np.sum(edges) / (edges.shape[0] * edges.shape[1]))
            
            # 4. Kontur sayısı
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            feature_vector.append(len(contours))
            
            # 5. En büyük kontur alanı
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                feature_vector.append(cv2.contourArea(largest_contour))
            else:
                feature_vector.append(0)
            
            # 6. Bölge özellikler (CNN pooling benzeri)
            h, w = gray.shape
            regions = [
                gray[:h//2, :w//2],      # Sol üst
                gray[:h//2, w//2:],      # Sağ üst
                gray[h//2:, :w//2],      # Sol alt
                gray[h//2:, w//2:]       # Sağ alt
            ]
            
            for region in regions:
                feature_vector.append(np.mean(region))
                feature_vector.append(np.std(region))
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def visualize_sample_data(self, X, y, num_samples=12):
        """Örnek verileri görselleştir"""
        class_names = ['Daire', 'Kare', 'Üçgen']
        
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        fig.suptitle('Örnek Veri Seti - CNN Kavramları', fontsize=16)
        
        for i in range(num_samples):
            row = i // 4
            col = i % 4
            axes[row, col].imshow(cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f'Sınıf: {class_names[y[i]]}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def simulate_cnn_layers(self, sample_image):
        """
        CNN katmanlarını simüle et
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('CNN Katmanları Simülasyonu', fontsize=16)
        
        # Orijinal görüntü
        axes[0, 0].imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Gri tonlama (Preprocessing)
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Preprocessing (Grayscale)')
        axes[0, 1].axis('off')
        
        # Edge Detection (Convolution benzeri)
        edges = cv2.Canny(gray, 50, 150)
        axes[0, 2].imshow(edges, cmap='gray')
        axes[0, 2].set_title('Feature Extraction (Edges)')
        axes[0, 2].axis('off')
        
        # Blurring (Pooling benzeri)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        axes[1, 0].imshow(blurred, cmap='gray')
        axes[1, 0].set_title('Pooling (Size Reduction)')
        axes[1, 0].axis('off')
        
        # Threshold (Activation benzeri)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        axes[1, 1].imshow(thresh, cmap='gray')
        axes[1, 1].set_title('Activation (Thresholding)')
        axes[1, 1].axis('off')
        
        # Contours (Feature Maps)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = sample_image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        axes[1, 2].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Feature Maps (Contours)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def train_simple_classifier(self, X_train, y_train):
        """
        Basit sınıflandırıcı eğitimi
        """
        print("Özellik çıkarma ve model eğitimi...")
        
        # Özellik çıkarma
        X_features = self.extract_features(X_train)
        
        # Random Forest sınıflandırıcı
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_features, y_train)
        
        print("Model eğitimi tamamlandı!")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Model değerlendirmesi
        """
        if self.model is None:
            print("Önce modeli eğitin!")
            return
        
        # Özellik çıkarma
        X_test_features = self.extract_features(X_test)
        
        # Tahminler
        y_pred = self.model.predict(X_test_features)
        
        # Performans değerlendirmesi
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Classification report
        class_names = ['Daire', 'Kare', 'Üçgen']
        print("\nSınıflandırma Raporu:")
        print("=" * 50)
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, cm[i, j], ha='center', va='center')
        plt.xticks(range(len(class_names)), class_names)
        plt.yticks(range(len(class_names)), class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        plt.show()

def simple_main():
    """
    TensorFlow olmadan CNN kavramlarını gösteren basit demo
    """
    print("Basit CNN Kavramları Demo (TensorFlow-free)")
    print("=" * 50)
    
    # Demo sınıfını başlat
    demo = SimpleCNNDemo()
    
    # 1. Veri oluşturma
    print("\n1. Örnek Veri Seti Oluşturma")
    X, y = demo.create_sample_data(num_samples=500, img_size=64)
    
    # 2. Veri görselleştirme
    print("\n2. Veri Görselleştirme")
    demo.visualize_sample_data(X, y)
    
    # 3. CNN katmanları simülasyonu
    print("\n3. CNN Katmanları Simülasyonu")
    sample_image = X[0]
    demo.simulate_cnn_layers(sample_image)
    
    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Model eğitimi
    print("\n4. Model Eğitimi (Feature-based Classification)")
    demo.train_simple_classifier(X_train, y_train)
    
    # 6. Model değerlendirmesi
    print("\n5. Model Performans Değerlendirmesi")
    demo.evaluate_model(X_test, y_test)
    
    print("\nBasit CNN Demo tamamlandı!")
    print("Bu örnekte CNN'lerin temel kavramları gösterildi:")
    print("- Görüntü ön işleme")
    print("- Özellik çıkarma (Convolution benzeri)")
    print("- Boyut azaltma (Pooling benzeri)")
    print("- Aktivasyon fonksiyonları (Thresholding)")
    print("- Sınıflandırma")

# Ana demo fonksiyonu - güncellenmiş
def main():
    """
    CNN Demo ana fonksiyonu
    TensorFlow varsa tam demo, yoksa basit demo çalıştırır
    """
    print("CNN (Convolutional Neural Network) Demo")
    print("=" * 50)
    
    if TENSORFLOW_AVAILABLE:
        print("TensorFlow mevcut. Tam CNN demo çalıştırılıyor...")
        
        # CNN demo sınıfını başlat
        cnn_demo = CNNDemo()
        
        # 1. Veri oluşturma
        print("\n1. Örnek Veri Seti Oluşturma")
        X, y = cnn_demo.create_sample_data(num_samples=500, img_size=64)  # Küçültüldü
        
        # Verileri normalize et (0-1 arası)
        X = X.astype('float32') / 255.0
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Eğitim seti: {X_train.shape}")
        print(f"Validation seti: {X_val.shape}")
        print(f"Test seti: {X_test.shape}")
        
        # 2. Veri görselleştirme
        print("\n2. Veri Görselleştirme")
        cnn_demo.visualize_sample_data((X*255).astype(np.uint8), y)
        
        # 3. Model oluşturma
        print("\n3. CNN Model Oluşturma")
        model = cnn_demo.create_simple_cnn()
        if model is not None:
            cnn_demo.visualize_model_architecture()
            
            # 4. Model eğitimi (kısa)
            print("\n4. Model Eğitimi")
            cnn_demo.train_model(X_train, y_train, X_val, y_val, epochs=5)
            
            if cnn_demo.history is not None:
                # 5. Eğitim görselleştirme
                print("\n5. Eğitim Sonuçları")
                cnn_demo.visualize_training_history()
                
                # 6. Model değerlendirme
                print("\n6. Model Performans Değerlendirmesi")
                cnn_demo.evaluate_model(X_test, y_test)
                
                # 7. Tahmin görselleştirme
                print("\n7. Model Tahminleri")
                cnn_demo.visualize_predictions(X_test, y_test)
        
        print("\nTensorFlow CNN Demo tamamlandı!")
    
    else:
        print("TensorFlow mevcut değil. Basit CNN kavramları demo çalıştırılıyor...")
        simple_main()

if __name__ == "__main__":
    main()