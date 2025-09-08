try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Installing simplified version...")
    TENSORFLOW_AVAILABLE = False

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
try:
    import seaborn as sns
except ImportError:
    print("Seaborn not available. Using matplotlib for plots.")
    sns = None

class CNNDemo:
    """
    CNN (Convolutional Neural Network) demonstration class
    Bu sınıf CNN'lerin nasıl çalıştığını pratik örneklerle gösterir
    """
    
    def __init__(self):
        self.model = None
        self.history = None
        
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
    
    def visualize_sample_data(self, X, y, num_samples=12):
        """Örnek verileri görselleştir"""
        class_names = ['Daire', 'Kare', 'Üçgen']
        
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        fig.suptitle('Örnek Veri Seti - CNN Sınıflandırma', fontsize=16)
        
        for i in range(num_samples):
            row = i // 4
            col = i % 4
            axes[row, col].imshow(X[i])
            axes[row, col].set_title(f'Sınıf: {class_names[y[i]]}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_simple_cnn(self, input_shape=(64, 64, 3), num_classes=3):
        """
        Basit CNN mimarisi oluşturma
        Her katmanın işlevini açıklayan model
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow mevcut değil. Lütfen 'pip install tensorflow' komutunu çalıştırın.")
            return None
            
        print("CNN Modeli oluşturuluyor...")
        
        model = keras.Sequential([
            # 1. Convolution + ReLU + Pooling bloğu
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                          name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            
            # 2. Convolution + ReLU + Pooling bloğu
            layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            
            # 3. Convolution + ReLU + Pooling bloğu
            layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            
            # Flatten - 3D'den 1D'ye dönüştürme
            layers.Flatten(name='flatten'),
            
            # Fully Connected katmanlar
            layers.Dense(128, activation='relu', name='fc1'),
            layers.Dropout(0.5, name='dropout'),  # Overfitting önleme
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Model derlemesi
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def visualize_model_architecture(self):
        """Model mimarisini görselleştir"""
        if self.model is None:
            print("Önce model oluşturun!")
            return
        
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow mevcut değil. Model mimarisi gösterilemiyor.")
            return
        
        print("\nCNN Model Mimarisi:")
        print("=" * 50)
        self.model.summary()
        
        # Model görselleştirme (optional)
        try:
            keras.utils.plot_model(
                self.model,
                to_file='cnn_architecture.png',
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB'
            )
            print("Model mimarisi 'cnn_architecture.png' dosyasına kaydedildi.")
        except Exception as e:
            print(f"Model görselleştirme hatası: {e}")
            print("Graphviz yüklü değil olabilir. Sadece text output gösteriliyor.")
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=10):
        """
        Model eğitimi
        """
        if self.model is None:
            print("Önce model oluşturun!")
            return None
            
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow mevcut değil. Model eğitilemez.")
            return None
        
        print(f"\nModel eğitimi başlıyor... ({epochs} epoch)")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        # Model eğitimi
        try:
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            print("Model eğitimi tamamlandı!")
            return self.history
        except Exception as e:
            print(f"Eğitim hatası: {e}")
            return None
    
    def visualize_training_history(self):
        """Eğitim geçmişini görselleştir"""
        if self.history is None:
            print("Önce modeli eğitin!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy grafiği
        ax1.plot(self.history.history['accuracy'], label='Eğitim Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss grafiği
        ax2.plot(self.history.history['loss'], label='Eğitim Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_feature_maps(self, X_sample):
        """
        Özellik haritalarını görselleştir
        CNN'in nasıl özellik çıkardığını gösterir
        """
        if self.model is None:
            print("Önce model oluşturun!")
            return
        
        # İlk birkaç katmanın çıktılarını al
        layer_outputs = [layer.output for layer in self.model.layers[:6]]
        activation_model = keras.Model(inputs=self.model.input, outputs=layer_outputs)
        
        # Aktivasyonları hesapla
        activations = activation_model.predict(X_sample.reshape(1, *X_sample.shape))
        
        layer_names = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('CNN Katmanları - Özellik Haritaları', fontsize=16)
        
        for i, (activation, layer_name) in enumerate(zip(activations, layer_names)):
            row = i // 3
            col = i % 3
            
            # İlk kanal veya havuzlama sonucu göster
            if len(activation.shape) == 4:  # Conv katmanı
                feature_map = activation[0, :, :, 0]  # İlk kanal
            else:  # Pooling katmanı
                feature_map = activation[0, :, :, 0]
            
            axes[row, col].imshow(feature_map, cmap='viridis')
            axes[row, col].set_title(f'{layer_name}\nŞekil: {feature_map.shape}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, X_test, y_test):
        """
        Model performansını değerlendir
        """
        if self.model is None:
            print("Önce model oluşturun ve eğitin!")
            return
        
        print("\nModel değerlendirmesi...")
        
        # Test accuracy
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Tahminler
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        class_names = ['Daire', 'Kare', 'Üçgen']
        print("\nSınıflandırma Raporu:")
        print("=" * 50)
        print(classification_report(y_test, y_pred_classes, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(8, 6))
        
        if sns is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
        else:
            # Manuel heatmap çizimi
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
    
    def visualize_predictions(self, X_test, y_test, num_samples=12):
        """Tahminleri görselleştir"""
        if self.model is None:
            print("Önce model oluşturun ve eğitin!")
            return
        
        class_names = ['Daire', 'Kare', 'Üçgen']
        
        # Rastgele örnekler seç
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        # Tahminler
        predictions = self.model.predict(X_test[indices])
        predicted_classes = np.argmax(predictions, axis=1)
        
        fig, axes = plt.subplots(3, 4, figsize=(15, 12))
        fig.suptitle('CNN Tahminleri', fontsize=16)
        
        for i, idx in enumerate(indices):
            row = i // 4
            col = i % 4
            
            axes[row, col].imshow(X_test[idx])
            
            # Doğru/yanlış tahmin rengi
            color = 'green' if predicted_classes[i] == y_test[idx] else 'red'
            
            title = f'Gerçek: {class_names[y_test[idx]]}\n'
            title += f'Tahmin: {class_names[predicted_classes[i]]}\n'
            title += f'Güven: {np.max(predictions[i]):.2f}'
            
            axes[row, col].set_title(title, color=color)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def transfer_learning_demo(self):
        """
        Transfer Learning örneği
        VGG16 kullanarak
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow mevcut değil. Transfer Learning örneği gösterilemiyor.")
            return None
            
        print("\nTransfer Learning Örneği (VGG16)")
        print("=" * 40)
        
        try:
            # VGG16 base model
            base_model = keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(64, 64, 3)
            )
            
            # Base model katmanlarını dondur
            base_model.trainable = False
            
            # Yeni model oluştur
            transfer_model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(3, activation='softmax')
            ])
            
            transfer_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Transfer Learning modeli hazır!")
            transfer_model.summary()
            
            return transfer_model
        except Exception as e:
            print(f"Transfer Learning hatası: {e}")
            print("Internet bağlantısı gerekebilir (ImageNet weights indirmek için).")
            return None

# Ana demo fonksiyonu
def main():
    """
    CNN Demo ana fonksiyonu
    Tüm CNN kavramlarını pratik örneklerle gösterir
    """
    print("CNN (Convolutional Neural Network) Demo")
    print("=" * 50)
    
    # CNN demo sınıfını başlat
    cnn_demo = CNNDemo()
    
    # 1. Veri oluşturma
    print("\n1. Örnek Veri Seti Oluşturma")
    X, y = cnn_demo.create_sample_data(num_samples=1000, img_size=64)
    
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
    cnn_demo.visualize_sample_data(X, y)
    
    # 3. Model oluşturma
    print("\n3. CNN Model Oluşturma")
    model = cnn_demo.create_simple_cnn()
    cnn_demo.visualize_model_architecture()
    
    # 4. Model eğitimi
    print("\n4. Model Eğitimi")
    cnn_demo.train_model(X_train, y_train, X_val, y_val, epochs=20)
    
    # 5. Eğitim görselleştirme
    print("\n5. Eğitim Sonuçları")
    cnn_demo.visualize_training_history()
    
    # 6. Özellik haritaları
    print("\n6. CNN Özellik Haritaları")
    sample_image = X_test[0]
    cnn_demo.visualize_feature_maps(sample_image)
    
    # 7. Model değerlendirme
    print("\n7. Model Performans Değerlendirmesi")
    cnn_demo.evaluate_model(X_test, y_test)
    
    # 8. Tahmin görselleştirme
    print("\n8. Model Tahminleri")
    cnn_demo.visualize_predictions(X_test, y_test)
    
    # 9. Transfer Learning demo
    print("\n9. Transfer Learning Örneği")
    transfer_model = cnn_demo.transfer_learning_demo()
    
    print("\nCNN Demo tamamlandı!")
    print("Bu örnekte gördükleriniz:")
    print("- CNN katmanlarının işleyişi")
    print("- Özellik haritalarının görselleştirilmesi")
    print("- Model eğitimi ve değerlendirmesi")
    print("- Transfer Learning kullanımı")

if __name__ == "__main__":
    main()