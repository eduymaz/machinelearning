import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage

class ImageSegmentation:
    def __init__(self, image_path=None):
        if image_path:
            self.image = cv2.imread(image_path)
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            # Create a sample image for demonstration
            self.image = self.create_sample_image()
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def create_sample_image(self):
        """Create a sample image with multiple objects for segmentation"""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Background gradient
        for i in range(400):
            for j in range(400):
                img[i, j] = [int(i/2), int(j/2), 100]
        
        # Add different colored objects
        cv2.circle(img, (150, 150), 60, (255, 100, 100), -1)  # Red circle
        cv2.rectangle(img, (250, 80), (350, 180), (100, 255, 100), -1)  # Green rectangle
        cv2.ellipse(img, (200, 300), (80, 40), 0, 0, 360, (100, 100, 255), -1)  # Blue ellipse
        
        # Add some noise
        noise = np.random.randint(0, 50, (400, 400, 3))
        img = cv2.add(img, noise.astype(np.uint8))
        
        return img
    
    def grabcut_segmentation(self, rect=None):
        """
        GrabCut algoritması ile ön plan - arka plan segmentasyonu
        rect: (x, y, width, height) tuple olarak segmentasyon yapılacak bölge
        """
        if rect is None:
            # Default rectangle for sample image
            rect = (50, 50, 300, 300)
        
        # Mask ve BGD/FGD modelleri için gerekli arrays
        mask = np.zeros(self.image_rgb.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # GrabCut algoritmasını çalıştır
        cv2.grabCut(self.image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Final mask oluştur (0,2: arka plan, 1,3: ön plan)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = self.image_rgb * mask2[:, :, np.newaxis]
        
        return result, mask2
    
    def kmeans_segmentation(self, k=4):
        """
        K-Means clustering ile renk tabanlı segmentasyon
        k: Küme sayısı
        """
        # Görüntüyü 2D array'e dönüştür
        data = self.image_rgb.reshape((-1, 3))
        data = np.float32(data)
        
        # K-Means clustering uygula
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Sonuçları yeniden şekillendir
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(self.image_rgb.shape)
        
        return segmented_image, labels.reshape(self.image_rgb.shape[:2])
    
    def watershed_segmentation(self):
        """
        Watershed algoritması ile segmentasyon
        """
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Gürültü azaltma
        gray = cv2.medianBlur(gray, 5)
        
        # Thresholding ile binary image elde et
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Watershed algoritması uygula
        markers = cv2.watershed(self.image, markers)
        result = self.image_rgb.copy()
        result[markers == -1] = [255, 0, 0]  # Boundaries kırmızı
        
        return result, markers
    
    def otsu_thresholding(self):
        """
        Otsu thresholding ile segmentasyon
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Otsu thresholding
        threshold_value, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary, threshold_value
    
    def adaptive_thresholding(self):
        """
        Adaptive thresholding ile segmentasyon
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
        
        return adaptive_thresh
    
    def region_growing(self, seed_point=None, threshold=30):
        """
        Region Growing algoritması ile segmentasyon
        seed_point: Başlangıç noktası (x, y)
        threshold: Benzerlik eşiği
        """
        if seed_point is None:
            seed_point = (200, 200)  # Default seed point
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Segmented image
        segmented = np.zeros((h, w), dtype=np.uint8)
        
        # Stack for region growing
        stack = [seed_point]
        seed_value = gray[seed_point[1], seed_point[0]]
        
        while stack:
            x, y = stack.pop()
            
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            
            if segmented[y, x] == 1:
                continue
            
            if abs(int(gray[y, x]) - int(seed_value)) <= threshold:
                segmented[y, x] = 1
                
                # Add neighboring pixels
                stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        
        return segmented * 255
    
    def display_all_segmentations(self):
        """Tüm segmentasyon sonuçlarını görüntüle"""
        # Segmentasyon işlemlerini gerçekleştir
        grabcut_result, grabcut_mask = self.grabcut_segmentation()
        kmeans_result, kmeans_labels = self.kmeans_segmentation()
        watershed_result, watershed_markers = self.watershed_segmentation()
        otsu_result, otsu_threshold = self.otsu_thresholding()
        adaptive_result = self.adaptive_thresholding()
        region_result = self.region_growing()
        
        # Sonuçları görüntüle
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Image Segmentation Techniques', fontsize=16)
        
        # Original
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # GrabCut
        axes[0, 1].imshow(grabcut_result)
        axes[0, 1].set_title('GrabCut Segmentation')
        axes[0, 1].axis('off')
        
        # K-Means
        axes[0, 2].imshow(kmeans_result)
        axes[0, 2].set_title('K-Means Clustering (k=4)')
        axes[0, 2].axis('off')
        
        # Watershed
        axes[1, 0].imshow(watershed_result)
        axes[1, 0].set_title('Watershed Segmentation')
        axes[1, 0].axis('off')
        
        # Otsu Thresholding
        axes[1, 1].imshow(otsu_result, cmap='gray')
        axes[1, 1].set_title(f'Otsu Thresholding (T={otsu_threshold:.1f})')
        axes[1, 1].axis('off')
        
        # Adaptive Thresholding
        axes[1, 2].imshow(adaptive_result, cmap='gray')
        axes[1, 2].set_title('Adaptive Thresholding')
        axes[1, 2].axis('off')
        
        # Region Growing
        axes[2, 0].imshow(region_result, cmap='gray')
        axes[2, 0].set_title('Region Growing')
        axes[2, 0].axis('off')
        
        # GrabCut Mask
        axes[2, 1].imshow(grabcut_mask, cmap='gray')
        axes[2, 1].set_title('GrabCut Mask')
        axes[2, 1].axis('off')
        
        # K-Means Labels
        axes[2, 2].imshow(kmeans_labels, cmap='tab10')
        axes[2, 2].set_title('K-Means Labels')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def interactive_grabcut_demo(self):
        """
        GrabCut için interaktif demo
        """
        print("GrabCut Segmentation Demo")
        print("========================")
        print("1. Orijinal görüntü")
        print("2. Dikdörtgen alanı tanımla")
        print("3. GrabCut algoritması uygula")
        print("4. Sonuç")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')
        
        # Rectangle area
        rect_img = self.image_rgb.copy()
        cv2.rectangle(rect_img, (50, 50), (350, 350), (255, 0, 0), 3)
        axes[0, 1].imshow(rect_img)
        axes[0, 1].set_title('2. Rectangle Selection')
        axes[0, 1].axis('off')
        
        # GrabCut result
        result, mask = self.grabcut_segmentation((50, 50, 300, 300))
        axes[1, 0].imshow(result)
        axes[1, 0].set_title('3. GrabCut Result')
        axes[1, 0].axis('off')
        
        # Mask
        axes[1, 1].imshow(mask, cmap='gray')
        axes[1, 1].set_title('4. Generated Mask')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
def main():
    """Ana fonksiyon - tüm segmentasyon tekniklerini demo eder"""
    print("Image Segmentation Techniques Demo")
    print("=" * 40)
    
    # Segmentation sınıfını başlat
    segmenter = ImageSegmentation()
    
    # Tüm segmentasyon tekniklerini göster
    segmenter.display_all_segmentations()
    
    # GrabCut interaktif demo
    segmenter.interactive_grabcut_demo()
    
    # Performans karşılaştırması
    print("\nSegmentation Techniques Comparison:")
    print("-" * 40)
    print("1. GrabCut: En iyi ön plan/arka plan ayrımı")
    print("2. K-Means: Renk tabanlı hızlı segmentasyon")
    print("3. Watershed: Birbirine yakın nesneleri ayırma")
    print("4. Otsu: Otomatik eşik değeri belirleme")
    print("5. Adaptive: Değişken aydınlatma koşulları")
    print("6. Region Growing: Homojen bölge segmentasyonu")

if __name__ == "__main__":
    main()