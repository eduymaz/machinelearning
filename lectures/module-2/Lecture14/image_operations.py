import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageOperations:
    def __init__(self, image_path=None):
        if image_path:
            self.image = cv2.imread(image_path)
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            # Create a sample image if no path provided
            self.image = self.create_sample_image()
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def create_sample_image(self):
        """Create a sample image for demonstration"""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)
        cv2.circle(img, (200, 200), 50, (0, 255, 0), -1)
        cv2.line(img, (0, 0), (300, 300), (0, 0, 255), 3)
        return img
    
    def resize_image(self, width, height):
        """Resize image to specified dimensions"""
        resized = cv2.resize(self.image, (width, height))
        return resized
    
    def rotate_image(self, angle):
        """Rotate image by specified angle"""
        height, width = self.image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.image, rotation_matrix, (width, height))
        return rotated
    
    def flip_image(self, flip_code):
        """Flip image (0: vertical, 1: horizontal, -1: both)"""
        flipped = cv2.flip(self.image, flip_code)
        return flipped
    
    def blur_image(self, kernel_size=15):
        """Apply Gaussian blur to image"""
        blurred = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        return blurred
    
    def histogram_equalization(self):
        """Apply histogram equalization for contrast enhancement"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return equalized
    
    def noise_reduction(self):
        """Apply noise reduction using Non-local Means Denoising"""
        denoised = cv2.fastNlMeansDenoisingColored(self.image, None, 10, 10, 7, 21)
        return denoised
    
    def edge_detection(self):
        """Detect edges using Canny edge detector"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges
    
    def threshold_image(self, threshold_value=127):
        """Apply binary thresholding"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return thresh
    
    def crop_image(self, x, y, width, height):
        """Crop image to specified region"""
        cropped = self.image[y:y+height, x:x+width]
        return cropped
    
    def convert_color_space(self, color_space='GRAY'):
        """Convert image to different color spaces"""
        if color_space == 'GRAY':
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        elif color_space == 'HSV':
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LAB':
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        else:
            return self.image
    
    def display_results(self):
        """Display all operations results"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('Computer Vision Operations', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Resize
        resized = cv2.cvtColor(self.resize_image(150, 150), cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(resized)
        axes[0, 1].set_title('Resized')
        axes[0, 1].axis('off')
        
        # Rotate
        rotated = cv2.cvtColor(self.rotate_image(45), cv2.COLOR_BGR2RGB)
        axes[0, 2].imshow(rotated)
        axes[0, 2].set_title('Rotated 45°')
        axes[0, 2].axis('off')
        
        # Flip
        flipped = cv2.cvtColor(self.flip_image(1), cv2.COLOR_BGR2RGB)
        axes[0, 3].imshow(flipped)
        axes[0, 3].set_title('Flipped Horizontal')
        axes[0, 3].axis('off')
        
        # Blur
        blurred = cv2.cvtColor(self.blur_image(), cv2.COLOR_BGR2RGB)
        axes[1, 0].imshow(blurred)
        axes[1, 0].set_title('Blurred')
        axes[1, 0].axis('off')
        
        # Histogram Equalization
        hist_eq = self.histogram_equalization()
        axes[1, 1].imshow(hist_eq, cmap='gray')
        axes[1, 1].set_title('Histogram Equalized')
        axes[1, 1].axis('off')
        
        # Noise Reduction
        denoised = cv2.cvtColor(self.noise_reduction(), cv2.COLOR_BGR2RGB)
        axes[1, 2].imshow(denoised)
        axes[1, 2].set_title('Noise Reduced')
        axes[1, 2].axis('off')
        
        # Edge Detection
        edges = self.edge_detection()
        axes[1, 3].imshow(edges, cmap='gray')
        axes[1, 3].set_title('Edge Detection')
        axes[1, 3].axis('off')
        
        # Thresholding
        thresh = self.threshold_image()
        axes[2, 0].imshow(thresh, cmap='gray')
        axes[2, 0].set_title('Thresholded')
        axes[2, 0].axis('off')
        
        # Crop
        cropped = cv2.cvtColor(self.crop_image(50, 50, 100, 100), cv2.COLOR_BGR2RGB)
        axes[2, 1].imshow(cropped)
        axes[2, 1].set_title('Cropped')
        axes[2, 1].axis('off')
        
        # Grayscale
        gray = self.convert_color_space('GRAY')
        axes[2, 2].imshow(gray, cmap='gray')
        axes[2, 2].set_title('Grayscale')
        axes[2, 2].axis('off')
        
        # HSV
        hsv = self.convert_color_space('HSV')
        axes[2, 3].imshow(hsv)
        axes[2, 3].set_title('HSV Color Space')
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create image operations instance with sample image
    img_ops = ImageOperations()
    
    # Display all operations
    img_ops.display_results()
    
    # Individual operation examples
    print("Image Operations Examples:")
    print("1. Original image shape:", img_ops.image.shape)
    print("2. Resized image shape:", img_ops.resize_image(100, 100).shape)
    print("3. Edge detection completed")
    print("4. All operations demonstrated successfully!")
