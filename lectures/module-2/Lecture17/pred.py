"""
YOLO8 Turkish License Plate Detection - Prediction Script
Bu script eğitilmiş YOLO8 modeli ile plaka tespiti yapar.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse
from pathlib import Path
import time

class LicensePlateDetector:
    """Türkçe plaka tespit sınıfı"""
    
    def __init__(self, model_path="runs/detect/turkish_license_plate/weights/best.pt"):
        """
        Detector'ı başlat
        
        Args:
            model_path (str): Eğitilmiş model dosyasının yolu
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Modeli yükle"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"✅ Model yüklendi: {self.model_path}")
            else:
                print(f"⚠️  Model dosyası bulunamadı: {self.model_path}")
                print("Varsayılan YOLOv8n modeli kullanılacak...")
                self.model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            self.model = YOLO("yolov8n.pt")
    
    def detect_single_image(self, image_path, save_path=None, conf_threshold=0.25):
        """
        Tek bir görüntüde plaka tespiti yap
        
        Args:
            image_path (str): Görüntü dosyasının yolu
            save_path (str): Sonucun kaydedileceği yol
            conf_threshold (float): Güven eşiği
        
        Returns:
            tuple: (detection_results, annotated_image)
        """
        if not os.path.exists(image_path):
            print(f"❌ Görüntü dosyası bulunamadı: {image_path}")
            return None, None
        
        try:
            # Görüntüyü yükle
            image = cv2.imread(image_path)
            original_image = image.copy()
            
            # Tahmin yap
            results = self.model(image, conf=conf_threshold)
            
            # Sonuçları işle
            detection_info = []
            annotated_image = image.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Bounding box koordinatları
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Güven skoru ve sınıf
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Tespit bilgilerini kaydet
                        detection_info.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.model.names[class_id] if hasattr(self.model, 'names') else 'license_plate'
                        })
                        
                        # Bounding box çiz
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Label ekle
                        label = f"Plaka {confidence:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(annotated_image, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Sonucu kaydet
            if save_path:
                cv2.imwrite(save_path, annotated_image)
                print(f"✅ Sonuç kaydedildi: {save_path}")
            
            # Tespit sayısını yazdır
            print(f"🎯 {len(detection_info)} plaka tespit edildi")
            for i, detection in enumerate(detection_info):
                bbox = detection['bbox']
                conf = detection['confidence']
                print(f"   Plaka {i+1}: Konum=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}), Güven={conf:.3f}")
            
            return detection_info, annotated_image
            
        except Exception as e:
            print(f"❌ Tespit hatası: {e}")
            return None, None
    
    def detect_video(self, video_path, output_path=None, conf_threshold=0.25, show_video=True):
        """
        Video üzerinde plaka tespiti yap
        
        Args:
            video_path (str): Video dosyasının yolu
            output_path (str): Çıktı video dosyasının yolu
            conf_threshold (float): Güven eşiği
            show_video (bool): Videoyu gerçek zamanlı göster
        """
        if not os.path.exists(video_path):
            print(f"❌ Video dosyası bulunamadı: {video_path}")
            return
        
        try:
            # Video yakalayıcısını başlat
            cap = cv2.VideoCapture(video_path)
            
            # Video özellikleri
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"📹 Video bilgileri: {width}x{height}, {fps} FPS, {total_frames} frame")
            
            # Video yazıcısını başlat
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = None
            if output_path:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Her frame'de tespit yap
                results = self.model(frame, conf=conf_threshold, verbose=False)
                
                # Sonuçları çiz
                annotated_frame = frame.copy()
                detection_count = 0
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            detection_count += 1
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = box.conf[0].cpu().numpy()
                            
                            # Bounding box ve label çiz
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Plaka {confidence:.2f}"
                            cv2.putText(annotated_frame, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Frame bilgilerini ekle
                info_text = f"Frame: {frame_count}/{total_frames} | Tespit: {detection_count}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Videoyu göster
                if show_video:
                    cv2.imshow('License Plate Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Frame'i kaydet
                if out:
                    out.write(annotated_frame)
                
                # İlerleme göster
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps_current = frame_count / elapsed_time
                    print(f"İşlenen frame: {frame_count}/{total_frames} ({fps_current:.1f} FPS)")
            
            # Kaynakları serbest bırak
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            print(f"✅ Video işleme tamamlandı: {total_time:.2f} saniye")
            if output_path:
                print(f"✅ Çıktı kaydedildi: {output_path}")
                
        except Exception as e:
            print(f"❌ Video işleme hatası: {e}")
    
    def detect_webcam(self, conf_threshold=0.25):
        """
        Webcam'den gerçek zamanlı plaka tespiti
        
        Args:
            conf_threshold (float): Güven eşiği
        """
        try:
            # Webcam'i başlat
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Webcam açılamadı!")
                return
            
            print("📷 Webcam başlatıldı. Çıkmak için 'q' tuşuna basın.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Tespit yap
                results = self.model(frame, conf=conf_threshold, verbose=False)
                
                # Sonuçları çiz
                detection_count = 0
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            detection_count += 1
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = box.conf[0].cpu().numpy()
                            
                            # Bounding box ve label çiz
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Plaka {confidence:.2f}"
                            cv2.putText(frame, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Bilgi metni ekle
                info_text = f"Tespit edilen plaka sayisi: {detection_count}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Cikmak icin 'q' tusuna basin", (10, frame.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Frame'i göster
                cv2.imshow('Webcam License Plate Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Webcam kapatıldı")
            
        except Exception as e:
            print(f"❌ Webcam hatası: {e}")

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='YOLO8 Turkish License Plate Detection')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'webcam'], 
                       default='image', help='Tespit modu')
    parser.add_argument('--source', type=str, help='Kaynak dosya yolu (görüntü veya video)')
    parser.add_argument('--model', type=str, 
                       default='runs/detect/turkish_license_plate/weights/best.pt',
                       help='Model dosya yolu')
    parser.add_argument('--output', type=str, help='Çıktı dosya yolu')
    parser.add_argument('--conf', type=float, default=0.25, help='Güven eşiği')
    parser.add_argument('--show', action='store_true', help='Sonuçları göster')
    
    args = parser.parse_args()
    
    print("🚗 YOLO8 Türkçe Plaka Tespit Sistemi")
    print("=" * 40)
    
    # Detector'ı başlat
    detector = LicensePlateDetector(args.model)
    
    try:
        if args.mode == 'image':
            if not args.source:
                print("❌ Görüntü modu için --source parametresi gerekli!")
                return
            
            # Çıktı yolu belirle
            output_path = args.output
            if not output_path:
                input_path = Path(args.source)
                output_path = str(input_path.parent / f"{input_path.stem}_detected{input_path.suffix}")
            
            # Tespit yap
            detections, annotated_img = detector.detect_single_image(
                args.source, output_path, args.conf
            )
            
            # Sonuçları göster
            if args.show and annotated_img is not None:
                cv2.imshow('Detection Result', annotated_img)
                print("Görüntüyü kapatmak için herhangi bir tuşa basın...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        elif args.mode == 'video':
            if not args.source:
                print("❌ Video modu için --source parametresi gerekli!")
                return
            
            # Çıktı yolu belirle
            output_path = args.output
            if not output_path:
                input_path = Path(args.source)
                output_path = str(input_path.parent / f"{input_path.stem}_detected{input_path.suffix}")
            
            # Video tespiti yap
            detector.detect_video(args.source, output_path, args.conf, args.show)
        
        elif args.mode == 'webcam':
            # Webcam tespiti yap
            detector.detect_webcam(args.conf)
        
        print("✅ İşlem tamamlandı!")
        
    except KeyboardInterrupt:
        print("\n⏸️  İşlem kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"❌ Genel hata: {e}")

if __name__ == "__main__":
    main()