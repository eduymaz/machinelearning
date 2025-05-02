import os
import sys
import subprocess
import time
from pathlib import Path

def check_file_exists(filename):
    """Dosyanın varlığını kontrol et"""
    if not os.path.exists(filename):
        print(f"HATA: {filename} bulunamadı!")
        return False
    return True

def install_requirements():
    """Gerekli kütüphaneleri yükle"""
    print("\nGerekli kütüphaneler yükleniyor...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Kütüphaneler başarıyla yüklendi!")
        return True
    except subprocess.CalledProcessError:
        print("HATA: Kütüphaneler yüklenirken bir hata oluştu!")
        return False

def check_python_files():
    """Python dosyalarının varlığını kontrol et"""
    required_files = [
        "data_loading.py",
        "data_preprocessing.py",
        "feature_engineering.py",
        "models.py",
        "api.py",
        "run_api.py"
    ]
    
    print("\nDosya kontrolleri yapılıyor...")
    all_files_exist = True
    for file in required_files:
        if check_file_exists(file):
            print(f"✓ {file} bulundu")
        else:
            all_files_exist = False
    
    return all_files_exist

def check_data_file():
    """Veri dosyasının varlığını kontrol et"""
    print("\nVeri dosyası kontrol ediliyor...")
    if check_file_exists("Northwind.csv"):
        print("✓ Northwind.csv bulundu")
        return True
    return False

def run_api():
    """API'yi başlat"""
    print("\nAPI başlatılıyor...")
    try:
        # API'yi başlat
        api_process = subprocess.Popen([sys.executable, "run_api.py"])
        
        # API'nin başlaması için biraz bekle
        time.sleep(3)
        
        print("\nAPI başarıyla başlatıldı!")
        print("\nErişilebilir endpoint'ler:")
        print("- Swagger UI: http://localhost:8880/docs")
        print("- ReDoc: http://localhost:8880/redoc")
        print("- Ana sayfa: http://localhost:8880/")
        print("- Tüm müşteriler: http://localhost:8880/customers")
        print("- Analiz sonuçları: http://localhost:8880/analysis")
        
        print("\nAPI'yi durdurmak için Ctrl+C tuşlarına basın...")
        
        # API çalışırken bekle
        api_process.wait()
        
    except KeyboardInterrupt:
        print("\nAPI durduruluyor...")
        api_process.terminate()
        print("API durduruldu.")
    except Exception as e:
        print(f"\nHATA: API başlatılırken bir hata oluştu: {str(e)}")
        return False

def main():
    """Ana program akışı"""
    print("=== Müşteri Risk Analizi API Kurulum ve Çalıştırma ===")
    
    # Çalışma dizinini kontrol et
    current_dir = Path.cwd()
    print(f"\nÇalışma dizini: {current_dir}")
    
    # Dosya kontrollerini yap
    if not check_python_files():
        print("\nHATA: Bazı gerekli dosyalar eksik!")
        return
    
    # Veri dosyasını kontrol et
    if not check_data_file():
        print("\nHATA: Veri dosyası eksik!")
        return
    
    # Gerekli kütüphaneleri yükle
    if not install_requirements():
        print("\nHATA: Kütüphaneler yüklenemedi!")
        return
    
    # API'yi başlat
    run_api()

if __name__ == "__main__":
    main() 