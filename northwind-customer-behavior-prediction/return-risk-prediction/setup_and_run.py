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
        "model.py",
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
        api_process = subprocess.Popen([sys.executable, "run_api.py"])
        time.sleep(3)
        print("\nAPI başarıyla başlatıldı!")
        print("\nErişilebilir endpoint'ler:")
        print("- Swagger UI: http://localhost:8787/docs")
        print("- ReDoc: http://localhost:8787/redoc")
        print("- Ana sayfa: http://localhost:8787/")
        print("- Tahmin yapmak için: http://localhost:8787/predict")
        print("- Model bilgileri için: http://localhost:8787/model-info")
        print("\nAPI'yi durdurmak için Ctrl+C tuşlarına basın...")
        api_process.wait()
    except KeyboardInterrupt:
        print("\nAPI durduruluyor...")
        api_process.terminate()
        print("API durduruldu.")
    except Exception as e:
        print(f"\nHATA: API başlatılırken bir hata oluştu: {str(e)}")
        return False

def main():
    """Ana fonksiyon"""
    print("Sipariş İade Riski Tahmin API'si Kurulum ve Çalıştırma Aracı")
    print("=" * 60)
    
    # Gerekli dosyaları kontrol et
    if not check_python_files():
        print("\nHATA: Bazı gerekli dosyalar eksik!")
        return
    
    if not check_data_file():
        print("\nHATA: Veri dosyası bulunamadı!")
        return
    
    # Kütüphaneleri yükle
    if not install_requirements():
        print("\nHATA: Kütüphaneler yüklenemedi!")
        return
    
    # API'yi başlat
    run_api()

if __name__ == "__main__":
    main() 