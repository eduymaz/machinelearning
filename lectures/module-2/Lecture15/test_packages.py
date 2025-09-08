#!/usr/bin/env python3
"""
Temel paket testi - CNN demo çalıştırmadan önce bu dosyayı test edin
"""

print("📦 Paket Kontrolü Başlıyor...")
print("=" * 30)

# Temel paketleri kontrol et
packages = {
    'numpy': 'Sayısal hesaplamalar için',
    'matplotlib': 'Görselleştirme için', 
    'cv2': 'OpenCV - Görüntü işleme için',
    'sklearn': 'Makine öğrenmesi için'
}

missing_packages = []

for package, description in packages.items():
    try:
        __import__(package)
        print(f"✅ {package:<12} - {description}")
    except ImportError:
        print(f"❌ {package:<12} - {description} - EKSİK!")
        missing_packages.append(package)

# İsteğe bağlı paketler
optional_packages = {
    'tensorflow': 'Tam CNN demo için',
    'seaborn': 'Gelişmiş görselleştirme için'
}

print("\n📋 İsteğe Bağlı Paketler:")
for package, description in optional_packages.items():
    try:
        __import__(package)
        print(f"✅ {package:<12} - {description}")
    except ImportError:
        print(f"⚠️  {package:<12} - {description} - Kurulu değil")

# Sonuç
print("\n" + "=" * 50)
if not missing_packages:
    print("🎉 Tüm temel paketler mevcut! Demo çalıştırılabilir.")
    print("▶️  Şimdi şu komutu çalıştırabilirsiniz:")
    print("   python run_demo.py")
else:
    print("❌ Eksik paketler var! Şu komutu çalıştırın:")
    pip_install = "pip install " + " ".join([
        "opencv-python" if pkg == "cv2" else 
        "scikit-learn" if pkg == "sklearn" else pkg 
        for pkg in missing_packages
    ])
    print(f"   {pip_install}")

print("\n💡 TensorFlow için tam demo istiyorsanız:")
print("   pip install tensorflow seaborn")