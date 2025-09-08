#!/usr/bin/env python3
"""
CNN Demo Başlatıcı
Bu dosya TensorFlow kurulu olup olmadığını kontrol eder ve uygun demo'yu çalıştırır
"""

print("CNN Demo Başlatılıyor...")
print("=" * 40)

# TensorFlow kontrolü
try:
    import tensorflow as tf
    print("✅ TensorFlow bulundu! Tam CNN demo çalıştırılacak.")
    
    # Ana CNN demo'yu çalıştır
    try:
        from cnn_demo import main
        main()
    except Exception as e:
        print(f"❌ Ana demo hatası: {e}")
        print("🔄 Basit demo'ya geçiliyor...")
        from simple_cnn_demo import simple_main
        simple_main()
        
except ImportError:
    print("⚠️  TensorFlow bulunamadı! Basit CNN kavramları demo çalıştırılacak.")
    print("💡 TensorFlow kurmak için: pip install tensorflow")
    print()
    
    # Basit demo'yu çalıştır
    try:
        from simple_cnn_demo import simple_main
        simple_main()
    except Exception as e:
        print(f"❌ Basit demo hatası: {e}")
        print("Lütfen gerekli paketlerin kurulu olduğundan emin olun:")
        print("pip install opencv-python numpy matplotlib scikit-learn")

print("\n🎉 Demo tamamlandı!")