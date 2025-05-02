import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path='Northwind.csv'):
    """
    Veriyi yükler ve ön işleme yapar.
    
    Args:
        file_path (str): Veri dosyasının yolu
        
    Returns:
        pd.DataFrame: Ön işlenmiş veri seti
    """
    # Veriyi yükle
    df = pd.read_csv(file_path)
    
    # Eksik değerleri doldur
    df['discount'].fillna(0, inplace=True)
    
    # Harcama hesapla
    df['spending'] = df['unit_price'] * df['quantity'] * (1 - df['discount'])
    
    # İade riski hedef değişkenini oluştur
    # Bu örnek için basit bir kural kullanıyoruz:
    # - Yüksek indirimli ürünler
    # - Yüksek miktarlı siparişler
    # - Düşük fiyatlı ürünler
    # daha yüksek iade riski taşır
    df['return_risk'] = (
        (df['discount'] > 0.2).astype(int) * 0.3 +
        (df['quantity'] > df['quantity'].quantile(0.75)).astype(int) * 0.3 +
        (df['unit_price'] < df['unit_price'].quantile(0.25)).astype(int) * 0.4
    )
    
    # Değerleri 0-1 arasına normalize et
    df['return_risk'] = (df['return_risk'] - df['return_risk'].min()) / \
                       (df['return_risk'].max() - df['return_risk'].min())
    
    return df 