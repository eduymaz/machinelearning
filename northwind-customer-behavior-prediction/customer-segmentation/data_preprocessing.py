import pandas as pd

def preprocess_data(df):
    """
    Veri setini ön işler ve gerekli dönüşümleri yapar.
    
    Args:
        df (pd.DataFrame): Ham veri seti
        
    Returns:
        pd.DataFrame: Ön işlenmiş veri seti
    """
    # Aynı müşteri ve aynı tarihteki siparişleri tekilleştir
    df_unique = df.drop_duplicates(subset=['customer_id', 'order_date'])
    
    # Sıralama
    df_unique = df_unique.sort_values(['customer_id', 'order_date']).reset_index(drop=True)
    
    # Bir sonraki siparişin tarihini bul
    df_unique['next_order_date'] = df_unique.groupby('customer_id')['order_date'].shift(-1)
    
    # Gün farkını hesapla
    df_unique['days_diff'] = (df_unique['next_order_date'] - df_unique['order_date']).dt.days
    
    # 180 günden az olanları işaretle
    df_unique['less_than_180'] = df_unique['days_diff'] < 180
    
    return df_unique 