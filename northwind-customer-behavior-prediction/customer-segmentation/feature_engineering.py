import pandas as pd

def create_summary_features(df_unique):
    """
    Özet özellikler oluşturur.
    
    Args:
        df_unique (pd.DataFrame): Ön işlenmiş veri seti
        
    Returns:
        pd.DataFrame: Özet özellikler
    """
    # Özet tablo oluştur
    summary = df_unique.groupby('customer_id').agg(
        total_orders=('order_date', 'count'),
        orders_less_than_180=('less_than_180', 'sum')
    ).reset_index()
    
    # Yüzde hesapla
    summary['percentage_less_than_180'] = 100 * summary['orders_less_than_180'] / summary['total_orders']
    summary = summary.round(2)
    
    # Hedef değişken oluştur
    summary['target'] = summary['percentage_less_than_180'].apply(lambda x: 1 if x >= 60 else 0)
    
    # Risk grupları oluştur
    summary['risk_group'] = pd.qcut(summary['percentage_less_than_180'], 
                                  q=4, 
                                  labels=['low', 'mid-low', 'mid-high', 'high'])
    
    return summary 