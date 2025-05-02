import pandas as pd

def load_data(file_path='Northwind.csv'):
    """
    Veri setini yükler ve temel dönüşümleri yapar.
    
    Args:
        file_path (str): Veri setinin yolu
        
    Returns:
        pd.DataFrame: Yüklenen veri seti
    """
    df = pd.read_csv(file_path)
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df 