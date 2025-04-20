import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_candidate_data(num_candidates=200):

    candidates = []
    
    for i in range(num_candidates):
        experience = round(random.uniform(0, 10), 1)
        
        technical_score = round(random.uniform(0, 100), 1)
        
        if experience < 2 and technical_score < 60:
            label = 1  # İşe alınmadı
        else:
            label = 0  # İşe alındı
            
        candidate = {
            'aday_id': i + 1,
            'tecrube_yili': experience,
            'teknik_puan': technical_score,
            'etiket': label,
            'basvuru_tarihi': datetime.now() - timedelta(days=random.randint(1, 365))
        }
        
        candidates.append(candidate)
    
    return pd.DataFrame(candidates)

def save_data(df, filename):
 
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df.to_csv(filename, index=False)
    print(f"Veri {filename} dosyasına kaydedildi.")
    
    print("\nKaydedilen dosyanın ilk 5 satırı:")
    print(pd.read_csv(filename).head())

def main():
    candidates_df = generate_candidate_data()
    
    save_data(candidates_df, './data/raw/candidates.csv')
    
    print("\nVeri İstatistikleri:")
    print(f"Toplam Aday Sayısı: {len(candidates_df)}")
    print(f"İşe Alınan Aday Sayısı: {len(candidates_df[candidates_df['etiket'] == 0])}")
    print(f"İşe Alınmayan Aday Sayısı: {len(candidates_df[candidates_df['etiket'] == 1])}")
    print("\nTecrübe Yılı İstatistikleri:")
    print(candidates_df['tecrube_yili'].describe())
    print("\nTeknik Puan İstatistikleri:")
    print(candidates_df['teknik_puan'].describe())

if __name__ == "__main__":
    main() 