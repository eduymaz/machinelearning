from data_loading import load_data
from data_preprocessing import preprocess_data
from feature_engineering import create_summary_features
from analysis import plot_distribution, print_summary_statistics

def main():
    """
    Ana program akışı
    """
    # Veri yükleme
    print("Veri yükleniyor...")
    df = load_data()
    
    # Veri ön işleme
    print("\nVeri ön işleniyor...")
    df_unique = preprocess_data(df)
    
    # Özellik mühendisliği
    print("\nÖzellikler oluşturuluyor...")
    summary = create_summary_features(df_unique)
    
    # Analiz ve görselleştirme
    print("\nAnaliz yapılıyor...")
    print_summary_statistics(summary)
    plot_distribution(summary)

if __name__ == "__main__":
    main() 