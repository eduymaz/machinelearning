import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(summary):
    """
    Dağılım grafiğini çizer.
    
    Args:
        summary (pd.DataFrame): Özet veri seti
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(summary['percentage_less_than_180'], bins=10, kde=True)
    plt.xlabel('% of orders < 180 days apart')
    plt.title('Dağılım: percentage_less_than_180')
    plt.show()

def print_summary_statistics(summary):
    """
    Özet istatistikleri yazdırır.
    
    Args:
        summary (pd.DataFrame): Özet veri seti
    """
    print("\nÖzet İstatistikler:")
    print(f"Toplam müşteri sayısı: {len(summary)}")
    print(f"Riskli müşteri sayısı (target=1): {summary['target'].sum()}")
    print("\nRisk gruplarına göre dağılım:")
    print(summary['risk_group'].value_counts())
    
    print("\nİlk 5 müşteri:")
    print(summary.head()) 