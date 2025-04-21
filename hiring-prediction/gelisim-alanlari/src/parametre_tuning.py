import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Veriyi yükler ve ölçeklendirir."""
    df = pd.read_csv('../data/raw/candidates.csv')
    X = df[['tecrube_yili', 'teknik_puan']]
    y = df['etiket']
    
    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def tune_parameters(X, y):
    """SVM parametrelerini optimize eder."""
    # Parametre grid'i
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1, 10],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # Grid search
    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    return grid_search

def plot_results(grid_search):
    """Sonuçları görselleştirir."""
    results = pd.DataFrame(grid_search.cv_results_)
    
    # En iyi parametreleri göster
    print("\nEn İyi Parametreler:")
    print(grid_search.best_params_)
    print("\nEn İyi Skor:", grid_search.best_score_)
    
    # C ve gamma için ısı haritası
    plt.figure(figsize=(10, 6))
    pivot = results.pivot_table(
        index='param_C',
        columns='param_gamma',
        values='mean_test_score'
    )
    sns.heatmap(pivot, annot=True, cmap='viridis')
    plt.title('C ve Gamma Parametreleri için Doğruluk Skorları')
    plt.savefig('../data/processed/parameter_heatmap.png')
    plt.close()

def main():
    # Veriyi yükle
    X, y, scaler = load_data()
    
    # Parametreleri optimize et
    grid_search = tune_parameters(X, y)
    
    # Sonuçları görselleştir
    plot_results(grid_search)
    
    # En iyi modeli kaydet
    best_model = grid_search.best_estimator_
    import joblib
    joblib.dump(best_model, '../data/processed/best_model.joblib')
    
    # Sonuçları kaydet
    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv('../data/processed/parameter_tuning_results.csv', index=False)
    
    print("\nSonuçlar kaydedildi:")
    print("- data/processed/parameter_tuning_results.csv")
    print("- data/processed/parameter_heatmap.png")
    print("- data/processed/best_model.joblib")

if __name__ == "__main__":
    main() 