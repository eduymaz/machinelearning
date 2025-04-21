import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def load_data():
    df = pd.read_csv('../data/raw/candidates.csv')
    X = df[['tecrube_yili', 'teknik_puan']]
    y = df['etiket']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def test_kernels(X, y):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    results = []
    
    for kernel in kernels:
        model = SVC(kernel=kernel)
        model.fit(X, y)
    
        y_pred = model.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        
        results.append({
            'kernel': kernel,
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score']
        })
        
        
        plot_decision_boundary(X, y, model, kernel)
    
    return pd.DataFrame(results)

def plot_decision_boundary(X, y, model, kernel):
    
    plt.figure(figsize=(10, 6))
    
    # Veri noktalarını çiz
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    
    # Karar sınırını çiz
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                        np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    plt.xlabel('Tecrübe Yılı (Ölçeklendirilmiş)')
    plt.ylabel('Teknik Puan (Ölçeklendirilmiş)')
    plt.title(f'SVM Karar Sınırı - {kernel} Kernel')
    plt.savefig(f'../data/processed/decision_boundary_{kernel}.png')
    plt.close()

def main():
    # Veriyi yükle
    X, y, scaler = load_data()
    
    # Kernel'ları test et
    results = test_kernels(X, y)
    
    # Sonuçları göster
    print("\nKernel Performans Karşılaştırması:")
    print(results)
    
    # Sonuçları kaydet
    results.to_csv('../data/processed/kernel_results.csv', index=False)
    print("\nSonuçlar kaydedildi: data/processed/kernel_results.csv")

if __name__ == "__main__":
    main() 