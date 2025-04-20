import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):

    X = df[['tecrube_yili', 'teknik_puan']]
    y = df['etiket']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):

    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    
    # Klasör yapısını oluşturma
    os.makedirs('./data/processed', exist_ok=True)
    plt.savefig('./data/processed/confusion_matrix.png')
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def plot_decision_boundary(model, X, y, scaler):
  
    # Veriyi ölçeklendirme
    X_scaled = scaler.transform(X)
    
    # Karar sınırını çizme
    plt.figure(figsize=(10, 8))
    
    # Veri noktalarını çizme
    plt.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], 
                color='blue', label='İşe Alındı')
    plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], 
                color='red', label='İşe Alınmadı')
    
    # Karar sınırını çizme
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], 
               alpha=0.5, linestyles=['--', '-', '--'])
    
    plt.xlabel('Tecrübe Yılı (Ölçeklendirilmiş)')
    plt.ylabel('Teknik Puan (Ölçeklendirilmiş)')
    plt.title('SVM Karar Sınırı')
    plt.legend()
    plt.savefig('./data/processed/decision_boundary.png')

def save_model(model, scaler, model_path, scaler_path):
  
    # Klasör yapısını oluşturma
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model {model_path} dosyasına kaydedildi.")
    print(f"Ölçeklendirici {scaler_path} dosyasına kaydedildi.")

def main():
    # Verinin yüklenmesi
    df = load_data('./data/raw/candidates.csv')
    
    # Veriyi ön işleme
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Modeli eğitme
    model = train_model(X_train, y_train)
    
    # Modeli değerlendirme
    evaluate_model(model, X_test, y_test)
    
    # Karar sınırını görselleştir
    plot_decision_boundary(model, df[['tecrube_yili', 'teknik_puan']], 
                          df['etiket'], scaler)
    
    save_model(model, scaler, 
              './data/processed/model.joblib', 
              './data/processed/scaler.joblib')

if __name__ == "__main__":
    main() 