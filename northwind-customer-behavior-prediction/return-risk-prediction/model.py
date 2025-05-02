import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class ReturnRiskModel(nn.Module):
    def __init__(self, input_size):
        super(ReturnRiskModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.relu(self.layer3(x))
        x = torch.sigmoid(self.output(x))
        return x

def prepare_data(df, test_size=0.2, random_state=42):
    """
    Veriyi model için hazırlar.
    
    Args:
        df (pd.DataFrame): Veri seti
        test_size (float): Test seti oranı
        random_state (int): Rastgele durum
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Özellikler ve hedef değişken
    features = ['unit_price', 'quantity', 'discount', 'spending']
    X = df[features].values
    y = df['return_risk'].values
    
    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Ölçeklendirme
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train, input_size, epochs=50, batch_size=32):
    """
    Modeli eğitir.
    
    Args:
        X_train (np.array): Eğitim verisi
        y_train (np.array): Eğitim etiketleri
        input_size (int): Giriş boyutu
        epochs (int): Epoch sayısı
        batch_size (int): Batch boyutu
        
    Returns:
        ReturnRiskModel: Eğitilmiş model
    """
    # Veriyi PyTorch formatına dönüştür
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    
    # DataLoader oluştur
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model oluştur
    model = ReturnRiskModel(input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Eğitim
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    
    return model

def predict(model, X):
    """
    Tahmin yapar.
    
    Args:
        model (ReturnRiskModel): Eğitilmiş model
        X (np.array): Tahmin verisi
        
    Returns:
        np.array: Tahminler
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor)
        return predictions.numpy() 