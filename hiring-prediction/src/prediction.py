import joblib
import numpy as np
import os

def load_model_and_scaler(model_path, scaler_path):
   
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Ölçeklendirici dosyası bulunamadı: {scaler_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def make_prediction(model, scaler, experience, technical_score):

    data = np.array([[experience, technical_score]])
    
    scaled_data = scaler.transform(data)
    
    prediction = model.predict(scaled_data)[0]
    
    return prediction

def get_prediction_message(prediction):

    if prediction == 0:
        return "Tebrikler! Başvurunuz başarılı olmuştur."
    else:
        return "Üzgünüz, başvurunuz bu sefer başarısız olmuştur."

def main():
    try:
        model, scaler = load_model_and_scaler(
            './data/processed/model.joblib',
            './data/processed/scaler.joblib'
        )
        
        print("Aday Değerlendirme Sistemi")
        print("--------------------------")
        
        try:
            experience = float(input("Tecrübe Yılı (0-10): "))
            technical_score = float(input("Teknik Puan (0-100): "))
            
            if not (0 <= experience <= 10 and 0 <= technical_score <= 100):
                print("Hata: Geçersiz değer aralığı!")
                return
            
            prediction = make_prediction(model, scaler, experience, technical_score)
            
            print("\nDeğerlendirme Sonucu:")
            print(f"Tecrübe Yılı: {experience}")
            print(f"Teknik Puan: {technical_score}")
            print(f"Sonuç: {get_prediction_message(prediction)}")
            
        except ValueError:
            print("Hata: Geçerli sayısal değer giriniz!")
            
    except FileNotFoundError as e:
        print(f"Hata: {e}")
        print("Lütfen önce model eğitimini çalıştırın.")

if __name__ == "__main__":
    main() 