import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, test_size=0.2, class_imbalance=0):
    X = df[["total_orders", "total_spent", "avg_order_value"]]
    y = df["will_order_again"]

    # Özellikleri (X) ölçekle
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Veriyi eğitim ve test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    if class_imbalance == 0:
        return X_train, X_test, y_train, y_test

    if class_imbalance == 1:
        print("Random Oversampling uygulanıyor...")
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        return X_resampled, X_test, y_resampled, y_test

    if class_imbalance == 2:
        print("Class Weight uygulanıyor...")
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f"Sınıf ağırlıkları: {class_weight_dict}")

        global class_weights_dict
        class_weights_dict = class_weight_dict

        return X_train, X_test, y_train, y_test
