import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('HW05_data.csv')
X = df[[f"X{i}" for i in range(1, 31)]].to_numpy()
y_raw = df['Y'].to_numpy()


le = LabelEncoder()
y = le.fit_transform(y_raw)



X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
MAX_K = 30
avg_error_rates = []


for k in range(1, MAX_K + 1):
    fold_errors = []
    
    for train_idx, val_idx in skf.split(X_dev, y_dev):
        X_fold_train, X_fold_val = X_dev[train_idx], X_dev[val_idx]
        y_fold_train, y_fold_val = y_dev[train_idx], y_dev[val_idx]
        
        scaler = StandardScaler()
        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = scaler.transform(X_fold_val)
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_fold_train_scaled, y_fold_train)
        y_pred = knn.predict(X_fold_val_scaled)
        
        fold_errors.append(1 - accuracy_score(y_fold_val, y_pred))
    
    avg_error_rates.append(np.mean(fold_errors))


best_k = np.argmin(avg_error_rates) + 1
print(f"Optimal K identified via CV: {best_k}")


plt.figure(figsize=(10, 6))
plt.plot(range(1, MAX_K + 1), avg_error_rates, color='teal', linestyle='-', marker='o')
plt.title('Optimization: Mean CV Error Rate vs. K Value')
plt.xlabel('K Neighbors')
plt.ylabel('Mean Error Rate (5-Fold)')
plt.grid(True, alpha=0.3)
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
plt.legend()
plt.show()


final_scaler = StandardScaler()
X_dev_scaled = final_scaler.fit_transform(X_dev)
X_test_scaled = final_scaler.transform(X_test)

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_dev_scaled, y_dev)
y_final_pred = final_knn.predict(X_test_scaled)


test_acc = accuracy_score(y_test, y_final_pred)
print(f"Optimal K: {best_k}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Error Rate: {1 - test_acc:.4f}")


fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_final_pred, 
                                        display_labels=le.classes_, 
                                        cmap='Blues', ax=ax, colorbar=False)
plt.title(f'Confusion Matrix (K={best_k})')
plt.show()