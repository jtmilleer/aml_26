import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('HW03_data.csv')

# Prepare features and target
y = df[['Y']].to_numpy()
X = df[['X1', 'X2', 'X3']].to_numpy()

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)

# Verify the split percentages
total_samples = len(X)
train_percent = len(X_train) / total_samples * 100
val_percent = len(X_val) / total_samples * 100
test_percent = len(X_test) / total_samples * 100

print(f'Training set: {len(X_train)} samples ({train_percent:.2f}%)')
print(f'Validation set: {len(X_val)} samples ({val_percent:.2f}%)')
print(f'Test set: {len(X_test)} samples ({test_percent:.2f}%)')

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


reg = KNeighborsRegressor(n_neighbors=9)
reg.fit(X_train_scaled, y_train)




