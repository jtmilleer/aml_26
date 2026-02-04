import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm


df = pd.read_csv('Auto.csv')

# gather data into numpy arrays
disp_data = df[['displacement']].to_numpy() 
cylinder_data = df[['cylinders']].to_numpy()
weight_data = df[['weight']].to_numpy()
accel_data = df[['acceleration']].to_numpy()
mpg_data = df[['mpg']].to_numpy()

# split data into different training and testing sets
# first split gets displacement and mpg data, the rest only split one dataset
disp_train, disp_test, mpg_train, mpg_test = train_test_split(disp_data, mpg_data, test_size=0.4 )

cylinder_train, cylinder_test, _, _ = train_test_split(cylinder_data, mpg_data, test_size=0.4 )
weight_train, weight_test, _, _ = train_test_split(weight_data, mpg_data, test_size=0.4 )
accel_train, accel_test, _, _ = train_test_split(accel_data, mpg_data, test_size=0.4 )

# create and train models for each feature vs mpg
disp_mpg_model = LinearRegression()
disp_mpg_model.fit(disp_train, mpg_train)

cylinder_mpg_model = LinearRegression()
cylinder_mpg_model.fit(cylinder_train, mpg_train)

weight_mpg_model = LinearRegression()
weight_mpg_model.fit(weight_train, mpg_train)

accel_mpg_model = LinearRegression()
accel_mpg_model.fit(accel_train, mpg_train)

# print slopes and intercepts
print("Displacement vs MPG")
print("slope:", disp_mpg_model.coef_[0])
print("intercept:", disp_mpg_model.intercept_)
print("r^2:", disp_mpg_model.score(disp_test, mpg_test))

print("\nCylinders vs MPG")
print("slope:", cylinder_mpg_model.coef_[0])
print("intercept:", cylinder_mpg_model.intercept_)
print("r^2:", cylinder_mpg_model.score(cylinder_test, mpg_test))

print("\nWeight vs MPG")
print("slope:", weight_mpg_model.coef_[0])
print("intercept:", weight_mpg_model.intercept_)
print("r^2:", weight_mpg_model.score(weight_test, mpg_test))

print("\nAcceleration vs MPG")
print("slope:", accel_mpg_model.coef_[0])
print("intercept:", accel_mpg_model.intercept_)
print("r^2:", accel_mpg_model.score(accel_test, mpg_test))


print("\nMSE Scores:")
# Displacement vs MPG
y_pred_mpg = disp_mpg_model.predict(disp_test)
mse_mpg = mean_squared_error(mpg_test, y_pred_mpg)

# Cylinders vs MPG
y_pred_cylinder = cylinder_mpg_model.predict(cylinder_test)
mse_cylinder = mean_squared_error(mpg_test, y_pred_cylinder)

# Weight vs MPG
y_pred_weight = weight_mpg_model.predict(weight_test)
mse_weight = mean_squared_error(mpg_test, y_pred_weight)

# Acceleration vs MPG
y_pred_accel = accel_mpg_model.predict(accel_test)
mse_accel = mean_squared_error(mpg_test, y_pred_accel)

# Store and rank MSE scores
mse_scores = [
    ("Displacement vs MPG", mse_mpg),
    ("Cylinders vs MPG", mse_cylinder),
    ("Weight vs MPG", mse_weight),
    ("Acceleration vs MPG", mse_accel)
]


mse_scores.sort(key=lambda x: x[1])

print("\nRanked MSE Scores (best to worst):")
for rank, (model_name, mse) in enumerate(mse_scores, 1):
    print(f"{rank}. {model_name} MSE: {mse}")

features = [
    ("Displacement", disp_train),
    ("Cylinders", cylinder_train),
    ("Weight", weight_train),
    ("Acceleration", accel_train)
]

print("\n--- Statistical Significance (t-tests) ---")
for name, data in features:
    # Add constant for intercept
    X = sm.add_constant(data)
    model = sm.OLS(mpg_train, X).fit()
    
    t_stat = model.tvalues[1]  # index 1 is the feature, index 0 is the constant
    p_val = model.pvalues[1]
    
    print(f"{name:12} | t-stat: {t_stat:8.4f} | p-value: {p_val:.4e}")
# Plot all regressions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Displacement vs MPG
axes[0, 0].scatter(disp_test, mpg_test, color='blue', alpha=0.5)
axes[0, 0].plot(disp_test, y_pred_mpg, color='red', linewidth=2)
axes[0, 0].set_xlabel('Displacement')
axes[0, 0].set_ylabel('MPG')
axes[0, 0].set_title('Displacement vs MPG')
axes[0, 0].grid(True)

# Plot 2: Cylinders vs MPG
axes[0, 1].scatter(cylinder_test, mpg_test, color='green', alpha=0.5)
axes[0, 1].plot(cylinder_test, y_pred_cylinder, color='red', linewidth=2)
axes[0, 1].set_xlabel('Cylinders')
axes[0, 1].set_ylabel('MPG')
axes[0, 1].set_title('Cylinders vs MPG')
axes[0, 1].grid(True)

# Plot 3: Weight vs MPG
axes[1, 0].scatter(weight_test, mpg_test, color='orange', alpha=0.5)
axes[1, 0].plot(weight_test, y_pred_weight, color='red', linewidth=2)
axes[1, 0].set_xlabel('Weight')
axes[1, 0].set_ylabel('MPG')
axes[1, 0].set_title('Weight vs MPG')
axes[1, 0].grid(True)

# Plot 4: Acceleration vs MPG
axes[1, 1].scatter(accel_test, mpg_test, color='purple', alpha=0.5)
axes[1, 1].plot(accel_test, y_pred_accel, color='red', linewidth=2)
axes[1, 1].set_xlabel('Acceleration')
axes[1, 1].set_ylabel('MPG')
axes[1, 1].set_title('Acceleration vs MPG')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()