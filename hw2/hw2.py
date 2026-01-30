import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Auto.csv')

disp_data = df[['displacement']].to_numpy() 
cylinder_data = df[['cylinders']].to_numpy()
weight_data = df[['weight']].to_numpy()
accel_data = df[['acceleration']].to_numpy()

mpg_data = df[['mpg']].to_numpy()

disp_train, disp_test, mpg_train, mpg_test = train_test_split(disp_data, mpg_data, test_size=0.4 )

cylinder_train, cylinder_test, _, _ = train_test_split(cylinder_data, mpg_data, test_size=0.4 )
weight_train, weight_test, _, _ = train_test_split(weight_data, mpg_data, test_size=0.4 )
accel_train, accel_test, _, _ = train_test_split(accel_data, mpg_data, test_size=0.4 )

disp_mpg_model = LinearRegression()
disp_mpg_model.fit(disp_train, mpg_train)

cylinder_mpg_model = LinearRegression()
cylinder_mpg_model.fit(cylinder_train, mpg_train)

weight_mpg_model = LinearRegression()
weight_mpg_model.fit(weight_train, mpg_train)

accel_mpg_model = LinearRegression()
accel_mpg_model.fit(accel_train, mpg_train)

print("Displacement vs MPG")
print("slope:", disp_mpg_model.coef_[0])
print("intercept:", disp_mpg_model.intercept_)

print("\nCylinders vs MPG")
print("slope:", cylinder_mpg_model.coef_[0])
print("intercept:", cylinder_mpg_model.intercept_)

print("\nWeight vs MPG")
print("slope:", weight_mpg_model.coef_[0])
print("intercept:", weight_mpg_model.intercept_)

print("\nAcceleration vs MPG")
print("slope:", accel_mpg_model.coef_[0])
print("intercept:", accel_mpg_model.intercept_)

#print("\nR^2 Scores:")
#print("Displacement vs MPG:", disp_mpg_model.score(disp_test, mpg_test))
#print("Cylinders vs MPG:", cylinder_mpg_model.score(cylinder_test, mpg_test))
#print("Weight vs MPG:", weight_mpg_model.score(weight_test, mpg_test))
#print("Acceleration vs MPG:", accel_mpg_model.score(accel_test, mpg_test))

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