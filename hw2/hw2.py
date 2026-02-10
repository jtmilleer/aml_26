import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# ai was used to write this code


# Load dataset
df = pd.read_csv('Auto.csv')

# Define features and target
features_list = ['displacement', 'cylinders', 'weight', 'acceleration']
X = df[features_list]
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


mse_scores = []


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten() 
colors = ['blue', 'green', 'orange', 'purple']

print("--- Individual Model Results ---")
for i, feature in enumerate(features_list):
   
    model = LinearRegression()
    model.fit(X_train[[feature]], y_train)
    

    y_pred = model.predict(X_test[[feature]])
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append((feature, mse))
    
    print(f"{feature.capitalize()}: R^2 = {model.score(X_test[[feature]], y_test):.4f}, MSE = {mse:.4f}")

   
    sort_idx = X_test[feature].argsort()
    x_sort = X_test[feature].iloc[sort_idx]
    y_sort_pred = y_pred[sort_idx]

    axes[i].scatter(X_test[feature], y_test, color=colors[i], alpha=0.5, label='Actual')
    axes[i].plot(x_sort, y_sort_pred, color='red', linewidth=2, label='Regression Line')
    axes[i].set_xlabel(feature.capitalize())
    axes[i].set_ylabel('MPG')
    axes[i].set_title(f"{feature.capitalize()} vs MPG")
    axes[i].legend()
    axes[i].grid(True)


print("\n--- Statistical Significance (t-tests) ---")
for feature in features_list:
    X_sm = sm.add_constant(X_train[feature])
    model_sm = sm.OLS(y_train, X_sm).fit()
    
    
    t_stat = model_sm.tvalues.iloc[1]
    p_val = model_sm.pvalues.iloc[1]
    print(f"{feature:12} | t-stat: {t_stat:8.4f} | p-value: {p_val:.4e}")

# Adjust layout and show the plots
plt.tight_layout()
plt.show()