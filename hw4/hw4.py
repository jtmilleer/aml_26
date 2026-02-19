import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('HW04_data.csv')

Y = df[['Y']].to_numpy()
X = df[[f"X{i}" for i in range(1, 101)]].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

scalar = StandardScaler()
#fit the scaler on the training data and fit the transform to X_Train
X_train_scaled = scalar.fit_transform(X_train)
#transfrom the test data
X_test_scaled = scalar.transform(X_test)

pca = PCA(n_components=X_train_scaled.shape[1])

X_train_pca = pca.fit_transform(X_train_scaled)

print(f"Explained variance : {pca.explained_variance_ratio_}")

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

m095 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Number of components to explain 95% variance: {m095}")


# reduce dimensions to m095 components
pca1 = PCA(n_components=m095)
X_train_pca1 = pca1.fit_transform(X_train_scaled)
X_test_pca1 = pca1.transform(X_test_scaled)

model = LinearRegression()
model.fit(X_train_pca1, Y_train)

#print(f"MSE training using {m095} components: {mean_squared_error(Y_train, model.predict(X_train_pca1)):.4f}")
#print(f"MSE testing using {m095} components: {mean_squared_error(Y_test, model.predict(X_test_pca1)):.4f}")

mse_train_values = []
mse_test_values = []

for i in range(1, X_train_scaled.shape[1] + 1):
    pca_i = PCA(n_components=i)
    X_train_pca_i = pca_i.fit_transform(X_train_scaled)
    X_test_pca_i = pca_i.transform(X_test_scaled)

    model_i = LinearRegression()
    model_i.fit(X_train_pca_i, Y_train)

    mse_train = mean_squared_error(Y_train, model_i.predict(X_train_pca_i))
    mse_test = mean_squared_error(Y_test, model_i.predict(X_test_pca_i))
    mse_train_values.append(mse_train)
    mse_test_values.append(mse_test)

    #print(f"Components: {i}, MSE Train: {mse_train:.4f}, MSE Test: {mse_test:.4f}")


mse_train_values = np.array(mse_train_values)
mse_test_values = np.array(mse_test_values) 
#print(mse_train_values)

print(f"--------MSE Values for Train and Test Sets as function of M----------")
for i in range(len(mse_train_values)):
    print(f"M: {i+1}, MSE Train: {mse_train_values[i]:.4f}, MSE Test: {mse_test_values[i]:.4f}")

print(f"M: {m095}, MSE Train: {mse_train_values[m095-1]:.4f}, MSE Test: {mse_test_values[m095-1]:.4f}")
print(f"M: 16, MSE Train: {mse_train_values[15]:.4f}, MSE Test: {mse_test_values[15]:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(range(1, X_train_scaled.shape[1] + 1), mse_train_values, label='Train MSE', marker='o')
plt.plot(range(1, X_train_scaled.shape[1] + 1), mse_test_values, label='Test MSE', marker='s')
plt.title('MSE vs Number of PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
