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

for rs in [42,24]:

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=rs)

    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=rs)

    # Verify the split percentages
    total_samples = len(X)
    train_percent = len(X_train) / total_samples * 100
    val_percent = len(X_validation) / total_samples * 100
    test_percent = len(X_test) / total_samples * 100

    print(f'Training set: {len(X_train)} samples ({train_percent:.2f}%)')
    print(f'Validation set: {len(X_validation)} samples ({val_percent:.2f}%)')
    print(f'Test set: {len(X_test)} samples ({test_percent:.2f}%)')

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validtion_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    maxR2 = 0
    minK = 0


    results = []

    for i in range(1, 15):
        knn = KNeighborsRegressor(n_neighbors=i)
        knn.fit(X_train_scaled, y_train)
        
        y_validation_predicted = knn.predict(X_validtion_scaled)
            
        # Predict on all sets
        y_train_pred = knn.predict(X_train_scaled)
        y_val_pred = knn.predict(X_validtion_scaled)
        y_test_pred = knn.predict(X_test_scaled)
        
        # Calculate R^2 scores
        r2_train = r2_score(y_train, y_train_pred)
        r2_validation = r2_score(y_validation, y_val_pred)
        r2_test = r2_score(y_test, y_test_pred)

        results.append({
            'k': i,
            'R2_Train': r2_train,
            'R2_Validation': r2_validation,
            'R2_Test': r2_test
        })

        if(r2_validation > maxR2):
            maxR2 = r2_validation
            minK = i

    # for(i in range (1,30))
    # knn = KNeighborsRegressor(n_neighbors=9)
    # knn.fit(X_train_scaled, y_train)

    #best based off of MSE on validation set
    print(f'Best k: {minK} R^2: {maxR2:.4f}')

    # Create a DataFrame for easy viewing and plotting
    results_df = pd.DataFrame(results)

    print(results_df)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k'], results_df['R2_Train'], label='Training $R^2$', marker='o')
    plt.plot(results_df['k'], results_df['R2_Validation'], label='Validation $R^2$', marker='s')
    plt.plot(results_df['k'], results_df['R2_Test'], label='Test $R^2$', marker='^')

    plt.title('$R^2$ Score vs. K Neighbors')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('$R^2$ Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()




