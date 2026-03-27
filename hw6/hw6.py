import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score,train_test_split






df = pd.read_csv('HW06_data.csv')
X = df[[f"X{i}" for i in range(1, 14)]].to_numpy()
y_raw = df['Y'].to_numpy()

X_dev, X_test, y_dev, y_test = train_test_split(X, y_raw, test_size=0.25, random_state=42, stratify=y_raw)

logReg = LogisticRegression(max_iter=1000)
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

models = [logReg, lda, qda]

for i, model in enumerate(models):
    scores = cross_val_score(model, X_dev, y_dev, cv=5)
    print(f"Classifier {i+1} Mean Accuracy: {scores.mean():.4f} ")



logReg.fit(X_dev, y_dev)
lda.fit(X_dev, y_dev)
qda.fit(X_dev, y_dev)

y_logReg_pred = logReg.predict(X_test)
y_lda_pred = lda.predict(X_test)    
y_qda_pred = qda.predict(X_test)

log_mean_acc = accuracy_score(y_test, y_logReg_pred)
lda_mean_acc = accuracy_score(y_test, y_lda_pred)
qda_mean_acc = accuracy_score(y_test, y_qda_pred)

print(f"Logistic Regression Mean Accuracy: {log_mean_acc:.4f}")
print(f"Linear Discriminant Analysis Mean Accuracy: {lda_mean_acc:.4f}")
print(f"Quadratic Discriminant Analysis Mean Accuracy: {qda_mean_acc:.4f}")