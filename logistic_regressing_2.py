
# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

print("Libraries imported successfully!")


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv("spotify_churn_dataset.csv")

print("\nDataset shape (rows, columns):", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())


# ==========================================
# 3. DATASET INFORMATION
# ==========================================

print("\nDataset Information:")
df.info()

print("\nMissing values in each column:")
print(df.isnull().sum())


# ==========================================
# 4. ENCODE CATEGORICAL VARIABLES
# ==========================================

df = pd.get_dummies(df, drop_first=True)

print("\nDataset shape after encoding:", df.shape)


# ==========================================
# 5. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("offline_listening", axis=1)
y = df["offline_listening"]

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)


# ==========================================
# 6. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ==========================================
# 7. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")


# ==========================================
# 8. CREATE AND TRAIN MODEL
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully!")


# ==========================================
# 9. PREDICT ALL TESTING VALUES
# ==========================================

y_pred = model.predict(X_test_scaled)

print("\nAll predictions completed!")


# ==========================================
# 10. DISPLAY APPROXIMATELY HALF DATA
# ==========================================

n = len(y_test) // 10

comparison = pd.DataFrame({
    "Actual Value": y_test.values[:n],
    "Predicted Value": y_pred[:n]
})

print("\nActual vs Predicted Values (Approximately Half):")
print(comparison.to_string(index=False))

print("\nNumber of rows displayed:", n)


# ==========================================
# 11. ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(14, 6))

plt.plot(
    y_test.values[:n],
    label="Actual Values",
    marker='o',
    linestyle='-',
    markersize=4
)

plt.plot(
    y_pred[:n],
    label="Predicted Values",
    marker='x',
    linestyle='--',
    markersize=4
)

plt.xlabel("Test Data Index")
plt.ylabel("Offline Listening")
plt.title("Actual vs Predicted Values - Logistic Regression")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================
# 12. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No", "Yes"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()


# ==========================================
# 13. MODEL ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# ==========================================
# 14. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(classification_report(y_test, y_pred))


# ==========================================
# 15. FINAL MODEL EVALUATION
# ==========================================

print("\n================================")
print("       MODEL EVALUATION")
print("================================")

print("Algorithm: Logistic Regression")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Displayed Predictions:", n)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nProgram executed successfully!")
