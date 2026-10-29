import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Paths
FOLDER_DIR = r"C:\Users\zinou\OneDrive\Bureau\US embassy"
FILE_NAME = "SentinelAI_Algeria_Fraud_Dataset_2500.xlsx"
DATASET_PATH = os.path.join(FOLDER_DIR, FILE_NAME)

print(f"[*] Loading dataset from: {DATASET_PATH}")
df = pd.read_excel(DATASET_PATH)

# 2. Features and Labels
df["is_declined"] = (df["Status"] == "Declined").astype(int)
df_encoded = pd.get_dummies(df, columns=["Merchant Category"], drop_first=True)

feature_cols = [
    "Amount (DZD)",
    "Velocity (KM/H)",
    "Attempt Number",
    "is_declined",
    "Latitude",
    "Longitude",
] + [c for c in df_encoded.columns if c.startswith("Merchant Category_")]

X = df_encoded[feature_cols].values.astype(np.float32)
y = df_encoded["Is Fraud"].values.astype(np.int32)

# 3. Three-way Split (60% Train, 20% CV, 20% Test)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_cv, y_train, y_cv = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full,
)

# 4. Normalization (Trained on Train set only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cv = scaler.transform(X_cv)
X_test = scaler.transform(X_test)

# Save artifacts for live server
joblib.dump(scaler, os.path.join(FOLDER_DIR, "scaler.pkl"))
joblib.dump(feature_cols, os.path.join(FOLDER_DIR, "feature_columns.pkl"))

# 5. Model Architecture
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# 6. Training
print("[*] Training neural network...")
model.fit(
    X_train,
    y_train,
    validation_data=(X_cv, y_cv),
    epochs=15,
    batch_size=64,
    verbose=0,
)

# 7. Evaluate Global Errors
j_train, acc_train = model.evaluate(X_train, y_train, verbose=0)
j_cv, acc_cv = model.evaluate(X_cv, y_cv, verbose=0)
j_test, acc_test = model.evaluate(X_test, y_test, verbose=0)

print(f"\n[+] Cross-Validation Error (J_cv): {j_cv:.5f}")
print(
    f"[+] Train Cost (J_train): {j_train:.5f} | Test Error (J_test):"
    f" {j_test:.5f}"
)

# 8. Save Model
model_path = os.path.join(FOLDER_DIR, "fraud_model.keras")
model.save(model_path)
print(f"[+] Production weights saved to: {model_path}\n")