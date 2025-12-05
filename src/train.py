import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
import os

DATA_DIR = "../data/"
MODEL_DIR = "../models/"

# -------------------------------------------------------
# LOAD ALL CSV FILES AUTOMATICALLY FROM data/ DIRECTORY
# -------------------------------------------------------
csv_files = [os.path.join(DATA_DIR, f) 
             for f in os.listdir(DATA_DIR) 
             if f.lower().endswith(".csv")]

if len(csv_files) == 0:
    raise FileNotFoundError("No CSV files found in ../data/. Make sure your dataset files are there.")

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(" •", f)

df = pd.concat([pd.read_csv(f, low_memory=False) for f in csv_files], ignore_index=True)

# -------------------------------------------------------
# SELECT FEATURES
# -------------------------------------------------------
keep_cols = ["dur", "proto", "sbytes", "dbytes", "spkts", "dpkts", "state", "attack"]
df = df[keep_cols]

# -------------------------------------------------------
# ENCODE CATEGORICAL COLUMNS
# -------------------------------------------------------
le_proto = LabelEncoder()
le_state = LabelEncoder()

df["proto"] = le_proto.fit_transform(df["proto"])
df["state"] = le_state.fit_transform(df["state"])

# -------------------------------------------------------
# SPLIT INPUTS / LABEL
# -------------------------------------------------------
X_raw = df[["dur", "proto", "sbytes", "dbytes", "spkts", "dpkts", "state"]]
y = df["attack"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# SCALE
# -------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# -------------------------------------------------------
# CLASS WEIGHTS (handles class imbalance)
# -------------------------------------------------------
classes = np.unique(y_train)
cw_array = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights = {cls: weight for cls, weight in zip(classes, cw_array)}

print("Class weights:", class_weights)

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    X_train, y_train,
    epochs=12,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights
)

# -------------------------------------------------------
# EVALUATE MODEL
# -------------------------------------------------------
pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, pred))

# -------------------------------------------------------
# SAVE ARTIFACTS
# -------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

model.save(os.path.join(MODEL_DIR, "iot_model.h5"))
pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb"))
pickle.dump(le_proto, open(os.path.join(MODEL_DIR, "le_proto.pkl"), "wb"))
pickle.dump(le_state, open(os.path.join(MODEL_DIR, "le_state.pkl"), "wb"))

print("Saved model + scaler + encoders to ../models/")
