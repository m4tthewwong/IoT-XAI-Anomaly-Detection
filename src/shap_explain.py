import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

MODEL_DIR = "../models/"
DATA_FILE = "../data/UNSW_2018_IoT_Botnet_Full5pc_1.csv"
OUT_FILE = "../explanations/shap_summary.png"

# -----------------------------
# LOAD MODEL + ARTIFACTS
# -----------------------------
model = tf.keras.models.load_model(MODEL_DIR + "iot_model.h5")
scaler = pickle.load(open(MODEL_DIR + "scaler.pkl", "rb"))
le_proto = pickle.load(open(MODEL_DIR + "le_proto.pkl", "rb"))
le_state = pickle.load(open(MODEL_DIR + "le_state.pkl", "rb"))

print("Loaded model + scaler + encoders.")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_FILE, low_memory=False)

keep_cols = ["dur", "proto", "sbytes", "dbytes", "spkts", "dpkts", "state"]
df = df[keep_cols]

df["proto"] = le_proto.transform(df["proto"])
df["state"] = le_state.transform(df["state"])

X = df.astype(np.float32)

X_base_np   = X.sample(100, random_state=42).to_numpy()
X_sample_np = X.sample(50, random_state=7).to_numpy()

def predict_fn(x):
    return model.predict(scaler.transform(x))

print("Computing SHAP values…")
explainer = shap.KernelExplainer(predict_fn, X_base_np)
shap_values = explainer.shap_values(X_sample_np)

plt.figure(figsize=(14, 6))
shap.summary_plot(
    shap_values,
    X_sample_np,
    feature_names=keep_cols,
    show=False
)

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=200)
plt.show()

print("Saved SHAP plot to:", OUT_FILE)
