import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from lime import lime_tabular

MODEL_DIR = "../models/"
DATA_FILE = "../data/UNSW_2018_IoT_Botnet_Full5pc_1.csv"
OUT_FILE = "../explanations/lime_explanation.html"

# -----------------------------
# LOAD MODEL + ARTIFACTS
# -----------------------------
model = tf.keras.models.load_model(MODEL_DIR + "iot_model.h5")
scaler = pickle.load(open(MODEL_DIR + "scaler.pkl", "rb"))
le_proto = pickle.load(open(MODEL_DIR + "le_proto.pkl", "rb"))
le_state = pickle.load(open(MODEL_DIR + "le_state.pkl", "rb"))

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_FILE, low_memory=False)

keep_cols = ["dur", "proto", "sbytes", "dbytes", "spkts", "dpkts", "state", "attack"]
df = df[keep_cols]

df["proto"] = le_proto.transform(df["proto"])
df["state"] = le_state.transform(df["state"])

X_raw = df[["dur", "proto", "sbytes", "dbytes", "spkts", "dpkts", "state"]].astype(np.float32)

def lime_predict(x):
    x_scaled = scaler.transform(x)
    pa = model.predict(x_scaled).flatten()
    pn = 1 - pa
    return np.vstack([pn, pa]).T

explainer = lime_tabular.LimeTabularExplainer(
    X_raw.values,
    feature_names=X_raw.columns.tolist(),
    class_names=["normal", "attack"],
    mode="classification"
)

exp = explainer.explain_instance(
    X_raw.values[10],
    lime_predict
)

exp.save_to_file(OUT_FILE)
print("Saved:", OUT_FILE)
