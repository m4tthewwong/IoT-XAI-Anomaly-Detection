import numpy as np
import pickle
import tensorflow as tf

MODEL_DIR = "../models/"

model = tf.keras.models.load_model(MODEL_DIR + "iot_model.h5")
scaler = pickle.load(open(MODEL_DIR + "scaler.pkl", "rb"))
le_proto = pickle.load(open(MODEL_DIR + "le_proto.pkl", "rb"))
le_state = pickle.load(open(MODEL_DIR + "le_state.pkl", "rb"))

print("Artifacts loaded successfully.\n")

def predict_single(dur, proto, sbytes, dbytes, spkts, dpkts, state):
    proto_enc = le_proto.transform([proto])[0]
    state_enc = le_state.transform([state])[0]

    x = np.array([[dur, proto_enc, sbytes, dbytes, spkts, dpkts, state_enc]])
    x_scaled = scaler.transform(x)

    prob_attack = model.predict(x_scaled)[0][0]
    cls = 1 if prob_attack > 0.5 else 0

    return cls, prob_attack

example_attack = {
    "dur": 7.056393,
    "proto": "tcp",
    "sbytes": 650,
    "dbytes": 1330,
    "spkts": 5,
    "dpkts": 3,
    "state": "RST"
}

example_normal = {
    "dur": 0.000131,
    "proto": "arp",
    "sbytes": 60,
    "dbytes": 60,
    "spkts": 1,
    "dpkts": 1,
    "state": "CON"
}

print("Running inference on VERIFIED ATTACK example:")
cls_a, p_a = predict_single(**example_attack)
print("Prediction:", "ATTACK" if cls_a == 1 else "NORMAL")
print("Confidence:", f"{p_a:.4f}\n")

print("Running inference on VERIFIED NORMAL example:")
cls_n, p_n = predict_single(**example_normal)
print("Prediction:", "ATTACK" if cls_n == 1 else "NORMAL")
print("Confidence:", f"{1 - p_n:.4f}")
