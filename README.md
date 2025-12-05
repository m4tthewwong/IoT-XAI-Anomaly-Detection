# IoT XAI Anomaly Detection

A lightweight, explainable anomaly-detection system for IoT network traffic.  
This project combines classical flow-based features with a compact deep-learning model designed for efficient edge deployment. SHAP-based explanations provide transparent insight into each prediction.

---

## Features
- End-to-end IoT anomaly detection pipeline  
- Lightweight model suitable for edge devices  
- SHAP explainability for alert interpretation  
- Modular, clean code structure  
- Supports both real-time and batch inference  

---

## Tech Stack
- Python  
- Scikit-learn  
- TensorFlow / Keras  
- SHAP  
- NumPy / Pandas  

---

## How It Works
1. Extracts relevant flow features from IoT traffic  
2. Preprocesses and scales numerical + categorical inputs  
3. Runs inference using a trained lightweight model  
4. Outputs predicted class + confidence  
5. Generates SHAP explanations to visualize feature influence  

---

## Adding datasets
- Download all four datasets [here](https://unsw-my.sharepoint.com/personal/z5131399_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5131399%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FBot%2DIoT%5FDataset%2FDataset%2F5%25%2FAll%20features&viewid=604d81f1%2D64a9%2D4a09%2D8464%2D3c45ff9ba8fe&ga=1).
- Create a folder called "data" inside project directory.
- Put the four csv files inside the data folder.

## Running SHAP
```bash
python src/shap_explain.py
```

## Running LIME
```bash
python src/lime_explain.py
```

## Running Inference
```bash
python src/inference.py
```

## Predictions include:
- Classification (NORMAL or ATTACK)
- Model confidence score

## Repository Structure
```bash
src/              # Main pipeline code
models/           # Model architectures and training scripts
explanations/     # SHAP utilities and explanation visuals
```

## Dataset
This project uses publicly available IoT intrusion datasets such as BoT-IoT.
Datasets are not included in this repository.

## Results
- High accuracy across attack categories
- Low-latency inference optimized for edge hardware
- SHAP visualizations enable transparent anomaly explanations
