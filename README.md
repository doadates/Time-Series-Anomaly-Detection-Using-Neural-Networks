# Neural Networks for Anomaly Detection in Time Series Data

## 📌 Overview
This project compares **two deep learning approaches** for anomaly detection in industrial time series sensor data:

1. **TimeGPT-1** – Transformer-based model for contextual anomaly detection using Nixtla’s API.
2. **LSTM Autoencoder** – Reconstruction-based model, tested with both static and dynamic thresholding.

The models are evaluated on the **Intel Berkeley Research Lab Sensor Dataset**, focusing on temperature, voltage, and humidity data.

---

## 📂 Project Structure
```
.
├── anomaly.py       # TimeGPT-1 anomaly detection on humidity data
├── auto.py          # LSTM Autoencoder with static thresholding (temperature data)
├── auto_dyn.py      # LSTM Autoencoder with dynamic thresholding (voltage data)
├── Doga_Deniz_Ates_Neural_Networks_Anomaly_Detection.pdf  # Full project report
└── data/            # Contains data.txt (Intel Berkeley Research Lab Sensor data)
```

---

## ⚙️ Methodology

### **1. TimeGPT-1 Anomaly Detection** (`anomaly.py`)
- Uses Nixtla’s `NixtlaClient` for anomaly detection.
- Detects anomalies by comparing predicted values to observed values.
- Well-suited for **contextual anomalies** and long-term dependencies.

```python
nixtla_client = NixtlaClient(api_key='YOUR_API_KEY')
anomalies_df = nixtla_client.detect_anomalies(
    df, time_col='timestamp', target_col='humidity', freq='H', level=97
)
```

---

### **2. LSTM Autoencoder – Static Thresholding** (`auto.py`)
- Encodes time series into a latent space and reconstructs it.
- Anomalies are detected where the **reconstruction error > threshold**.
- **Static threshold** = `mean + 1.3 * std` of reconstruction error.
- Good for detecting **short-term anomalies**.

---

### **3. LSTM Autoencoder – Dynamic Thresholding** (`auto_dyn.py`)
- Same as static version but uses a **rolling mean + k * std** over a window.
- More sensitive to **gradual drifts** and **non-stationary data**.
- Detects context-dependent anomalies.

---

## 📊 Results Summary
- **LSTM Autoencoder**: High sensitivity, good for small or sudden anomalies, but may produce more false positives.
- **TimeGPT-1**: Captures broader contextual deviations, fewer false alarms, better interpretability.
- **Dynamic Thresholding**: Improves detection in drifting or noisy environments.

---

## 📦 Requirements
Install dependencies with:
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib nixtla
```

---

## ▶️ How to Run
1. Place `data.txt` inside the `data/` folder.
2. For TimeGPT-1, set your Nixtla API key in `anomaly.py`.
3. Run each script:
```bash
python anomaly.py      # TimeGPT-1 on humidity
python auto.py         # LSTM Autoencoder (static)
python auto_dyn.py     # LSTM Autoencoder (dynamic)
```

---

## 📄 Full Report
For a detailed explanation of the models, dataset, and results, see:  
📄 **[Doga_Deniz_Ates_Neural_Networks_Anomaly_Detection.pdf](./Doga_Deniz_Ates_Neural_Networks_Anomaly_Detection.pdf)**

---

## 👤 Author
**Doga Deniz Ates**  
MSc Student – Intelligent Adaptive Systems, University of Hamburg
