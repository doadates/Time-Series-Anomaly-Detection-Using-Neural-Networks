# Neural Networks for Anomaly Detection in Time Series Data

## 📌 Overview
This project compares **two deep learning approaches** for anomaly detection in industrial time series sensor data:

1. **TimeGPT-1** – Transformer-based model for contextual anomaly detection using Nixtla’s API.
2. **LSTM Autoencoder** – Reconstruction-based model, tested with both static and dynamic thresholding.

The models are evaluated on the **Intel Berkeley Research Lab Sensor Dataset**, focusing on temperature, voltage, humidity and light data.

---

## 📂 Project Structure
```
.
├── anomaly.py       # TimeGPT-1 anomaly detection on humidity data
├── auto.py          # LSTM Autoencoder with static thresholding (temperature data)
├── auto_dyn.py      # LSTM Autoencoder with dynamic thresholding (voltage data)
├── Doga_Deniz_Ates_Neural_Networks_Anomaly_Detection.pdf  # Full project report
└── data/            # Contains data.txt (Intel Berkeley Research Lab Sensor data) (https://db.csail.mit.edu/labdata/labdata.html)
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
<img width="1500" height="500" alt="volt_temp" src="https://github.com/user-attachments/assets/0d77f1bd-2442-4f23-9ee7-1250f910e1f7" />
<img width="1500" height="500" alt="temp_timegpt1" src="https://github.com/user-attachments/assets/d985529b-1612-4176-9075-0303d16e245c" />

---

### **2. LSTM Autoencoder – Static Thresholding** (`auto.py`)
- Encodes time series into a latent space and reconstructs it.
- Anomalies are detected where the **reconstruction error > threshold**.
- **Static threshold** = `mean + 1.3 * std` of reconstruction error.
- Good for detecting **short-term anomalies**.
<img width="1500" height="500" alt="temp_auto" src="https://github.com/user-attachments/assets/56d2bbba-f1aa-4701-8fe7-faffe3ff7ab8" />
<img width="1500" height="500" alt="volt_auto" src="https://github.com/user-attachments/assets/75120153-0a74-46e3-a08f-047fdc836176" />

---

### **3. LSTM Autoencoder – Dynamic Thresholding** (`auto_dyn.py`)
- Same as static version but uses a **rolling mean + k * std** over a window.
- More sensitive to **gradual drifts** and **non-stationary data**.
- Detects context-dependent anomalies.
<img width="1500" height="500" alt="volt_auto_dyn" src="https://github.com/user-attachments/assets/f3a86d2b-8a5d-4fef-9235-40bb6401307e" />
<img width="1500" height="500" alt="temp_auto_dyn" src="https://github.com/user-attachments/assets/3facf708-0450-452a-a53d-0a1f82006b30" />

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
python anomaly.py      # TimeGPT-1
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
