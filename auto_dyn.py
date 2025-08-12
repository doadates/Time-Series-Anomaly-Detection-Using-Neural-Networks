import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv(
    "data/data.txt",
    sep=r'\s+',
    header=None,
    engine='python',
    on_bad_lines='skip'
)

df.columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']
df = df[df['moteid'] == 1]
df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
df.dropna(subset=['timestamp'], inplace=True)
df = df[['timestamp', 'voltage']]
df.set_index('timestamp', inplace=True)
df = df.resample('1h').mean()
df['voltage'] = df['voltage'].interpolate(method='time')
df.reset_index(inplace=True)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['voltage']])

def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        xs.append(x)
    return np.array(xs)

SEQUENCE_LENGTH = 30
X = create_sequences(data_scaled, SEQUENCE_LENGTH)

timesteps = X.shape[1]
n_features = X.shape[2]

input_layer = Input(shape=(timesteps, n_features))
encoded = LSTM(64, activation='relu')(input_layer)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(n_features))(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
autoencoder.compile(optimizer=optimizer, loss='mse')
autoencoder.fit(X, X, epochs=10, batch_size=32, validation_split=0.1, shuffle=True)

X_pred = autoencoder.predict(X)
mse = np.mean(np.power(X - X_pred, 2), axis=(1, 2))

window_size = 48  # geçmiş 48 saatlik pencere
k = 1.5  # eşik hassasiyet katsayısı

dynamic_thresholds = []
for i in range(len(mse)):
    if i < window_size:
        dynamic_thresholds.append(np.nan)
    else:
        window = mse[i - window_size:i]
        mean = np.mean(window)
        std = np.std(window)
        threshold = mean + k * std
        dynamic_thresholds.append(threshold)

dynamic_thresholds = np.array(dynamic_thresholds)
dynamic_anomalies = mse > dynamic_thresholds

timestamps = df['timestamp'][SEQUENCE_LENGTH:]

# === Plot reconstruction error with dynamic thresholding ===
plt.rcParams.update({
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

plt.figure(figsize=(15, 5))
plt.plot(timestamps, mse, label='Reconstruction Error', color='steelblue')
plt.plot(timestamps, dynamic_thresholds, label='Dynamic Threshold', color='red', linestyle='--')
plt.scatter(timestamps[dynamic_anomalies], mse[dynamic_anomalies], color='orange', label='Anomalies', zorder=5)

plt.xlabel("Time")
plt.ylabel("Reconstruction Error")
plt.title("Voltage Data Anomaly Detection with Dynamic Thresholding")
plt.legend()

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()
