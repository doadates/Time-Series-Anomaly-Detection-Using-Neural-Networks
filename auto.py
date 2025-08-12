import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
import matplotlib.pyplot as plt
import os
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

df = df[['timestamp', 'temperature']]
df.set_index('timestamp', inplace=True)

df = df.resample('1h').mean()
df['temperature'] = df['temperature'].interpolate(method='time')


df.reset_index(inplace=True)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['temperature']])

def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        xs.append(x)
    return np.array(xs)

SEQUENCE_LENGTH = 30
X = create_sequences(data_scaled, SEQUENCE_LENGTH)

print("NaNs in input:", np.isnan(X).sum())  # Should be 0

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
autoencoder.summary()

autoencoder.fit(X, X, epochs=10, batch_size=32, validation_split=0.1, shuffle=True)

X_pred = autoencoder.predict(X)
mse = np.mean(np.power(X - X_pred, 2), axis=(1, 2))

threshold = np.mean(mse) +1.3 * np.std(mse)
anomalies = mse > threshold

timestamps = df['timestamp'][SEQUENCE_LENGTH:]
"""
plt.figure(figsize=(15, 5))
plt.plot(timestamps, mse, label='Reconstruction error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(timestamps[anomalies], mse[anomalies], color='orange', label='Anomalies')
plt.title("Temperature Data Anomaly Detection")
plt.legend()
plt.show()
"""

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
mse_df = pd.DataFrame({
    'timestamp': timestamps.values,
    'mse': mse
})
print(mse_df.head())

mse_df.to_csv("temperature_mse_auto.csv", index=False)

plt.figure(figsize=(15, 5))
plt.plot(timestamps, mse, label='Reconstruction Error', color='steelblue')
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.scatter(timestamps[anomalies], mse[anomalies], color='orange', label='Anomalies', zorder=5)

plt.xlabel("Time")
plt.ylabel("Reconstruction Error")
plt.title("Temperature Data Anomaly Detection")
plt.legend()


ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))  
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.gcf().autofmt_xdate()  
plt.tight_layout()
plt.show()

"""
Optional: Save anomalies to CSV
anomaly_df = df.iloc[SEQUENCE_LENGTH:].copy()
anomaly_df['reconstruction_error'] = mse
anomaly_df['anomaly'] = anomalies
anomaly_df[anomaly_df['anomaly']].to_csv("voltage_anomalies.csv", index=False)
"""