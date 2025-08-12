import pandas as pd
from nixtla import NixtlaClient
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 1. Instantiate the NixtlaClient
nixtla_client = NixtlaClient(api_key = '#yourapikey')

df = pd.read_csv(
    "data/data.txt",
    sep=r'\s+',
    header=None,
    engine='python',
    on_bad_lines='skip'
)

df.columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity','light', 'voltage']

df = df[df['moteid'] ==1]

df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
df.dropna(subset=['timestamp'], inplace=True)

df = df[['timestamp', 'humidity']]
df.set_index('timestamp', inplace=True)

df = df.resample('h').mean()

df['humidity'] = df['humidity'].interpolate(method='time')

df.reset_index(inplace=True)


anomalies_df = nixtla_client.detect_anomalies(df, time_col='timestamp', target_col='humidity', freq='H', level=97)

nixtla_client.plot(df, anomalies_df,time_col='timestamp', target_col='humidity')


only_anomalies = anomalies_df[anomalies_df['anomaly'] == 1]


plt.figure(figsize=(15, 5))
plt.plot(df['timestamp'], df['humidity'], label='Actual', color='black')
plt.scatter(only_anomalies['timestamp'], only_anomalies['humidity'], color='red', label='Anomaly')

plt.xlabel("Time")
plt.ylabel("humidity")
plt.title("humidity Data Anomaly Detection")
plt.legend()
plt.grid(True)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))  
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.gcf().autofmt_xdate()  
plt.tight_layout()


plt.show()
