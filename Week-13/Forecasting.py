from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos
df = pd.read_csv('Dataset\PS_2025.11.03_17.08.05.csv', comment='#')

# Serie temporal: Planetas descubiertos por año
time_series = df.groupby('disc_year').size().reset_index()
time_series.columns = ['year', 'discoveries']

# Preparar datos para forecasting
X_time = time_series['year'].values.reshape(-1, 1)
y_time = time_series['discoveries'].values

# Modelo de regresión para forecasting
time_model = LinearRegression()
time_model.fit(X_time, y_time)

# Predecir años futuros
future_years = np.array(range(2026, 2031)).reshape(-1, 1)
future_predictions = time_model.predict(future_years)

# Gráfico
plt.figure(figsize=(12, 6))
plt.plot(time_series['year'], time_series['discoveries'], 'bo-', label='Datos Reales')
plt.plot(future_years, future_predictions, 'ro--', label='Predicciones')
plt.xlabel('Año')
plt.ylabel('Descubrimientos')
plt.title('Forecasting de Descubrimientos de Exoplanetas')
plt.legend()
plt.grid(True)
plt.show()

print("Predicciones para 2026-2030:")
for year, pred in zip(future_years.flatten(), future_predictions):
    print(f"Año {year}: {pred:.1f} descubrimientos predichos")