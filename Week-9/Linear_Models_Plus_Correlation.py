from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('Dataset\PS_2025.11.03_17.08.05.csv', comment='#')

# Preparar datos para modelo lineal
model_df = df.copy()

# Codificar método de descubrimiento
le = LabelEncoder()
model_df['method_encoded'] = le.fit_transform(model_df['discoverymethod'])

# Variables para el modelo
X = model_df[['disc_year', 'method_encoded', 'sy_snum']]
y = model_df['sy_pnum']  # Predecir número de planetas

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Modelo Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,      # Número de árboles
    max_depth=10,          # Profundidad máxima para evitar overfitting
    min_samples_split=5,   # Mínimo de muestras para dividir un nodo
    min_samples_leaf=2,    # Mínimo de muestras en hoja
    random_state=42,       # Para reproducibilidad
    n_jobs=-1             # Usar todos los cores del CPU
)

# Entrenar el modelo
rf_model.fit(X_train, y_train)

# Predicciones
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Métricas COMPARATIVAS
print("=" * 50)
print("COMPARACIÓN DE MODELOS")
print("=" * 50)

# Métricas Random Forest
r2_train_rf = r2_score(y_train, y_pred_train)
r2_test_rf = r2_score(y_test, y_pred_test)
mse_test_rf = mean_squared_error(y_test, y_pred_test)

print("RANDOM FOREST:")
print(f"   R² Train: {r2_train_rf:.4f}")
print(f"   R² Test:  {r2_test_rf:.4f}")
print(f"   MSE Test: {mse_test_rf:.4f}")

# Importancia de las características
importances = rf_model.feature_importances_
feature_names = X.columns

print("\nIMPORTANCIA DE CARACTERÍSTICAS:")
for feature, importance in zip(feature_names, importances):
    print(f"   {feature}: {importance:.4f}")

# Modelo lineal original para comparar
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
r2_linear = r2_score(y_test, y_pred_linear)

print("\nMODELO LINEAL ORIGINAL:")
print(f"   R² Test: {r2_linear:.4f}")
print(f"   Coeficientes: {linear_model.coef_}")
print(f"   Intercepto: {linear_model.intercept_:.4f}")

# Gráficos mejorados
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Valores Reales vs Predichos (Random Forest)
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, color='blue', label='Random Forest')
axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Valores Reales')
axes[0, 0].set_ylabel('Valores Predichos (RF)')
axes[0, 0].set_title('Random Forest: Valores Reales vs Predichos')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuos (Random Forest)
residuals_rf = y_test - y_pred_test
axes[0, 1].scatter(y_pred_test, residuals_rf, alpha=0.6, color='blue')
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Valores Predichos (RF)')
axes[0, 1].set_ylabel('Residuos')
axes[0, 1].set_title('Random Forest: Análisis de Residuos')
axes[0, 1].grid(True, alpha=0.3)

# 3. Comparación de predicciones RF vs Lineal
axes[1, 0].scatter(y_pred_linear, y_pred_test, alpha=0.6, color='green')
axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Predicciones Modelo Lineal')
axes[1, 0].set_ylabel('Predicciones Random Forest')
axes[1, 0].set_title('Comparación: Predicciones Lineal vs RF')
axes[1, 0].grid(True, alpha=0.3)

# 4. Importancia de características
y_pos = np.arange(len(feature_names))
axes[1, 1].barh(y_pos, importances, color='skyblue')
axes[1, 1].set_yticks(y_pos)
axes[1, 1].set_yticklabels(feature_names)
axes[1, 1].set_xlabel('Importancia')
axes[1, 1].set_title('Importancia de Características (Random Forest)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Análisis de mejora
improvement = ((r2_test_rf - r2_linear) / abs(r2_linear)) * 100
print(f"\nMejora: Random Forest es {improvement:+.1f}% mejor que el modelo lineal")

# Predicciones de ejemplo
print("\nPREDICCIONES DE EJEMPLO:")
sample_idx = X_test.sample(5, random_state=42).index
for idx in sample_idx:
    real = y.loc[idx]
    X_sample = pd.DataFrame([X.loc[idx]], columns=X.columns)
    pred_rf = rf_model.predict(X_sample)[0]
    pred_linear = linear_model.predict(X_sample)[0]
    print(f"   Real: {real:.1f} | RF: {pred_rf:.1f} | Lineal: {pred_linear:.1f}")