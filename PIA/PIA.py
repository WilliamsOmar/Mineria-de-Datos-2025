# =============================================================================
# ANÁLISIS PREDICTIVO: ¿QUÉ MÉTODOS DOMINARÁN EL FUTURO DE LA EXOPLANETOLOGÍA?
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8')
print("=== PREDICCIÓN DEL FUTURO DE MÉTODOS DE DETECCIÓN ===\n")

# 1. CARGA Y FILTRADO DE DATOS

df = pd.read_csv('Dataset\PS_2025.11.03_17.08.05.csv', comment='#')

print("1. FILTRANDO DATOS PARA MODELO ESTABLE...")

# Filtrar métodos con suficientes datos (mínimo 50 ejemplos)
method_counts = df['discoverymethod'].value_counts()
valid_methods = method_counts[method_counts >= 50].index
df_filtered = df[df['discoverymethod'].isin(valid_methods)]

print(f"   - Métodos originales: {len(method_counts)}")
print(f"   - Métodos con ≥50 muestras: {len(valid_methods)}")
print(f"   - Registros totales: {len(df_filtered):,}")

# 2. ENGINEERING DE CARACTERÍSTICAS AVANZADAS

print("\n2. CREANDO CARACTERÍSTICAS TEMPORALES AVANZADAS...")

# Características básicas
features_df = df_filtered[['disc_year', 'sy_snum', 'sy_pnum', 'discoverymethod']].copy()

# ENGINEERING AVANZADO: Características temporales
features_df['decade'] = (features_df['disc_year'] // 10) * 10
features_df['year_normalized'] = (features_df['disc_year'] - 1990) / (2025 - 1990)

# Patrones cíclicos anuales (para capturar tendencias)
features_df['year_sin'] = np.sin(2 * np.pi * features_df['year_normalized'])
features_df['year_cos'] = np.cos(2 * np.pi * features_df['year_normalized'])

# Épocas tecnológicas importantes
features_df['is_early_era'] = (features_df['disc_year'] < 2000).astype(int)
features_df['is_kepler_era'] = ((features_df['disc_year'] >= 2009) & (features_df['disc_year'] <= 2018)).astype(int)
features_df['is_tess_era'] = (features_df['disc_year'] >= 2018).astype(int)

# Características de complejidad del sistema
features_df['system_complexity'] = features_df['sy_pnum'] * (features_df['sy_snum'] + 1)
features_df['is_multi_planet'] = (features_df['sy_pnum'] > 1).astype(int)
features_df['is_multi_star'] = (features_df['sy_snum'] > 1).astype(int)
features_df['planet_density'] = features_df['sy_pnum'] / (features_df['sy_snum'] + 1)

# Codificar variable objetivo
le = LabelEncoder()
y = le.fit_transform(features_df['discoverymethod'])

# Seleccionar características finales
feature_columns = [
    'disc_year', 'sy_snum', 'sy_pnum', 
    'decade', 'year_normalized', 'year_sin', 'year_cos',
    'is_early_era', 'is_kepler_era', 'is_tess_era',
    'system_complexity', 'is_multi_planet', 'is_multi_star', 'planet_density'
]

X = features_df[feature_columns]

print(f"   - Características creadas: {len(feature_columns)}")
print(f"   - Métodos a predecir: {len(le.classes_)}")
print(f"   - Distribución de clases:")
for i, method in enumerate(le.classes_):
    count = np.sum(y == i)
    print(f"     * {method}: {count} muestras")

# 3. MODELO PREDICTIVO: RANDOM FOREST

print("\n3. ENTRENANDO MODELO PREDICTIVO (Random Forest)...")

# Dividir datos 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 
)

# Modelo avanzado
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predicciones
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# Evaluación
accuracy = np.mean(y_pred == y_test)
cv_scores = cross_val_score(rf_model, X, y, cv=5)

print(f"   - Precisión: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   - Validación cruzada: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 4. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS

print("\n4. ANALIZANDO QUÉ FACTORES DETERMINAN EL MÉTODO:")

# Importancia de características
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('IMPORTANCIA DE CARACTERÍSTICAS\n¿Qué determina el método de descubrimiento?')
plt.xlabel('Importancia Relativa')
plt.tight_layout()
plt.show()

print("   CARACTERÍSTICAS MÁS IMPORTANTES:")
for i, row in feature_importance.head(5).iterrows():
    print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")

# 5. PREDICCIÓN DE TENDENCIAS FUTURAS (2025-2035)

print("\n5. SIMULANDO DESCUBRIMIENTOS FUTUROS (2025-2035)...")

def create_future_data(year, system_type):
    """Crear datos para un sistema futuro"""
    year_norm = (year - 1990) / (2035 - 1990)
    
    # Crear diccionario con TODAS las características en el ORDEN CORRECTO
    base_data = {
        'disc_year': year,
        'sy_snum': 1,  # Valores por defecto
        'sy_pnum': 1,
        'decade': (year // 10) * 10,
        'year_normalized': year_norm,
        'year_sin': np.sin(2 * np.pi * year_norm),
        'year_cos': np.cos(2 * np.pi * year_norm),
        'is_early_era': 0,
        'is_kepler_era': 0,
        'is_tess_era': 1 if year >= 2018 else 0,
        'system_complexity': 1,
        'is_multi_planet': 0,
        'is_multi_star': 0,
        'planet_density': 0.5
    }
    
    # Modificar según el tipo de sistema
    if system_type == 'simple':
        base_data.update({
            'sy_snum': 1, 'sy_pnum': 1,
            'system_complexity': 1, 'is_multi_planet': 0, 
            'is_multi_star': 0, 'planet_density': 0.5
        })
    elif system_type == 'multiple':
        base_data.update({
            'sy_snum': 1, 'sy_pnum': 3,
            'system_complexity': 3, 'is_multi_planet': 1, 
            'is_multi_star': 0, 'planet_density': 3.0
        })
    elif system_type == 'binary':
        base_data.update({
            'sy_snum': 2, 'sy_pnum': 2,
            'system_complexity': 4, 'is_multi_planet': 1, 
            'is_multi_star': 1, 'planet_density': 1.0
        })
    elif system_type == 'complex':
        base_data.update({
            'sy_snum': 1, 'sy_pnum': 5,
            'system_complexity': 5, 'is_multi_planet': 1, 
            'is_multi_star': 0, 'planet_density': 5.0
        })
    
    # Retornar en el ORDEN EXACTO de feature_columns
    return [base_data[col] for col in feature_columns]

# Verificar que la función funciona correctamente
test_features = create_future_data(2025, 'simple')
print(f"   - Características generadas: {len(test_features)} (debe ser {len(feature_columns)})")
print(f"   - Primeras 5 características: {test_features[:5]}")

# Simular futuros descubrimientos
future_years = list(range(2025, 2036))
future_predictions = {}
system_types = ['simple', 'multiple', 'binary', 'complex']

for year in future_years:
    year_predictions = []
    for system_type in system_types:
        try:
            future_features = create_future_data(year, system_type)
            pred = rf_model.predict([future_features])[0]
            proba = rf_model.predict_proba([future_features])[0]
            confidence = np.max(proba)
            
            year_predictions.append({
                'system_type': system_type,
                'method': le.classes_[pred],
                'confidence': confidence
            })
        except Exception as e:
            print(f"   ERROR en año {year}, tipo {system_type}: {e}")
            continue
    
    future_predictions[year] = year_predictions

# Analizar tendencias
method_trends = {method: [] for method in le.classes_}
for year in future_years:
    for method in le.classes_:
        count = sum(1 for pred in future_predictions[year] if pred['method'] == method)
        method_trends[method].append(count)

# Visualizar tendencias futuras
plt.figure(figsize=(14, 8))
for method, counts in method_trends.items():
    if sum(counts) > 0:
        plt.plot(future_years, counts, 'o-', linewidth=3, markersize=8, label=method)

plt.xlabel('Año', fontsize=12)
plt.ylabel('Frecuencia Predicha (de 4 sistemas tipo)', fontsize=12)
plt.title('PREDICCIÓN: Evolución de Métodos de Descubrimiento (2025-2035)\nBasado en Patrones Históricos y Complejidad del Sistema', 
          fontsize=14, pad=20)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(future_years, rotation=45)
plt.tight_layout()
plt.show()

# 6. ANÁLISIS DE "NICHOS" DE MÉTODOS

print("\n6. IDENTIFICANDO NICHOS ESPECIALIZADOS DE MÉTODOS:")

# Analizar en qué condiciones cada método es dominante
method_dominance = {}

for method_idx, method_name in enumerate(le.classes_):
    method_mask = (y == method_idx)
    if np.sum(method_mask) > 10:  # Solo métodos con suficientes datos
        method_data = X[method_mask]
        
        # Características promedio donde este método domina
        avg_features = method_data.mean()
        
        method_dominance[method_name] = {
            'año_promedio': avg_features['disc_year'],
            'complejidad_promedio': avg_features['system_complexity'],
            'planetas_promedio': avg_features['sy_pnum'],
            'estrellas_promedio': avg_features['sy_snum'],
            'muestras': len(method_data)
        }

# Mostrar nichos
print("   NICHOS ESPECIALIZADOS IDENTIFICADOS:")
for method, stats in method_dominance.items():
    print(f"\n    {method}:")
    print(f"      • Época: {stats['año_promedio']:.1f}")
    print(f"      • Complejidad: {stats['complejidad_promedio']:.2f}")
    print(f"      • Planetas: {stats['planetas_promedio']:.2f}")
    print(f"      • Estrellas: {stats['estrellas_promedio']:.2f}")
    print(f"      • Muestras: {stats['muestras']:,}")

# 7. SIMULACIÓN DE IMPACTO TECNOLÓGICO

print("\n7. SIMULANDO EL IMPACTO DE NUEVAS TECNOLOGÍAS:")

# Crear escenarios con diferentes niveles de complejidad
scenarios = {
    'Tecnología Actual': 1.0,
    'Mejora Moderada (+30%)': 1.3,
    'Telescopios Avanzados (+70%)': 1.7,
    'Tecnología Revolucionaria (+120%)': 2.2
}

scenario_results = {}
year_test = 2030

for scenario_name, complexity_multiplier in scenarios.items():
    scenario_predictions = []
    
    for system_type in system_types:
        try:
            features = create_future_data(year_test, system_type)
            # Aplicar multiplicador de complejidad - CORREGIDO
            complexity_idx = feature_columns.index('system_complexity')
            density_idx = feature_columns.index('planet_density')
            
            features[complexity_idx] *= complexity_multiplier
            features[density_idx] *= complexity_multiplier
            
            pred = rf_model.predict([features])[0]
            scenario_predictions.append(le.classes_[pred])
        except Exception as e:
            print(f"   ERROR en escenario {scenario_name}: {e}")
            continue
    
    scenario_results[scenario_name] = scenario_predictions

# Visualizar impacto tecnológico
scenario_df = pd.DataFrame(scenario_results, index=system_types)
scenario_counts = scenario_df.apply(pd.Series.value_counts).fillna(0)

plt.figure(figsize=(12, 8))
scenario_counts.T.plot(kind='bar', stacked=True, ax=plt.gca(), 
                      colormap='Set3', width=0.8)
plt.title('IMPACTO DE AVANCES TECNOLÓGICOS EN DETECCIÓN (2030)\n¿Cómo mejoras técnicas afectan los métodos preferidos?',
          fontsize=13, pad=20)
plt.ylabel('Número de Sistemas Tipo Predichos', fontsize=11)
plt.xlabel('Escenario Tecnológico', fontsize=11)
plt.legend(title='Método de Detección', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 8. CONCLUSIONES Y PREDICCIONES PRINCIPALES

print("\n" + "="*80)
print(" PREDICCIONES Y CONCLUSIONES PRINCIPALES ")
print("="*80)

# Análisis de dominancia futura - CORREGIDO
dominance_analysis = {}
for year in [2025, 2030, 2035]:
    if year in future_predictions:
        method_counts = {}
        for pred in future_predictions[year]:
            method = pred['method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts:
            # CORRECCIÓN: Convertir a Series de pandas correctamente
            method_series = pd.Series(method_counts)
            dominant_method = method_series.idxmax()
            dominant_count = method_series.max()
            dominance_analysis[year] = (dominant_method, dominant_count)

print("\n PREDICCIÓN DE DOMINANCIA POR AÑO:")
for year, (method, count) in dominance_analysis.items():
    percentage = (count / len(system_types)) * 100
    print(f"   • {year}: {method} ({count}/4 sistemas = {percentage:.0f}%)")

# Métodos en crecimiento vs declive - CORREGIDO
method_growth = {}
for method in le.classes_:
    if 2025 in future_predictions and 2035 in future_predictions:
        start_count = sum(1 for pred in future_predictions[2025] if pred['method'] == method)
        end_count = sum(1 for pred in future_predictions[2035] if pred['method'] == method)
        
        if start_count > 0:
            growth = ((end_count - start_count) / start_count) * 100
            method_growth[method] = growth

if method_growth:
    print("\n TENDENCIAS DE CRECIMIENTO (2025-2035):")
    for method, growth in sorted(method_growth.items(), key=lambda x: x[1], reverse=True):
        trend = "📈" if growth > 0 else "📉"
        print(f"   {trend} {method}: {growth:+.1f}%")

# Insights basados en importancia de características
top_feature = feature_importance.iloc[0]
print(f"\n INSIGHT PRINCIPAL:")
print(f"   La característica más importante es: '{top_feature['feature']}'")
print(f"   (Importancia: {top_feature['importance']:.3f})")

if dominance_analysis:
    final_dominant = dominance_analysis[2035][0] if 2035 in dominance_analysis else "No disponible"
    print(f"\n PREDICCIÓN FINAL:")
    print(f"   Para 2035, el método dominante será: {final_dominant}")
    print(f"   Basado en patrones históricos y tendencias de complejidad creciente")
