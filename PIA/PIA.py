# =============================================================================
# ANÁLISIS INTEGRADO: DESCUBRIMIENTO DE PATRONES OCULTOS EN EXOPLANETAS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8')

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

# 2. MODELO PREDICTIVO: RANDOM FOREST

print("\n2. ENTRENANDO MODELO PREDICTIVO (Random Forest)...")

# Engineering de características
features_df = df_filtered.copy()
le = LabelEncoder()
features_df['method_encoded'] = le.fit_transform(features_df['discoverymethod'])
features_df['system_complexity'] = features_df['sy_pnum'] * (features_df['sy_snum'] + 1)

# Características para el modelo
X = features_df[['disc_year', 'system_complexity', 'sy_snum', 'sy_pnum']]
y = features_df['method_encoded']

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluación
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Importancia de características
feature_importance = rf_model.feature_importances_
year_importance = feature_importance[0]
complexity_importance = feature_importance[1]
importance_ratio = year_importance / complexity_importance

print(f"      • Precisión del modelo: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"      • Importancia 'Año': {year_importance:.4f}")
print(f"      • Importancia 'Complejidad': {complexity_importance:.4f}")
print(f"      • Ratio Año/Complejidad: {importance_ratio:.2f}x")

# 3. ANÁLISIS DE CLUSTERING: K-MEANS

# Preparar datos para clustering
cluster_features = ['disc_year', 'system_complexity', 'method_encoded']
cluster_data = features_df[cluster_features]

# Estandarizar
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# Añadir clusters al dataframe
features_df['cluster'] = clusters

# Análisis de clusters por método
cluster_analysis = features_df.groupby('cluster')['disc_year'].agg(['mean', 'std'])
method_by_cluster = features_df.groupby('cluster')['discoverymethod'].apply(lambda x: x.mode()[0])

print("      • 4 clusters identificados (silhouette: {:.3f})".format(
    silhouette_score(scaled_data, clusters)))
print("      • Caracterización de clusters:")
for cluster_id in range(4):
    cluster_data = features_df[features_df['cluster'] == cluster_id]
    avg_year = cluster_data['disc_year'].mean()
    main_method = cluster_data['discoverymethod'].mode()[0]
    print(f"        Cluster {cluster_id}: Año promedio={avg_year:.0f}, Método principal={main_method}")

# 4. ANOVA

print("\n   PRUEBA ESTADÍSTICA: DIFERENCIAS TEMPORALES (ANOVA)")

# ANOVA: ¿Los años son diferentes entre métodos?
method_groups = []
for method in valid_methods:
    method_years = features_df[features_df['discoverymethod'] == method]['disc_year']
    if len(method_years) > 1:
        method_groups.append(method_years)

# Ejecutar ANOVA
f_stat, p_value = stats.f_oneway(*method_groups)

print(f"      • F-statistic: {f_stat:.4f}")
print(f"      • p-value: {p_value:.4e}")
print(f"      • Conclusión: {'Diferencia significativa' if p_value < 0.05 else 'Sin diferencia significativa'}")

top_methods = method_counts.head(3).index

# 5. VISUALIZACIONES

print("\n 3. VISUALIZACIÓN DE HALLAZGOS:")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 5.1 Importancia de características
feature_names = ['Año', 'Complejidad', 'Estrellas', 'Planetas']
axes[0,0].bar(feature_names, feature_importance, color=['red', 'blue', 'green', 'orange'])
axes[0,0].set_title('IMPORTANCIA RELATIVA: AÑO vs COMPLEJIDAD', fontsize=12, pad=15)
axes[0,0].set_ylabel('Importancia', fontsize=11)
axes[0,0].text(0, year_importance*0.9, f'{year_importance:.3f}', ha='center', fontweight='bold', color='white')
axes[0,0].text(1, complexity_importance*0.9, f'{complexity_importance:.3f}', ha='center', fontweight='bold', color='white')
axes[0,0].text(0.5, max(feature_importance)*1.05, f'Ratio: {importance_ratio:.1f}x', 
              ha='center', fontsize=11, fontweight='bold', color='red')

# 5.2 Clusters en espacio 2D
scatter = axes[0,1].scatter(features_df['disc_year'], features_df['system_complexity'], 
                           c=features_df['cluster'], cmap='viridis', alpha=0.6, s=30)
axes[0,1].set_xlabel('Año de Descubrimiento', fontsize=11)
axes[0,1].set_ylabel('Complejidad del Sistema', fontsize=11)
axes[0,1].set_title('CLUSTERS: AÑO vs COMPLEJIDAD\n(Agrupación natural de métodos)', fontsize=12, pad=15)
plt.colorbar(scatter, ax=axes[0,1], label='Cluster')

# 5.3 Evolución temporal por método
for method in top_methods:
    method_data = features_df[features_df['discoverymethod'] == method]
    yearly_counts = method_data.groupby('disc_year').size()
    axes[1,0].plot(yearly_counts.index, yearly_counts.values, 'o-', label=method, markersize=4)

axes[1,0].set_xlabel('Año', fontsize=11)
axes[1,0].set_ylabel('Descubrimientos', fontsize=11)
axes[1,0].set_title('EVOLUCIÓN TEMPORAL POR MÉTODO\n(Dominios temporales claros)', fontsize=12, pad=15)
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 5.4 Boxplot de años por método
method_data_list = []
method_labels = []
for method in top_methods:
    method_years = features_df[features_df['discoverymethod'] == method]['disc_year']
    if len(method_years) > 10:
        method_data_list.append(method_years)
        method_labels.append(method)

box = axes[1,1].boxplot(method_data_list, labels=method_labels, patch_artist=True)
colors = ['lightcoral', 'lightblue', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1,1].set_ylabel('Año de Descubrimiento', fontsize=11)
axes[1,1].set_title('DISTRIBUCIÓN TEMPORAL POR MÉTODO\n(ANOVA: p-value = {:.2e})'.format(p_value), 
                    fontsize=12, pad=15)
axes[1,1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 6. CONCLUSIONES 

print("\n" + "="*80)
print(" CONCLUSIONES PRINCIPALES ")
print("="*80)

print("\n1. HALLAZGO CONFIRMADO:")
print(f" 'El AÑO es {importance_ratio:.1f} veces más importante que la COMPLEJIDAD'")
print(f"  Evidencia: Importancia Año={year_importance:.3f} vs Complejidad={complexity_importance:.3f}")

print("\n2. EVIDENCIA ESTADÍSTICA CONSOLIDADA:")
print(f"  PREDICCIÓN: Modelo Random Forest con {accuracy*100:.1f}% de precisión")
print(f"  CLUSTERING: 4 grupos naturales (silhouette={silhouette_score(scaled_data, clusters):.3f})")
print(f"  ANOVA: Diferencias significativas entre métodos (p-value={p_value:.2e})")

print("\n3. IMPLICACIONES CIENTÍFICAS:")
print(f"  La tecnología disponible determina el método más que las características del sistema")
print(f"  Cada método tiene su 'ventana temporal' óptima")