from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== CLUSTERING DE SISTEMAS EXOPLANETARIOS ===\n")

# Cargar datos
df = pd.read_csv('Dataset/PS_2025.11.03_17.08.05.csv', comment='#')

print("Información inicial:")
print(f"   - Total de registros: {len(df):,}")
print(f"   - Planetas únicos: {df['pl_name'].nunique():,}")
print(f"   - Métodos de descubrimiento: {df['discoverymethod'].nunique()}")
print(f"   - Rango de años: {df['disc_year'].min()} - {df['disc_year'].max()}\n")

# Crear copia del DataFrame
model_df = df.copy()

# Codificar método de descubrimiento si no existe
if 'method_encoded' not in model_df.columns:
    le = LabelEncoder()
    model_df['method_encoded'] = le.fit_transform(model_df['discoverymethod'])
    print("Método de descubrimiento codificado:")
    print(f"   - Métodos únicos: {len(le.classes_)}")
    print(f"   - Clases: {list(le.classes_)}")

# Seleccionar características para clustering
features = ['disc_year', 'sy_snum', 'sy_pnum', 'method_encoded']
cluster_data = model_df[features]

print(f"\nCaracterísticas para clustering:")
print(f"   - Año de descubrimiento (disc_year)")
print(f"   - Número de estrellas (sy_snum)")
print(f"   - Número de planetas (sy_pnum)")
print(f"   - Método de descubrimiento codificado (method_encoded)")
print(f"   - Forma de los datos: {cluster_data.shape}")

# Estandarizar características
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

print(f"\n4. ESTADÍSTICAS ANTES/DESPUÉS DE ESTANDARIZAR:")
original_stats = cluster_data.describe().loc[['mean', 'std']]
print("   Original:")
print(original_stats.round(3))

scaled_stats = pd.DataFrame(scaled_data, columns=features).describe().loc[['mean', 'std']]
print("\n   Estandarizado:")
print(scaled_stats.round(3))

print("\nBuscando número óptimo de clusters...")

# Método del codo
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Graficar método del codo
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo\n(Punto óptimo ≈ donde la curva se flexiona)')
plt.grid(True, alpha=0.3)

# Método de la silueta
silhouette_scores = []
k_range_silhouette = []  # Nuevo: solo k válidos para silhouette

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    
    # Silhouette score solo se calcula para k > 1
    if k > 1:
        score = silhouette_score(scaled_data, labels)
        silhouette_scores.append(score)
        k_range_silhouette.append(k)  # Guardar k correspondiente

# Graficar ambos métodos
plt.figure(figsize=(12, 5))

# Gráfico 1: Método del codo
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo\n(Punto óptimo ≈ donde la curva se flexiona)')
plt.grid(True, alpha=0.3)

# Añadir anotaciones para puntos de inflexión
elbow_point = 4  # Puedes ajustar esto basado en el gráfico
plt.axvline(x=elbow_point, color='red', linestyle='--', alpha=0.7, 
           label=f'Posible codo: k={elbow_point}')
plt.legend()

# Gráfico 2: Método de la silueta 
plt.subplot(1, 2, 2)
plt.plot(k_range_silhouette, silhouette_scores, 'ro-', linewidth=2, markersize=6)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Puntuación de Silueta')
plt.title('Método de la Silueta\n(Mayor = Mejor)')
plt.grid(True, alpha=0.3)

# Resaltar el mejor k según silhouette
best_k_silhouette = k_range_silhouette[np.argmax(silhouette_scores)]
best_score = np.max(silhouette_scores)
plt.axvline(x=best_k_silhouette, color='green', linestyle='--', alpha=0.7,
           label=f'Mejor k: {best_k_silhouette} (score: {best_score:.3f})')
plt.legend()

plt.tight_layout()
plt.show()

# Seleccionar k óptimo
print(f"   - K sugerido por método del codo: {elbow_point}")
print(f"   - K sugerido por método de silueta: {best_k_silhouette}")

# Usar el que tenga mejor silueta o el codo más claro
if best_score > 0.5:  # Si la silueta es buena
    optimal_k = best_k_silhouette
else:
    optimal_k = elbow_point

print(f"   - K seleccionado para el modelo: {optimal_k}")
print(f"   - Puntuación de silueta para k={optimal_k}: {best_score:.4f}")

# Entrenar modelo KMeans con k óptimo
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# Añadir clusters al dataframe
model_df['cluster'] = clusters

print(f"\nResultados del clustering (k={optimal_k}):")
print(f"   - Tamaño de cada cluster:")
for cluster_id in range(optimal_k):
    cluster_size = np.sum(clusters == cluster_id)
    percentage = (cluster_size / len(clusters)) * 100
    print(f"     * Cluster {cluster_id}: {cluster_size:,} registros ({percentage:.1f}%)")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Clusters en espacio 2D (Año vs Planetas)
scatter1 = axes[0,0].scatter(model_df['disc_year'], model_df['sy_pnum'], 
                           c=model_df['cluster'], cmap='viridis', alpha=0.6, s=30)
axes[0,0].set_xlabel('Año de Descubrimiento')
axes[0,0].set_ylabel('Número de Planetas en el Sistema')
axes[0,0].set_title('Clusters: Año vs Complejidad del Sistema')
plt.colorbar(scatter1, ax=axes[0,0], label='Cluster')

# Clusters en espacio 2D (Año vs Estrellas)
scatter2 = axes[0,1].scatter(model_df['disc_year'], model_df['sy_snum'], 
                           c=model_df['cluster'], cmap='plasma', alpha=0.6, s=30)
axes[0,1].set_xlabel('Año de Descubrimiento')
axes[0,1].set_ylabel('Número de Estrellas en el Sistema')
axes[0,1].set_title('Clusters: Año vs Tipo de Sistema Estelar')
plt.colorbar(scatter2, ax=axes[0,1], label='Cluster')

# Distribución de métodos por cluster
cluster_methods = pd.crosstab(model_df['cluster'], model_df['discoverymethod'])
cluster_methods_percent = cluster_methods.div(cluster_methods.sum(axis=1), axis=0) * 100

# Tomar solo métodos principales para mejor visualización
top_methods = model_df['discoverymethod'].value_counts().head(4).index
cluster_methods_top = cluster_methods_percent[top_methods]

cluster_methods_top.plot(kind='bar', ax=axes[1,0], width=0.8)
axes[1,0].set_xlabel('Cluster')
axes[1,0].set_ylabel('Porcentaje (%)')
axes[1,0].set_title('Distribución de Métodos por Cluster\n(Solo métodos principales)')
axes[1,0].legend(title='Método', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1,0].tick_params(axis='x', rotation=0)

# Características promedio por cluster
cluster_means = model_df.groupby('cluster')[features].mean()

# Heatmap de características por cluster
sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='YlOrRd', 
            ax=axes[1,1], cbar_kws={'label': 'Valor Promedio'})
axes[1,1].set_title('Características Promedio por Cluster\n(Estandarizado)')
axes[1,1].set_ylabel('Característica')

plt.tight_layout()
plt.show()

print("\nAnálisis detallado de clusters:")

# Centroides en escala original (aproximados)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids_original, columns=features)

print("\n   Centroides (valores aproximados en escala original):")
print(centroids_df.round(2))

print("\n   Interpretacion de clusters:")
for cluster_id in range(optimal_k):
    cluster_data = model_df[model_df['cluster'] == cluster_id]
    
    avg_year = cluster_data['disc_year'].mean()
    avg_stars = cluster_data['sy_snum'].mean()
    avg_planets = cluster_data['sy_pnum'].mean()
    main_method = cluster_data['discoverymethod'].mode().iloc[0]
    
    print(f"\n   Cluster {cluster_id}:")
    print(f"     • Año promedio: {avg_year:.1f}")
    print(f"     • Estrellas promedio: {avg_stars:.2f}")
    print(f"     • Planetas promedio: {avg_planets:.2f}")
    print(f"     • Método principal: {main_method}")
    print(f"     • Tamaño: {len(cluster_data):,} registros")
    
    # Caracterizar el cluster
    if avg_year < 2005:
        era = "ERA TEMPRANA"
    elif avg_year < 2015:
        era = "ERA INTERMEDIA" 
    else:
        era = "ERA MODERNA"
        
    if avg_planets > 2:
        complexity = "SISTEMAS COMPLEJOS"
    elif avg_planets > 1:
        complexity = "SISTEMAS MÚLTIPLES"
    else:
        complexity = "SISTEMAS SIMPLES"
        
    print(f"     • Caracterización: {era} + {complexity}")

# Evaluación del clustering
silhouette_avg = silhouette_score(scaled_data, clusters)
print(f"\nEvaluación del clustering:")
print(f"   - Puntuación de silueta: {silhouette_avg:.4f}")

if silhouette_avg > 0.7:
    print("   Estructura fuerte")
elif silhouette_avg > 0.5:
    print("   Estructura razonable") 
elif silhouette_avg > 0.25:
    print("   Estructura débil")
else:
    print("   Sin estructura clara")

print(f"\nPredicciones de ejemplo - ¿A qué cluster pertenecerían?")

new_examples = [
    [1995, 1, 1, 1],   # Sistema antiguo simple
    [2010, 1, 3, 5],   # Sistema Kepler múltiple
    [2020, 2, 1, 5],   # Sistema binario moderno
    [2023, 1, 5, 5]    # Sistema complejo actual
]

example_df = pd.DataFrame(new_examples, columns=features)
examples_scaled = scaler.transform(example_df)
predicted_clusters = kmeans.predict(examples_scaled)

for i, (example, cluster) in enumerate(zip(new_examples, predicted_clusters)):
    print(f"   Ejemplo {i+1}: {example[:3]} → Cluster {cluster}")

print(f"\nConclusiones:")
print(f"   Clustering completado exitosamente con {optimal_k} clusters")
print(f"   Se identificaron patrones temporales y estructurales")
print(f"   Los clusters representan diferentes 'eras' y tipos de sistemas")
print(f"   Calidad del clustering: {silhouette_avg:.3f} (escala: -1 a 1)")