from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Cargar datos
df = pd.read_csv('Dataset\PS_2025.11.03_17.08.05.csv', comment='#')

# Crear copia del DataFrame
model_df = df.copy()

# Configuración de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

#### PREPARACION PARA CLUSTERING ####

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

# -- Distribución de métodos de descubrimiento (Pie chart)
method_counts = df['discoverymethod'].value_counts()
total = method_counts.sum()

# Filtrar métodos con menos del 2% y agruparlos en "Otros"
threshold = 0.02  # 2%
major_methods = method_counts[method_counts / total >= threshold]
minor_methods = method_counts[method_counts / total < threshold]

if len(minor_methods) > 0:
    major_methods['Otros'] = minor_methods.sum()

# Crear pie chart con etiquetas filtradas
def custom_autopct(pct):
    return f'{pct:.1f}%' if pct >= threshold*100 else ''

wedges, texts, autotexts = axes[0,0].pie(
    major_methods.values, 
    labels=major_methods.index, 
    autopct=custom_autopct,
    startangle=90,
    labeldistance=1.1
)

# Mejorar la legibilidad de las etiquetas
for text in texts:
    text.set_fontsize(9)
    text.set_ha('center')
    
for autotext in autotexts:
    autotext.set_fontsize(8)
    autotext.set_color('white')
    autotext.set_fontweight('bold')

axes[0,0].set_title('Distribución de Métodos de Descubrimiento\n(Solo métodos > 2%)', pad=20)

# -- Boxplot de años por método
top_4_methods = method_counts.head(4).index

box_plot_data = []
box_labels = []

for method in top_4_methods:
    method_data = df[df['discoverymethod'] == method]['disc_year']
    if len(method_data) > 1:  # Solo métodos con suficientes datos
        box_plot_data.append(method_data)
        box_labels.append(method)

# Crear el boxplot
box = axes[1,0].boxplot(box_plot_data, 
                       labels=box_labels, 
                       patch_artist=True)

# Colorear las cajas
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700']  # Colores distintos para cada método
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Mejorar la presentación
axes[1,0].set_ylabel('Año de Descubrimiento')
axes[1,0].set_title('Distribución de Años por los 4 Métodos Más Usados', fontsize=12, pad=15)
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(True, alpha=0.3, axis='y')

# Clusters en espacio 2D (Año vs Planetas)
scatter1 = axes[0,1].scatter(model_df['disc_year'], model_df['sy_pnum'], 
                           c=model_df['cluster'], cmap='viridis', alpha=0.6, s=30)
axes[0,1].set_xlabel('Año de Descubrimiento')
axes[0,1].set_ylabel('Número de Planetas en el Sistema')
axes[0,1].set_title('Clusters: Año vs Complejidad del Sistema')
plt.colorbar(scatter1, ax=axes[0,1], label='Cluster')

# Clusters en espacio 2D (Año vs Estrellas)
scatter2 = axes[1,1].scatter(model_df['disc_year'], model_df['sy_snum'], 
                           c=model_df['cluster'], cmap='plasma', alpha=0.6, s=30)
axes[1,1].set_xlabel('Año de Descubrimiento')
axes[1,1].set_ylabel('Número de Estrellas en el Sistema')
axes[1,1].set_title('Clusters: Año vs Tipo de Sistema Estelar')
plt.colorbar(scatter2, ax=axes[1,1], label='Cluster')

plt.tight_layout()
plt.show()