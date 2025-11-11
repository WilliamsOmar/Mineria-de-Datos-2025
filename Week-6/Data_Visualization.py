import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Cargar datos
df = pd.read_csv('PS_2025.11.03_17.08.05.csv', comment='#')

# Configuración de gráficos
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(3, 2, figsize=(18, 12))

# 1. Distribución de métodos de descubrimiento (Pie chart)
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

# 2. Evolución temporal de descubrimientos por método
yearly_methods = df.groupby(['disc_year', 'discoverymethod']).size().unstack(fill_value=0)

# Tomamos solo los métodos principales para mayor claridad
top_methods = method_counts.head(4).index  # Top 4 métodos
top_methods_sorted = list(top_methods)[::-1]  # Esto invierte el orden
yearly_top = yearly_methods[top_methods_sorted]

axes[0,1].stackplot(yearly_top.index, yearly_top.T, labels=top_methods_sorted, alpha=0.8)
axes[0,1].set_xlabel('Año de Descubrimiento')
axes[0,1].set_ylabel('Número de Planetas Descubiertos')
axes[0,1].set_title('Evolución de Descubrimientos por Método')
axes[0,1].legend(loc='upper left', fontsize=8)
axes[0,1].grid(True, alpha=0.3)

# 3. Boxplot de años por método
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

# 4. Scatter plot: Año vs Número de Estrellas
scatter = axes[1,1].scatter(df['disc_year'], df['sy_pnum'], 
                           c=df['sy_snum'], cmap='viridis', 
                           alpha=0.6, s=30)
axes[1,1].set_xlabel('Año de Descubrimiento')
axes[1,1].set_ylabel('Número de Planetas en el Sistema')
axes[1,1].set_title('Relación: Año vs Complejidad del Sistema')
plt.colorbar(scatter, ax=axes[1,1], label='Número de Estrellas')

# 5. Gráfico de barras: Planetas por año
yearly_planets = df.groupby('disc_year')['pl_name'].nunique()
axes[2,0].bar(yearly_planets.index, yearly_planets.values)
axes[2,0].set_xlabel('Año')
axes[2,0].set_ylabel('Planetas Descubiertos')
axes[2,0].set_title('Planetas Descubiertos por Año')

# 6. Heatmap de correlación (para variables numéricas)
numeric_df = df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[2,1])
    axes[2,1].set_title('Matriz de Correlación')

plt.tight_layout()
plt.show()