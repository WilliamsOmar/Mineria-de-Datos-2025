import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

"""
Planeta (pl_name) → Pertenece a → Sistema Estelar (sy_snum)
Planeta → Descubierto por → Método (discoverymethod)
Planeta → Descubierto en → Año (disc_year)
Planeta → Publicado en → Fecha (releasedate)
"""

# Cargar datos
df = pd.read_csv('PS_2025.11.03_17.08.05.csv', comment='#')

print("=== ESTADÍSTICAS DESCRIPTIVAS ===")
print(f"Total de registros: {len(df)}/n")
print(f"Total de planetas únicos: {df['pl_name'].nunique()}/n")
print(f"Rango de años: {df['disc_year'].min()} - {df['disc_year'].max()}/n")
print(f"Métodos de descubrimiento: {df['discoverymethod'].unique()}/n")

# Estadísticas por método de descubrimiento
print("\n=== ESTADÍSTICAS POR MÉTODO DE DESCUBRIMIENTO ===")
method_stats = df.groupby('discoverymethod').agg({
    'pl_name': 'nunique',
    'disc_year': ['count', 'min', 'max', 'mean'],
    'sy_snum': ['mean', 'min', 'max']
}).round(2)
print(method_stats)

# Estadísticas por número de estrellas
print("\n=== ESTADÍSTICAS POR NÚMERO DE ESTRELLAS ===")
star_stats = df.groupby('sy_snum').agg({
    'disc_year': ['count', 'min', 'max', 'mean'],
    'discoverymethod': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
})
print(star_stats)