import pandas as pd
from scipy import stats

# Cargar datos
df = pd.read_csv('Dataset\PS_2025.11.03_17.08.05.csv', comment='#')

# ANOVA: ¿Los años de descubrimiento son diferentes entre métodos?
methods = df['discoverymethod'].unique()
method_groups = [df[df['discoverymethod'] == method]['disc_year'] for method in methods]

# ANOVA paramétrico
f_stat, p_value = stats.f_oneway(*method_groups)
print(f"ANOVA - F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

# Si ANOVA es significativo, hacer pruebas post-hoc t-test
if p_value < 0.05:
    print("\nPruebas T post-hoc:")
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            t_stat, p_val = stats.ttest_ind(method_groups[i], method_groups[j])
            print(f"{methods[i]} vs {methods[j]}: p-value = {p_val:.4f}")
            """
            p < 0.05: Diferencia significativa
            p < 0.01: Diferencia muy significativa
            p < 0.001: Diferencia altamente significativa 
            p > 0.05: No hay diferencia significativa 
            """

# Kruskal-Wallis (no paramétrico). Confirma diferencias significativas
h_stat, kw_pvalue = stats.kruskal(*method_groups)
print(f"\nKruskal-Wallis - H-statistic: {h_stat:.4f}, p-value: {kw_pvalue:.4f}")