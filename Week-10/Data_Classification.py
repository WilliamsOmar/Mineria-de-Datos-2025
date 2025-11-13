import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Configuración de visualización
plt.style.use('seaborn-v0_8')
print("=== CLASIFICACIÓN KNN - MÉTODOS DE DESCUBRIMIENTO ===\n")

# Cargar datos
df = pd.read_csv('Dataset\PS_2025.11.03_17.08.05.csv', comment='#')

print("Análisis inicial:")
print(f"   - Total de registros: {len(df)}")
print(f"   - Métodos únicos: {df['discoverymethod'].nunique()}")
print(f"   - Distribución de métodos:")
method_counts = df['discoverymethod'].value_counts()
for method, count in method_counts.items():
    print(f"     * {method}: {count} registros")

# Filtrar métodos con suficientes muestras (al menos 10)
min_samples = 10
valid_methods = method_counts[method_counts >= min_samples].index
df_filtered = df[df['discoverymethod'].isin(valid_methods)]

print(f"\nDatos para clasificación:")
print(f"   - Métodos con ≥{min_samples} muestras: {len(valid_methods)}")
print(f"   - Registros totales: {len(df_filtered)}")
print(f"   - Métodos excluidos por pocos datos: {len(method_counts) - len(valid_methods)}")

# Codificar la variable objetivo
le = LabelEncoder()
y_class = le.fit_transform(df_filtered['discoverymethod'])

# Características para el modelo
X_class = df_filtered[['disc_year', 'sy_snum', 'sy_pnum']]

print(f"\nCaracterísticas del modelo:")
print(f"   - Características: {list(X_class.columns)}")
print(f"   - Clases: {len(le.classes_)}")
print(f"   - Nombres de clases: {list(le.classes_)}")

# Dividir datos - stratify para mantener proporciones
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
)

print(f"\nDivisión de datos:")
print(f"   - Entrenamiento: {X_train.shape[0]} muestras")
print(f"   - Prueba: {X_test.shape[0]} muestras")
print(f"   - Clases en entrenamiento: {len(np.unique(y_train))}")
print(f"   - Clases en prueba: {len(np.unique(y_test))}")

# Modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred_knn)
print(f"\nResultados de clasificación:")
print(f"   Precisión KNN: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Obtener las clases que realmente están en y_test
unique_classes = np.unique(y_test)
available_class_names = le.classes_[unique_classes]

print(f"\nReporte de clasificación:")
print(f"   - Clases presentes en prueba: {len(unique_classes)}")
print(f"   - Clases disponibles: {list(available_class_names)}")

# Reporte de clasificación
print(classification_report(y_test, y_pred_knn, target_names=available_class_names, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=available_class_names, 
            yticklabels=available_class_names, ax=axes[0])
axes[0].set_title('Matriz de Confusión - Clasificación KNN')
axes[0].set_xlabel('Predicción')
axes[0].set_ylabel('Real')
axes[0].tick_params(axis='x', rotation=45)

# Precisión por clase
class_accuracy = cm.diagonal() / cm.sum(axis=1)
y_pos = np.arange(len(available_class_names))

axes[1].barh(y_pos, class_accuracy, color='lightgreen')
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(available_class_names)
axes[1].set_xlabel('Precisión')
axes[1].set_title('Precisión por Método de Descubrimiento')
axes[1].axvline(x=accuracy, color='red', linestyle='--', 
                label=f'Precisión general: {accuracy:.3f}')
axes[1].legend()

plt.tight_layout()
plt.show()

print("\nAnálisis detallado por clase:")
for i, class_name in enumerate(available_class_names):
    class_idx = np.where(le.classes_ == class_name)[0][0]
    class_mask = (y_test == class_idx)
    
    if np.sum(class_mask) > 0:  # Solo si hay ejemplos en prueba
        class_accuracy = accuracy_score(y_test[class_mask], y_pred_knn[class_mask])
        support = np.sum(class_mask)
        
        print(f"   - {class_name}:")
        print(f"     * Precisión: {class_accuracy:.3f}")
        print(f"     * Muestras en prueba: {support}")
        print(f"     * Representación: {support/len(y_test)*100:.1f}%")

print("\nPredicciones de ejemplo:")
print("   (Año, Estrellas, Planetas) -> Método Predicho")

examples = [
    [1998, 1, 1],   # Época temprana
    [2015, 1, 3],   # Época Kepler
    [2022, 2, 1],   # Época moderna
]

for i, example in enumerate(examples, 1):
    prediction = knn.predict([example])[0]
    predicted_method = le.classes_[prediction]
    probabilities = knn.predict_proba([example])[0]
    confidence = np.max(probabilities)
    
    print(f"   Ejemplo {i}: {example}")
    print(f"     → {predicted_method} (confianza: {confidence:.3f})")
    
    # Mostrar top 3 predicciones
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    for j, idx in enumerate(top_3_idx):
        if probabilities[idx] > 0.05:  # Solo mostrar si probabilidad > 5%
            print(f"       {j+1}. {le.classes_[idx]}: {probabilities[idx]:.3f}")