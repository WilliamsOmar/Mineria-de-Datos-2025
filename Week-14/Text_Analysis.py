from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('Dataset\PS_2025.11.03_17.08.05.csv', comment='#')

# Análisis de texto de nombres de planetas
planet_names = ' '.join(df['pl_name'].dropna().astype(str))

# Nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(planet_names)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras - Nombres de Exoplanetas')
plt.show()

# Análisis de frecuencias
words = planet_names.split()
word_freq = Counter(words)
print("Palabras más comunes en nombres de exoplanetas:")
for word, freq in word_freq.most_common(10):
    print(f"{word}: {freq}")