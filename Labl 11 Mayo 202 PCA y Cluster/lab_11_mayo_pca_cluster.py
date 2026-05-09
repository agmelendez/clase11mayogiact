# -*- coding: utf-8 -*-
"""Laboratorio PCA + Clúster aplicado a gestión marina costera.

Este archivo está diseñado para ejecutarse en Google Colab o localmente.
Genera datos sintéticos, ejecuta PCA, selecciona componentes, calcula clústeres,
valida k mediante silhouette y produce salidas interpretables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

SEED = 123
np.random.seed(SEED)
N = 180

variables = [
    "presion_turistica",
    "esfuerzo_pesquero",
    "calidad_agua",
    "biodiversidad",
    "gobernanza_local",
    "cumplimiento_normativo",
    "conflictividad",
    "infraestructura",
]

# 1. Generación de factores latentes
F_presion = np.random.normal(0, 1, N)
F_condicion = np.random.normal(0, 1, N)
F_capacidad = np.random.normal(0, 1, N)

# 2. Dataset sintético con estructura sustantiva
df = pd.DataFrame({
    "presion_turistica": 60 + 15 * (0.70 * F_presion + np.random.normal(0, 0.55, N)),
    "esfuerzo_pesquero": 55 + 18 * (0.68 * F_presion + np.random.normal(0, 0.60, N)),
    "conflictividad": 48 + 16 * (0.62 * F_presion - 0.25 * F_capacidad + np.random.normal(0, 0.65, N)),
    "calidad_agua": 70 + 12 * (0.70 * F_condicion - 0.30 * F_presion + np.random.normal(0, 0.55, N)),
    "biodiversidad": 66 + 13 * (0.75 * F_condicion - 0.35 * F_presion + np.random.normal(0, 0.60, N)),
    "gobernanza_local": 52 + 14 * (0.72 * F_capacidad + np.random.normal(0, 0.55, N)),
    "cumplimiento_normativo": 58 + 16 * (0.65 * F_capacidad - 0.20 * F_presion + np.random.normal(0, 0.65, N)),
    "infraestructura": 62 + 15 * (0.55 * F_capacidad + np.random.normal(0, 0.70, N)),
})

print("Primeras filas del dataset sintético:")
print(df.head())
print("\nMatriz de correlaciones redondeada:")
print(df[variables].corr().round(2))

# 3. Estandarización
scaler = StandardScaler()
X = scaler.fit_transform(df[variables])

# 4. PCA completo para revisar varianza
pca_full = PCA().fit(X)
var_ratio = pca_full.explained_variance_ratio_
var_acum = np.cumsum(var_ratio)

print("\nVarianza explicada por componente:")
for i, v in enumerate(var_ratio, start=1):
    print(f"PC{i}: {v:.3f} | acumulada: {var_acum[i-1]:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(var_ratio) + 1), var_acum, marker="o")
plt.axhline(0.75, linestyle="--")
plt.title("PCA: varianza acumulada")
plt.xlabel("Número de componentes")
plt.ylabel("Varianza acumulada")
plt.show()

# 5. PCA final
n_comp = int(np.argmax(var_acum >= 0.75) + 1)
pca = PCA(n_components=n_comp).fit(X)
scores = pca.transform(X)
loadings = pd.DataFrame(
    pca.components_.T,
    index=variables,
    columns=[f"PC{i}" for i in range(1, n_comp + 1)],
)
print(f"\nComponentes retenidos: {n_comp}")
print("\nCargas principales:")
print(loadings.round(3))

# 6. Selección de número de clústeres
silhouette = {}
for k in range(2, 8):
    labels = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit_predict(scores)
    silhouette[k] = silhouette_score(scores, labels)

print("\nSilhouette por k:")
print(silhouette)
k_opt = max(silhouette, key=silhouette.get)
print(f"K óptimo sugerido: {k_opt}")

# 7. K-means final
kmeans = KMeans(n_clusters=k_opt, n_init=100, random_state=SEED)
df["cluster"] = kmeans.fit_predict(scores)

print("\nTamaño de los clústeres:")
print(df["cluster"].value_counts().sort_index())
print("\nCentroides por clúster en escala original:")
print(df.groupby("cluster")[variables].mean().round(2))

plt.figure(figsize=(8, 6))
y_coord = scores[:, 1] if n_comp > 1 else np.zeros(N)
plt.scatter(scores[:, 0], y_coord, c=df["cluster"], alpha=0.75)
plt.xlabel("PC1")
plt.ylabel("PC2" if n_comp > 1 else "0")
plt.title("Clústeres proyectados sobre componentes principales")
plt.show()

# 8. Contraste jerárquico
Z = linkage(scores, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=20)
plt.title("Dendrograma jerárquico sobre scores PCA")
plt.xlabel("Grupos agregados")
plt.ylabel("Distancia")
plt.show()

# 9. Interpretación sugerida
print("\nGuía de interpretación:")
print("1. Nombre PC1, PC2, etc. según las variables con mayores cargas absolutas.")
print("2. Etiquete clústeres comparando centroides: alta presión, buena condición, baja capacidad, etc.")
print("3. No convierta automáticamente clústeres estadísticos en categorías de política pública sin validación territorial.")
