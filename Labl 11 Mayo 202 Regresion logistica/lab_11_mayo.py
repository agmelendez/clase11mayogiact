# -*- coding: utf-8 -*-
"""
Laboratorio 11 de mayo - Regresión Logística Binaria aplicada a gestión marina costera
Ejecutable en Google Colab.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    RocCurveDisplay, precision_recall_fscore_support, brier_score_loss
)
from sklearn.calibration import calibration_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =========================================================
# 1. GENERACIÓN DEL DATASET SINTÉTICO
# =========================================================
np.random.seed(42)
n = 180

df = pd.DataFrame({
    "id_unidad": np.arange(1, n + 1),
    "capacitacion": np.random.binomial(1, 0.48, n),
    "experiencia_anios": np.random.uniform(1, 35, n),
    "distancia_km": np.random.normal(18, 7, n),
    "indice_presion_turistica": np.random.normal(55, 15, n),
    "participacion_comunitaria": np.random.choice([1, 2, 3, 4, 5], n, p=[.10, .20, .30, .25, .15])
})

# DGP logístico: variable dependiente binaria
eta = (
    -1.20
    + 1.05 * df["capacitacion"]
    + 0.04 * df["experiencia_anios"]
    - 0.05 * df["distancia_km"]
    - 0.015 * df["indice_presion_turistica"]
    + 0.38 * df["participacion_comunitaria"]
)
df["prob_adopcion"] = 1 / (1 + np.exp(-eta))
df["adopta_buena_practica"] = np.random.binomial(1, df["prob_adopcion"])

print("Primeras filas:")
print(df.head())
print("\nPrevalencia del evento:")
print(df["adopta_buena_practica"].value_counts(normalize=True).rename("proporcion"))

# =========================================================
# 2. AJUSTE DEL MODELO LOGÍSTICO
# =========================================================
formula = (
    "adopta_buena_practica ~ capacitacion + experiencia_anios + "
    "distancia_km + indice_presion_turistica + participacion_comunitaria"
)
modelo = smf.logit(formula, data=df).fit(maxiter=200)
print(modelo.summary())

conf = modelo.conf_int()
or_table = pd.DataFrame({
    "OR": np.exp(modelo.params),
    "IC_2.5%": np.exp(conf[0]),
    "IC_97.5%": np.exp(conf[1]),
    "p_value": modelo.pvalues
})
print("\nOdds ratios e intervalos de confianza:")
print(or_table)

# =========================================================
# 3. DESEMPEÑO PREDICTIVO Y MATRIZ DE CONFUSIÓN
# =========================================================
df["p_hat"] = modelo.predict(df)
threshold = 0.50
df["pred_50"] = (df["p_hat"] >= threshold).astype(int)

print("\nMatriz de confusión con umbral 0.50:")
print(confusion_matrix(df["adopta_buena_practica"], df["pred_50"]))
print("\nReporte de clasificación:")
print(classification_report(df["adopta_buena_practica"], df["pred_50"], zero_division=0))
print("ROC-AUC:", round(roc_auc_score(df["adopta_buena_practica"], df["p_hat"]), 3))
print("Brier score:", round(brier_score_loss(df["adopta_buena_practica"], df["p_hat"]), 3))

RocCurveDisplay.from_predictions(df["adopta_buena_practica"], df["p_hat"])
plt.title("Curva ROC - Modelo logístico")
plt.show()

# =========================================================
# 4. ANÁLISIS DE UMBRALES
# =========================================================
print("\nComparación de umbrales:")
for t in [0.30, 0.40, 0.50, 0.60, 0.70]:
    pred = (df["p_hat"] >= t).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        df["adopta_buena_practica"], pred, average="binary", zero_division=0
    )
    print(f"Umbral={t:.2f} | Precisión={precision:.3f} | Sensibilidad={recall:.3f} | F1={f1:.3f}")

# =========================================================
# 5. MULTICOLINEALIDAD Y CALIBRACIÓN
# =========================================================
cols_x = ["capacitacion", "experiencia_anios", "distancia_km", "indice_presion_turistica", "participacion_comunitaria"]
X = sm.add_constant(df[cols_x])
vif = pd.DataFrame({
    "variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print("\nVIF:")
print(vif)

prob_true, prob_pred = calibration_curve(df["adopta_buena_practica"], df["p_hat"], n_bins=6)
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Probabilidad predicha")
plt.ylabel("Frecuencia observada")
plt.title("Curva de calibración")
plt.show()

# =========================================================
# 6. VALIDACIÓN TRAIN/TEST Y COMPARACIÓN CON INTERACCIÓN
# =========================================================
train, test = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["adopta_buena_practica"]
)
modelo_train = smf.logit(formula, data=train).fit(maxiter=200, disp=False)
test = test.copy()
test["p_hat"] = modelo_train.predict(test)
print("\nROC-AUC en prueba:", round(roc_auc_score(test["adopta_buena_practica"], test["p_hat"]), 3))

formula_inter = (
    "adopta_buena_practica ~ capacitacion * participacion_comunitaria + "
    "experiencia_anios + distancia_km + indice_presion_turistica"
)
modelo_inter = smf.logit(formula_inter, data=df).fit(maxiter=200, disp=False)
print("\nComparación AIC:")
print(pd.DataFrame({
    "modelo": ["Base", "Con interacción"],
    "AIC": [modelo.aic, modelo_inter.aic],
    "BIC": [modelo.bic, modelo_inter.bic]
}))

# Exportar datos simulados
archivo = "datos_simulados_regresion_logistica.csv"
df.to_csv(archivo, index=False)
print(f"\nArchivo exportado: {archivo}")
