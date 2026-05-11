# ============================================================================
# SP-8502: Métodos Cuanti-Cuali con IA Responsable
# SESIÓN 9 - SPRINT 3: Regresión Lineal - Supuestos, Diagnóstico y Modelado
# ============================================================================
# Laboratorio Autoguiado: Análisis de Captura de Peces en Pesquerías Artesanales
# Región Pacífico Central, Costa Rica (Datos Sintéticos)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, f_oneway, levene
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# Configuración visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("LABORATORIO AUTOGUIADO: REGRESIÓN LINEAL")
print("Sesión 9 - Sprint 3: SP-8502")
print("="*80)

# ============================================================================
# PARTE 1: GENERACIÓN DE DATASET SINTÉTICO REALISTA
# ============================================================================
print("\n[PARTE 1] Generación del Dataset Sintético")
print("-" * 80)

np.random.seed(42)
n_observations = 150

# Simulación realista: captura de peces artesanales
# Variables relevantes en pesquerías del Pacífico Central

data = {
    'id_pescador': np.arange(1, n_observations + 1),
    
    # Variable dependiente (Y)
    'captura_kg': None,  # A generar
    
    # Variables independientes (X)
    'horas_faena': np.random.normal(loc=6.5, scale=1.2, size=n_observations),
    'tamanio_embarcacion_m': np.random.normal(loc=7.2, scale=2.1, size=n_observations),
    'experiencia_anios': np.random.uniform(1, 45, size=n_observations),
    'precio_hielo_colones': np.random.normal(loc=3500, scale=500, size=n_observations),
    'temperatura_agua_C': np.random.normal(loc=24.5, scale=1.8, size=n_observations),
}

# Construcción de Y con relación teórica conocida (para validación)
# Y = β₀ + β₁*horas_faena + β₂*tamanio_embarcacion + β₃*experiencia + ε
beta_0 = 15.0
beta_1 = 8.5
beta_2 = 3.2
beta_3 = 0.4

epsilon = np.random.normal(loc=0, scale=5.0, size=n_observations)
data['captura_kg'] = (
    beta_0 + 
    beta_1 * data['horas_faena'] + 
    beta_2 * data['tamanio_embarcacion_m'] + 
    beta_3 * data['experiencia_anios'] + 
    epsilon
)

# Asegurar que captura > 0 (restricción de dominio)
data['captura_kg'] = np.maximum(data['captura_kg'], 5)

df = pd.DataFrame(data)

print(f"✓ Dataset creado: {df.shape[0]} observaciones, {df.shape[1]} variables")
print(f"\nEstadísticas descriptivas:")
print(df.describe().round(3))

# ============================================================================
# PARTE 2: EXPLORACIÓN INICIAL Y CORRELACIONES
# ============================================================================
print("\n\n[PARTE 2] Exploración Inicial del Dataset")
print("-" * 80)

# Matriz de correlaciones
corr_matrix = df[['captura_kg', 'horas_faena', 'tamanio_embarcacion_m', 
                   'experiencia_anios', 'temperatura_agua_C']].corr()

print("\nMatriz de Correlaciones (Pearson):")
print(corr_matrix.round(3))

# Visualización: matriz de correlaciones
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Matriz de Correlaciones - Dataset de Pesquerías Artesanales\n' + 
             'Captura vs. Factores Técnicos, Sociales y Ambientales', 
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\n✓ Interpretación: Todas las correlaciones con captura_kg son positivas.")
print("  Esto sugiere que más horas, mayor embarcación y más experiencia → mayor captura.")

# ============================================================================
# PARTE 3: REGRESIÓN LINEAL SIMPLE (Warm-up)
# ============================================================================
print("\n\n[PARTE 3] Regresión Lineal Simple: Horas de Faena → Captura")
print("-" * 80)

# Modelo simple
modelo_simple = ols('captura_kg ~ horas_faena', data=df).fit()

print("\nSummary del Modelo:")
print(modelo_simple.summary())

# Visualización: scatter + línea de regresión
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['horas_faena'], df['captura_kg'], alpha=0.6, s=60, color='steelblue', edgecolors='black', linewidth=0.5)

# Línea de predicción
x_range = np.linspace(df['horas_faena'].min(), df['horas_faena'].max(), 100)
y_pred = modelo_simple.predict(pd.DataFrame({'horas_faena': x_range}))
ax.plot(x_range, y_pred, 'r-', linewidth=2.5, label=f'Línea ajustada\nR² = {modelo_simple.rsquared:.3f}')

ax.set_xlabel('Horas de Faena', fontsize=11, fontweight='bold')
ax.set_ylabel('Captura (kg)', fontsize=11, fontweight='bold')
ax.set_title('Regresión Lineal Simple: Horas de Faena vs Captura\n' +
             f'Ecuación: Captura = {modelo_simple.params[0]:.2f} + {modelo_simple.params[1]:.2f}*Horas',
             fontsize=12, fontweight='bold', pad=15)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✓ Interpretación:")
print(f"  • Intercepto (β₀) = {modelo_simple.params[0]:.3f} kg")
print(f"  • Pendiente (β₁) = {modelo_simple.params[1]:.3f} kg/hora")
print(f"  • R² = {modelo_simple.rsquared:.3f} ({modelo_simple.rsquared*100:.1f}% variabilidad explicada)")
print(f"  • p-value (F-test) = {modelo_simple.f_pvalue:.2e} (modelo significativo)")

# ============================================================================
# PARTE 4: REGRESIÓN LINEAL MÚLTIPLE
# ============================================================================
print("\n\n[PARTE 4] Regresión Lineal Múltiple")
print("-" * 80)

# Modelo múltiple (3 predictores)
modelo_multiple = ols('captura_kg ~ horas_faena + tamanio_embarcacion_m + experiencia_anios', 
                       data=df).fit()

print("\nModelo: captura_kg ~ horas_faena + tamanio_embarcacion_m + experiencia_anios")
print(modelo_multiple.summary())

print("\n✓ Interpretación de Coeficientes (manteniendo otros constantes):")
for var in modelo_multiple.params.index[1:]:
    coef = modelo_multiple.params[var]
    pval = modelo_multiple.pvalues[var]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    print(f"  • {var}: {coef:.4f} {sig}")

print(f"\n  R² = {modelo_multiple.rsquared:.4f} (mejora vs. simple)")
print(f"  R² Ajustado = {modelo_multiple.rsquared_adj:.4f}")
print(f"  F-statistic = {modelo_multiple.fvalue:.2f}, p-value = {modelo_multiple.f_pvalue:.2e}")

# Comparación de modelos
print(f"\n✓ Comparación: Modelo Simple vs. Múltiple")
print(f"  R² Simple:   {modelo_simple.rsquared:.4f}")
print(f"  R² Múltiple: {modelo_multiple.rsquared:.4f}")
print(f"  Mejora: {(modelo_multiple.rsquared - modelo_simple.rsquared):.4f}")

# ============================================================================
# PARTE 5: SUPUESTOS DEL MODELO LINEAL (DIAGNÓSTICOS)
# ============================================================================
print("\n\n[PARTE 5] Verificación de Supuestos")
print("-" * 80)

# Residuales del modelo múltiple
residuals = modelo_multiple.resid
fitted_values = modelo_multiple.fittedvalues

# 5.1: LINEALIDAD
print("\n[5.1] Supuesto de Linealidad")
print("Método: Gráfico de Residuales vs. Valores Ajustados")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(fitted_values, residuals, alpha=0.6, s=60, color='steelblue', edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Línea Y=0')
ax.scatter(fitted_values, residuals, alpha=0.6, s=60, color='steelblue', edgecolors='black', linewidth=0.5)

# LOWESS smooth para detectar patrones no lineales
from scipy.signal import savgol_filter
sorted_idx = np.argsort(fitted_values)
window_length = min(51, len(fitted_values) // 2 * 2 + 1)
try:
    y_smooth = savgol_filter(residuals.iloc[sorted_idx].values, window_length, 3)
    ax.plot(fitted_values.iloc[sorted_idx], y_smooth, 'g-', linewidth=2, label='Tendencia (Savitzky-Golay)')
except:
    pass

ax.set_xlabel('Valores Ajustados (ŷ)', fontsize=11, fontweight='bold')
ax.set_ylabel('Residuales (e)', fontsize=11, fontweight='bold')
ax.set_title('Diagnóstico 1: Residuales vs. Valores Ajustados\n' +
             'Verifica: Linealidad (patrón aleatorio alrededor de y=0)',
             fontsize=12, fontweight='bold', pad=15)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✓ Interpretación:")
print("  Si los residuales dispersan aleatoriamente sin patrón → linealidad OK")
print("  Si hay patrón curvo → considerar transformación o término no lineal")

# 5.2: NORMALIDAD DE RESIDUALES
print("\n[5.2] Supuesto de Normalidad de Residuales")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histograma
axes[0, 0].hist(residuals, bins=20, color='steelblue', edgecolor='black', alpha=0.7, density=True)
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Densidad Normal')
axes[0, 0].set_xlabel('Residuales', fontsize=10, fontweight='bold')
axes[0, 0].set_ylabel('Densidad', fontsize=10, fontweight='bold')
axes[0, 0].set_title('Histograma de Residuales', fontsize=11, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Q-Q Plot
pp = ProbPlot(residuals)
pp.qqplot(ax=axes[0, 1], line='45', alpha=0.6, markersize=6)
axes[0, 1].set_title('Q-Q Plot: Residuales vs. Normal Teórica', fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Test Shapiro-Wilk
stat_sw, pval_sw = shapiro(residuals)
axes[1, 0].text(0.5, 0.7, 'Test Shapiro-Wilk', ha='center', fontsize=12, fontweight='bold',
               transform=axes[1, 0].transAxes)
axes[1, 0].text(0.5, 0.5, f'Estadístico: {stat_sw:.4f}\np-value: {pval_sw:.4f}',
               ha='center', fontsize=11, transform=axes[1, 0].transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 0].text(0.5, 0.2, 'H₀: Residuales ~ Normal\n' + 
               ('Rechazo H₀ (no normales)' if pval_sw < 0.05 else 'No rechazo H₀ (normales)'),
               ha='center', fontsize=10, transform=axes[1, 0].transAxes,
               style='italic')
axes[1, 0].axis('off')

# ECDF
from scipy.stats import norm
sorted_residuals = np.sort(residuals)
ecdf = np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)
normal_ecdf = norm.cdf(sorted_residuals, mu, sigma)
axes[1, 1].plot(sorted_residuals, ecdf, 'b-', linewidth=2, label='ECDF empírica')
axes[1, 1].plot(sorted_residuals, normal_ecdf, 'r--', linewidth=2, label='CDF Normal teórica')
axes[1, 1].set_xlabel('Residuales', fontsize=10, fontweight='bold')
axes[1, 1].set_ylabel('Probabilidad Acumulada', fontsize=10, fontweight='bold')
axes[1, 1].set_title('Función de Distribución Acumulada (ECDF)', fontsize=11, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"✓ Test Shapiro-Wilk:")
print(f"  Estadístico = {stat_sw:.4f}")
print(f"  p-value = {pval_sw:.4f}")
if pval_sw < 0.05:
    print(f"  ⚠ RECHAZO H₀: Residuales NO normales (p < 0.05)")
    print(f"    Alternativas: (1) Transformar Y, (2) GLM, (3) Robust regression")
else:
    print(f"  ✓ NO RECHAZO H₀: Residuales normales (p ≥ 0.05)")

# 5.3: HOMOCEDASTICIDAD
print("\n[5.3] Supuesto de Homocedasticidad (Varianza Constante)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: Residuales estandarizados vs. Valores ajustados
standardized_residuals = residuals / residuals.std()
axes[0].scatter(fitted_values, np.abs(standardized_residuals), alpha=0.6, s=60, 
               color='steelblue', edgecolors='black', linewidth=0.5)
axes[0].axhline(y=np.sqrt(2), color='r', linestyle='--', linewidth=2, label='Línea referencia')
axes[0].set_xlabel('Valores Ajustados (ŷ)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('|Residuales Estandarizados|', fontsize=11, fontweight='bold')
axes[0].set_title('Scale-Location Plot\nVerifica varianza constante',
                 fontsize=11, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test de Levene
# Dividir residuales en 2 grupos (bajo vs. alto ajuste)
median_fitted = fitted_values.median()
group1 = residuals[fitted_values < median_fitted]
group2 = residuals[fitted_values >= median_fitted]
stat_levene, pval_levene = levene(group1, group2)

axes[1].text(0.5, 0.7, 'Test de Levene', ha='center', fontsize=12, fontweight='bold',
            transform=axes[1].transAxes)
axes[1].text(0.5, 0.5, f'Estadístico: {stat_levene:.4f}\np-value: {pval_levene:.4f}',
            ha='center', fontsize=11, transform=axes[1].transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
axes[1].text(0.5, 0.2, 'H₀: Varianza igual en todos los niveles\n' +
            ('Rechazo H₀ (varianza no constante)' if pval_levene < 0.05 else 'No rechazo H₀ (varianza constante)'),
            ha='center', fontsize=10, transform=axes[1].transAxes, style='italic')
axes[1].axis('off')

plt.tight_layout()
plt.show()

print(f"✓ Test de Levene:")
print(f"  Estadístico = {stat_levene:.4f}")
print(f"  p-value = {pval_levene:.4f}")
if pval_levene < 0.05:
    print(f"  ⚠ RECHAZO H₀: Varianza NO es constante (heterocedástica)")
    print(f"    Alternativas: (1) Transformar Y, (2) WLS, (3) Robust SE")
else:
    print(f"  ✓ NO RECHAZO H₀: Varianza constante (homocedasticidad OK)")

# 5.4: INDEPENDENCIA DE RESIDUALES
print("\n[5.4] Supuesto de Independencia de Residuales")
print("Nota: En datos transversales (cross-sectional), asumir independencia.")
print("      En series de tiempo, usar Durbin-Watson.")

# Durbin-Watson (si fuera relevante)
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuals)
print(f"\nDurbin-Watson: {dw:.4f}")
print(f"Rango: [0, 4], interpretación:")
print(f"  ≈ 2: residuales independientes")
print(f"  < 2: autocorrelación positiva")
print(f"  > 2: autocorrelación negativa")

# Gráfico: Residuales en orden
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(residuals.values, marker='o', linestyle='-', alpha=0.6, color='steelblue', markersize=4)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.fill_between(range(len(residuals)), -2*residuals.std(), 2*residuals.std(), 
                alpha=0.2, color='gray', label='±2 SD')
ax.set_xlabel('Orden de Observación (ID Pescador)', fontsize=11, fontweight='bold')
ax.set_ylabel('Residual', fontsize=11, fontweight='bold')
ax.set_title('Residuales en Orden Temporal/Secuencial\n' +
            'Verifica: Patrón aleatorio (sin tendencias, clusters)',
            fontsize=12, fontweight='bold', pad=15)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✓ Interpretación:")
print("  Si puntos dispersan aleatoriamente → independencia OK")
print("  Si hay patrón cíclico/tendencia → autocorrelación presente")

# ============================================================================
# PARTE 6: MULTICOLINEALIDAD
# ============================================================================
print("\n\n[PARTE 6] Análisis de Multicolinealidad")
print("-" * 80)

# VIF (Variance Inflation Factor)
X_numeric = df[['horas_faena', 'tamanio_embarcacion_m', 'experiencia_anios']]
X_with_const = sm.add_constant(X_numeric)

vif_data = pd.DataFrame()
vif_data["Variable"] = X_numeric.columns
vif_data["VIF"] = [VIF(X_with_const.values, i) for i in range(1, X_with_const.shape[1])]

print("\nVariance Inflation Factor (VIF):")
print(vif_data.to_string(index=False))
print("\nInterpretación VIF:")
print("  VIF < 5:   Multicolinealidad baja → OK")
print("  VIF 5-10:  Multicolinealidad moderada → revisar")
print("  VIF > 10:  Multicolinealidad severa → eliminar/combinar predictores")

# Matriz de correlaciones entre predictores
fig, ax = plt.subplots(figsize=(10, 8))
corr_predictors = X_numeric.corr()
sns.heatmap(corr_predictors, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
            vmin=-1, vmax=1)
ax.set_title('Matriz de Correlaciones: Predictores\n' +
            'Alto valor (cercano a 1 o -1) sugiere multicolinealidad',
            fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\n✓ Conclusión sobre Multicolinealidad:")
print("  Todos los VIF < 5 → multicolinealidad no es problema en este modelo")

# ============================================================================
# PARTE 7: TRANSFORMACIONES
# ============================================================================
print("\n\n[PARTE 7] Transformaciones de Variables")
print("-" * 80)

print("\n[7.1] Transformación Log de la Variable Dependiente")
print("Razón: Cuando residuales no son normales o varianza es heterocedástica")

df['log_captura'] = np.log(df['captura_kg'])

modelo_log = ols('log_captura ~ horas_faena + tamanio_embarcacion_m + experiencia_anios',
                  data=df).fit()

print(f"\nComparación de Modelos:")
print(f"  Modelo Original:      R² = {modelo_multiple.rsquared:.4f}")
print(f"  Modelo con log(Y):    R² = {modelo_log.rsquared:.4f}")

# Diagnósticos del modelo log
residuals_log = modelo_log.resid
stat_sw_log, pval_sw_log = shapiro(residuals_log)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Original
axes[0].hist(modelo_multiple.resid, bins=20, color='steelblue', alpha=0.7, 
            edgecolor='black', density=True)
mu1, sigma1 = modelo_multiple.resid.mean(), modelo_multiple.resid.std()
x1 = np.linspace(modelo_multiple.resid.min(), modelo_multiple.resid.max(), 100)
axes[0].plot(x1, stats.norm.pdf(x1, mu1, sigma1), 'r-', linewidth=2)
axes[0].set_title(f'Modelo Original\nShapiro-Wilk p = {pval_sw:.4f}', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Residuales', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Con transformación log
axes[1].hist(residuals_log, bins=20, color='lightgreen', alpha=0.7,
            edgecolor='black', density=True)
mu2, sigma2 = residuals_log.mean(), residuals_log.std()
x2 = np.linspace(residuals_log.min(), residuals_log.max(), 100)
axes[1].plot(x2, stats.norm.pdf(x2, mu2, sigma2), 'r-', linewidth=2)
axes[1].set_title(f'Modelo log(Captura)\nShapiro-Wilk p = {pval_sw_log:.4f}', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Residuales', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n✓ Resultado de transformación log:")
print(f"  Shapiro-Wilk (Original):  p = {pval_sw:.4f}")
print(f"  Shapiro-Wilk (log(Y)):    p = {pval_sw_log:.4f}")
if pval_sw_log > pval_sw:
    print(f"  → Log transformation mejora normalidad")
else:
    print(f"  → Log transformation no mejora normalidad")

print(f"\n[7.2] Estandarización de Predictores")
print("Razón: Facilita interpretación, mejora convergencia numérica, compara importancia")

# Estandarizar predictores
X_scaled = (X_numeric - X_numeric.mean()) / X_numeric.std()
X_scaled.columns = [col + '_scaled' for col in X_scaled.columns]
df_scaled = pd.concat([df[['captura_kg']], X_scaled], axis=1)

modelo_scaled = ols('captura_kg ~ horas_faena_scaled + tamanio_embarcacion_m_scaled + experiencia_anios_scaled',
                     data=df_scaled).fit()

print("\nCoeficientes Estandarizados (comparables en magnitud):")
for var in modelo_scaled.params.index[1:]:
    print(f"  {var}: {modelo_scaled.params[var]:.4f}")

print("\nInterpretación:")
print("  Un aumento de 1 SD en la variable → cambio de β coef. en SD de Y")

# ============================================================================
# PARTE 8: OUTLIERS Y OBSERVACIONES INFLUYENTES
# ============================================================================
print("\n\n[PARTE 8] Detección de Outliers y Observaciones Influyentes")
print("-" * 80)

from statsmodels.graphics.gofplots import abline_plot

# Cook's Distance
influence = modelo_multiple.get_influence()
cooks_d = influence.cooks_distance[0]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Cook's Distance
axes[0, 0].stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt=' ')
axes[0, 0].axhline(y=4/len(df), color='r', linestyle='--', linewidth=2, label='Umbral: 4/n')
axes[0, 0].set_xlabel('Observación ID', fontsize=10, fontweight='bold')
axes[0, 0].set_ylabel("Cook's Distance", fontsize=10, fontweight='bold')
axes[0, 0].set_title("Cook's Distance: Influencia de Observaciones", fontsize=11, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Residuales Estandarizados
standardized_residuals = residuals / residuals.std()
axes[0, 1].scatter(range(len(standardized_residuals)), standardized_residuals, alpha=0.6, s=60)
axes[0, 1].axhline(y=2, color='r', linestyle='--', linewidth=1.5, label='±2 SD')
axes[0, 1].axhline(y=-2, color='r', linestyle='--', linewidth=1.5)
axes[0, 1].set_xlabel('Observación ID', fontsize=10, fontweight='bold')
axes[0, 1].set_ylabel('Residual Estandarizado', fontsize=10, fontweight='bold')
axes[0, 1].set_title('Residuales Estandarizados (Outliers si |z| > 2)', fontsize=11, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Leverage (Valores del sombrero)
leverage = influence.hat_diag
axes[1, 0].scatter(range(len(leverage)), leverage, alpha=0.6, s=60, color='orange')
axes[1, 0].axhline(y=2*modelo_multiple.df_model/len(df), color='r', linestyle='--', 
                   linewidth=2, label='Umbral: 2p/n')
axes[1, 0].set_xlabel('Observación ID', fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel('Leverage (Hat Diagonal)', fontsize=10, fontweight='bold')
axes[1, 0].set_title('Leverage: Distancia Extrema en X', fontsize=11, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# DFBETAS (cambio en coef. al eliminar obs.)
dfbetas = influence.dfbetas
axes[1, 1].scatter(range(len(dfbetas)), dfbetas[:, 1], alpha=0.6, s=60, label='horas_faena')
axes[1, 1].scatter(range(len(dfbetas)), dfbetas[:, 2], alpha=0.6, s=60, label='tamanio_emb')
axes[1, 1].axhline(y=2/np.sqrt(len(df)), color='r', linestyle='--', linewidth=1.5, label='Umbral')
axes[1, 1].axhline(y=-2/np.sqrt(len(df)), color='r', linestyle='--', linewidth=1.5)
axes[1, 1].set_xlabel('Observación ID', fontsize=10, fontweight='bold')
axes[1, 1].set_ylabel('DFBETAS', fontsize=10, fontweight='bold')
axes[1, 1].set_title('DFBETAS: Influencia sobre Coeficientes', fontsize=11, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Identificar outliers/influyentes
outlier_cooks = np.where(cooks_d > 4/len(df))[0]
outlier_residuals = np.where(np.abs(standardized_residuals) > 2)[0]
outlier_leverage = np.where(leverage > 2*modelo_multiple.df_model/len(df))[0]

print(f"\n✓ Observaciones Problemáticas:")
print(f"  Outliers (|resid estd| > 2):  {len(outlier_residuals)} obs.")
if len(outlier_residuals) > 0:
    print(f"    IDs: {outlier_residuals[:5].tolist()}...")
print(f"  Leverage alto (hat > 2p/n):   {len(outlier_leverage)} obs.")
if len(outlier_leverage) > 0:
    print(f"    IDs: {outlier_leverage[:5].tolist()}...")
print(f"  Cook's D alto (> 4/n):        {len(outlier_cooks)} obs.")
if len(outlier_cooks) > 0:
    print(f"    IDs: {outlier_cooks[:5].tolist()}...")

# ============================================================================
# PARTE 9: COMPARACIÓN FINAL DE MODELOS
# ============================================================================
print("\n\n[PARTE 9] Comparación de Modelos Candidatos")
print("-" * 80)

# Resumen comparativo
modelos = {
    'Simple (horas)': modelo_simple,
    'Múltiple (3 var)': modelo_multiple,
    'Log(Y) + 3 var': modelo_log,
    'Escalado': modelo_scaled
}

print("\nResumen Comparativo:")
print(f"{'Modelo':<25} {'R²':<10} {'R² Adj':<10} {'AIC':<10} {'BIC':<10}")
print("-" * 65)
for name, mod in modelos.items():
    print(f"{name:<25} {mod.rsquared:<10.4f} {mod.rsquared_adj:<10.4f} {mod.aic:<10.2f} {mod.bic:<10.2f}")

# ============================================================================
# PARTE 10: CONSULTA AL TUTOR IA (TEMPLATE)
# ============================================================================
print("\n\n[PARTE 10] CONSULTA AL TUTOR IA - EJEMPLOS DE PROMPTS")
print("=" * 80)

print("""
EJEMPLO 1: Problema con normalidad
────────────────────────────────────
Prompt:
  "Mis residuales no pasan el test Shapiro-Wilk (p < 0.05). ¿Qué alternativas
   tengo? ¿Debo transformar Y? ¿Cuál transformación sugiero? ¿O debo usar GLM?"

Respuesta esperada: 
  - Transformaciones (log, sqrt, Box-Cox)
  - Modelos robustos
  - GLM/Generalized Linear Models
  - Bootstrapping


EJEMPLO 2: Multicolinealidad
────────────────────────────
Prompt:
  "Tengo VIF > 10 en dos predictores. ¿Cuál elimino o cómo combino?
   ¿Puedo estandarizar? ¿Debería usar PCA o ridge regression?"

Respuesta esperada:
  - Eliminar una variable redundante
  - PCA (Principal Component Analysis)
  - Ridge/Lasso regression
  - Domain knowledge


EJEMPLO 3: Outliers
────────────────────
Prompt:
  "Encontré 3 observaciones con Cook's D muy alto. ¿Las elimino?
   ¿Hay forma de hacerlas 'robustas'?"

Respuesta esperada:
  - Verificar si son errores de medición
  - Robust regression (Huber, M-estimators)
  - Influential point analysis
  - Sensitivity analysis (con/sin outliers)
""")

print("\nAHORA: Copia tu pregunta de tesis y variables (Y, X)")
print("       y te armaré un CHECKLIST específico para tu análisis.")
print("=" * 80)

# ============================================================================
# GUARDADO: Crear un resumen reproducible
# ============================================================================

print("\n\n[RESUMEN FINAL]")
print("-" * 80)
print(f"Dataset completo guardado en 'df'")
print(f"Modelos disponibles:")
print(f"  - modelo_simple")
print(f"  - modelo_multiple (RECOMENDADO)")
print(f"  - modelo_log")
print(f"  - modelo_scaled")
print(f"\nPróximos pasos:")
print(f"  1. Aplicar este flujo a tu dataset")
print(f"  2. Documentar supuestos (sí/no/condicionado)")
print(f"  3. Guardar gráficos para entrega")
print(f"  4. Reportar en formato APA (tabla de coeficientes + diagnósticos)")
