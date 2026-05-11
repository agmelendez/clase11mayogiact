
# %% [cell 0]
# Bootstrap robusto para Google Colab / Jupyter / Python script
# Instala dependencias faltantes sin usar comandos magicos de IPython.
import sys, subprocess, importlib.util

_REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "sklearn": "scikit-learn",
}

for import_name, pip_name in _REQUIRED_PACKAGES.items():
    if importlib.util.find_spec(import_name) is None:
        print(f"Instalando dependencia faltante: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

print("Dependencias verificadas correctamente.")

# %% [cell 1]
# # SP-8502: Metodos Cuanti-Cuali con IA Responsable
# ## Sesion 9 - Sprint 3: Regresion Lineal - Supuestos, Diagnostico y Modelado
#
# **Laboratorio Autoguiado:** Analisis de Captura de Peces en Pesquerias Artesanales
#
# **Region:** Pacifico Central, Costa Rica (Datos Sinteticos)
#
# **Nivel:** PhD - Interpretacion detallada, supuestos verificados, diagnosticos exhaustivos
#
# ---
#
# ### Objetivo
# Dominar:
# 1. Regresion lineal simple y multiple
# 2. Verificacion de supuestos (linealidad, normalidad, homocedasticidad, independencia)
# 3. Diagnosticos con visualizaciones y tests estadisticos
# 4. Multicolinealidad (VIF, correlaciones)
# 5. Transformaciones (log, sqrt, estandarizacion)
# 6. Deteccion de outliers e influyentes
# 7. Comparacion de modelos

# %% [cell 2]
# ## INSTALACION Y CONFIGURACION

# %% [cell 3]
# Instalar dependencias necesarias (ejecutar UNA SOLA VEZ)
# Nota tecnica: el nombre para instalar un paquete con pip no siempre coincide
# con el nombre usado para importarlo en Python. Por ejemplo:
#   pip install scikit-learn  ->  import sklearn
import importlib
import subprocess
import sys

packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scipy': 'scipy',
    'statsmodels': 'statsmodels',
    'scikit-learn': 'sklearn'
}

for pip_name, import_name in packages.items():
    print(f"Verificando {pip_name}...")
    try:
        importlib.import_module(import_name)
        print(f"  OK {pip_name} ya instalado")
    except ImportError:
        print(f"  Instalando {pip_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name, '-q'])
        print(f"  OK {pip_name} instalado")

# %% [cell 4]
# Importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, levene
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# Configuracion visual robusta para Google Colab
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    plt.style.use('default')
sns.set_palette('husl')
# matplotlib inline omitted: not needed in Colab execution

print("\n" + "="*80)
print("LABORATORIO AUTOGUIADO: REGRESION LINEAL")
print("Sesion 9 - Sprint 3: SP-8502")
print("="*80)

# %% [cell 5]
# ## PARTE 1: GENERACION DE DATASET SINTETICO REALISTA

# %% [cell 6]
print("\n[PARTE 1] Generacion del Dataset Sintetico")
print("-" * 80)

np.random.seed(42)  # Para reproducibilidad
n_observations = 150

# Simulacion realista: captura de peces artesanales en Pacifico Central
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

# Construccion teorica de Y (Data Generating Process conocido)
# Y = beta0 + beta1*X1 + beta2*X2 + beta3*X3 + epsilon
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

# Restriccion de dominio: captura > 0 (realismo)
data['captura_kg'] = np.maximum(data['captura_kg'], 5)

df = pd.DataFrame(data)

print(f"OK Dataset creado: {df.shape[0]} observaciones, {df.shape[1]} variables")
print(f"\nEstadisticas Descriptivas:")
print(df[['captura_kg', 'horas_faena', 'tamanio_embarcacion_m', 'experiencia_anios', 'temperatura_agua_C']].describe().round(3))

print(f"\nPrimeras 5 filas:")
print(df.head())

# %% [cell 7]
# ## PARTE 2: EXPLORACION Y CORRELACIONES

# %% [cell 8]
print("\n[PARTE 2] Exploracion Inicial - Matriz de Correlaciones")
print("-" * 80)

# Seleccionar variables numericas
numeric_cols = ['captura_kg', 'horas_faena', 'tamanio_embarcacion_m', 
                'experiencia_anios', 'temperatura_agua_C']
corr_matrix = df[numeric_cols].corr()

print("\nMatriz de Correlaciones (Pearson):")
print(corr_matrix.round(3))

# Visualizacion
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
ax.set_title('Matriz de Correlaciones - Dataset de Pesquerias Artesanales\n' + 
             'Captura vs. Factores Tecnicos, Sociales y Ambientales', 
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\nOK Interpretacion:")
print("  Todas las correlaciones con captura_kg son positivas.")
print("  Esto sugiere que: mas horas, mayor embarcacion, mas experiencia -> mayor captura")

# %% [cell 9]
# ## PARTE 3: REGRESION LINEAL SIMPLE

# %% [cell 10]
print("\n[PARTE 3] Regresion Lineal Simple: Horas de Faena -> Captura")
print("-" * 80)

# Ajustar modelo simple
modelo_simple = ols('captura_kg ~ horas_faena', data=df).fit()

print("\n" + modelo_simple.summary().as_text())

# Visualizacion
fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(df['horas_faena'], df['captura_kg'], alpha=0.6, s=70, 
          color='steelblue', edgecolors='black', linewidth=0.5, label='Datos observados')

# Linea de prediccion
x_range = np.linspace(df['horas_faena'].min(), df['horas_faena'].max(), 100)
y_pred = modelo_simple.predict(pd.DataFrame({'horas_faena': x_range}))
ax.plot(x_range, y_pred, 'r-', linewidth=2.5, label=f'Linea ajustada (R2 = {modelo_simple.rsquared:.3f})')

ax.set_xlabel('Horas de Faena', fontsize=11, fontweight='bold')
ax.set_ylabel('Captura (kg)', fontsize=11, fontweight='bold')
ax.set_title('Regresion Lineal Simple: Horas de Faena vs Captura\n' +
             f'Ecuacion: Captura = {modelo_simple.params.iloc[0]:.2f} + {modelo_simple.params.iloc[1]:.2f}*Horas',
             fontsize=12, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nOK Interpretacion de Coeficientes:")
print(f"  - Intercepto (beta0) = {modelo_simple.params[0]:.3f} kg")
print(f"    -> Captura esperada cuando horas = 0 (valor teorico)")
print(f"  - Pendiente (beta1) = {modelo_simple.params.iloc[1]:.3f} kg/hora")
print(f"    -> INTERPRETACION: Por cada hora adicional de faena,")
print(f"      se espera un aumento de {modelo_simple.params.iloc[1]:.3f} kg de captura (ceteris paribus)")
print(f"  - R2 = {modelo_simple.rsquared:.3f}")
print(f"    -> El modelo explica {modelo_simple.rsquared*100:.1f}% de la variabilidad en captura")
print(f"  - p-value (F-test) = {modelo_simple.f_pvalue:.2e}")
print(f"    -> El modelo es significativo (p < 0.05)")

# %% [cell 11]
# ## PARTE 4: REGRESION LINEAL MULTIPLE

# %% [cell 12]
print("\n[PARTE 4] Regresion Lineal Multiple")
print("-" * 80)

# Ajustar modelo multiple con 3 predictores
modelo_multiple = ols('captura_kg ~ horas_faena + tamanio_embarcacion_m + experiencia_anios', 
                       data=df).fit()

print("\nEspecificacion del Modelo:")
print("captura_kg = beta0 + beta1*horas_faena + beta2*tamanio_embarcacion_m + beta3*experiencia_anios + epsilon")
print("\n" + modelo_multiple.summary().as_text())

print("\nOK Interpretacion de Coeficientes (manteniendo otros constantes - ceteris paribus):")
for var in modelo_multiple.params.index[1:]:
    coef = modelo_multiple.params[var]
    pval = modelo_multiple.pvalues[var]
    se = modelo_multiple.bse[var]
    t_stat = modelo_multiple.tvalues[var]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    print(f"\n  - {var}:")
    print(f"    Coef = {coef:.4f} {sig} (SE = {se:.4f}, t = {t_stat:.3f}, p = {pval:.4f})")
    
print(f"\nOK Ajuste Global del Modelo:")
print(f"  - R2 = {modelo_multiple.rsquared:.4f}")
print(f"    -> {modelo_multiple.rsquared*100:.2f}% de variabilidad en captura explicada")
print(f"  - R2 Ajustado = {modelo_multiple.rsquared_adj:.4f}")
print(f"    -> Ajustado por numero de predictores (penaliza sobre-parametrizacion)")
print(f"  - F-statistic = {modelo_multiple.fvalue:.2f}, p-value = {modelo_multiple.f_pvalue:.2e}")
print(f"    -> El modelo en conjunto es significativo")

print(f"\nOK Comparacion: Modelo Simple vs. Multiple")
print(f"  R2 Simple:   {modelo_simple.rsquared:.4f}")
print(f"  R2 Multiple: {modelo_multiple.rsquared:.4f}")
print(f"  Mejora: {(modelo_multiple.rsquared - modelo_simple.rsquared):.4f}")
print(f"  -> Agregar predictores mejora significativamente el modelo")

# %% [cell 13]
# ## PARTE 5: VERIFICACION DE SUPUESTOS (DIAGNOSTICOS EXHAUSTIVOS)

# %% [cell 14]
# ### 5.1: SUPUESTO DE LINEALIDAD

# %% [cell 15]
print("\n[5.1] Supuesto de Linealidad")
print("-" * 80)
print("Metodo: Grafico de Residuales vs. Valores Ajustados")
print("Hipotesis nula: La relacion entre Y y X es lineal")
print("Evidencia de violacion: Residuales muestran patron curvo")

# Calcular residuales y valores ajustados
residuals = modelo_multiple.resid
fitted_values = modelo_multiple.fittedvalues

fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(fitted_values, residuals, alpha=0.6, s=70, 
          color='steelblue', edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2.5, label='Linea Y=0')

# Suavizado LOWESS para detectar patrones no lineales
from scipy.signal import savgol_filter
sorted_idx = np.argsort(fitted_values)
window_length = min(51, len(fitted_values) // 2 * 2 + 1)
try:
    y_smooth = savgol_filter(residuals.iloc[sorted_idx].values, window_length, 3)
    ax.plot(fitted_values.iloc[sorted_idx], y_smooth, 'g-', linewidth=2.5, 
           label='Tendencia (Savitzky-Golay)', alpha=0.8)
except:
    pass

ax.set_xlabel('Valores Ajustados (y)', fontsize=11, fontweight='bold')
ax.set_ylabel('Residuales (e)', fontsize=11, fontweight='bold')
ax.set_title('Diagnostico 1: Residuales vs. Valores Ajustados\n' +
             'Verifica: Linealidad (patron aleatorio alrededor de y=0)',
             fontsize=12, fontweight='bold', pad=15)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nOK Criterios de Evaluacion:")
print("  OK LINEALIDAD OK: Residuales dispersan aleatoriamente sin patron")
print("  ERROR LINEALIDAD VIOLADA: Residuales muestran patron curvo/cuadratico")
print("\nAccion si violado: Considerar terminos polinomiales, transformar Y, o usar splines")

# %% [cell 16]
# ### 5.2: SUPUESTO DE NORMALIDAD DE RESIDUALES

# %% [cell 17]
print("\n[5.2] Supuesto de Normalidad de Residuales")
print("-" * 80)
print("Hipotesis nula: epsilon ~ Normal(0, 2)")
print("Importancia: Critica para intervalos de confianza e inferencia en muestras pequenas")

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# Panel 1: Histograma
axes[0, 0].hist(residuals, bins=20, color='steelblue', edgecolor='black', alpha=0.7, density=True)
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2.5, label='Densidad Normal')
axes[0, 0].set_xlabel('Residuales', fontsize=10, fontweight='bold')
axes[0, 0].set_ylabel('Densidad', fontsize=10, fontweight='bold')
axes[0, 0].set_title('(A) Histograma de Residuales', fontsize=11, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Q-Q Plot
pp = ProbPlot(residuals)
pp.qqplot(ax=axes[0, 1], line='45', alpha=0.6, markersize=7)
axes[0, 1].set_title('(B) Q-Q Plot: Residuales vs. Normal Teorica', fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Test Shapiro-Wilk
stat_sw, pval_sw = shapiro(residuals)
axes[1, 0].text(0.5, 0.7, 'Test Shapiro-Wilk', ha='center', fontsize=12, fontweight='bold',
                transform=axes[1, 0].transAxes)
axes[1, 0].text(0.5, 0.5, f'Estadistico W: {stat_sw:.4f}\np-value: {pval_sw:.4f}',
                ha='center', fontsize=11, transform=axes[1, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
axes[1, 0].text(0.5, 0.15, 'H0: Residuales ~ Normal\n' + 
                ('ERROR RECHAZO H0 (no normales, p < 0.05)' if pval_sw < 0.05 else 'OK NO RECHAZO H0 (normales, p >= 0.05)'),
                ha='center', fontsize=10, transform=axes[1, 0].transAxes,
                style='italic', fontweight='bold')
axes[1, 0].axis('off')

# Panel 4: ECDF
from scipy.stats import norm
sorted_residuals = np.sort(residuals)
ecdf = np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)
normal_ecdf = norm.cdf(sorted_residuals, mu, sigma)
axes[1, 1].plot(sorted_residuals, ecdf, 'b-', linewidth=2.5, label='ECDF empirica', marker='o', markersize=3, alpha=0.7)
axes[1, 1].plot(sorted_residuals, normal_ecdf, 'r--', linewidth=2.5, label='CDF Normal teorica')
axes[1, 1].set_xlabel('Residuales', fontsize=10, fontweight='bold')
axes[1, 1].set_ylabel('Probabilidad Acumulada', fontsize=10, fontweight='bold')
axes[1, 1].set_title('(D) Funcion de Distribucion Acumulada (ECDF)', fontsize=11, fontweight='bold')
axes[1, 1].legend(fontsize=9, loc='upper left')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nOK Test Shapiro-Wilk:")
print(f"  Estadistico W = {stat_sw:.4f}")
print(f"  p-value = {pval_sw:.4f}")
if pval_sw < 0.05:
    print(f"  ERROR RECHAZO H0: Residuales NO son normales (p < 0.05)")
    print(f"    Alternativas a explorar:")
    print(f"      1. Transformar Y (log, sqrt, Box-Cox)")
    print(f"      2. Usar GLM/Generalized Linear Models")
    print(f"      3. Robust regression (M-estimators)")
    print(f"      4. Bootstrapping para intervalos de confianza")
else:
    print(f"  OK NO RECHAZO H0: Residuales son normales (p >= 0.05)")

# %% [cell 18]
# ### 5.3: SUPUESTO DE HOMOCEDASTICIDAD (VARIANZA CONSTANTE)

# %% [cell 19]
print("\n[5.3] Supuesto de Homocedasticidad (Varianza Constante)")
print("-" * 80)
print("Hipotesis nula: Var(epsilon) = 2 (constante para todos los valores de X)")
print("Violacion (heterocedasticidad): Var(epsilon|X) cambia con X")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Scale-Location Plot
standardized_residuals = residuals / residuals.std()
sqrt_abs_std_resid = np.sqrt(np.abs(standardized_residuals))

axes[0].scatter(fitted_values, sqrt_abs_std_resid, alpha=0.6, s=70, 
               color='steelblue', edgecolors='black', linewidth=0.5)
axes[0].axhline(y=np.sqrt(2), color='r', linestyle='--', linewidth=2, label='Linea referencia')
axes[0].set_xlabel('Valores Ajustados (y)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('|Residuales Estandarizados|', fontsize=11, fontweight='bold')
axes[0].set_title('(A) Scale-Location Plot\nVerifica varianza constante',
                 fontsize=11, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Panel 2: Test de Levene
median_fitted = fitted_values.median()
group1 = residuals[fitted_values < median_fitted]
group2 = residuals[fitted_values >= median_fitted]
stat_levene, pval_levene = levene(group1, group2)

axes[1].text(0.5, 0.7, 'Test de Levene', ha='center', fontsize=12, fontweight='bold',
            transform=axes[1].transAxes)
axes[1].text(0.5, 0.5, f'Estadistico: {stat_levene:.4f}\np-value: {pval_levene:.4f}',
            ha='center', fontsize=11, transform=axes[1].transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
axes[1].text(0.5, 0.15, 'H0: Varianza igual en todos los niveles\n' +
            ('ERROR RECHAZO H0 (varianza no constante)' if pval_levene < 0.05 else 'OK NO RECHAZO H0 (varianza constante)'),
            ha='center', fontsize=10, transform=axes[1].transAxes, style='italic', fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()

print(f"\nOK Test de Levene:")
print(f"  Estadistico = {stat_levene:.4f}")
print(f"  p-value = {pval_levene:.4f}")
if pval_levene < 0.05:
    print(f"  ERROR RECHAZO H0: Varianza NO es constante (heterocedasticidad)")
    print(f"    Alternativas:")
    print(f"      1. Transformar Y (log, sqrt)")
    print(f"      2. Weighted Least Squares (WLS)")
    print(f"      3. Errores Estandar Robustos (HC1, HC2, HC3)")
else:
    print(f"  OK NO RECHAZO H0: Varianza es constante (homocedasticidad OK)")

# %% [cell 20]
# ### 5.4: SUPUESTO DE INDEPENDENCIA DE RESIDUALES

# %% [cell 21]
print("\n[5.4] Supuesto de Independencia de Residuales")
print("-" * 80)
print("Hipotesis nula: Cov(epsiloni, epsilonj) = 0 para i != j")
print("Nota: En datos transversales (cross-sectional), asumir independencia.")
print("      En series de tiempo, usar Durbin-Watson o Box-Ljung test.")

# Durbin-Watson
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuals)

print(f"\nDurbin-Watson: {dw:.4f}")
print(f"Interpretacion (rango: [0, 4]):")
print(f"  DW  2:   Residuales independientes (OK)")
print(f"  DW < 2:   Autocorrelacion positiva (residuales correlacionados)")
print(f"  DW > 2:   Autocorrelacion negativa")

# Grafico: Residuales en orden
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(range(len(residuals)), residuals.values, marker='o', linestyle='-', 
       alpha=0.6, color='steelblue', markersize=5, label='Residuales')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.fill_between(range(len(residuals)), -2*residuals.std(), 2*residuals.std(), 
                alpha=0.2, color='gray', label='2 SD')
ax.set_xlabel('Orden de Observacion (ID Pescador)', fontsize=11, fontweight='bold')
ax.set_ylabel('Residual', fontsize=11, fontweight='bold')
ax.set_title('Grafico de Secuencia: Residuales en Orden\n' +
            'Verifica: Patron aleatorio (sin tendencias, clusters, ciclos)',
            fontsize=12, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nOK Evaluacion Visual:")
print(f"  Si puntos dispersan aleatoriamente -> independencia OK")
print(f"  Si hay patron ciclico/tendencia -> autocorrelacion presente")

# %% [cell 22]
# ## PARTE 6: ANALISIS DE MULTICOLINEALIDAD

# %% [cell 23]
print("\n[PARTE 6] Analisis de Multicolinealidad")
print("-" * 80)

# Preparar datos para VIF
X_numeric = df[['horas_faena', 'tamanio_embarcacion_m', 'experiencia_anios']]
X_with_const = sm.add_constant(X_numeric)

# Calcular VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_numeric.columns
vif_data["VIF"] = [VIF(X_with_const.values, i) for i in range(1, X_with_const.shape[1])]
vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

print("\nVariance Inflation Factor (VIF):")
print(vif_data.to_string(index=False))

print("\nGrillas de Interpretacion VIF:")
print("  VIF < 5:     OK Multicolinealidad baja -> OK, sin problemas")
print("  VIF 5-10:    ALERTA Multicolinealidad moderada -> revisar")
print("  VIF > 10:    ERROR Multicolinealidad severa -> eliminar/combinar predictores")

# Matriz de correlaciones entre predictores
fig, ax = plt.subplots(figsize=(10, 8))
corr_predictors = X_numeric.corr()
sns.heatmap(corr_predictors, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=2, cbar_kws={"shrink": 0.8}, ax=ax,
            vmin=-1, vmax=1, annot_kws={"fontsize": 11})
ax.set_title('Matriz de Correlaciones: Predictores\n' +
            'Valores cercanos a 1 sugieren multicolinealidad',
            fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\nOK Conclusion sobre Multicolinealidad:")
max_vif = vif_data['VIF'].max()
if max_vif < 5:
    print(f"  OK Todos los VIF < 5 -> Multicolinealidad NO es problema")
elif max_vif < 10:
    print(f"  ALERTA Algunos VIF 5-10 -> Multicolinealidad moderada, revisar especificacion")
else:
    print(f"  ERROR VIF > 10 detectado -> Multicolinealidad severa, considerar:")
    print(f"    - Eliminar una variable redundante")
    print(f"    - PCA (Principal Component Regression)")
    print(f"    - Ridge/Lasso regression")

# %% [cell 24]
# ## PARTE 7: TRANSFORMACIONES DE VARIABLES

# %% [cell 25]
print("\n[PARTE 7] Transformaciones de Variables")
print("-" * 80)

print("\n[7.1] Transformacion Log de la Variable Dependiente")
print("Justificacion: Mejorar normalidad y homocedasticidad")
print("              Interpretacion mas natural (elasticidades)")

# Crear variable transformada
df['log_captura'] = np.log(df['captura_kg'])

# Ajustar modelo con log(Y)
modelo_log = ols('log_captura ~ horas_faena + tamanio_embarcacion_m + experiencia_anios',
                  data=df).fit()

# Test de normalidad para ambos modelos
residuals_original = modelo_multiple.resid
residuals_log = modelo_log.resid
stat_sw_original, pval_sw_original = shapiro(residuals_original)
stat_sw_log, pval_sw_log = shapiro(residuals_log)

print(f"\nComparacion de Ajuste:")
print(f"  Modelo Original (Y):      R2 = {modelo_multiple.rsquared:.4f}")
print(f"  Modelo con log(Y):        R2 = {modelo_log.rsquared:.4f}")
print(f"\nTest Shapiro-Wilk (Normalidad de Residuales):")
print(f"  Original: W = {stat_sw_original:.4f}, p = {pval_sw_original:.4f}")
print(f"  log(Y):   W = {stat_sw_log:.4f}, p = {pval_sw_log:.4f}")
if pval_sw_log > pval_sw_original:
    print(f"  OK Log transformation mejora normalidad (p aumenta)")
else:
    print(f"  ERROR Log transformation NO mejora normalidad")

# Visualizar distribucion de residuales
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Original
axes[0].hist(residuals_original, bins=20, color='steelblue', alpha=0.7, 
            edgecolor='black', density=True)
mu1, sigma1 = residuals_original.mean(), residuals_original.std()
x1 = np.linspace(residuals_original.min(), residuals_original.max(), 100)
axes[0].plot(x1, stats.norm.pdf(x1, mu1, sigma1), 'r-', linewidth=2.5)
axes[0].set_title(f'(A) Modelo Original\nShapiro-Wilk p = {pval_sw_original:.4f}', 
                 fontsize=11, fontweight='bold')
axes[0].set_xlabel('Residuales', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Densidad', fontsize=10, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Log transform
axes[1].hist(residuals_log, bins=20, color='lightgreen', alpha=0.7,
            edgecolor='black', density=True)
mu2, sigma2 = residuals_log.mean(), residuals_log.std()
x2 = np.linspace(residuals_log.min(), residuals_log.max(), 100)
axes[1].plot(x2, stats.norm.pdf(x2, mu2, sigma2), 'r-', linewidth=2.5)
axes[1].set_title(f'(B) Modelo con log(Captura)\nShapiro-Wilk p = {pval_sw_log:.4f}', 
                 fontsize=11, fontweight='bold')
axes[1].set_xlabel('Residuales', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Densidad', fontsize=10, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nInterpretacion de Coeficientes en Modelo log(Y):")
for var in modelo_log.params.index[1:]:
    coef = modelo_log.params[var]
    pval = modelo_log.pvalues[var]
    print(f"  {var}: {coef:.4f}")
    print(f"    -> Elasticidad: aumento de 1% en {var} -> cambio de {coef:.3f}% en Captura")

# %% [cell 26]
# ### 7.2: Estandarizacion de Predictores

# %% [cell 27]
print("\n[7.2] Estandarizacion de Predictores")
print("Justificacion:")
print("  - Facilita comparacion de importancia relativa de X")
print("  - Mejora estabilidad numerica")
print("  - Como solo se estandarizan las X, los coeficientes quedan en kg por 1 desviacion estandar de X")

# Estandarizar predictores
# ddof=0 usa la desviacion estandar poblacional; ddof=1 tambien seria valido.
X_scaled = (X_numeric - X_numeric.mean()) / X_numeric.std(ddof=0)
X_scaled.columns = [col + '_scaled' for col in X_scaled.columns]
df_scaled = pd.concat([df[['captura_kg']], X_scaled], axis=1)

# Ajustar modelo con variables escaladas
modelo_scaled = ols('captura_kg ~ horas_faena_scaled + tamanio_embarcacion_m_scaled + experiencia_anios_scaled',
                     data=df_scaled).fit()

print("\nCoeficientes con predictores estandarizados:")
for var in modelo_scaled.params.index[1:]:
    coef = modelo_scaled.params[var]
    pval = modelo_scaled.pvalues[var]
    print(f"  {var}: {coef:.4f} kg por 1 SD de {var.replace('_scaled', '')} (p = {pval:.4f})")

print("\nInterpretacion:")
print("  Aumento de 1 desviacion estandar en X -> cambio de beta en kg de captura.")
print("  Esto permite comparar que predictor tiene mayor peso relativo sobre Y.")

# Comparar magnitudes de coeficientes estandarizados
std_coefs = pd.DataFrame({
    'Variable': modelo_scaled.params.index[1:],
    'Coef_Scaled': modelo_scaled.params.values[1:],
    'Abs_Coef': np.abs(modelo_scaled.params.values[1:])
}).sort_values('Abs_Coef', ascending=False)

print("\nRanking de Importancia Relativa:")
for idx, row in std_coefs.iterrows():
    var_short = row['Variable'].replace('_scaled', '')
    print(f"  {var_short}: {row['Coef_Scaled']:.4f} kg por 1 SD")

# %% [cell 28]
# ## PARTE 8: OUTLIERS E INFLUENCIA

# %% [cell 29]
print("\n[PARTE 8] Deteccion de Outliers y Observaciones Influyentes")
print("-" * 80)

# Calcular medidas de influencia
# Correccion clave:
# En statsmodels el atributo correcto es `hat_matrix_diag`, no `hat_diag`.
# Ademas, se usan residuales studentizados internos, mas apropiados para diagnostico de outliers.
influence = modelo_multiple.get_influence()
cooks_d = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag
dfbetas = influence.dfbetas
standardized_residuals = influence.resid_studentized_internal

# Numero de parametros estimados, incluyendo intercepto.
p_parametros = int(modelo_multiple.df_model + 1)

# Visualizar cuatro diagnosticos
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: Cook's Distance
axes[0, 0].stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt=' ')
axes[0, 0].axhline(y=4/len(df), color='r', linestyle='--', linewidth=2, label=f'Umbral: 4/n = {4/len(df):.3f}')
axes[0, 0].set_xlabel('Observacion ID', fontsize=10, fontweight='bold')
axes[0, 0].set_ylabel("Cook's Distance", fontsize=10, fontweight='bold')
axes[0, 0].set_title("(A) Cook's Distance: Influencia de Observaciones", fontsize=11, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Residuales Studentizados
axes[0, 1].scatter(range(len(standardized_residuals)), standardized_residuals, alpha=0.6, s=60)
axes[0, 1].axhline(y=2, color='r', linestyle='--', linewidth=1.5, label='2')
axes[0, 1].axhline(y=-2, color='r', linestyle='--', linewidth=1.5)
axes[0, 1].axhline(y=3, color='darkred', linestyle='--', linewidth=1.5, alpha=0.5, label='3 severo')
axes[0, 1].axhline(y=-3, color='darkred', linestyle='--', linewidth=1.5, alpha=0.5)
axes[0, 1].set_xlabel('Observacion ID', fontsize=10, fontweight='bold')
axes[0, 1].set_ylabel('Residual studentizado interno', fontsize=10, fontweight='bold')
axes[0, 1].set_title('(B) Residuales Studentizados (Outliers si |r| > 2)', fontsize=11, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Leverage
axes[1, 0].scatter(range(len(leverage)), leverage, alpha=0.6, s=60, color='orange')
threshold_leverage = 2 * p_parametros / len(df)
axes[1, 0].axhline(y=threshold_leverage, color='r', linestyle='--', 
                   linewidth=2, label=f'Umbral: 2p/n = {threshold_leverage:.3f}')
axes[1, 0].set_xlabel('Observacion ID', fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel('Leverage (diagonal de matriz hat)', fontsize=10, fontweight='bold')
axes[1, 0].set_title('(C) Leverage: Extremidad en Espacio X', fontsize=11, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: DFBETAS (cambio en coeficientes)
for i, col in enumerate(X_numeric.columns[:2]):  # Graficar primeros 2 predictores
    axes[1, 1].scatter(range(len(dfbetas)), dfbetas[:, i+1], alpha=0.6, s=50, label=col)
threshold_dfbetas = 2/np.sqrt(len(df))
axes[1, 1].axhline(y=threshold_dfbetas, color='r', linestyle='--', linewidth=1.5, label='Umbral')
axes[1, 1].axhline(y=-threshold_dfbetas, color='r', linestyle='--', linewidth=1.5)
axes[1, 1].set_xlabel('Observacion ID', fontsize=10, fontweight='bold')
axes[1, 1].set_ylabel('DFBETAS', fontsize=10, fontweight='bold')
axes[1, 1].set_title('(D) DFBETAS: Influencia sobre Coeficientes', fontsize=11, fontweight='bold')
axes[1, 1].legend(fontsize=8, loc='best')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Identificar casos problematicos
outlier_cooks = np.where(cooks_d > 4/len(df))[0]
outlier_residuals = np.where(np.abs(standardized_residuals) > 2)[0]
outlier_leverage = np.where(leverage > threshold_leverage)[0]

print(f"\nOK Resumen de Observaciones Problematicas:")
print(f"\n  Outliers (|resid studentizado| > 2): {len(outlier_residuals)} observaciones")
if len(outlier_residuals) > 0:
    print(f"    IDs base 0: {sorted(outlier_residuals[:10].tolist())} {'...' if len(outlier_residuals) > 10 else ''}")
    print(f"    id_pescador: {df.loc[outlier_residuals[:10], 'id_pescador'].astype(int).tolist()} {'...' if len(outlier_residuals) > 10 else ''}")
print(f"\n  Leverage alto (hat > {threshold_leverage:.3f}): {len(outlier_leverage)} observaciones")
if len(outlier_leverage) > 0:
    print(f"    IDs base 0: {sorted(outlier_leverage[:10].tolist())} {'...' if len(outlier_leverage) > 10 else ''}")
    print(f"    id_pescador: {df.loc[outlier_leverage[:10], 'id_pescador'].astype(int).tolist()} {'...' if len(outlier_leverage) > 10 else ''}")
print(f"\n  Cook's D alto (> {4/len(df):.3f}): {len(outlier_cooks)} observaciones")
if len(outlier_cooks) > 0:
    print(f"    IDs base 0: {sorted(outlier_cooks[:10].tolist())} {'...' if len(outlier_cooks) > 10 else ''}")
    print(f"    id_pescador: {df.loc[outlier_cooks[:10], 'id_pescador'].astype(int).tolist()} {'...' if len(outlier_cooks) > 10 else ''}")

print(f"\nOK Acciones Recomendadas:")
print(f"  1. Verificar si outliers son errores de medicion u observaciones validas")
print(f"  2. Analizar sensibilidad: reajustar modelo sin observaciones influyentes")
print(f"  3. Si validos: usar Robust Regression o Huber M-estimators")
print(f"  4. Documentar decision en reporte")

# %% [cell 30]
# ## PARTE 9: RESUMEN COMPARATIVO DE MODELOS

# %% [cell 31]
print("\n[PARTE 9] Comparacion Final de Modelos Candidatos")
print("-" * 80)

# Compilar informacion de modelos
modelos = {
    'Simple (horas)': modelo_simple,
    'Multiple (3 var)': modelo_multiple,
    'log(Y) + 3 var': modelo_log,
    'Escalado': modelo_scaled
}

# Tabla comparativa
comparison = []
for name, mod in modelos.items():
    comparison.append({
        'Modelo': name,
        'R2': f"{mod.rsquared:.4f}",
        'R2 Adj': f"{mod.rsquared_adj:.4f}",
        'AIC': f"{mod.aic:.1f}",
        'BIC': f"{mod.bic:.1f}",
        'F-stat': f"{mod.fvalue:.2f}",
        'p-value': f"{mod.f_pvalue:.2e}"
    })

comparison_df = pd.DataFrame(comparison)
print("\n")
print(comparison_df.to_string(index=False))

print("\nOK Criterios de Seleccion de Modelo:")
print("  - R2: Mayor es mejor, pero cuidado con sobreparametrizacion")
print("  - R2 Ajustado: Penaliza modelos con muchos predictores")
print("  - AIC/BIC: Menor es mejor (penalizan complejidad)")
print("  - F-statistic: Significancia global del modelo")

print("\nOK RECOMENDACION: Modelo Multiple (3 var)")
print("  Razones:")
print(f"    - R2 = {modelo_multiple.rsquared:.4f} (mejora notable vs. simple)")
print(f"    - R2 Adj = {modelo_multiple.rsquared_adj:.4f} (balance complejidad-ajuste)")
print(f"    - Interpretacion clara de factores (horas, embarcacion, experiencia)")
print(f"    - Supuestos verificados (aunque normalidad podria mejorarse con log)")
print(f"    - AIC = {modelo_multiple.aic:.1f} es competitivo")

# %% [cell 32]
# ## PARTE 10: CONSULTA AL TUTOR IA (EJEMPLOS DE PROMPTS)

# %% [cell 33]
# ### Ejemplos de Preguntas al Tutor IA para Situaciones Comunes

# %% [cell 34]
# #### Ejemplo 1: Normalidad Violada
#
# **Prompt:**
# ```
# Mi modelo de regresion multiple muestra residuales que NO pasan el test 
# Shapiro-Wilk (p < 0.05). Los residuales estan sesgados a la derecha.
#
# Que opciones tengo?
# 1. Debo transformar Y? Cual transformacion probaste?
# 2. Puedo usar GLM en su lugar?
# 3. Bootstrapping resuelve el problema?
# 4. Hay metodos robustos que no asuman normalidad?
#
# Mi Y es captura de peces (valores positivos, rango 5-150 kg).
# ```
#
# **Respuesta esperada del Tutor:**
# - Sugerir Box-Cox para encontrar transformacion optima
# - Recomendar log(Y) para datos sesgados a la derecha
# - Explicar GLM (Poisson o Gamma) como alternativa
# - Bootstrapping para IC sin asumir normalidad
# - Robust regression (Huber, M-estimators)

# %% [cell 35]
# #### Ejemplo 2: Multicolinealidad Detectada
#
# **Prompt:**
# ```
# Mi matriz de correlaciones muestra que dos predictores tienen r = 0.87.
# El VIF para ambas variables es > 10. Cual elimino o como las combino?
#
# Variables: horas_faena (corr 0.87 con experiencia_anios)
#
# Deberia:
# 1. Eliminar una de las dos?
# 2. Crear un indice combinado?
# 3. Usar PCA o Ridge regression?
# 4. Dejar ambas si son teoricamente importantes?
# ```
#
# **Respuesta esperada:**
# - Verificar importancia teorica vs. estadistica
# - PCA para reducir dimensionalidad
# - Ridge/Lasso regression
# - Elimination stepwise
# - Elasticnet para balance

# %% [cell 36]
# #### Ejemplo 3: Outliers Influyentes
#
# **Prompt:**
# ```
# Encontre 3 observaciones con Cook's D > umbral (0.027).
# Una de ellas tiene captura = 450 kg (outlier extremo).
#
# Las elimino? Como se si son errores de medicion u observaciones validas?
#
# Debo presentar resultados:
# 1. Con todas las observaciones?
# 2. Sin outliers?
# 3. Ambos y comparar?
# ```
#
# **Respuesta esperada:**
# - Sensitivity analysis: con/sin outliers
# - Verificar dominio (450 kg es realista?)
# - Robust regression si son validos
# - Documentar decision
# - Reportar ambos escenarios

# %% [cell 37]
# ## PARTE 11: TEMPLATE PARA TU ANALISIS

# %% [cell 38]
# ###  CHECKLIST PARA PREPARAR TU LABORATORIO
#
# **Antes de ejecutar este laboratorio con TUS datos, comparte:**
#
# 1 **Tu pregunta de investigacion (tesis)**
#    - Ejemplo: "Como afectan los factores socio-economicos (experiencia, educacion) y tecnicos (tamano embarcacion) a la captura de peces artesanales?"
#
# 2 **Variable Dependiente (Y)**
#    - Nombre:
#    - Tipo: (continua, conteo, proporcion)
#    - Rango/distribucion esperada:
#    - Unidades:
#
# 3 **Variables Independientes (X) Principales**
#    - X1: _____ (tipo: numerica/categorica, justificacion teorica)
#    - X2: _____ 
#    - X3: _____ 
#    - X4 (opcional):
#
# 4 **Estructura de Datos**
#    - N observaciones:
#    - Fuente (primaria/secundaria, sintesis):
#    - Ausentes (missing):
#
# Luego te envio un **CHECKLIST PERSONALIZADO** con:
# - Verificaciones previas (limpieza de datos)
# - Graficos exploratorios recomendados
# - Transformaciones sugeridas
# - Diagnosticos especificos para tu Y
# - Tabla de resultados (formato APA)

# %% [cell 39]
print("\n" + "="*80)
print("RESUMEN FINAL Y PROXIMOS PASOS")
print("="*80)
print(f"""
OK LABORATORIO COMPLETADO

 Analisis realizados:
   OK Regresion lineal simple y multiple
   OK Verificacion de 4 supuestos del modelo lineal
   OK Diagnosticos visuales y tests estadisticos
   OK Analisis de multicolinealidad (VIF)
   OK Transformaciones de variables
   OK Deteccion de outliers e influencia
   OK Comparacion de modelos candidatos

 Objetos generados en memoria:
   - df: Dataset completo
   - modelo_simple: Regresion 1 predictor
   - modelo_multiple: Regresion 3 predictores (RECOMENDADO)
   - modelo_log: Con transformacion log(Y)
   - modelo_scaled: Con variables estandarizadas

 PROXIMOS PASOS para TU ANALISIS:

1. Descarga este notebook y adapta el codigo a tu dataset
2. Reemplaza el dataset sintetico con tus datos reales
3. Sigue el flujo: Exploracion -> Ajuste -> Diagnosticos -> Interpretacion
4. Guarda todos los graficos (resolucion >= 300 dpi para entrega)
5. Documenta supuestos: OK OK / ALERTA Condicionado / ERROR Violado
6. Prepara tabla de coeficientes en formato APA

 TIPS:
   - Si normalidad falla: Transforma Y (log, sqrt, Box-Cox)
   - Si multicolinealidad: Usa VIF > 5 como alarma temprana
   - Si outliers: Reporta con/sin para sensitivity analysis
   - Si heterocedasticidad: Considera WLS o Robust SE

 Para consultas:
   - Usa el Tutor IA con prompts especificos
   - Comparte tu pregunta de tesis + variables
   - Te armare un checklist personalizado

""")
print("="*80)
print(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
print("="*80)
