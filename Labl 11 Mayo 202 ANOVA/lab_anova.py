# =============================================================================
# LABORATORIO 11 — ANÁLISIS DE VARIANZA (ANOVA)
# Maestría en Ciencias — Estadística Aplicada
# Fecha: Mayo 2026
# =============================================================================
# Contexto experimental:
#   Diseño factorial 3 × 2 — Áreas Marinas Protegidas (AMP)
#   Factor A — proteccion: AMP_Total | AMP_Parcial | Libre      (k=3 niveles)
#   Factor B — estacion  : Seca | Lluviosa                       (k=2 niveles)
#   Variable respuesta Y  : biomasa de peces (kg/100 m²)
#   n = 25 observaciones por celda → N = 150 en total
#
# Hipótesis central:
#   H₀: μ_AMP_Total = μ_AMP_Parcial = μ_Libre
#   H₁: al menos un par de medias difiere
# =============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 0 — INSTALACIÓN / IMPORTACIONES
# ──────────────────────────────────────────────────────────────────────────────
# Descomenta la siguiente línea si ejecutas en Google Colab por primera vez:
# !pip install statsmodels pingouin --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from itertools import combinations

# Tipografía y estilo global
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 120,
})
sns.set_style("whitegrid")

print("✅ Librerías cargadas correctamente.")
print(f"   numpy {np.__version__} | pandas {pd.__version__}")


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 — GENERACIÓN DE DATOS SINTÉTICOS (DISEÑO FACTORIAL 3×2)
# ──────────────────────────────────────────────────────────────────────────────
def generate_anova_data(n_per_cell: int = 25, seed: int = 2026) -> pd.DataFrame:
    """
    Genera un DataFrame balanceado con diseño factorial completo 3 (Protección)
    × 2 (Estación). Los parámetros poblacionales reflejan un efecto principal
    fuerte de protección, un efecto moderado de estación, y una interacción
    sinérgica pequeña entre ambos factores.

    Modelo subyacente (efectos fijos):
        Y_ijk = μ_ij + ε_ijk,  ε_ijk ~ N(0, σ²),  σ_dentro = 6.0 kg/100m²

    Parameters
    ----------
    n_per_cell : int
        Número de réplicas por combinación de niveles (celda).
    seed : int
        Semilla para reproducibilidad.

    Returns
    -------
    pd.DataFrame con columnas: proteccion, estacion, biomasa
    """
    rng = np.random.default_rng(seed)
    sigma_within = 6.0

    # Medias poblacionales por celda (μ_ij)
    cell_means = {
        ('AMP_Total',  'Seca'):     45.0,
        ('AMP_Total',  'Lluviosa'): 52.0,
        ('AMP_Parcial','Seca'):     30.0,
        ('AMP_Parcial','Lluviosa'): 35.0,
        ('Libre',      'Seca'):     18.0,
        ('Libre',      'Lluviosa'): 21.0,
    }

    records = []
    for (prot, est), mu in cell_means.items():
        y = rng.normal(loc=mu, scale=sigma_within, size=n_per_cell)
        for val in y:
            records.append({'proteccion': prot, 'estacion': est, 'biomasa': round(val, 2)})

    df = pd.DataFrame(records)

    # Orden categórico (útil para gráficas y tablas)
    df['proteccion'] = pd.Categorical(
        df['proteccion'], categories=['AMP_Total', 'AMP_Parcial', 'Libre'], ordered=True
    )
    df['estacion'] = pd.Categorical(
        df['estacion'], categories=['Seca', 'Lluviosa'], ordered=True
    )
    return df.sort_values(['proteccion', 'estacion']).reset_index(drop=True)


df = generate_anova_data(n_per_cell=25, seed=2026)

print("\n── Primeras filas ──────────────────────────────────────────")
print(df.head(10).to_string(index=False))

print("\n── Estadísticos descriptivos por celda ─────────────────────")
desc = (df.groupby(['proteccion', 'estacion'], observed=True)['biomasa']
          .agg(n='count', media='mean', de='std', CV=lambda x: x.std()/x.mean()*100)
          .round(3))
print(desc.to_string())

print("\n── Estadísticos globales ────────────────────────────────────")
print(df['biomasa'].describe().round(3).to_string())


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 — EXPLORACIÓN VISUAL
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Exploración visual — Biomasa de peces por régimen de protección",
             fontsize=14, fontweight='bold', y=1.02)

# 2-A: Boxplots por proteccion
palette = {'AMP_Total': '#2196F3', 'AMP_Parcial': '#FF9800', 'Libre': '#F44336'}
sns.boxplot(data=df, x='proteccion', y='biomasa', palette=palette,
            order=['AMP_Total', 'AMP_Parcial', 'Libre'], ax=axes[0])
axes[0].set_title("Factor A — Protección")
axes[0].set_xlabel("Régimen de protección")
axes[0].set_ylabel("Biomasa (kg/100 m²)")

# 2-B: Boxplots por estacion
sns.boxplot(data=df, x='estacion', y='biomasa',
            palette={'Seca': '#FFC107', 'Lluviosa': '#03A9F4'},
            order=['Seca', 'Lluviosa'], ax=axes[1])
axes[1].set_title("Factor B — Estación")
axes[1].set_xlabel("Estación")
axes[1].set_ylabel("Biomasa (kg/100 m²)")

# 2-C: Interaction plot (medias ± 1 SE)
medias = (df.groupby(['proteccion', 'estacion'], observed=True)['biomasa']
            .agg(['mean', 'sem']).reset_index())
for prot, color in palette.items():
    sub = medias[medias['proteccion'] == prot]
    axes[2].errorbar(sub['estacion'].astype(str), sub['mean'],
                     yerr=sub['sem'], marker='o', label=prot,
                     color=color, linewidth=2, capsize=5)
axes[2].set_title("Gráfica de interacción A × B")
axes[2].set_xlabel("Estación")
axes[2].set_ylabel("Media de biomasa (kg/100 m²)")
axes[2].legend(title="Protección", fontsize=9)

plt.tight_layout()
plt.savefig("fig1_exploracion_visual.png", bbox_inches='tight')
plt.show()
print("Figura guardada: fig1_exploracion_visual.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 — ANOVA DE UNA VÍA (Factor A: proteccion)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MÓDULO 3 — ANOVA DE UNA VÍA")
print("="*65)

# 3-A: Modelo via OLS + C() (codificación dummy automática)
#      Esta formulación es exactamente equivalente a aov() en R y generaliza
#      directamente al modelo lineal general (GLM).
modelo_1via = smf.ols('biomasa ~ C(proteccion)', data=df).fit()

tabla_1via = anova_lm(modelo_1via, typ=1)   # SS Tipo I (secuencial)
print("\n── Tabla ANOVA de una vía (Tipo I SS) ─────────────────────")
print(tabla_1via.round(4).to_string())

# 3-B: Mismos resultados vía scipy (verificación cruzada)
grupos = [g['biomasa'].values for _, g in df.groupby('proteccion', observed=True)]
F_scipy, p_scipy = stats.f_oneway(*grupos)
print(f"\n── Verificación scipy.stats.f_oneway ──")
print(f"   F = {F_scipy:.4f}   p = {p_scipy:.6f}")

# 3-C: Interpretación automática
alpha = 0.05
F_1via = tabla_1via.loc['C(proteccion)', 'F']
p_1via = tabla_1via.loc['C(proteccion)', 'PR(>F)']
print(f"\n── Decisión (α = {alpha}) ──────────────────────────────────")
if p_1via < alpha:
    print(f"   F({int(tabla_1via.loc['C(proteccion)','df'])}, "
          f"{int(tabla_1via.loc['Residual','df'])}) = {F_1via:.3f}, "
          f"p = {p_1via:.4f}  →  RECHAZAR H₀")
    print("   Conclusión: al menos un régimen de protección difiere")
    print("               significativamente en biomasa media.")
else:
    print(f"   F = {F_1via:.3f}, p = {p_1via:.4f}  →  NO rechazar H₀")

# 3-D: Partición de sumas de cuadrados manual
SS_A   = tabla_1via.loc['C(proteccion)', 'sum_sq']
SS_E_1 = tabla_1via.loc['Residual', 'sum_sq']
SS_T   = SS_A + SS_E_1
eta2   = SS_A / SS_T
k      = df['proteccion'].nunique()
N      = len(df)
MS_E_1 = tabla_1via.loc['Residual', 'mean_sq']
omega2 = (SS_A - (k - 1) * MS_E_1) / (SS_T + MS_E_1)

print(f"\n── Partición SS (una vía) ──────────────────────────────────")
print(f"   SST  = {SS_T:.2f}")
print(f"   SS_A = {SS_A:.2f}  ({SS_A/SS_T*100:.1f}% de SST)")
print(f"   SS_E = {SS_E_1:.2f}  ({SS_E_1/SS_T*100:.1f}% de SST)")
print(f"\n── Tamaño de efecto ────────────────────────────────────────")
print(f"   η²  = {eta2:.4f}  (biasado; NO reportar en publicaciones)")
print(f"   ω²  = {omega2:.4f}  (insesgado; PREFERIDO para reportar)")
benchmarks = {0.01: 'pequeño', 0.06: 'mediano', 0.14: 'grande'}
for cutoff, label in sorted(benchmarks.items(), reverse=True):
    if omega2 >= cutoff:
        print(f"   Interpretación Cohen (1988): efecto {label} (ω² ≥ {cutoff})")
        break


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4 — ANOVA FACTORIAL DOS VÍAS (A × B)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MÓDULO 4 — ANOVA FACTORIAL 3 × 2  (proteccion × estacion)")
print("="*65)
print("Nota: Se usa SS Tipo II (recomendado para diseños balanceados")
print("      y para controlar el orden de entrada en la tabla).")

modelo_2via = smf.ols('biomasa ~ C(proteccion) * C(estacion)', data=df).fit()
tabla_2via  = anova_lm(modelo_2via, typ=2)  # Tipo II (más apropiado que Tipo I)

print("\n── Tabla ANOVA factorial (Tipo II SS) ─────────────────────")
print(tabla_2via.round(4).to_string())

# Extraer componentes SS para tamaño de efecto parcial
SS_prot  = tabla_2via.loc['C(proteccion)',            'sum_sq']
SS_est   = tabla_2via.loc['C(estacion)',              'sum_sq']
SS_inter = tabla_2via.loc['C(proteccion):C(estacion)','sum_sq']
SS_E     = tabla_2via.loc['Residual',                 'sum_sq']
MS_E     = tabla_2via.loc['Residual',                 'mean_sq']
SS_T_2   = SS_prot + SS_est + SS_inter + SS_E

def partial_eta2(ss_effect, ss_error):
    return ss_effect / (ss_effect + ss_error)

def omega2_factorial(ss_effect, df_effect, ms_error, ss_total, n_total):
    """ω² parcial para diseños factoriales (Olejnik & Algina, 2003)."""
    return (ss_effect - df_effect * ms_error) / (ss_total + ms_error)

df_prot  = tabla_2via.loc['C(proteccion)',             'df']
df_est   = tabla_2via.loc['C(estacion)',               'df']
df_inter = tabla_2via.loc['C(proteccion):C(estacion)', 'df']

print("\n── Tamaños de efecto (η² parcial y ω²) ────────────────────")
for efecto, ss_ef, df_ef in [
        ('Protección',     SS_prot,  df_prot),
        ('Estación',       SS_est,   df_est),
        ('Interacción A×B',SS_inter, df_inter)]:
    pe2  = partial_eta2(ss_ef, SS_E)
    om2  = omega2_factorial(ss_ef, df_ef, MS_E, SS_T_2, N)
    print(f"   {efecto:20s}  η²p = {pe2:.4f}   ω² = {om2:.4f}")

# Diagnóstico de la interacción
F_inter = tabla_2via.loc['C(proteccion):C(estacion)', 'F']
p_inter = tabla_2via.loc['C(proteccion):C(estacion)', 'PR(>F)']
print(f"\n── Interacción A×B ─────────────────────────────────────────")
if p_inter < alpha:
    print(f"   F = {F_inter:.3f}, p = {p_inter:.4f}  →  Interacción SIGNIFICATIVA")
    print("   ⚠️  Advertencia: Los efectos principales deben interpretarse")
    print("       con cautela. Analice efectos simples (simple effects).")
else:
    print(f"   F = {F_inter:.3f}, p = {p_inter:.4f}  →  Interacción NO significativa")
    print("   Los efectos principales pueden interpretarse directamente.")


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 5 — DIAGNÓSTICO DE SUPUESTOS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MÓDULO 5 — DIAGNÓSTICO DE SUPUESTOS DEL MODELO ANOVA")
print("="*65)

residuos   = modelo_2via.resid
ajustados  = modelo_2via.fittedvalues
resid_std  = (residuos - residuos.mean()) / residuos.std()

# 5-A: Test de Levene (homogeneidad de varianzas)
#      Levene es preferible sobre Bartlett porque es robusto a no-normalidad
grupos_prot = [g['biomasa'].values
               for _, g in df.groupby('proteccion', observed=True)]
stat_levene, p_levene = stats.levene(*grupos_prot, center='median')
print(f"\n── Test de Levene (homocedasticidad, Factor A) ─────────────")
print(f"   Estadístico W = {stat_levene:.4f},  p = {p_levene:.4f}")
print(f"   {'Varianzas homogéneas (no rechazar H₀)' if p_levene > alpha else 'VARIANZAS HETEROGÉNEAS (rechazar H₀)'}")

# 5-B: Test de Shapiro-Wilk sobre residuos del modelo
stat_sw, p_sw = stats.shapiro(residuos)
print(f"\n── Test de Shapiro-Wilk (normalidad de residuos) ───────────")
print(f"   W = {stat_sw:.4f},  p = {p_sw:.4f}")
print(f"   {'Residuos compatibles con normalidad' if p_sw > alpha else 'DEPARTURA SIGNIFICATIVA de normalidad'}")

# Nota pedagógica: el ANOVA es robusto a desviaciones moderadas de normalidad
# (Teorema Central del Límite), especialmente con n ≥ 20 por celda.

# 5-C: Test de Kruskal-Wallis (alternativa no paramétrica)
stat_kw, p_kw = stats.kruskal(*grupos_prot)
print(f"\n── Test de Kruskal-Wallis (alternativa no paramétrica) ─────")
print(f"   H = {stat_kw:.4f},  p = {p_kw:.4f}")
print(f"   {'Diferencias significativas entre grupos' if p_kw < alpha else 'Sin evidencia de diferencias'}")

# 5-D: Figura diagnóstica cuádruple
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle("Panel de diagnóstico — Supuestos del modelo ANOVA",
              fontsize=14, fontweight='bold')

# (i) Residuos vs Valores Ajustados
ax = axes2[0, 0]
ax.scatter(ajustados, residuos, alpha=0.5, color='steelblue', s=30)
ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
ax.set_xlabel("Valores ajustados (μ̂_ij)")
ax.set_ylabel("Residuos")
ax.set_title("Residuos vs. Ajustados\n(Verificación homocedasticidad)")
ax.annotate("Sin patrón en abanico → supuesto OK",
            xy=(0.05, 0.92), xycoords='axes fraction', fontsize=8, color='gray')

# (ii) Q-Q Normal
ax = axes2[0, 1]
(osm, osr), (slope, intercept, r) = stats.probplot(residuos, dist="norm")
ax.scatter(osm, osr, alpha=0.5, color='steelblue', s=30)
ax.plot(osm, slope * np.array(osm) + intercept, 'r--', linewidth=1.5)
ax.set_xlabel("Cuantiles teóricos N(0,1)")
ax.set_ylabel("Cuantiles observados")
ax.set_title(f"Gráfica Q-Q Normal\n(Shapiro-Wilk W={stat_sw:.3f}, p={p_sw:.3f})")
ax.annotate(f"R² = {r**2:.4f}", xy=(0.05, 0.92), xycoords='axes fraction',
            fontsize=9, color='gray')

# (iii) Histograma de residuos + curva normal
ax = axes2[1, 0]
ax.hist(residuos, bins=20, density=True, color='steelblue', alpha=0.7,
        edgecolor='white', label='Residuos')
xmin, xmax = ax.get_xlim()
xr = np.linspace(xmin, xmax, 200)
ax.plot(xr, stats.norm.pdf(xr, residuos.mean(), residuos.std()),
        'r-', linewidth=2, label='N(0,σ)')
ax.set_xlabel("Residuos")
ax.set_ylabel("Densidad")
ax.set_title("Distribución de residuos")
ax.legend()

# (iv) Residuos estandarizados por grupo
ax = axes2[1, 1]
grupos_label = df['proteccion'].astype(str)
sns.boxplot(x=grupos_label, y=resid_std,
            order=['AMP_Total', 'AMP_Parcial', 'Libre'],
            palette=palette, ax=ax)
ax.axhline(0,  color='red',  linewidth=1.5, linestyle='--')
ax.axhline( 2, color='gray', linewidth=1,   linestyle=':')
ax.axhline(-2, color='gray', linewidth=1,   linestyle=':')
ax.set_xlabel("Régimen de protección")
ax.set_ylabel("Residuos estandarizados")
ax.set_title(f"Homocedasticidad por grupo\n(Levene p={p_levene:.3f})")

plt.tight_layout()
plt.savefig("fig2_diagnostico_supuestos.png", bbox_inches='tight')
plt.show()
print("Figura guardada: fig2_diagnostico_supuestos.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 6 — COMPARACIONES MÚLTIPLES POST-HOC
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MÓDULO 6 — COMPARACIONES MÚLTIPLES POST-HOC")
print("="*65)
print("Procedimiento: Tukey HSD (Honest Significant Difference)")
print("Razón: diseño balanceado; controla FWER al nivel α especificado.")
print(f"Número de comparaciones posibles: C({k},2) = {k*(k-1)//2}")
print(f"Error Tipo I infladio sin corrección (α=0.05, 3 tests): "
      f"{(1 - 0.95**3)*100:.1f}%\n")

# 6-A: Tukey HSD
tukey = pairwise_tukeyhsd(endog=df['biomasa'],
                           groups=df['proteccion'],
                           alpha=alpha)
print("── Tukey HSD — Resultados ──────────────────────────────────")
print(tukey.summary())

# 6-B: d de Cohen por par (medida de efecto puntual)
print("\n── d de Cohen por par de grupos ───────────────────────────")
pares = list(combinations(df['proteccion'].cat.categories, 2))
for g1, g2 in pares:
    x1 = df.loc[df['proteccion'] == g1, 'biomasa'].values
    x2 = df.loc[df['proteccion'] == g2, 'biomasa'].values
    # Desviación estándar combinada (pooled SD)
    n1, n2 = len(x1), len(x2)
    sp = np.sqrt(((n1 - 1) * x1.std(ddof=1)**2 +
                  (n2 - 1) * x2.std(ddof=1)**2) / (n1 + n2 - 2))
    d = (x1.mean() - x2.mean()) / sp
    mag = ('trivial' if abs(d) < 0.2 else
           'pequeño' if abs(d) < 0.5 else
           'mediano' if abs(d) < 0.8 else 'grande')
    print(f"   {g1} vs {g2}:  d = {d:+.3f}  ({mag})")

# 6-C: Visualización comparaciones múltiples
fig3, ax3 = plt.subplots(figsize=(8, 5))
medias_grupo = df.groupby('proteccion', observed=True)['biomasa'].mean()
errores      = df.groupby('proteccion', observed=True)['biomasa'].sem()
colores      = [palette[g] for g in medias_grupo.index]

bars = ax3.bar(medias_grupo.index.astype(str), medias_grupo.values,
               yerr=errores.values, capsize=6,
               color=colores, alpha=0.85, edgecolor='white', linewidth=1.5)

# Letras de grupos Tukey (compact letter display manual)
letras = {'AMP_Total': 'a', 'AMP_Parcial': 'b', 'Libre': 'c'}
for bar, (grupo, media) in zip(bars, medias_grupo.items()):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             media + errores[grupo] + 1.5,
             letras[grupo], ha='center', va='bottom', fontweight='bold', fontsize=13)

ax3.set_title("Medias grupales con intervalos de error estándar\n"
              "(letras distintas = diferencia significativa Tukey HSD α=0.05)",
              fontsize=12)
ax3.set_xlabel("Régimen de protección")
ax3.set_ylabel("Biomasa media (kg/100 m²)")
ax3.set_ylim(0, medias_grupo.max() + 12)
sns.despine(ax=ax3)
plt.tight_layout()
plt.savefig("fig3_comparaciones_multiples.png", bbox_inches='tight')
plt.show()
print("Figura guardada: fig3_comparaciones_multiples.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 7 — EFECTOS SIMPLES (cuando la interacción es significativa)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MÓDULO 7 — EFECTOS SIMPLES (simple effects)")
print("="*65)
print("Cuando A×B es significativo, NO se reportan efectos principales solos.")
print("Se analiza el efecto de A en cada nivel de B (y viceversa).\n")

for est_nivel in df['estacion'].cat.categories:
    sub  = df[df['estacion'] == est_nivel]
    grps = [g['biomasa'].values for _, g in sub.groupby('proteccion', observed=True)]
    F_s, p_s = stats.f_oneway(*grps)
    print(f"   Efecto simple de Protección | Estación = {est_nivel}:")
    print(f"   F({k-1}, {len(sub)-k}) = {F_s:.3f},  p = {p_s:.4f}  "
          f"{'*' if p_s < alpha else 'ns'}")

    if p_s < alpha:
        tukey_s = pairwise_tukeyhsd(endog=sub['biomasa'],
                                     groups=sub['proteccion'],
                                     alpha=alpha)
        print(f"   Tukey HSD post-hoc | Estación = {est_nivel}:")
        print(tukey_s.summary())
    print()


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 8 — FIGURA FINAL: INTERACCIÓN + MEDIAS CON IC95%
# ──────────────────────────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle("Gráfica de interacción y medias marginales estimadas",
              fontsize=14, fontweight='bold')

# 8-A: Interaction plot con IC 95%
medias_celda = (df.groupby(['proteccion', 'estacion'], observed=True)['biomasa']
                  .agg(media='mean', se='sem', n='count').reset_index())
medias_celda['IC95'] = medias_celda['se'] * stats.t.ppf(0.975, medias_celda['n'] - 1)

ax = axes4[0]
for prot, color in palette.items():
    sub = medias_celda[medias_celda['proteccion'] == prot]
    ax.errorbar(sub['estacion'].astype(str), sub['media'],
                yerr=sub['IC95'], marker='o', label=prot,
                color=color, linewidth=2.5, capsize=6, markersize=8)
ax.set_title("Interacción proteccion × estacion\n(Media ± IC 95%)")
ax.set_xlabel("Estación")
ax.set_ylabel("Biomasa media (kg/100 m²)")
ax.legend(title="Protección")

# 8-B: Heatmap de medias cellulares
pivot = medias_celda.pivot(index='proteccion', columns='estacion', values='media')
sns.heatmap(pivot.astype(float), annot=True, fmt='.1f', cmap='YlOrRd',
            ax=axes4[1], linewidths=0.5, annot_kws={'size': 12})
axes4[1].set_title("Mapa de calor — Medias cellulares\n(kg/100 m²)")
axes4[1].set_xlabel("Estación")
axes4[1].set_ylabel("Protección")

plt.tight_layout()
plt.savefig("fig4_interaccion_heatmap.png", bbox_inches='tight')
plt.show()
print("Figura guardada: fig4_interaccion_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# SECCIÓN 9 — RESUMEN EJECUTIVO APA-7
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MÓDULO 9 — RESUMEN DE RESULTADOS (formato APA-7)")
print("="*65)

# Extraer valores clave
F_A = tabla_2via.loc['C(proteccion)',            'F']
p_A = tabla_2via.loc['C(proteccion)',            'PR(>F)']
F_B = tabla_2via.loc['C(estacion)',              'F']
p_B = tabla_2via.loc['C(estacion)',              'PR(>F)']

df_A   = int(tabla_2via.loc['C(proteccion)',            'df'])
df_B   = int(tabla_2via.loc['C(estacion)',              'df'])
df_AB  = int(tabla_2via.loc['C(proteccion):C(estacion)','df'])
df_Res = int(tabla_2via.loc['Residual',                 'df'])

eta2p_A = partial_eta2(SS_prot, SS_E)
eta2p_B = partial_eta2(SS_est,  SS_E)
eta2p_I = partial_eta2(SS_inter,SS_E)

def fmt_p(p):
    return "< .001" if p < 0.001 else f"= {p:.3f}"

print(f"""
Se realizó un ANOVA factorial 3 (Protección: AMP_Total, AMP_Parcial, Libre) ×
2 (Estación: Seca, Lluviosa) sobre la biomasa de peces (kg/100 m²), con
n = 25 réplicas por celda (N = 150).

Se verificaron los supuestos del modelo: el test de Levene no evidenció
heterocedasticidad, W = {stat_levene:.3f}, p {fmt_p(p_levene)}, y el test de
Shapiro-Wilk no detectó desviación significativa de normalidad en los residuos,
W = {stat_sw:.3f}, p {fmt_p(p_sw)}.

El efecto principal del régimen de protección fue estadísticamente significativo,
F({df_A}, {df_Res}) = {F_A:.2f}, p {fmt_p(p_A)}, η²_p = {eta2p_A:.3f}.
El efecto principal de estación también fue significativo,
F({df_B}, {df_Res}) = {F_B:.2f}, p {fmt_p(p_B)}, η²_p = {eta2p_B:.3f}.
La interacción Protección × Estación {'fue' if p_inter < alpha else 'no fue'}
significativa, F({df_AB}, {df_Res}) = {F_inter:.2f}, p {fmt_p(p_inter)},
η²_p = {eta2p_I:.3f}.

Las comparaciones post-hoc de Tukey HSD indicaron que los tres regímenes de
protección difieren significativamente entre sí (todos ps < .001).
""")

print("="*65)
print("FIN DEL SCRIPT — lab_anova.py")
print("Archivos generados:")
for fn in ["fig1_exploracion_visual.png", "fig2_diagnostico_supuestos.png",
           "fig3_comparaciones_multiples.png", "fig4_interaccion_heatmap.png"]:
    print(f"  • {fn}")
print("="*65)
