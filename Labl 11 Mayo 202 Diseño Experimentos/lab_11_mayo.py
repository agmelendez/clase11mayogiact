# ============================================================
# Laboratorio autoguiado: Diseño de Experimentos aplicado a
# Gestión Marina Costera
# Compatible con Google Colab / Python local
# ============================================================

import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except Exception:
    sns = None

# ------------------------------------------------------------
# 1. Simulación de un experimento factorial marino-costero
# ------------------------------------------------------------
def simular_experimento_marino(
    semilla=2026,
    replicas=5,
    incluir_bloques=True,
    escenario="base",
    error_sigma=4.5,
):
    """Genera un diseño factorial 3 x 2 x 2 con réplicas y bloques.

    Factores:
    - proteccion: Baja, Media, Alta
    - restauracion: Sin restauracion, Con restauracion
    - presion_turistica: Baja, Alta

    Respuesta:
    - biodiversidad: índice sintético 0-100 de biodiversidad/recuperación ecológica

    Escenarios:
    - base: supuestos razonables
    - interaccion_fuerte: efecto conjunto protección x restauración más alto
    - heterocedastico: varianza distinta por presión turística
    - outliers: incorpora observaciones influyentes
    - desbalanceado: elimina algunas celdas al azar para practicar diseños no balanceados
    """
    rng = np.random.default_rng(semilla)
    niveles = {
        "proteccion": ["Baja", "Media", "Alta"],
        "restauracion": ["Sin", "Con"],
        "presion_turistica": ["Baja", "Alta"],
    }
    bloques = ["Bahia", "Estuario", "Arrecife"] if incluir_bloques else ["General"]

    filas = []
    for bloque, prot, rest, turismo in itertools.product(
        bloques, niveles["proteccion"], niveles["restauracion"], niveles["presion_turistica"]
    ):
        for r in range(1, replicas + 1):
            filas.append({
                "bloque_zona": bloque,
                "proteccion": prot,
                "restauracion": rest,
                "presion_turistica": turismo,
                "replica": r,
            })
    df = pd.DataFrame(filas)
    df = df.sample(frac=1, random_state=semilla).reset_index(drop=True)
    df["orden_aleatorio"] = np.arange(1, len(df) + 1)

    # Efectos sustantivos simulados
    base = 52
    efecto_prot = {"Baja": -5, "Media": 4, "Alta": 11}
    efecto_rest = {"Sin": 0, "Con": 8}
    efecto_tur = {"Baja": 4, "Alta": -7}
    efecto_bloque = {"Bahia": -2, "Estuario": 1.5, "Arrecife": 4, "General": 0}

    eta = (
        base
        + df["proteccion"].map(efecto_prot)
        + df["restauracion"].map(efecto_rest)
        + df["presion_turistica"].map(efecto_tur)
        + df["bloque_zona"].map(efecto_bloque)
    ).astype(float)

    # Interacciones pedagógicas
    interaccion = (
        (df["proteccion"].eq("Alta") & df["restauracion"].eq("Con")) * 5
        + (df["proteccion"].eq("Baja") & df["presion_turistica"].eq("Alta")) * -4
    )
    if escenario == "interaccion_fuerte":
        interaccion = interaccion * 2.2
    eta = eta + interaccion

    if escenario == "heterocedastico":
        sigma = np.where(df["presion_turistica"].eq("Alta"), error_sigma * 1.9, error_sigma * 0.75)
    else:
        sigma = np.repeat(error_sigma, len(df))

    df["biodiversidad"] = eta + rng.normal(0, sigma, len(df))
    df["biodiversidad"] = df["biodiversidad"].clip(0, 100)

    if escenario == "outliers":
        idx = rng.choice(df.index, size=max(2, len(df)//20), replace=False)
        df.loc[idx, "biodiversidad"] = rng.choice([8, 96], size=len(idx))
        df.loc[idx, "observacion_atipica_simulada"] = 1
        df["observacion_atipica_simulada"] = df["observacion_atipica_simulada"].fillna(0).astype(int)

    if escenario == "desbalanceado":
        quitar = df.query("proteccion == 'Baja' and restauracion == 'Con' and presion_turistica == 'Alta'").sample(
            frac=0.45, random_state=semilla
        ).index
        df = df.drop(quitar).reset_index(drop=True)

    return df

# ------------------------------------------------------------
# 2. Ajuste del modelo experimental
# ------------------------------------------------------------
def ajustar_modelo_experimental(df, incluir_bloques=True):
    formula = "biodiversidad ~ C(proteccion) * C(restauracion) * C(presion_turistica)"
    if incluir_bloques and df["bloque_zona"].nunique() > 1:
        formula += " + C(bloque_zona)"
    modelo = smf.ols(formula, data=df).fit()
    return modelo, formula

# ------------------------------------------------------------
# 3. Diagnóstico estadístico del modelo
# ------------------------------------------------------------
def diagnosticar_modelo(modelo):
    resid = modelo.resid
    fitted = modelo.fittedvalues
    shapiro_stat, shapiro_p = stats.shapiro(resid)
    bp = het_breuschpagan(resid, modelo.model.exog)
    influence = modelo.get_influence()
    cooks = influence.cooks_distance[0]
    out = pd.DataFrame({
        "prueba": ["Shapiro-Wilk", "Breusch-Pagan", "Cook > 4/n"],
        "indicador": ["Normalidad residual", "Homocedasticidad", "Influencia"],
        "valor": [shapiro_p, bp[1], float(np.sum(cooks > 4/len(resid)))],
        "criterio": ["p > 0.05 deseable", "p > 0.05 deseable", "menor cantidad posible"],
    })
    return out, resid, fitted, cooks

# ------------------------------------------------------------
# 4. Visualizaciones clave
# ------------------------------------------------------------
def graficos_experimentales(df, modelo):
    resid = modelo.resid
    fitted = modelo.fittedvalues
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if sns:
        sns.pointplot(data=df, x="proteccion", y="biodiversidad", hue="restauracion", errorbar="se", ax=axes[0,0])
        sns.boxplot(data=df, x="presion_turistica", y="biodiversidad", hue="proteccion", ax=axes[0,1])
    else:
        df.groupby(["proteccion", "restauracion"])["biodiversidad"].mean().unstack().plot(ax=axes[0,0], marker="o")
        df.boxplot(column="biodiversidad", by="presion_turistica", ax=axes[0,1])

    axes[0,0].set_title("Interacción: protección x restauración")
    axes[0,1].set_title("Distribución por presión turística y protección")

    axes[1,0].scatter(fitted, resid, alpha=0.65)
    axes[1,0].axhline(0, linestyle="--")
    axes[1,0].set_title("Residuales vs valores ajustados")
    axes[1,0].set_xlabel("Valores ajustados")
    axes[1,0].set_ylabel("Residuales")

    sm.qqplot(resid, line="45", ax=axes[1,1])
    axes[1,1].set_title("Q-Q plot de residuales")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 5. Ejecución completa del laboratorio
# ------------------------------------------------------------
df = simular_experimento_marino(
    semilla=2026,
    replicas=5,
    incluir_bloques=True,
    escenario="base",  # base, interaccion_fuerte, heterocedastico, outliers, desbalanceado
    error_sigma=4.5,
)

print("Primeras filas del diseño aleatorizado:")
print(df.head())
print("\nTamaño por celda experimental:")
print(pd.crosstab([df["proteccion"], df["restauracion"]], df["presion_turistica"]))

modelo, formula = ajustar_modelo_experimental(df, incluir_bloques=True)
print("\nFórmula del modelo:", formula)
print(modelo.summary())

print("\nANOVA tipo II:")
tabla_anova = anova_lm(modelo, typ=2)
print(tabla_anova)

print("\nDiagnóstico del modelo:")
diagnostico, residuales, ajustados, cooks_d = diagnosticar_modelo(modelo)
print(diagnostico)

graficos_experimentales(df, modelo)

print("\nComparaciones múltiples Tukey para factor protección:")
print(pairwise_tukeyhsd(endog=df["biodiversidad"], groups=df["proteccion"], alpha=0.05))

# ------------------------------------------------------------
# 6. Guía de interpretación mínima
# ------------------------------------------------------------
print("""
LECTURA PARA EL REPORTE:
1. Revise primero la tabla de balance: cada combinación de factores debe tener réplicas suficientes.
2. En ANOVA, p < 0.05 sugiere diferencias estadísticamente detectables entre niveles del factor.
3. Una interacción significativa implica que el efecto de un factor depende del nivel de otro factor.
4. Los diagnósticos no son accesorios: si fallan normalidad, homocedasticidad o influencia, documente el problema y considere transformaciones, GLM, errores robustos o modelos mixtos.
5. La conclusión debe traducir el resultado estadístico a una decisión de gestión: protección, restauración, control turístico, priorización territorial o necesidad de más evidencia.
""")
