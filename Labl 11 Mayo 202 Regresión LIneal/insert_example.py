import re

with open("Ejercicio Lab.html", "r", encoding="utf-8") as f:
    content = f.read()

# Update Sidebar
sidebar_old = """        <ul class="nav-links">
            <li><a href="#intro" class="active">Introducción</a></li>
            <li><a href="#teoria-intro">I. T: Modelado Costero</a></li>
            <li><a href="#teoria-modelos">II. T: Ecuaciones OLS</a></li>
            <li><a href="#teoria-supuestos">III. T: Supuestos y Pruebas</a></li>
            <li><a href="#teoria-transformaciones">IV. T: Mitigación Log</a></li>
            <li><a href="#modulo1">1. L: Generación Datos</a></li>"""

sidebar_new = """        <ul class="nav-links">
            <li><a href="#intro" class="active">Introducción</a></li>
            <li><a href="#teoria-intro">I. T: Modelado Costero</a></li>
            <li><a href="#teoria-modelos">II. T: Ecuaciones OLS</a></li>
            <li><a href="#teoria-supuestos">III. T: Supuestos y Pruebas</a></li>
            <li><a href="#teoria-transformaciones">IV. T: Mitigación Log</a></li>
            <li><a href="#teoria-ejemplo">V. T: Ejemplo Aplicado</a></li>
            <li><a href="#modulo1">1. L: Generación Datos</a></li>"""

content = content.replace(sidebar_old, sidebar_new)

# Insert the new section
section_html = """
        <section id="teoria-ejemplo">
            <div class="card" style="font-size: 0.95rem; line-height: 1.7;">
                <h2 class="section-title"><span class="badge">Teoría 5</span> Ejemplo Práctico: Caso Golfo de Nicoya</h2>
                <p>Para consolidar los conceptos antes de pasar al código en Python, analicemos cómo se vería un reporte estadístico real interpretado paso a paso con la metodología.</p>
                
                <h4 style="margin-top: 1rem;">1. Planteamiento</h4>
                <p>La autoridad pesquera necesita saber qué factores afectan la <strong>Captura Diaria (kg)</strong> de la flota artesanal. Se corrió un modelo de Regresión Lineal Múltiple con 150 observaciones. Las variables predictoras son: Horas en el Mar, Eslora del bote (metros) y Temperatura Superficial del Mar (°C).</p>

                <h4 style="margin-top: 1.5rem;">2. Resultados del Modelo (Output Simulado OLS)</h4>
                <div class="code-container" style="background-color: var(--bg-color); padding: 1rem; border-radius: 8px; font-family: 'Fira Code', monospace; overflow-x: auto; margin-bottom: 1rem;">
<pre style="margin: 0; font-size: 0.85rem; color: #a9b7c6;">
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            Captura_kg   R-squared:                       0.712
Model:                           OLS   Adj. R-squared:                  0.706
Method:                Least Squares   F-statistic:                     120.4
Date:               Sat, 09 May 2026   Prob (F-statistic):           2.3e-39
No. Observations:                150   Durbin-Watson:                   1.95
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      45.20       12.50      3.616      0.000      20.49       69.91
Horas_Mar      12.40        2.10      5.904      0.000       8.24       16.56
Eslora_m        8.15        3.40      2.397      0.018       1.43       14.87
Temp_C         -5.30        1.80     -2.944      0.004      -8.86       -1.74
==============================================================================
</pre>
                </div>

                <h4 style="margin-top: 1.5rem;">3. Diagnóstico de Supuestos</h4>
                <ul style="margin-left: 2rem; margin-bottom: 1rem;">
                    <li><strong>Shapiro-Wilk:</strong> p = 0.12 (Mayor a 0.05, por tanto <strong>se asume Normalidad</strong> de residuos).</li>
                    <li><strong>Breusch-Pagan:</strong> p = 0.08 (Mayor a 0.05, por tanto <strong>se asume Homocedasticidad</strong>).</li>
                    <li><strong>Durbin-Watson:</strong> 1.95 (Cerca de 2, <strong>ausencia de autocorrelación</strong>).</li>
                    <li><strong>VIF Máximo:</strong> 2.4 (Menor a 5, <strong>sin multicolinealidad severa</strong>).</li>
                </ul>
                <div class="alert alert-success">
                    <strong>Conclusión de Fiabilidad:</strong> El modelo cumple adecuadamente todos los supuestos matemáticos. No hay problemas que inflen falsamente la significancia estadística.
                </div>

                <h4 style="margin-top: 1.5rem;">4. Interpretación Metodológica (El Argumento)</h4>
                <ol style="margin-left: 2rem; margin-bottom: 1rem;">
                    <li><strong>Ajuste y Signficancia Global:</strong> El modelo es significativo (Prob F < 0.001). Las tres variables incluidas logran explicar en conjunto un <strong>71.2%</strong> de la variabilidad observada en las capturas diarias.</li>
                    <li><strong>Horas en el Mar (Esfuerzo):</strong> Muestra un efecto <em>positivo</em> y muy significativo (p < 0.001). Controlando por las demás variables, por cada hora adicional en el mar la captura aumenta en promedio <strong>12.4 kg</strong>.</li>
                    <li><strong>Eslora (Capacidad Física):</strong> Efecto <em>positivo</em> y significativo (p = 0.018). En promedio, las embarcaciones que son un metro más largas logran <strong>8.15 kg</strong> adicionales de captura diaria.</li>
                    <li><strong>Temperatura (°C):</strong> Muestra un efecto <em>negativo</em> y significativo (p = 0.004). Ceteris paribus, un aumento de 1°C en la temperatura del mar reduce la captura esperada en <strong>5.3 kg</strong>, posiblemente debido al hundimiento o migración de la biomasa pelágica por el estrés térmico.</li>
                </ol>
            </div>
        </section>

        <!-- MÓDULO 1 -->"""

content = content.replace("<!-- MÓDULO 1 -->", section_html)

with open("Ejercicio Lab.html", "w", encoding="utf-8") as f:
    f.write(content)

print("Ejemplo insertado con éxito.")
