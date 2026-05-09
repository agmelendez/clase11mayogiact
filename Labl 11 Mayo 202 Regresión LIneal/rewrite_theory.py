import re

with open("Ejercicio Lab.html", "r", encoding="utf-8") as f:
    content = f.read()

# Replace sidebar navigation
sidebar_old = """        <ul class="nav-links">
            <li><a href="#intro" class="active">Introducción</a></li>
            <li><a href="#teoria">Marco Teórico</a></li>
            <li><a href="#modulo1">1. Generación de Datos</a></li>
            <li><a href="#modulo2">2. Modelo Base (OLS)</a></li>
            <li><a href="#modulo3">3. Diagnóstico (PhD Level)</a></li>
            <li><a href="#modulo4">4. Multicolinealidad (VIF)</a></li>
            <li><a href="#modulo5">5. Modelos Alternativos</a></li>
        </ul>"""

sidebar_new = """        <ul class="nav-links">
            <li><a href="#intro" class="active">Introducción</a></li>
            <li><a href="#teoria-intro">I. T: Modelado Costero</a></li>
            <li><a href="#teoria-modelos">II. T: Ecuaciones OLS</a></li>
            <li><a href="#teoria-supuestos">III. T: Supuestos y Pruebas</a></li>
            <li><a href="#teoria-transformaciones">IV. T: Mitigación Log</a></li>
            <li><a href="#modulo1">1. L: Generación Datos</a></li>
            <li><a href="#modulo2">2. L: Modelo Base (OLS)</a></li>
            <li><a href="#modulo3">3. L: Diagnóstico (PhD)</a></li>
            <li><a href="#modulo4">4. L: Multicolinealidad</a></li>
            <li><a href="#modulo5">5. L: Modelos Alternativos</a></li>
        </ul>"""

content = content.replace(sidebar_old, sidebar_new)

# Replace the entire theory section using regex
pattern = re.compile(r'<!-- MARCO TEÓRICO -->\s*<section id="teoria">.*?</section>\s*<!-- MÓDULO 1 -->', re.DOTALL)

theory_new = """<!-- MARCO TEÓRICO: SECCIONES DIVIDIDAS -->
        <section id="teoria-intro">
            <div class="card" style="font-size: 0.95rem; line-height: 1.7;">
                <h2 class="section-title"><span class="badge">Teoría 1</span> Introducción al Modelado en Gestión Marina Costera</h2>
                <p>En el ámbito de la <strong>Gestión Marina Costera</strong>, muchas veces no basta con realizar análisis descriptivos. Afirmar que <em>"las áreas con alta presión turística tienen menor biomasa de peces"</em> es útil, pero insuficiente para el diseño de políticas públicas o planes de manejo basados en evidencia.</p>
                <p>El análisis profundo requiere utilizar modelos estadísticos que permitan estimar interacciones complejas:</p>
                <ul style="margin-left: 2rem; margin-bottom: 1rem;">
                    <li>¿Existe una relación estadísticamente significativa entre el esfuerzo pesquero y la captura total?</li>
                    <li>¿Esa relación se mantiene constante cuando controlamos la temperatura superficial del mar?</li>
                    <li>¿El modelo de predicción de capturas es confiable a lo largo del tiempo?</li>
                </ul>
                <div class="alert alert-info">
                    <strong>La pregunta central de esta sesión:</strong><br>
                    ¿Cómo podemos estimar, evaluar y proyectar la dinámica de las pesquerías artesanales considerando factores ambientales y socioeconómicos simultáneos?
                </div>
            </div>
        </section>

        <section id="teoria-modelos">
            <div class="card" style="font-size: 0.95rem; line-height: 1.7;">
                <h2 class="section-title"><span class="badge">Teoría 2</span> Modelos de Regresión: Ecuaciones y Ejemplos Costeros</h2>
                
                <h4 style="margin-top: 1rem;">1. Regresión Lineal Simple</h4>
                <p>Analiza el efecto de una única variable independiente (X) sobre una variable dependiente (Y). Su estructura matemática es:</p>
                <div class="code-container" style="background-color: var(--bg-color); padding: 1rem; border-radius: 8px; text-align: center; font-size: 1.2rem; font-family: 'Fira Code', monospace; margin-bottom: 1rem; color: var(--accent-color);">
                    Y = β₀ + β₁X + ε
                </div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr><th>Parámetro</th><th>Aplicación en Biología Pesquera</th></tr>
                        </thead>
                        <tbody>
                            <tr><td><strong>Y</strong></td><td>Captura total desembarcada (en kilogramos).</td></tr>
                            <tr><td><strong>X</strong></td><td>Horas en el mar (medida de esfuerzo pesquero).</td></tr>
                            <tr><td><strong>β₀</strong> (Intercepto)</td><td>Captura teórica si el esfuerzo fuera cero (suele no tener interpretación biológica directa).</td></tr>
                            <tr><td><strong>β₁</strong> (Pendiente)</td><td>Aumento esperado en kg de captura por cada hora adicional en el mar.</td></tr>
                            <tr><td><strong>ε</strong> (Error)</td><td>Factores aleatorios no observados en el modelo (corrientes repentinas, variabilidad climática).</td></tr>
                        </tbody>
                    </table>
                </div>

                <h4 style="margin-top: 1.5rem;">2. Regresión Lineal Múltiple</h4>
                <p>Permite analizar el efecto simultáneo de múltiples factores, <strong>controlando</strong> estadísticamente el efecto de cada uno para evitar conclusiones erróneas.</p>
                <div class="code-container" style="background-color: var(--bg-color); padding: 1rem; border-radius: 8px; text-align: center; font-size: 1.2rem; font-family: 'Fira Code', monospace; margin-bottom: 1rem; color: var(--accent-color);">
                    Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + ε
                </div>
                <blockquote style="border-left: 4px solid var(--accent-color); padding-left: 1rem; margin: 1rem 0; color: var(--text-muted); font-style: italic;">
                    <strong>Ejemplo Oceanográfico:</strong> <br><code>Captura = β₀ + β₁(Horas en el Mar) + β₂(Años de Experiencia) + β₃(Temperatura del Agua) + ε</code>
                </blockquote>
                
                <h4 style="margin-top: 1.5rem;">3. Bondad de Ajuste (R² y Prueba F)</h4>
                <ul style="margin-left: 2rem; margin-bottom: 1rem;">
                    <li><strong>R² (Coeficiente de Determinación):</strong> Mide la proporción de la varianza en la captura que es explicada por nuestro conjunto de variables ambientales y de esfuerzo. Un R² de 0.65 indica que el modelo explica el 65% de las variaciones en las capturas.</li>
                    <li><strong>Prueba F de Fisher:</strong> Evalúa la significancia global del modelo. Contrasta empíricamente si predecir con nuestras variables (esfuerzo, temperatura) es estadísticamente mejor que predecir utilizando únicamente el promedio histórico de capturas (p &lt; 0.05).</li>
                </ul>
            </div>
        </section>

        <section id="teoria-supuestos">
            <div class="card" style="font-size: 0.95rem; line-height: 1.7;">
                <h2 class="section-title"><span class="badge">Teoría 3</span> Diagnóstico: Supuestos y Pruebas Estadísticas Formales</h2>
                <p>En el manejo adaptativo marino, un modelo de predicción solo es válido si los residuos (ε) cumplen rigurosamente ciertas propiedades estocásticas. Evaluar esto requiere tanto gráficos de diagnóstico como pruebas estadísticas formales.</p>

                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Supuesto de OLS</th>
                                <th>Interpretación Aplicada</th>
                                <th>Prueba Estadística Formal</th>
                                <th>Diagnóstico Gráfico</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>1. Normalidad de Residuos</strong></td>
                                <td>Los errores de captura imprevistos deben distribuirse como una campana de Gauss, sin valores atípicos extremos sistemáticos.</td>
                                <td><strong>Prueba de Shapiro-Wilk:</strong><br>H0: Los residuos son normales.<br>(Buscamos p > 0.05 para cumplir supuesto).</td>
                                <td>Gráfico Q-Q (Normal Q-Q Plot).</td>
                            </tr>
                            <tr>
                                <td><strong>2. Homocedasticidad</strong></td>
                                <td>La variabilidad de los errores debe ser constante. El modelo no debe ser mucho más impreciso prediciendo capturas grandes en buques industriales que en lanchas pequeñas.</td>
                                <td><strong>Prueba de Breusch-Pagan:</strong><br>H0: Varianza constante.<br>(Buscamos p > 0.05).</td>
                                <td>Residuals vs Fitted / Scale-Location Plot.</td>
                            </tr>
                            <tr>
                                <td><strong>3. Independencia (Ausencia de Autocorrelación)</strong></td>
                                <td>La captura reportada por una lancha hoy no debe determinar el residuo de la captura reportada mañana (frecuente en series de tiempo climáticas).</td>
                                <td><strong>Prueba de Durbin-Watson:</strong><br>Valores cercanos a 2 indican independencia (Rango aceptable: 1.5 - 2.5).</td>
                                <td>ACF Plot (Función de Autocorrelación).</td>
                            </tr>
                            <tr>
                                <td><strong>4. Ausencia de Multicolinealidad</strong></td>
                                <td>Las variables independientes no deben ser linealmente redundantes (ej. incluir "Eslora de embarcación" y "Tonelaje bruto" al mismo tiempo infla los errores estándar).</td>
                                <td><strong>Factor de Inflación de la Varianza (VIF):</strong><br>Valores VIF > 5 sugieren colinealidad. Valores > 10 indican un problema severo.</td>
                                <td>Matriz de Correlación Térmica (Heatmap).</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <h4 style="margin-top: 1.5rem;">Riesgo de Gestión al Violar Supuestos</h4>
                <div class="alert alert-warning">
                    Si un modelo pesquero sufre de <strong>Heterocedasticidad</strong> o <strong>Multicolinealidad</strong>, los estimadores pueden seguir siendo insesgados, pero los errores estándar serán calculados erróneamente. Esto puede llevarnos a declarar una variable climática como estadísticamente significativa cuando en realidad no lo es, induciendo políticas de vedas o asignación de cuotas injustificadas.
                </div>
            </div>
        </section>

        <section id="teoria-transformaciones">
            <div class="card" style="font-size: 0.95rem; line-height: 1.7;">
                <h2 class="section-title"><span class="badge">Teoría 4</span> Mitigación: Transformaciones Log-Lineales y Elasticidad</h2>
                <p>En oceanografía y biología pesquera, las variables rara vez se comportan linealmente de forma natural. Por ejemplo, la distribución de ingresos o capturas totales suele presentar una fuerte <strong>asimetría positiva</strong> (unas pocas lanchas industriales pescan el 80% de la biomasa, mientras muchas artesanales pescan muy poco).</p>

                <h4 style="margin-top: 1rem;">La Transformación Logarítmica Natural (ln)</h4>
                <p>Cuando la relación es exponencial o los residuos violan severamente la normalidad y homocedasticidad, es práctica estándar aplicar el logaritmo natural a la variable dependiente (ej. Ingresos) o a variables independientes (ej. Distancia al puerto).</p>
                
                <div class="code-container" style="background-color: var(--bg-color); padding: 1rem; border-radius: 8px; text-align: center; font-size: 1.2rem; font-family: 'Fira Code', monospace; margin-bottom: 1rem; color: var(--accent-color);">
                    ln(Captura) = β₀ + β₁·ln(Esfuerzo) + ε
                </div>

                <h4 style="margin-top: 1.5rem;">El Concepto Económico de Elasticidad</h4>
                <p>La ventaja de la transformación <strong>Log-Log</strong> en el manejo de recursos naturales es su interpretación económica directa. Los coeficientes ya no representan unidades absolutas (kg por hora), sino elasticidades porcentuales.</p>
                <blockquote style="border-left: 4px solid var(--accent-color); padding-left: 1rem; margin: 1rem 0; color: var(--text-muted); font-style: italic;">
                    <strong>Interpretación:</strong> Si un modelo pesquero Log-Log arroja un coeficiente β₁ de <strong>0.85</strong> para la variable esfuerzo, significa que: <em>"Ceteris paribus, ante un aumento del 1% en el esfuerzo pesquero, se espera un aumento promedio del 0.85% en las capturas totales".</em>
                </blockquote>
                <p>Esta interpretación marginal es la piedra angular de modelos clásicos de dinámica poblacional, como el modelo biológico de producción excedente de Schaefer.</p>
            </div>
        </section>

        <!-- MÓDULO 1 -->"""

content = pattern.sub(theory_new, content)

with open("Ejercicio Lab.html", "w", encoding="utf-8") as f:
    f.write(content)

print("HTML Reescrito Exitosamente")
