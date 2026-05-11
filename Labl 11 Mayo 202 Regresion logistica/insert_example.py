# -*- coding: utf-8 -*-
"""Utilidad opcional: inserta un ejemplo logístico aplicado en un laboratorio HTML."""
from pathlib import Path

path = Path("index.html")
content = path.read_text(encoding="utf-8")
example = """
<section id="teoria-ejemplo-logit">
  <div class="card">
    <h2 class="section-title"><span class="badge">Ejemplo</span> Regresión logística aplicada</h2>
    <p>Evento Y=1: adopción de una buena práctica de manejo costero. Predictores: capacitación, experiencia, distancia e índice de presión turística.</p>
    <pre><code class="language-python">modelo = smf.logit('adopta_buena_practica ~ capacitacion + experiencia_anios + distancia_km', data=df).fit()</code></pre>
  </div>
</section>
"""
content = content.replace("</main>", example + "\n</main>")
path.write_text(content, encoding="utf-8")
print("Ejemplo logístico insertado.")
