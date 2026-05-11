# -*- coding: utf-8 -*-
"""Utilidad opcional: reemplaza teoría OLS por teoría de regresión logística en un HTML base."""
from pathlib import Path
import re

path = Path("index.html")
content = path.read_text(encoding="utf-8")
content = re.sub(r"Regresión Lineal|Modelado Lineal|OLS|MCO", "Regresión Logística", content, flags=re.I)
content = content.replace("Ecuaciones OLS", "Modelo logit")
content = content.replace("Modelo Base (OLS)", "Modelo Logit")
path.write_text(content, encoding="utf-8")
print("Teoría actualizada a regresión logística en index.html")
