"""
Genera el dataset PROPIO del proyecto: clientes de la cafetería universitaria "Café La Corona".

IMPORTANTE: los datos son SIMULADOS (sintéticos). Se diseñaron con intención para tener
problemas reales de calidad (valores faltantes y outliers) y así poder aplicar U3.
Se declara como simulado en el reporte por honestidad académica.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2505)  # semilla fija -> resultados reproducibles
N = 320

# Cuatro perfiles de cliente "ocultos" (el K-means deberá redescubrirlos sin conocerlos)
perfiles = {
    #  nombre             (visitas, gasto, antigüedad), peso
    "rutina_ahorro":   dict(vis=(18, 2.5), gas=(38, 6),  ant=(14, 4), p=0.30),
    "gourmet_ocasional": dict(vis=(5, 1.8),  gas=(98, 12), ant=(10, 4), p=0.20),
    "nuevo_curioso":   dict(vis=(7, 2.0),  gas=(52, 8),  ant=(2, 1),  p=0.25),
    "leal_premium":    dict(vis=(15, 2.5), gas=(82, 10), ant=(22, 5), p=0.25),
}
nombres = list(perfiles)
probs = [perfiles[n]["p"] for n in nombres]
seg = rng.choice(nombres, size=N, p=probs)

# Tiempo de espera en fila (min): afecta negativamente las visitas (hipótesis de regresión)
espera = np.clip(rng.normal(8, 3.2, N), 1, None)

visitas, gasto, antig = [], [], []
for s, w in zip(seg, espera):
    d = perfiles[s]
    v = rng.normal(*d["vis"]) - 0.7 * (w - 8)      # cada minuto extra de espera ≈ -0.7 visitas/mes
    visitas.append(max(1, round(v)))
    gasto.append(max(15, rng.normal(*d["gas"])))
    antig.append(max(1, round(rng.normal(*d["ant"]))))

# Tarjeta de lealtad: más probable en perfiles leales; además gastan un poco más con ella
p_tarjeta = {"rutina_ahorro": 0.55, "gourmet_ocasional": 0.30, "nuevo_curioso": 0.20, "leal_premium": 0.85}
tarjeta = np.array([rng.random() < p_tarjeta[s] for s in seg])
gasto = np.array(gasto) + np.where(tarjeta, 6, 0)

tipo = rng.choice(["Estudiante", "Docente", "Administrativo"], size=N, p=[0.70, 0.18, 0.12])
turno = rng.choice(["Mañana", "Tarde"], size=N, p=[0.55, 0.45])
satisf = np.clip(np.round(8.4 - 0.12 * (espera - 8) + rng.normal(0, 1.0, N)), 1, 10)

df = pd.DataFrame({
    "id_cliente": [f"C{str(i+1).zfill(4)}" for i in range(N)],   # ID anónimo (sin nombres)
    "tipo_cliente": tipo,
    "tarjeta_lealtad": np.where(tarjeta, "Sí", "No"),
    "turno_preferido": turno,
    "visitas_mes": visitas,
    "gasto_promedio": np.round(gasto, 2),
    "tiempo_espera_min": np.round(espera, 1),
    "antiguedad_meses": antig,
    "satisfaccion": satisf.astype(int),
})

# ---- Problemas de calidad INTENCIONALES (U3) ----
# 1) Valores faltantes (la app de la cafetería a veces no registra estos campos)
for col, frac in [("tiempo_espera_min", 0.07), ("gasto_promedio", 0.05), ("satisfaccion", 0.04)]:
    idx = rng.choice(N, size=int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

# 2) Outliers: pedidos de catering de docentes (gasto altísimo) y fallas en caja (esperas enormes)
idx_cat = rng.choice(df.index[df.tipo_cliente != "Estudiante"], size=5, replace=False)
df.loc[idx_cat, "gasto_promedio"] = rng.uniform(260, 340, 5).round(2)
idx_fila = rng.choice(N, size=4, replace=False)
df.loc[idx_fila, "tiempo_espera_min"] = [38.0, 45.5, 52.0, 41.0]

df.to_csv("data/clientes_cafeteria_raw.csv", index=False)
print(df.shape)
print(df.isna().sum())
print(df.describe().round(2))
