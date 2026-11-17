# ☕ Café La Corona: ¿a quién y cómo promocionar?
Proyecto final · INC-2505 Análisis y exploración de datos

Segmentación de clientes de una cafetería universitaria (dataset propio y **simulado**) con estadística descriptiva,
limpieza de datos, inferencia (t de Student y regresión) y K-means, presentado en un dashboard de Streamlit.

**Dashboard publicado:** _(pega aquí tu enlace de Streamlit Community Cloud)_

## Contenido
| Archivo | Qué es | Rúbrica |
|---|---|---|
| `Proyecto_INC2505.ipynb` | Análisis completo con interpretaciones | U1–U5 |
| `app.py` | Dashboard de Streamlit (6 pestañas, filtros, simulador) | U6 |
| `data/clientes_cafeteria_raw.csv` | Dataset crudo (con nulos y outliers intencionales) | U1, U3 |
| `generar_datos.py` | Script que genera el dataset (semilla fija) | U1 |
| `requirements.txt` | Dependencias | U6 |
| `.streamlit/config.toml` | Tema visual del dashboard | U6 |

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
