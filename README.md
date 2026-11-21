# TFG Mercado

Software de escritorio para analizar bolsa y cripto con deep learning.
Estructura: `core/` (C++), `backend/` (Python), `ui/` (web), `data/` (no se sube a git).

## Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python -m backend.data.binance --symbol BTCUSDT --interval 1m --start 2024-01 --end 2024-03
```

El último comando descarga velas de Binance Vision y crea
`data/parquet/binance_spot/BTCUSDT_1m.parquet`.

## Interfaz de prototipo

Con datos ya descargados, desde la raíz del repo:

```bash
uvicorn backend.api.main:app --reload
```

Abre http://127.0.0.1:8000 (gráfico de velas) o http://127.0.0.1:8000/docs (documentación de la API).

## Acciones (Alpaca)

1. Crea una cuenta gratuita en Alpaca (basta con *paper trading*) y genera una clave de API.
2. `cp .env.example .env` y pega tus claves en `.env` (este archivo no se sube a git).
3. Descarga velas diarias de la lista de seguimiento (GOOGL, NVDA, XOM, PLD):

```bash
python -m backend.data.alpaca --interval 1d --start 2020-01-01
```

Los precios se descargan ajustados por splits y dividendos (`--adjustment all`).

## Series de referencia y volatilidad

Mercado (SPY) y ETF sectoriales (tecnología XLK, comunicaciones XLC, energía XLE, inmobiliario XLRE):

```bash
python -m backend.data.alpaca --symbols SPY XLK XLC XLE XLRE --interval 1d --start 2020-01-01
```

Evaluaciones (necesitan las series descargadas):

```bash
python -m backend.training.evaluate_baselines    # retorno diario
python -m backend.training.evaluate_volatility   # rango diario: baselines y HAR
python -m backend.training.evaluate_har_x        # HAR + mercado, sector y volumen
```

## Eventos (comentarios de CEOs)

1. Comprueba la herramienta con fechas al azar (no necesita eventos):

```bash
python -m backend.events.evaluate_events --placebo
```

2. Escribe tus eventos: `mkdir -p data/events && cp docs/events.example.csv data/events/events.csv`
   y añade una fila por comentario (acción, fecha y hora exactas con zona horaria, categoría).
3. Analízalos:

```bash
python -m backend.events.evaluate_events
```

### Eventos automáticos desde EDGAR (8-K)

Los 8-K son los avisos que las empresas presentan a la SEC (resultados, cambios de directivos,
presentaciones a inversores...). Traen fecha y hora exactas. Necesitas poner en `.env` tu nombre y
tu correo (la SEC lo exige):

```bash
# en .env:  SEC_USER_AGENT=Tu Nombre tu@correo.com
python -m backend.data.edgar --start 2020-01-01        # crea data/events/edgar_8k.csv
python -m backend.events.evaluate_events --events data/events/edgar_8k.csv
```

Categorías: `resultados` (item 2.02), `directivos` (5.02), `regfd` (7.01), `otros_eventos` (8.01) y `otros`.

Notas sobre los eventos de EDGAR:
- **XOM:** Exxon Mobil cambió de registrante el 1 de julio de 2026 (ExxonMobil Holdings, Texas), así que
  su historial anterior está bajo otro CIK (34088). El ingestor lo une solo; para otros casos:
  `python -m backend.data.edgar --extra-cik SÍMBOLO=CIK`. Vuelve a ejecutarlo para completar XOM.
- El análisis fusiona los avisos del mismo día y descarta los que están a menos de 5 sesiones de otro de mayor
  prioridad (`--min-gap`, `--prioridad`), porque comparten reacción y exagerarían la significancia.
- `--detalle` lista todos los eventos; por defecto solo muestra el resumen y los de mayor reacción.

## Deep learning (PyTorch y GPU)

```bash
nvidia-smi                          # debe mostrar tu GPU y la "CUDA Version" que admite el driver
pip install torch numpy             # la rueda de PyPI ya incluye CUDA
python -m backend.training.check_gpu
```

Si `nvidia-smi` muestra una versión de CUDA 12.x, usa una rueda para tu driver:
`pip install torch --index-url https://download.pytorch.org/whl/cu128` (o `cu126`).
Si pip no encuentra ninguna rueda para tu versión de Python, crea un entorno con Python 3.13 o 3.12
(Omarchy trae `mise`):

```bash
mise exec python@3.13 -- python -m venv ~/Work/TFG/TFGvenv313
source ~/Work/TFG/TFGvenv313/bin/activate
pip install -e ".[dev]" && pip install torch
```

`backend/models/dataset.py` prepara las ventanas de días y el escalado para las redes neuronales sin
usar PyTorch, de modo que los tests puedan comprobar que no se filtra información del futuro.

## Volatilidad con calendario de resultados (HAR-E)

Primero vuelve a generar los eventos (une el historial antiguo de XOM) y luego evalúa:

```bash
python -m backend.data.edgar --start 2020-01-01
python -m backend.training.evaluate_har_e
```

Compara HAR con HAR-E (que sabe si mañana o hoy hay resultados) por acción y, en conjunto, en la
víspera de resultados, el día de resultados y el resto de días.

## Tono de los comunicados de resultados

```bash
python -m backend.data.edgar_text --category resultados   # descarga el texto de cada comunicado (unos minutos)
python -m backend.events.evaluate_tone                    # ¿predice el tono la reacción del precio?
```

El primero guarda los textos en `data/raw/edgar_text/` y un índice en `data/events/texts_index.csv`, y avisa
si algún texto parece solo la carátula del 8-K. El segundo usa un lexicón propio (baseline) y compara el tono
de cada comunicado con el de los anteriores de la misma empresa.

Si `check_gpu` dice que tu GPU es demasiado antigua (por ejemplo una MX250, capacidad 6.1), no es un fallo:
las ruedas actuales de PyTorch solo traen kernels desde la capacidad 7.5 (serie RTX 20 en adelante).
En ese equipo se trabaja con la CPU; las RTX 30, 40 y 50 sí están cubiertas.

## Pantalla "¿Y si…?"

```bash
python -m backend.events.build_reactions        # una fila por evento + fechas al azar de referencia
uvicorn backend.api.main:app --reload           # y abre http://127.0.0.1:8000/whatif.html
```

Muestra, para una acción y un tipo de evento (y un tono, si has descargado los textos), cuánto se ha movido
la acción en la historia frente a un día normal, hacia dónde y con qué incertidumbre. Describe el pasado: no
es una predicción. Hay que volver a ejecutar `build_reactions` cuando cambien los eventos o los textos.

## Red neuronal de volatilidad (paso 16)

```bash
python -m backend.training.evaluate_nn                  # perceptrón multicapa, 3 semillas, usa la GPU si la ve
python -m backend.training.evaluate_nn --arch gru       # red recurrente
python -m backend.training.evaluate_nn --device cpu     # forzar CPU
```

Compara la red con `har_enext` (HAR + "mañana hay resultados") en cada acción y, en conjunto, en la víspera de
resultados, el día de resultados y el resto de días. Guarda las predicciones en `data/models/nn_predictions.parquet`.
Los tests de PyTorch se ejecutan solos si PyTorch está instalado (`pytest tests/test_nn_vol.py`).
