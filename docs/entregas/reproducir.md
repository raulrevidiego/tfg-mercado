# Cómo reproducir los resultados

Todos los comandos se ejecutan desde la raíz del repositorio (`tfg-mercado/`) con el entorno virtual activado. Los datos no se suben a git (`/data/` está en `.gitignore`): hay que descargarlos con los pasos 1 y 2.

## 0. Requisitos

- Python 3.14 (el proyecto se desarrolló con 3.14.4). Para la red neuronal, PyTorch 2.15 con CUDA 13.2; con `--device cpu` funciona sin GPU, más despacio.
- Para el núcleo C++ opcional: un compilador de C++ (g++ o clang++).
- Una cuenta gratuita de Alpaca (basta con *paper trading*) y las claves en `.env` (copia `.env.example`; el archivo `.env` no se sube a git). La SEC exige identificarse con nombre y correo en las peticiones a EDGAR: también va en `.env`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install torch numpy          # solo para la red neuronal
pip install -e ".[core]"         # solo para el núcleo C++ (pybind11, setuptools)
pip install -e ".[nlp]"          # solo para FinBERT (transformers)
pytest
```

`pytest` debe terminar sin fallos. Se saltan solos los tests de PyTorch si no está instalado, los de C++ si no está compilado y uno opcional con el modelo real de FinBERT (`TFG_TEST_FINBERT=1`).

## 1. Datos

```bash
python -m backend.data.binance --symbol BTCUSDT --interval 1m --start 2024-01 --end 2024-03
python -m backend.data.alpaca --interval 1d --start 2020-01-01
python -m backend.data.alpaca --symbols SPY XLK XLC XLE XLRE --interval 1d --start 2020-01-01
python -m backend.data.quality
python -m backend.data.edgar --start 2020-01-01
python -m backend.data.edgar_text --category resultados
```

- `quality` revisa las velas (huecos, valores raros). Tras corregir el dato erróneo de NVDA del 10 de junio de 2024 debe dar 0 velas sospechosas.
- `edgar` crea `data/events/edgar_8k.csv`. Exxon Mobil cambió de registrante el 1 de julio de 2026: el ingestor une automáticamente el CIK antiguo (34088); si XOM sale con muy pocos eventos, vuelve a ejecutarlo.
- `edgar_text` descarga el texto de cada comunicado de resultados (unos minutos).
- Las descargas se actualizan fusionando con lo ya guardado: repetirlas no duplica datos. Los resultados de este documento usan los datos hasta el 2 de noviembre de 2026; con datos posteriores las cifras cambiarán algo.

## 2. Resultados, en el orden del capítulo

| Apartado | Comando |
|---|---|
| 2. Dirección del precio | `python -m backend.training.evaluate_baselines` |
| 3. Volatilidad (media, EWMA, HAR) | `python -m backend.training.evaluate_volatility` |
| 3.1 Mercado, sector y volumen | `python -m backend.training.evaluate_har_x` |
| 3.2 Calendario de resultados | `python -m backend.training.evaluate_har_e` |
| 3.3 Red neuronal | `python -m backend.training.evaluate_nn` |
| 4. Eventos | `python -m backend.events.evaluate_events --events data/events/edgar_8k.csv --motor cpp --draws 200000` |
| 4.1 Robustez | `python -m backend.events.evaluate_events --events data/events/edgar_8k.csv --motor cpp --draws 200000 --robustez` |
| 5. Tono (las cuatro variantes) | ver más abajo |

Las cuatro variantes del tono se ejecutan todas y siempre con `--variantes 4`, para que el umbral de Bonferroni (p < 0,0031) cuente las cuatro:

```bash
python -m backend.events.evaluate_tone --metodo lexicon --parte all --variantes 4
python -m backend.events.evaluate_tone --metodo lexicon --parte quotes --variantes 4
python -m backend.events.evaluate_tone --metodo finbert --parte all --variantes 4
python -m backend.events.evaluate_tone --metodo finbert --parte quotes --variantes 4
```

La primera vez, FinBERT se descarga (unos 440 MB) de Hugging Face. Para comprobar que funciona: `python -m backend.events.tone_finbert`. Debe puntuar bien tres de las cuatro frases de ejemplo (la segunda es un caso difícil que el modelo falla).

Otros comandos de apoyo:

```bash
python -m backend.training.check_gpu          # comprueba PyTorch y la GPU
python -m backend.events.benchmark_core       # Python frente a C++ (necesita python core/build.py)
python -m backend.events.build_reactions      # tabla de la pantalla «¿Y si…?»
uvicorn backend.api.main:app --reload         # interfaz en http://127.0.0.1:8000
```

## 3. Para que los números coincidan

- **Motor del test de permutaciones.** Los p del estudio de eventos de este capítulo se calcularon con `--motor cpp --draws 200000`. El motor `python` (el valor por defecto) usa otro generador aleatorio: con la misma semilla da p que difieren solo por el ruido del Monte Carlo (del orden de √(p(1−p)/remuestreos)), no idénticos. Dentro de cada motor, el resultado es exactamente el mismo con la misma semilla; en C++ además no depende del número de hilos ni de la máquina.
- **Semilla.** `--seed 7` por defecto.
- **Cuántos remuestreos.** Con 200.000, el p mínimo posible es 1/200.001 (por eso se escribe «< 0,00001»). Con `--motor python` y esa cantidad, `--robustez` tarda mucho: usa `--motor cpp`.
- **Red neuronal.** Se promedian 3 semillas. En GPU, y entre máquinas distintas, las cifras pueden variar ligeramente de una ejecución a otra; los márgenes de error (±) indican cuánto.
- **Datos.** Las cifras dependen de la fecha de descarga (los datos de Alpaca se ajustan por splits y dividendos) y de que los 8-K estén completos. Compara los recuentos con los del capítulo (353 eventos analizados, 295 independientes, 1.718 velas por acción) antes de comparar los resultados.
