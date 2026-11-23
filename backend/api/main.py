"""API local del TFG: sirve velas desde los Parquet y la interfaz de prototipo.

Arranque (desde la raíz del repo):

    uvicorn backend.api.main:app --reload

Después abre http://127.0.0.1:8000 (interfaz) o http://127.0.0.1:8000/docs (documentación
automática de la API, generada por FastAPI).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from backend.data.common import INTERVAL_MS
from backend.data.store import load_series
from backend.events.reactions import load_reactions, what_if
from backend.models.volatility import forecast_vol_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = REPO_ROOT / "ui" / "prototype"
SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,20}$")  # evita rutas raras como "../algo"
SOURCE_RE = re.compile(r"^[a-z0-9_]{1,40}$")

DAY_MS = 86_400_000
# Temporalidades que se pueden VER: las de los datos más semana y mes (aproximado a 30 días).
TF_MS = INTERVAL_MS | {"1w": 7 * DAY_MS, "1M": 30 * DAY_MS}
POLARS_EVERY = {"1M": "1mo"}  # cómo llama Polars a cada temporalidad cuando difiere

app = FastAPI(title="TFG Mercado API")


def parquet_root() -> Path:
    """Carpeta con los Parquet, una subcarpeta por fuente (binance_spot, alpaca_stocks...).

    Se lee la variable de entorno en cada llamada (útil en tests).
    """
    data_dir = Path(os.environ.get("TFG_DATA_DIR", REPO_ROOT / "data"))
    return data_dir / "parquet"


def resample(df: pl.DataFrame, tf: str) -> pl.DataFrame:
    """Agrupa velas finas en velas más gruesas (p. ej. 60 velas de 1m -> 1 de 1h).

    Una vela agregada toma: open de la primera, high el máximo, low el mínimo,
    close de la última y volume la suma.
    """
    every = POLARS_EVERY.get(tf, tf)
    return df.group_by_dynamic("open_time", every=every, closed="left", label="left").agg(
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    )


@app.get("/api/datasets")
def list_datasets() -> list[dict[str, str]]:
    """Series descargadas, p. ej. [{"source": "alpaca_stocks", "symbol": "NVDA", "interval": "1d"}]."""
    found = []
    root = parquet_root()
    for source_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        for f in sorted(source_dir.glob("*_*.parquet")):
            symbol, interval = f.stem.rsplit("_", 1)
            found.append({"source": source_dir.name, "symbol": symbol, "interval": interval})
    return found


@app.get("/api/candles")
def candles(
    source: str = "binance_spot",
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    tf: str = "1h",
    limit: int = Query(1500, ge=1, le=5000),
) -> list[dict]:
    """Últimas `limit` velas de `symbol` (de la fuente `source`), agrupadas en la temporalidad `tf`.

    `interval` es la resolución del Parquet descargado y `tf` la que quieres ver.
    `time` va en segundos Unix (UTC), el formato que espera Lightweight Charts.
    """
    symbol = symbol.upper()
    if not SYMBOL_RE.match(symbol) or not SOURCE_RE.match(source):
        raise HTTPException(400, "Símbolo o fuente no válidos")
    if interval not in INTERVAL_MS or tf not in TF_MS:
        raise HTTPException(400, f"Temporalidades admitidas: {', '.join(TF_MS)}")
    if TF_MS[tf] < INTERVAL_MS[interval] or TF_MS[tf] % INTERVAL_MS[interval]:
        raise HTTPException(400, "tf debe ser igual o múltiplo de interval")

    path = parquet_root() / source / f"{symbol}_{interval}.parquet"
    if not path.exists():
        raise HTTPException(404, f"No hay datos de {symbol} {interval}: ejecuta antes el ingestor")

    df = pl.read_parquet(path, columns=["open_time", "open", "high", "low", "close", "volume"])
    if tf != interval:
        df = resample(df, tf)
    return (
        df.tail(limit)
        .select(
            time=pl.col("open_time").dt.epoch("s"),
            open="open", high="high", low="low", close="close", volume="volume",
        )
        .to_dicts()
    )


@app.get("/api/watchlist")
def watchlist(source: str = "alpaca_stocks") -> list[dict]:
    """Resumen de cada serie diaria de una fuente: último cierre y variaciones en %.

    Las variaciones se miden en velas (sesiones): 1 = día anterior, 5 = una semana de
    mercado, 21 = aproximadamente un mes.
    """
    if not SOURCE_RE.match(source):
        raise HTTPException(400, "Fuente no válida")
    rows = []
    for f in sorted((parquet_root() / source).glob("*_1d.parquet")):
        df = pl.read_parquet(f, columns=["open_time", "close"])
        if df.height < 2:
            continue
        close = df["close"]
        last = close[-1]

        def change(n: int) -> float | None:
            return (last / close[-1 - n] - 1) * 100 if df.height > n else None

        rows.append({
            "symbol": f.stem.rsplit("_", 1)[0],
            "date": df["open_time"][-1].date().isoformat(),
            "close": last,
            "change_1d": change(1),
            "change_5d": change(5),
            "change_1m": change(21),
        })
    return rows


# ------------------------------------------------------------------ pantalla "¿y si...?"

CATEGORY_RE = re.compile(r"^[a-z0-9_]{1,30}$")
_BASELINE_CACHE: dict[tuple, float | None] = {}


def data_root() -> Path:
    return Path(os.environ.get("TFG_DATA_DIR", REPO_ROOT / "data"))


def _reactions() -> list[dict]:
    path = data_root() / "events" / "reactions.csv"
    if not path.exists():
        raise HTTPException(404, "No hay tabla de reacciones: ejecuta python -m backend.events.build_reactions")
    return load_reactions(path)


def _baseline_range(symbol: str) -> float | None:
    """Rango diario que el HAR espera para la próxima sesión (sin evento). None si no hay datos."""
    path = data_root() / "parquet" / "alpaca_stocks" / f"{symbol}_1d.parquet"
    if not path.exists():
        return None
    key = (str(path), path.stat().st_mtime)
    if key not in _BASELINE_CACHE:
        try:
            last = forecast_vol_frame(load_series(symbol, data_dir=data_root()))["har"][-1]
        except ValueError:
            last = None
        _BASELINE_CACHE[key] = last
    return _BASELINE_CACHE[key]


@app.get("/api/whatif/options")
def whatif_options() -> dict:
    """Qué se puede consultar: acciones, categorías (con su número de eventos) y si hay datos de tono."""
    events = [r for r in _reactions() if r["kind"] == "evento"]
    return {
        "symbols": sorted({r["symbol"] for r in events}),
        "categories": dict(Counter(r["category"] for r in events)),
        "has_tone": any(r["tone_rel"] is not None for r in events),
    }


@app.get("/api/whatif")
def whatif(symbol: str = "NVDA", category: str = "resultados", tone: str = "any", scope: str = "symbol") -> dict:
    """Qué ha pasado en la historia tras eventos de este tipo, y qué rango cabe esperar si ocurre uno."""
    symbol = symbol.upper()
    if not SYMBOL_RE.match(symbol) or not CATEGORY_RE.match(category):
        raise HTTPException(400, "Símbolo o categoría no válidos")
    if tone not in {"any", "positivo", "negativo", "neutro"} or scope not in {"symbol", "all"}:
        raise HTTPException(400, "tone: any, positivo, negativo o neutro; scope: symbol o all")
    result = what_if(_reactions(), symbol, category, tone, scope)
    baseline = _baseline_range(symbol) if scope == "symbol" else None
    result["baseline_range"] = baseline
    if baseline is not None and "range" in result:
        result["expected_range_event"] = baseline * result["range"]["median_multiplier"]
    return result


# La interfaz se monta la última: así /api/... tiene prioridad sobre los archivos estáticos.
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
