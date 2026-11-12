"""Descarga velas de acciones desde la API de datos de Alpaca (plan gratuito Basic).

Necesita una clave de API gratuita (ver README): APCA_API_KEY_ID y APCA_API_SECRET_KEY,
en variables de entorno o en un archivo .env en la raíz del repo (que no se sube a git).

Uso desde la raíz del repo:

    python -m backend.data.alpaca --interval 1d --start 2020-01-01
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import polars as pl

from backend.config import load_env_file
from backend.data.common import INTERVAL_MS
from backend.data.quality import log_repairs, repair_spikes

log = logging.getLogger(__name__)

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
SOURCE = "alpaca_stocks"  # subcarpeta dentro de data/parquet/
DEFAULT_SYMBOLS = ["GOOGL", "NVDA", "XOM", "PLD"]  # Alphabet, NVIDIA, Exxon Mobil, Prologis

# Nuestro nombre de intervalo -> el que usa Alpaca.
TIMEFRAMES = {
    "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
    "1h": "1Hour", "4h": "4Hour", "1d": "1Day",
}

# El plan gratuito no deja pedir los últimos 15 minutos de datos SIP: pedimos hasta 20 min atrás.
DELAY = timedelta(minutes=20)


class MissingKeysError(RuntimeError):
    """No hay claves de API configuradas."""


def get_headers() -> dict[str, str]:
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise MissingKeysError(
            "Faltan APCA_API_KEY_ID y APCA_API_SECRET_KEY. "
            "Copia .env.example como .env y rellénalo con tus claves de Alpaca."
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def normalize_date(value: str) -> str:
    """'2024-01-01' -> '2024-01-01T00:00:00Z'; si ya lleva hora, se deja igual."""
    return f"{value}T00:00:00Z" if len(value) == 10 else value


def fetch_bars(
    client: httpx.Client,
    symbols: list[str],
    interval: str,
    start: str,
    end: str,
    adjustment: str = "all",
    feed: str = "sip",
) -> dict[str, list[dict]]:
    """Pide las velas de varias acciones de una vez y sigue la paginación hasta el final.

    Alpaca devuelve como máximo 10.000 velas por respuesta; si hay más, incluye un
    `next_page_token` que hay que reenviar para recibir la siguiente página.
    """
    params = {
        "symbols": ",".join(symbols),
        "timeframe": TIMEFRAMES[interval],
        "start": start,
        "end": end,
        "limit": 10_000,
        "adjustment": adjustment,
        "feed": feed,
        "sort": "asc",
    }
    result: dict[str, list[dict]] = {s: [] for s in symbols}
    while True:
        resp = client.get(BARS_URL, params=params)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Alpaca devolvió {resp.status_code}: {resp.text[:300]}\n"
                "Si es un 401/403 revisa las claves; si habla de datos SIP, prueba --feed iex."
            )
        body = resp.json()
        for symbol, bars in (body.get("bars") or {}).items():
            result.setdefault(symbol, []).extend(bars)
        token = body.get("next_page_token")
        if not token:
            return result
        params["page_token"] = token


def bars_to_frame(bars: list[dict]) -> pl.DataFrame:
    """Convierte la lista de velas de Alpaca al formato común (open_time, open, ... volume)."""
    return pl.DataFrame(bars).select(
        pl.col("t")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ")
        .dt.replace_time_zone("UTC")
        .alias("open_time"),
        pl.col("o").cast(pl.Float64).alias("open"),
        pl.col("h").cast(pl.Float64).alias("high"),
        pl.col("l").cast(pl.Float64).alias("low"),
        pl.col("c").cast(pl.Float64).alias("close"),
        pl.col("v").cast(pl.Float64).alias("volume"),
        pl.col("n").cast(pl.Int64).alias("n_trades"),
        pl.col("vw").cast(pl.Float64).alias("vwap"),
    )


def build_dataset(
    symbols: list[str],
    interval: str,
    start: str,
    end: str | None = None,
    feed: str = "sip",
    adjustment: str = "all",
    data_dir: Path = Path("data"),
) -> list[Path]:
    """Descarga las acciones pedidas y las fusiona con lo que ya hubiera guardado."""
    if interval not in TIMEFRAMES:
        raise ValueError(f"Intervalo no soportado: {interval}. Usa uno de {list(TIMEFRAMES)}")
    symbols = [s.upper() for s in symbols]
    end = end or (datetime.now(timezone.utc) - DELAY).strftime("%Y-%m-%dT%H:%M:%SZ")

    with httpx.Client(headers=get_headers(), timeout=60) as client:
        data = fetch_bars(
            client, symbols, interval, normalize_date(start), normalize_date(end), adjustment, feed
        )

    written: list[Path] = []
    for symbol in symbols:
        if not data.get(symbol):
            log.warning("Sin datos para %s en ese rango", symbol)
            continue
        df, changes = repair_spikes(bars_to_frame(data[symbol]))
        for row in changes.iter_rows(named=True):
            log.warning(
                "Pico sospechoso corregido en %s %s (%s): %.2f -> %.2f",
                symbol, f"{row['open_time']:%Y-%m-%d}", row["field"], row["old"], row["new"],
            )
        log_repairs(data_dir, SOURCE, symbol, interval, changes)
        out = data_dir / "parquet" / SOURCE / f"{symbol}_{interval}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():  # fusionar con lo anterior sin duplicar velas
            df = pl.concat([pl.read_parquet(out), df])
        df = df.unique(subset="open_time", keep="last").sort("open_time")
        df.write_parquet(out)
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> None:
    load_env_file()
    p = argparse.ArgumentParser(description="Descarga velas de acciones de Alpaca a Parquet.")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--interval", default="1d", choices=sorted(TIMEFRAMES, key=INTERVAL_MS.get))
    p.add_argument("--start", default="2020-01-01", help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (por defecto, hasta hace 20 minutos)")
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--adjustment", default="all", choices=["raw", "split", "dividend", "all"])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        paths = build_dataset(
            args.symbols, args.interval, args.start, args.end, args.feed, args.adjustment, args.data_dir
        )
    except (MissingKeysError, RuntimeError, ValueError) as e:
        sys.exit(f"\nError: {e}")

    print()
    for path in paths:
        df = pl.read_parquet(path)
        print(
            f"{path.stem:<12} {df.height:>7,} velas  "
            f"{df['open_time'].min():%Y-%m-%d} → {df['open_time'].max():%Y-%m-%d}  "
            f"último cierre {df['close'][-1]:,.2f}"
        )


if __name__ == "__main__":
    main()
