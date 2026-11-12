"""Lectura de las series guardadas en data/parquet/."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_series(
    symbol: str, interval: str = "1d", source: str = "alpaca_stocks", data_dir: Path = Path("data")
) -> pl.DataFrame:
    """Carga una serie de velas ordenada por fecha."""
    path = data_dir / "parquet" / source / f"{symbol.upper()}_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Descárgala antes con el ingestor.")
    return pl.read_parquet(path).sort("open_time")
