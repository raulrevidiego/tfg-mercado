"""Descarga y normaliza velas (klines) de Binance Vision.

Binance Vision es un archivo público de ficheros ZIP: no necesita clave de API.
URL de los ficheros mensuales (mercado spot):

    https://data.binance.vision/data/spot/monthly/klines/<SYM>/<INT>/<SYM>-<INT>-<YYYY>-<MM>.zip

Uso desde la raíz del repo:

    python -m backend.data.binance --symbol BTCUSDT --interval 1m --start 2024-01 --end 2024-03
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

import httpx
import polars as pl

from backend.data.common import INTERVAL_MS

log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

# Orden de columnas de los CSV de Binance (12 columnas).
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

# Los timestamps de spot pasaron de milisegundos a microsegundos en 2025.
# Un timestamp en ms vale ~1.7e12 y en µs ~1.7e15, así que 1e14 los separa sin ambigüedad.
_MICROSECONDS_THRESHOLD = 10**14


# --------------------------------------------------------------------------- URLs

def month_range(start: str, end: str) -> list[str]:
    """Devuelve ['2024-01', '2024-02', ...] desde `start` hasta `end`, ambos incluidos."""
    y, m = (int(p) for p in start.split("-"))
    y_end, m_end = (int(p) for p in end.split("-"))
    months: list[str] = []
    while (y, m) <= (y_end, m_end):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return months


def monthly_url(symbol: str, interval: str, year_month: str) -> str:
    symbol = symbol.upper()
    return f"{BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{year_month}.zip"


# ----------------------------------------------------------------------- descarga

def download_month(
    client: httpx.Client, symbol: str, interval: str, year_month: str, raw_dir: Path
) -> Path | None:
    """Descarga un ZIP mensual a `raw_dir`. Si ya existe, no vuelve a bajarlo (caché).

    Devuelve None si Binance no lo tiene (404): típicamente el mes en curso, porque
    los ficheros mensuales se publican a principios del mes siguiente.
    """
    dest = raw_dir / f"{symbol.upper()}-{interval}-{year_month}.zip"
    if dest.exists():
        return dest

    url = monthly_url(symbol, interval, year_month)
    resp = client.get(url)
    if resp.status_code == 404:
        log.warning("No disponible: %s (¿mes en curso, o símbolo/intervalo inexistente?)", url)
        return None
    resp.raise_for_status()

    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")  # se escribe aparte y se renombra: evita ZIPs a medias
    tmp.write_bytes(resp.content)
    tmp.replace(dest)
    log.info("Descargado %s", dest.name)
    return dest


# ------------------------------------------------------------------------ parseo

def _to_ms(col: str) -> pl.Expr:
    """Normaliza un timestamp entero a milisegundos (acepta ms o µs)."""
    c = pl.col(col).cast(pl.Int64)
    return pl.when(c >= _MICROSECONDS_THRESHOLD).then(c // 1000).otherwise(c)


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Tipos correctos: tiempos como datetime UTC, precios/volúmenes como Float64."""
    return (
        df.with_columns(
            _to_ms("open_time").alias("open_time"),
            _to_ms("close_time").alias("close_time"),
        )
        .with_columns(
            pl.from_epoch("open_time", time_unit="ms").dt.replace_time_zone("UTC"),
            pl.from_epoch("close_time", time_unit="ms").dt.replace_time_zone("UTC"),
            pl.col(
                "open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote",
            ).cast(pl.Float64),
            pl.col("n_trades").cast(pl.Int64),
        )
        .drop("ignore")
    )


def parse_klines_csv(data: bytes) -> pl.DataFrame:
    """Lee el contenido de un CSV de klines (con o sin fila de cabecera)."""
    first_line = data.split(b"\n", 1)[0]
    has_header = not first_line[:1].isdigit()  # sin cabecera, la 1ª línea empieza por un número
    df = pl.read_csv(io.BytesIO(data), has_header=has_header, infer_schema_length=10_000)
    if df.width != len(KLINE_COLUMNS):
        raise ValueError(f"Se esperaban {len(KLINE_COLUMNS)} columnas y hay {df.width}")
    df.columns = KLINE_COLUMNS
    return normalize(df)


def read_zip(path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(path) as zf:
        csvs = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"{path.name}: se esperaba 1 CSV y hay {len(csvs)}")
        return parse_klines_csv(zf.read(csvs[0]))


# ------------------------------------------------------------------------ calidad

def find_gaps(df: pl.DataFrame, interval: str) -> pl.DataFrame:
    """Huecos en la serie: velas que faltan entre dos velas consecutivas."""
    step = INTERVAL_MS[interval]
    return (
        df.select("open_time")
        .with_columns(prev=pl.col("open_time").shift(1))
        .with_columns(gap_ms=(pl.col("open_time") - pl.col("prev")).dt.total_milliseconds())
        .filter(pl.col("gap_ms") > step)
        .select(
            pl.col("prev").alias("gap_start"),
            pl.col("open_time").alias("gap_end"),
            (pl.col("gap_ms") // step - 1).alias("missing_candles"),
        )
    )


# ----------------------------------------------------------------------- dataset

def build_dataset(
    symbol: str, interval: str, start: str, end: str, data_dir: Path = Path("data")
) -> Path:
    """Descarga los meses pedidos y genera un único Parquet ordenado y sin duplicados."""
    symbol = symbol.upper()
    raw_dir = data_dir / "raw" / "binance_spot" / symbol / interval
    out = data_dir / "parquet" / "binance_spot" / f"{symbol}_{interval}.parquet"

    frames: list[pl.DataFrame] = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for ym in month_range(start, end):
            zip_path = download_month(client, symbol, interval, ym, raw_dir)
            if zip_path is not None:
                frames.append(read_zip(zip_path))

    if not frames:
        raise RuntimeError("No se pudo descargar ningún mes: revisa símbolo, intervalo y fechas.")

    df = pl.concat(frames).unique(subset="open_time", keep="last").sort("open_time")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Descarga velas de Binance Vision a Parquet.")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1m", choices=sorted(INTERVAL_MS, key=INTERVAL_MS.get))
    p.add_argument("--start", required=True, help="mes inicial, formato YYYY-MM")
    p.add_argument("--end", required=True, help="mes final (incluido), formato YYYY-MM")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = build_dataset(args.symbol, args.interval, args.start, args.end, args.data_dir)

    df = pl.read_parquet(out)
    gaps = find_gaps(df, args.interval)
    print(f"\nGuardado: {out}")
    print(f"Velas: {df.height:,}  |  {df['open_time'].min()}  →  {df['open_time'].max()}")
    print(f"Huecos: {gaps.height}  (velas que faltan: {gaps['missing_candles'].sum() or 0})")
    print(df.tail(3))


if __name__ == "__main__":
    main()
