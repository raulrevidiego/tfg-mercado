"""Controles de calidad de datos: velas con picos sospechosos.

Un máximo o un mínimo que se aleja mucho de los cierres de la vela anterior, de la propia
vela y de la siguiente suele ser un dato erróneo del proveedor, no un movimiento real.

Uso desde la raíz del repo:

    python -m backend.data.quality
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def find_suspicious_candles(df: pl.DataFrame, threshold: float = 0.20) -> pl.DataFrame:
    """Velas cuyo máximo supera en más de `threshold` (20 %) al mayor de los tres cierres
    vecinos, o cuyo mínimo queda más de `threshold` por debajo del menor."""
    return (
        df.with_columns(prev=pl.col("close").shift(1), nxt=pl.col("close").shift(-1))
        .with_columns(
            ref_hi=pl.max_horizontal("prev", "close", "nxt"),
            ref_lo=pl.min_horizontal("prev", "close", "nxt"),
        )
        .with_columns(
            up_spike=pl.col("high") / pl.col("ref_hi") - 1,
            down_spike=1 - pl.col("low") / pl.col("ref_lo"),
        )
        .filter((pl.col("up_spike") > threshold) | (pl.col("down_spike") > threshold))
        .select("open_time", "open", "high", "low", "close", "up_spike", "down_spike")
    )


def repair_spikes(df: pl.DataFrame, threshold: float = 0.20) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Corrige los picos sospechosos recortándolos al rango apertura-cierre de su vela.

    No conocemos el máximo (o mínimo) real de esa vela, pero sí que no puede ser un pico
    aislado; max(open, close) es una estimación prudente. Devuelve la serie corregida y
    una tabla con cada cambio (open_time, field, old, new) para dejar constancia.
    """
    marks = find_suspicious_candles(df, threshold).select(
        "open_time",
        (pl.col("up_spike") > threshold).alias("_fix_high"),
        (pl.col("down_spike") > threshold).alias("_fix_low"),
    )
    joined = df.join(marks, on="open_time", how="left").with_columns(
        pl.col("_fix_high", "_fix_low").fill_null(False)
    )
    new_high = pl.max_horizontal("open", "close")
    new_low = pl.min_horizontal("open", "close")
    fixed = joined.with_columns(
        high=pl.when(pl.col("_fix_high")).then(new_high).otherwise(pl.col("high")),
        low=pl.when(pl.col("_fix_low")).then(new_low).otherwise(pl.col("low")),
    ).drop("_fix_high", "_fix_low")
    changes = pl.concat([
        joined.filter(pl.col("_fix_high")).select(
            "open_time", pl.lit("high").alias("field"), pl.col("high").alias("old"), new_high.alias("new")
        ),
        joined.filter(pl.col("_fix_low")).select(
            "open_time", pl.lit("low").alias("field"), pl.col("low").alias("old"), new_low.alias("new")
        ),
    ]).sort("open_time")
    return fixed, changes


def log_repairs(
    data_dir: Path, source: str, symbol: str, interval: str, changes: pl.DataFrame
) -> Path | None:
    """Anota las correcciones en data/quality/repairs.csv (sin duplicar si se repite)."""
    if changes.height == 0:
        return None
    path = data_dir / "quality" / "repairs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = changes.select(
        pl.lit(source).alias("source"),
        pl.lit(symbol).alias("symbol"),
        pl.lit(interval).alias("interval"),
        pl.col("open_time").dt.strftime("%Y-%m-%d %H:%M:%S").alias("open_time"),
        "field", "old", "new",
    )
    if path.exists():
        rows = pl.concat([pl.read_csv(path), rows]).unique(maintain_order=True)
    rows.write_csv(path)
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Busca velas sospechosas en todos los Parquet.")
    p.add_argument("--threshold", type=float, default=0.20)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    files = sorted((args.data_dir / "parquet").glob("*/*.parquet"))
    total = 0
    for f in files:
        sus = find_suspicious_candles(pl.read_parquet(f), args.threshold)
        total += sus.height
        if sus.height:
            print(f"\n{f.parent.name}/{f.name}: {sus.height} vela(s) sospechosa(s)")
            print(sus)
    print(f"\nRevisados {len(files)} archivos; velas sospechosas: {total}")


if __name__ == "__main__":
    main()
