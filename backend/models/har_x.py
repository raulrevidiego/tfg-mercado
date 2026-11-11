"""HAR ampliado (HAR-X): HAR más variables del mercado, del sector y del volumen.

Pregunta que responde: ¿saber cuánto se mueve hoy el mercado (SPY), el sector de la acción o
cuánto volumen hay aporta algo a predecir el rango de mañana, además de la propia historia?

Todas las variables de un día t usan solo datos hasta t, y el modelo se reajusta con walk-forward
exactamente igual que HAR (ver `volatility.har_forecasts`).
"""

from __future__ import annotations

import polars as pl

from backend.models.volatility import EWMA_LAMBDA, har_forecasts

MARKET_ETF = "SPY"
# ETF sectorial de referencia de cada acción de la lista de seguimiento.
SECTOR_ETF = {"GOOGL": "XLC", "NVDA": "XLK", "XOM": "XLE", "PLD": "XLRE"}

HAR = ["rng", "mean_5", "mean_22"]      # el propio HAR: hoy, semana, mes
MKT = ["mkt_rng", "mkt_5"]              # rango del mercado hoy y media de 5 días
SEC = ["sec_rng", "sec_5"]              # lo mismo para el ETF del sector
VOL = ["relvol"]                        # volumen de hoy frente a su media de 22 días (en logaritmo)


def forecast_har_x_frame(
    stock: pl.DataFrame,
    market: pl.DataFrame,
    sector: pl.DataFrame | None = None,
    warmup: int = 500,
    refit_every: int = 21,
) -> tuple[pl.DataFrame, list[str]]:
    """Devuelve (tabla con las predicciones de cada variante, nombres de los modelos).

    Las series se alinean por fecha (solo se usan los días presentes en todas) ANTES de calcular
    nada, para que "mañana" sea siempre el siguiente día común.
    """
    joined = stock.select("open_time", "high", "low", "volume").join(
        market.select("open_time", mh=pl.col("high"), ml=pl.col("low")), on="open_time", how="inner"
    )
    if sector is not None:
        joined = joined.join(
            sector.select("open_time", sh=pl.col("high"), sl=pl.col("low")), on="open_time", how="inner"
        )
    joined = joined.sort("open_time")

    ranges = [
        (pl.col("high") / pl.col("low")).log().alias("rng"),
        (pl.col("mh") / pl.col("ml")).log().alias("mkt_rng"),
    ]
    if sector is not None:
        ranges.append((pl.col("sh") / pl.col("sl")).log().alias("sec_rng"))
    frame = joined.with_columns(ranges)

    derived = [
        pl.col("rng").shift(-1).alias("target"),
        pl.col("rng").rolling_mean(5).alias("mean_5"),
        pl.col("rng").rolling_mean(22).alias("mean_22"),
        pl.col("rng").ewm_mean(alpha=1 - EWMA_LAMBDA, adjust=False).alias("ewma_094"),
        pl.col("mkt_rng").rolling_mean(5).alias("mkt_5"),
        pl.when(pl.col("volume") > 0)
        .then((pl.col("volume") / pl.col("volume").rolling_mean(22)).log())
        .alias("relvol"),
    ]
    if sector is not None:
        derived.append(pl.col("sec_rng").rolling_mean(5).alias("sec_5"))
    frame = frame.with_columns(derived)

    variants = {"har": HAR, "har_mkt": HAR + MKT, "har_vol": HAR + VOL}
    if sector is not None:
        variants["har_sector"] = HAR + SEC
    variants["har_all"] = HAR + MKT + VOL + (SEC if sector is not None else [])

    target = frame["target"].to_list()
    predictions = []
    for name, cols in variants.items():
        features = list(zip(*(frame[c].to_list() for c in cols)))
        preds = har_forecasts(features, target, warmup, refit_every)
        predictions.append(pl.Series(name, preds, dtype=pl.Float64))
    return frame.with_columns(predictions), list(variants)
