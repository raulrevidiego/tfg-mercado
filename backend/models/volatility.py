"""Baselines de volatilidad: predicen cuánto se va a mover la acción mañana.

Cantidad a predecir: el **rango diario**, ln(máximo / mínimo) del día. Es intuitivo ("mañana
oscilará un 2 % entre su máximo y su mínimo") y usa más información que el retorno de cierre a
cierre, así que es menos ruidoso.

Convención idéntica a baselines.py: la predicción hecha el día t usa SOLO datos hasta el día t y
apunta al rango del día t+1. La volatilidad tiene memoria (los días agitados tienden a seguir a
otros agitados), por eso los modelos que miran el pasado reciente deberían ganar a la media global.
"""

from __future__ import annotations

import polars as pl

from backend.models import ols

VOL_BASELINES = {
    "expanding_mean": "Mañana oscila lo de siempre (media de toda la historia)",
    "last_value": "Mañana oscila lo mismo que hoy",
    "mean_5": "Media de los últimos 5 días",
    "mean_20": "Media de los últimos 20 días",
    "ewma_094": "Media que pesa más los días recientes (lambda = 0,94)",
    "har": "Regresión HAR: combina hoy, la semana y el mes (se reajusta cada 21 días)",
}

EWMA_LAMBDA = 0.94      # valor clásico de RiskMetrics para datos diarios
FLOOR = 1e-4            # un rango previsto nunca baja del 0,01 %


def har_forecasts(
    features: list[tuple], target: list[float | None], start: int, refit_every: int = 21,
    min_train: int = 100,
) -> list[float | None]:
    """Predicciones del modelo HAR con walk-forward.

    HAR (Corsi, 2009) es una regresión lineal: rango de mañana ≈ b0 + b1·hoy + b2·media de la
    semana + b3·media del mes. Se reajusta cada `refit_every` días usando SOLO las filas cuyo
    objetivo ya se conocía en ese momento; entre reajustes se aplican los mismos coeficientes.
    """
    n = len(features)
    out: list[float | None] = [None] * n
    t0 = start
    while t0 < n:
        # Filas i <= t0-1: su objetivo es el rango del día i+1 <= t0, ya conocido en t0.
        idx = [i for i in range(t0) if None not in features[i] and target[i] is not None]
        if len(idx) >= min_train:
            coef = ols.fit([list(features[i]) for i in idx], [target[i] for i in idx])
            for t in range(t0, min(t0 + refit_every, n)):
                if None not in features[t]:
                    out[t] = max(ols.predict(coef, list(features[t])), FLOOR)
        t0 += refit_every
    return out


def forecast_vol_frame(df: pl.DataFrame, warmup: int = 500, refit_every: int = 21) -> pl.DataFrame:
    """Tabla con el rango de cada día (`rng`), el valor a acertar (`target`, el rango de mañana)
    y la predicción de cada baseline hecha ese día. HAR solo predice desde `warmup`."""
    frame = (
        df.sort("open_time")
        .select("open_time", "high", "low")
        .with_columns(rng=(pl.col("high") / pl.col("low")).log())
        .with_columns(
            target=pl.col("rng").shift(-1),
            expanding_mean=pl.col("rng").cum_sum() / pl.int_range(1, pl.len() + 1),
            last_value=pl.col("rng"),
            mean_5=pl.col("rng").rolling_mean(5),
            mean_20=pl.col("rng").rolling_mean(20),
            mean_22=pl.col("rng").rolling_mean(22),  # "el mes" de HAR (22 sesiones)
            ewma_094=pl.col("rng").ewm_mean(alpha=1 - EWMA_LAMBDA, adjust=False),
        )
    )
    har = har_forecasts(
        list(zip(frame["rng"], frame["mean_5"], frame["mean_22"])),
        frame["target"].to_list(), warmup, refit_every,
    )
    return frame.with_columns(har=pl.Series("har", har, dtype=pl.Float64))
