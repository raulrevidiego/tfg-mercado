"""HAR con calendario de eventos (HAR-E): ¿saber que mañana hay resultados mejora la predicción?

Los estudios de eventos mostraron que los días de resultados la acción oscila ~1,7 veces más de lo
previsto. Un HAR normal no puede saberlo: solo mira el pasado. Pero la fecha de resultados SE CONOCE
con antelación (las empresas la anuncian). HAR-E añade dos variables al HAR:

- `earn_next`: 1 si MAÑANA es una sesión de reacción a resultados.
- `earn_today`: 1 si HOY lo es (el pico de hoy lo causó el evento, no un cambio de régimen: conviene
  no extrapolarlo a mañana).

Supuesto honesto: se usa la fecha REAL del 8-K como si se hubiera conocido de antemano. Si una empresa
cambiara la fecha a última hora, el modelo no se enteraría.
"""

from __future__ import annotations

import polars as pl

from backend.events.table import Event, first_affected_session
from backend.models.volatility import forecast_vol_frame, har_forecasts

MIN_EVENTS = 8  # con menos eventos no hay datos para estimar el efecto


def forecast_har_e_frame(
    stock: pl.DataFrame,
    events: list[Event],
    symbol: str,
    category: str = "resultados",
    warmup: int = 500,
    refit_every: int = 21,
) -> tuple[pl.DataFrame, list[str]]:
    """Devuelve (tabla con las predicciones, nombres de los modelos).

    `har` es el HAR original; `har_enext` añade solo "mañana hay evento"; `har_e` añade también
    "hoy hubo evento". Las tres se reajustan con el mismo walk-forward.
    """
    # Primero se cuentan los eventos (barato): sin suficientes, no merece la pena calcular nada más.
    sessions = (
        stock.sort("open_time")["open_time"].dt.convert_time_zone("America/New_York").dt.date().to_list()
    )
    reaction = {
        first_affected_session(e.when, sessions)
        for e in events if e.symbol == symbol and e.category == category
    } - {None}
    if len(reaction) < MIN_EVENTS:
        raise ValueError(f"{symbol}: solo {len(reaction)} eventos '{category}' (hacen falta {MIN_EVENTS})")

    frame = forecast_vol_frame(stock, warmup=warmup, refit_every=refit_every)
    today = [1.0 if s in reaction else 0.0 for s in sessions]
    next_day = today[1:] + [0.0]          # "mañana" es la fila siguiente; el último día no tiene mañana
    base = list(zip(frame["rng"], frame["mean_5"], frame["mean_22"]))
    target = frame["target"].to_list()
    only_next = [(*b, n) for b, n in zip(base, next_day)]
    both = [(*b, n, t) for b, n, t in zip(base, next_day, today)]
    return frame.with_columns(
        pl.Series("earn_today", today, dtype=pl.Float64),
        pl.Series("earn_next", next_day, dtype=pl.Float64),
        pl.Series("har_enext", har_forecasts(only_next, target, warmup, refit_every), dtype=pl.Float64),
        pl.Series("har_e", har_forecasts(both, target, warmup, refit_every), dtype=pl.Float64),
    ), ["har", "har_enext", "har_e"]
