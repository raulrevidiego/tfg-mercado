"""Modelos de referencia (baselines): los más simples posibles.

Cualquier modelo más complejo (redes neuronales, etc.) tiene que superarlos para justificar
su existencia.

Convención: en cada día t, cada baseline predice el retorno logarítmico del día t+1 usando
SOLO información hasta el día t. Usar datos del futuro (*look-ahead*) invalidaría el resultado.

Retorno logarítmico del día t = ln(cierre_t / cierre_{t-1}). Se usa porque los retornos
se suman en el tiempo y se parecen al % diario cuando es pequeño (un +1 % ~ 0,00995).
"""

from __future__ import annotations

import polars as pl

BASELINES = {
    "zero": "Mañana no cambia nada (paseo aleatorio)",
    "persistence": "Mañana repite lo de hoy",
    "expanding_mean": "Mañana rinde la media histórica hasta hoy",
}


def forecast_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Tabla con el retorno de cada día (`ret`), el valor a acertar (`target`, el retorno del
    día siguiente) y la predicción de cada baseline hecha ese día."""
    return (
        df.sort("open_time")
        .select("open_time", "close")
        .with_columns(ret=(pl.col("close") / pl.col("close").shift(1)).log())
        .drop_nulls("ret")  # el primer día no tiene día anterior
        .with_columns(
            target=pl.col("ret").shift(-1),  # el último día no tiene "siguiente": queda nulo
            zero=pl.lit(0.0),
            persistence=pl.col("ret"),
            expanding_mean=pl.col("ret").cum_sum() / pl.int_range(1, pl.len() + 1),
        )
    )
