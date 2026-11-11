"""Estudio de eventos: ¿se movió la acción más de lo normal tras un comentario?

Idea (versión diaria):
1. Con los ~120 días ANTERIORES al evento (dejando 10 de margen para no contaminarlos) se ajusta
   el *modelo de mercado*: retorno de la acción ≈ alfa + beta · retorno de SPY.
2. Ese modelo dice qué retorno era de esperar cada día. La diferencia con el retorno real es el
   *retorno anormal* (AR): lo que el mercado general no explica.
3. Se suman los AR del día del evento y los siguientes: *retorno anormal acumulado* (CAR).
4. Para saber si un CAR es raro o normal se compara con el de muchas fechas al azar (*placebo*).
5. Aparte, el *rango anormal* compara cuánto oscilaron máximo y mínimo ese día con lo que predecía
   la EWMA del día anterior (1,0 = lo esperado; 2,0 = el doble).
"""

from __future__ import annotations

import math
import random

import polars as pl

from backend.models import ols
from backend.models.volatility import EWMA_LAMBDA


def prepare(stock: pl.DataFrame, market: pl.DataFrame) -> dict[str, list]:
    """Alinea acción y mercado por fecha y devuelve listas listas para analizar.

    Claves: session (fecha de la sesión en hora de Nueva York), ret, mkt_ret (retornos
    logarítmicos), rng (rango diario) y ewma_prev (predicción EWMA del rango hecha el día anterior).
    """
    frame = (
        stock.select("open_time", "close", "high", "low")
        .join(market.select("open_time", mclose=pl.col("close")), on="open_time", how="inner")
        .sort("open_time")
        .with_columns(
            session=pl.col("open_time").dt.convert_time_zone("America/New_York").dt.date(),
            ret=(pl.col("close") / pl.col("close").shift(1)).log(),
            mkt_ret=(pl.col("mclose") / pl.col("mclose").shift(1)).log(),
            rng=(pl.col("high") / pl.col("low")).log(),
        )
        .with_columns(ewma=pl.col("rng").ewm_mean(alpha=1 - EWMA_LAMBDA, adjust=False))
        .with_columns(ewma_prev=pl.col("ewma").shift(1))
        .slice(1)  # el primer día no tiene retorno
    )
    return frame.select("session", "ret", "mkt_ret", "rng", "ewma_prev").to_dict(as_series=False)


def analyze_event(
    data: dict[str, list], idx: int, est_len: int = 120, gap: int = 10, min_est: int = 60
) -> dict | None:
    """Analiza el evento ocurrido en la sesión número `idx`. None si no hay datos suficientes
    (poca historia antes o menos de 4 sesiones después)."""
    n = len(data["ret"])
    end = idx - gap
    start = max(0, end - est_len)
    if end - start < min_est or idx + 4 >= n:
        return None

    x = [[m] for m in data["mkt_ret"][start:end]]
    y = data["ret"][start:end]
    alpha, beta = ols.fit(x, y)
    resid = [yy - (alpha + beta * xx[0]) for xx, yy in zip(x, y)]
    sigma = math.sqrt(sum(r * r for r in resid) / (len(resid) - 2))

    ar = [data["ret"][idx + k] - (alpha + beta * data["mkt_ret"][idx + k]) for k in range(5)]
    car1 = sum(ar[:2])
    return {
        "alpha": alpha, "beta": beta, "sigma": sigma,
        "car0": ar[0], "car1": car1, "car4": sum(ar),
        "z1": car1 / (sigma * math.sqrt(2)) if sigma > 0 else None,
        "abn_range": data["rng"][idx] / data["ewma_prev"][idx],
    }


def placebo_pool(
    data: dict[str, list], n: int, exclude_idx: set[int] | list[int], seed: int = 7, margin: int = 10,
    est_len: int = 120, gap: int = 10, min_est: int = 60,
) -> list[dict]:
    """Analiza `n` fechas al azar (sin eventos reales cerca) como si hubiera pasado algo.

    Muestra cómo se comporta un CAR cuando NO ha pasado nada: la distribución contra la que
    se compara el CAR de un evento real.
    """
    banned: set[int] = set()
    for e in exclude_idx:
        banned.update(range(e - margin, e + margin + 1))
    candidates = [i for i in range(gap + min_est, len(data["ret"]) - 4) if i not in banned]
    rnd = random.Random(seed)
    picks = rnd.sample(candidates, min(n, len(candidates)))
    out = []
    for i in picks:
        r = analyze_event(data, i, est_len, gap, min_est)
        if r is not None:
            out.append({**r, "idx": i})
    return out


def permutation_pvalue(
    observed_mean: float, symbols: list[str], pools: dict[str, list[float]], draws: int = 5000, seed: int = 7
) -> float:
    """¿Con qué frecuencia un conjunto de fechas AL AZAR da una media de CAR tan extrema?

    Cada evento real se sustituye por un CAR placebo de la MISMA acción (para respetar que unas
    acciones son más volátiles que otras). p pequeño = difícil de explicar por casualidad.
    """
    rnd = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        mean = sum(rnd.choice(pools[s]) for s in symbols) / len(symbols)
        extreme += abs(mean) >= abs(observed_mean)
    return (extreme + 1) / (draws + 1)


def magnitude_pvalue(
    observed_mean_abs: float, symbols: list[str], pools: dict[str, list[float]], draws: int = 5000, seed: int = 7
) -> float:
    """Como `permutation_pvalue`, pero sobre la MAGNITUD de la reacción (|CAR|), sin importar el signo.

    Los eventos buenos y malos se compensan en la media con signo (+10 % y -10 % dan 0 %), pero
    la reacción fue enorme. Aquí se pregunta: ¿se mueve la acción, hacia donde sea, más de lo que
    se movería en fechas al azar? Es un contraste de una cola (solo importa que sea mayor).
    """
    rnd = random.Random(seed)
    at_least = 0
    for _ in range(draws):
        mean = sum(abs(rnd.choice(pools[s])) for s in symbols) / len(symbols)
        at_least += mean >= observed_mean_abs
    return (at_least + 1) / (draws + 1)


def select_independent_events(
    events: list[dict], min_gap: int = 5, priority: list[str] | tuple[str, ...] = ()
) -> tuple[list[dict], int, int]:
    """Deja solo eventos independientes entre sí (cada `event` lleva symbol, idx y category).

    Dos problemas inflarían la significancia si se ignoraran:
    1. Varios avisos el mismo día en la misma acción comparten exactamente la misma reacción:
       se fusionan en uno solo, con la categoría de mayor prioridad.
    2. Eventos separados por menos de `min_gap` sesiones comparten días de reacción: de cada grupo
       solapado se conserva el de mayor prioridad (y, a igualdad, el más antiguo).
    Devuelve (eventos conservados, fusionados por el mismo día, descartados por solapamiento).
    """
    rank = {c: i for i, c in enumerate(priority)}

    def key(e: dict) -> tuple[int, int]:
        return rank.get(e["category"], len(priority)), e["idx"]

    by_day: dict[tuple[str, int], dict] = {}
    for e in sorted(events, key=key):
        by_day.setdefault((e["symbol"], e["idx"]), e)  # el primero (mayor prioridad) se queda
    kept: list[dict] = []
    taken: dict[str, list[int]] = {}
    for e in sorted(by_day.values(), key=key):
        mine = taken.setdefault(e["symbol"], [])
        if all(abs(e["idx"] - i) >= min_gap for i in mine):
            mine.append(e["idx"])
            kept.append(e)
    kept.sort(key=lambda e: (e["idx"], e["symbol"]))
    return kept, len(events) - len(by_day), len(by_day) - len(kept)


def summarize(results: list[dict], pools: dict[str, list[float]], draws: int = 5000, seed: int = 7) -> list[dict]:
    """Resumen por categoría de los eventos analizados (cada `result` lleva symbol y category)."""
    rows = []
    for cat in sorted({r["category"] for r in results}):
        group = [r for r in results if r["category"] == cat]
        cars = [r["car1"] for r in group]
        n = len(cars)
        mean = sum(cars) / n
        sd = math.sqrt(sum((c - mean) ** 2 for c in cars) / (n - 1)) if n > 1 else None
        symbols = [r["symbol"] for r in group]
        mean_abs = sum(abs(c) for c in cars) / n
        expected_abs = sum(sum(abs(v) for v in pools[sym]) / len(pools[sym]) for sym in symbols) / n
        rows.append({
            "category": cat, "n": n, "mean_car1": mean,
            "mean_abs_car1": mean_abs, "expected_abs": expected_abs,
            "p_abs": magnitude_pvalue(mean_abs, symbols, pools, draws, seed),
            "margin": 1.96 * sd / math.sqrt(n) if sd is not None else None,
            "share_pos": sum(c > 0 for c in cars) / n,
            "mean_abn_range": sum(r["abn_range"] for r in group) / n,
            "p": permutation_pvalue(mean, [r["symbol"] for r in group], pools, draws, seed),
        })
    return rows
