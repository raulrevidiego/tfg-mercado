"""Métricas para evaluar predicciones de retornos. Python puro, para que se lean fácil."""

from __future__ import annotations

import math


def mae(pred: list[float], actual: list[float]) -> float:
    """Error absoluto medio: cuánto se equivoca la predicción de media, en las mismas unidades."""
    return sum(abs(p - a) for p, a in zip(pred, actual)) / len(pred)


def rmse(pred: list[float], actual: list[float]) -> float:
    """Raíz del error cuadrático medio: como el MAE pero castiga más los errores grandes."""
    return math.sqrt(sum((p - a) ** 2 for p, a in zip(pred, actual)) / len(pred))


def directional_accuracy(pred: list[float], actual: list[float]) -> tuple[float | None, int]:
    """Fracción de días en que la predicción acierta si sube o baja.

    Solo cuentan los días en que el modelo se moja (predicción distinta de 0) y en que el
    precio se movió de verdad. Devuelve (acierto, días contados); acierto es None si no hay días.
    """
    hits = calls = 0
    for p, a in zip(pred, actual):
        if p == 0 or a == 0:
            continue
        calls += 1
        hits += (p > 0) == (a > 0)
    return (hits / calls if calls else None), calls


def proportion_margin(p: float, n: int, z: float = 1.96) -> float:
    """Margen de error (±) de una proporción con ~95 % de confianza: z * sqrt(p(1-p)/n).

    Con pocos días, un 53 % de aciertos no se distingue de tirar una moneda al aire.
    """
    return z * math.sqrt(p * (1 - p) / n)


def qlike_losses(forecast: list[float], actual: list[float]) -> list[float]:
    """Pérdida QLIKE día a día para predicciones de volatilidad (menor es mejor).

    Con y = actual² y f = forecast²: y/f - ln(y/f) - 1. Vale 0 si acierta, castiga más
    quedarse corto (subestimar el riesgo) que pasarse, y es poco sensible al ruido del día.
    """
    out = []
    for f, a in zip(forecast, actual):
        ratio = (a * a) / (f * f)
        out.append(ratio - math.log(ratio) - 1)
    return out


def mean_diff_margin(diffs: list[float], lags: int = 5, z: float = 1.96) -> float:
    """Margen de error (±, ~95 %) de la media de una serie de diferencias de pérdida.

    Las pérdidas de días seguidos se parecen entre sí (la volatilidad tiene memoria), así que
    el margen "ingenuo" sería demasiado optimista. Se usa el estimador de Newey-West: suma
    también las autocorrelaciones hasta `lags` días, con pesos decrecientes.
    """
    n = len(diffs)
    mean = sum(diffs) / n
    x = [d - mean for d in diffs]
    var = sum(v * v for v in x) / n
    for k in range(1, lags + 1):
        cov = sum(x[t] * x[t - k] for t in range(k, n)) / n
        var += 2 * (1 - k / (lags + 1)) * cov
    return z * math.sqrt(max(var, 0.0) / n)
