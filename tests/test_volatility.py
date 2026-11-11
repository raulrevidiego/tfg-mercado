"""Tests de OLS, baselines de volatilidad, QLIKE y evaluación."""

import math
import random

import polars as pl
import pytest

from backend.models import ols
from backend.models.volatility import VOL_BASELINES, forecast_vol_frame
from backend.training import metrics
from backend.training.evaluate_volatility import REFERENCE, evaluate


def _ohlc(ranges: list[float]) -> pl.DataFrame:
    """Serie con el rango diario indicado: high = low * exp(rango)."""
    low = 100.0
    return pl.DataFrame({
        "open_time": list(range(len(ranges))),
        "high": [low * math.exp(r) for r in ranges],
        "low": [low] * len(ranges),
    })


def _persistente(n: int, seed: int = 7) -> list[float]:
    """Rangos con memoria (clustering): el logaritmo del rango sigue un AR(1) con ruido."""
    rnd = random.Random(seed)
    x, out = 0.0, []
    for _ in range(n):
        x = 0.95 * x + rnd.gauss(0, 0.25)
        out.append(0.015 * math.exp(x))
    return out


# --------------------------------------------------------------------------- OLS

def test_ols_recupera_los_coeficientes_exactos():
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0], [5.0, 8.0]]
    y = [1 + 2 * a + 3 * b for a, b in X]
    coef = ols.fit(X, y)
    assert coef == pytest.approx([1.0, 2.0, 3.0])
    assert ols.predict(coef, [10.0, 10.0]) == pytest.approx(51.0)


def test_ols_detecta_regresores_colineales():
    X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]  # la 2.ª columna es el doble de la 1.ª
    with pytest.raises(ValueError):
        ols.fit(X, [1.0, 2.0, 3.0, 4.0])


# ------------------------------------------------------------------- construcción

def test_columnas_de_los_baselines_se_calculan_bien():
    r = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    f = forecast_vol_frame(_ohlc(r), warmup=1000)
    assert f["rng"].to_list() == pytest.approx(r)
    assert f["target"].to_list()[:6] == pytest.approx(r[1:])          # el objetivo es el rango de mañana
    assert f["target"][-1] is None
    assert f["last_value"].to_list() == pytest.approx(r)
    assert f["expanding_mean"][3] == pytest.approx(sum(r[:4]) / 4)
    assert f["mean_5"][4] == pytest.approx(sum(r[:5]) / 5)
    assert f["mean_5"][3] is None                                      # aún no hay 5 días
    ewma = [r[0]]
    for v in r[1:]:
        ewma.append(0.06 * v + 0.94 * ewma[-1])
    assert f["ewma_094"].to_list() == pytest.approx(ewma)
    assert all(v is None for v in f["har"].to_list())                  # warmup enorme: HAR no predice


def test_sin_look_ahead_cambiar_el_futuro_no_cambia_el_pasado():
    base = _persistente(800)
    corte = 700
    alterada = base[:corte] + [0.5] * (800 - corte)  # solo cambia el futuro desde el día 700
    a = forecast_vol_frame(_ohlc(base), warmup=300)
    b = forecast_vol_frame(_ohlc(alterada), warmup=300)
    for col in VOL_BASELINES:
        for dia, (x, y) in enumerate(zip(a[col].to_list()[:corte], b[col].to_list()[:corte])):
            assert (x is None and y is None) or x == pytest.approx(y), f"{col} usa datos del futuro (día {dia})"


def test_har_solo_predice_desde_el_warmup():
    f = forecast_vol_frame(_ohlc(_persistente(700)), warmup=400)
    har = f["har"].to_list()
    assert all(v is None for v in har[:400])
    assert all(v is not None and v >= 1e-4 for v in har[400:])


# -------------------------------------------------------------------------- métricas

def test_qlike_cero_si_acierta_y_castiga_mas_quedarse_corto():
    assert metrics.qlike_losses([0.02], [0.02]) == pytest.approx([0.0])
    corto = metrics.qlike_losses([0.01], [0.02])[0]   # prever la mitad
    largo = metrics.qlike_losses([0.04], [0.02])[0]   # prever el doble
    assert corto > largo > 0


def test_margen_newey_west():
    assert metrics.mean_diff_margin([0.5] * 100) == pytest.approx(0.0, abs=1e-12)
    rnd = random.Random(1)
    ruido = [rnd.gauss(0, 1) for _ in range(2000)]
    m = metrics.mean_diff_margin(ruido)
    assert m == pytest.approx(1.96 / math.sqrt(2000), rel=0.25)       # ~ el margen ingenuo si no hay memoria
    # con memoria (valores repetidos en bloques) el margen debe ser mayor que el ingenuo
    bloques = [v for v in ruido[:400] for _ in range(5)]
    ingenuo = 1.96 * (sum((x - sum(bloques) / 2000) ** 2 for x in bloques) / 2000) ** 0.5 / math.sqrt(2000)
    assert metrics.mean_diff_margin(bloques, lags=10) > ingenuo


# ----------------------------------------------------------------------- evaluación

def test_con_volatilidad_persistente_los_modelos_con_memoria_ganan_a_la_media():
    frame = forecast_vol_frame(_ohlc(_persistente(1500)), warmup=500)
    rows, mean_range, n = evaluate(frame, warmup=500)
    by = {r["model"]: r for r in rows}
    assert n == 999 and mean_range > 0
    assert by[REFERENCE]["abs_gain"] == pytest.approx(0.0, abs=1e-9)
    for name in ("mean_20", "ewma_094", "har"):
        assert by[name]["abs_gain"] > 5 and by[name]["qlike_gain"] > 5, name


def test_warmup_demasiado_grande():
    frame = forecast_vol_frame(_ohlc(_persistente(300)), warmup=100)
    with pytest.raises(ValueError):
        evaluate(frame, warmup=5000)
