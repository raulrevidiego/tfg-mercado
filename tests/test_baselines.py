"""Tests de los baselines, las métricas y la evaluación."""

import math

import polars as pl
import pytest

from backend.models.baselines import forecast_frame
from backend.training import metrics
from backend.training.evaluate_baselines import evaluate


def _df(closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"open_time": list(range(len(closes))), "close": closes})


def test_forecast_frame_calcula_retornos_y_objetivo():
    f = forecast_frame(_df([100.0, 110.0, 121.0, 108.9]))
    r1, r2, r3 = math.log(1.1), math.log(1.1), math.log(0.9)
    assert f["ret"].to_list() == pytest.approx([r1, r2, r3])
    assert f["target"].to_list()[:2] == pytest.approx([r2, r3])   # el objetivo es el retorno de mañana
    assert f["target"][-1] is None                                # el último día no tiene mañana
    assert f["zero"].to_list() == [0.0, 0.0, 0.0]
    assert f["persistence"].to_list() == pytest.approx([r1, r2, r3])
    assert f["expanding_mean"].to_list() == pytest.approx([r1, (r1 + r2) / 2, (r1 + r2 + r3) / 3])


def test_sin_look_ahead_cambiar_el_futuro_no_cambia_el_pasado():
    base = [100.0, 101.0, 99.0, 102.0, 104.0, 103.0]
    alterada = base[:4] + [500.0, 1.0]  # solo cambian los dos últimos días
    a, b = forecast_frame(_df(base)), forecast_frame(_df(alterada))
    # Las predicciones de los días 0..2 (retornos 1..3) dependen solo de datos hasta ese día.
    for col in ("persistence", "expanding_mean"):
        assert a[col].to_list()[:3] == pytest.approx(b[col].to_list()[:3])


def test_metricas():
    assert metrics.mae([0, 0], [0.01, -0.03]) == pytest.approx(0.02)
    assert metrics.rmse([0, 0], [3.0, 4.0]) == pytest.approx(math.sqrt(12.5))
    acc, calls = metrics.directional_accuracy([1, -1, 0, 1], [1, 1, 1, 0])
    assert (acc, calls) == (0.5, 2)       # el 3.º no se moja (pred 0) y el 4.º no se movió
    assert metrics.directional_accuracy([0, 0], [1, -1]) == (None, 0)
    assert metrics.proportion_margin(0.5, 100) == pytest.approx(0.098, abs=1e-3)


def test_serie_alternante_persistence_es_el_doble_de_mala():
    # +1 %, -1 %, +1 %... aproximado: los retornos alternan de signo con igual tamaño.
    closes = [100.0]
    for i in range(200):
        closes.append(closes[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
    rows, up_rate, n = evaluate(forecast_frame(_df(closes)), warmup=20)
    by = {r["model"]: r for r in rows}
    assert by["zero"]["mae"] == pytest.approx(math.log(1.01), rel=1e-6)
    assert by["persistence"]["mae"] == pytest.approx(2 * math.log(1.01), rel=1e-6)
    assert by["persistence"]["acc"] == 0.0           # siempre acierta lo contrario
    assert by["persistence"]["vs_zero"] == pytest.approx(-100.0)
    assert by["zero"]["acc"] is None
    assert up_rate == pytest.approx(0.5, abs=0.02)


def test_warmup_demasiado_grande():
    with pytest.raises(ValueError):
        evaluate(forecast_frame(_df([100.0, 101.0, 102.0])), warmup=50)
