"""Tests del detector de velas sospechosas."""

import polars as pl

from backend.data.quality import find_suspicious_candles


def _serie(n: int = 10, base: float = 100.0) -> pl.DataFrame:
    return pl.DataFrame({
        "open_time": list(range(n)),
        "open": [base] * n, "high": [base + 1] * n,
        "low": [base - 1] * n, "close": [base] * n,
    })


def test_serie_normal_no_tiene_sospechosas():
    assert find_suspicious_candles(_serie()).height == 0


def test_detecta_un_pico_de_maximo_aislado():
    df = _serie().with_columns(
        high=pl.when(pl.col("open_time") == 4).then(190.0).otherwise(pl.col("high"))
    )
    sus = find_suspicious_candles(df)
    assert sus["open_time"].to_list() == [4]
    assert sus["up_spike"][0] > 0.8


def test_detecta_un_pico_de_minimo_aislado():
    df = _serie().with_columns(
        low=pl.when(pl.col("open_time") == 6).then(40.0).otherwise(pl.col("low"))
    )
    assert find_suspicious_candles(df)["open_time"].to_list() == [6]


def test_un_salto_real_con_cierres_vecinos_altos_no_se_marca():
    # Sube un 30 % y se queda arriba: el máximo coincide con los cierres siguientes.
    closes = [100.0] * 5 + [130.0] * 5
    df = pl.DataFrame({
        "open_time": list(range(10)), "open": closes,
        "high": [c + 1 for c in closes], "low": [c - 1 for c in closes], "close": closes,
    })
    assert find_suspicious_candles(df).height == 0


# ------------------------------------------------------------------ reparación ---

from backend.data.quality import log_repairs, repair_spikes  # noqa: E402


def _con_picos() -> pl.DataFrame:
    return _serie().with_columns(
        high=pl.when(pl.col("open_time") == 4).then(190.0).otherwise(pl.col("high")),
        low=pl.when(pl.col("open_time") == 6).then(40.0).otherwise(pl.col("low")),
    )


def test_repair_recorta_al_rango_apertura_cierre():
    fixed, changes = repair_spikes(_con_picos())
    assert fixed.filter(pl.col("open_time") == 4)["high"][0] == 100.0   # max(open, close)
    assert fixed.filter(pl.col("open_time") == 6)["low"][0] == 100.0    # min(open, close)
    assert find_suspicious_candles(fixed).height == 0
    assert sorted(changes["field"].to_list()) == ["high", "low"]
    assert changes.filter(pl.col("field") == "high")["old"][0] == 190.0


def test_repair_no_toca_una_serie_sana():
    df = _serie()
    fixed, changes = repair_spikes(df)
    assert fixed.equals(df) and changes.height == 0


def test_log_repairs_es_idempotente(tmp_path):
    from datetime import datetime, timezone
    changes = pl.DataFrame({
        "open_time": [datetime(2024, 6, 10, 4, tzinfo=timezone.utc)],
        "field": ["high"], "old": [195.39], "new": [121.44],
    })
    log_repairs(tmp_path, "alpaca_stocks", "NVDA", "1d", changes)
    path = log_repairs(tmp_path, "alpaca_stocks", "NVDA", "1d", changes)  # repetir no duplica
    log = pl.read_csv(path)
    assert log.height == 1
    assert log["open_time"][0] == "2024-06-10 04:00:00" and log["symbol"][0] == "NVDA"
