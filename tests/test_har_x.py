"""Tests de HAR-X (HAR + mercado, sector y volumen) y de su evaluación."""

import math
import random

import polars as pl
import pytest

from backend.models.har_x import forecast_har_x_frame
from backend.training import evaluate_har_x
from backend.training.evaluate_volatility import evaluate


def _ohlcv(times: list[int], ranges: list[float], volumes: list[float] | None = None) -> pl.DataFrame:
    """Serie con el rango diario indicado (high = low * exp(rango)) y volumen opcional."""
    vols = volumes or [1_000_000.0] * len(times)
    return pl.DataFrame({
        "open_time": times, "high": [100.0 * math.exp(r) for r in ranges],
        "low": [100.0] * len(times), "volume": vols,
    })


def _mercado_y_accion(n: int, seed: int = 3):
    """Mercado con memoria; el rango de la acción de mañana depende del mercado de hoy."""
    rnd = random.Random(seed)
    x, market = 0.0, []
    for _ in range(n):
        x = 0.9 * x + rnd.gauss(0, 0.3)
        market.append(0.012 * math.exp(x))
    stock = [0.01] + [0.9 * market[t] + abs(rnd.gauss(0, 0.001)) + 0.002 for t in range(n - 1)]
    times = list(range(n))
    return _ohlcv(times, stock, [1e6 * math.exp(rnd.gauss(0, 0.3)) for _ in times]), _ohlcv(times, market)


def _sector_parecido(market: pl.DataFrame, seed: int = 11) -> pl.DataFrame:
    """Serie sectorial correlacionada con el mercado pero con ruido propio."""
    rnd = random.Random(seed)
    rngs = [0.7 * math.log(h / l) + abs(rnd.gauss(0, 0.003)) + 0.002
            for h, l in zip(market["high"].to_list(), market["low"].to_list())]
    return _ohlcv(market["open_time"].to_list(), rngs)


def test_alinea_por_fecha_antes_de_calcular_nada():
    times = list(range(40))
    stock = _ohlcv(times, [0.02] * 40)
    market = _ohlcv(times[:20] + times[21:], [0.01] * 39)   # al mercado le falta el día 20
    frame, _ = forecast_har_x_frame(stock, market, warmup=1000)
    assert frame.height == 39 and 20 not in frame["open_time"].to_list()
    # "mañana" del día 19 es el siguiente día común (el 21), no el día 20 que falta.
    assert frame["target"].to_list()[19] == pytest.approx(0.02)


def test_columnas_y_variantes_con_y_sin_sector():
    times = list(range(60))
    stock, market, sector = _ohlcv(times, [0.02] * 60), _ohlcv(times, [0.01] * 60), _ohlcv(times, [0.015] * 60)
    f1, m1 = forecast_har_x_frame(stock, market, sector, warmup=1000)
    assert m1 == ["har", "har_mkt", "har_vol", "har_sector", "har_all"]
    assert f1["sec_rng"][0] == pytest.approx(0.015) and f1["mkt_rng"][0] == pytest.approx(0.01)
    f2, m2 = forecast_har_x_frame(stock, market, None, warmup=1000)
    assert m2 == ["har", "har_mkt", "har_vol", "har_all"] and "sec_rng" not in f2.columns


def test_volumen_relativo_y_volumen_cero():
    times = list(range(30))
    vols = [1000.0] * 29 + [2000.0]
    frame, _ = forecast_har_x_frame(_ohlcv(times, [0.02] * 30, vols), _ohlcv(times, [0.01] * 30), warmup=1000)
    # El día 29: volumen 2000 frente a la media de 22 días (21 de 1000 y 1 de 2000).
    assert frame["relvol"][29] == pytest.approx(math.log(2000 / ((21 * 1000 + 2000) / 22)))
    sin_volumen = _ohlcv(times, [0.02] * 30, [0.0] * 30)
    frame0, _ = forecast_har_x_frame(sin_volumen, _ohlcv(times, [0.01] * 30), warmup=1000)
    assert all(v is None for v in frame0["relvol"].to_list())   # sin volumen no se inventa nada


def test_sin_look_ahead_cambiar_el_futuro_no_cambia_el_pasado():
    stock, market = _mercado_y_accion(800)
    corte = 700
    # Cambiamos SOLO el futuro (desde el día 700) de las tres series.
    futuro = list(range(corte, 800))
    sector = _sector_parecido(market)
    stock2 = pl.concat([stock.slice(0, corte), _ohlcv(futuro, [0.3] * 100, [9e9] * 100)])
    market2 = pl.concat([market.slice(0, corte), _ohlcv(futuro, [0.4] * 100)])
    sector2 = pl.concat([sector.slice(0, corte), _ohlcv(futuro, [0.5] * 100)])
    a, names = forecast_har_x_frame(stock, market, sector, warmup=300)
    b, _ = forecast_har_x_frame(stock2, market2, sector2, warmup=300)
    for col in names:
        for dia, (x, y) in enumerate(zip(a[col].to_list()[:corte], b[col].to_list()[:corte])):
            assert (x is None and y is None) or x == pytest.approx(y), f"{col} usa el futuro (día {dia})"


def test_si_el_mercado_informa_har_mkt_gana_a_har():
    stock, market = _mercado_y_accion(1500)
    frame, names = forecast_har_x_frame(stock, market, warmup=500)
    rows, _, n = evaluate(frame, 500, models=[*names, "ewma_094"], reference="har")
    by = {r["model"]: r for r in rows}
    assert by["har"]["abs_gain"] == pytest.approx(0.0, abs=1e-9)
    assert by["har_mkt"]["abs_gain"] > 10 and by["har_mkt"]["abs_gain"] > by["har_mkt"]["abs_margin"]


def test_cli_de_extremo_a_extremo(tmp_path, capsys):
    stock, market = _mercado_y_accion(900)
    out = tmp_path / "parquet" / "alpaca_stocks"
    out.mkdir(parents=True)
    # El ingestor guarda fechas con zona horaria; aquí basta con enteros ordenados.
    for sym, df in {"NVDA": stock, "SPY": market, "XLK": _sector_parecido(market)}.items():
        df.with_columns(close=pl.col("high"), open=pl.col("low")).write_parquet(out / f"{sym}_1d.parquet")
    evaluate_har_x.main(["--symbols", "NVDA", "--warmup", "400", "--data-dir", str(tmp_path)])
    salida = capsys.readouterr().out
    assert "NVDA [mercado: SPY, sector: XLK]" in salida
    for modelo in ("har_mkt", "har_vol", "har_sector", "har_all", "ewma_094", "(referencia)"):
        assert modelo in salida


def test_cli_avisa_si_falta_el_etf_sectorial(tmp_path, capsys):
    stock, market = _mercado_y_accion(900)
    out = tmp_path / "parquet" / "alpaca_stocks"
    out.mkdir(parents=True)
    for sym, df in {"GOOGL": stock, "SPY": market}.items():   # falta XLC
        df.write_parquet(out / f"{sym}_1d.parquet")
    evaluate_har_x.main(["--symbols", "GOOGL", "--warmup", "400", "--data-dir", str(tmp_path)])
    salida = capsys.readouterr().out
    assert "no hay ETF sectorial" in salida and "XLC" in salida
    assert "\n  har_sector" not in salida        # no hay fila de har_sector en la tabla
    assert "\n  har_all" in salida


def test_cli_sin_spy_da_un_error_claro(tmp_path):
    with pytest.raises(SystemExit, match="SPY"):
        evaluate_har_x.main(["--symbols", "NVDA", "--data-dir", str(tmp_path)])
