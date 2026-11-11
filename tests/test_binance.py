"""Tests del ingestor de Binance con datos sintéticos (no usan red)."""

import io
import zipfile
from datetime import datetime, timezone

import polars as pl

from backend.data import binance as b

T0_MS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC en milisegundos
MIN = 60_000


def _csv(open_times: list[int], unit_factor: int = 1, header: bool = False) -> bytes:
    """Construye un CSV de klines. `unit_factor`=1000 simula timestamps en microsegundos."""
    lines = []
    if header:
        lines.append(",".join(b.KLINE_COLUMNS))
    for i, t in enumerate(open_times):
        o = 100 + i
        row = [
            t * unit_factor, o, o + 2, o - 1, o + 1, 5.5,
            (t + MIN - 1) * unit_factor, 550.0, 42, 2.5, 250.0, 0,
        ]
        lines.append(",".join(str(x) for x in row))
    return ("\n".join(lines) + "\n").encode()


def _zip_bytes(csv: bytes, name: str = "BTCUSDT-1m-2024-01.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, csv)
    return buf.getvalue()


def test_month_range_cruza_de_anio():
    assert b.month_range("2023-11", "2024-02") == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_url_mensual():
    assert b.monthly_url("btcusdt", "1m", "2024-01") == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    )


def test_parseo_milisegundos():
    df = b.parse_klines_csv(_csv([T0_MS, T0_MS + MIN, T0_MS + 2 * MIN]))
    assert df.height == 3
    assert "ignore" not in df.columns
    assert df["open_time"][0] == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert df["open"].dtype == pl.Float64
    assert df["n_trades"].dtype == pl.Int64


def test_microsegundos_se_normalizan_igual_que_milisegundos():
    times = [T0_MS, T0_MS + MIN]
    en_ms = b.parse_klines_csv(_csv(times))
    en_us = b.parse_klines_csv(_csv(times, unit_factor=1000))
    assert en_ms["open_time"].to_list() == en_us["open_time"].to_list()
    assert en_ms["close_time"].to_list() == en_us["close_time"].to_list()


def test_csv_con_cabecera():
    df = b.parse_klines_csv(_csv([T0_MS, T0_MS + MIN], header=True))
    assert df.height == 2
    assert df.columns[0] == "open_time"


def test_read_zip(tmp_path):
    p = tmp_path / "x.zip"
    p.write_bytes(_zip_bytes(_csv([T0_MS, T0_MS + MIN])))
    assert b.read_zip(p).height == 2


def test_huecos():
    # Faltan las velas de los minutos 2 y 3.
    df = b.parse_klines_csv(_csv([T0_MS, T0_MS + MIN, T0_MS + 4 * MIN]))
    gaps = b.find_gaps(df, "1m")
    assert gaps.height == 1
    assert gaps["missing_candles"][0] == 2


def test_serie_continua_sin_huecos():
    df = b.parse_klines_csv(_csv([T0_MS + i * MIN for i in range(10)]))
    assert b.find_gaps(df, "1m").height == 0


def test_build_dataset_con_red_simulada(tmp_path, monkeypatch):
    """Flujo completo con un servidor falso: descarga, caché, 404 y Parquet final."""
    import httpx

    peticiones = []

    def handler(request: httpx.Request) -> httpx.Response:
        peticiones.append(request.url.path)
        if request.url.path.endswith("2024-01.zip"):
            return httpx.Response(200, content=_zip_bytes(_csv([T0_MS, T0_MS + MIN])))
        if request.url.path.endswith("2024-02.zip"):
            return httpx.Response(200, content=_zip_bytes(_csv([T0_MS + 31 * 1440 * MIN], unit_factor=1000)))
        return httpx.Response(404)  # mes aún no publicado

    real_client = httpx.Client
    monkeypatch.setattr(
        b.httpx, "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )

    out = b.build_dataset("BTCUSDT", "1m", "2024-01", "2024-03", data_dir=tmp_path)
    df = pl.read_parquet(out)
    assert df.height == 3
    assert df["open_time"].is_sorted()
    assert len(peticiones) == 3  # enero, febrero y marzo (404)

    b.build_dataset("BTCUSDT", "1m", "2024-01", "2024-03", data_dir=tmp_path)
    assert len(peticiones) == 4  # solo reintenta el 404: enero y febrero salen de la caché
