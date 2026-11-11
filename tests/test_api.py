"""Tests de la API con un Parquet sintético (no usan red)."""

import polars as pl
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.data import binance as b

T0_MS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC
T0_S = T0_MS // 1000
MIN = 60_000


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Cliente de pruebas con 120 velas de 1 minuto (2 horas) guardadas en un Parquet temporal."""
    lines = []
    for i in range(120):
        o = 100 + i
        lines.append(
            f"{T0_MS + i * MIN},{o},{o + 2},{o - 1},{o + 1},5.5,"
            f"{T0_MS + i * MIN + MIN - 1},550.0,42,2.5,250.0,0"
        )
    df = b.parse_klines_csv(("\n".join(lines) + "\n").encode())
    out = tmp_path / "parquet" / "binance_spot"
    out.mkdir(parents=True)
    df.write_parquet(out / "BTCUSDT_1m.parquet")
    monkeypatch.setenv("TFG_DATA_DIR", str(tmp_path))
    return TestClient(app)


def test_datasets(client):
    assert client.get("/api/datasets").json() == [
        {"source": "binance_spot", "symbol": "BTCUSDT", "interval": "1m"}
    ]


def test_velas_sin_agrupar(client):
    data = client.get("/api/candles", params={"tf": "1m", "limit": 5}).json()
    assert len(data) == 5
    assert data[-1]["time"] == T0_S + 119 * 60
    assert data[-1]["open"] == 219.0


def test_agrupar_en_horas_calcula_bien_el_ohlc(client):
    data = client.get("/api/candles", params={"tf": "1h"}).json()
    assert len(data) == 2
    primera, segunda = data
    assert primera["time"] == T0_S
    assert primera["open"] == 100.0       # open de la primera vela de la hora
    assert primera["high"] == 161.0       # máximo de los highs (102 + 59)
    assert primera["low"] == 99.0         # mínimo de los lows (100 - 1)
    assert primera["close"] == 160.0      # close de la última vela (159 + 1)
    assert primera["volume"] == pytest.approx(330.0)  # 60 velas * 5.5
    assert segunda["open"] == 160.0 and segunda["close"] == 220.0


def test_limit_devuelve_las_ultimas(client):
    data = client.get("/api/candles", params={"tf": "1h", "limit": 1}).json()
    assert [d["time"] for d in data] == [T0_S + 3600]


def test_errores_de_validacion(client):
    assert client.get("/api/candles", params={"symbol": "../x"}).status_code == 400
    assert client.get("/api/candles", params={"tf": "7m"}).status_code == 400
    assert client.get("/api/candles", params={"interval": "1h", "tf": "5m"}).status_code == 400
    assert client.get("/api/candles", params={"symbol": "ETHUSDT"}).status_code == 404


def test_la_interfaz_se_sirve_en_la_raiz(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "TFG Mercado" in r.text


def test_otra_fuente_acciones_diarias(client, tmp_path):
    """Una serie diaria de otra fuente aparece en el listado y se puede consultar."""
    dia = 86_400_000
    df = pl.DataFrame({
        "open_time": [T0_MS, T0_MS + dia, T0_MS + 2 * dia],
        "open": [10.0, 11.0, 12.0], "high": [11.0, 12.0, 13.0],
        "low": [9.0, 10.0, 11.0], "close": [11.0, 12.0, 13.0], "volume": [1.0, 2.0, 3.0],
    }).with_columns(pl.from_epoch("open_time", time_unit="ms").dt.replace_time_zone("UTC"))
    out = tmp_path / "parquet" / "alpaca_stocks"
    out.mkdir(parents=True)
    df.write_parquet(out / "NVDA_1d.parquet")

    sets = client.get("/api/datasets").json()
    assert {"source": "alpaca_stocks", "symbol": "NVDA", "interval": "1d"} in sets
    data = client.get(
        "/api/candles", params={"source": "alpaca_stocks", "symbol": "NVDA", "interval": "1d", "tf": "1d"}
    ).json()
    assert [d["close"] for d in data] == [11.0, 12.0, 13.0]


def test_fuente_invalida(client):
    assert client.get("/api/candles", params={"source": "../etc"}).status_code == 400


# ---------------------------------------------------------------- series diarias ---

DIA = 86_400_000


def _guardar_diario(tmp_path, symbol: str, closes: list[float], start_ms: int = T0_MS) -> None:
    n = len(closes)
    df = pl.DataFrame({
        "open_time": [start_ms + i * DIA for i in range(n)],
        "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes, "volume": [10.0] * n,
    }).with_columns(pl.from_epoch("open_time", time_unit="ms").dt.replace_time_zone("UTC"))
    out = tmp_path / "parquet" / "alpaca_stocks"
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / f"{symbol}_1d.parquet")


def test_agrupar_dias_en_semanas(client, tmp_path):
    # 2024-01-01 es lunes: 14 días seguidos = 2 semanas completas.
    _guardar_diario(tmp_path, "NVDA", [float(100 + i) for i in range(14)])
    data = client.get(
        "/api/candles",
        params={"source": "alpaca_stocks", "symbol": "NVDA", "interval": "1d", "tf": "1w"},
    ).json()
    assert len(data) == 2
    assert data[0]["time"] == T0_S
    assert data[0]["open"] == 100.0 and data[0]["close"] == 106.0   # lunes -> domingo
    assert data[0]["high"] == 107.0 and data[0]["low"] == 99.0
    assert data[0]["volume"] == pytest.approx(70.0)                  # 7 días * 10


def test_agrupar_dias_en_meses(client, tmp_path):
    _guardar_diario(tmp_path, "NVDA", [100.0] * 60)  # 1 ene - 29 feb 2024 (bisiesto)
    data = client.get(
        "/api/candles",
        params={"source": "alpaca_stocks", "symbol": "NVDA", "interval": "1d", "tf": "1M"},
    ).json()
    assert [d["time"] for d in data] == [T0_S, T0_S + 31 * 86400]    # 1 ene y 1 feb
    assert data[0]["volume"] == pytest.approx(310.0)                 # 31 días * 10
    assert data[1]["volume"] == pytest.approx(290.0)                 # 29 días * 10


def test_no_se_puede_ver_mas_fino_que_los_datos(client, tmp_path):
    _guardar_diario(tmp_path, "NVDA", [100.0] * 5)
    r = client.get(
        "/api/candles",
        params={"source": "alpaca_stocks", "symbol": "NVDA", "interval": "1d", "tf": "1h"},
    )
    assert r.status_code == 400


def test_watchlist(client, tmp_path):
    closes = [float(i) for i in range(1, 31)]  # 1, 2, ..., 30
    _guardar_diario(tmp_path, "NVDA", closes)
    _guardar_diario(tmp_path, "XOM", [50.0])   # una sola vela: se ignora
    rows = client.get("/api/watchlist").json()
    assert [r["symbol"] for r in rows] == ["NVDA"]
    r = rows[0]
    assert r["close"] == 30.0
    assert r["change_1d"] == pytest.approx((30 / 29 - 1) * 100)
    assert r["change_5d"] == pytest.approx((30 / 25 - 1) * 100)
    assert r["change_1m"] == pytest.approx((30 / 9 - 1) * 100)       # 21 velas atrás
    assert r["date"] == "2024-01-30"


def test_watchlist_fuente_invalida(client):
    assert client.get("/api/watchlist", params={"source": "../x"}).status_code == 400
