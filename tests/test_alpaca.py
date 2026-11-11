"""Tests del ingestor de Alpaca con un servidor simulado (no usan red ni claves reales)."""

import httpx
import polars as pl
import pytest

from backend import config
from backend.data import alpaca as a


def _bar(t: str, o: float) -> dict:
    return {"t": t, "o": o, "h": o + 2, "l": o - 1, "c": o + 1, "v": 1000, "n": 50, "vw": o + 0.5}


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "clave-de-prueba")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secreto-de-prueba")


REAL_CLIENT = httpx.Client  # guardado al importar, antes de que ningún test lo sustituya


def _simular_alpaca(monkeypatch, handler):
    monkeypatch.setattr(
        a.httpx, "Client", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw)
    )


def test_bars_to_frame():
    df = a.bars_to_frame([_bar("2024-01-02T05:00:00Z", 100.0), _bar("2024-01-03T05:00:00Z", 101.0)])
    assert df.columns == ["open_time", "open", "high", "low", "close", "volume", "n_trades", "vwap"]
    assert df["open_time"].dtype.time_zone == "UTC"
    assert df["open_time"][0].isoformat() == "2024-01-02T05:00:00+00:00"
    assert df["close"].to_list() == [101.0, 102.0]


def test_normalize_date():
    assert a.normalize_date("2024-01-01") == "2024-01-01T00:00:00Z"
    assert a.normalize_date("2024-01-01T10:00:00Z") == "2024-01-01T10:00:00Z"


def test_faltan_claves(monkeypatch, tmp_path):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(a.MissingKeysError):
        a.build_dataset(["NVDA"], "1d", "2024-01-01", data_dir=tmp_path)


def test_descarga_con_paginacion_y_fusion(keys, monkeypatch, tmp_path):
    peticiones = []

    def handler(request: httpx.Request) -> httpx.Response:
        peticiones.append(request)
        assert request.headers["APCA-API-KEY-ID"] == "clave-de-prueba"
        q = request.url.params
        assert q["timeframe"] == "1Day" and q["adjustment"] == "all" and q["symbols"] == "NVDA,XOM"
        if "page_token" not in q:  # primera página: NVDA y token para seguir
            body = {"bars": {"NVDA": [_bar("2024-01-02T05:00:00Z", 100.0)]}, "next_page_token": "p2"}
        else:  # segunda página: XOM, sin token = fin
            assert q["page_token"] == "p2"
            body = {"bars": {"XOM": [_bar("2024-01-02T05:00:00Z", 50.0)]}, "next_page_token": None}
        return httpx.Response(200, json=body)

    _simular_alpaca(monkeypatch, handler)
    paths = a.build_dataset(["nvda", "xom"], "1d", "2024-01-01", end="2024-02-01", data_dir=tmp_path)
    assert len(peticiones) == 2
    assert sorted(p.name for p in paths) == ["NVDA_1d.parquet", "XOM_1d.parquet"]
    assert pl.read_parquet(tmp_path / "parquet" / "alpaca_stocks" / "NVDA_1d.parquet").height == 1

    # Segunda ejecución con una vela nueva y otra repetida: se fusionan sin duplicar.
    def handler2(request: httpx.Request) -> httpx.Response:
        bars = [_bar("2024-01-02T05:00:00Z", 100.0), _bar("2024-01-03T05:00:00Z", 101.0)]
        return httpx.Response(200, json={"bars": {"NVDA": bars}, "next_page_token": None})

    _simular_alpaca(monkeypatch, handler2)
    a.build_dataset(["NVDA"], "1d", "2024-01-01", end="2024-02-01", data_dir=tmp_path)
    df = pl.read_parquet(tmp_path / "parquet" / "alpaca_stocks" / "NVDA_1d.parquet")
    assert df.height == 2 and df["open_time"].is_sorted()


def test_error_de_alpaca_da_mensaje_claro(keys, monkeypatch, tmp_path):
    _simular_alpaca(monkeypatch, lambda r: httpx.Response(403, text="forbidden"))
    with pytest.raises(RuntimeError, match="403"):
        a.build_dataset(["NVDA"], "1d", "2024-01-01", data_dir=tmp_path)


def test_load_env_file_no_pisa_variables_existentes(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comentario\nFOO_TEST=desde_archivo\nexport BAR_TEST='con comillas'\n\nSIN_IGUAL\n")
    monkeypatch.setenv("FOO_TEST", "ya_definida")
    monkeypatch.delenv("BAR_TEST", raising=False)
    config.load_env_file(env)
    import os
    assert os.environ["FOO_TEST"] == "ya_definida"
    assert os.environ["BAR_TEST"] == "con comillas"


def test_un_pico_erroneo_se_corrige_y_se_anota(keys, monkeypatch, tmp_path):
    bars = [_bar(f"2024-06-{d:02d}T04:00:00Z", 120.0) for d in range(3, 13)]
    bars[7]["h"] = 195.39  # 10 de junio: máximo imposible, como el de NVIDIA el día del split
    body = {"bars": {"NVDA": bars}, "next_page_token": None}
    _simular_alpaca(monkeypatch, lambda r: httpx.Response(200, json=body))

    a.build_dataset(["NVDA"], "1d", "2024-06-01", end="2024-07-01", data_dir=tmp_path)

    df = pl.read_parquet(tmp_path / "parquet" / "alpaca_stocks" / "NVDA_1d.parquet")
    dia = df.filter(pl.col("open_time").dt.day() == 10)
    assert dia["high"][0] == 121.0          # max(open=120, close=121), no 195.39
    log = pl.read_csv(tmp_path / "quality" / "repairs.csv")
    assert log["field"].to_list() == ["high"] and log["old"][0] == 195.39
