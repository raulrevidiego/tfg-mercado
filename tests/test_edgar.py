"""Tests del ingestor de EDGAR (8-K -> tabla de eventos) con un servidor simulado."""

from datetime import date

import httpx
import pytest

from backend.data import edgar
from backend.events.table import load_events

REAL_CLIENT = httpx.Client


def _recent(rows):
    """Construye el formato 'por columnas' de la SEC a partir de filas."""
    keys = edgar.COLUMNS
    return {k: [r[k] for r in rows] for k in keys}


def _fila(acc, fecha, aceptado, form="8-K", items="2.02,9.01", doc="d.htm"):
    return {"accessionNumber": acc, "filingDate": fecha, "acceptanceDateTime": aceptado,
            "form": form, "items": items, "primaryDocument": doc}


TICKERS = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
           "1": {"cik_str": 34088, "ticker": "XOM", "title": "EXXON MOBIL CORP"}}


def _servidor(monkeypatch, handler):
    monkeypatch.setattr(edgar.httpx, "Client", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(edgar, "PAUSE", 0)


@pytest.fixture
def ua(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Alumno Prueba alumno@correo.com")


def test_columnas_a_filas_y_categorias():
    rows = edgar.columns_to_rows(_recent([_fila("a-1", "2024-01-02", "2024-01-02T21:05:47.000Z"),
                                          _fila("a-2", "2024-02-02", "2024-02-02T13:00:00.000Z", items="5.02")]))
    assert [r["accessionNumber"] for r in rows] == ["a-1", "a-2"] and rows[1]["items"] == "5.02"
    assert edgar.category_for_items("2.02,9.01") == "resultados"
    assert edgar.category_for_items("5.02,7.01") == "directivos"       # prioridad: 5.02 antes que 7.01
    assert edgar.category_for_items("7.01") == "regfd"
    assert edgar.category_for_items("8.01,9.01") == "otros_eventos"
    assert edgar.category_for_items("1.01") == "otros" and edgar.category_for_items("") == "otros"


def test_to_events_filtra_formularios_y_fechas_y_arma_la_url():
    rows = edgar.columns_to_rows(_recent([
        _fila("0001045810-24-000029", "2024-02-21", "2024-02-21T21:05:47.000Z", items="2.02,9.01", doc="nvda-8k.htm"),
        _fila("0001045810-24-000030", "2024-02-22", "2024-02-22T12:00:00.000Z", form="10-K"),     # no es 8-K
        _fila("0001045810-19-000001", "2019-12-31", "2019-12-31T12:00:00.000Z"),                   # anterior a start
    ]))
    ev = edgar.to_events("NVDA", 1045810, rows, date(2020, 1, 1))
    assert len(ev) == 1
    assert ev[0] == {
        "symbol": "NVDA", "datetime": "2024-02-21T21:05:47Z", "category": "resultados",
        "source": "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-8k.htm",
        "note": "8-K items 2.02,9.01",
    }


def test_user_agent_obligatorio(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(edgar.MissingUserAgentError, match="SEC_USER_AGENT"):
        edgar.build_events(["NVDA"], date(2020, 1, 1))
    monkeypatch.setenv("SEC_USER_AGENT", "sin correo")
    with pytest.raises(edgar.MissingUserAgentError):
        edgar.get_user_agent()


def test_simbolo_desconocido_da_un_error_claro(ua, monkeypatch):
    _servidor(monkeypatch, lambda r: httpx.Response(200, json=TICKERS))
    with pytest.raises(RuntimeError, match="ZZZZ"):
        edgar.build_events(["NVDA", "ZZZZ"], date(2020, 1, 1))


def test_error_de_la_sec_menciona_el_user_agent(ua, monkeypatch):
    _servidor(monkeypatch, lambda r: httpx.Response(403, text="forbidden"))
    with pytest.raises(RuntimeError, match="User-Agent"):
        edgar.build_events(["NVDA"], date(2020, 1, 1))


def test_recientes_mas_archivos_antiguos_solo_si_hacen_falta(ua, monkeypatch):
    pedidas, agentes = [], set()

    def handler(request: httpx.Request) -> httpx.Response:
        pedidas.append(request.url.path)
        agentes.add(request.headers["user-agent"])
        p = request.url.path
        if p.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        if p.endswith("CIK0001045810.json"):
            recent = _recent([_fila("a-new", "2024-03-01", "2024-03-01T13:30:00.000Z", items="7.01")])
            files = [
                {"name": "CIK0001045810-submissions-001.json", "filingFrom": "2021-01-01", "filingTo": "2023-12-31"},
                {"name": "CIK0001045810-submissions-002.json", "filingFrom": "2010-01-01", "filingTo": "2019-12-31"},
            ]
            return httpx.Response(200, json={"filings": {"recent": recent, "files": files}})
        if p.endswith("submissions-001.json"):
            return httpx.Response(200, json=_recent([_fila("a-old", "2022-05-05", "2022-05-05T20:10:00.000Z", items="5.02")]))
        raise AssertionError(f"petición inesperada: {p}")

    _servidor(monkeypatch, handler)
    ev = edgar.build_events(["nvda"], date(2020, 1, 1))
    assert sorted(e["category"] for e in ev) == ["directivos", "regfd"]
    assert not any("submissions-002" in p for p in pedidas)          # el archivo de 2010-2019 no se pide
    assert agentes == {"Alumno Prueba alumno@correo.com"}


def test_el_csv_generado_lo_entiende_el_cargador_de_eventos(tmp_path):
    ev = [
        {"symbol": "XOM", "datetime": "2024-05-02T10:00:00Z", "category": "regfd", "source": "https://x", "note": "n, con coma"},
        {"symbol": "NVDA", "datetime": "2024-02-21T21:05:47Z", "category": "resultados", "source": "https://y", "note": ""},
    ]
    path = tmp_path / "events" / "edgar_8k.csv"
    edgar.write_events_csv(path, ev)
    cargados = load_events(path)
    assert [(e.symbol, e.category) for e in cargados] == [("NVDA", "resultados"), ("XOM", "regfd")]   # ordenados por fecha
    assert cargados[0].when.isoformat() == "2024-02-21T21:05:47+00:00" and cargados[1].note == "n, con coma"


def test_cli_de_extremo_a_extremo(ua, monkeypatch, tmp_path, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        if p.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        rows = [_fila("a-1", "2024-02-21", "2024-02-21T21:05:47.000Z"),
                _fila("a-2", "2024-04-02", "2024-04-02T13:00:00.000Z", items="5.02")]
        return httpx.Response(200, json={"filings": {"recent": _recent(rows), "files": []}})

    _servidor(monkeypatch, handler)
    edgar.main(["--symbols", "NVDA", "XOM", "--start", "2024-01-01", "--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Guardados 4 eventos" in out and "resultados" in out and "directivos" in out
    assert len(load_events(tmp_path / "events" / "edgar_8k.csv")) == 4


def test_une_el_cik_nuevo_con_el_antiguo_y_no_duplica(ua, monkeypatch):
    """Caso real: XOM cambió de registrante en julio de 2026; el historial previo está en el CIK 34088."""
    tickers = {"0": {"cik_str": 2115436, "ticker": "XOM", "title": "ExxonMobil Holdings Corp"}}
    pedidas = []

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        pedidas.append(p)
        if p.endswith("company_tickers.json"):
            return httpx.Response(200, json=tickers)
        if p.endswith("CIK0002115436.json"):    # registrante nuevo: solo lo posterior a la reorganización
            rows = [_fila("0001-26-000001", "2026-07-08", "2026-07-08T13:00:00.000Z", items="7.01"),
                    _fila("0000-26-999999", "2026-07-09", "2026-07-09T13:00:00.000Z", items="8.01")]
        elif p.endswith("CIK0000034088.json"):  # Exxon Mobil Corp: historial anterior (y un filing compartido)
            rows = [_fila("0000-24-000002", "2024-05-02", "2024-05-02T20:30:00.000Z", items="2.02,9.01"),
                    _fila("0000-26-999999", "2026-07-09", "2026-07-09T13:00:00.000Z", items="8.01")]
        else:
            raise AssertionError(f"petición inesperada: {p}")
        return httpx.Response(200, json={"filings": {"recent": _recent(rows), "files": []}})

    _servidor(monkeypatch, handler)
    ev = edgar.build_events(["XOM"], date(2020, 1, 1))
    assert sorted(e["datetime"] for e in ev) == [
        "2024-05-02T20:30:00Z", "2026-07-08T13:00:00Z", "2026-07-09T13:00:00Z"]   # el compartido, una sola vez
    urls = {e["source"].split("/data/")[1].split("/")[0] for e in ev}
    assert urls == {"2115436", "34088"}                                           # cada enlace con su CIK
    assert any("CIK0000034088" in p for p in pedidas)


def test_extra_cik_por_linea_de_comandos(ua, monkeypatch, tmp_path, capsys):
    pedidas = []

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        pedidas.append(p)
        if p.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        rows = [_fila("a-1", "2024-02-21", "2024-02-21T21:05:47.000Z")]
        return httpx.Response(200, json={"filings": {"recent": _recent(rows), "files": []}})

    _servidor(monkeypatch, handler)
    edgar.main(["--symbols", "NVDA", "--extra-cik", "nvda=99999", "--start", "2024-01-01", "--data-dir", str(tmp_path)])
    assert any("CIK0000099999" in p for p in pedidas)
