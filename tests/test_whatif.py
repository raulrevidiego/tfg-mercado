"""Tests de la tabla de reacciones, de '¿y si…?' (estadísticas, API y página)."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.events import reactions
from backend.events.reactions import COLUMNS, load_reactions, quantile, save_reactions, tone_group, what_if
from test_events import NY, _mercado_y_accion  # noqa: E402
from test_har_e import _escenario  # noqa: E402
from test_tone import NEGATIVO, POSITIVO  # noqa: E402


def _fila(kind, symbol, category, car1, abn=1.0, tone=None, session="2024-01-02"):
    return {"kind": kind, "symbol": symbol, "session": session, "category": category, "car0": car1 / 2,
            "car1": car1, "car4": car1, "abn_range": abn, "z1": 0.0, "tone_rel": tone}


def _historia():
    cars = [0.06, -0.05, 0.04, -0.07, 0.05, -0.04, 0.06, -0.03, 0.05, -0.06, 0.04, -0.05, 0.03, -0.04, 0.06, -0.05]
    rows = [_fila("evento", "NVDA", "resultados", c, 1.5 + i / 10, 0.1 if c > 0 else -0.1) for i, c in enumerate(cars)]
    rows += [_fila("evento", "XOM", "resultados", c, 1.2) for c in (0.02, -0.03, 0.01)]
    rows += [_fila("azar", "NVDA", "azar", 0.01 * (-1) ** i) for i in range(40)]
    rows += [_fila("azar", "XOM", "azar", 0.005 * (-1) ** i) for i in range(40)]
    return rows


# ---------------------------------------------------------------------- estadísticas

def test_grupos_de_tono_y_cuantiles():
    assert [tone_group(t) for t in (0.2, 0.05, 0.0, -0.05, -0.2, None)] == [
        "positivo", "neutro", "neutro", "neutro", "negativo", "sin_dato"]
    assert quantile([1, 2, 3, 4], 0.5) == 2.5 and quantile([1, 2, 3, 4], 0.9) == pytest.approx(3.7)
    assert quantile([7], 0.9) == 7


def test_what_if_compara_con_un_dia_normal_y_dice_si_la_direccion_se_distingue():
    r = what_if(_historia(), "NVDA", "resultados")
    assert r["n"] == 16 and r["symbol"] == "NVDA" and r["scope"] == "symbol"
    assert r["reaction"]["median_abs"] == pytest.approx(0.05)
    assert r["reaction"]["normal_median_abs"] == pytest.approx(0.01)            # solo las fechas al azar de NVDA
    assert r["direction"]["share_up"] == 0.5 and r["direction"]["mean"] == pytest.approx(0.0)
    assert r["direction"]["distinguishable"] is False
    assert r["range"]["median_multiplier"] == pytest.approx(quantile([1.5 + i / 10 for i in range(16)], 0.5))
    assert any("no se distingue de cero" in w for w in r["warnings"])
    assert not any("Solo" in w for w in r["warnings"])                         # 16 eventos: suficientes


def test_what_if_filtra_por_tono_y_avisa_si_hay_pocos_eventos():
    r = what_if(_historia(), "NVDA", "resultados", tone="positivo")
    assert r["n"] == 8 and r["direction"]["share_up"] == 1.0 and r["direction"]["distinguishable"] is True
    assert any("Solo 8 eventos" in w for w in r["warnings"])
    assert what_if(_historia(), "NVDA", "resultados", tone="negativo")["direction"]["share_up"] == 0.0


def test_what_if_con_todas_las_acciones_y_sin_eventos():
    todas = what_if(_historia(), "NVDA", "resultados", scope="all")
    assert todas["n"] == 19 and todas["symbol"] == "(todas)"
    vacio = what_if(_historia(), "NVDA", "directivos")
    assert vacio["n"] == 0 and "No hay eventos" in vacio["warnings"][0] and "reaction" not in vacio


# ------------------------------------------------------------------ tabla de reacciones

def _datos(tmp_path, n_eventos=10, tonos=False):
    """Datos sintéticos: una acción con choques en cada evento y, opcionalmente, textos con tono."""
    shocks = {150 + 40 * i: (0.05 if i % 2 == 0 else -0.05) for i in range(n_eventos)}
    stock, market, sesiones = _mercado_y_accion(150 + 40 * n_eventos + 20, shocks)
    out = tmp_path / "parquet" / "alpaca_stocks"
    out.mkdir(parents=True)
    stock.write_parquet(out / "AAA_1d.parquet")
    market.write_parquet(out / "SPY_1d.parquet")
    (tmp_path / "events").mkdir()
    (tmp_path / "raw" / "edgar_text").mkdir(parents=True)
    filas, indice = ["symbol,datetime,category,source,note"], ["symbol,datetime,accession,document,n_chars,used_exhibit,path"]
    for i, k in enumerate(shocks):
        previa = sesiones[k] - timedelta(days=1)
        while previa.weekday() >= 5:
            previa -= timedelta(days=1)
        when = datetime(previa.year, previa.month, previa.day, 17, 0, tzinfo=NY)
        iso = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        filas.append(f"AAA,{iso},resultados,,")
        if i == 4:  # un segundo aviso el mismo día: debe fusionarse
            filas.append(f"AAA,{iso},otros,,duplicado")
        if tonos:
            path = tmp_path / "raw" / "edgar_text" / f"AAA_{i:018d}.txt"
            path.write_text("#document: ex991.htm\n" + (POSITIVO if i % 2 == 0 else NEGATIVO), encoding="utf-8")
            indice.append(f"AAA,{iso},{i:018d},ex991.htm,{path.stat().st_size},1,{path}")
    (tmp_path / "events" / "edgar_8k.csv").write_text("\n".join(filas) + "\n")
    if tonos:
        (tmp_path / "events" / "texts_index.csv").write_text("\n".join(indice) + "\n")


def test_la_tabla_tiene_eventos_independientes_y_fechas_al_azar(tmp_path, capsys):
    _datos(tmp_path)
    reactions.main(["--data-dir", str(tmp_path), "--placebo-n", "60"])
    out = capsys.readouterr().out
    assert "10 eventos independientes (0 con tono)" in out and "60 fechas al azar" in out
    rows = load_reactions(tmp_path / "events" / "reactions.csv")
    ev = [r for r in rows if r["kind"] == "evento"]
    assert len(ev) == 10 and {r["category"] for r in ev} == {"resultados"}      # el duplicado 'otros' se fusionó
    assert all(abs(r["car1"]) > 0.02 for r in ev)                                # los choques de ±5 % se ven
    assert sum(r["kind"] == "azar" for r in rows) == 60 and set(rows[0]) == set(COLUMNS)


def test_el_tono_relativo_se_guarda_cuando_hay_textos(tmp_path):
    _datos(tmp_path, n_eventos=12, tonos=True)
    reactions.main(["--data-dir", str(tmp_path), "--placebo-n", "30"])
    ev = [r for r in load_reactions(tmp_path / "events" / "reactions.csv") if r["kind"] == "evento"]
    ev.sort(key=lambda r: r["session"])
    assert all(r["tone_rel"] is None for r in ev[:3])                            # sin referencia previa
    con_tono = [r for r in ev if r["tone_rel"] is not None]
    assert len(con_tono) == 9
    assert all((r["tone_rel"] > 0) == (r["car1"] > 0) for r in con_tono)         # tono y reacción van a la par


def test_sin_eventos_pide_generarlos(tmp_path):
    with pytest.raises(SystemExit, match="python -m backend.data.edgar"):
        reactions.main(["--data-dir", str(tmp_path)])


def test_guardar_y_cargar_conserva_los_valores_vacios(tmp_path):
    path = tmp_path / "r.csv"
    save_reactions([_fila("evento", "A", "resultados", 0.03, tone=None)], path)
    assert load_reactions(path)[0]["tone_rel"] is None and load_reactions(path)[0]["car1"] == 0.03


# ------------------------------------------------------------------------------- API

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TFG_DATA_DIR", str(tmp_path))
    (tmp_path / "events").mkdir()
    save_reactions(_historia(), tmp_path / "events" / "reactions.csv")
    return TestClient(app)


def test_opciones(client):
    o = client.get("/api/whatif/options").json()
    assert o["symbols"] == ["NVDA", "XOM"] and o["categories"] == {"resultados": 19} and o["has_tone"] is True


def test_whatif_por_api(client):
    r = client.get("/api/whatif", params={"symbol": "nvda", "category": "resultados"}).json()
    assert r["n"] == 16 and r["reaction"]["median_abs"] == pytest.approx(0.05)
    assert r["baseline_range"] is None                                           # no hay serie de precios de NVDA aquí
    assert client.get("/api/whatif", params={"scope": "all"}).json()["n"] == 19


def test_whatif_con_rango_base_y_rango_esperado(client, tmp_path):
    stock, _, _, _ = _escenario(n=900)            # rangos diarios que varían (con rango constante el HAR no se puede ajustar)
    out = tmp_path / "parquet" / "alpaca_stocks"
    out.mkdir(parents=True)
    stock.write_parquet(out / "NVDA_1d.parquet")
    r = client.get("/api/whatif", params={"symbol": "NVDA"}).json()
    assert 0.001 < r["baseline_range"] < 0.1
    assert r["expected_range_event"] == pytest.approx(r["baseline_range"] * r["range"]["median_multiplier"])


def test_errores_de_la_api(client, tmp_path, monkeypatch):
    assert client.get("/api/whatif", params={"tone": "eufórico"}).status_code == 400
    assert client.get("/api/whatif", params={"scope": "mundo"}).status_code == 400
    assert client.get("/api/whatif", params={"symbol": "../x"}).status_code == 400
    assert client.get("/api/whatif", params={"category": "A B"}).status_code == 400
    monkeypatch.setenv("TFG_DATA_DIR", str(tmp_path / "vacio"))
    assert "build_reactions" in client.get("/api/whatif/options").json()["detail"]


def test_la_pagina_de_whatif_se_sirve_y_se_enlaza(client):
    assert "¿Y si" in client.get("/whatif.html").text
    assert 'href="whatif.html"' in client.get("/").text
