"""Tests de las rutas /api/app/* (la interfaz lee JSON precalculados)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.app import build
from backend.events import fastperm
from test_app_build import AHORA, _mundo  # noqa: E402  (mundo sintético del test de la capa de datos)


@pytest.fixture(scope="module")
def datos(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    eventos, _ = _mundo(tmp)
    build.build_all(tmp, eventos, tmp / "app", ["AAA", "BBB"], draws=300, motor="python", warmup=300, now=AHORA)
    return tmp


@pytest.fixture
def client(datos, monkeypatch):
    monkeypatch.setenv("TFG_DATA_DIR", str(datos))
    return TestClient(app)


def test_meta_y_motor(client):
    meta = client.get("/api/app/meta").json()
    assert meta["symbols"] == ["AAA", "BBB"] and meta["generated_at"].startswith("2026-11-25")
    engine = client.get("/api/app/engine").json()
    assert engine["cpp"] is fastperm.available() and engine["threads"] >= 1
    assert engine["engine"] == ("cpp" if fastperm.available() else "python")


def test_events_filtra_por_categoria_y_valida_el_simbolo(client):
    todo = client.get("/api/app/events", params={"symbol": "aaa"}).json()  # minúsculas: se normaliza
    assert todo["symbol"] == "AAA" and todo["categories"] == ["directivos", "resultados"]
    solo = client.get("/api/app/events", params={"symbol": "AAA", "categories": "resultados"}).json()
    assert solo["events"] and {e["category"] for e in solo["events"]} == {"resultados"}
    assert len(solo["events"]) < len(todo["events"])
    assert client.get("/api/app/events", params={"symbol": "ZZZ"}).status_code == 404
    assert client.get("/api/app/events", params={"symbol": "../etc"}).status_code == 400
    assert client.get("/api/app/events").status_code == 422  # falta el símbolo


def test_volatility_devuelve_series_y_rango_de_manana(client):
    vol = client.get("/api/app/volatility", params={"symbol": "BBB"}).json()
    assert vol["symbol"] == "BBB" and len(vol["sessions"]) == len(vol["real"]) == len(vol["har_enext"])
    assert vol["next"]["har_e_event"] > vol["next"]["har_e_no_event"]
    assert vol["day_types"]["1"].startswith("reacción a resultados")
    assert client.get("/api/app/volatility", params={"symbol": "NOPE"}).status_code == 404


def test_resumen_de_eventos_y_modelos(client):
    resumen = client.get("/api/app/events-summary").json()
    assert resumen["headline"] and {c["category"] for c in resumen["categories"]} == {"directivos", "resultados"}
    modelos = client.get("/api/app/models").json()
    assert modelos["reference"] == "har_enext" and modelos["headline"]


def test_sin_datos_dice_que_comando_ejecutar(tmp_path, monkeypatch):
    monkeypatch.setenv("TFG_DATA_DIR", str(tmp_path / "vacio"))
    for ruta in ("meta", "events-summary", "models"):
        r = TestClient(app).get(f"/api/app/{ruta}")
        assert r.status_code == 404 and "python -m backend.app.build" in r.json()["detail"]
    assert TestClient(app).get("/api/app/engine").status_code == 200  # el motor no depende de los JSON


def test_los_json_no_llevan_nan_ni_infinitos(datos):
    for nombre in ("meta", "events", "events_summary", "volatility", "models"):
        texto = (datos / "app" / f"{nombre}.json").read_text(encoding="utf-8")
        assert "NaN" not in texto and "Infinity" not in texto
        json.loads(texto)  # JSON válido estricto
