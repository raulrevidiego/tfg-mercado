"""La interfaz (ui/app): archivos locales, rutas de la API que usa y tests de JavaScript.

No hay navegador en los tests: aquí se comprueba lo que se puede comprobar sin él (que todo lo que index.html y los
módulos referencian existe, que las URL de la API que pide el JavaScript son rutas reales, que la paleta de CSS y de
JavaScript coincide y que pasan los tests de las funciones puras si Node está instalado)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

RAIZ = Path(__file__).resolve().parents[1]
UI = RAIZ / "ui" / "app"
JS = sorted((UI / "js").glob("*.js"))
INDEX = (UI / "index.html").read_text(encoding="utf-8")


def test_index_solo_referencia_archivos_locales_que_existen():
    refs = re.findall(r'(?:src|href)="([^"]+)"', INDEX)
    assert refs, "index.html no referencia nada"
    for ref in refs:
        assert not re.match(r"(?:https?:)?//", ref), f"{ref}: la interfaz debe funcionar sin conexión (nada de CDN)"
        if ref.startswith("/"):  # enlace a otra parte de la misma API, como /prototype/whatif.html
            continue
        assert (UI / ref).exists(), f"index.html referencia {ref}, que no existe"


def test_los_modulos_importan_archivos_que_existen():
    for modulo in JS:
        for ruta in re.findall(r"from '(\./[^']+)'", modulo.read_text(encoding="utf-8")):
            assert (modulo.parent / ruta).exists(), f"{modulo.name} importa {ruta}, que no existe"


def test_las_urls_de_la_api_que_pide_el_javascript_son_rutas_reales():
    rutas = set(app.openapi()["paths"])  # incluye las rutas de los routers añadidos con include_router
    usadas = set()
    for modulo in JS:
        usadas |= set(re.findall(r"[`'\"](/api/[A-Za-z0-9/_-]+)", modulo.read_text(encoding="utf-8")))
    assert {"/api/watchlist", "/api/datasets", "/api/candles", "/api/app/meta", "/api/app/events",
            "/api/app/volatility", "/api/app/events-summary", "/api/app/models", "/api/app/engine"} <= usadas
    assert sorted(usadas - rutas) == []


def test_la_paleta_de_css_y_de_javascript_coincide():
    css = (UI / "css" / "app.css").read_text(encoding="utf-8")
    tema = (UI / "js" / "theme.js").read_text(encoding="utf-8")
    nombres = {"bg": "--bg", "card": "--card", "border": "--border", "blue": "--blue", "magenta": "--magenta", "ink": "--ink", "muted": "--muted"}
    for js_name, css_name in nombres.items():
        en_css = re.search(re.escape(css_name) + r":\s*(#[0-9A-Fa-f]{6})", css).group(1).upper()
        en_js = re.search(js_name + r":\s*'(#[0-9A-Fa-f]{6})'", tema).group(1).upper()
        assert en_css == en_js, f"{js_name}: {en_css} en CSS y {en_js} en theme.js"


def test_la_api_sirve_la_interfaz_la_libreria_y_el_simulador_provisional():
    client = TestClient(app)
    home = client.get("/")
    assert home.status_code == 200 and "TFG Mercado" in home.text and "Mercado y eventos" in home.text
    js = client.get("/js/main.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]  # los navegadores exigen este tipo para los módulos
    assert client.get("/vendor/lightweight-charts.standalone.production.js").status_code == 200
    simulador = client.get("/prototype/whatif.html")
    assert simulador.status_code == 200 and 'href="/"' in simulador.text
    assert client.get("/prototype/").text.count("whatif.html") == 1  # antes el enlace estaba pegado tres veces


def _node_ok() -> bool:
    node = shutil.which("node")
    if not node:
        return False
    version = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v")
    return bool(version) and int(version.split(".")[0]) >= 20


@pytest.mark.skipif(not _node_ok(), reason="Node 20 o superior no está instalado")
def test_los_tests_de_javascript_pasan():
    archivos = sorted(str(p.relative_to(UI)) for p in (UI / "tests").glob("*.test.js"))
    assert len(archivos) >= 3
    r = subprocess.run(["node", "--test", *archivos], cwd=UI, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
