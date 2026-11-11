"""Tests de la descarga de textos de EDGAR, el tono, las estadísticas y la evaluación del tono."""

import csv
import random
import re
from datetime import timedelta, timezone

import httpx
import pytest

from backend.data import edgar, edgar_text
from backend.events import evaluate_tone
from backend.events.tone import relative_tone, strip_boilerplate, tone_score
from test_events import NY, _bars, _mercado_y_accion  # noqa: E402  (utilidades de datos sintéticos)

REAL_CLIENT = httpx.Client


@pytest.fixture
def ua(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Alumno Prueba alumno@correo.com")
    monkeypatch.setattr(edgar, "PAUSE", 0)


def _servidor(monkeypatch, handler):
    monkeypatch.setattr(edgar_text.httpx, "Client", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))


# ------------------------------------------------------------------ textos de EDGAR

def test_parse_source_url_y_eleccion_del_anexo():
    url = "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-8k.htm"
    assert edgar_text.parse_source_url(url) == (1045810, "000104581024000029", "nvda-8k.htm")
    assert edgar_text.parse_source_url("https://ejemplo.com/otra.htm") is None
    nombres = ["nvda-8k.htm", "ex992.htm", "nvda-20240222xex991.htm", "R1.htm", "logo.jpg"]
    assert edgar_text.pick_document(nombres, "nvda-8k.htm") == "nvda-20240222xex991.htm"   # 99.1 antes que 99.2
    assert edgar_text.pick_document(["nvda-8k.htm", "R1.htm"], "nvda-8k.htm") == "nvda-8k.htm"


def test_html_a_texto_sin_scripts_ni_estilos():
    html = ("<html><head><style>p{color:red}</style><script>var x=1;</script></head><body>"
            "<p>Record&nbsp;revenue   of $26.0 billion</p><div>&amp; strong growth</div><p></p><p>Fin</p></body></html>")
    assert edgar_text.html_to_text(html) == "Record revenue of $26.0 billion\n& strong growth\nFin"


def _handler(peticiones, indice_ok=True):
    def handler(request: httpx.Request) -> httpx.Response:
        peticiones.append(request.url.path)
        if request.url.path.endswith("index.json"):
            if not indice_ok:
                return httpx.Response(404)
            return httpx.Response(200, json={"directory": {"item": [{"name": "nvda-8k.htm"}, {"name": "ex991.htm"}]}})
        if request.url.path.endswith("ex991.htm"):
            return httpx.Response(200, text="<p>Record revenue and strong growth. " + "Texto largo. " * 200 + "</p>")
        return httpx.Response(200, text="<p>Carátula del 8-K</p>")
    return handler


def test_fetch_text_usa_el_anexo_y_la_cache(ua, tmp_path):
    peticiones: list[str] = []
    cliente = REAL_CLIENT(transport=httpx.MockTransport(_handler(peticiones)))
    a = edgar_text.fetch_text(cliente, "NVDA", 1045810, "000104581024000029", "nvda-8k.htm", tmp_path)
    assert a["document"] == "ex991.htm" and a["used_exhibit"] and a["text"].startswith("Record revenue")
    n = len(peticiones)
    b = edgar_text.fetch_text(cliente, "NVDA", 1045810, "000104581024000029", "nvda-8k.htm", tmp_path)
    assert b == a and len(peticiones) == n            # la segunda vez sale de la caché: ninguna petición


def test_fetch_text_sin_indice_usa_el_documento_principal(ua, tmp_path):
    cliente = REAL_CLIENT(transport=httpx.MockTransport(_handler([], indice_ok=False)))
    r = edgar_text.fetch_text(cliente, "NVDA", 1, "000000000000000001", "nvda-8k.htm", tmp_path)
    assert r["document"] == "nvda-8k.htm" and not r["used_exhibit"] and r["text"] == "Carátula del 8-K"


def test_cli_de_textos_de_extremo_a_extremo(ua, monkeypatch, tmp_path, capsys):
    (tmp_path / "events").mkdir()
    base = "https://www.sec.gov/Archives/edgar/data/1045810/"
    (tmp_path / "events" / "edgar_8k.csv").write_text(
        "symbol,datetime,category,source,note\n"
        f"NVDA,2024-02-21T21:05:47Z,resultados,{base}000104581024000029/nvda-8k.htm,\n"
        f"NVDA,2024-05-22T20:30:00Z,resultados,{base}000104581024000050/nvda-8k.htm,\n"
        f"NVDA,2024-06-01T13:00:00Z,regfd,{base}000104581024000060/nvda-8k.htm,\n"       # otra categoría: se ignora
    )
    _servidor(monkeypatch, _handler([]))
    edgar_text.main(["--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Textos descargados: 2" in out and "con anexo de comunicado: 2" in out
    filas = list(csv.DictReader((tmp_path / "events" / "texts_index.csv").open(encoding="utf-8")))
    assert [f["datetime"] for f in filas] == ["2024-02-21T21:05:47Z", "2024-05-22T20:30:00Z"]
    assert filas[0]["document"] == "ex991.htm" and int(filas[0]["n_chars"]) > 1500


def test_cli_de_textos_avisa_si_el_texto_es_corto(ua, monkeypatch, tmp_path, capsys):
    (tmp_path / "events").mkdir()
    (tmp_path / "events" / "edgar_8k.csv").write_text(
        "symbol,datetime,category,source,note\nNVDA,2024-02-21T21:05:47Z,resultados,"
        "https://www.sec.gov/Archives/edgar/data/1/000000000000000001/nvda-8k.htm,\n")
    _servidor(monkeypatch, _handler([], indice_ok=False))
    edgar_text.main(["--data-dir", str(tmp_path)])
    assert "tienen menos de 1,500 caracteres" in capsys.readouterr().out


# ------------------------------------------------------------------------------ tono

def test_tono_cuenta_palabras_y_la_negacion_mas_simple():
    assert tone_score("Record growth and strong demand.")["score"] == 1.0
    assert tone_score("Revenue declined with weak demand and headwinds.")["score"] == -1.0
    assert tone_score("Results were not strong.")["score"] == -1.0          # negación: 'strong' pasa a negativa
    assert tone_score("We saw no decline.")["score"] == 1.0                  # negación: 'decline' pasa a positiva
    assert tone_score("Not a very good strong quarter.")["score"] == 1.0     # la negación llega a 3 palabras, no a 4
    assert tone_score("")["score"] == 0.0 and tone_score("Nada relevante aquí.")["n_words"] == 3


def test_el_aviso_legal_no_cuenta():
    cuerpo = "Record growth and strong results. " * 20
    legal = "Forward-looking statements involve risks: adverse declines, losses and headwinds."
    assert strip_boilerplate(cuerpo + legal) == cuerpo                       # al final: se corta
    assert tone_score(cuerpo + legal)["score"] == 1.0
    inicio = "Forward-looking statements " + "x " * 1300 + "record growth"     # al principio: se quitan 2.500 caracteres
    assert strip_boilerplate(inicio) == inicio[2500:] and "record growth" in strip_boilerplate(inicio)


def test_tono_relativo_frente_a_los_comunicados_anteriores():
    rel = relative_tone([0.5, 0.6, 0.4, 0.9, 0.1], window=3, min_prev=3)
    assert rel[:3] == [None, None, None]
    assert rel[3] == pytest.approx(0.9 - 0.5) and rel[4] == pytest.approx(0.1 - (0.6 + 0.4 + 0.9) / 3)


# ------------------------------------------------------------------------ estadísticas

def test_rangos_con_empates_y_spearman():
    assert evaluate_tone._ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
    x = [1, 2, 3, 4, 5]
    assert evaluate_tone.spearman(x, [10, 20, 25, 300, 1000]) == pytest.approx(1.0)    # monótona (no hace falta que sea lineal)
    assert evaluate_tone.spearman(x, [9, 7, 5, 3, 1]) == pytest.approx(-1.0)
    assert evaluate_tone.spearman(x, [4, 4, 4, 4, 4]) == 0.0


def test_pvalor_de_permutaciones_y_diferencia_de_terciles():
    x = list(range(30))
    assert evaluate_tone.permutation_p(evaluate_tone.spearman, x, [v * 2 for v in x], draws=800) == pytest.approx(1 / 801)
    rnd = random.Random(5)
    y = [rnd.gauss(0, 1) for _ in x]
    assert evaluate_tone.permutation_p(evaluate_tone.spearman, x, y, draws=800) > 0.05
    assert evaluate_tone.tercile_gap(list(range(1, 10)), [0, 0, 0, 5, 5, 5, 10, 10, 10]) == pytest.approx(10.0)


# ------------------------------------------------------------- evaluación de extremo a extremo

POSITIVO = "Record growth and strong demand with robust momentum. " * 6
NEGATIVO = "Revenue declined on weak demand, headwinds and pressure. " * 6


def _escenario(tmp_path, correlacionado: bool, n_sesiones=2400, cada=30, seed=3):
    rnd = random.Random(seed)
    reaccion = list(range(150, n_sesiones - 10, cada))
    signos = [rnd.choice([-1, 1]) for _ in reaccion]
    shocks = {k: 0.04 * s for k, s in zip(reaccion, signos)}
    stock, market, sesiones = _mercado_y_accion(n_sesiones, shocks, seed=seed)
    out = tmp_path / "parquet" / "alpaca_stocks"
    out.mkdir(parents=True)
    stock.write_parquet(out / "AAA_1d.parquet")
    market.write_parquet(out / "SPY_1d.parquet")

    (tmp_path / "events").mkdir()
    (tmp_path / "raw" / "edgar_text").mkdir(parents=True)
    eventos, indice = ["symbol,datetime,category,source,note"], ["symbol,datetime,accession,document,n_chars,used_exhibit,path"]
    for i, (k, s) in enumerate(zip(reaccion, signos)):
        previa = sesiones[k] - timedelta(days=1)
        while previa.weekday() >= 5:
            previa -= timedelta(days=1)
        when = __import__("datetime").datetime(previa.year, previa.month, previa.day, 17, 0, tzinfo=NY)
        iso = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        bueno = (s > 0) if correlacionado else rnd.random() < 0.5
        path = tmp_path / "raw" / "edgar_text" / f"AAA_{i:018d}.txt"
        path.write_text("#document: ex991.htm\n" + (POSITIVO if bueno else NEGATIVO), encoding="utf-8")
        eventos.append(f"AAA,{iso},resultados,,")
        indice.append(f"AAA,{iso},{i:018d},ex991.htm,{path.stat().st_size},1,{path}")
    (tmp_path / "events" / "edgar_8k.csv").write_text("\n".join(eventos) + "\n")
    (tmp_path / "events" / "texts_index.csv").write_text("\n".join(indice) + "\n")


def _numero(salida: str, etiqueta: str) -> float:
    m = re.search(rf"{re.escape(etiqueta)}.*?([+-]\d+\.\d+)\s+(\d\.\d+)", salida)
    assert m, f"no encuentro '{etiqueta}'"
    return float(m.group(1)), float(m.group(2))


def test_si_el_tono_acompana_a_la_reaccion_se_detecta(tmp_path, capsys):
    _escenario(tmp_path, correlacionado=True)
    evaluate_tone.main(["--data-dir", str(tmp_path), "--draws", "1500"])
    out = capsys.readouterr().out
    rho, p = _numero(out, "1. correlación tono relativo y CAR con signo")
    assert rho > 0.4 and p < 0.01
    assert "Bonferroni" in out and "4. signo del tono = signo del CAR" in out
    acierto = float(re.search(r"4\. signo del tono.*?(\d+\.\d) %", out).group(1))
    assert acierto > 70


def test_si_el_tono_es_ruido_no_se_inventa_un_efecto(tmp_path, capsys):
    _escenario(tmp_path, correlacionado=False)
    evaluate_tone.main(["--data-dir", str(tmp_path), "--draws", "1500"])
    rho, _ = _numero(capsys.readouterr().out, "1. correlación tono relativo y CAR con signo")
    assert abs(rho) < 0.35


def test_evaluate_tone_pide_los_archivos_que_faltan(tmp_path):
    with pytest.raises(SystemExit, match="python -m backend.data.edgar"):
        evaluate_tone.main(["--data-dir", str(tmp_path)])
    (tmp_path / "events").mkdir()
    (tmp_path / "events" / "edgar_8k.csv").write_text("symbol,datetime,category\n")
    with pytest.raises(SystemExit, match="edgar_text"):
        evaluate_tone.main(["--data-dir", str(tmp_path)])
