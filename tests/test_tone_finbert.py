"""Tests del tono con FinBERT y de la cita del CEO. El modelo real NO se descarga en los tests normales:
se usa un puntuador de mentira. Para probar el modelo de verdad: TFG_TEST_FINBERT=1 pytest tests/test_tone_finbert.py
"""

from __future__ import annotations

import os
import re
import sys

import pytest

from backend.events import evaluate_tone, tone_finbert
from backend.events.tone_finbert import extract_quotes, finbert_tone, make_tone_fn, split_sentences
from test_tone import NEGATIVO, POSITIVO, _escenario, _numero  # noqa: E402  (escenario sintético de test_tone)

AVISO = "This release contains forward-looking statements about risks and uncertainties. " * 3


def fake_scorer(sentences):
    """Puntuador de mentira: 'growth' = positiva, 'declined' = negativa, el resto neutra."""
    out = []
    for s in sentences:
        if "growth" in s:
            out.append((0.8, 0.1, 0.1))
        elif "declined" in s:
            out.append((0.1, 0.8, 0.1))
        else:
            out.append((0.1, 0.1, 0.8))
    return out


# ------------------------------------------------------------------ frases y citas

def test_split_sentences_quita_fragmentos_cortos_y_tablas_de_cifras():
    texto = (
        "Revenue grew strongly across all of our business segments this quarter. Net income $ 1,234 5,678 9,012 3,456\n"
        "2024 2025 2026 | 12.5 | 13.1 | 14.0 | 15.2 | 16.1\n"
        "See table. Operating margin improved because of lower costs and better pricing? Yes!"
    )
    frases = split_sentences(texto)
    assert frases == [
        "Revenue grew strongly across all of our business segments this quarter.",
        "Operating margin improved because of lower costs and better pricing?",
    ]
    assert len(split_sentences("This is a long enough sentence for the test. " * 50, max_sentences=7)) == 7
    assert len(split_sentences("word " * 400, max_chars=100)[0]) <= 100


def test_extract_quotes_comillas_curvas_y_rectas_y_descarta_lo_corto():
    curvas = f"Headline. “We delivered record growth and expect momentum to continue next year,” said the CEO. “Record.” {AVISO}"
    assert extract_quotes(curvas) == ["We delivered record growth and expect momentum to continue next year,"]
    rectas = 'Headline. "We delivered record growth and expect momentum to continue next year," said the CEO.'
    assert extract_quotes(rectas) == ["We delivered record growth and expect momentum to continue next year,"]
    assert extract_quotes("No quotes anywhere in this press release about the quarter.") == []


def test_extract_quotes_ignora_las_comillas_del_aviso_legal():
    texto = "Headline of the release. " * 20 + AVISO + ' “A quote inside the legal notice that has many words in it for sure.”'
    assert extract_quotes(texto) == []


# ------------------------------------------------------------------ tono con puntuador de mentira

def test_finbert_tone_es_la_media_de_positiva_menos_negativa():
    texto = "Our growth this quarter was excellent across all segments. Sales declined in one smaller region this year."
    t = finbert_tone(texto, fake_scorer)
    assert t["n_sentences"] == 2
    assert t["pos"] == pytest.approx(0.45) and t["neg"] == pytest.approx(0.45)
    assert t["score"] == pytest.approx(0.0)
    solo_bueno = finbert_tone("Our growth this quarter was excellent across all segments.", fake_scorer)
    assert solo_bueno["score"] == pytest.approx(0.7)
    assert finbert_tone("short", fake_scorer) is None  # nada que puntuar


def test_finbert_tone_parte_quotes_solo_puntua_las_citas():
    llamadas = []

    def scorer(sentences):
        llamadas.append(list(sentences))
        return fake_scorer(sentences)

    texto = ("Sales declined in one smaller region this year and we remain cautious overall. "
             "“Our growth this quarter was excellent across all of our segments,” said the CEO.")
    t = finbert_tone(texto, scorer, part="quotes")
    assert llamadas == [["Our growth this quarter was excellent across all of our segments,"]]
    assert t["score"] == pytest.approx(0.7)
    assert finbert_tone("No quotes here at all in this long enough sentence.", scorer, part="quotes") is None


def test_make_tone_fn_lexicon_todo_y_citas():
    texto = (f"Revenue declined on weak demand, headwinds, pressure, losses and shortfall this quarter. "
             f"“We delivered record growth and strong momentum for the whole company,” said the CEO.")
    todo = make_tone_fn("lexicon", "all")(texto)
    citas = make_tone_fn("lexicon", "quotes")(texto)
    assert citas["score"] > 0 > todo["score"]
    assert make_tone_fn("lexicon", "quotes")("Sin citas en este comunicado de prensa corto.") is None


def test_make_tone_fn_valida_y_pide_transformers(monkeypatch):
    with pytest.raises(ValueError, match="Método"):
        make_tone_fn("gpt")
    with pytest.raises(ValueError, match="Parte"):
        make_tone_fn("lexicon", "titulo")
    monkeypatch.setitem(sys.modules, "transformers", None)  # simula que no está instalado
    with pytest.raises(RuntimeError, match=r"\.\[nlp\]"):
        make_tone_fn("finbert")


# ------------------------------------------------------------------ evaluate_tone de extremo a extremo

def _con_citas(tmp_path):
    """Envuelve el texto de cada comunicado del escenario en una cita del CEO."""
    for path in (tmp_path / "raw" / "edgar_text").glob("*.txt"):
        cabecera, _, cuerpo = path.read_text(encoding="utf-8").partition("\n")
        path.write_text(f"{cabecera}\nResults for the quarter. “{cuerpo.strip()}” said the CEO.\n", encoding="utf-8")


def test_evaluate_tone_con_finbert_y_citas_detecta_el_efecto(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(tone_finbert, "load_scorer", lambda *a, **k: fake_scorer)
    _escenario(tmp_path, correlacionado=True)
    _con_citas(tmp_path)
    evaluate_tone.main(["--data-dir", str(tmp_path), "--draws", "1500", "--metodo", "finbert", "--parte", "quotes"])
    out = capsys.readouterr().out
    rho, p = _numero(out, "1. correlación tono relativo y CAR con signo")
    assert rho > 0.4 and p < 0.01
    assert "Tono: FinBERT; texto: solo las citas" in out


def test_evaluate_tone_sin_citas_omite_los_comunicados_y_avisa(tmp_path):
    _escenario(tmp_path, correlacionado=True)  # los textos no tienen comillas
    with pytest.raises(SystemExit, match=r"sin citas entre comillas.*al menos 20"):
        evaluate_tone.main(["--data-dir", str(tmp_path), "--parte", "quotes"])


def test_evaluate_tone_cuenta_los_omitidos_y_pide_transformers_sin_traza(tmp_path, capsys, monkeypatch):
    _escenario(tmp_path, correlacionado=True)
    _con_citas(tmp_path)
    textos = sorted((tmp_path / "raw" / "edgar_text").glob("*.txt"))
    textos[0].write_text("#document: ex991.htm\nSin citas en este comunicado de prensa.\n", encoding="utf-8")
    evaluate_tone.main(["--data-dir", str(tmp_path), "--draws", "300", "--parte", "quotes"])
    assert "1 comunicados omitidos por no tener citas" in capsys.readouterr().out
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(SystemExit, match=r"\.\[nlp\]"):
        evaluate_tone.main(["--data-dir", str(tmp_path), "--metodo", "finbert"])


def test_bonferroni_se_ajusta_al_numero_de_variantes(tmp_path, capsys):
    _escenario(tmp_path, correlacionado=True)
    evaluate_tone.main(["--data-dir", str(tmp_path), "--draws", "300"])
    assert "menor de 0,0125" in capsys.readouterr().out
    evaluate_tone.main(["--data-dir", str(tmp_path), "--draws", "300", "--variantes", "4"])
    salida = capsys.readouterr().out
    assert "4 contrastes × 4 variantes (16 en total)" in salida and "menor de 0,0031" in salida


# ------------------------------------------------------------------ modelo real (opt-in: descarga ~440 MB)

@pytest.mark.skipif(os.environ.get("TFG_TEST_FINBERT") != "1", reason="solo con TFG_TEST_FINBERT=1 (descarga el modelo)")
def test_finbert_real_distingue_frases_buenas_malas_y_neutras():
    scorer = tone_finbert.load_scorer()
    bueno, malo, neutro = scorer([
        "Revenue grew 22% year over year, well above our guidance, driven by strong demand.",
        "We recorded a significant impairment charge and lowered our full-year outlook.",
        "The company will hold its annual shareholder meeting on June 5 in Houston.",
    ])
    assert bueno[0] > bueno[1] and malo[1] > malo[0] and neutro[2] > max(neutro[0], neutro[1])
    assert all(abs(sum(p) - 1) < 1e-4 for p in (bueno, malo, neutro))
