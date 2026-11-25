"""Tests de las pruebas de robustez del estudio de eventos (backend/events/robustness.py)."""

from __future__ import annotations

import random
import re
from datetime import datetime

import pytest

from backend.events import evaluate_events, fastperm, robustness
from test_events import NY, _guardar, _mercado_y_accion  # noqa: E402  (datos sintéticos de test_events)

STOCKS = ["AAA", "BBB", "CCC", "DDD"]
THRESHOLD = 0.005


def _pools(seed=1):
    rnd = random.Random(seed)
    return {s: [rnd.gauss(0, 0.015) for _ in range(300)] for s in STOCKS}


def _events(category, n_per_stock, scale, seed, special=None):
    """Eventos con CAR = ±|N(0, scale)|; `special` = {acción: escala} para dar a una acción otra escala."""
    rnd = random.Random(seed)
    out = []
    for s in STOCKS:
        sc = (special or {}).get(s, scale)
        for _ in range(n_per_stock):
            out.append({"symbol": s, "category": category, "car1": rnd.choice([-1, 1]) * abs(rnd.gauss(0, sc))})
    return out


def _report(results, engine="python"):
    return robustness.robustness(results, _pools(), draws=3000, seed=7, engine=engine, threshold=THRESHOLD)


def test_un_efecto_real_en_todas_las_acciones_sale_robusto():
    r = _report(_events("real", 10, 0.045, seed=2))["real"]
    assert r["verdict"] == "robusta" and r["exceptions"] == [] and r["reasons"] == []
    assert len(r["by_stock"]) == 4 and all(x["ratio"] > 1 for x in r["by_stock"])
    assert len(r["without_stock"]) == 4 and all(x["p"] < THRESHOLD for x in r["without_stock"])
    assert r["trimmed"]["p"] < THRESHOLD and len(r["trimmed"]["dropped"]) == 3


def test_un_efecto_que_solo_viene_de_una_accion_sale_fragil():
    r = _report(_events("una", 10, 0.015, seed=8, special={"AAA": 0.07}))["una"]
    assert r["overall"]["p"] < THRESHOLD  # en conjunto parece significativo...
    assert r["verdict"] == "frágil"
    sin_aaa = next(x for x in r["without_stock"] if x["without"] == "AAA")
    assert sin_aaa["p"] > 0.05  # ...pero sin AAA desaparece
    assert any(m.startswith("sin AAA") for m in r["reasons"])


def test_un_efecto_que_no_aparece_en_una_accion_sale_robusto_con_excepciones():
    # como "resultados": aguanta sin cada acción, pero en DDD las reacciones son las de un día normal
    r = _report(_events("exc", 10, 0.045, seed=2, special={"DDD": 0.015}))["exc"]
    assert r["overall"]["p"] < THRESHOLD and r["reasons"] == []
    assert r["verdict"] == "robusta con excepciones" and r["exceptions"] == ["DDD"]
    texto = robustness.format_robustness({"exc": r}, THRESHOLD, 3000)
    assert "→ ROBUSTA CON EXCEPCIONES" in texto and "no aparece en DDD" in texto


def test_un_unico_evento_gigante_sale_fragil():
    eventos = _events("outlier", 8, 0.015, seed=4)
    eventos[0]["car1"] = 0.6
    r = _report(eventos)["outlier"]
    assert r["overall"]["p"] < THRESHOLD
    assert r["trimmed"]["p"] > 0.05 and r["verdict"] == "frágil"
    assert any(m.startswith("sin los 3 mayores") for m in r["reasons"])
    assert r["trimmed"]["dropped"][0] == ("AAA", 0.6)


def test_sin_efecto_no_se_investiga_y_con_pocos_eventos_se_omite():
    rep = _report(_events("nulo", 10, 0.015, seed=5) + _events("poco", 1, 0.2, seed=6))
    assert rep["nulo"]["verdict"] == "sin efecto detectable" and rep["nulo"]["by_stock"] == []
    assert "poco" not in rep  # 4 eventos < MIN_EVENTS


def test_el_informe_se_lee_y_dice_el_veredicto():
    rep = _report(_events("real", 10, 0.045, seed=2) + _events("una", 10, 0.015, seed=8, special={"AAA": 0.07}))
    texto = robustness.format_robustness(rep, THRESHOLD, 3000)
    assert "→ ROBUSTA" in texto and "→ FRÁGIL" in texto and "sin AAA" in texto and "sin los 3 mayores" in texto
    assert "p < 0.00033" in texto  # el p mínimo con 3000 remuestreos se muestra como cota, no como 0
    assert "ninguna categoría" in robustness.format_robustness({}, THRESHOLD, 3000)


@pytest.mark.skipif(not fastperm.available(), reason="núcleo C++ sin compilar (python core/build.py)")
def test_con_el_motor_cpp_los_veredictos_son_los_mismos():
    todo = _events("real", 10, 0.045, seed=2) + _events("una", 10, 0.015, seed=8, special={"AAA": 0.07})
    todo += _events("nulo", 10, 0.015, seed=5)
    py, cpp = _report(todo, "python"), _report(todo, "cpp")
    assert {k: v["verdict"] for k, v in py.items()} == {k: v["verdict"] for k, v in cpp.items()}


def test_la_cli_imprime_la_robustez(tmp_path, capsys):
    ks = [150 + 60 * i for i in range(12)]
    stock, market, sesiones = _mercado_y_accion(900, {k: (0.06 if i % 2 else -0.06) for i, k in enumerate(ks)})
    _guardar(tmp_path, "NVDA", stock)
    _guardar(tmp_path, "SPY", market)
    filas = "".join(
        f"NVDA,{datetime(sesiones[k].year, sesiones[k].month, sesiones[k].day, 10, tzinfo=NY).isoformat()},positivo,,\n"
        for k in ks)
    (tmp_path / "events").mkdir()
    (tmp_path / "events" / "events.csv").write_text("symbol,datetime,category,source,note\n" + filas)
    evaluate_events.main(["--data-dir", str(tmp_path), "--draws", "600", "--placebo-n", "150", "--robustez"])
    out = capsys.readouterr().out
    assert "Robustez de la reacción en tamaño" in out and "positivo · 12 eventos" in out
    assert re.search(r"→ (ROBUSTA|FRÁGIL)", out) and "solo NVDA" in out and "sin los 3 mayores" in out
    evaluate_events.main(["--data-dir", str(tmp_path), "--draws", "600", "--placebo-n", "150"])
    assert "Robustez de la reacción" not in capsys.readouterr().out  # solo con --robustez
