"""Tests de la tabla de eventos, el estudio de eventos y su CLI (con datos sintéticos)."""

import math
import random
import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from backend.events import evaluate_events
from backend.events.study import (
    analyze_event, magnitude_pvalue, permutation_pvalue, placebo_pool, prepare,
    select_independent_events, summarize,
)
from backend.events.table import first_affected_session, load_events

NY = ZoneInfo("America/New_York")


# ------------------------------------------------------------------ datos sintéticos

def _sessions(n: int, start: date = date(2022, 1, 3)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _bars(sessions: list[date], closes: list[float]) -> pl.DataFrame:
    """Velas diarias con la hora que usa Alpaca: medianoche de Nueva York expresada en UTC."""
    times = [datetime(d.year, d.month, d.day, tzinfo=NY).astimezone(timezone.utc) for d in sessions]
    return pl.DataFrame({
        "open_time": times, "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes, "volume": [1e6] * len(closes),
    })


def _mercado_y_accion(n: int, shocks: dict[int, float] | None = None, seed: int = 5, beta: float = 1.2):
    """Retorno de la acción = beta * mercado + ruido (+ choques puntuales en los días indicados)."""
    rnd = random.Random(seed)
    sessions = _sessions(n)
    m = [rnd.gauss(0, 0.01) for _ in range(n)]
    s = [beta * m[i] + rnd.gauss(0, 0.008) + (shocks or {}).get(i, 0.0) for i in range(n)]
    cm, cs, mk, st = 100.0, 100.0, [], []
    for i in range(n):
        cm *= math.exp(m[i])
        cs *= math.exp(s[i])
        mk.append(cm)
        st.append(cs)
    return _bars(sessions, st), _bars(sessions, mk), sessions


# ------------------------------------------------------------------------- la tabla

def test_load_events_zonas_horarias_y_comentarios(tmp_path):
    f = tmp_path / "events.csv"
    f.write_text(
        "# comentario\n"
        "symbol,datetime,category,source,note\n"
        "nvda,2024-06-10T13:30:00Z,Positivo,https://x,hola\n"
        "\n"
        "XOM,2024-06-10T09:00:00-04:00,negativo,,\n"
        "GOOGL,2024-06-10T10:00:00,neutral,,\n"
    )
    ev = load_events(f)
    assert [e.symbol for e in ev] == ["NVDA", "XOM", "GOOGL"]
    assert ev[0].category == "positivo" and ev[0].source == "https://x"
    assert ev[0].when == datetime(2024, 6, 10, 13, 30, tzinfo=timezone.utc)
    assert ev[1].when == datetime(2024, 6, 10, 13, 0, tzinfo=timezone.utc)   # 09:00 EDT = 13:00 UTC
    assert ev[2].when.utcoffset() == timedelta(hours=-4)                      # sin zona: Nueva York (verano)


def test_load_events_errores_claros(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text("symbol,datetime\nNVDA,2024-06-10T10:00:00\n")
    with pytest.raises(ValueError, match="category"):
        load_events(f)
    f.write_text("symbol,datetime,category\nNVDA,2024-06-10T10:00:00,ok\nXOM,ayer por la tarde,ok\n")
    with pytest.raises(ValueError, match="evento nº 2"):
        load_events(f)
    f.write_text("symbol,datetime,category\n,2024-06-10T10:00:00,ok\n")
    with pytest.raises(ValueError, match="vacíos"):
        load_events(f)


def test_primera_sesion_afectada():
    sesiones = [d for d in _sessions(40, date(2024, 1, 2)) if d != date(2024, 1, 15)]  # sin festivo 15 ene
    sesion = lambda *a, **k: first_affected_session(datetime(*a, **k), sesiones)
    assert sesion(2024, 1, 9, 10, 0, tzinfo=NY) == date(2024, 1, 9)        # con la bolsa abierta
    assert sesion(2024, 1, 9, 8, 0, tzinfo=NY) == date(2024, 1, 9)         # antes de la apertura
    assert sesion(2024, 1, 9, 16, 0, tzinfo=NY) == date(2024, 1, 10)       # a las 16:00 ya está cerrada
    assert sesion(2024, 1, 9, 17, 30, tzinfo=NY) == date(2024, 1, 10)      # cerrada: siguiente sesión
    assert sesion(2024, 1, 12, 17, 0, tzinfo=NY) == date(2024, 1, 16)      # viernes tarde: lunes festivo
    assert sesion(2024, 1, 13, 11, 0, tzinfo=NY) == date(2024, 1, 16)      # sábado
    assert sesion(2030, 1, 1, tzinfo=NY) is None                           # fuera de los datos


def test_horario_de_verano_cambia_la_hora_utc_del_cierre():
    sesiones = _sessions(400, date(2024, 1, 2))
    # 20:30 UTC: en invierno son las 15:30 (abierta) y en verano las 16:30 (cerrada).
    invierno = first_affected_session(datetime(2024, 1, 9, 20, 30, tzinfo=timezone.utc), sesiones)
    verano = first_affected_session(datetime(2024, 7, 9, 20, 30, tzinfo=timezone.utc), sesiones)
    assert invierno == date(2024, 1, 9) and verano == date(2024, 7, 10)


# ------------------------------------------------------------------- el estudio

def test_prepare_alinea_y_usa_la_fecha_de_nueva_york():
    stock, market, sessions = _mercado_y_accion(30)
    data = prepare(stock, market.slice(0, 29))             # al mercado le falta el último día
    assert len(data["ret"]) == 28                           # 29 comunes menos el primero (sin retorno)
    assert data["session"] == sessions[1:29]                # fechas de Nueva York, no del UTC de las 04:00/05:00
    assert all(v is not None for v in data["ewma_prev"])


def test_car_exacto_con_datos_deterministas():
    m = [0.01 * ((i % 4) - 1.5) for i in range(200)]       # mercado con variación
    ret = [0.001 + 1.5 * x for x in m]                      # relación exacta alfa=0.001, beta=1.5
    idx, shocks = 150, [0.04, 0.01, -0.02, 0.0, 0.03]
    for k, sh in enumerate(shocks):
        ret[idx + k] += sh
    data = {"ret": ret, "mkt_ret": m, "rng": [0.02] * 200, "ewma_prev": [0.01] * 200, "session": list(range(200))}
    r = analyze_event(data, idx)
    assert r["alpha"] == pytest.approx(0.001) and r["beta"] == pytest.approx(1.5)
    assert r["car0"] == pytest.approx(0.04)
    assert r["car1"] == pytest.approx(0.05)
    assert r["car4"] == pytest.approx(0.06)
    assert r["abn_range"] == pytest.approx(2.0)


def test_detecta_un_choque_y_no_inventa_efectos():
    stock, market, _ = _mercado_y_accion(500, shocks={300: 0.06})
    data = prepare(stock, market)
    con = analyze_event(data, 299)                          # idx = k - 1 por el primer día descartado
    sin = analyze_event(data, 200)
    assert con["beta"] == pytest.approx(1.2, abs=0.15)
    assert con["car0"] == pytest.approx(0.06, abs=0.03) and con["z1"] > 3
    assert abs(sin["car1"]) < 0.04


def test_sin_datos_suficientes_devuelve_none():
    stock, market, _ = _mercado_y_accion(300)
    data = prepare(stock, market)
    assert analyze_event(data, 30) is None                  # poca historia antes
    assert analyze_event(data, len(data["ret"]) - 2) is None  # menos de 4 sesiones después


def test_placebo_respeta_exclusiones_y_la_semilla():
    stock, market, _ = _mercado_y_accion(600)
    data = prepare(stock, market)
    a = placebo_pool(data, 100, exclude_idx=[300], seed=1)
    b = placebo_pool(data, 100, exclude_idx=[300], seed=1)
    assert [r["idx"] for r in a] == [r["idx"] for r in b] and len(a) == 100
    assert all(abs(r["idx"] - 300) > 10 for r in a)


def test_permutaciones():
    rnd = random.Random(3)
    pools = {"A": [rnd.gauss(0, 0.01) for _ in range(500)]}
    assert permutation_pvalue(0.0, ["A", "A"], pools, draws=2000) > 0.9
    assert permutation_pvalue(0.10, ["A", "A"], pools, draws=2000) == pytest.approx(1 / 2001)
    assert permutation_pvalue(0.01, ["A"], pools, 500, seed=9) == permutation_pvalue(0.01, ["A"], pools, 500, seed=9)


def test_resumen_por_categoria():
    res = [
        {"symbol": "A", "category": "positivo", "car1": 0.04, "abn_range": 2.0},
        {"symbol": "A", "category": "positivo", "car1": 0.02, "abn_range": 1.0},
        {"symbol": "A", "category": "negativo", "car1": -0.03, "abn_range": 3.0},
    ]
    rows = {r["category"]: r for r in summarize(res, {"A": [0.0, 0.001, -0.001] * 50}, draws=200)}
    pos, neg = rows["positivo"], rows["negativo"]
    assert pos["n"] == 2 and pos["mean_car1"] == pytest.approx(0.03) and pos["share_pos"] == 1.0
    assert pos["margin"] == pytest.approx(1.96 * math.sqrt(0.0002) / math.sqrt(2))   # sd = 0.01414
    assert pos["mean_abn_range"] == pytest.approx(1.5)
    assert neg["margin"] is None and neg["share_pos"] == 0.0


def test_calibracion_sin_efecto_real_el_5_por_ciento_supera_z_196():
    stock, market, _ = _mercado_y_accion(1800, seed=21)
    data = prepare(stock, market)
    res = placebo_pool(data, 500, [], seed=2)
    zs = [r["z1"] for r in res]
    share = sum(abs(z) > 1.96 for z in zs) / len(zs)
    assert 0.02 < share < 0.10
    assert abs(sum(r["car1"] for r in res) / len(res)) < 0.004


# ---------------------------------------------------------------------- el CLI

def _guardar(tmp_path, nombre, df):
    out = tmp_path / "parquet" / "alpaca_stocks"
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / f"{nombre}_1d.parquet")


def test_cli_de_extremo_a_extremo(tmp_path, capsys):
    shocks = {300: 0.06, 400: 0.07, 500: -0.07}
    stock, market, sesiones = _mercado_y_accion(600, shocks)
    _guardar(tmp_path, "NVDA", stock)
    _guardar(tmp_path, "SPY", market)

    def a(k, hora, minuto=0):
        d = sesiones[k]
        return datetime(d.year, d.month, d.day, hora, minuto, tzinfo=NY).isoformat()

    (tmp_path / "events").mkdir()
    (tmp_path / "events" / "events.csv").write_text(
        "symbol,datetime,category,source,note\n"
        f"NVDA,{a(300, 10)},positivo,,en horario de bolsa\n"
        f"NVDA,{a(300, 11)},positivo,,otro aviso el mismo día: se fusiona\n"
        f"NVDA,{a(399, 18)},positivo,,tras el cierre: reacciona el día siguiente (400)\n"
        f"NVDA,{a(500, 11)},negativo,,\n"
        f"NVDA,{a(597, 10)},neutral,,demasiado cerca del final de los datos\n"
    )
    argv = ["--data-dir", str(tmp_path), "--draws", "800", "--placebo-n", "150"]
    evaluate_events.main(argv)
    out = capsys.readouterr().out
    assert "4 eventos analizados (1 omitidos" in out
    assert "→ 3 independientes: 1 fusionados" in out and "0 descartados por estar a menos de 5" in out
    pos = re.search(r"positivo\s+2 eventos\s+CAR medio\s+([+-]\d+\.\d+)", out)
    neg = re.search(r"negativo\s+1 eventos\s+CAR medio\s+([+-]\d+\.\d+)", out)
    assert pos and float(pos.group(1)) > 3.5          # en % (choques de +6 % y +7 %)
    assert neg and float(neg.group(1)) < -3.5
    assert "|CAR| medio" in out and "Bonferroni" in out and "p = 0.00" in out
    assert "Los 3 con mayor reacción" in out and "(omitido)" not in out
    evaluate_events.main([*argv, "--detalle"])
    detalle = capsys.readouterr().out
    assert "Todos los eventos independientes" in detalle and "(omitido) NVDA" in detalle


def test_cli_placebo_y_errores(tmp_path, capsys):
    stock, market, _ = _mercado_y_accion(900)
    _guardar(tmp_path, "NVDA", stock)
    _guardar(tmp_path, "SPY", market)
    evaluate_events.main(["--placebo", "--symbols", "NVDA", "--placebo-n", "100", "--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Prueba de calibración" in out and "NVDA" in out and "|z| > 1,96" in out
    with pytest.raises(SystemExit, match="No existe"):
        evaluate_events.main(["--data-dir", str(tmp_path)])


# ------------------------------------------------- eventos independientes y magnitud

def _ev(symbol, idx, category):
    return {"symbol": symbol, "idx": idx, "category": category}


PRIORIDAD = ["resultados", "directivos", "regfd", "otros_eventos", "otros"]


def test_seleccion_fusiona_el_mismo_dia_y_descarta_solapes():
    eventos = [
        _ev("A", 100, "otros"), _ev("A", 100, "resultados"),   # mismo día: se queda "resultados"
        _ev("A", 103, "directivos"),                           # a 3 sesiones de uno de mayor prioridad: fuera
        _ev("A", 106, "regfd"),                                # a 6 sesiones: independiente
        _ev("B", 100, "otros"),                                # otra acción: no interfiere
    ]
    kept, mismo_dia, solapados = select_independent_events(eventos, min_gap=5, priority=PRIORIDAD)
    assert [(e["symbol"], e["idx"], e["category"]) for e in kept] == [
        ("A", 100, "resultados"), ("B", 100, "otros"), ("A", 106, "regfd")]
    assert (mismo_dia, solapados) == (1, 1)


def test_en_un_solape_manda_la_prioridad_no_el_orden_cronologico():
    kept, _, solapados = select_independent_events(
        [_ev("A", 100, "otros"), _ev("A", 102, "resultados")], min_gap=5, priority=PRIORIDAD)
    assert [e["category"] for e in kept] == ["resultados"] and solapados == 1
    # Con min_gap=1 solo se fusionan los del mismo día: ambos se conservan.
    kept, _, _ = select_independent_events(
        [_ev("A", 100, "otros"), _ev("A", 102, "resultados")], min_gap=1, priority=PRIORIDAD)
    assert len(kept) == 2


def test_categorias_desconocidas_van_despues_y_el_desempate_es_estable():
    kept, _, _ = select_independent_events(
        [_ev("A", 100, "positivo"), _ev("A", 101, "resultados")], min_gap=5, priority=PRIORIDAD)
    assert [e["category"] for e in kept] == ["resultados"]


def test_la_magnitud_detecta_efectos_que_se_compensan():
    rnd = random.Random(8)
    pools = {"A": [rnd.gauss(0, 0.01) for _ in range(600)]}
    res = [{"symbol": "A", "category": "mixto", "car1": c, "abn_range": 1.0} for c in (0.06, -0.06, 0.05, -0.05)]
    fila = summarize(res, pools, draws=1500)[0]
    assert abs(fila["mean_car1"]) < 0.01 and fila["p"] > 0.5              # con signo: nada
    assert fila["mean_abs_car1"] == pytest.approx(0.055)
    assert fila["expected_abs"] == pytest.approx(0.008, abs=0.002)         # lo esperado al azar (~0,8 %)
    assert fila["p_abs"] == pytest.approx(1 / 1501)                        # en magnitud: imposible por azar


def test_pvalor_de_magnitud_con_efecto_nulo_es_alto():
    rnd = random.Random(2)
    pools = {"A": [rnd.gauss(0, 0.01) for _ in range(600)]}
    esperado = sum(abs(v) for v in pools["A"]) / 600
    assert magnitude_pvalue(esperado, ["A"] * 4, pools, draws=1500) > 0.3
    assert magnitude_pvalue(10 * esperado, ["A"] * 4, pools, draws=500) == pytest.approx(1 / 501)
