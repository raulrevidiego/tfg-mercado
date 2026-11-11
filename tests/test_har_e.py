"""Tests de HAR-E (HAR con calendario de resultados) y de su evaluación."""

import math
import random
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from backend.events.table import Event
from backend.models.har_e import forecast_har_e_frame
from backend.training import evaluate_har_e
from backend.training.evaluate_volatility import evaluate

NY = ZoneInfo("America/New_York")


def _sessions(n: int, start: date = date(2020, 1, 2)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _bars(sessions: list[date], ranges: list[float]) -> pl.DataFrame:
    times = [datetime(d.year, d.month, d.day, tzinfo=NY).astimezone(timezone.utc) for d in sessions]
    return pl.DataFrame({
        "open_time": times, "open": [100.0] * len(times), "close": [100.0] * len(times),
        "high": [100.0 * math.exp(r) for r in ranges], "low": [100.0] * len(times),
        "volume": [1e6] * len(times),
    })


def _evento(symbol: str, sesion: date, categoria: str = "resultados") -> Event:
    """Comunicado a las 17:00 de Nueva York del día ANTERIOR: la reacción cae en `sesion`."""
    previa = sesion - timedelta(days=1)
    while previa.weekday() >= 5:
        previa -= timedelta(days=1)
    return Event(symbol, datetime(previa.year, previa.month, previa.day, 17, 0, tzinfo=NY), categoria)


def _escenario(n: int = 1500, cada: int = 63, factor: float = 3.0, seed: int = 4):
    """Volatilidad persistente y, cada `cada` sesiones, un día de resultados donde el rango se multiplica."""
    rnd = random.Random(seed)
    sesiones = _sessions(n)
    x, rangos = 0.0, []
    for _ in range(n):
        x = 0.9 * x + rnd.gauss(0, 0.2)
        rangos.append(0.012 * math.exp(x))
    reaccion = list(range(40, n - 5, cada))
    for k in reaccion:
        rangos[k] *= factor
    return _bars(sesiones, rangos), [_evento("AAA", sesiones[k]) for k in reaccion], sesiones, reaccion


def test_las_variables_de_evento_apuntan_a_la_vispera_y_al_dia():
    stock, eventos, sesiones, reaccion = _escenario()
    frame, _ = forecast_har_e_frame(stock, eventos, "AAA", warmup=1000)
    hoy, manana = frame["earn_today"].to_list(), frame["earn_next"].to_list()
    assert [i for i, v in enumerate(hoy) if v == 1.0] == reaccion
    assert [i for i, v in enumerate(manana) if v == 1.0] == [k - 1 for k in reaccion]


def test_comunicado_de_viernes_tras_el_cierre_reacciona_el_lunes():
    sesiones = _sessions(60)
    viernes = next(s for s in sesiones[20:] if s.weekday() == 4)
    lunes = viernes + timedelta(days=3)
    eventos = [Event("AAA", datetime(viernes.year, viernes.month, viernes.day, 17, 30, tzinfo=NY), "resultados")]
    eventos += [_evento("AAA", sesiones[k]) for k in range(3, 20, 2)]            # para superar el mínimo de eventos
    frame, _ = forecast_har_e_frame(_bars(sesiones, [0.02] * 60), eventos, "AAA", warmup=1000)
    i = sesiones.index(lunes)
    assert frame["earn_today"][i] == 1.0 and frame["earn_next"][i - 1] == 1.0     # víspera = el viernes
    assert frame["earn_today"][i - 1] == 0.0


def test_solo_cuenta_la_categoria_y_la_accion_pedidas():
    stock, eventos, sesiones, reaccion = _escenario()
    ruido = [_evento("AAA", sesiones[k + 7], "regfd") for k in reaccion] + [_evento("BBB", sesiones[k + 9]) for k in reaccion]
    frame, _ = forecast_har_e_frame(stock, eventos + ruido, "AAA", warmup=1000)
    assert [i for i, v in enumerate(frame["earn_today"].to_list()) if v == 1.0] == reaccion


def test_con_pocos_eventos_avisa_en_lugar_de_fallar_con_un_error_oscuro():
    stock, eventos, _, _ = _escenario()
    with pytest.raises(ValueError, match="solo 3 eventos"):
        forecast_har_e_frame(stock, eventos[:3], "AAA", warmup=500)


def test_quitar_un_evento_futuro_no_cambia_las_predicciones_anteriores():
    stock, eventos, sesiones, reaccion = _escenario()
    ultimo = reaccion[-1]
    a, _ = forecast_har_e_frame(stock, eventos, "AAA", warmup=500)
    b, _ = forecast_har_e_frame(stock, eventos[:-1], "AAA", warmup=500)
    for col in ("har", "har_enext", "har_e"):
        for dia in range(ultimo - 1):          # el día ultimo-1 ya sabe que "mañana" hay evento
            x, y = a[col][dia], b[col][dia]
            assert (x is None and y is None) or x == pytest.approx(y), f"{col} cambia en el día {dia}"


def test_si_los_resultados_amplifican_el_rango_har_e_gana_sobre_todo_en_la_vispera():
    stock, eventos, _, _ = _escenario(n=1800)
    frame, models = forecast_har_e_frame(stock, eventos, "AAA", warmup=500)
    todos, _, _ = evaluate(frame, 500, models=models, reference="har")
    vispera, _, n_v = evaluate(frame.slice(500).filter(pl.col("earn_next") == 1), 0, models=models, reference="har")
    por = {r["model"]: r for r in vispera}
    assert n_v >= 15
    assert por["har_e"]["abs_gain"] > 30 and por["har_e"]["qlike_gain"] > 30          # la víspera mejora muchísimo
    assert por["har_e"]["abs_gain"] > por["har_e"]["abs_margin"]
    assert {r["model"]: r for r in todos}["har_e"]["abs_gain"] > 0                     # y en conjunto también


def _guardar(tmp_path, nombre, df):
    out = tmp_path / "parquet" / "alpaca_stocks"
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / f"{nombre}_1d.parquet")


def test_cli_de_extremo_a_extremo(tmp_path, capsys):
    stock, eventos, _, _ = _escenario(n=1500)
    _guardar(tmp_path, "AAA", stock)
    _guardar(tmp_path, "BBB", _bars(_sessions(1500), [0.02] * 1500))        # sin eventos: se omite con aviso
    (tmp_path / "events").mkdir()
    filas = ["symbol,datetime,category,source,note"]
    filas += [f"AAA,{e.when.isoformat()},{e.category},," for e in eventos]
    (tmp_path / "events" / "edgar_8k.csv").write_text("\n".join(filas) + "\n")
    evaluate_har_e.main(["--symbols", "AAA", "BBB", "--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "AAA · todos los días" in out and "(aviso) BBB: solo 0 eventos" in out
    for corte in ("víspera de resultados", "día de resultados", "resto de días"):
        assert f"CONJUNTO · {corte}" in out
    assert "har_enext" in out and "har_e " in out and "(referencia)" in out


def test_cli_sin_archivo_de_eventos(tmp_path):
    with pytest.raises(SystemExit, match="python -m backend.data.edgar"):
        evaluate_har_e.main(["--data-dir", str(tmp_path)])
