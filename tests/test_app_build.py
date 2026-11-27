"""Tests de los datos precalculados de la interfaz (backend/app/build.py)."""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from backend.app import build
from test_events import NY, _bars, _guardar, _sessions  # noqa: E402  (datos sintéticos de test_events)

AHORA = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def _antes(sesion):
    previa = sesion - timedelta(days=1)
    while previa.weekday() >= 5:
        previa -= timedelta(days=1)
    return previa


def _mundo(tmp_path, simbolos=("AAA", "BBB"), n=900):
    """Mercado y acciones sintéticos: cada 63 sesiones hay resultados (rango ×3 y salto de ±6 %) y, 30 sesiones
    después, un aviso de directivos sin ningún efecto. Devuelve (ruta de eventos, sesiones)."""
    sesiones = _sessions(n)
    rm = random.Random(1)
    mercado = [rm.gauss(0, 0.008) for _ in range(n)]
    cierres_m, c = [], 100.0
    for r in mercado:
        c *= math.exp(r)
        cierres_m.append(c)
    _guardar(tmp_path, "SPY", _bars(sesiones, cierres_m))
    filas = ["symbol,datetime,category,source,note"]
    reaccion = list(range(150, n - 5, 63))
    for j, simbolo in enumerate(simbolos):
        rnd = random.Random(10 + j)
        x, c, cierres = 0.0, 100.0, []
        for i in range(n):
            x = 0.95 * x + rnd.gauss(0, 0.1)
            es_resultados = i in reaccion
            r = 1.2 * mercado[i] + rnd.gauss(0, 0.004 * math.exp(x))
            if es_resultados:
                r += rnd.choice([-0.06, 0.06])
            c *= math.exp(r)
            cierres.append(c)
        df = _bars(sesiones, cierres)
        # el rango diario sigue un proceso persistente y se triplica el día de resultados
        rangos = [0.012 * math.exp(0.5 * math.sin(i / 25 + j)) * (3.0 if i in reaccion else 1.0) for i in range(n)]
        df = df.with_columns(
            high=pl.Series([c_ * math.exp(r / 2) for c_, r in zip(cierres, rangos)]),
            low=pl.Series([c_ * math.exp(-r / 2) for c_, r in zip(cierres, rangos)]),
        )
        _guardar(tmp_path, simbolo, df)
        for categoria, indices in (("resultados", reaccion), ("directivos", [k + 30 for k in reaccion])):
            for k in indices:
                if k < n - 5:
                    d = _antes(sesiones[k])
                    filas.append(f"{simbolo},{datetime(d.year, d.month, d.day, 17, tzinfo=NY).isoformat()},{categoria},test,")
    ruta = tmp_path / "events"
    ruta.mkdir()
    (ruta / "edgar_8k.csv").write_text("\n".join(filas) + "\n", encoding="utf-8")
    return ruta / "edgar_8k.csv", sesiones


def _construir(tmp_path, **kwargs):
    eventos, sesiones = _mundo(tmp_path)
    salida = tmp_path / "app"
    archivos = build.build_all(
        tmp_path, eventos, salida, ["AAA", "BBB"], draws=300, motor="python", warmup=300, now=AHORA, **kwargs)
    return {k: json.loads(v.read_text(encoding="utf-8")) for k, v in archivos.items()}, sesiones


@pytest.fixture(scope="module")
def mundo(tmp_path_factory):
    return _construir(tmp_path_factory.mktemp("datos"))


# ------------------------------------------------------------------ utilidades

def test_clean_pasa_nan_a_null_redondea_y_convierte_fechas():
    out = build.clean({"a": float("nan"), "b": [1.23456789, float("inf")], "c": datetime(2026, 1, 2).date(), "d": (1, 2)})
    assert out == {"a": None, "b": [1.234568, None], "c": "2026-01-02", "d": [1, 2]}
    assert build._es(3.8712) == "3,87" and build._es(2.35, 1) == "2,4"


# ------------------------------------------------------------------ los cinco JSON

def test_se_escriben_los_cinco_json(mundo):
    datos, _ = mundo
    assert set(datos) == {"meta.json", "events.json", "events_summary.json", "volatility.json", "models.json"}


def test_meta_resume_los_datos_y_el_motor(mundo):
    meta, sesiones = mundo[0]["meta.json"], mundo[1]
    assert meta["schema"] == build.SCHEMA and meta["symbols"] == ["AAA", "BBB"]
    assert meta["data_as_of"] == sesiones[-1].isoformat() and meta["generated_at"].startswith("2026-11-25T12:00")
    assert meta["engine"] == "python" and meta["draws"] == 300 and meta["has_nn"] is False
    assert "asesoramiento" in meta["disclaimer"]


def test_events_solo_lleva_eventos_independientes_con_su_reaccion(mundo):
    events = mundo[0]["events.json"]
    assert set(events) == {"AAA", "BBB"}
    resultados = [e for e in events["AAA"] if e["category"] == "resultados"]
    assert len(resultados) >= 8 and [e["session"] for e in events["AAA"]] == sorted(e["session"] for e in events["AAA"])
    assert all(abs(e["car1"]) > 0.03 for e in resultados)  # el salto sintético de ±6 % se ve
    assert {"session", "category", "car0", "car1", "car4", "z1", "abn_range"} <= set(resultados[0])


def test_events_summary_distingue_resultados_de_directivos(mundo):
    resumen = mundo[0]["events_summary.json"]
    por_cat = {c["category"]: c for c in resumen["categories"]}
    assert por_cat["resultados"]["p_abs"] < resumen["threshold"] <= 0.05
    assert por_cat["resultados"]["badge"]["level"] in {"ok", "mixed"}
    assert por_cat["resultados"]["robustness"]["verdict"].startswith("robusta")
    assert por_cat["directivos"]["badge"]["level"] == "none"  # sin efecto: no hay nada que comprobar
    assert resumen["counts"]["independent"] == sum(len(v) for v in mundo[0]["events.json"].values())
    assert resumen["headline"].startswith("Mueven la acción en tamaño: resultados (×")
    assert "directivos" not in resumen["headline"].split("Ninguna")[0]


def test_volatility_alinea_cada_prediccion_con_el_dia_que_predice(mundo):
    vol = mundo[0]["volatility.json"]["symbols"]["AAA"]
    n = len(vol["sessions"])
    assert n > 0 and all(len(vol[k]) == n for k in ("real", "har", "har_enext", "type"))
    assert vol["nn"] is None
    assert vol["har"][0] is None and vol["har_enext"][0] is None  # antes del calentamiento no hay predicción
    assert any(v is not None for v in vol["har_enext"])
    assert sorted(set(vol["type"])) == [0, 1, 2]
    # los días de reacción a resultados (tipo 1) son los de mayor rango real
    reales = {t: [r for r, k in zip(vol["real"], vol["type"]) if k == t] for t in (0, 1)}
    assert sum(reales[1]) / len(reales[1]) > 2 * sum(reales[0]) / len(reales[0])


def test_el_rango_esperado_de_mañana_sube_si_hay_resultados(mundo):
    nxt = mundo[0]["volatility.json"]["symbols"]["AAA"]["next"]
    assert nxt["as_of"] == mundo[1][-1].isoformat()
    assert nxt["har_e_event"] > 2 * nxt["har_e_no_event"] > 0  # el calendario triplica el rango en los datos sintéticos
    assert nxt["har"] > 0 and nxt["rng_today"] > 0


def test_models_compara_har_con_har_e_y_dice_que_har_e_gana_en_la_vispera(mundo):
    models = mundo[0]["models.json"]
    assert models["reference"] == "har_enext" and models["models"] == ["har", "har_enext"]
    claves = [s["key"] for s in models["samples"]]
    assert {"stock_AAA", "stock_BBB", "pool_eve", "pool_day", "pool_rest"} <= set(claves)
    eve = next(s for s in models["samples"] if s["key"] == "pool_eve")
    har = next(r for r in eve["rows"] if r["model"] == "har")
    assert har["mae_gain"] < 0 and har["mae_gain"] + har["mae_margin"] < 0  # HAR pierde con claridad
    assert models["cards"]["best_classic"]["reduction"] > 0.2
    assert models["headline"].startswith("HAR-E es el mejor modelo clásico") and "víspera de resultados" in models["headline"]
    assert models["cards"]["nn"] is None


def test_con_predicciones_de_la_red_la_comparacion_incluye_nn(tmp_path):
    eventos, sesiones = _mundo(tmp_path)
    datos = {}
    for s in ("AAA", "BBB"):  # una "red" que siempre predice el doble de lo que debería: pierde claramente
        panel = build.build_panel({s: build.load_series(s, data_dir=tmp_path)}, build.load_series("SPY", data_dir=tmp_path),
                                  build.load_events(eventos), "resultados", 300, 21)[s]
        datos[s] = panel.select("session", nn=pl.col("har_enext") * 2.0).with_columns(symbol=pl.lit(s))
    (tmp_path / "models").mkdir()
    pl.concat(datos.values()).drop_nulls().write_parquet(tmp_path / "models" / "nn_predictions.parquet")
    archivos = build.build_all(tmp_path, eventos, tmp_path / "app", ["AAA", "BBB"], 300, motor="python", warmup=300, now=AHORA)
    models = json.loads(archivos["models.json"].read_text(encoding="utf-8"))
    vol = json.loads(archivos["volatility.json"].read_text(encoding="utf-8"))
    assert models["models"] == ["har", "har_enext", "nn"] and json.loads(archivos["meta.json"].read_text())["has_nn"] is True
    assert models["cards"]["nn"] == {"stocks": 2, "mae_worse_in": 2, "better_in": 0}
    assert "no supera a HAR-E" in models["headline"] and "2 de 2" in models["headline"]
    assert any(v is not None for v in vol["symbols"]["AAA"]["nn"])


# ------------------------------------------------------------------ textos de "lectura rápida"

def _cat(nombre, p_abs, p, media, azar, nivel, etiqueta, detalle=""):
    return {"category": nombre, "p_abs": p_abs, "p": p, "mean_abs_car1": media, "expected_abs": azar,
            "badge": {"level": nivel, "label": etiqueta, "detail": detalle}}


def test_category_badge_cubre_los_cinco_casos():
    fila = {"p_abs": 0.0001}
    assert build.category_badge({"p_abs": 0.3}, None, 0.005)["level"] == "none"
    assert build.category_badge(fila, None, 0.005)["level"] == "unchecked"
    assert build.category_badge(fila, {"verdict": "robusta"}, 0.005)["label"] == "ROBUSTA"
    mixta = build.category_badge(fila, {"verdict": "robusta con excepciones", "exceptions": ["XOM"]}, 0.005)
    assert mixta["level"] == "mixed" and "XOM" in mixta["detail"]
    fragil = build.category_badge(fila, {"verdict": "frágil", "reasons": ["sin XOM: p = 0.0293"]}, 0.005)
    assert fragil["level"] == "fragile" and "sin XOM" in fragil["detail"]


def test_events_headline_se_escribe_desde_los_numeros():
    cats = [
        _cat("resultados", 0.0, 0.86, 0.0387, 0.0164, "mixed", "ROBUSTA CON EXCEPCIONES", "Aguanta, pero no aparece en XOM."),
        _cat("regfd", 0.0, 0.19, 0.0282, 0.0177, "fragile", "FRÁGIL", "Deja de pasar el umbral en: sin XOM."),
        _cat("directivos", 0.5, 0.44, 0.0167, 0.0168, "none", "SIN EFECTO SIGNIFICATIVO"),
    ]
    texto = build.events_headline(cats, 0.005)
    assert "resultados (×2,4: 3,87 % frente a 1,64 % de un día normal; robusta con excepciones, aguanta, pero no aparece en XOM)" in texto
    assert "regfd (×1,6: 2,82 % frente a 1,77 % de un día normal; frágil)" in texto
    assert "directivos" not in texto and "Ninguna tiene una dirección media detectable" in texto
    cats[2]["p"] = 0.001
    assert "Con dirección media detectable: directivos." in build.events_headline(cats, 0.005)
    assert build.events_headline([cats[2]], 0.005).startswith("Ninguna categoría")


def _rows(**modelos):
    base = {"mae": 0.01, "mae_gain": 0.0, "mae_margin": 0.0, "qlike": 0.3, "qlike_gain": 0.0, "qlike_margin": 0.0}
    return [{"model": m, **base, **vals} for m, vals in modelos.items()]


def test_models_headline_cambia_con_los_datos():
    eve = {"key": "pool_eve", "n": 81, "rows": _rows(
        har={"mae": 0.01854, "mae_gain": -65.6, "mae_margin": 20.6}, har_enext={"mae": 0.0112})}
    acciones = [{"key": f"stock_{s}", "rows": _rows(
        har_enext={}, nn={"mae_gain": -4.5, "mae_margin": 2.5, "qlike_gain": 3.0, "qlike_margin": 4.0})} for s in "ABCD"]
    texto, tarjetas = build.models_headline([eve, *acciones], ["har", "har_enext", "nn"])
    assert "reduce el error un 39,6 %" in texto and "1,85 frente a 1,12 puntos de rango, 81 días" in texto
    assert "no supera a HAR-E: es peor en MAE en 4 de 4 acciones" in texto
    assert tarjetas["nn"] == {"stocks": 4, "mae_worse_in": 4, "better_in": 0}
    # si la red ganara con claridad en una acción, el texto lo dice
    acciones[0]["rows"][1].update(mae_gain=6.0, mae_margin=2.0)
    assert "mejora a HAR-E en 1 de 4 acciones" in build.models_headline([eve, *acciones], ["har", "har_enext", "nn"])[0]
    # si HAR no pierde con claridad, no se afirma que HAR-E gane
    eve["rows"][0].update(mae_gain=-5.0, mae_margin=20.0)
    assert "no se distingue de HAR" in build.models_headline([eve], ["har", "har_enext"])[0]


def test_el_build_avisa_si_no_se_ejecuta_desde_la_carpeta_del_proyecto(tmp_path):
    with pytest.raises(SystemExit, match="carpeta del proyecto"):
        build.main(["--data-dir", str(tmp_path / "no-existe")])
