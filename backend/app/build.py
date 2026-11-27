"""Datos precalculados de la interfaz (paso 22).

La interfaz no calcula nada pesado: lee unos JSON pequeños que genera este módulo una sola vez a partir de lo
ya descargado (velas, 8-K, predicciones de la red neuronal) y de los modelos y contrastes del proyecto.

    python -m backend.app.build                 # escribe data/app/*.json

Archivos (todas las cifras son fracciones: 0,01 = 1 %; los rangos son ln(máximo/mínimo)):
- meta.json            fecha de los datos, acciones, motor del test de permutaciones, remuestreos y semilla.
- events.json          reacción (CAR) de cada evento independiente, por acción.
- events_summary.json  resumen por categoría con su p, el veredicto de robustez, la insignia y la "lectura rápida".
- volatility.json      por acción, rango real y previsto por HAR, HAR-E y la red, día a día, más el rango
                       esperado para la siguiente sesión con y sin resultados.
- models.json          comparación de modelos (MAE y QLIKE frente a HAR-E, con margen) y su "lectura rápida".

Los textos de "lectura rápida" se generan desde los números (funciones `events_headline` y `models_headline`):
si cambian los datos, el texto cambia con ellos y la interfaz no dice nada que los datos no digan.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from backend.data.alpaca import DEFAULT_SYMBOLS
from backend.data.store import load_series
from backend.events import fastperm
from backend.events.evaluate_events import DEFAULT_PRIORITY, MARKET, analyze_events
from backend.events.robustness import robustness
from backend.events.study import summarize
from backend.events.table import load_events
from backend.models import ols
from backend.models.nn_vol import build_panel
from backend.models.volatility import FLOOR
from backend.training.evaluate_volatility import evaluate

SCHEMA = 1
REFERENCE = "har_enext"
DAY_TYPES = {
    "0": "resto de días",
    "1": "reacción a resultados (previsto la víspera)",
    "2": "día siguiente al de resultados",
}
DISCLAIMER = "Herramienta de estudio con datos históricos. No es asesoramiento financiero."


# ------------------------------------------------------------------ utilidades

def _es(x: float, decimals: int = 2) -> str:
    """Número con coma decimal, como en el resto de la documentación."""
    return f"{x:.{decimals}f}".replace(".", ",")


def clean(obj):
    """Prepara un objeto para JSON: NaN e infinitos pasan a null y los decimales se redondean."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 6)
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return obj


def write_json(out_dir: Path, name: str, obj) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


# ------------------------------------------------------------------ eventos

def category_badge(row: dict, robust: dict | None, threshold: float) -> dict:
    """Insignia de una categoría: nivel (none, unchecked, ok, mixed, fragile) y texto."""
    if row["p_abs"] >= threshold:
        return {"level": "none", "label": "SIN EFECTO SIGNIFICATIVO", "detail": ""}
    if robust is None:
        return {"level": "unchecked", "label": "SIGNIFICATIVA, SIN COMPROBAR", "detail": "Pocos eventos para repetir el contraste."}
    verdict = robust["verdict"]
    if verdict == "robusta":
        return {"level": "ok", "label": "ROBUSTA", "detail": "Aguanta sin cada acción y sin los mayores eventos."}
    if verdict == "robusta con excepciones":
        return {"level": "mixed", "label": "ROBUSTA CON EXCEPCIONES",
                "detail": f"Aguanta, pero no aparece en {', '.join(robust['exceptions'])}."}
    return {"level": "fragile", "label": "FRÁGIL", "detail": "Deja de pasar el umbral en: " + "; ".join(robust["reasons"]) + "."}


def events_headline(categories: list[dict], threshold: float) -> str:
    """"Lectura rápida" de la pantalla de eventos, escrita a partir de los números."""
    significant = [c for c in categories if c["badge"]["level"] != "none"]
    if not significant:
        return "Ninguna categoría de aviso a la SEC mueve la acción más que un día cualquiera."
    parts = []
    for c in significant:
        ratio = c["mean_abs_car1"] / c["expected_abs"]
        extra = f"; {c['badge']['label'].lower()}" + (f", {c['badge']['detail'][0].lower()}{c['badge']['detail'][1:].rstrip('.')}"
                                                      if c["badge"]["level"] in ("mixed",) else "")
        parts.append(f"{c['category']} (×{_es(ratio, 1)}: {_es(c['mean_abs_car1'] * 100)} % frente a "
                     f"{_es(c['expected_abs'] * 100)} % de un día normal{extra})")
    directional = [c["category"] for c in categories if c["p"] < threshold]
    tail = ("Ninguna tiene una dirección media detectable: los buenos y los malos avisos se compensan."
            if not directional else "Con dirección media detectable: " + ", ".join(directional) + ".")
    return "Mueven la acción en tamaño: " + "; ".join(parts) + ". " + tail


def build_events(
    data_dir: Path, events_path: Path, draws: int, seed: int, motor: str, min_gap: int = 5, placebo_n: int = 300
) -> tuple[dict, dict]:
    """(events.json, events_summary.json)."""
    analysis = analyze_events(load_events(events_path), data_dir, min_gap, DEFAULT_PRIORITY.split(","), placebo_n, seed)
    if not analysis.analyzed:
        raise SystemExit("Ningún evento se pudo analizar: ¿están descargadas las velas de esas acciones?")
    usable = [r for r in analysis.kept if analysis.pools.get(r["symbol"])]
    rows = summarize(usable, analysis.pools, draws, seed, motor)
    threshold = 0.05 / (2 * max(len(rows), 1))
    report = robustness(usable, analysis.pools, draws, seed, motor, threshold=threshold)

    events: dict[str, list[dict]] = {}
    for r in sorted(usable, key=lambda r: (r["symbol"], r["session"])):
        events.setdefault(r["symbol"], []).append({
            "session": r["session"], "category": r["category"], "car0": r.get("car0"), "car1": r["car1"],
            "car4": r.get("car4"), "z1": r.get("z1"), "abn_range": r.get("abn_range"),
        })

    categories = []
    for row in rows:
        rob = report.get(row["category"])
        categories.append({**row, "robustness": rob, "badge": category_badge(row, rob, threshold)})
    summary = {
        "threshold": threshold, "draws": draws, "seed": seed, "engine": fastperm.resolve_engine(motor),
        "counts": {"analyzed": len(analysis.analyzed), "skipped": len(analysis.skipped), "independent": len(analysis.kept),
                   "merged_same_day": analysis.n_same_day, "dropped_overlap": analysis.n_overlap},
        "categories": categories, "headline": events_headline(categories, threshold),
    }
    return events, summary


# ------------------------------------------------------------------ volatilidad y modelos

def next_forecast(frame: pl.DataFrame) -> dict | None:
    """Rango esperado para la siguiente sesión con HAR y con HAR-E (sin y con resultados mañana).

    Ajusta las dos regresiones con todas las filas de objetivo conocido; la última fila (hoy) aporta las variables."""
    rows = frame.drop_nulls(["target", "rng", "mean_5", "mean_22", "earn_next"])
    if rows.height < 100:
        return None
    base = [list(x) for x in zip(rows["rng"], rows["mean_5"], rows["mean_22"])]
    y = rows["target"].to_list()
    c_har = ols.fit(base, y)
    c_e = ols.fit([[*x, e] for x, e in zip(base, rows["earn_next"])], y)
    last = frame.row(-1, named=True)
    x = [last["rng"], last["mean_5"], last["mean_22"]]
    no_event = max(ols.predict(c_e, [*x, 0.0]), FLOOR)
    event = max(ols.predict(c_e, [*x, 1.0]), FLOOR)
    return {"as_of": last["session"], "rng_today": last["rng"], "mean_5": last["mean_5"], "mean_22": last["mean_22"],
            "har": max(ols.predict(c_har, x), FLOOR), "har_e_no_event": no_event, "har_e_event": event}


def volatility_series(frame: pl.DataFrame, nn: dict | None) -> dict:
    """Series día a día alineadas con el día PREDICHO: la predicción de la fila i-1 es para la sesión i."""
    sessions = frame["session"].to_list()
    rng, har, harE = frame["rng"].to_list(), frame["har"].to_list(), frame["har_enext"].to_list()
    nxt, today = frame["earn_next"].to_list(), frame["earn_today"].to_list()
    out = {"sessions": sessions[1:], "real": rng[1:], "har": har[:-1], "har_enext": harE[:-1],
           "nn": [nn.get(s) for s in sessions[:-1]] if nn else None,
           "type": [1 if nxt[i] == 1 else (2 if today[i] == 1 else 0) for i in range(len(sessions) - 1)]}
    return out


def _model_rows(rows: list[dict]) -> list[dict]:
    return [{"model": r["model"], "mae": r["abs"], "mae_gain": r["abs_gain"], "mae_margin": r["abs_margin"],
             "qlike": r["qlike"], "qlike_gain": r["qlike_gain"], "qlike_margin": r["qlike_margin"]} for r in rows]


def models_headline(samples: list[dict], models: list[str]) -> tuple[str, dict]:
    """("Lectura rápida" de la pantalla de modelos, datos de las tarjetas), desde los números."""
    by_key = {s["key"]: s for s in samples}
    cards: dict = {"best_classic": None, "nn": None}
    parts = []
    eve = by_key.get("pool_eve")
    if eve:
        har = next(r for r in eve["rows"] if r["model"] == "har")
        har_e = next(r for r in eve["rows"] if r["model"] == REFERENCE)
        reduction = 1 - har_e["mae"] / har["mae"]
        loses = har["mae_gain"] + har["mae_margin"] < 0
        cards["best_classic"] = {"mae_har": har["mae"], "mae_har_e": har_e["mae"], "reduction": reduction, "days": eve["n"], "significant": loses}
        if loses:
            parts.append(f"HAR-E es el mejor modelo clásico: en la víspera de resultados reduce el error un {_es(reduction * 100, 1)} % "
                         f"frente a HAR ({_es(har['mae'] * 100, 2)} frente a {_es(har_e['mae'] * 100, 2)} puntos de rango, {eve['n']} días).")
        else:
            parts.append("HAR-E no se distingue de HAR en la víspera de resultados con los datos disponibles.")
    if "nn" in models:
        stocks = [s for s in samples if s["key"].startswith("stock_")]
        nn_rows = [next(r for r in s["rows"] if r["model"] == "nn") for s in stocks]
        n = len(nn_rows)
        mae_worse = sum(r["mae_gain"] + r["mae_margin"] < 0 for r in nn_rows)
        better = sum((r["mae_gain"] - r["mae_margin"] > 0) or (r["qlike_gain"] - r["qlike_margin"] > 0) for r in nn_rows)
        cards["nn"] = {"stocks": n, "mae_worse_in": mae_worse, "better_in": better}
        if better == 0:
            parts.append(f"La red neuronal no supera a HAR-E: es peor en MAE en {mae_worse} de {n} acciones y no gana en ninguna.")
        else:
            parts.append(f"La red neuronal mejora a HAR-E en {better} de {n} acciones y es peor en MAE en {mae_worse}.")
    return " ".join(parts), cards


def build_models(panel: dict[str, pl.DataFrame], nn: dict[str, dict], warmup: int) -> dict:
    has_nn = bool(nn) and all(nn.get(s) for s in panel)
    models = ["har", REFERENCE] + (["nn"] if has_nn else [])
    frames = {}
    for sym, f in panel.items():
        if has_nn:
            f = f.with_columns(pl.Series("nn", [nn[sym].get(d) for d in f["session"].to_list()], dtype=pl.Float64))
        frames[sym] = f
    samples = []
    for sym, f in frames.items():
        rows, mean_range, n = evaluate(f, warmup, models, REFERENCE)
        samples.append({"key": f"stock_{sym}", "label": f"{sym} · todos los días", "n": n, "mean_range": mean_range, "rows": _model_rows(rows)})
    cols = ["target", "earn_next", "earn_today", *models]
    pooled = pl.concat([f.slice(warmup).select(cols) for f in frames.values()])
    masks = {
        "pool_eve": ("Conjunto · víspera de resultados", (pl.col("earn_next") == 1)),
        "pool_day": ("Conjunto · día de resultados", (pl.col("earn_today") == 1)),
        "pool_rest": ("Conjunto · resto de días", (pl.col("earn_next") == 0) & (pl.col("earn_today") == 0)),
    }
    for key, (label, mask) in masks.items():
        try:
            rows, mean_range, n = evaluate(pooled.filter(mask), 0, models, REFERENCE)
        except ValueError:  # sin días de ese tipo
            continue
        samples.append({"key": key, "label": label, "n": n, "mean_range": mean_range, "rows": _model_rows(rows)})
    headline, cards = models_headline(samples, models)
    return {"reference": REFERENCE, "models": models, "samples": samples, "headline": headline, "cards": cards, "day_types": DAY_TYPES}


def load_nn_predictions(data_dir: Path) -> dict[str, dict]:
    """Predicciones de la red guardadas por `evaluate_nn` ({acción: {sesión: rango previsto}}); vacío si no existen."""
    path = data_dir / "models" / "nn_predictions.parquet"
    if not path.exists():
        return {}
    df = pl.read_parquet(path)
    out: dict[str, dict] = {}
    for sym, session, value in zip(df["symbol"], df["session"], df["nn"]):
        out.setdefault(sym, {})[session] = value
    return out


def build_volatility(
    data_dir: Path, events_path: Path, symbols: list[str], category: str, warmup: int, refit_every: int
) -> tuple[dict, dict, date]:
    """(volatility.json, models.json, última sesión de los datos)."""
    stocks = {s: load_series(s, data_dir=data_dir) for s in symbols}
    panel = build_panel(stocks, load_series(MARKET, data_dir=data_dir), load_events(events_path), category, warmup, refit_every)
    nn = load_nn_predictions(data_dir)
    series = {}
    for sym, frame in panel.items():
        series[sym] = {**volatility_series(frame, nn.get(sym)), "next": next_forecast(frame)}
    models = build_models(panel, nn, warmup)
    last = max(f["session"].max() for f in panel.values())
    return {"day_types": DAY_TYPES, "symbols": series}, models, last


# ------------------------------------------------------------------ todo junto

def build_all(
    data_dir: Path, events_path: Path, out_dir: Path, symbols: list[str], draws: int, seed: int = 7,
    motor: str = "auto", category: str = "resultados", warmup: int = 500, refit_every: int = 21,
    now: datetime | None = None,
) -> dict[str, Path]:
    """Genera todos los JSON y devuelve {nombre: ruta}."""
    vol, models, last = build_volatility(data_dir, events_path, symbols, category, warmup, refit_every)
    events, summary = build_events(data_dir, events_path, draws, seed, motor)
    meta = {
        "schema": SCHEMA, "generated_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "data_as_of": last, "symbols": symbols, "reference": MARKET, "category": category,
        "events_file": str(events_path), "draws": draws, "seed": seed, "engine": summary["engine"],
        "warmup": warmup, "refit_every": refit_every, "has_nn": "nn" in models["models"],
        "disclaimer": DISCLAIMER,
    }
    return {
        "meta.json": write_json(out_dir, "meta.json", meta),
        "events.json": write_json(out_dir, "events.json", events),
        "events_summary.json": write_json(out_dir, "events_summary.json", summary),
        "volatility.json": write_json(out_dir, "volatility.json", vol),
        "models.json": write_json(out_dir, "models.json", models),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Genera los datos precalculados de la interfaz (data/app/*.json).")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--events", type=Path, default=None, help="tabla de eventos (por defecto data/events/edgar_8k.csv)")
    p.add_argument("--out", type=Path, default=None, help="carpeta de salida (por defecto <data-dir>/app)")
    p.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p.add_argument("--draws", type=int, default=None, help="remuestreos del test de permutaciones (200000 con C++, 20000 sin él)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--motor", choices=list(fastperm.ENGINES), default="auto")
    p.add_argument("--category", default="resultados", help="categoría de evento para el calendario de HAR-E")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--refit-every", type=int, default=21)
    args = p.parse_args(argv)

    if not (args.data_dir / "parquet").exists():
        raise SystemExit(
            f"\nNo encuentro {args.data_dir / 'parquet'}: ejecuta el comando desde la carpeta del proyecto (la que contiene "
            "backend/ y data/) o indica dónde están los datos con --data-dir.")
    engine = fastperm.resolve_engine(args.motor)
    draws = args.draws or (200_000 if engine == "cpp" else 20_000)
    events_path = args.events or args.data_dir / "events" / "edgar_8k.csv"
    out_dir = args.out or args.data_dir / "app"
    print(f"\nGenerando datos de la interfaz en {out_dir} (motor {engine}, {draws:,} remuestreos)...")
    files = build_all(args.data_dir, events_path, out_dir, args.symbols, draws, args.seed, args.motor,
                      args.category, args.warmup, args.refit_every)
    for name, path in files.items():
        print(f"  {name:<22}{path.stat().st_size / 1024:>8.0f} KB")
    for name, title in (("events_summary.json", "Lectura rápida de eventos"), ("models.json", "Lectura rápida de modelos")):
        print(f"\n{title}: " + json.loads(files[name].read_text(encoding="utf-8"))["headline"])
    if not (args.data_dir / "models" / "nn_predictions.parquet").exists():
        print("\nAviso: no hay data/models/nn_predictions.parquet (ejecuta `python -m backend.training.evaluate_nn`): "
              "la interfaz no mostrará la red neuronal.")
    print("\nListo. Arranca la interfaz con: uvicorn backend.api.main:app")


if __name__ == "__main__":
    main()
