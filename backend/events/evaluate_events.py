"""Analiza el efecto de los eventos (comunicados, comentarios de CEOs...) sobre el precio.

Dos modos, desde la raíz del repo:

    python -m backend.events.evaluate_events --events data/events/edgar_8k.csv   # tus eventos
    python -m backend.events.evaluate_events --placebo                          # calibración con fechas al azar

El modo placebo no necesita eventos: finge eventos en fechas aleatorias y comprueba cuántas veces
el método "ve" un efecto que no existe. Es la forma de saber hasta qué punto fiarse del método.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from backend.data.alpaca import DEFAULT_SYMBOLS
from backend.data.store import load_series
from backend.events.study import (
    analyze_event, placebo_pool, prepare, select_independent_events, summarize,
)
from backend.events.table import first_affected_session, load_events

MARKET = "SPY"
DEFAULT_PRIORITY = "resultados,directivos,regfd,otros_eventos,otros"


def _pct(x: float) -> float:
    return x * 100


def _event_line(r: dict) -> str:
    z = "–" if r["z1"] is None else f"{r['z1']:+.1f}"
    return (
        f"{r['session']!s:<12}{r['symbol']:<8}{r['category']:<15}{_pct(r['car0']):>+10.2f}"
        f"{_pct(r['car1']):>+13.2f}{z:>7}{r['abn_range']:>14.2f}×"
    )


@dataclass
class Analysis:
    """Resultado de analizar una tabla de eventos (ver `analyze_events`)."""

    analyzed: list[dict]            # un resultado por evento analizable
    skipped: list[tuple]            # (evento, motivo) de los que no se pudieron analizar
    kept: list[dict]                # los independientes (tras fusionar y descartar solapes)
    n_same_day: int                 # fusionados por coincidir el mismo día en la misma acción
    n_overlap: int                  # descartados por estar cerca de otro de mayor prioridad
    pools: dict[str, list[float]]   # CAR de fechas al azar (placebo) por acción
    cache: dict = field(default_factory=dict)


def analyze_events(
    events: list, data_dir: Path, min_gap: int, priority: list[str], placebo_n: int, seed: int
) -> Analysis:
    """Analiza todos los eventos (sesión de reacción, modelo de mercado, CAR), se queda con los
    independientes y construye las fechas al azar de referencia. No imprime nada."""
    market = load_series(MARKET, data_dir=data_dir)
    cache: dict[str, dict] = {}
    analyzed, skipped = [], []

    for ev in events:
        if ev.symbol not in cache:
            cache[ev.symbol] = prepare(load_series(ev.symbol, data_dir=data_dir), market)
        data = cache[ev.symbol]
        session = first_affected_session(ev.when, data["session"])
        if session is None:
            skipped.append((ev, "no hay sesiones posteriores en los datos"))
            continue
        idx = data["session"].index(session)
        res = analyze_event(data, idx)
        if res is None:
            skipped.append((ev, "faltan datos antes (menos de 70 sesiones) o después (menos de 5)"))
            continue
        res.update(symbol=ev.symbol, category=ev.category, session=session, idx=idx)
        analyzed.append(res)

    if not analyzed:
        return Analysis(analyzed, skipped, [], 0, 0, {}, cache)
    kept, n_same_day, n_overlap = select_independent_events(analyzed, min_gap, priority)
    # Las fechas al azar nunca caen cerca de NINGÚN evento real, se conserve o no.
    pools = {}
    for sym, data in cache.items():
        mine = [r["idx"] for r in analyzed if r["symbol"] == sym]
        pools[sym] = [p["car1"] for p in placebo_pool(data, placebo_n, mine, seed)]
    return Analysis(analyzed, skipped, kept, n_same_day, n_overlap, pools, cache)


def run_events(args: argparse.Namespace) -> None:
    path = args.events or args.data_dir / "events" / "events.csv"
    if not path.exists():
        raise SystemExit(
            f"\nNo existe {path}.\nCrea el archivo con las columnas symbol,datetime,category,source,note "
            "(ver docs/events.example.csv), genera uno con `python -m backend.data.edgar` "
            "o prueba el modo --placebo."
        )
    events = load_events(path)
    priority = [c.strip().lower() for c in args.prioridad.split(",") if c.strip()]
    analysis = analyze_events(events, args.data_dir, args.min_gap, priority, args.placebo_n, args.seed)
    analyzed, skipped, kept = analysis.analyzed, analysis.skipped, analysis.kept
    n_same_day, n_overlap, pools = analysis.n_same_day, analysis.n_overlap, analysis.pools

    if not analyzed:
        raise SystemExit("\nNingún evento se pudo analizar:\n" + "\n".join(
            f"  {e.symbol} {e.when:%Y-%m-%d}: {why}" for e, why in skipped))

    print(f"\n{len(analyzed)} eventos analizados ({len(skipped)} omitidos por falta de datos)")
    print(
        f"→ {len(kept)} independientes: {n_same_day} fusionados por coincidir el mismo día en la misma acción y "
        f"{n_overlap} descartados por estar a menos de {args.min_gap} sesiones de otro de mayor prioridad"
    )

    rows = summarize([r for r in kept if pools.get(r["symbol"])], pools, args.draws, args.seed, args.motor)

    print("\nResumen por categoría. CAR = retorno que el mercado (SPY) no explica, día del evento + siguiente.")
    print("  CAR medio: con signo, margen de error 95 % y p frente a fechas al azar.")
    print("  |CAR| medio: tamaño de la reacción sin importar el signo, frente a lo esperado en fechas al azar.")
    for s in rows:
        margin = "–" if s["margin"] is None else f"{_pct(s['margin']):.2f}"
        print(
            f"  {s['category']:<14}{s['n']:>3} eventos   CAR medio {_pct(s['mean_car1']):>+6.2f} ± {margin} % "
            f"(p = {s['p']:.4f})   |CAR| medio {_pct(s['mean_abs_car1']):>5.2f} % "
            f"(azar {_pct(s['expected_abs']):.2f} %, p = {s['p_abs']:.4f})   rango ×{s['mean_abn_range']:.2f}"
        )

    if args.robustez:
        from backend.events.robustness import format_robustness, robustness

        threshold = 0.05 / (2 * max(len(rows), 1))
        usable = [r for r in kept if pools.get(r["symbol"])]
        report = robustness(usable, pools, args.draws, args.seed, args.motor, threshold=threshold)
        print(format_robustness(report, threshold, args.draws))

    shown = kept if args.detalle else sorted(kept, key=lambda r: -abs(r["car1"]))[: args.top]
    shown = sorted(shown, key=lambda r: (r["session"], r["symbol"]))
    print(f"\n{'Todos los eventos independientes' if args.detalle else f'Los {len(shown)} con mayor reacción'}")
    print(f"{'sesión':<12}{'acción':<8}{'categoría':<15}{'CAR día0 %':>10}{'CAR[0,+1] %':>13}{'z':>7}{'rango real/previsto':>20}")
    for r in shown:
        print(_event_line(r))

    if skipped:
        why = Counter(w for _, w in skipped)
        print("\nOmitidos: " + "; ".join(f"{n} ({w})" for w, n in why.items()))
        if args.detalle:
            for ev, w in skipped:
                print(f"  (omitido) {ev.symbol} {ev.when:%Y-%m-%d %H:%M}: {w}")

    k = max(len(rows), 1)
    print(
        "\nLectura: p pequeño = difícil de explicar por casualidad. Pero aquí se hacen "
        f"{2 * k} contrastes ({k} categorías × 2): para que un p resista la corrección de Bonferroni "
        f"hace falta p < {0.05 / (2 * k):.4f}. Un p suelto de 0,02 entre tantos no prueba nada."
        "\nLa reacción media con signo suele salir cercana a 0 porque los eventos buenos y malos se compensan;"
        "\nla magnitud (|CAR|) y el rango son los que detectan que 'pasó algo'. Saber SI fue bueno o malo"
        "\nrequiere el tono del comunicado, no solo su categoría."
    )


def run_placebo(args: argparse.Namespace) -> None:
    market = load_series(MARKET, data_dir=args.data_dir)
    print(f"\nPrueba de calibración: {args.placebo_n} fechas al azar por acción, como si hubiera habido un evento")
    print(f"{'acción':<8}{'fechas':>8}{'CAR[0,+1] medio %':>19}{'|z| > 1,96':>13}{'rango real/previsto':>21}")
    for sym in args.symbols:
        data = prepare(load_series(sym, data_dir=args.data_dir), market)
        res = placebo_pool(data, args.placebo_n, [], args.seed)
        zs = [r["z1"] for r in res if r["z1"] is not None]
        print(
            f"{sym.upper():<8}{len(res):>8}{_pct(sum(r['car1'] for r in res) / len(res)):>+19.3f}"
            f"{sum(abs(z) > 1.96 for z in zs) / len(zs) * 100:>12.1f} %{sum(r['abn_range'] for r in res) / len(res):>20.2f}×"
        )
    print(
        "\nLectura: sin efecto real, el CAR medio debería rondar 0 y en torno a un 5 % de las fechas superaría"
        "\n|z| > 1,96 por pura casualidad. Si el porcentaje sale bastante mayor, el test por evento es"
        "\ndemasiado optimista (los días reales son más volátiles y con más colas que una campana normal),"
        "\ny por eso el análisis usa fechas al azar como referencia, no solo el z."
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Efecto de eventos sobre el precio (estudio de eventos diario).")
    p.add_argument("--events", type=Path, default=None, help="CSV de eventos (por defecto data/events/events.csv)")
    p.add_argument("--placebo", action="store_true", help="modo de calibración con fechas al azar")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--min-gap", type=int, default=5, help="sesiones mínimas entre eventos de una misma acción")
    p.add_argument("--prioridad", default=DEFAULT_PRIORITY, help="categorías de mayor a menor prioridad, separadas por comas")
    p.add_argument("--top", type=int, default=10, help="cuántos eventos con mayor reacción listar")
    p.add_argument("--detalle", action="store_true", help="listar todos los eventos y los omitidos")
    p.add_argument("--placebo-n", type=int, default=300, help="fechas al azar por acción")
    p.add_argument("--draws", type=int, default=5000, help="remuestreos del test de permutaciones")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--robustez", action="store_true",
                   help="repite el contraste de magnitud por acción, sin cada acción y sin los mayores (ver README)")
    p.add_argument(
        "--motor", choices=["python", "cpp", "auto"], default="python",
        help="motor del test de permutaciones: python (por defecto), cpp (núcleo C++, ver README) o auto",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)
    (run_placebo if args.placebo else run_events)(args)


if __name__ == "__main__":
    main()
