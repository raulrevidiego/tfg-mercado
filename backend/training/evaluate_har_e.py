"""¿Mejora HAR si sabe que mañana hay resultados?

Compara HAR-E con el HAR original (la referencia), con su margen de error del 95 %, en tres cortes:
todos los días, y el conjunto de las acciones separado en víspera de resultados, día de resultados y
resto de días. Necesita el CSV de eventos de EDGAR (python -m backend.data.edgar).

Uso desde la raíz del repo:

    python -m backend.training.evaluate_har_e
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from backend.data.alpaca import DEFAULT_SYMBOLS
from backend.data.store import load_series
from backend.events.table import load_events
from backend.models.har_e import forecast_har_e_frame
from backend.training.evaluate_volatility import evaluate, format_report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evalúa HAR con el calendario de resultados.")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--events", type=Path, default=None, help="por defecto data/events/edgar_8k.csv")
    p.add_argument("--category", default="resultados")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    path = args.events or args.data_dir / "events" / "edgar_8k.csv"
    if not path.exists():
        raise SystemExit(f"\nNo existe {path}. Genéralo con: python -m backend.data.edgar")
    events = load_events(path)

    tests = []
    for symbol in args.symbols:
        try:
            frame, models = forecast_har_e_frame(
                load_series(symbol, data_dir=args.data_dir), events, symbol.upper(), args.category, args.warmup
            )
        except ValueError as e:
            print(f"\n(aviso) {e}: se omite.")
            continue
        rows, mean_range, n = evaluate(frame, args.warmup, models=models, reference="har")
        print(format_report(f"{symbol.upper()} · todos los días", rows, mean_range, n, reference="har"))
        tests.append(
            frame.slice(args.warmup).drop_nulls(["target", *models]).with_columns(symbol=pl.lit(symbol.upper()))
        )

    if not tests:
        raise SystemExit("\nNinguna acción tenía suficientes eventos.")
    pooled = pl.concat(tests)
    print("\n" + "=" * 100 + "\nConjunto de las acciones, por tipo de día")
    for label, mask in [
        ("víspera de resultados (mañana hay evento)", pl.col("earn_next") == 1),
        ("día de resultados (hoy hubo evento)", pl.col("earn_today") == 1),
        ("resto de días", (pl.col("earn_next") == 0) & (pl.col("earn_today") == 0)),
    ]:
        sub = pooled.filter(mask)
        if sub.height < 10:
            print(f"\n{label}: solo {sub.height} días, demasiado pocos para comparar")
            continue
        rows, mean_range, n = evaluate(sub, 0, models=models, reference="har")
        print(format_report(f"CONJUNTO · {label}", rows, mean_range, n, reference="har"))
    print(
        "\nLectura: 'mejora' > 0 % = pierde menos que el HAR original; el ± es el margen de error (95 %)."
        "\nSe supone que la fecha de resultados se conoce con antelación (se usa la fecha real del 8-K)."
    )


if __name__ == "__main__":
    main()
