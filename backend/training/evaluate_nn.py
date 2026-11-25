"""¿Mejora una red neuronal a HAR-E? Walk-forward con las 4 acciones y los mismos márgenes de error.

Compara `nn` con `har_enext` (el mejor modelo clásico hasta ahora: HAR + "mañana hay resultados"), que es
la referencia, y con `har`. Necesita PyTorch (en la GPU va mucho más rápido) y el CSV de eventos de EDGAR.

Uso desde la raíz del repo:
    python -m backend.training.evaluate_nn 
                    # perceptrón multicapa, 3 semillas
    python -m backend.training.evaluate_nn --arch gru      # red recurrente
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from backend.data.alpaca import DEFAULT_SYMBOLS
from backend.data.store import load_series
from backend.events.table import load_events
from backend.models.nn_vol import build_panel, resolve_device, walk_forward
from backend.training.evaluate_volatility import evaluate, format_report

MODELS = ["har", "har_enext", "nn"]
REFERENCE = "har_enext"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evalúa una red neuronal de volatilidad frente a HAR-E.")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--events", type=Path, default=None, help="por defecto data/events/edgar_8k.csv")
    p.add_argument("--category", default="resultados")
    p.add_argument("--arch", choices=["mlp", "gru"], default="mlp")
    p.add_argument("--window", type=int, default=22)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--refit-every", type=int, default=21)
    p.add_argument("--device", default="auto", help="auto, cpu o cuda")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    path = args.events or args.data_dir / "events" / "edgar_8k.csv"
    if not path.exists():
        raise SystemExit(f"\nNo existe {path}. Genéralo con: python -m backend.data.edgar")
    try:
        device = resolve_device(args.device)
        stocks = {s.upper(): load_series(s, data_dir=args.data_dir) for s in args.symbols}
        panel = build_panel(
            stocks, load_series("SPY", data_dir=args.data_dir), load_events(path), args.category,
            args.warmup, args.refit_every,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise SystemExit(f"\nError: {e}")

    def progress(done: int, total: int, day: int, seconds: float) -> None:
        if done == 1 or done % 5 == 0 or done == total:
            print(f"  tramo {done}/{total} (día {day}) · {seconds:.0f} s")

    print(f"\nRed {args.arch} · {args.seeds} semillas · dispositivo {device} · ventana {args.window} días")
    preds = walk_forward(
        panel, args.window, args.warmup, args.refit_every, args.arch, args.hidden, args.dropout,
        args.epochs, seeds=args.seeds, device=device, progress=progress,
    )

    tests, saved = [], []
    for symbol, frame in panel.items():
        frame = frame.with_columns(pl.Series("nn", preds[symbol], dtype=pl.Float64))
        saved.append(frame.select("session", "nn").with_columns(symbol=pl.lit(symbol)))
        rows, mean_range, n = evaluate(frame, args.warmup, models=MODELS, reference=REFERENCE)
        print(format_report(f"{symbol} · todos los días", rows, mean_range, n, reference=REFERENCE))
        tests.append(frame.slice(args.warmup).drop_nulls(["target", *MODELS]).with_columns(symbol=pl.lit(symbol)))

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
        rows, mean_range, n = evaluate(sub, 0, models=MODELS, reference=REFERENCE)
        print(format_report(f"CONJUNTO · {label}", rows, mean_range, n, reference=REFERENCE))

    out = args.data_dir / "models" / "nn_predictions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(saved).write_parquet(out)
    print(f"\nPredicciones guardadas en {out}")
    print(
        "Lectura: 'mejora' > 0 % = pierde menos que HAR-E (la referencia); el ± es el margen de error (95 %)."
        "\nSi el margen es mayor que la mejora, la red no se puede decir que sea mejor."
    )


if __name__ == "__main__":
    main()
