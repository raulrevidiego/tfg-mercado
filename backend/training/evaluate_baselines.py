"""Evalúa los baselines sobre las acciones descargadas.

Como los baselines no se entrenan, que cada predicción use solo datos hasta ese día equivale
a un walk-forward con paso de un día. Con modelos que sí se entrenan, reentrenaremos por ventanas.

Uso desde la raíz del repo:

    python -m backend.training.evaluate_baselines
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from backend.data.alpaca import DEFAULT_SYMBOLS
from backend.data.store import load_series
from backend.models.baselines import BASELINES, forecast_frame
from backend.training.metrics import directional_accuracy, mae, proportion_margin, rmse


def evaluate(frame: pl.DataFrame, warmup: int) -> tuple[list[dict], float, int]:
    """Evalúa todos los baselines desde el día `warmup` (los anteriores solo sirven de historia).

    Devuelve (filas de resultados, fracción de días de subida, número de días evaluados).
    """
    test = frame.slice(warmup).drop_nulls("target")
    if test.height == 0:
        raise ValueError("No quedan días para evaluar: reduce --warmup")
    actual = test["target"].to_list()
    moves = [a for a in actual if a != 0]
    up_rate = sum(a > 0 for a in moves) / len(moves)

    rows: list[dict] = []
    zero_mae = mae(test["zero"].to_list(), actual)
    for name in BASELINES:
        pred = test[name].to_list()
        acc, calls = directional_accuracy(pred, actual)
        m = mae(pred, actual)
        rows.append({
            "model": name,
            "mae": m,
            "rmse": rmse(pred, actual),
            "acc": acc,
            "margin": proportion_margin(acc, calls) if acc is not None else None,
            "calls": calls,
            "vs_zero": (1 - m / zero_mae) * 100,
        })
    return rows, up_rate, test.height


def format_report(symbol: str, rows: list[dict], up_rate: float, n: int) -> str:
    lines = [
        f"\n{symbol}  ({n:,} días de test; días de subida: {up_rate * 100:.1f} %)",
        f"  {'modelo':<15}{'MAE %':>8}{'RMSE %':>9}{'dirección %':>19}{'días':>8}{'MAE vs zero':>14}",
    ]
    for r in rows:
        acc = "–" if r["acc"] is None else f"{r['acc'] * 100:.1f} ± {r['margin'] * 100:.1f}"
        lines.append(
            f"  {r['model']:<15}{r['mae'] * 100:>8.2f}{r['rmse'] * 100:>9.2f}"
            f"{acc:>19}{r['calls']:>8,}{r['vs_zero']:>13.1f} %"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evalúa los baselines con retornos diarios.")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--warmup", type=int, default=500, help="días iniciales usados solo como historia")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    for symbol in args.symbols:
        frame = forecast_frame(load_series(symbol, data_dir=args.data_dir))
        rows, up_rate, n = evaluate(frame, args.warmup)
        print(format_report(symbol.upper(), rows, up_rate, n))
    print(
        "\nLectura: 'MAE vs zero' > 0 % = mejor que 'mañana no cambia nada'; < 0 % = peor."
        "\n'dirección' es el % de días en que acierta si sube o baja, con su margen de error (95 %)."
    )


if __name__ == "__main__":
    main()
