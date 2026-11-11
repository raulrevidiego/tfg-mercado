"""Evalúa los baselines de volatilidad sobre las acciones descargadas.

Referencia: `expanding_mean` ("mañana oscila lo de siempre"). Cada modelo se compara con ella.
Las mejoras llevan su margen de error al 95 %: si el margen es mayor que la mejora, no se puede
afirmar que el modelo sea mejor.

Uso desde la raíz del repo:

    python -m backend.training.evaluate_volatility
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from backend.data.alpaca import DEFAULT_SYMBOLS
from backend.data.store import load_series
from backend.models.volatility import VOL_BASELINES, forecast_vol_frame
from backend.training.metrics import mean_diff_margin, qlike_losses

REFERENCE = "expanding_mean"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def evaluate(
    frame: pl.DataFrame, warmup: int, models: list[str] | None = None, reference: str = REFERENCE
) -> tuple[list[dict], float, int]:
    """Devuelve (filas de resultados, rango medio real, días evaluados).

    `models` son las columnas a evaluar (por defecto, los baselines de volatilidad) y
    `reference` la columna con la que se compara cada una (debe estar en `models`).
    """
    models = models or list(VOL_BASELINES)
    test = frame.slice(warmup).drop_nulls(["target", *models])
    if test.height == 0:
        raise ValueError("No quedan días para evaluar: reduce --warmup")
    actual = test["target"].to_list()

    losses = {}
    for name in models:
        pred = test[name].to_list()
        losses[name] = {
            "abs": [abs(p - a) for p, a in zip(pred, actual)],
            "qlike": qlike_losses(pred, actual),
        }

    ref = losses[reference]
    rows = []
    for name in models:
        row = {"model": name}
        for key in ("abs", "qlike"):
            mine, base = losses[name][key], ref[key]
            diffs = [b - m for b, m in zip(base, mine)]  # > 0: el modelo pierde menos que la referencia
            row[key] = _mean(mine)
            row[f"{key}_gain"] = _mean(diffs) / _mean(base) * 100
            row[f"{key}_margin"] = mean_diff_margin(diffs) / _mean(base) * 100
        rows.append(row)
    return rows, _mean(actual), test.height


def format_report(
    symbol: str, rows: list[dict], mean_range: float, n: int, reference: str = REFERENCE
) -> str:
    lines = [
        f"\n{symbol}  ({n:,} días de test; rango diario medio real: {mean_range * 100:.2f} %)",
        f"  {'modelo':<15}{'MAE (pp)':>9}{'mejora MAE':>19}{'QLIKE':>9}{'mejora QLIKE':>20}",
    ]
    for r in rows:
        if r["model"] == reference:
            gains = f"{'(referencia)':>19}", f"{'(referencia)':>20}"
        else:
            gains = (
                f"{r['abs_gain']:>+10.1f} ± {r['abs_margin']:.1f} %",
                f"{r['qlike_gain']:>+11.1f} ± {r['qlike_margin']:.1f} %",
            )
        lines.append(f"  {r['model']:<15}{r['abs'] * 100:>9.3f}{gains[0]:>19}{r['qlike']:>9.3f}{gains[1]:>20}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evalúa baselines de volatilidad (rango diario).")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--warmup", type=int, default=500, help="días iniciales usados solo como historia")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    for symbol in args.symbols:
        frame = forecast_vol_frame(load_series(symbol, data_dir=args.data_dir), warmup=args.warmup)
        rows, mean_range, n = evaluate(frame, args.warmup)
        print(format_report(symbol.upper(), rows, mean_range, n))
    print(
        "\nLectura: 'mejora' > 0 % = pierde menos que la referencia; el ± es el margen de error (95 %)."
        "\nSi el margen es mayor que la mejora, no se puede afirmar que el modelo sea mejor."
        "\nMAE (pp) = error medio en puntos porcentuales de rango; QLIKE: menor es mejor."
    )


if __name__ == "__main__":
    main()
