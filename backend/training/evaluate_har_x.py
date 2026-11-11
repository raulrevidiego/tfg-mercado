"""¿Añaden algo el mercado, el sector y el volumen a la predicción de volatilidad?

Compara variantes de HAR con el HAR original (la referencia) y con la EWMA. Las mejoras llevan su
margen de error al 95 %: si el margen es mayor que la mejora, no se puede afirmar que aporten.

Necesita descargar antes las series de referencia (SPY y los ETF sectoriales) con el ingestor de
Alpaca (ver README). Uso desde la raíz del repo:

    python -m backend.training.evaluate_har_x
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.data.alpaca import DEFAULT_SYMBOLS
from backend.data.store import load_series
from backend.models.har_x import MARKET_ETF, SECTOR_ETF, forecast_har_x_frame
from backend.training.evaluate_volatility import evaluate, format_report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evalúa HAR con variables de mercado, sector y volumen.")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    try:
        market = load_series(MARKET_ETF, data_dir=args.data_dir)
    except FileNotFoundError as e:
        raise SystemExit(f"\nError: {e}\nDescarga SPY y los ETF sectoriales (ver README).")

    for symbol in args.symbols:
        etf = SECTOR_ETF.get(symbol.upper())
        try:
            sector = load_series(etf, data_dir=args.data_dir) if etf else None
        except FileNotFoundError:
            sector = None
        if sector is None:
            print(f"\n(aviso) {symbol.upper()}: no hay ETF sectorial descargado ({etf}); "
                  "'har_all' usa solo mercado y volumen.")
        stock = load_series(symbol, data_dir=args.data_dir)
        frame, models = forecast_har_x_frame(stock, market, sector, warmup=args.warmup)
        rows, mean_range, n = evaluate(
            frame, args.warmup, models=[*models, "ewma_094"], reference="har"
        )
        label = f"{symbol.upper()} [mercado: {MARKET_ETF}" + (f", sector: {etf}]" if sector is not None else "]")
        print(format_report(label, rows, mean_range, n, reference="har"))
    print(
        "\nLectura: 'mejora' > 0 % = pierde menos que el HAR original; el ± es el margen de error (95 %)."
        "\nSi el margen es mayor que la mejora, esa variable no se puede decir que aporte."
        "\nhar_mkt = + mercado; har_sector = + sector; har_vol = + volumen; har_all = todo lo disponible."
    )


if __name__ == "__main__":
    main()
