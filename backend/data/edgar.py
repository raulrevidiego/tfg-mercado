"""Descarga las notificaciones 8-K de EDGAR y las convierte en una tabla de eventos.

Un 8-K es el aviso que una empresa cotizada en EEUU debe presentar ante la SEC cuando ocurre algo
relevante (resultados, cambios de directivos, presentaciones a inversores...). EDGAR da, gratis,
la fecha y HORA EXACTAS de aceptación y los códigos de los asuntos tratados ("items"). Eso las
convierte en eventos fechados al segundo, listos para el estudio de eventos.

Requisitos de la SEC: identificarse con un User-Agent que incluya nombre y correo, y no pasar de
10 peticiones por segundo. Pon en .env una línea como:

    SEC_USER_AGENT=Tu Nombre tu@correo.com

Uso desde la raíz del repo:

    python -m backend.data.edgar --start 2020-01-01
    python -m backend.events.evaluate_events --events data/events/edgar_8k.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

from backend.config import load_env_file
from backend.data.alpaca import DEFAULT_SYMBOLS

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
OLDER_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# Empresas que cambiaron de CIK al reorganizarse: su historial anterior está bajo el CIK antiguo.
# Exxon Mobil pasó a ExxonMobil Holdings (Texas) el 1 de julio de 2026 y el símbolo XOM apunta ya al
# CIK nuevo, que solo tiene filings desde esa fecha; los anteriores están en Exxon Mobil Corp (34088).
EXTRA_CIKS = {"XOM": [34088]}

PAUSE = 0.12  # segundos entre peticiones: unas 8 por segundo, por debajo del límite de 10

# Asuntos (items) de un 8-K, por orden de prioridad cuando un mismo 8-K trae varios.
CATEGORY_BY_ITEM = [
    ("2.02", "resultados"),      # resultados financieros
    ("5.02", "directivos"),      # salida o nombramiento de directivos y consejeros
    ("7.01", "regfd"),           # divulgación a inversores (presentaciones, discursos)
    ("8.01", "otros_eventos"),   # otros eventos que la empresa considera importantes
]
COLUMNS = ("accessionNumber", "filingDate", "acceptanceDateTime", "form", "items", "primaryDocument")


class MissingUserAgentError(RuntimeError):
    """No se ha configurado SEC_USER_AGENT."""


def get_user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        raise MissingUserAgentError(
            "La SEC exige identificarte. Añade a tu .env una línea como:\n"
            "  SEC_USER_AGENT=Tu Nombre tu@correo.com"
        )
    return ua


def _get_json(client: httpx.Client, url: str) -> dict:
    resp = client.get(url)
    time.sleep(PAUSE)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"La SEC devolvió {resp.status_code} para {url}. "
            "Un 403 suele indicar un User-Agent no válido o demasiadas peticiones."
        )
    return resp.json()


def resolve_ciks(client: httpx.Client, symbols: list[str]) -> dict[str, int]:
    """Convierte símbolos bursátiles en CIK (el identificador de la SEC)."""
    table = {row["ticker"].upper(): int(row["cik_str"]) for row in _get_json(client, TICKERS_URL).values()}
    missing = [s for s in symbols if s.upper() not in table]
    if missing:
        raise RuntimeError(f"La SEC no conoce estos símbolos: {', '.join(missing)}")
    return {s.upper(): table[s.upper()] for s in symbols}


def columns_to_rows(cols: dict) -> list[dict]:
    """La SEC entrega los filings 'por columnas' (una lista por campo); aquí se pasan a filas."""
    n = len(cols.get("accessionNumber", []))
    return [{k: (cols.get(k) or [""] * n)[i] for k in COLUMNS} for i in range(n)]


def fetch_filings(client: httpx.Client, cik: int, start: date) -> list[dict]:
    """Todos los filings de una empresa desde `start`: los recientes y los archivos antiguos que hagan falta."""
    data = _get_json(client, SUBMISSIONS_URL.format(cik=cik))
    rows = columns_to_rows(data["filings"]["recent"])
    for shard in data["filings"].get("files", []):
        if shard.get("filingTo", "9999-12-31") >= start.isoformat():  # solo si cubre fechas posteriores a start
            rows += columns_to_rows(_get_json(client, OLDER_URL.format(name=shard["name"])))
    return rows


def category_for_items(items: str) -> str:
    codes = {c.strip() for c in items.split(",")}
    return next((cat for code, cat in CATEGORY_BY_ITEM if code in codes), "otros")


def to_events(symbol: str, cik: int, rows: list[dict], start: date) -> list[dict]:
    """Filas de 8-K -> filas del CSV de eventos (symbol, datetime, category, source, note)."""
    events = []
    for r in rows:
        if r["form"] != "8-K" or date.fromisoformat(r["filingDate"]) < start or not r["acceptanceDateTime"]:
            continue
        accepted = datetime.fromisoformat(r["acceptanceDateTime"]).astimezone(timezone.utc)
        events.append({
            "symbol": symbol,
            "datetime": accepted.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "category": category_for_items(r["items"]),
            "source": ARCHIVE_URL.format(
                cik=cik, accession=r["accessionNumber"].replace("-", ""), document=r["primaryDocument"]
            ),
            "note": f"8-K items {r['items']}",
        })
    return events


def write_events_csv(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["symbol", "datetime", "category", "source", "note"])
        writer.writeheader()
        writer.writerows(sorted(events, key=lambda e: (e["datetime"], e["symbol"])))


def build_events(
    symbols: list[str], start: date, extra_ciks: dict[str, list[int]] | None = None
) -> list[dict]:
    """Eventos 8-K de cada símbolo, uniendo el CIK actual con los antiguos (`EXTRA_CIKS` y `extra_ciks`)."""
    headers = {"User-Agent": get_user_agent(), "Accept-Encoding": "gzip, deflate"}
    extras = {**EXTRA_CIKS, **(extra_ciks or {})}
    events: list[dict] = []
    with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
        ciks = resolve_ciks(client, symbols)
        for symbol, cik in ciks.items():
            seen: set[str] = set()
            found: list[dict] = []
            for c in dict.fromkeys([cik, *extras.get(symbol, [])]):  # sin repetir, en orden
                rows = [r for r in fetch_filings(client, c, start) if r["accessionNumber"] not in seen]
                seen.update(r["accessionNumber"] for r in rows)
                found += to_events(symbol, c, rows, start)
            log.info("%s: %d 8-K desde %s", symbol, len(found), start)
            events += found
    return events


def main(argv: list[str] | None = None) -> None:
    load_env_file()
    p = argparse.ArgumentParser(description="Genera una tabla de eventos con los 8-K de EDGAR.")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--start", default="2020-01-01", help="YYYY-MM-DD")
    p.add_argument("--output", type=Path, default=None, help="por defecto data/events/edgar_8k.csv")
    p.add_argument("--extra-cik", action="append", default=[], metavar="SÍMBOLO=CIK",
                   help="CIK antiguo de una empresa que cambió de registrante (se puede repetir)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = args.output or args.data_dir / "events" / "edgar_8k.csv"
    extra: dict[str, list[int]] = {}
    for item in args.extra_cik:
        symbol, _, cik = item.partition("=")
        extra.setdefault(symbol.strip().upper(), []).append(int(cik))
    try:
        events = build_events([s.upper() for s in args.symbols], date.fromisoformat(args.start), extra)
    except (MissingUserAgentError, RuntimeError) as e:
        sys.exit(f"\nError: {e}")
    write_events_csv(out, events)

    print(f"\nGuardados {len(events)} eventos en {out}\n")
    by = Counter((e["symbol"], e["category"]) for e in events)
    cats = sorted({c for _, c in by})
    print(f"{'acción':<8}" + "".join(f"{c:>15}" for c in cats))
    for symbol in sorted({s for s, _ in by}):
        print(f"{symbol:<8}" + "".join(f"{by[(symbol, c)]:>15}" for c in cats))


if __name__ == "__main__":
    main()
