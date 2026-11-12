"""Descarga el TEXTO de los comunicados que acompañan a los 8-K (por defecto, los de resultados).

Un 8-K de resultados suele ser una carátula corta; el comunicado de prensa (con la cita del CEO y los
números) va en un anexo, normalmente el "Exhibit 99.1". Este módulo localiza ese anexo, lo convierte a
texto plano y lo guarda en `data/raw/edgar_text/`, más un índice en `data/events/texts_index.csv` que
usa el análisis del tono.

Misma política que `edgar.py`: User-Agent con nombre y correo (SEC_USER_AGENT) y menos de 10 peticiones
por segundo.

Uso desde la raíz del repo:

    python -m backend.data.edgar_text --category resultados
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
import time
from datetime import timezone
from datetime import timezone
from html.parser import HTMLParser
from pathlib import Path

import httpx

from backend.config import load_env_file
from backend.data import edgar
from backend.events.table import load_events

INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
SOURCE_RE = re.compile(r"/edgar/data/(\d+)/(\d{18})/([^/]+)$")
EXHIBIT_RE = re.compile(r"(ex(hibit)?[-_ ]?99|99[-_.]?0?1)", re.I)
PRESS_RELEASE_RE = re.compile(r"99[-_.]?0?1", re.I)
BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "table"}
SHORT_TEXT = 1500  # menos caracteres que esto: probablemente solo la carátula, no el comunicado


def parse_source_url(url: str) -> tuple[int, str, str] | None:
    """'.../edgar/data/1045810/000104581024000029/doc.htm' -> (1045810, '000104581024000029', 'doc.htm')."""
    m = SOURCE_RE.search(url)
    return (int(m.group(1)), m.group(2), m.group(3)) if m else None


def pick_document(names: list[str], primary: str) -> str:
    """Elige el anexo del comunicado (Exhibit 99.1 mejor que 99.2); si no hay, el documento principal."""
    candidates = [n for n in names if n.lower().endswith((".htm", ".html", ".txt")) and EXHIBIT_RE.search(n)]
    if not candidates:
        return primary
    return sorted(candidates, key=lambda n: (not PRESS_RELEASE_RE.search(n), n))[0]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Texto plano de un HTML: sin scripts ni estilos, con espacios y saltos de línea ordenados."""
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def _cache_path(cache_dir: Path, symbol: str, accession: str) -> Path:
    return cache_dir / f"{symbol}_{accession}.txt"


def fetch_text(
    client: httpx.Client, symbol: str, cik: int, accession: str, primary: str, cache_dir: Path
) -> dict:
    """Texto del comunicado de un 8-K. Devuelve {document, text, used_exhibit}. Usa caché si existe."""
    cached = _cache_path(cache_dir, symbol, accession)
    if cached.exists():
        first, _, text = cached.read_text(encoding="utf-8").partition("\n")
        document = first.removeprefix("#document: ")
        return {"document": document, "text": text, "used_exhibit": document != primary}

    names: list[str] = []
    resp = client.get(INDEX_URL.format(cik=cik, accession=accession))
    time.sleep(edgar.PAUSE)
    if resp.status_code < 400:
        names = [item["name"] for item in resp.json().get("directory", {}).get("item", [])]
    document = pick_document(names, primary)

    resp = client.get(DOC_URL.format(cik=cik, accession=accession, document=document))
    time.sleep(edgar.PAUSE)
    if resp.status_code >= 400:
        raise RuntimeError(f"La SEC devolvió {resp.status_code} para {document} ({symbol} {accession})")
    text = resp.text if document.lower().endswith(".txt") else html_to_text(resp.text)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(f"#document: {document}\n{text}", encoding="utf-8")
    return {"document": document, "text": text, "used_exhibit": document != primary}


def build_texts(events_csv: Path, category: str, data_dir: Path) -> list[dict]:
    """Descarga el texto de los eventos de una categoría y escribe el índice texts_index.csv."""
    headers = {"User-Agent": edgar.get_user_agent(), "Accept-Encoding": "gzip, deflate"}
    cache_dir = data_dir / "raw" / "edgar_text"
    rows: list[dict] = []
    with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
        for ev in load_events(events_csv):
            parsed = parse_source_url(ev.source)
            if ev.category != category or parsed is None:
                continue
            cik, accession, primary = parsed
            got = fetch_text(client, ev.symbol, cik, accession, primary, cache_dir)
            rows.append({
                "symbol": ev.symbol,
                "datetime": ev.when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "accession": accession, "document": got["document"], "n_chars": len(got["text"]),
                "used_exhibit": int(got["used_exhibit"]), "path": str(_cache_path(cache_dir, ev.symbol, accession)),
            })
    index = data_dir / "events" / "texts_index.csv"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["symbol", "datetime", "accession", "document", "n_chars", "used_exhibit", "path"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> None:
    load_env_file()
    p = argparse.ArgumentParser(description="Descarga el texto de los comunicados de los 8-K.")
    p.add_argument("--events", type=Path, default=None, help="por defecto data/events/edgar_8k.csv")
    p.add_argument("--category", default="resultados")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    events = args.events or args.data_dir / "events" / "edgar_8k.csv"
    if not events.exists():
        sys.exit(f"\nNo existe {events}. Genéralo con: python -m backend.data.edgar")
    try:
        rows = build_texts(events, args.category, args.data_dir)
    except (edgar.MissingUserAgentError, RuntimeError) as e:
        sys.exit(f"\nError: {e}")
    if not rows:
        sys.exit(f"\nNo hay eventos '{args.category}' con enlace a un documento de EDGAR.")

    short = [r for r in rows if r["n_chars"] < SHORT_TEXT]
    print(f"\nTextos descargados: {len(rows)}  |  con anexo de comunicado: {sum(r['used_exhibit'] for r in rows)}"
          f"  |  longitud mediana: {int(statistics.median(r['n_chars'] for r in rows)):,} caracteres")
    for sym in sorted({r["symbol"] for r in rows}):
        mine = [r for r in rows if r["symbol"] == sym]
        print(f"  {sym:<6}{len(mine):>4} comunicados  (anexo: {sum(r['used_exhibit'] for r in mine)}, "
              f"mediana {int(statistics.median(r['n_chars'] for r in mine)):,} caracteres)")
    if short:
        print(f"\nAviso: {len(short)} textos tienen menos de {SHORT_TEXT:,} caracteres (¿solo la carátula del 8-K?):")
        for r in short[:8]:
            print(f"  {r['symbol']} {r['datetime'][:10]}  {r['document']}  ({r['n_chars']} caracteres)")
    print(f"\nÍndice: {args.data_dir / 'events' / 'texts_index.csv'}")


if __name__ == "__main__":
    main()
