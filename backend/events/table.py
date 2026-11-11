"""Tabla de eventos: qué se dijo, de qué empresa y cuándo.

Formato del archivo CSV (data/events/events.csv):

    symbol,datetime,category,source,note
    NVDA,2024-06-10T13:30:00Z,positivo,https://...,texto o resumen del comentario

- `datetime`: fecha y hora del comentario en ISO 8601. Con zona horaria ("Z" = UTC, "-04:00"...)
  o, si no la lleva, se entiende hora de Nueva York (la de la bolsa).
- `category`: la etiqueta que quieras (positivo, negativo, neutral, producto...). Se agrupa por ella.
- Las líneas que empiezan por # se ignoran.

La clave es decidir QUÉ SESIÓN de bolsa recoge la reacción: un comentario hecho con la bolsa cerrada
afecta a la siguiente sesión (ver `first_affected_session`).
"""

from __future__ import annotations

import csv
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)  # hora de Nueva York

REQUIRED = ("symbol", "datetime", "category")


@dataclass(frozen=True)
class Event:
    symbol: str
    when: datetime  # siempre con zona horaria
    category: str
    source: str = ""
    note: str = ""


def parse_when(text: str) -> datetime:
    """Lee una fecha ISO 8601. Sin zona horaria se asume hora de Nueva York."""
    dt = datetime.fromisoformat(text.strip())  # acepta 'Z' desde Python 3.11
    return dt.replace(tzinfo=NY) if dt.tzinfo is None else dt


def load_events(path: Path) -> list[Event]:
    lines = [
        ln for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    reader = csv.DictReader(lines)
    missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"{path.name}: faltan las columnas {', '.join(missing)}")
    events = []
    for n, row in enumerate(reader, start=1):
        try:
            symbol, category = row["symbol"].strip().upper(), row["category"].strip().lower()
            if not symbol or not category:
                raise ValueError("symbol y category no pueden estar vacíos")
            events.append(Event(
                symbol, parse_when(row["datetime"]), category,
                (row.get("source") or "").strip(), (row.get("note") or "").strip(),
            ))
        except ValueError as e:
            raise ValueError(f"{path.name}, evento nº {n}: {e}") from e
    return events


def first_affected_session(when: datetime, sessions: list[date]) -> date | None:
    """Primera sesión de bolsa que recoge la reacción a un comentario.

    - Durante la sesión, o antes de que abra: la propia sesión (la vela diaria lo incluye).
    - Con la bolsa cerrada (después de las 16:00 de Nueva York): la siguiente sesión.
    - Fin de semana o festivo: la siguiente sesión.
    `sessions` es la lista ordenada de días con vela. Devuelve None si no hay sesión posterior.
    """
    local = when.astimezone(NY)
    day = local.date() + (timedelta(days=1) if local.time() >= MARKET_CLOSE else timedelta(0))
    i = bisect_left(sessions, day)
    return sessions[i] if i < len(sessions) else None
