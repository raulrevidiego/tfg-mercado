"""Rutas de la interfaz (/api/app/*): sirven los JSON precalculados por `python -m backend.app.build`.

No calculan nada pesado: leen data/app/*.json (unos cientos de KB). Si faltan, devuelven un 404 que dice qué
comando ejecutar. El motor del test de permutaciones (C++ o Python) sí se consulta en vivo.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from backend.events import fastperm

router = APIRouter(prefix="/api/app", tags=["interfaz"])

SYMBOL_RE = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
HINT = "No hay datos para la interfaz: ejecuta python -m backend.app.build"


def app_dir() -> Path:
    """Carpeta data/app (la variable de entorno se lee en cada llamada, útil en tests)."""
    data = Path(os.environ.get("TFG_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
    return data / "app"


def _read(name: str):
    path = app_dir() / name
    if not path.exists():
        raise HTTPException(404, f"{HINT} (falta {name})")
    return json.loads(path.read_text(encoding="utf-8"))


def _symbol(symbol: str, available) -> str:
    if not SYMBOL_RE.match(symbol):
        raise HTTPException(400, "Símbolo no válido")
    symbol = symbol.upper()
    if symbol not in available:
        raise HTTPException(404, f"{symbol} no está en los datos de la interfaz (hay: {', '.join(sorted(available))})")
    return symbol


@router.get("/meta")
def meta() -> dict:
    """Fecha de los datos, acciones, motor y remuestreos con los que se generó todo."""
    return _read("meta.json")


@router.get("/engine")
def engine() -> dict:
    """Motor del test de permutaciones disponible AHORA en esta máquina (no el que generó los JSON)."""
    return {"engine": "cpp" if fastperm.available() else "python", "cpp": fastperm.available(), "threads": fastperm.max_threads()}


@router.get("/events")
def events(symbol: str, categories: str | None = Query(None, description="categorías separadas por comas")) -> dict:
    """Eventos independientes de una acción con su reacción (CAR), opcionalmente filtrados por categoría."""
    data = _read("events.json")
    symbol = _symbol(symbol, data)
    wanted = {c.strip() for c in categories.split(",") if c.strip()} if categories else None
    rows = [e for e in data[symbol] if wanted is None or e["category"] in wanted]
    return {"symbol": symbol, "categories": sorted({e["category"] for e in data[symbol]}), "events": rows}


@router.get("/events-summary")
def events_summary() -> dict:
    """Resumen por categoría: p, veredicto de robustez, insignia y lectura rápida."""
    return _read("events_summary.json")


@router.get("/volatility")
def volatility(symbol: str) -> dict:
    """Rango real y previsto (HAR, HAR-E, red) día a día, y el rango esperado para la siguiente sesión."""
    data = _read("volatility.json")
    symbol = _symbol(symbol, data["symbols"])
    return {"symbol": symbol, "day_types": data["day_types"], **data["symbols"][symbol]}


@router.get("/models")
def models() -> dict:
    """Comparación de modelos frente a HAR-E (MAE y QLIKE con margen de error) y su lectura rápida."""
    return _read("models.json")
